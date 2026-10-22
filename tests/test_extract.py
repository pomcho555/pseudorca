from __future__ import annotations

import ast
import base64
import bz2
import gzip
import hashlib
import io
import json
import lzma
import pickle
import random
import shutil
import tarfile
import time
import tracemalloc
import zipfile
import zlib
from collections.abc import Iterable
from pathlib import Path

import pytest
from click.testing import CliRunner

import pseudorca.extract as extract_mod
from pseudorca.cli import main
from pseudorca.extract import EXTRACTOR_VERSION, Limits, extract_bytes, extract_file, payloads, tree_for
from pseudorca.extract.magics import cell_magic, transform_python
from pseudorca.extract.payloads import _NotDecodable, decompress, pattern_candidates
from pseudorca.notebooks import register_bytes, register_file
from pseudorca.store import Store, StoreError
from pseudorca.tree import Hit, Tree

# ----- synthetic inputs ------------------------------------------------------------------------------


def notebook(*cells: str | tuple[str, str], metadata: dict | None = None) -> bytes:
    """nbformat 4 JSON; a cell is code source, or (cell_type, source)."""
    out = []
    for cell in cells:
        kind, source = cell if isinstance(cell, tuple) else ("code", cell)
        entry = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            entry.update(outputs=[{"output_type": "stream", "text": ["OUTPUT_TEXT\n"]}], execution_count=1)
        out.append(entry)
    meta = metadata if metadata is not None else {"kernelspec": {"language": "python", "name": "python3"}}
    return json.dumps({"cells": out, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}).encode()


def extract(*cells: str | tuple[str, str], limits: Limits | None = None) -> Tree:
    return extract_bytes(notebook(*cells), "test.ipynb", limits)


def py_source(tag: str, n: int = 30) -> str:
    """Synthetic Python that does not compress to almost nothing."""
    rng = random.Random(tag)
    defs = [
        f"def f_{tag}_{i}(x):\n    return x * {rng.randint(1000, 999999)} + {rng.random():.6f}\n"
        for i in range(n)
    ]
    return f"import math\n\n{''.join(defs)}"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def tar_of(members: dict[str, bytes], extra: Iterable[tarfile.TarInfo] = ()) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for info in extra:
            tf.addfile(info)
    return buf.getvalue()


def zip_of(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def payload_files(tree: Tree, cell: int = 0) -> list[str]:
    return tree.paths(f"payloads/cell_{cell:03d}/*")


def with_alone_dict(stream: bytes, size: int) -> bytes:
    """An lzma-alone stream whose header declares another dictionary size (the data needs no more than its
    own length, so it still decodes)."""
    return stream[:1] + size.to_bytes(4, "little") + stream[5:]


def with_xz_dict(stream: bytes, code: int) -> bytes:
    """A one-block xz stream, as lzma.compress makes it, whose LZMA2 filter declares dictionary `code`
    (32 = 256 MiB, 36 = 1 GiB, 40 = 4 GiB - 1); the block header's CRC32 is recomputed."""
    out = bytearray(stream)
    end = 12 + (out[12] + 1) * 4
    at = out.index(b"\x21\x01", 14, end - 4) + 2
    out[at] = code
    out[end - 4 : end] = zlib.crc32(out[12 : end - 4]).to_bytes(4, "little")
    return bytes(out)


def _vli(n: int) -> bytes:
    out = bytearray()
    while n >= 0x80:
        out.append(n & 0x7F | 0x80)
        n >>= 7
    return bytes([*out, n])


def xz_blocks(chunks: list[bytes], code: int, *, sizes: bool) -> bytes:
    """An xz stream built by hand with one LZMA2 block per chunk, each header declaring dictionary `code`
    and, with `sizes`, the block's compressed and uncompressed size (as multi-threaded xz writes them)."""
    flags = b"\x00\x01"  # CRC32 checks
    out = bytearray(b"\xfd7zXZ\x00" + flags + zlib.crc32(flags).to_bytes(4, "little"))
    records = []
    for chunk in chunks:
        packed = lzma.compress(
            chunk, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 1 << 16}]
        )
        body = bytes([0xC0 if sizes else 0x00])
        body += (_vli(len(packed)) + _vli(len(chunk)) if sizes else b"") + bytes([0x21, 0x01, code])
        size = -(-(len(body) + 5) // 4) * 4
        header = bytes([size // 4 - 1]) + body
        header += bytes(size - 4 - len(header))
        header += zlib.crc32(header).to_bytes(4, "little")
        out += header + packed + bytes(-len(packed) % 4) + zlib.crc32(chunk).to_bytes(4, "little")
        records.append((len(header) + len(packed) + 4, len(chunk)))
    index = bytearray(b"\x00" + _vli(len(records)))
    for unpadded, unpacked in records:
        index += _vli(unpadded) + _vli(unpacked)
    index += bytes(-len(index) % 4)
    index += zlib.crc32(index).to_bytes(4, "little")
    backward = (len(index) // 4 - 1).to_bytes(4, "little")
    out += index + zlib.crc32(backward + flags).to_bytes(4, "little") + backward + flags + b"YZ"
    return bytes(out)


def symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links cannot be made here")


# ----- cells and magics ------------------------------------------------------------------------------


def test_cells_markdown_and_meta():
    data = notebook(("markdown", "# Plan\nUse **X**."), "x = 1\n", ("raw", "raw text"), "   \n", "y = 2")
    t = extract_bytes(data, "plan.ipynb")
    assert t.paths() == ["cells/cell_001.py", "cells/cell_004.py", "markdown/cell_000.md"]
    assert t.text("markdown/cell_000.md") == "# Plan\nUse **X**."
    assert t.files["cells/cell_001.py"].origin == "cell 1"
    assert "OUTPUT_TEXT" not in "".join(f.text for f in t.files.values())  # outputs are ignored
    assert t.meta == {"extractor_version": EXTRACTOR_VERSION, "format": "ipynb", "cells": 5, "env": []}
    assert t.source_sha256 == hashlib.sha256(data).hexdigest()
    assert t.warnings == []


def test_writefile_and_append():
    t = extract(
        "%%writefile src/agent.py\nimport os\nPROMPT = 'v1'\n",
        "%%writefile -a src/agent.py\nprint(PROMPT)\n",
        "%%writefile /opt/work/run.sh\necho start\n",
        "%%file --append notes.txt\nfirst\n",
    )
    assert t.text("files/src/agent.py") == "import os\nPROMPT = 'v1'\nprint(PROMPT)\n"
    assert t.files["files/src/agent.py"].origin == (
        "cell 0 > %%writefile src/agent.py; cell 1 > %%writefile -a src/agent.py"
    )
    assert t.text("files/opt/work/run.sh") == "echo start\n"  # the leading "/" is dropped
    assert t.text("files/notes.txt") == "first\n"  # %%file is IPython's alias, -a creates a missing file
    stub = t.text("cells/cell_000.py")
    assert stub.count("\n") == 1 and stub.startswith("# %%writefile src/agent.py")
    assert ast.parse(stub)


def test_writefile_overwrite_keeps_the_last_version():
    t = extract("%%writefile a.py\nv = 1\n", "%%writefile a.py\nv = 2\n")
    assert t.text("files/a.py") == "v = 2\n"
    assert t.paths("files/*") == ["files/a.py"]


def test_writefile_bodies_end_with_a_newline_as_in_ipython():
    # Jupyter saves a cell's last line without "\n" (the helper keeps that); IPython adds one before
    # %%writefile runs, so an appended cell starts on a line of its own.
    blob = b64(zlib.compress(py_source("appended").encode()))
    t = extract(
        "%%writefile main.py\nimport os\nMAX_STEPS = 64",
        f"%%writefile -a main.py\ndef agent(obs):\n    return policy(obs)\nBLOB = '{blob}'",
        "%%writefile empty.py",
    )
    assert t.text("files/main.py") == (
        f"import os\nMAX_STEPS = 64\ndef agent(obs):\n    return policy(obs)\nBLOB = '{blob}'\n"
    )
    assert t.grep("def agent(obs):") == [Hit("files/main.py", 3, "def agent(obs):")]
    (path,) = payload_files(t, 1)
    assert t.files[path].origin == "cell 1 > %%writefile -a main.py > line 5 str > base64 > zlib"
    assert t.text("files/empty.py") == ""

    # appended Python writes count line breaks as the tree does ("\r" too)
    t = extract(
        "open('gen.py', 'w').write('import zlib\\r')\n"
        f"open('gen.py', 'a').write('X = 1\\rBLOB = \"{blob}\"\\r')\n"
    )
    assert t.grep("BLOB = ", path_glob="files/*") == [Hit("files/gen.py", 3, f'BLOB = "{blob}"')]
    (path,) = payload_files(t)
    assert t.files[path].origin.startswith("cell 0 > open().write gen.py > line 3 str")


def test_cell_magic_bodies_agree_with_ipython():
    tm = pytest.importorskip("IPython.core.inputtransformer2")
    sources = [
        "%%writefile -a agent.py\ndef solve(task):\n    return task",
        "%%writefile a.py\nx = 1\n",
        "\n\n%%bash\nset -e\npip install x",
        "%%writefile a.py",
        "%%writefile a.py\nx = 1\r",
        "%%writefile a.py\r\nx = 1\r\ny = 2",
        "%%html\n<b>x</b>\n\n",
    ]
    for source in sources:
        call = ast.parse(tm.TransformerManager().transform_cell(source)).body[0].value
        assert isinstance(call, ast.Call) and call.func.attr == "run_cell_magic", source
        name, args, body = (a.value for a in call.args)
        magic = cell_magic(source)
        assert magic is not None and (magic.name, magic.args, magic.body) == (name, args, body), source


def test_shell_and_other_cell_magics():
    t = extract(
        "%%bash\necho hi\nls -la\n",
        "%%sh\nexport X=1\n",
        "%%script bash --no-raise-error\necho via script\n",
        "%%script /usr/bin/env sh\necho via env\n",
        "%%html\n<b>bold</b>\n",
        "%%javascript\nconsole.log(1)\n",
    )
    assert t.text("cells/cell_000.sh") == "# %%bash\necho hi\nls -la\n"
    assert t.text("cells/cell_001.sh") == "# %%sh\nexport X=1\n"
    assert t.text("cells/cell_002.sh") == "# %%script bash --no-raise-error\necho via script\n"
    assert t.text("cells/cell_003.sh") == "# %%script /usr/bin/env sh\necho via env\n"
    assert t.text("cells/cell_004.html.txt") == "# %%html\n<b>bold</b>\n"
    assert t.text("cells/cell_005.javascript.txt") == "# %%javascript\nconsole.log(1)\n"
    assert t.files["cells/cell_000.sh"].origin == "cell 0 > %%bash"
    assert not t.paths("cells/*.py") and t.warnings == []


def test_shell_and_other_magic_files_keep_the_cell_line_numbers():
    # the lines up to the magic line stay (the magic line as a comment), so line N of the file is line N
    # of the cell, for evidence and for payload origins alike
    blob = b64(gzip.compress(py_source("sh-lines").encode()))
    t = extract(
        "\n%%bash\nset -e\npip install vllm==0.6.3\n",
        f"%%html\n<p>hi</p>\n<!-- {blob} -->",
        f" \n%%sh\necho {blob} | base64 -d > a.py",
        "\r\n%%bash\r\necho crlf\r\n",
    )
    assert t.text("cells/cell_000.sh") == "\n# %%bash\nset -e\npip install vllm==0.6.3\n"
    assert t.grep("vllm==0.6.3") == [Hit("cells/cell_000.sh", 4, "pip install vllm==0.6.3")]
    assert t.text("cells/cell_001.html.txt") == f"# %%html\n<p>hi</p>\n<!-- {blob} -->\n"
    assert t.files["payloads/cell_001/0.py"].origin == "cell 1 > %%html > line 3 str > base64 > gzip"
    assert t.text("cells/cell_002.sh").startswith(" \n# %%sh\necho ")
    assert t.files["payloads/cell_002/0.py"].origin == "cell 2 > %%sh > line 3 str > base64 > gzip"
    assert t.text("cells/cell_003.sh") == "\r\n# %%bash\r\necho crlf\r\n"
    assert t.grep("echo crlf") == [Hit("cells/cell_003.sh", 3, "echo crlf")]


def test_time_and_capture_bodies_stay_python():
    t = extract("%%time\nx = 1\n%pip install foo\n", "%%capture --no-stderr out\nimport os\n")
    assert t.text("cells/cell_000.py") == "# %%time\nx = 1\n# %pip install foo\n"
    assert t.text("cells/cell_001.py") == "# %%capture --no-stderr out\nimport os\n"
    assert t.warnings == []


def test_line_magics_are_commented_and_line_numbers_kept():
    source = (
        "import os\n"
        "!pip install -q pkg==1.0\n"
        "%matplotlib inline\n"
        "files = !ls -la\n"
        "here = %pwd\n"
        "try:\n"
        "    import pkg\n"
        "except ImportError:\n"
        "    !pip install pkg\n"
        "np.mean?\n"
        "x = 5 % 3  # !not a magic\n"
        "y = (x\n"
        "     % 2)\n"
        'PROMPT = """\n'
        "!keep this line\n"
        "%keep this too\n"
        '"""\n'
        "%pip install a \\\n"
        "    b\n"
        "z = x != 1\n"
    )
    t = extract(source)
    text = t.text("cells/cell_000.py")
    lines = text.splitlines()
    assert len(lines) == len(source.splitlines())
    assert lines[1] == "# !pip install -q pkg==1.0"
    assert lines[3] == "# files = !ls -la"
    assert lines[4] == "# here = %pwd"
    assert lines[8] == "    pass  # !pip install pkg"  # the except block still has a statement
    assert lines[9] == "# np.mean?"
    assert lines[10] == "x = 5 % 3  # !not a magic"
    assert lines[12] == "     % 2)"
    assert lines[14:16] == ["!keep this line", "%keep this too"]
    assert lines[17:19] == ["# %pip install a \\", "    # b"]
    assert lines[19] == "z = x != 1"
    assert ast.parse(text) and t.warnings == []


def test_magic_handling_agrees_with_ipython():
    tm = pytest.importorskip("IPython.core.inputtransformer2")
    cells = [
        "import os\n!pip install -q foo\n%time x = 1\nfiles = !ls\ny = %pwd\n",
        "if True:\n    !echo hi\n    %env A=1\nelse:\n    pass\n",
        's = """\n!not a magic\n%not either\n"""\nt = (1\n     % 2)\n',
        "x = 1  # !not magic\nd = {'a': 1}\nd['a'] = !echo 2\nok = x != 1\n",
        "np.mean?\nval = 3\n",
    ]
    for source in cells:
        ipython = tm.TransformerManager().transform_cell(source)
        assert ast.parse(ipython)
        text, _ = transform_python(source)
        assert ast.parse(text)
        old = source.splitlines()
        changed_by_ipython = {
            i for i, (a, b) in enumerate(zip(old, ipython.splitlines(), strict=True)) if a != b
        }
        changed_by_us = {i for i, (a, b) in enumerate(zip(old, text.splitlines(), strict=True)) if a != b}
        assert changed_by_us == changed_by_ipython, source


def test_env_collection():
    t = extract(
        "%env MODEL_DIR=/data/models/small\n%env TOKENIZERS_PARALLELISM false\n%set_env A=1\n%env HOME\n",
        "import os\n"
        "os.environ['B'] = '2'\n"
        "os.environ.setdefault('C', '3')\n"
        "os.environ.update({'D': '4'}, )\n"
        "os.environ['E'] = some_variable\n"
        "os.environ[f'{prefix}_F'] = '6'\n",
    )
    assert t.meta["env"] == [
        {"cell": 0, "key": "MODEL_DIR", "value": "/data/models/small"},
        {"cell": 0, "key": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"cell": 0, "key": "A", "value": "1"},
        {"cell": 1, "key": "B", "value": "2"},
        {"cell": 1, "key": "C", "value": "3"},
        {"cell": 1, "key": "D", "value": "4"},
    ]


# ----- literal writes --------------------------------------------------------------------------------


def test_literal_file_writes():
    t = extract(
        "from pathlib import Path\n"
        "import pathlib, textwrap\n"
        "Path('out/a.py').write_text('print(1)\\n')\n"
        "pathlib.Path('out') / 'b.txt'\n"
        "(pathlib.Path('out') / 'b.txt').write_text(textwrap.dedent('''\n    one\n    two\n'''))\n"
        "open('c.txt', 'w').write('c' + 'd')\n"
        "with open('d.cfg', 'w', encoding='utf-8') as f:\n"
        "    f.write('[a]\\n')\n"
        "    f.write('k = 1\\n')\n"
        "with open('log.txt', 'a') as f:\n"
        "    f.write('first')\n"
        "open('log.txt', 'a').write(' second')\n"
        "Path('w.bin').write_bytes(b'\\x00\\x01\\x02')\n"
    )
    assert t.text("files/out/a.py") == "print(1)\n"
    assert t.files["files/out/a.py"].origin == "cell 0 > write_text out/a.py"
    assert t.text("files/out/b.txt") == "\none\ntwo\n"
    assert t.text("files/c.txt") == "cd"
    assert t.files["files/c.txt"].origin == "cell 0 > open().write c.txt"
    assert t.text("files/d.cfg") == "[a]\nk = 1\n"
    assert t.text("files/log.txt") == "first second"
    assert t.files["files/w.bin"].binary and t.files["files/w.bin"].size == 3


def test_non_literal_writes_are_skipped():
    t = extract(
        "from pathlib import Path\n"
        "d = 'out'\n"
        "Path(f'{d}/a.py').write_text('x = 1')\n"
        "Path('ok.py').write_text(f'{d}')\n"
        "open(name, 'w').write('x')\n"
        "open('read.txt').write('x')\n"
        "with open('mixed.txt', 'w') as f:\n"
        "    f.write('a')\n"
        "    f.write(d)\n"
        "with open('dumped.json', 'w') as f:\n"
        "    f.write('{')\n"
        "    json.dump({}, f)\n"
    )
    assert t.paths("files/*") == []


def test_written_python_is_scanned_for_payloads():
    inner = b64(zlib.compress(py_source("inner").encode()))
    t = extract(
        f"from pathlib import Path\nPath('gen.py').write_text('''import zlib\nBLOB = \"{inner}\"\n''')\n"
    )
    assert "files/gen.py" in t
    (path,) = payload_files(t)
    assert path == "payloads/cell_000/0.py"
    assert t.files[path].origin == "cell 0 > write_text gen.py > line 2 str > base64 > zlib"


def test_written_files_that_are_not_python_are_searched_for_payloads():
    # %%writefile searched a written run.sh or config for payloads; write_text and open().write did not,
    # and nothing written as bytes was searched at all, not even Python
    src = py_source("written")
    blob = b64(src.encode())
    sh = f"#!/bin/bash\\necho {blob} | base64 -d > helper.py\\n"
    t = extract(
        f'from pathlib import Path\nPath("run.sh").write_text("{sh}")\n',
        f'with open("cfg.yaml", "w") as f:\n    f.write("name: x\\nblob: {blob}\\n")\n',
        f'open("log.sh", "w").write("#!/bin/bash\\n")\nopen("log.sh", "a").write("echo {blob}\\n")\n',
        f'open("run.sh", "wb").write(b"{sh}")\n',
        f'from pathlib import Path\nPath("gen.py").write_bytes(b"import base64\\nBLOB = \'{blob}\'\\n")\n',
        f"%%writefile run.sh\n#!/bin/bash\necho {blob} | base64 -d > helper.py\n",
    )
    origins = [
        "cell 0 > write_text run.sh > line 2 str > base64",
        "cell 1 > open().write cfg.yaml > line 2 str > base64",
        "cell 2 > open().write log.sh > line 2 str > base64",  # the append continues the file's lines
        "cell 3 > open().write run.sh > line 2 str > base64",
        "cell 4 > write_bytes gen.py > line 2 str > base64",
        "cell 5 > %%writefile run.sh > line 2 str > base64",  # as before: the same content, the same payload
    ]
    for cell, origin in enumerate(origins):
        (path,) = payload_files(t, cell)
        assert t.text(path) == src and t.files[path].origin == origin, cell
    assert t.warnings == []

    # max_nesting bounds written files that are not Python too
    t = extract(
        f'from pathlib import Path\nPath("run.sh").write_text("{sh}")\n', limits=Limits(max_nesting=0)
    )
    assert payload_files(t) == []
    assert t.warnings == ["cell 0 > write_text run.sh: not searched for nested payloads (max_nesting 0)"]


# ----- payloads --------------------------------------------------------------------------------------


def test_base64_lzma_tar_payload(store: Store, tmp_path: Path):
    members = {
        "src/agent.py": py_source("agent").encode(),
        "src/prompts/system.txt": b"You are a careful solver.\n",
        "../../escape.py": b"ESCAPED = True\n",
        "/etc/abs.cfg": b"key = value\n",
    }
    link = tarfile.TarInfo("src/link.py")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    device = tarfile.TarInfo("dev/null")
    device.type = tarfile.CHRTYPE
    blob = b64(lzma.compress(tar_of(members, [link, device])))
    data = notebook("x = 1\n", f"import base64, lzma, tarfile, io\nCODE = '{blob}'\n")
    t = extract_bytes(data, "agent.ipynb")
    assert payload_files(t, 1) == [
        "payloads/cell_001/0/escape.py",
        "payloads/cell_001/0/etc/abs.cfg",
        "payloads/cell_001/0/src/agent.py",
        "payloads/cell_001/0/src/prompts/system.txt",
    ]
    assert t.files["payloads/cell_001/0/src/agent.py"].origin == (
        "cell 1 > line 2 str > base64 > lzma > tar member src/agent.py"
    )
    assert t.text("payloads/cell_001/0/escape.py") == "ESCAPED = True\n"
    assert all(".." not in p.split("/") for p in t.paths())

    # through the cache: nothing lands outside cache/trees/<id>/
    nb, _ = register_bytes(store, data, name="agent.ipynb")
    after_register = {p for p in tmp_path.rglob("*")}
    cached = tree_for(store, nb)
    new = {p for p in tmp_path.rglob("*")} - after_register
    cache_dir = store.root / "cache" / "trees" / nb.id
    assert new and all(p == cache_dir or cache_dir in p.parents or p in cache_dir.parents for p in new)
    assert not any(p.name == "escape.py" for p in tmp_path.rglob("*"))
    assert cached.paths() == t.paths()


def test_base64_zlib_text_and_nested_payload():
    inner_src = py_source("inner")
    inner = b64(zlib.compress(inner_src.encode()))
    outer_src = f"import zlib, base64\n\nINNER = '{inner}'\n{py_source('outer', 5)}"
    outer = b64(zlib.compress(outer_src.encode()))
    t = extract(f"import zlib, base64\nexec_me = '{outer}'\n")
    assert payload_files(t) == ["payloads/cell_000/0.py", "payloads/cell_000/1.py"]
    assert t.text("payloads/cell_000/0.py") == outer_src
    assert t.text("payloads/cell_000/1.py") == inner_src
    assert t.files["payloads/cell_000/1.py"].origin == (
        "cell 0 > line 2 str > base64 > zlib > line 3 str > base64 > zlib"
    )


def test_gzip_bytes_literal():
    raw = gzip.compress(
        b'{"model": "synthetic", "layers": [' + b", ".join(b"%d" % i for i in range(300)) + b"]}"
    )
    t = extract(f"DATA = {raw!r}\n")
    (path,) = payload_files(t)
    assert path == "payloads/cell_000/0.json"
    assert json.loads(t.text(path))["model"] == "synthetic"
    assert t.files[path].origin == "cell 0 > line 1 bytes > gzip"


def test_base85_ascii85_hex_and_urlsafe():
    src = py_source("enc").encode()
    t = extract(
        f"A = {base64.b85encode(src).decode()!r}\n",
        f"B = {base64.a85encode(src).decode()!r}\n",
        f"C = '{zlib.compress(src).hex()}'\n",
        f"D = '{base64.urlsafe_b64encode(bz2.compress(src)).decode().rstrip('=')}'\n",
    )
    assert t.text("payloads/cell_000/0.py") == src.decode()
    assert t.files["payloads/cell_000/0.py"].origin == "cell 0 > line 1 str > base85"
    assert t.files["payloads/cell_001/0.py"].origin == "cell 1 > line 1 str > ascii85"
    assert t.files["payloads/cell_002/0.py"].origin == "cell 2 > line 1 str > hex > zlib"
    assert t.text("payloads/cell_003/0.py") == src.decode()


def test_joined_chunks_are_decoded_once():
    blob = b64(zlib.compress(py_source("chunks").encode()))
    chunks = [blob[i : i + 300] for i in range(0, len(blob), 300)]
    plus = " +\n    ".join(repr(c) for c in chunks)
    t = extract(f"A = ({plus})\nB = ''.join([{', '.join(repr(c) for c in chunks)}])\n")
    assert payload_files(t) == ["payloads/cell_000/0.py", "payloads/cell_000/1.py"]
    assert t.warnings == []  # the chunks were not also tried one by one


def test_zip_payload_and_member_decompression():
    archive = zip_of(
        {
            "pkg/model.py": py_source("zip").encode(),
            "data/cfg.json.gz": gzip.compress(b'{"seed": 7, "name": "synthetic"}'),
            "pkg/": b"",
        }
    )
    t = extract(f"Z = '{b64(archive)}'\n")
    assert payload_files(t) == ["payloads/cell_000/0/data/cfg.json", "payloads/cell_000/0/pkg/model.py"]
    assert t.files["payloads/cell_000/0/data/cfg.json"].origin == (
        "cell 0 > line 1 str > base64 > zip member data/cfg.json.gz > gzip"
    )


def test_pickle_is_recorded_and_never_loaded(tmp_path: Path):
    marker = tmp_path / "created-if-unpickled"

    class CreatesMarker:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    raw = pickle.dumps(CreatesMarker(), protocol=4)
    t = extract(f"STATE = '{b64(raw)}'\n", f"STATE_RAW = {raw!r}\n", limits=Limits(min_payload_chars=40))
    for path in ("payloads/cell_000/0.pkl", "payloads/cell_001/0.pkl"):
        f = t.files[path]
        assert f.binary and f.text == "" and f.size == len(raw)
    assert not marker.exists()


def test_random_base64_text_is_not_a_payload():
    rng = random.Random(1)
    junk = b64(bytes(rng.randrange(256) for _ in range(600)))
    prose = "You are a helpful assistant. Answer briefly and cite the source. " * 5
    t = extract(f"A = '{junk}'\nB = '{prose}'\nC = '{'A' * 400}'\n")
    assert payload_files(t) == [] and t.warnings == []


def test_unparseable_cell_falls_back_to_patterns():
    blob = b64(zlib.compress(py_source("fallback").encode()))
    t = extract(f"def broken(:\n    pass\nCODE = '{blob}'  # it's here\n")
    assert any("cell 0: not valid Python" in w and "pattern" in w for w in t.warnings)
    (path,) = payload_files(t)
    assert t.files[path].origin == "cell 0 > line 3 str > base64 > zlib"


def test_shell_cell_payloads_are_found():
    blob = b64(gzip.compress(py_source("shell").encode()))
    t = extract(f"%%bash\necho {blob} | base64 -d | gunzip > agent.py\n")
    (path,) = payload_files(t)
    assert path == "payloads/cell_000/0.py"
    assert t.files[path].origin == "cell 0 > %%bash > line 2 str > base64 > gzip"


def test_pattern_scan_strings_and_lines():
    blob = b64(zlib.compress(py_source("lines").encode()))
    # a one-quote string ends at a line end that no backslash escapes, even after an escaped one
    (cand,) = pattern_candidates(f'x = "abc\\\ndef\ny = "{blob}"\n', 200)
    assert (cand.line, cand.col, cand.value) == (3, 4, blob)
    # an unclosed string does not hide the strings after it
    (cand,) = pattern_candidates(f"say \"it's\n'{blob}'", 200)
    assert (cand.line, cand.value) == (2, blob)
    assert [c.value for c in pattern_candidates(f'"""{blob}"""\n\'{blob}\'', 200)] == [blob, blob]
    assert pattern_candidates(f"# '{blob}'\n", 200) == []
    # lines are counted as the tree counts them: \r\n, \r and \n
    (cand,) = pattern_candidates(f"a = 1\rb = 2\r\nc = '{blob}'", 200)
    assert (cand.line, cand.col) == (3, 4)


# ----- limits and hostile input ----------------------------------------------------------------------


def test_decompression_bombs_are_bounded():
    zeros = bytes(8 << 20)
    bombs = [
        zlib.compress(zeros, 9),
        bz2.compress(zeros, 9),
        lzma.compress(zeros, preset=1),
        lzma.compress(tar_of({"big.bin": zeros}), preset=1),
        zip_of({"big.bin": zeros}),
    ]
    limits = Limits(min_payload_chars=40, max_payload_bytes=128 << 10, max_total_bytes=1 << 20)
    data = notebook(*(f"B = '{b64(b)}'\n" for b in bombs))
    tracemalloc.start()
    try:
        t = extract_bytes(data, "bombs.ipynb", limits)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 << 20
    assert sum("truncated" in w for w in t.warnings) >= 3
    assert any("big.bin: skipped" in w for w in t.warnings)
    assert all(f.size <= 128 << 10 for f in t.files.values())


def test_total_budget_member_and_depth_limits():
    members = {f"m{i}.py": f"V = {i}\n".encode() for i in range(10)}
    t = extract(f"T = '{b64(tar_of(members))}'\n", limits=Limits(max_members=3))
    assert len(payload_files(t)) == 3
    assert any("max_members (3) reached" in w for w in t.warnings)

    deep = b64(zlib.compress(gzip.compress(py_source("deep").encode())))
    t = extract(f"D = '{deep}'\n", limits=Limits(max_depth=2))
    assert payload_files(t) == ["payloads/cell_000/0.gz"]
    assert any("not decompressed (max_depth 2 reached)" in w for w in t.warnings)

    a = b64(zlib.compress(py_source("a").encode()))
    b = b64(zlib.compress(py_source("b").encode()))
    t = extract(f"A = '{a}'\n", f"B = '{b}'\n", limits=Limits(max_total_bytes=1000))
    assert any("max_total_bytes" in w for w in t.warnings)
    assert len(t.paths("payloads/*")) <= 1


def test_hostile_payloads_never_raise():
    good = zlib.compress(py_source("h").encode())
    tar = tar_of({"a.py": py_source("t").encode()})
    cases = [
        good[: len(good) // 2],  # stream ended early
        good[:2] + bytes(200),  # zlib magic, garbage after
        b"\x1f\x8b" + bytes(300),  # gzip magic, garbage
        b"BZh9" + bytes(300),
        b"\xfd7zXZ\x00" + bytes(300),
        b"\x5d\x00\x00" + b"\xff" * 300,  # lzma-alone header claiming a huge dictionary
        tar[:700],  # tar cut inside a member
        tar[:257] + b"ustar" + bytes(300),
        b"PK\x03\x04" + bytes(400),
        zip_of({"a.py": b"x = 1\n" * 50})[:-30],  # damaged central directory
    ]
    t = extract(*(f"X = '{b64(c)}'\n" for c in cases), limits=Limits(min_payload_chars=40))
    assert t.meta["cells"] == len(cases)
    assert not any("extraction failed" in w or "not decoded (" in w for w in t.warnings)


def test_big_lzma_dictionaries_are_decoded_within_the_memory_limit():
    # a header may declare any dictionary size, and lzma.decompress has no memory limit: a stream that
    # declared 128 MiB or more once vanished from the tree, without a warning
    src = py_source("big_dict").encode()
    small = [{"id": lzma.FILTER_LZMA2, "dict_size": 1 << 16}]
    alone = lzma.compress(
        src, format=lzma.FORMAT_ALONE, filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 16}]
    )
    bcj = lzma.compress(src, filters=[{"id": lzma.FILTER_X86}, *small])  # two filters in the block header
    cases = [
        with_alone_dict(alone, 256 << 20),
        with_alone_dict(alone, 3 << 30),
        with_xz_dict(lzma.compress(src, filters=small), 36),  # 1 GiB
        with_xz_dict(bcj, 40),  # 4 GiB - 1
        xz_blocks([src[:3000], src[3000:]], 32, sizes=True),  # 256 MiB in every block header
    ]
    assert lzma.decompress(xz_blocks([src[:3000], src[3000:]], 4, sizes=True)) == src  # the builder is sound
    t = extract(*(f"X = '{b64(c)}'\n" for c in cases))
    for cell in range(len(cases)):
        assert t.text(f"payloads/cell_{cell:03d}/0.py") == src.decode(), cell
        assert t.files[f"payloads/cell_{cell:03d}/0.py"].origin == f"cell {cell} > line 1 str > base64 > lzma"
    assert t.warnings == []

    # past max_depth such a stream is still recognised and kept as a binary file
    t = extract(f"X = '{b64(cases[2])}'\n", limits=Limits(max_depth=1))
    assert payload_files(t) == ["payloads/cell_000/0.xz"] and t.files["payloads/cell_000/0.xz"].binary
    assert t.warnings == ["cell 0 > line 1 str > base64 > lzma: not decompressed (max_depth 1 reached)"]


def test_lzma_dictionaries_too_big_even_when_lowered_are_kept_with_a_warning():
    # with max_payload_bytes this high, a dictionary big enough for the output would pass LZMA_MEMLIMIT
    src = py_source("too_big").encode()
    alone = lzma.compress(
        src, format=lzma.FORMAT_ALONE, filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 16}]
    )
    single = with_xz_dict(lzma.compress(src, filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 1 << 16}]), 36)
    cases = [
        with_alone_dict(alone, 256 << 20),
        single,
        f"X = {single!r}\n",
        with_alone_dict(alone, 0xFFFF0000),  # not a size an lzma header can have: not lzma at all
    ]
    limits = Limits(max_payload_bytes=200 << 20)
    t = extract(*(c if isinstance(c, str) else f"X = '{b64(c)}'\n" for c in cases), limits=limits)
    limit = payloads.LZMA_MEMLIMIT
    assert t.paths("payloads/*") == [
        "payloads/cell_000/0.lzma",
        "payloads/cell_001/0.xz",
        "payloads/cell_002/0.xz",
    ]
    assert all(t.files[p].binary for p in t.paths("payloads/*"))
    assert t.warnings == [
        f"cell 0 > line 1 str > base64 > lzma: not decompressed (its dictionary needs more than "
        f"LZMA_MEMLIMIT, {limit} bytes)",
        f"cell 1 > line 1 str > base64 > lzma: not decompressed (its dictionary needs more than "
        f"LZMA_MEMLIMIT, {limit} bytes)",
        f"cell 2 > line 1 bytes > lzma: not decompressed (its dictionary needs more than "
        f"LZMA_MEMLIMIT, {limit} bytes)",
    ]

    # the blocks after the first are found only through the sizes their headers record: without them the
    # second block's dictionary stays as declared, and the stream is kept with a warning
    t = extract(f"X = '{b64(xz_blocks([src[:3000], src[3000:]], 32, sizes=False))}'\n")
    assert payload_files(t) == ["payloads/cell_000/0.xz"]
    assert t.warnings == [
        f"cell 0 > line 1 str > base64 > lzma: not decompressed (its dictionary needs more than "
        f"LZMA_MEMLIMIT, {limit} bytes)"
    ]


def test_quoted_string_scan_takes_linear_time():
    # an opening quote followed by many escaped ones once made every escaped quote rescan the line: O(n²)
    n = 6000
    hostile = [
        '"' + '\\"' * n,
        "'" + "\\'" * n,
        '"""' + '\\"""' * n,
        "'''" + "\\'''" * n,
        "\"'" + "\\\"\\'" * n,
    ]
    start = time.perf_counter()
    for text in hostile:
        assert pattern_candidates(text, 200) == []
    t = extract(f"%%bash\necho {hostile[0]}\n")
    elapsed = time.perf_counter() - start
    assert payload_files(t) == []
    assert elapsed < 1.5


def test_concatenated_and_empty_streams_take_linear_time():
    # each stream's end once copied the rest of the input, and empty or one-byte streams trip no byte
    # limit: N streams back to back cost O(N²) time
    start = time.perf_counter()
    for kind, member, n in [
        ("zlib", zlib.compress(b""), 160_000),
        ("gzip", gzip.compress(b""), 80_000),
        ("bz2", bz2.compress(b""), 80_000),
        ("xz", lzma.compress(b""), 80_000),
    ]:
        with pytest.raises(_NotDecodable):
            decompress(kind, member * n, 1 << 20)
    out, problem = decompress("zlib", zlib.compress(b"x") * 160_000, 1 << 20)
    t = extract(f"E = '{b64(zlib.compress(b'') * 160_000)}'\n")
    elapsed = time.perf_counter() - start
    assert out == b"x" * payloads.MAX_STREAMS and problem == f"more than {payloads.MAX_STREAMS} streams"
    assert payload_files(t) == [] and t.warnings == []
    assert elapsed < 1.5


def test_concatenated_streams_are_followed_up_to_a_limit():
    a, b = py_source("first").encode(), py_source("second").encode()
    big = random.Random(5).randbytes(300_000)  # compresses to more than one input window
    for kind, compress in [
        ("zlib", zlib.compress),
        ("gzip", gzip.compress),
        ("bz2", bz2.compress),
        ("xz", lzma.compress),
    ]:
        assert decompress(kind, compress(a) + compress(b), 1 << 20) == (a + b, ""), kind
        assert decompress(kind, compress(big) + compress(a), 1 << 20) == (big + a, ""), kind
        # an empty stream ends the decoding; nothing is lost when nothing follows it
        assert decompress(kind, compress(a) + compress(b"") + compress(b), 1 << 20) == (a, "empty stream")
        assert decompress(kind, compress(a) + compress(b""), 1 << 20) == (a, "")
    n = payloads.MAX_STREAMS
    parts = [b"%d," % i for i in range(n + 5)]
    out, problem = decompress("zlib", b"".join(zlib.compress(p) for p in parts), 1 << 20)
    assert out == b"".join(parts[:n]) and problem == f"more than {n} streams"

    t = extract(f"S = '{b64(b''.join(gzip.compress(py_source(str(i)).encode()) for i in range(n + 1)))}'\n")
    (path,) = payload_files(t)
    assert t.text(path) == "".join(py_source(str(i)) for i in range(n))
    assert t.warnings == [
        f"cell 0 > line 1 str > base64 > gzip: more than {n} streams after {len(t.text(path))} bytes; "
        "kept what was decoded"
    ]


def test_malformed_notebooks():
    with pytest.raises(ValueError, match=r"bad\.ipynb"):
        extract_bytes(b'{"cells": [', "bad.ipynb")
    with pytest.raises(ValueError, match="no cells"):
        extract_bytes(b'{"metadata": {}}', "empty.ipynb")
    with pytest.raises(ValueError):
        extract_bytes(b"[" * 100_000 + b"]" * 100_000, "deep.ipynb")
    with pytest.raises(ValueError):
        extract_bytes(b"\xff\xfe\x00garbage", "binary.ipynb")
    t = extract_bytes(
        json.dumps(
            {"cells": [5, {"cell_type": "code", "source": 7}, {"cell_type": "code", "source": "a = 1"}]}
        ).encode(),
        "odd.ipynb",
    )
    assert t.paths() == ["cells/cell_002.py"] and len(t.warnings) == 2 and t.meta["cells"] == 3


def test_nbformat3_notebook():
    data = json.dumps(
        {
            "nbformat": 3,
            "metadata": {"name": "old"},
            "worksheets": [
                {
                    "cells": [
                        {"cell_type": "heading", "level": 2, "source": ["Setup"]},
                        {"cell_type": "code", "language": "python", "input": ["!pip install x\n", "a = 1\n"]},
                        {"cell_type": "markdown", "source": "notes"},
                    ]
                }
            ],
        }
    ).encode()
    t = extract_bytes(data, "old.ipynb")
    assert t.text("cells/cell_001.py") == "# !pip install x\na = 1\n"
    assert t.text("markdown/cell_000.md") == "## Setup"
    assert t.text("markdown/cell_002.md") == "notes"
    assert t.meta["cells"] == 3


def test_non_python_kernel_cells_are_kept_verbatim():
    data = notebook(
        "!x <- TRUE\nprint(x)\n",
        metadata={"kernelspec": {"language": "R"}, "language_info": {"name": "R", "file_extension": ".r"}},
    )
    t = extract_bytes(data, "r.ipynb")
    assert t.text("cells/cell_000.r") == "!x <- TRUE\nprint(x)\n"
    assert any("kernel language is 'r'" in w for w in t.warnings)


def test_script_kernel(tmp_path: Path):
    blob = b64(zlib.compress(py_source("script").encode()))
    source = (
        "import os\n"
        "from pathlib import Path\n"
        "os.environ['MODE'] = 'fast'\n"
        f"CODE = '{blob}'\n"
        "Path('helper.py').write_text('X = 1\\n')\n"
    )
    path = tmp_path / "kernel.py"
    path.write_text(source)
    t = extract_file(path)
    assert t.text("script.py") == source
    assert t.meta == {
        "extractor_version": EXTRACTOR_VERSION,
        "format": "script",
        "cells": 0,
        "env": [{"cell": None, "key": "MODE", "value": "fast"}],
    }
    assert t.text("files/helper.py") == "X = 1\n"
    assert t.paths("payloads/*") == ["payloads/script/0.py"]
    assert t.files["payloads/script/0.py"].origin == "script > line 4 str > base64 > zlib"

    r_script = extract_bytes(b"df %>% filter(x > 1)\n", "analysis.R")
    assert r_script.text("script.py") == "df %>% filter(x > 1)\n"
    assert any(w.startswith("script: not valid Python") for w in r_script.warnings)


def test_lone_surrogates_and_undecodable_names_survive_the_cache(store: Store):
    member = tar_of({"bad\udcff\nname.py": b"V = 1\n"})  # stored as the raw bytes b"bad\xff\nname.py"
    cells = [
        {"cell_type": "code", "source": "%%writefile out\ud800.py\nx = 1\n"},
        {"cell_type": "code", "source": f"%env K=v\ud800\ny = '\udfff'\nT = '{b64(member)}'\n"},
        {"cell_type": "markdown", "source": "note \ud800"},
    ]
    data = json.dumps({"cells": cells}).encode()
    nb, _ = register_bytes(store, data, name="odd.ipynb")
    fresh = tree_for(store, nb)
    assert "files/out?.py" in fresh and fresh.meta["env"] == [{"cell": 1, "key": "K", "value": "v?"}]
    assert fresh.paths("payloads/*") == ["payloads/cell_001/0/bad??name.py"]
    assert fresh.warnings == []
    cached = tree_for(store, nb)
    assert cached.paths() == fresh.paths()
    assert [cached.text(p) for p in cached.paths()] == [fresh.text(p) for p in fresh.paths()]


# ----- tree_for --------------------------------------------------------------------------------------


def test_tree_for_caches_and_rebuilds_on_version_bump(store: Store, monkeypatch: pytest.MonkeyPatch):
    nb, _ = register_bytes(store, notebook("x = 1\n"), name="main.ipynb")
    first = tree_for(store, nb)
    assert (store.root / "cache" / "trees" / nb.id / "manifest.json").is_file()

    def must_not_run(*args, **kwargs):
        raise AssertionError("extracted again")

    monkeypatch.setattr(extract_mod, "extract_bytes", must_not_run)
    again = tree_for(store, nb)
    assert again.text("cells/cell_000.py") == first.text("cells/cell_000.py") == "x = 1\n"
    assert again.meta["extractor_version"] == EXTRACTOR_VERSION

    calls = []

    def counting(*args, **kwargs):
        calls.append(args[1])
        return extract_bytes(*args, **kwargs)

    monkeypatch.setattr(extract_mod, "extract_bytes", counting)
    monkeypatch.setattr(extract_mod, "EXTRACTOR_VERSION", EXTRACTOR_VERSION + 1)
    rebuilt = tree_for(store, nb)
    assert calls == ["main.ipynb"] and rebuilt.meta["extractor_version"] == EXTRACTOR_VERSION + 1
    tree_for(store, nb)
    assert len(calls) == 1


def test_tree_for_refuses_linked_cache_folders(store: Store, tmp_path: Path):
    # a cache entry can arrive with the repository: a linked files/ under a stale manifest once made
    # tree_for write into, and then empty, a folder of the link's choosing
    nb, _ = register_bytes(store, notebook("x = 1\n"), name="main.ipynb")
    docs = tmp_path / "home" / "Documents"
    docs.mkdir(parents=True)
    for name in ("notes.txt", "thesis.docx"):
        (docs / name).write_text("keep me\n")
    entry = store.root / "cache" / "trees" / nb.id
    entry.mkdir(parents=True)
    stale = {
        "format": 1,
        "source_sha256": nb.id,
        "meta": {"extractor_version": 1},
        "warnings": [],
        "files": [],
    }
    (entry / "manifest.json").write_text(json.dumps(stale))
    for link in (entry / "files", entry, entry.parent, store.root / "cache"):  # files/, <id>/, trees/, cache/
        shutil.rmtree(link, ignore_errors=True)
        symlink(link, docs)
        with pytest.raises(StoreError) as e:
            tree_for(store, nb)
        assert link.name in str(e.value)
        assert sorted(p.name for p in docs.iterdir()) == ["notes.txt", "thesis.docx"], link
        link.unlink()
    assert tree_for(store, nb).text("cells/cell_000.py") == "x = 1\n"


def test_tree_for_never_follows_links_inside_its_cache_entry(store: Store, tmp_path: Path):
    nb, _ = register_bytes(store, notebook("x = 1\n"), name="main.ipynb")
    fresh = tree_for(store, nb)
    entry = store.root / "cache" / "trees" / nb.id
    content = entry / "files" / fresh.files["cells/cell_000.py"].sha256

    # a content file linked to a file elsewhere is neither read nor kept
    secret = tmp_path / "credentials"
    secret.write_text("aws_secret_access_key = outside\n")
    content.unlink()
    symlink(content, secret)
    assert tree_for(store, nb).text("cells/cell_000.py") == "x = 1\n"
    assert not content.is_symlink() and secret.read_text() == "aws_secret_access_key = outside\n"

    # dangling links where the next save writes (a content file, the temporary manifest): nothing is created
    planted = tmp_path / "planted"
    planted.mkdir()
    content.unlink()
    symlink(content, planted / "content")
    symlink(entry / "manifest.json.tmp", planted / "manifest")
    assert tree_for(store, nb).text("cells/cell_000.py") == "x = 1\n"
    assert list(planted.iterdir()) == []
    assert not content.is_symlink() and not (entry / "manifest.json").is_symlink()

    # a linked manifest is not read either, even one that looks fresh
    crafted = tmp_path / "crafted.json"
    planted_manifest = {
        "format": 1,
        "source_sha256": nb.id,
        "meta": {"extractor_version": EXTRACTOR_VERSION},
        "warnings": ["planted"],
        "files": [],
    }
    crafted.write_text(json.dumps(planted_manifest))
    (entry / "manifest.json").unlink()
    symlink(entry / "manifest.json", crafted)
    t = tree_for(store, nb)
    assert t.paths() == ["cells/cell_000.py"] and t.warnings == []
    assert not (entry / "manifest.json").is_symlink() and json.loads(crafted.read_text()) == planted_manifest

    # a rebuild removes stale contents, but only regular files directly inside files/
    stale = entry / "files" / ("0" * 64)
    stale.write_text("stale\n")
    (entry / "files" / "sub").mkdir()
    (entry / "files" / "sub" / "inner.txt").write_text("not a cached content\n")
    (entry / "manifest.json").unlink()
    tree_for(store, nb)
    assert not stale.exists()
    assert (entry / "files" / "sub" / "inner.txt").read_text() == "not a cached content\n"


def test_tree_for_reads_no_store_notebooks_from_their_path(store: Store):
    f = store.root.parent / "nbs" / "main.ipynb"
    f.parent.mkdir()
    f.write_bytes(notebook("y = 2\n"))
    nb, _ = register_file(store, f, keep_bytes=False)
    assert not nb.stored and not store.has_blob(nb.id)
    assert tree_for(store, nb).text("cells/cell_000.py") == "y = 2\n"

    shutil.rmtree(store.root / "cache" / "trees" / nb.id)
    f.write_bytes(notebook("y = 3\n"))
    with pytest.raises(StoreError, match=r"sha256 differs.*without --no-store"):
        tree_for(store, nb)
    f.unlink()
    with pytest.raises(StoreError, match=r"no longer exists.*without --no-store"):
        tree_for(store, nb)

    stored, _ = register_bytes(store, notebook("z = 4\n"), name="kept.ipynb")
    store.blob_path(stored.id).unlink()  # e.g. the blobs were not pulled yet
    with pytest.raises(StoreError, match=r"missing from the store's blobs"):
        tree_for(store, stored)


def test_tree_for_detects_scripts_by_name(store: Store):
    nb, _ = register_bytes(store, b"import os\nprint(os.getcwd())\n", name="kernel.py")
    t = tree_for(store, nb)
    assert t.meta["format"] == "script" and t.text("script.py").startswith("import os")


def test_only_ipynb_and_py_suffixes_pick_the_format():
    # any other name (v1.5 has the suffix ".5") leaves it to the bytes: a JSON object with "cells" or
    # "worksheets" is a notebook, anything else a script
    data = notebook("x = 1\n")
    for name in ("v1.5", "qwen2.5-7b", "agent.irnb", "solver v2.1", "", "NEXT.IPYNB"):
        assert extract_bytes(data, name).meta["format"] == "ipynb", name
    v3 = json.dumps({"nbformat": 3, "worksheets": [{"cells": [{"cell_type": "code", "input": "y = 2"}]}]})
    assert extract_bytes(v3.encode(), "old.v3").text("cells/cell_000.py") == "y = 2"
    assert extract_bytes(b"\xef\xbb\xbf \n" + data, "bom.v2").meta["format"] == "ipynb"
    for content in (b"import os\n", b'{"seed": 7}', b"[1, 2]", b"{" * 100_000, b"\xff\xfe{garbage"):
        assert extract_bytes(content, "v1.5").meta["format"] == "script", content[:20]
    assert extract_bytes(data, "kernel.py").meta["format"] == "script"  # a recognised suffix wins
    with pytest.raises(ValueError, match="not a valid notebook"):
        extract_bytes(b"import os\n", "agent.ipynb")


def test_a_dotted_name_does_not_make_a_notebook_a_script(store: Store):
    data = notebook("TEMPERATURE = 0.7\n", "%%writefile agent.py\nSTEPS = 64\n")
    f = store.root.parent / "nbs" / "next.ipynb"
    f.parent.mkdir()
    f.write_bytes(data)
    nb, _ = register_file(store, f, name="v1.5")
    t = tree_for(store, nb)
    assert t.meta["format"] == "ipynb"
    assert t.paths() == ["cells/cell_000.py", "cells/cell_001.py", "files/agent.py"]
    assert t.grep("TEMPERATURE = 0.7") == [Hit("cells/cell_000.py", 1, "TEMPERATURE = 0.7")]

    # renamed later (`nb update --name`) and read on a clone whose cache/ is empty
    store.append("nb.update", {"id": nb.id, "name": "solver v2.1"})
    shutil.rmtree(store.root / "cache" / "trees" / nb.id)
    renamed = store.state().notebooks[nb.id]
    assert renamed.name == "solver v2.1" and tree_for(store, renamed).meta["format"] == "ipynb"

    # the path comes first, then the name, then the bytes
    cases = [
        (notebook("a = 1\n"), "main.ipynb.bak", "kernel.py", "script"),
        (b"import os\n", "kernel.py", "qwen2.5-7b", "script"),
        (notebook("b = 2\n"), "agent.irnb", "llama3.1-sft", "ipynb"),
        (notebook("c = 3\n"), "", "exp-0.7", "ipynb"),
        (b"import sys\n", "", "exp-0.8", "script"),
    ]
    for content, path, name, fmt in cases:
        nb, _ = register_bytes(store, content, name=name, path=path)
        assert tree_for(store, nb).meta["format"] == fmt, (path, name)


def test_check_with_a_dotted_name_reads_the_notebook(store: Store):
    # `check FILE --name qwen2.5-7b` once read the .ipynb as a script and missed a rejected idea in it
    fp = r"re:^sampling = SamplingParams\(temperature=0\.7"
    store.append(
        "idea.add", {"id": "temp-07", "title": "t = 0.7", "status": "rejected", "fingerprints": [fp]}
    )
    f = store.root.parent / "next.ipynb"
    f.write_bytes(notebook("from vllm import SamplingParams\nsampling = SamplingParams(temperature=0.7)\n"))
    args = ["--store", str(store.root), "check", str(f), "--name", "qwen2.5-7b", "--llm", "never"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1, result.output
    assert "FORBIDDEN PRESENT" in result.output and "temp-07" in result.output
