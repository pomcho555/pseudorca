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
import tracemalloc
import zipfile
import zlib
from collections.abc import Iterable
from pathlib import Path

import pytest

import pseudorca.extract as extract_mod
from pseudorca.extract import EXTRACTOR_VERSION, Limits, extract_bytes, extract_file, tree_for
from pseudorca.extract.magics import transform_python
from pseudorca.notebooks import register_bytes, register_file
from pseudorca.store import Store, StoreError
from pseudorca.tree import Tree

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


def test_shell_and_other_cell_magics():
    t = extract(
        "%%bash\necho hi\nls -la\n",
        "%%sh\nexport X=1\n",
        "%%script bash --no-raise-error\necho via script\n",
        "%%script /usr/bin/env sh\necho via env\n",
        "%%html\n<b>bold</b>\n",
        "%%javascript\nconsole.log(1)\n",
    )
    assert t.text("cells/cell_000.sh") == "echo hi\nls -la\n"
    assert t.text("cells/cell_001.sh") == "export X=1\n"
    assert t.text("cells/cell_002.sh") == "echo via script\n"
    assert t.text("cells/cell_003.sh") == "echo via env\n"
    assert t.text("cells/cell_004.html.txt") == "<b>bold</b>\n"
    assert t.text("cells/cell_005.javascript.txt") == "console.log(1)\n"
    assert t.files["cells/cell_000.sh"].origin == "cell 0 > %%bash"
    assert not t.paths("cells/*.py") and t.warnings == []


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
    assert t.files[path].origin == "cell 0 > %%bash > line 1 str > base64 > gzip"


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
