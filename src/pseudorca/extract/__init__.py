"""Static extraction: notebook or script bytes -> Tree. Nothing is ever executed or unpickled.

A notebook's code cells (IPython magics turned into plain Python), its markdown, the files it writes with
literal paths and contents (%%writefile, Path.write_text, open().write) and the payloads embedded in its
string constants (base64/base85/hex, zlib/gzip/xz/lzma/bz2, tar/zip) become files of a virtual Tree. See
`pseudorca.tree` for the path scheme.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pseudorca import util
from pseudorca.store import StoreError
from pseudorca.tree import Tree

if TYPE_CHECKING:
    from pseudorca.model import Notebook
    from pseudorca.store import Store

EXTRACTOR_VERSION = 2  # bump when extraction output changes, so cached trees are rebuilt
FORMAT_SUFFIXES = {".ipynb": "ipynb", ".py": "script"}  # the only suffixes that pick the format


@dataclass(frozen=True)
class Limits:
    min_payload_chars: int = 200  # shorter string constants are not tried as payloads
    max_payload_bytes: int = 64 << 20  # decoded size cap per payload, after decompression
    max_total_bytes: int = 256 << 20  # cap on everything decoded from one notebook
    max_depth: int = 4  # decode steps per payload (base64 > lzma > tar counts as 3)
    max_members: int = 5000  # archive members per notebook
    max_nesting: int = 3  # how often decoded or written Python is searched again for payloads and writes
    max_parse_chars: int = 4 << 20  # longer decoded Python is searched by pattern instead of parsed


def suffix_format(name: str | None) -> str:
    """The format a name ending in .ipynb ("ipynb") or .py ("script") picks, in any case; else "".

    Any other suffix decides nothing: `v1.5`, `qwen2.5-7b` and `solver v2.1` are labels, not formats.
    """
    return FORMAT_SUFFIXES.get(PurePosixPath(name or "").suffix.lower(), "")


def extract_bytes(data: bytes, name: str = "notebook.ipynb", limits: Limits | None = None) -> Tree:
    """Extract a notebook (.ipynb JSON) or a script given as bytes.

    A `name` ending in .ipynb or .py picks the format. Any other name leaves it to the bytes: a JSON object
    with "cells" or "worksheets" is a notebook, anything else a script. Raises ValueError when an .ipynb is
    not notebook JSON. Hostile payloads never raise: what cannot be decoded within `limits` is recorded in
    `tree.warnings`.
    """
    from pseudorca.extract.engine import Extractor
    from pseudorca.extract.notebook import sniff_notebook

    ex = Extractor(limits or Limits())
    fmt = suffix_format(name)
    doc = None if fmt else sniff_notebook(data)
    if fmt == "ipynb" or doc is not None:
        ex.notebook(data if doc is None else doc, PurePosixPath(name or "").name or "notebook")
        fmt = "ipynb"
    else:
        ex.script(data)
        fmt = "script"
    tree = ex.finish()
    tree.source_sha256 = util.sha256_bytes(data)
    tree.meta = {"extractor_version": EXTRACTOR_VERSION, "format": fmt, "cells": ex.cells, "env": ex.env}
    return tree


def extract_file(path: Path, limits: Limits | None = None) -> Tree:
    p = Path(path)
    return extract_bytes(p.read_bytes(), p.name, limits)


def tree_for(store: Store, nb: Notebook) -> Tree:
    """The tree of a registered notebook: from cache/trees/<id>/ when fresh, else extracted and cached."""
    if not util.HEX_RE.match(nb.id or ""):
        raise StoreError(f"notebook id {str(nb.id)[:80]!r} is not a sha256")
    cache = store.cache_dir("trees", nb.id)
    tree = Tree.load(cache)
    if (
        tree is not None
        and tree.meta.get("extractor_version") == EXTRACTOR_VERSION
        and tree.source_sha256 == nb.id
    ):
        return tree
    data = _notebook_bytes(store, nb)
    tree = extract_bytes(data, _format_name(nb))
    tree.save(cache)
    _prune(cache, tree)
    return tree


def _notebook_bytes(store: Store, nb: Notebook) -> bytes:
    """The notebook's bytes: its blob, else (added with --no-store) its file if that still has the same sha256."""
    if store.has_blob(nb.id):
        return store.get_blob(nb.id)
    path = store.root.parent / nb.path if nb.path else None
    if path is not None and path.is_file():
        data = None if nb.size and path.stat().st_size != nb.size else path.read_bytes()
        if data is not None and util.sha256_bytes(data) == nb.id:
            return data
        problem = f"{nb.path} has changed since it was added (its sha256 differs)"
    else:
        problem = f"{nb.path} no longer exists" if nb.path else "no file path was recorded for it"
    if nb.stored:
        raise StoreError(
            f"the bytes of notebook {nb.label} are missing from the store's blobs and {problem}; "
            "pull the latest store, or add the notebook again"
        )
    raise StoreError(
        f"notebook {nb.label} was added with --no-store and {problem}; restore that version, "
        "or add the notebook again without --no-store so the store keeps its bytes"
    )


def _format_name(nb: Notebook) -> str:
    """The name to extract a registered notebook under: its path, else its name, whichever ends in .ipynb
    or .py. With neither, extract_bytes looks at the bytes, so a name like `v1.5` cannot turn a notebook
    into a script."""
    names = [n for n in (nb.path, nb.name) if n]
    return next((n for n in names if suffix_format(n)), names[0] if names else "")


def _prune(directory: Path, tree: Tree) -> None:
    """Remove cached contents a previous extraction left behind and this tree no longer uses."""
    keep = {f.sha256 for f in tree.files.values() if not f.binary}
    with contextlib.suppress(OSError):
        for p in (directory / "files").iterdir():
            if p.name not in keep:
                p.unlink()
