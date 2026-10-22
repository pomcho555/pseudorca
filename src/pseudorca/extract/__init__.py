"""Static extraction: notebook or script bytes -> Tree. Nothing is ever executed or unpickled.

CONTRACT STUB: the extraction module replaces the bodies below and keeps these signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pseudorca.tree import Tree

if TYPE_CHECKING:
    from pseudorca.model import Notebook
    from pseudorca.store import Store

EXTRACTOR_VERSION = 1  # bump when extraction output changes, so cached trees are rebuilt


@dataclass(frozen=True)
class Limits:
    min_payload_chars: int = 200  # shorter string constants are not tried as payloads
    max_payload_bytes: int = 64 << 20  # decoded size cap per payload, after decompression
    max_total_bytes: int = 256 << 20  # cap on everything decoded from one notebook
    max_depth: int = 4  # decode steps per payload (base64 > lzma > tar counts as 3)
    max_members: int = 5000  # archive members per notebook


def extract_bytes(data: bytes, name: str = "notebook.ipynb", limits: Limits | None = None) -> Tree:
    """Extract a notebook (.ipynb JSON) or a script (.py) given as bytes; `name`'s suffix picks the format."""
    raise NotImplementedError


def extract_file(path: Path, limits: Limits | None = None) -> Tree:
    p = Path(path)
    return extract_bytes(p.read_bytes(), p.name, limits)


def tree_for(store: Store, nb: Notebook) -> Tree:
    """The tree of a registered notebook: from cache/trees/<id>/ when fresh, else extracted and cached."""
    raise NotImplementedError
