"""Registering notebook files in the ledger (shared by `nb add`, `check` and `kaggle pull`)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pseudorca import util
from pseudorca.model import Notebook
from pseudorca.store import Store


def display_path(store: Store, path: Path) -> str:
    """The path to record: relative to the repository holding the store, else only the file name.

    Absolute paths would put local user names into a shared ledger.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(store.root.parent).as_posix()
    except ValueError:
        return p.name


def register_file(
    store: Store, path: Path, *, name: str | None = None, **kwargs: Any
) -> tuple[Notebook, bool]:
    """Register a notebook file; see `register_bytes`."""
    p = Path(path)
    return register_bytes(store, p.read_bytes(), name=name or p.name, path=display_path(store, p), **kwargs)


def register_bytes(
    store: Store,
    data: bytes,
    *,
    name: str,
    path: str = "",
    parent: str | None = None,
    applies: Iterable[str] = (),
    tags: Iterable[str] = (),
    kernel: str = "",
    version: str = "",
    note: str = "",
    keep_bytes: bool = True,
) -> tuple[Notebook, bool]:
    """Add a notebook to the ledger, or merge new metadata into the existing entry with the same sha256.

    `parent` must be the full id of a registered notebook. Returns (notebook, created).
    """
    applies, tags = list(dict.fromkeys(applies)), list(dict.fromkeys(tags))
    sha = util.sha256_bytes(data)
    st = store.state()
    unknown = [a for a in applies if a not in st.ideas]
    if unknown:
        raise LookupError(f"unknown idea(s): {', '.join(unknown)} (add them with `pseudorca idea add`)")
    if parent is not None and parent not in st.notebooks:
        raise LookupError(f"parent {parent[:12]} is not a registered notebook")
    if parent == sha:
        raise ValueError("a notebook cannot be its own parent")

    existing = st.notebooks.get(sha)
    if existing is not None:
        changes: dict[str, Any] = {}
        if new_tags := [t for t in tags if t not in existing.tags]:
            changes["tags"] = [*existing.tags, *new_tags]
        if new_applies := [a for a in applies if a not in existing.applies]:
            changes["applies"] = [*existing.applies, *new_applies]
        for key, value in (("kernel", kernel), ("version", version), ("note", note), ("parent", parent)):
            if value and getattr(existing, key) != value:
                changes[key] = value
        if keep_bytes and not existing.stored:
            store.put_blob(data)
            changes["stored"] = True
        if changes:
            store.append("nb.update", {"id": sha, **changes})
        return store.state().notebooks[sha], False

    if keep_bytes:
        store.put_blob(data)
    store.append(
        "nb.add",
        {
            "id": sha,
            "name": name,
            "path": path,
            "kernel": kernel,
            "version": version,
            "parent": parent,
            "applies": applies,
            "tags": tags,
            "note": note,
            "stored": keep_bytes,
            "size": len(data),
        },
    )
    return store.state().notebooks[sha], True
