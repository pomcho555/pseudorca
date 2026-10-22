"""Reading Jupyter notebooks (nbformat 4, and 3 on a best-effort basis) into plain cells. Outputs are ignored."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Cell:
    index: int  # 0-based position in the notebook
    kind: str  # "code", "markdown", "raw", ...
    source: str


@dataclass
class NotebookCells:
    cells: list[Cell]
    count: int  # cells in the notebook, including unreadable ones
    language: str = ""  # kernel language, lower-case ("" when the notebook does not say)
    file_extension: str = ""  # e.g. ".r", from language_info
    warnings: list[str] = field(default_factory=list)

    @property
    def is_python(self) -> bool:
        return not self.language or "python" in self.language


def _text(value: Any) -> str | None:
    """A cell source: a string or a list of strings (None for anything else)."""
    if value is None:
        return ""
    if isinstance(value, list):
        value = "".join(v for v in value if isinstance(v, str))
    return _encodable(value) if isinstance(value, str) else None


def _encodable(text: str) -> str:
    """Lone surrogates (JSON allows "\\ud800"; UTF-8 does not) replaced by "?", keeping every offset."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", "replace").decode("utf-8")
    return text


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def read_notebook(data: bytes, name: str = "notebook.ipynb") -> NotebookCells:
    """The cells of a notebook. Raises ValueError when the bytes are not notebook JSON."""
    try:
        nb = json.loads(data)
    except (ValueError, RecursionError) as e:
        raise ValueError(f"{name} is not a valid notebook: its JSON cannot be read ({e})") from None
    if not isinstance(nb, dict):
        raise ValueError(f"{name} is not a Jupyter notebook: the JSON is not an object")
    raw = nb.get("cells")
    if not isinstance(raw, list):
        sheets = nb.get("worksheets")  # nbformat 3
        if not isinstance(sheets, list):
            raise ValueError(f"{name} is not a Jupyter notebook: it has no cells")
        raw = []
        for sheet in sheets:
            cells = _dict(sheet).get("cells")
            if isinstance(cells, list):
                raw.extend(cells)

    meta = _dict(nb.get("metadata"))
    kernelspec, info = _dict(meta.get("kernelspec")), _dict(meta.get("language_info"))
    language = kernelspec.get("language") or info.get("name") or meta.get("language") or ""
    out = NotebookCells(
        cells=[],
        count=len(raw),
        language=str(language).strip().lower(),
        file_extension=str(info.get("file_extension") or ""),
    )
    for i, cell in enumerate(raw):
        if not isinstance(cell, dict):
            out.warnings.append(f"cell {i}: not a cell object; skipped")
            continue
        kind = str(cell.get("cell_type") or "")
        if kind == "code":
            source = _text(cell["source"] if "source" in cell else cell.get("input"))  # v4 / v3
        elif kind == "heading":  # nbformat 3
            level = cell.get("level")
            hashes = "#" * (level if isinstance(level, int) and 1 <= level <= 6 else 1)
            text = _text(cell.get("source"))
            source, kind = (None if text is None else f"{hashes} {text}"), "markdown"
        else:
            source = _text(cell.get("source"))
        if source is None:
            out.warnings.append(f"cell {i}: its source is neither text nor a list of lines; skipped")
            continue
        out.cells.append(Cell(i, kind, source))
    return out
