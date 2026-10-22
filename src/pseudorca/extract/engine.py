"""Building a Tree from a notebook or a script: cells, magics, literal file writes and embedded payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pseudorca.extract import magics
from pseudorca.extract.notebook import read_notebook
from pseudorca.extract.payloads import (
    Budget,
    Candidate,
    Decoder,
    Result,
    ast_candidates,
    brief,
    guess_text_ext,
    label,
    pattern_candidates,
)
from pseudorca.extract.writes import Write, find_env, find_writes, parse_python
from pseudorca.tree import Tree, safe_path

if TYPE_CHECKING:
    from pseudorca.extract import Limits

MAX_WARNINGS = 200
_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,10}")


@dataclass(frozen=True)
class _Source:
    """Text being scanned: its cell, its origin and where its payloads go."""

    cell: int | None  # None for a script
    origin: str  # e.g. "cell 4" or "cell 4 > %%writefile src/agent.py"
    pdir: str  # e.g. "payloads/cell_004"
    level: int = 0  # how many decode or write steps away from the notebook's own text

    def nested(self, origin: str) -> _Source:
        return _Source(self.cell, origin, self.pdir, self.level + 1)


class Extractor:
    """Collects one notebook's (or script's) files into a Tree. Nothing is ever run."""

    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.tree = Tree()
        self.env: list[dict[str, Any]] = []
        self.cells = 0
        self.budget = Budget(limits.max_total_bytes, limits.max_members)
        self.decoder = Decoder(limits, self.budget)
        self._payload_count: dict[str, int] = {}
        self._said: set[str] = set()
        self._dropped = 0

    # ----- entry points ----------------------------------------------------------------------------

    def notebook(self, data: bytes, name: str) -> None:
        """Extract an .ipynb (raises ValueError when it is not notebook JSON)."""
        nb = read_notebook(data, name)
        self.cells = nb.count
        for w in nb.warnings:
            self.warn(w)
        ext = nb.file_extension if _EXT_RE.fullmatch(nb.file_extension) else ".txt"
        if not nb.is_python:
            self.warn(
                f"kernel language is {label(nb.language)!r}: code cells were kept as {ext} files "
                "and searched for payloads by pattern only"
            )
        for cell in nb.cells:
            try:
                if cell.kind == "markdown" and cell.source.strip():
                    self.tree.add_text(
                        f"markdown/cell_{cell.index:03d}.md", cell.source, f"cell {cell.index}"
                    )
                elif cell.kind == "code" and cell.source.strip():
                    if nb.is_python:
                        self._code_cell(cell.index, cell.source)
                    else:
                        src = self._cell_source(cell.index)
                        self.tree.add_text(f"cells/cell_{cell.index:03d}{ext}", cell.source, src.origin)
                        self._scan_text(cell.source, src)
            except Exception as e:  # one broken cell must not cost the others
                self.warn(f"cell {cell.index}: extraction failed ({brief(e)})")

    def script(self, data: bytes) -> None:
        """Extract a script kernel: `script.py`, plus its writes and payloads."""
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("utf-8", "replace")
            self.warn("script: not valid UTF-8; undecodable bytes were replaced")
        self.tree.add_text("script.py", text, "script")
        try:
            self._scan_python(text, _Source(None, "script", "payloads/script"))
        except Exception as e:
            self.warn(f"script: extraction failed ({brief(e)})")

    def finish(self) -> Tree:
        if self._dropped:
            self.tree.warnings.append(f"{self._dropped} more warnings were not recorded")
        return self.tree

    # ----- warnings --------------------------------------------------------------------------------

    def warn(self, message: str) -> None:
        if len(self.tree.warnings) < MAX_WARNINGS:
            self.tree.warnings.append(message if len(message) <= 1000 else message[:999] + "…")
        else:
            self._dropped += 1

    def warn_once(self, key: str, message: str) -> None:
        if key not in self._said:
            self._said.add(key)
            self.warn(message)

    # ----- cells -----------------------------------------------------------------------------------

    @staticmethod
    def _cell_source(i: int) -> _Source:
        return _Source(i, f"cell {i}", f"payloads/cell_{i:03d}")

    def _code_cell(self, i: int, source: str) -> None:
        stem = f"cells/cell_{i:03d}"
        src = self._cell_source(i)
        magic = magics.cell_magic(source)
        kind = "python" if magic is None else magics.classify(magic)
        if magic is None or kind == "python":
            text, env = magics.transform_python(source)
            self.tree.add_text(f"{stem}.py", text, src.origin)
            self._scan_python(text, src, env=env)
            return
        if kind == "writefile":
            target = magics.writefile_target(magic.args)
            if target is not None:
                self._writefile(i, magic, *target)
                return
            self.warn(f"cell {i}: %%{label(magic.name)} without a file name")
        body = _Source(i, f"cell {i} > %%{label(magic.name)}", src.pdir)
        if kind == "shell":
            self.tree.add_text(f"{stem}.sh", magic.body, body.origin)
        else:
            slug = _SLUG_RE.sub("_", magic.name).strip("_") or "magic"
            self.tree.add_text(f"{stem}.{slug[:40]}.txt", magic.body, body.origin)
        self._scan_text(magic.body, body)

    def _writefile(self, i: int, magic: magics.CellMagic, path: str, append: bool) -> None:
        target = "files/" + safe_path(path)
        origin = f"cell {i} > %%{label(magic.name)} {'-a ' if append else ''}{label(path)}"
        existing = self.tree.files.get(target)
        offset = 0
        if append and existing is not None and not existing.binary:
            offset = existing.text.count("\n")
            self.tree.add_text(
                target, existing.text + magic.body, f"{existing.origin}; {origin}", replace=True
            )
        else:
            self.tree.add_text(target, magic.body, origin, replace=True)
        self.tree.add_text(
            f"cells/cell_{i:03d}.py", f"# {label(magic.head, 500)}  -> {target}\n", f"cell {i}"
        )
        body = _Source(i, origin, f"payloads/cell_{i:03d}")
        if target.endswith(".py"):
            self._scan_python(magic.body, body, offset=offset)
        else:
            self._scan_text(magic.body, body, offset=offset)

    # ----- scanning --------------------------------------------------------------------------------

    def _scan_python(
        self,
        text: str,
        src: _Source,
        *,
        env: list[tuple[int, str, str]] | None = None,
        offset: int = 0,
    ) -> None:
        """Environment settings, literal writes and payloads of Python code (a pattern scan if it does
        not parse)."""
        if src.level and len(text) > self.limits.max_parse_chars:
            module, why = None, f"not parsed (longer than max_parse_chars {self.limits.max_parse_chars})"
        else:
            module, error = parse_python(text)
            why = f"not valid Python ({error})"
        settings = [(line, 0, key, value) for line, key, value in env or ()]
        writes: list[Write] = []
        if module is None:
            self.warn(f"{src.origin}: {why}; searched for payloads by pattern only")
            candidates = pattern_candidates(text, self.limits.min_payload_chars)
        else:
            settings.extend(find_env(module))
            writes = find_writes(module)
            candidates = ast_candidates(module, self.limits.min_payload_chars)
            del module
        for _, _, key, value in sorted(settings, key=lambda s: (s[0], s[1])):
            self.env.append({"cell": src.cell, "key": key, "value": value})
        for w in writes:
            self._write(w, src)
        self._payloads(candidates, src, offset)

    def _scan_text(self, text: str, src: _Source, *, offset: int = 0) -> None:
        """Payloads in non-Python text (shell, other cell magics, written data files)."""
        self._payloads(pattern_candidates(text, self.limits.min_payload_chars, bare_runs=True), src, offset)

    def _scan_nested(self, text: str, src: _Source, *, offset: int = 0) -> None:
        """Python that was decoded or written by the notebook, scanned again up to max_nesting levels."""
        if src.level > self.limits.max_nesting:
            self.warn(
                f"{src.origin}: not searched for nested payloads (max_nesting {self.limits.max_nesting})"
            )
            return
        self._scan_python(text, src, offset=offset)

    def _write(self, w: Write, src: _Source) -> None:
        target = "files/" + safe_path(w.path)
        origin = f"{src.origin} > {w.via} {label(w.path)}"
        existing = self.tree.files.get(target)
        base = existing if w.append and existing is not None and not existing.binary else None
        if isinstance(w.content, bytes):
            data = w.content if base is None else base.text.encode("utf-8", "replace") + w.content
            self.tree.add_bytes(
                target, data, origin if base is None else f"{base.origin}; {origin}", replace=True
            )
            return
        if base is None:
            self.tree.add_text(target, w.content, origin, replace=True)
            offset = 0
        else:
            self.tree.add_text(target, base.text + w.content, f"{base.origin}; {origin}", replace=True)
            offset = base.text.count("\n")
        if target.endswith(".py"):
            self._scan_nested(w.content, src.nested(origin), offset=offset)

    def _payloads(self, candidates: list[Candidate], src: _Source, offset: int) -> None:
        consumed: set[int] = set()
        for cand in candidates:
            if cand.node and cand.node in consumed:
                continue
            if self.budget.total_left <= 0:
                self.warn_once(
                    "total",
                    f"max_total_bytes ({self.limits.max_total_bytes}) reached: later payloads were not decoded",
                )
                return
            kind = "bytes" if isinstance(cand.value, bytes) else "str"
            origin = f"{src.origin} > line {cand.line + offset} {kind}"
            try:
                res = self.decoder.decode(cand.value, origin)
            except Exception as e:  # hostile input must never stop the extraction
                self.warn(f"{origin}: payload not decoded ({brief(e)})")
                continue
            if res:
                consumed |= cand.parts
                self._commit(res, src)

    def _commit(self, res: Result, src: _Source) -> None:
        """Add a decoded payload as payloads/<cell>/<k>.<ext> (or <k>/<member>), then scan its Python."""
        for w in res.warnings:
            self.warn(w)
        if not res.items:
            return
        k = self._payload_count.get(src.pdir, 0)
        self._payload_count[src.pdir] = k + 1
        single = len(res.items) == 1 and not res.items[0].rel
        python: list[tuple[str, str]] = []
        for item in res.items:
            if item.text is None:
                path = f"{src.pdir}/{k}.{item.ext}" if single else f"{src.pdir}/{k}/{item.rel}"
                self.tree.add_bytes(path, item.data, item.origin)
                continue
            if single:
                ext = guess_text_ext(item.text, self.limits.max_parse_chars)
                path, is_python = f"{src.pdir}/{k}.{ext}", ext == "py"
            else:
                path, is_python = f"{src.pdir}/{k}/{item.rel}", item.rel.endswith(".py")
            self.tree.add_text(path, item.text, item.origin)
            if is_python:
                python.append((item.text, item.origin))
        res.items.clear()
        for text, origin in python:
            self._scan_nested(text, src.nested(origin))
