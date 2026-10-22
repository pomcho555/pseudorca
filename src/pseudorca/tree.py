"""A read-only virtual file tree extracted from a notebook.

Nothing in a tree is ever executed: it is text to search, read and quote. Paths are virtual (they never
become file names on disk; the cache stores contents under their sha256), so a hostile path in an archive
cannot write anywhere.

Path scheme used by the extractor:
    cells/cell_004.py            code cell 4 after IPython magics were turned into plain Python
    cells/cell_004.sh            a %%bash / %%sh cell (the magic line kept as "# %%bash", so lines match the cell)
    cells/cell_004.<magic>.txt   any other cell magic (same line numbering)
    markdown/cell_000.md         markdown cell 0 (context for humans and the LLM; never evidence)
    files/<path>                 a file the notebook writes (%%writefile, Path.write_text, open().write)
    payloads/cell_004/<n>.<ext>  an embedded string payload after decoding (base64, zlib, lzma, ...)
    payloads/cell_004/<n>/<member>  a member of an embedded archive
    script.py                    the code of a script kernel
"""

from __future__ import annotations

import bisect
import fnmatch
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pseudorca import util

TREE_FORMAT = 1
CONTEXT_ONLY_PREFIXES = ("markdown/",)  # never used as evidence by deterministic checks


@dataclass(frozen=True)
class TreeFile:
    path: str
    text: str  # "" for binary files
    origin: str  # how it was obtained, e.g. "cell 4 > %%writefile src/agent.py"
    sha256: str
    size: int
    binary: bool = False


@dataclass(frozen=True)
class Hit:
    path: str
    line: int  # 1-based
    text: str


def safe_path(path: str) -> str:
    """Normalise an untrusted path to a relative POSIX path without '.', '..' or empty parts."""
    parts: list[str] = []
    for part in PurePosixPath(str(path).replace("\\", "/")).parts:
        if part in ("", ".", "/"):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts) or "_"


_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def split_lines(text: str) -> list[str]:
    """Lines split only at \\r\\n, \\r and \\n, as Python's tokenizer counts them (str.splitlines also splits
    at form feeds and other separators, which would shift line numbers)."""
    lines = _LINE_BREAK.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _write_new(target: Path, text: str) -> None:
    """Write `target` through a fresh temporary file: O_EXCL never follows a planted link, os.replace swaps the
    name itself, and newline="" keeps \r\n and \r exactly."""
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{util.new_ulid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with open(fd, "w", encoding="utf-8", errors="replace", newline="") as fh:
        fh.write(text)
    os.replace(tmp, target)


def normalize_line(line: str) -> str:
    """Collapse all whitespace runs to one space and strip the ends."""
    return " ".join(line.split())


def decode_text(data: bytes) -> str | None:
    """UTF-8 text if `data` looks like text, else None (binary)."""
    if b"\x00" in data[:8192]:
        return None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    sample = text[:8192]
    if not sample:
        return text
    printable = sum(1 for c in sample if c.isprintable() or c in "\n\r\t\f\v")
    return text if printable / len(sample) >= 0.95 else None


class Tree:
    """Files extracted from one notebook, with search helpers."""

    def __init__(
        self, source_sha256: str = "", warnings: Iterable[str] = (), meta: dict[str, Any] | None = None
    ):
        self.source_sha256 = source_sha256
        self.files: dict[str, TreeFile] = {}
        self.warnings: list[str] = list(warnings)
        self.meta: dict[str, Any] = dict(meta or {})
        self._norm: dict[str, list[tuple[int, str]]] = {}
        self._join: dict[str, tuple[str, list[int], list[int]]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def __contains__(self, path: object) -> bool:
        return path in self.files

    # ----- building --------------------------------------------------------------------------------

    def _unique(self, path: str) -> str:
        if path not in self.files:
            return path
        n = 2
        while f"{path}~{n}" in self.files:
            n += 1
        return f"{path}~{n}"

    def add_text(self, path: str, text: str, origin: str, *, replace: bool = False) -> TreeFile:
        """Add a text file. An existing path gets a `~2`-style suffix unless `replace` is true."""
        path = safe_path(path)
        if not replace:
            path = self._unique(path)
        raw = text.encode("utf-8", "replace")
        f = TreeFile(path=path, text=text, origin=origin, sha256=util.sha256_bytes(raw), size=len(raw))
        self.files[path] = f
        self._norm.pop(path, None)
        self._join.pop(path, None)
        return f

    def add_bytes(self, path: str, data: bytes, origin: str, *, replace: bool = False) -> TreeFile:
        """Add a file from bytes: text if it decodes as text, otherwise a binary entry without content."""
        text = decode_text(data)
        if text is not None:
            return self.add_text(path, text, origin, replace=replace)
        path = safe_path(path)
        if not replace:
            path = self._unique(path)
        f = TreeFile(
            path=path, text="", origin=origin, sha256=util.sha256_bytes(data), size=len(data), binary=True
        )
        self.files[path] = f
        self._norm.pop(path, None)
        self._join.pop(path, None)
        return f

    # ----- reading ---------------------------------------------------------------------------------

    def paths(self, pattern: str = "*") -> list[str]:
        """Sorted paths matching a glob (`*` also matches `/`)."""
        return sorted(p for p in self.files if fnmatch.fnmatchcase(p, pattern))

    def evidence_paths(self) -> list[str]:
        """Text files that can count as evidence (not markdown, not binary)."""
        return [
            p for p in self.paths() if not self.files[p].binary and not p.startswith(CONTEXT_ONLY_PREFIXES)
        ]

    def text(self, path: str) -> str:
        return self.files[path].text

    def lines(self, path: str) -> list[str]:
        return split_lines(self.files[path].text)

    def read(self, path: str, start: int = 1, end: int | None = None) -> str:
        """Lines `start`..`end` (1-based, inclusive) with line numbers, like `cat -n`."""
        lines = self.lines(path)
        start = max(1, start)
        end = len(lines) if end is None else min(end, len(lines))
        width = len(str(max(end, 1)))
        return "\n".join(f"{i:>{width}}| {lines[i - 1]}" for i in range(start, end + 1))

    def grep(
        self,
        pattern: str,
        *,
        regex: bool = False,
        ignore_case: bool = False,
        path_glob: str = "*",
        paths: Iterable[str] | None = None,
        max_hits: int = 200,
    ) -> list[Hit]:
        """Lines containing `pattern` (a substring, or a regex when `regex` is true).

        Only trusted callers may pass regex=True: Python regexes have no timeout.
        """
        flags = re.IGNORECASE if ignore_case else 0
        rx = re.compile(pattern if regex else re.escape(pattern), flags)
        hits: list[Hit] = []
        for p in self.paths(path_glob) if paths is None else paths:
            f = self.files.get(p)
            if f is None or f.binary:
                continue
            for i, line in enumerate(split_lines(f.text), 1):
                if rx.search(line):
                    hits.append(Hit(p, i, line))
                    if len(hits) >= max_hits:
                        return hits
        return hits

    def _normalized(self, path: str) -> list[tuple[int, str]]:
        if path not in self._norm:
            self._norm[path] = [
                (i, normalize_line(line))
                for i, line in enumerate(split_lines(self.files[path].text), 1)
                if line.strip()
            ]
        return self._norm[path]

    def find_snippet(
        self, snippet: str, paths: Iterable[str] | None = None, *, partial_ends: bool = True
    ) -> list[tuple[str, int]]:
        """Where a snippet occurs, ignoring whitespace differences and blank lines.

        With `partial_ends` the snippet's first line may be the end of a line and its last line the start
        of one (a quote cut mid-line); otherwise every line must match a whole line. A one-line snippet with
        `partial_ends` matches anywhere inside a line. Returns (path, line of the snippet's first line).
        """
        want = [normalize_line(line) for line in split_lines(snippet) if line.strip()]
        if not want:
            return []
        # The file's normalized non-blank lines joined by "\n": a substring match of the joined snippet is
        # exactly a match with partial ends (the first line may end a line, the last may start one); padding
        # both with "\n" makes every line whole. One str.find pass per occurrence keeps this linear.
        needle = "\n".join(want)
        if not partial_ends:
            needle = f"\n{needle}\n"
        found: list[tuple[str, int]] = []
        for p in self.paths() if paths is None else paths:
            f = self.files.get(p)
            if f is None or f.binary:
                continue
            joined, offsets, numbers = self._joined(p)
            hay = joined if partial_ends else f"\n{joined}\n"
            start = hay.find(needle)
            while start >= 0:
                found.append((p, numbers[bisect.bisect_right(offsets, start) - 1]))
                start = hay.find(needle, start + 1)
        return found

    def _joined(self, path: str) -> tuple[str, list[int], list[int]]:
        if path not in self._join:
            norm = self._normalized(path)
            offsets, pos = [], 0
            for _, text in norm:
                offsets.append(pos)
                pos += len(text) + 1
            self._join[path] = ("\n".join(t for _, t in norm), offsets, [i for i, _ in norm])
        return self._join[path]

    # ----- cache -----------------------------------------------------------------------------------

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {"path": f.path, "origin": f.origin, "sha256": f.sha256, "size": f.size, "binary": f.binary}
            for f in (self.files[p] for p in self.paths())
        ]

    def save(self, directory: Path) -> None:
        """Write the tree to a cache directory: contents under their sha256, plus manifest.json."""
        content_dir = directory / "files"
        content_dir.mkdir(parents=True, exist_ok=True)
        for f in self.files.values():
            target = content_dir / f.sha256
            if f.binary:
                continue
            if target.is_symlink():
                target.unlink()
            if not target.is_file():
                _write_new(target, f.text)
        manifest = {
            "format": TREE_FORMAT,
            "source_sha256": self.source_sha256,
            "meta": self.meta,
            "warnings": self.warnings,
            "files": self.manifest(),
        }
        _write_new(directory / "manifest.json", json.dumps(manifest, ensure_ascii=True))

    @classmethod
    def load(cls, directory: Path) -> Tree | None:
        """A tree saved with `save`, or None if there is none or it has another format."""
        mp = directory / "manifest.json"
        if mp.is_symlink() or not mp.is_file():
            return None
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            if m.get("format") != TREE_FORMAT:
                return None
            tree = cls(m.get("source_sha256", ""), m.get("warnings", []), m.get("meta", {}))
            content_dir = directory / "files"
            for e in m["files"]:
                # the cache is untrusted input: content names must be hashes, contents regular files
                if not (
                    util.is_sha256(e["sha256"])
                    and isinstance(e["path"], str)
                    and isinstance(e["origin"], str)
                    and isinstance(e["binary"], bool)
                    and isinstance(e["size"], int)
                ):
                    return None
                text = ""
                if not e["binary"]:
                    f = content_dir / e["sha256"]
                    if f.is_symlink() or not f.is_file():
                        return None
                    with open(f, encoding="utf-8", newline="") as fh:  # keep \r\n and \r as they were
                        text = fh.read()
                tree.files[e["path"]] = TreeFile(
                    e["path"], text, e["origin"], e["sha256"], e["size"], e["binary"]
                )
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return tree
