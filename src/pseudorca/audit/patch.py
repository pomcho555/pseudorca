"""Patch matching: is a unified diff applied in a notebook, reverted, or neither?

Each hunk splits into added blocks (runs of "+" lines) and removed blocks (runs of "-" lines). Blocks are
matched as whole lines after whitespace normalisation, wherever the notebook keeps the code: a diff made
against a repository file still matches the same code inlined into a cell, written with %%writefile or
embedded as a payload. Candidate files are tried in order: paths ending with the diff's path, then the same
basename, then every evidence path.

Hunk states:
    applied   every non-trivial added block is found; a removal-only hunk counts as applied when its
              removed blocks are gone and a context line is still found
    reverted  no added block is found and every removed block is; an add-only hunk counts as reverted
              when a context line is still found
    unknown   anything else (the code drifted, or the file is not in the notebook)
    trivial   only blank or trivial lines change, or only whitespace; ignored
A hunk without non-trivial context lines is located by its file instead: the file must be in the notebook
(same path or basename), or for a new or deleted file, must not be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from pseudorca.audit.types import Evidence, Finding
from pseudorca.tree import Tree, normalize_line

MAX_EVIDENCE = 5
QUOTE_CHARS = 300
TRIVIAL_LINES = frozenset(
    {"pass", "else:", "try:", "finally:", "except:", "return", "continue", "break", "..."}
)
_BRACKETS = frozenset("()[]{}<>,;:")
_HUNK_RE = re.compile(r"^@@+ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_TIMESTAMP_RE = re.compile(r"\s+\d{4}-\d\d-\d\d[ T]\d\d:\d\d.*$")
_DUP_SUFFIX_RE = re.compile(r"~\d+$")  # Tree._unique's suffix for a path written twice


@dataclass
class Hunk:
    header: str  # the "@@ ... @@" line
    context: list[str] = field(default_factory=list)
    added: list[list[str]] = field(default_factory=list)  # runs of consecutive "+" lines, without the "+"
    removed: list[list[str]] = field(default_factory=list)  # runs of consecutive "-" lines, without the "-"


@dataclass
class FilePatch:
    path: str  # the new path, or the old one for a deletion ("" when the diff names no file)
    old_path: str | None  # None for a new file (/dev/null)
    new_path: str | None  # None for a deleted file
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def is_new(self) -> bool:
        return self.old_path is None

    @property
    def is_deleted(self) -> bool:
        return self.new_path is None


def parse(diff_text: str) -> list[FilePatch]:
    """The file patches of a unified diff (git or plain). Text outside file sections and hunks is ignored.

    Hunk line counts are trusted while they last, so a removed "-- x" line is not taken for a file header
    (unless a "@@" line follows the pair). Hand-edited diffs often miscount, so after the counts run out,
    hunk lines still continue the hunk.
    """
    lines = diff_text.splitlines()
    files: list[FilePatch] = []
    fp: FilePatch | None = None
    hunk: Hunk | None = None
    run: list[str] = []
    run_kind = ""
    old_left = new_left = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        counting = hunk is not None and (old_left > 0 or new_left > 0)
        if (
            line.startswith("--- ")
            and i + 1 < len(lines)
            and lines[i + 1].startswith("+++ ")
            and (not counting or (i + 2 < len(lines) and lines[i + 2].startswith("@@")))
        ):
            old, new = _diff_path(line[4:]), _diff_path(lines[i + 1][4:])
            fp = FilePatch(new or old or "", old, new)
            files.append(fp)
            hunk = None
            i += 2
            continue
        if line.startswith("@@") and not counting:
            if fp is None:  # a bare hunk without file headers
                fp = FilePatch("", "", "")
                files.append(fp)
            hunk = Hunk(line)
            fp.hunks.append(hunk)
            m = _HUNK_RE.match(line)
            old_left = int(m.group(2) or 1) if m else 0
            new_left = int(m.group(4) or 1) if m else 0
            run_kind = ""
            i += 1
            continue
        if hunk is not None:
            tag = line[:1]
            if tag in ("+", "-"):
                if run_kind != tag:
                    run = []
                    (hunk.added if tag == "+" else hunk.removed).append(run)
                    run_kind = tag
                run.append(line[1:])
                if tag == "+":
                    new_left -= 1
                else:
                    old_left -= 1
            elif tag in (" ", ""):  # editors sometimes strip the space of a blank context line
                hunk.context.append(line[1:])
                run_kind = ""
                old_left -= 1
                new_left -= 1
            elif tag != "\\":  # "\ No newline at end of file" is skipped; anything else ends the hunk
                hunk = None
                continue
        i += 1
    return files


def _diff_path(raw: str) -> str | None:
    """A path from a ---/+++ line without a/ or b/ prefix or time stamp; None for /dev/null."""
    raw = raw.split("\t", 1)[0].strip()
    if raw.startswith('"'):
        raw = _unquote(raw)
    else:
        raw = _TIMESTAMP_RE.sub("", raw)
    if raw == "/dev/null":
        return None
    return raw[2:] if raw.startswith(("a/", "b/")) else raw


_ESCAPES = {
    "n": b"\n",
    "t": b"\t",
    "r": b"\r",
    '"': b'"',
    "\\": b"\\",
    "a": b"\a",
    "b": b"\b",
    "f": b"\f",
    "v": b"\v",
}


def _unquote(raw: str) -> str:
    """Undo git's C-style path quoting, including octal escapes of UTF-8 bytes."""
    body = raw[1:-1] if len(raw) > 1 and raw.endswith('"') else raw[1:]
    out = bytearray()
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            octal = re.match(r"[0-7]{3}", body[i + 1 : i + 4])
            if octal:
                out.append(int(octal.group(), 8) & 0xFF)
                i += 4
                continue
            out += _ESCAPES.get(body[i + 1], body[i + 1].encode())
            i += 2
            continue
        out += c.encode()
        i += 1
    return out.decode("utf-8", "replace")


def is_trivial(line: str) -> bool:
    """Blank lines, lines under 4 characters, lone brackets and bare keywords such as `pass` or `else:`."""
    s = normalize_line(line)
    return len(s) < 4 or s in TRIVIAL_LINES or all(c in _BRACKETS for c in s)


def _trivial_block(block: list[str]) -> bool:
    return all(is_trivial(line) for line in block)


def _normalized(blocks: list[list[str]]) -> list[str]:
    return [s for block in blocks for line in block if (s := normalize_line(line))]


class _Matcher:
    """Finds blocks in the candidate files of one diff path: same path, then same basename, then anywhere."""

    def __init__(self, tree: Tree, diff_path: str, evidence_paths: list[str]) -> None:
        self.tree = tree
        target = diff_path.strip("/")
        base = PurePosixPath(target).name if target else ""
        suffix: list[str] = []
        same_name: list[str] = []
        rest: list[str] = []
        for p in evidence_paths:
            q = _DUP_SUFFIX_RE.sub("", p)
            if target and (q == target or q.endswith("/" + target)):
                suffix.append(p)
            elif base and PurePosixPath(q).name == base:
                same_name.append(p)
            else:
                rest.append(p)
        self.levels = [level for level in (suffix, same_name, rest) if level]
        self.file_exists = bool(suffix or same_name)
        self.found_in: set[str] = set()
        self._cache: dict[tuple[str, ...], tuple[str, int] | None] = {}

    def find(self, block: list[str]) -> tuple[str, int] | None:
        key = tuple(block)
        if key not in self._cache:
            snippet = "\n".join(block)
            self._cache[key] = next(
                (
                    hits[0]
                    for paths in self.levels
                    if (hits := self.tree.find_snippet(snippet, paths, partial_ends=False))
                ),
                None,
            )
        hit = self._cache[key]
        if hit is not None:
            self.found_in.add(hit[0])
        return hit


def _located(hunk: Hunk, fp: FilePatch, m: _Matcher, *, removal: bool) -> bool:
    """Whether the hunk's place is in the notebook, to tell "applied/reverted" from "drifted"."""
    context = [c for c in hunk.context if not is_trivial(c)]
    if context:
        return any(m.find([c]) for c in context)
    if (removal and fp.is_deleted) or (not removal and fp.is_new):
        return not m.file_exists
    return m.file_exists


def _evidence(tree: Tree, hit: tuple[str, int], block: list[str]) -> Evidence:
    """The block's first non-trivial line where the block was found."""
    path, line = hit
    target = next((normalize_line(x) for x in block if not is_trivial(x)), "")
    lines = tree.lines(path)
    for j in range(line - 1, len(lines)):
        if normalize_line(lines[j]) == target:
            return Evidence(path, j + 1, lines[j].strip()[:QUOTE_CHARS])
    return Evidence(path, line, lines[line - 1].strip()[:QUOTE_CHARS])


def hunk_state(hunk: Hunk, fp: FilePatch, m: _Matcher) -> tuple[str, list[Evidence]]:
    """(state, evidence) of one hunk; see the module docstring for the states."""
    added = [b for b in hunk.added if not _trivial_block(b)]
    removed = [b for b in hunk.removed if not _trivial_block(b)]
    if _normalized(added) == _normalized(removed):  # only trivial lines or whitespace changed
        return "trivial", []
    added_hits = [m.find(b) for b in added]
    first = added_hits[0] if added_hits else None
    if first is not None and all(added_hits):
        return "applied", [_evidence(m.tree, first, added[0])]
    if any(added_hits):
        return "unknown", []
    removed_hits = [m.find(b) for b in removed]
    if removed and all(removed_hits):
        return "reverted", [_evidence(m.tree, h, b) for h, b in zip(removed_hits, removed, strict=True) if h]
    if not added and not any(removed_hits) and _located(hunk, fp, m, removal=True):
        return "applied", []
    if not removed and _located(hunk, fp, m, removal=False):
        return "reverted", []
    return "unknown", []


def check(diff_text: str, tree: Tree) -> Finding:
    """The patch verdict for one idea: present, absent, partial or unknown, with per-hunk counts."""
    files = [f for f in parse(diff_text) if f.hunks]
    if not files:
        return Finding("unknown", "patch", note="no hunks found: the patch is empty or not a unified diff")
    evidence_paths = tree.evidence_paths()
    if not evidence_paths:
        return Finding("unknown", "patch", note="the notebook has no code to search")

    counts = {"applied": 0, "reverted": 0, "unknown": 0, "trivial": 0}
    per_file: dict[str, dict[str, Any]] = {}
    applied_ev: list[Evidence] = []
    reverted_ev: list[Evidence] = []
    for fp in files:
        m = _Matcher(tree, fp.path, evidence_paths)
        states = []
        for hunk in fp.hunks:
            state, ev = hunk_state(hunk, fp, m)
            counts[state] += 1
            states.append(state)
            (applied_ev if state == "applied" else reverted_ev).extend(ev)
        d = per_file.setdefault(
            fp.path, {"hunks": [], "found_in": [], "new_file": fp.is_new, "deleted": fp.is_deleted}
        )
        d["hunks"].extend(states)
        d["found_in"] = sorted(set(d["found_in"]) | m.found_in)

    a, r, u = counts["applied"], counts["reverted"], counts["unknown"]
    n = a + r + u
    if n == 0:
        status, note = "unknown", "the patch changes only blank or trivial lines"
    else:
        if a == n:
            status = "present"
        elif r == n:
            status = "absent"
        elif a == 0 and r == 0:
            status = "unknown"
        else:
            status = "partial"
        note = f"{a}/{n} hunks applied" + (f", {r} reverted" if r else "") + (f", {u} unknown" if u else "")
    evidence = applied_ev[:MAX_EVIDENCE]
    evidence += reverted_ev[: MAX_EVIDENCE - len(evidence)]
    detail = {"hunks": n + counts["trivial"], **counts, "files": per_file}
    return Finding(status, "patch", evidence=evidence, note=note, detail=detail)
