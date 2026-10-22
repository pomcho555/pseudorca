"""Patch matching: is a unified diff applied in a notebook, reverted, or neither?

Lines are compared as whole lines after whitespace normalisation, and blank lines are skipped, wherever the
notebook keeps the code: a diff made against a repository file still matches the same code inlined into a
cell, written with %%writefile or embedded as a payload. Candidate files are tried in order: paths ending
with the diff's path, then the same basename, then every evidence path.

Trivial lines (`pass`, `else:`, lone brackets, lines under 4 characters) are never anchors, and a change of
only trivial lines (or only whitespace) is ignored. Next to changed lines they are part of the change's
text, compared like any other line.

A change is judged at its place. The non-trivial context lines of a hunk split it into changes, and each
change is anchored by the nearest non-trivial context line before and after it:
    new side in place  the change's added lines next to one of its anchors (between both of them when
                       the old side is in place too); for a change that only removes lines, the two
                       anchors next to each other
    old side in place  the removed lines between both anchors; for a change that only adds lines, the
                       two anchors next to each other
A change at the start or end of a hunk has one anchor. When that anchor alone would have to stand for the
place, it must be the first or last line of its file.

Hunk states:
    applied   every change's new side is in place (and, for a change that only removes lines, its old
              side is not)
    reverted  every change's old side is in place, and its new side is not
    unknown   anything else: the code drifted, the file is not in the notebook, or added lines are found
              only away from their context
    trivial   only blank or trivial lines change, or only whitespace; ignored
A hunk without non-trivial context lines (a new or deleted file, a diff made with -U0, or context of only
trivial lines) has no place, so its blocks (runs of "+" or "-" lines) are matched alone, in the diff's own
file when the notebook has it (anywhere for a new or deleted file): applied when every added block is
found, reverted when every removed block is found and no added block is. When it only adds or only removes
lines, the other side is "almost none of these lines is found" (at most ALMOST_NONE of them): anywhere, and
without a file of that name, for a new or deleted file; in the file itself otherwise.

Every line of the notebook is normalised and indexed once per check, so finding a run of lines costs about
the number of places its rarest line occurs: matching takes time near-linear in the diff and the notebook.
"""

from __future__ import annotations

import operator
import re
import weakref
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, NamedTuple

from pseudorca.audit.types import Evidence, Finding
from pseudorca.tree import Tree, TreeFile, normalize_line, split_lines

MATCHER_VERSION = 2  # bump when matching changes, so that recorded patch verdicts are redone
MAX_EVIDENCE = 5
QUOTE_CHARS = 300
ALMOST_NONE = 0.2  # share of a context-free hunk's lines that may be found while it counts as not there
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
    lines: list[tuple[str, str]] = field(default_factory=list)  # every line in order: (" ", "+" or "-", text)


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
    lines = split_lines(diff_text.removeprefix("\ufeff"))  # a BOM would hide the first file header
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
                hunk.lines.append((tag, line[1:]))
                if tag == "+":
                    new_left -= 1
                else:
                    old_left -= 1
            elif tag in (" ", ""):  # editors sometimes strip the space of a blank context line
                hunk.context.append(line[1:])
                hunk.lines.append((" ", line[1:]))
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


def _blocks(lines: list[tuple[str, str]], tag: str) -> list[list[str]]:
    """Runs of consecutive lines with this tag ("+" or "-") that are not all trivial."""
    runs: list[list[str]] = []
    prev = ""
    for t, text in lines:
        if t == tag:
            if prev != tag:
                runs.append([])
            runs[-1].append(text)
        prev = t
    return [b for b in runs if not _trivial_block(b)]


class _Index:
    """The non-blank lines of every evidence path, normalised once and indexed: built lazily, and shared by
    the matchers of all diff files, and by every check of the same tree (see `_index_for`).

    `norm` holds the lines of all paths, one path after another; path r owns positions starts[r] to
    starts[r + 1]. `at` maps each normalised line to its positions, so finding a run of lines only looks at
    the places where its rarest line occurs.
    """

    def __init__(self, files: list[TreeFile]) -> None:
        self.files = files  # never the Tree itself: the cache holds indexes weakly by their tree
        self.paths = [f.path for f in files]
        self.bare = [_DUP_SUFFIX_RE.sub("", p) for p in self.paths]  # a file written twice: its first path
        self.names = [PurePosixPath(q).name for q in self.bare]
        self._built = False
        self.starts: list[int] = [0]
        self.norm: list[str] = []
        self.line: list[int] = []  # 1-based line in its file, per position
        self.at: dict[str, list[int]] = {}
        self._raw: dict[int, list[str]] = {}
        self._sets: dict[tuple[str, ...] | None, set[str]] = {}

    def build(self) -> _Index:
        if not self._built:
            self._built = True
            at, norm, line = self.at, self.norm, self.line
            for f in self.files:
                for i, raw in enumerate(split_lines(f.text), 1):
                    if s := normalize_line(raw):
                        at.setdefault(s, []).append(len(norm))
                        norm.append(s)
                        line.append(i)
                self.starts.append(len(norm))
        return self

    def owner(self, pos: int) -> int:
        """The rank (index in `paths`) of the path that holds position `pos`."""
        return bisect_right(self.starts, pos) - 1

    def raw(self, rank: int) -> list[str]:
        """A path's lines as written, for quotes (split once, and only for paths that give evidence)."""
        if rank not in self._raw:
            self._raw[rank] = split_lines(self.files[rank].text)
        return self._raw[rank]

    def line_set(self, ranks: tuple[int, ...] | None) -> set[str]:
        """The normalised lines of these paths (None: of every path)."""
        key = None if ranks is None else tuple(self.paths[r] for r in ranks)
        if key not in self._sets:
            self.build()
            if ranks is None:
                self._sets[key] = set(self.at)
            else:
                self._sets[key] = {s for r in ranks for s in self.norm[self.starts[r] : self.starts[r + 1]]}
        return self._sets[key]


_INDEXES: weakref.WeakKeyDictionary[Tree, _Index] = weakref.WeakKeyDictionary()


def _index_for(tree: Tree, paths: list[str]) -> _Index:
    """The index of these paths of `tree`, reused while the tree holds the same files (an audit checks
    every idea's patch against one tree)."""
    files = [tree.files[p] for p in paths]
    ix = _INDEXES.get(tree)
    if ix is None or len(ix.files) != len(files) or any(map(operator.is_not, ix.files, files)):
        ix = _INDEXES[tree] = _Index(files)
    return ix


class _Hit(NamedTuple):
    path: str
    line: int  # 1-based line of the first line found
    pos: int  # its position in the index
    size: int  # how many (non-blank) lines were found from there


SAME_PATH, SAME_NAME, ANYWHERE = 0, 1, 2  # candidate levels, tried in this order


class _Matcher:
    """Finds lines in the candidate files of one diff path: same path, then same basename, then anywhere."""

    def __init__(self, index: _Index, diff_path: str) -> None:
        self.ix = index
        target = diff_path.strip("/")
        base = PurePosixPath(target).name if target else ""
        self.level: list[int] = []  # per rank in index.paths
        for q, name in zip(index.bare, index.names, strict=True):
            if target and (q == target or q.endswith("/" + target)):
                self.level.append(SAME_PATH)
            elif base and name == base:
                self.level.append(SAME_NAME)
            else:
                self.level.append(ANYWHERE)
        # the diff's own file, when the notebook has it: same path first, then same basename
        self.located = tuple(r for lv in (SAME_PATH, SAME_NAME) for r, x in enumerate(self.level) if x == lv)
        self.file_exists = bool(self.located)
        self.found_in: set[str] = set()
        self._cache: dict[tuple[tuple[str, ...], bool, bool, bool], _Hit | None] = {}

    def find(
        self, lines: list[str], *, at_start: bool = False, at_end: bool = False, local: bool = False
    ) -> _Hit | None:
        """Where `lines` occur as consecutive whole lines (blank lines ignored), in the first candidate level
        that has them, and there the first place; with at_start/at_end, only where nothing but blank lines
        comes before/after them. `local`: only in the diff's own file."""
        key = (tuple(lines), at_start, at_end, local)
        if key not in self._cache:
            want = [s for x in lines if (s := normalize_line(x))]
            self._cache[key] = self._search(want, at_start, at_end, SAME_NAME if local else ANYWHERE)
        hit = self._cache[key]
        if hit is not None:
            self.found_in.add(hit.path)
        return hit

    def _search(self, want: list[str], at_start: bool, at_end: bool, last_level: int) -> _Hit | None:
        if not want:
            return None
        ix = self.ix.build()
        n, norm, at = len(want), ix.norm, ix.at
        # look only where the rarest of the lines occurs (a line found nowhere settles it at once)
        j = min(range(n), key=lambda k: len(at.get(want[k], ())))
        best: tuple[int, int] | None = None  # (level, position): positions run in path order, then line
        for q in at.get(want[j], ()):
            rank = ix.owner(q)
            level = self.level[rank]
            if level > last_level or (best is not None and level >= best[0]):
                continue
            pos, start, stop = q - j, ix.starts[rank], ix.starts[rank + 1]
            if pos < start or pos + n > stop or (at_start and pos != start) or (at_end and pos + n != stop):
                continue
            if n > 1 and norm[pos : pos + n] != want:
                continue
            best = (level, pos)
            if level == SAME_PATH:
                break
        if best is None:
            return None
        pos = best[1]
        return _Hit(ix.paths[ix.owner(pos)], ix.line[pos], pos, n)

    def share(self, blocks: list[list[str]], *, everywhere: bool) -> tuple[int, int]:
        """(found, total) for the distinct non-trivial lines of `blocks`, as whole lines of the diff's file
        (or of any evidence path)."""
        wanted = {normalize_line(x) for b in blocks for x in b if not is_trivial(x)}
        found = self.ix.line_set(None if everywhere else self.located)
        return len(wanted & found), len(wanted)

    def evidence(self, hit: _Hit, block: list[str]) -> Evidence:
        """The block's first non-trivial line, among the lines found at `hit`."""
        ix = self.ix
        target = next((normalize_line(x) for x in block if not is_trivial(x)), "")
        line = next(
            (ix.line[p] for p in range(hit.pos, hit.pos + hit.size) if ix.norm[p] == target), hit.line
        )
        return Evidence(hit.path, line, ix.raw(ix.owner(hit.pos))[line - 1].strip()[:QUOTE_CHARS])


@dataclass
class _Change:
    """The lines of a hunk between two non-trivial context lines (its anchors)."""

    before: str | None  # None when the change starts the hunk
    after: str | None  # None when the change ends the hunk
    lines: list[tuple[str, str]]  # the changed lines and the trivial context lines around them

    def side(self, drop: str) -> list[str]:
        """The new side (drop="-") or the old side (drop="+") of the change, without its anchors."""
        return [text for tag, text in self.lines if tag != drop]


def _changes(hunk: Hunk) -> list[_Change]:
    out: list[_Change] = []
    anchor: str | None = None
    cur: list[tuple[str, str]] = []
    for tag, text in hunk.lines:
        if tag == " " and not is_trivial(text):
            if any(t != " " for t, _ in cur):
                out.append(_Change(anchor, text, cur))
            anchor, cur = text, []
        else:
            cur.append((tag, text))
    if any(t != " " for t, _ in cur):
        out.append(_Change(anchor, None, cur))
    return out


def _in_place(m: _Matcher, c: _Change, side: list[str], *, own: bool, both: bool) -> _Hit | None:
    """Where one side of a change is in place.

    `own`: the side has non-trivial lines of its own, so `both` decides whether one anchor next to them is
    enough or both are needed (a change with one anchor has only that one). Otherwise the side is only a
    place: both anchors next to each other, or a lone anchor at the start or end of a file.
    """
    if own and not both:
        tries = ([[c.before, *side]] if c.before is not None else []) + (
            [[*side, c.after]] if c.after is not None else []
        )
        return next((hit for lines in tries if (hit := m.find(lines))), None)
    lines = [x for x in (c.before, *side, c.after) if x is not None]
    if own or sum(1 for x in lines if x.strip()) > 1:
        return m.find(lines)
    return m.find(lines, at_start=c.before is None, at_end=c.after is None)


def _change_state(c: _Change, m: _Matcher) -> tuple[str, list[Evidence]]:
    added, removed = _blocks(c.lines, "+"), _blocks(c.lines, "-")
    if _normalized(added) == _normalized(removed):  # only trivial lines or whitespace changed
        return "trivial", []
    new_hit = _in_place(m, c, c.side("-"), own=bool(added), both=False)
    old_hit = _in_place(m, c, c.side("+"), own=bool(removed), both=True)
    if added:
        if new_hit and old_hit:
            # the old side is in place too, so added lines next to one anchor may be another place's (a
            # similar function, a helper, a copy of an anchor): they must sit between both anchors
            new_hit = _in_place(m, c, c.side("-"), own=True, both=True)
            if new_hit is None:
                return "unknown", []
        if new_hit:
            return "applied", [m.evidence(new_hit, added[0])]
        if old_hit:
            return "reverted", [m.evidence(old_hit, b) for b in removed]
        return "unknown", []
    # the change only removes lines: its new side is just the place, so both found is unclear
    if old_hit and not new_hit:
        return "reverted", [m.evidence(old_hit, b) for b in removed]
    if new_hit and not old_hit:
        return "applied", []
    return "unknown", []


def _free_state(hunk: Hunk, fp: FilePatch, m: _Matcher) -> tuple[str, list[Evidence], tuple[int, int] | None]:
    """A hunk without non-trivial context: (state, evidence, (found, total) lines when they decided)."""
    added, removed = _blocks(hunk.lines, "+"), _blocks(hunk.lines, "-")
    if _normalized(added) == _normalized(removed):
        return "trivial", [], None
    # the blocks have no place of their own: look for them in the diff's file when the notebook has it, so
    # that unrelated cells cannot decide (a new or deleted file is judged by its lines anywhere)
    local = m.file_exists and not (fp.is_new or fp.is_deleted)
    added_hits = [m.find(b, local=local) for b in added]
    if added and all(added_hits):
        hit = added_hits[0]
        assert hit is not None
        return "applied", [m.evidence(hit, added[0])], None
    removed_hits = [m.find(b, local=local) for b in removed]
    if removed and all(removed_hits) and not any(added_hits):
        return "reverted", [m.evidence(h, b) for h, b in zip(removed_hits, removed, strict=True) if h], None
    if added and removed:
        return "unknown", [], None
    # only adds or only removes lines: the other side is "almost none of them is found"
    whole_file = fp.is_new if added else fp.is_deleted
    if whole_file and m.file_exists:  # a file of that name with other content: the code drifted
        return "unknown", [], None
    if not whole_file and not m.file_exists:  # lines of a file that is not in the notebook
        return "unknown", [], None
    found, total = m.share(added or removed, everywhere=whole_file)
    if found > ALMOST_NONE * total:
        return "unknown", [], (found, total)
    return ("reverted" if added else "applied"), [], (found, total)


def hunk_state(hunk: Hunk, fp: FilePatch, m: _Matcher) -> tuple[str, list[Evidence]]:
    """(state, evidence) of one hunk; see the module docstring for the states."""
    state, evidence, _ = _hunk_state(hunk, fp, m)
    return state, evidence


def _hunk_state(hunk: Hunk, fp: FilePatch, m: _Matcher) -> tuple[str, list[Evidence], tuple[int, int] | None]:
    if not any(tag == " " and not is_trivial(text) for tag, text in hunk.lines):
        return _free_state(hunk, fp, m)
    states = [s for s in (_change_state(c, m) for c in _changes(hunk)) if s[0] != "trivial"]
    kinds = {state for state, _ in states}
    if not kinds:
        return "trivial", [], None
    if kinds == {"applied"}:
        return "applied", next((ev[:1] for _, ev in states if ev), []), None
    if kinds == {"reverted"}:
        return "reverted", [e for _, ev in states for e in ev], None
    return "unknown", [], None


def check(diff_text: str, tree: Tree) -> Finding:
    """The patch verdict for one idea: present, absent, partial or unknown, with per-hunk counts."""
    files = [f for f in parse(diff_text) if f.hunks]
    if not files:
        return Finding("unknown", "patch", note="no hunks found: the patch is empty or not a unified diff")
    evidence_paths = tree.evidence_paths()
    if not evidence_paths:
        return Finding("unknown", "patch", note="the notebook has no code to search")

    index = _index_for(tree, evidence_paths)
    counts = {"applied": 0, "reverted": 0, "unknown": 0, "trivial": 0}
    per_file: dict[str, dict[str, Any]] = {}
    applied_ev: list[Evidence] = []
    reverted_ev: list[Evidence] = []
    for fp in files:
        m = _Matcher(index, fp.path)
        states = []
        shares: list[tuple[int, int]] = []
        for hunk in fp.hunks:
            state, ev, share = _hunk_state(hunk, fp, m)
            counts[state] += 1
            states.append(state)
            if share is not None:
                shares.append(share)
            (applied_ev if state == "applied" else reverted_ev).extend(ev)
        d = per_file.setdefault(
            fp.path, {"hunks": [], "found_in": [], "new_file": fp.is_new, "deleted": fp.is_deleted}
        )
        d["hunks"].extend(states)
        d["found_in"] = sorted(set(d["found_in"]) | m.found_in)
        if shares:  # lines of a context-free hunk found, for a new or deleted file judged by them
            found, total = d.get("lines_found", [0, 0])
            d["lines_found"] = [found + sum(s[0] for s in shares), total + sum(s[1] for s in shares)]

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
