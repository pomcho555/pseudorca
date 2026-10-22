"""Fingerprints: snippets that must (or must not) occur in a notebook's code.

    text         the substring must occur in a line; text spanning lines must occur in consecutive lines
                 (whitespace differences and blank lines are ignored, the first line may be the end of a
                 line and the last one the start of a line)
    re:<regex>   the regex must match a line; a regex containing a newline (typed, or written as \\n or
                 \\r) or the (?m) flag is searched over each whole file, with re.MULTILINE
    !text        the substring must not occur
    !re:<regex>  the regex must not match

A leading "!" always negates, so a literal "!" at the start needs a regex (`re:!pip`). Only evidence paths
are searched: markdown is a claim, not evidence, and binary files have no text. Each piece of evidence
quotes the lines its match touches, without their outer whitespace (cut around the match when long). Its
line is the quote's first line, and its column is where the match starts in the quote, so displays can
show that part of a long line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pseudorca.audit.types import Evidence, Finding
from pseudorca.tree import Tree, normalize_line, split_lines

MATCHER_VERSION = 2  # bump when matching changes, so that recorded fingerprint verdicts are redone
MAX_EVIDENCE = 5
QUOTE_CHARS = 300
_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_NEWLINE_ESCAPE = re.compile(r"(?<!\\)(?:\\\\)*\\[nr]")  # \n or \r in a regex, not an escaped backslash


@dataclass(frozen=True)
class Spec:
    text: str  # the fingerprint as written
    negative: bool
    regex: bool
    pattern: str


def parse(fingerprint: str) -> Spec:
    negative = fingerprint.startswith("!")
    body = fingerprint[1:] if negative else fingerprint
    regex = body.startswith("re:")
    return Spec(fingerprint, negative, regex, body[3:] if regex else body)


def _compile(spec: Spec) -> re.Pattern[str]:
    return re.compile(spec.pattern if spec.regex else re.escape(spec.pattern))


def spans_lines(spec: Spec) -> bool:
    """Whether the fingerprint is matched across lines rather than line by line (see the module doc)."""
    if "\n" in spec.pattern or "\r" in spec.pattern:
        return True
    return spec.regex and (
        _NEWLINE_ESCAPE.search(spec.pattern) is not None or bool(_compile(spec).flags & re.MULTILINE)
    )


def check(fingerprints: list[str], tree: Tree) -> Finding | None:
    """The fingerprint verdict for one idea, or None when it has no fingerprints."""
    specs = [parse(f) for f in dict.fromkeys(fingerprints) if f.strip()]
    if not specs:
        return None
    invalid = []
    for s in specs:
        try:
            if not s.pattern.strip():
                raise ValueError("nothing to look for")
            _compile(s)
        except (re.error, ValueError) as e:
            invalid.append(f"{s.text} ({e})")
    if invalid:
        return Finding(
            "unknown",
            "fingerprint",
            note="invalid fingerprint: " + "; ".join(invalid),
            detail={"invalid": invalid},
        )
    paths = tree.evidence_paths()
    if not paths:
        return Finding("unknown", "fingerprint", note="the notebook has no code to search")

    hits = {s.text: _find(s, tree, paths) for s in specs}
    positives = [s for s in specs if not s.negative]
    negatives = [s for s in specs if s.negative]
    found = [s.text for s in positives if hits[s.text]]
    missing = [s.text for s in positives if not hits[s.text]]
    forbidden = [s.text for s in negatives if hits[s.text]]
    if positives:
        status = "absent" if not found else "present" if not missing and not forbidden else "partial"
    else:
        status = "present" if not forbidden else "absent" if len(forbidden) == len(negatives) else "partial"

    notes = []
    if positives:
        notes.append(
            f"found {len(found)}/{len(positives)}" + (f" (missing: {', '.join(missing)})" if missing else "")
        )
    if negatives:
        notes.append(
            f"{len(forbidden)}/{len(negatives)} forbidden found"
            + (f" ({', '.join(forbidden)})" if forbidden else "")
        )
    shown = [s for s in specs if hits[s.text]]
    return Finding(
        status,
        "fingerprint",
        evidence=_evidence(shown, hits),
        note="; ".join(notes),
        detail={"found": found, "missing": missing, "forbidden_found": forbidden},
    )


def _find(spec: Spec, tree: Tree, paths: list[str]) -> list[Evidence]:
    """Up to MAX_EVIDENCE places where one fingerprint's pattern occurs."""
    if not spans_lines(spec):
        rx = _compile(spec)
        out = []
        for h in tree.grep(spec.pattern, regex=spec.regex, paths=paths, max_hits=MAX_EVIDENCE):
            m = rx.search(h.text)
            quote, col, _ = _quote(h.text, m.start() if m else 0)
            out.append(Evidence(h.path, h.line, quote, col))
        return out
    if spec.regex:
        return _regex_hits(re.compile(spec.pattern, re.MULTILINE), tree, paths)
    return _snippet_hits(spec.pattern, tree, paths)


def _regex_hits(rx: re.Pattern[str], tree: Tree, paths: list[str]) -> list[Evidence]:
    """Matches of a regex over each whole file; the quote is the lines the match touches, and the line the
    quote's first one (a match may start on a blank line, or with the line break before its text)."""
    out: list[Evidence] = []
    for p in paths:
        text = _LINE_BREAK.sub("\n", tree.text(p))  # line numbers as tree.split_lines counts them
        for m in rx.finditer(text):
            begin = text.rfind("\n", 0, m.start()) + 1
            end = m.end() - 1 if m.end() > m.start() and text[m.end() - 1] == "\n" else m.end()
            stop = text.find("\n", max(end, m.start()))
            quote, col, offset = _quote(text[begin : len(text) if stop < 0 else stop], m.start() - begin)
            first = begin + offset if quote else m.start()  # where the quote starts in the file
            out.append(Evidence(p, text.count("\n", 0, first) + 1, quote, col))
            if len(out) >= MAX_EVIDENCE:
                return out
    return out


def _snippet_hits(pattern: str, tree: Tree, paths: list[str]) -> list[Evidence]:
    """Places where text spanning lines occurs (Tree.find_snippet with partial ends)."""
    want = [normalize_line(x) for x in split_lines(pattern) if x.strip()]
    out: list[Evidence] = []
    for path, line in tree.find_snippet(pattern, paths, partial_ends=True)[:MAX_EVIDENCE]:
        lines = tree.lines(path)
        j, seen = line - 1, 0
        while j < len(lines) and seen < len(want):  # the file lines the snippet covers
            seen += bool(lines[j].strip())
            j += 1
        first = normalize_line(lines[line - 1])
        start = first.find(want[0]) if len(want) == 1 else len(first) - len(want[0])
        text = "\n".join(lines[line - 1 : j])
        quote, col, offset = _quote(text, _raw_index(lines[line - 1], start))
        out.append(Evidence(path, line + text.count("\n", 0, offset), quote, col))
    return out


def _raw_index(line: str, k: int) -> int:
    """Where character k of normalize_line(line) is in `line`."""
    i, n = len(line) - len(line.lstrip()), 0
    while i < len(line) and n < k:
        if line[i].isspace():
            while i < len(line) and line[i].isspace():
                i += 1
        else:
            i += 1
        n += 1
    return i


def _evidence(specs: list[Spec], hits: dict[str, list[Evidence]]) -> list[Evidence]:
    """Up to MAX_EVIDENCE hits, taking each fingerprint's first hit before anyone's second."""
    out: list[Evidence] = []
    seen: set[tuple[str, int | None]] = set()
    for rank in range(MAX_EVIDENCE):
        for s in specs:
            if rank < len(hits[s.text]) and len(out) < MAX_EVIDENCE:
                e = hits[s.text][rank]
                if (e.path, e.line) not in seen:
                    seen.add((e.path, e.line))
                    out.append(e)
    return out


def _quote(text: str, start: int) -> tuple[str, int, int]:
    """(quote, col, offset): `text` without its outer whitespace, cut to QUOTE_CHARS around position `start`
    when longer; where `start` is in that quote; and where the quote starts in `text`."""
    lead = len(text) - len(text.lstrip())
    body = text.strip()
    col = min(max(0, start - lead), len(body))
    if len(body) <= QUOTE_CHARS:
        return body, col, lead
    begin = min(max(0, col - QUOTE_CHARS // 3), len(body) - QUOTE_CHARS)
    window = body[begin : begin + QUOTE_CHARS]
    quote = window.strip()
    skip = len(window) - len(window.lstrip())
    return quote, min(max(0, col - begin - skip), len(quote)), lead + begin + skip
