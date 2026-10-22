"""Fingerprints: snippets that must (or must not) occur in a notebook's code.

    text         the substring must occur in a line
    re:<regex>   the regex must match a line
    !text        the substring must not occur
    !re:<regex>  the regex must not match any line

A leading "!" always negates, so a literal "!" at the start needs a regex (`re:!pip`). Only evidence paths
are searched: markdown is a claim, not evidence, and binary files have no text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pseudorca.audit.types import Evidence, Finding
from pseudorca.tree import Hit, Tree

MAX_EVIDENCE = 5
QUOTE_CHARS = 300


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


def check(fingerprints: list[str], tree: Tree) -> Finding | None:
    """The fingerprint verdict for one idea, or None when it has no fingerprints."""
    specs = [parse(f) for f in dict.fromkeys(fingerprints) if f.strip()]
    if not specs:
        return None
    invalid = []
    for s in specs:
        try:
            if not s.pattern:
                raise ValueError("empty pattern")
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

    hits = {s.text: tree.grep(s.pattern, regex=s.regex, paths=paths, max_hits=MAX_EVIDENCE) for s in specs}
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


def _evidence(specs: list[Spec], hits: dict[str, list[Hit]]) -> list[Evidence]:
    """Up to MAX_EVIDENCE hits, taking each fingerprint's first hit before anyone's second."""
    out: list[Evidence] = []
    seen: set[tuple[str, int]] = set()
    for rank in range(MAX_EVIDENCE):
        for s in specs:
            if rank < len(hits[s.text]) and len(out) < MAX_EVIDENCE:
                h = hits[s.text][rank]
                if (h.path, h.line) not in seen:
                    seen.add((h.path, h.line))
                    out.append(Evidence(h.path, h.line, _quote(s, h.text)))
    return out


def _quote(spec: Spec, line: str) -> str:
    """The line, cut to QUOTE_CHARS around the match when it is longer."""
    text = line.strip()
    if len(text) <= QUOTE_CHARS:
        return text
    m = _compile(spec).search(text)
    start = min(max(0, (m.start() if m else 0) - QUOTE_CHARS // 3), len(text) - QUOTE_CHARS)
    return text[start : start + QUOTE_CHARS].strip()
