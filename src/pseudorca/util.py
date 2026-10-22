"""Small shared helpers: ids, hashes, time stamps, plain-text tables and numbers for display."""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import math
import os
import re
import time
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_RAND_MASK = (1 << 80) - 1
_last_ulid = [0, 0]


def new_ulid(min_ms: int = 0) -> str:
    """A 26-character ULID: 48 bits of milliseconds then 80 random bits, Crockford base32.

    Ids sort by creation time, and they stay strictly increasing within one process, so events written
    in the same millisecond keep their order. `min_ms` raises the time part (a hybrid logical clock: the
    store passes one more than the newest id it has seen, so a new event always sorts after them, even
    when this machine's clock is behind).
    """
    now = time.time_ns() // 1_000_000
    if min_ms > now:
        # a store whose newest event is ahead of this clock: order after it, but keep that future time out of
        # the process-wide counter, so other stores this process writes to get ordinary ids
        ms, rnd = min_ms, int.from_bytes(os.urandom(10), "big")
    else:
        ms = now
        if ms <= _last_ulid[0]:
            ms = _last_ulid[0]
            rnd = (_last_ulid[1] + 1) & _RAND_MASK
        else:
            rnd = int.from_bytes(os.urandom(10), "big")
        _last_ulid[0], _last_ulid[1] = ms, rnd
    value = (ms << 80) | rnd
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def ulid_ms(ulid: str) -> int:
    """The millisecond time part of a ULID (0 if it is not one)."""
    value = 0
    for ch in ulid[:10].upper():
        idx = _CROCKFORD.find(ch)
        if idx < 0:
            return 0
        value = value * 32 + idx
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    """Local time with its UTC offset, to the second."""
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def local_time(iso: str) -> str:
    """An event time ("2026-01-05T18:30:00+09:00") as "2026-01-05 10:30" in the viewer's time zone.

    Writers record their own offset; converting keeps a team spread over time zones in order.
    """
    if not isinstance(iso, str) or len(iso) <= 10:  # a bare date stays a date
        return str(iso or "")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso[:16].replace("T", " ")
    if dt.tzinfo is not None:
        with contextlib.suppress(OverflowError, OSError, ValueError):  # a malformed time near year 1 or 9999
            dt = dt.astimezone()
    return minute_text(dt)


def minute_text(dt: datetime) -> str:
    """A time as "2026-01-05 10:30". Not strftime: glibc's %Y does not pad years below 1000 to four digits."""
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}"


def local_day(iso: str) -> str:
    return local_time(iso)[:10]


_EARLIEST = datetime.min.replace(tzinfo=UTC)


def instant(iso: object) -> datetime:
    """A ledger time as an aware datetime, for ordering. No offset means the viewer's local time; anything
    unparseable sorts first. (Times written in different zones must never be compared as strings.)"""
    if not isinstance(iso, str) or not iso:
        return _EARLIEST
    try:
        dt = datetime.fromisoformat(iso)
        return dt if dt.tzinfo is not None else dt.astimezone()
    except (OverflowError, OSError, ValueError):
        return _EARLIEST


INSTALL_SOURCE = "git+https://github.com/pomcho555/pseudorca"


def install_hint(extra: str) -> str:
    """How to add an optional extra, for both pip and uv tool installs."""
    spec = f'"pseudorca[{extra}] @ {INSTALL_SOURCE}"'
    return f"add the {extra} extra: pip install {spec}, or uv tool install --reinstall {spec}"


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
IDEA_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
HEX_RE = re.compile(r"\A[0-9a-f]+\Z")
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.match(value) is not None


def slug(text: str, max_len: int = 40) -> str:
    """A file-name-safe version of free text (used for the actor part of event file names)."""
    s = _SLUG_RE.sub("-", text.strip()).strip("-.")
    return s[:max_len] or "unknown"


def short(sha: str | None, n: int = 10) -> str:
    return (sha or "")[:n]


# control characters (all but tab) and bidirectional overrides: printed as they are, they could move the
# cursor, recolour, retitle or rewrite the terminal, or make text read in another order than it is
_UNSAFE_RE = re.compile("[\x00-\x08\x0a-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def clean(text: str) -> str:
    """`text` safe to print on one terminal line: each unsafe character becomes "?". Ledger strings come
    through git from teammates, or from hand-edited events."""
    return _UNSAFE_RE.sub("?", text)


def _char_width(c: str) -> int:
    return 2 if unicodedata.east_asian_width(c) in "WF" else 1


def display_width(text: str) -> int:
    """Terminal columns a string takes (wide East Asian characters count twice)."""
    return sum(_char_width(c) for c in text)


def clip(text: str, width: int) -> str:
    """Cut text to a display width, marking the cut with an ellipsis (unsafe characters become "?")."""
    text = clean(text)
    if display_width(text) <= width:
        return text
    out, used = [], 0
    for c in text:
        w = _char_width(c)
        if used + w > width - 1:
            break
        out.append(c)
        used += w
    return "".join(out) + "…"


def excerpt(text: str, col: int | None, width: int, match_len: int = 0) -> str:
    """One line of `text` clipped to `width`, keeping the part starting at `col` (the match) visible."""
    line = " ".join(text.split()) if col is None else text
    if col is None or display_width(line) <= width:
        return clip(" ".join(line.split()), width)
    # start about a quarter of the width before the match, counted in display columns (wide characters
    # count twice), then move back while the rest of the line still fits
    start, used = max(0, min(col, len(line))), 0
    while start > 0 and used + _char_width(line[start - 1]) <= width // 4:
        start -= 1
        used += _char_width(line[start])
    tail = display_width(line[start:])
    while start > 0 and tail + _char_width(line[start - 1]) <= width - 1:
        start -= 1
        tail += _char_width(line[start])
    window = line[start:]
    out = clip(" ".join(window.split()), width - (1 if start else 0))
    return ("…" + out) if start else out


def format_table(
    headers: list[str],
    rows: list[list[object]],
    max_width: int = 60,
    col_widths: dict[int, int] | None = None,
) -> str:
    """A plain-text table. Cells are flattened to one line and clipped to `max_width` display columns,
    or to `col_widths[i]` for column i."""
    limits = col_widths or {}
    cells = [
        [
            clip(" ".join(("" if v is None else str(v)).split()), limits.get(i, max_width))  # clip cleans
            for i, v in enumerate(row)
        ]
        for row in rows
    ]
    heads = [clean(str(h)) for h in headers]
    widths = [display_width(h) for h in heads]
    for row in cells:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], display_width(v))

    def line(vals: list[str]) -> str:
        return "  ".join(v + " " * (widths[i] - display_width(v)) for i, v in enumerate(vals)).rstrip()

    out = [line(heads), line(["-" * w for w in widths])]
    out.extend(line(r) for r in cells)
    return "\n".join(out)


# ----- numbers for display -------------------------------------------------------------------------

MINUS = "−"
COMPACT = 1e6  # from this magnitude on, a number is shown with 4 to 6 significant digits, not decimals
MAX_DECIMALS = 10


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def number(x: float | None, dec: int | None = None) -> str:
    """A score: `dec` decimals, or 6 significant digits (always for 1e6 and more); empty when missing."""
    if not _finite(x):
        return ""
    assert x is not None
    return f"{x:.6g}" if dec is None or abs(x) >= COMPACT else f"{x:.{dec}f}"


def _magnitude(x: float, dec: int) -> str:
    return f"{x:.4g}" if x >= COMPACT else f"{x:.{dec}f}"


def plain_number(x: float | None, dec: int) -> str:
    """A value with a Unicode minus ("n/a" when missing)."""
    if not _finite(x):
        return "n/a"
    assert x is not None
    text = _magnitude(abs(x), dec)
    return MINUS + text if x < 0 and float(text) else text


def signed_number(x: float | None, dec: int) -> str:
    """A difference: always signed, except a zero ("n/a" when missing)."""
    if not _finite(x):
        return "n/a"
    assert x is not None
    text = _magnitude(abs(x), dec)
    if not float(text):
        return text
    return ("+" if x > 0 else MINUS) + text


def interval_decimals(items: Iterable[Any], fallback: Iterable[float | None] = ()) -> int:
    """Decimals that show intervals (of anything with ci_low and ci_high: terms, steps) to about two
    significant digits; more when a bound that excludes 0 would otherwise print as 0."""
    items = list(items)
    widths = [
        t.ci_high - t.ci_low
        for t in items
        if _finite(getattr(t, "ci_low", None)) and _finite(getattr(t, "ci_high", None))
    ]
    bases = [w for w in widths if w > 0 and math.isfinite(w)]  # two huge finite bounds can be inf apart
    if not bases:
        values = [abs(getattr(t, "estimate", getattr(t, "delta", 0.0)) or 0.0) for t in items]
        values += [abs(v) for v in fallback if _finite(v)]  # type: ignore[arg-type]
        bases = [v for v in values if math.isfinite(v) and v > 0]
    dec = min(MAX_DECIMALS, max(0, 1 - math.floor(math.log10(min(bases))))) if bases else 3
    for t in items:
        lo, hi = getattr(t, "ci_low", None), getattr(t, "ci_high", None)
        for bound, excludes in ((lo, _finite(lo) and lo > 0), (hi, _finite(hi) and hi < 0)):
            while excludes and dec < MAX_DECIMALS and not float(f"{abs(bound):.{dec}f}"):
                dec += 1
    return dec


def score_columns(
    scores: Iterable[float | None], fitted: Iterable[float | None], digits: int = 6
) -> tuple[int, int]:
    """Decimals for scores (to `digits` significant digits: 6 for scores as recorded, fewer for means
    computed from a table) and for fitted values: up to 2 more than the scores' (a fitted value is a mean,
    but hardly more precise than the scores it comes from)."""
    dec = score_decimals(scores, digits=digits)
    return dec, min(dec + 2, max(dec, score_decimals(fitted, digits=digits)))


def running_column(start: float, deltas: list[float], end: float, dec: int) -> tuple[str, list[str], str]:
    """A start, the steps added to it and the end, at `dec` decimals. Each step is shown as the difference
    of the rounded running totals, so the column adds up as printed (a step is then at most one unit in
    the last place from its own rounding). Compact numbers (1e6 and more) are rounded one by one."""
    totals = [start]
    for d in deltas[:-1]:
        totals.append(totals[-1] + d)
    totals.append(end)
    if not all(_finite(v) and abs(v) < COMPACT for v in totals):
        return plain_number(start, dec), [signed_number(d, dec) for d in deltas], plain_number(end, dec)
    shown = [round(v, dec) for v in totals]
    steps = [signed_number(b - a, dec) for a, b in itertools.pairwise(shown)]
    return plain_number(shown[0], dec), steps[: len(deltas)], plain_number(shown[-1], dec)


def score_decimals(values: Iterable[float | None], cap: int = 8, digits: int = 6) -> int:
    """Decimals that show every value to `digits` significant digits, without trailing zeros (at most
    `cap`)."""
    dec = 0
    for v in values:
        if not _finite(v) or v == 0 or abs(v) >= COMPACT:  # type: ignore[arg-type]
            continue
        assert v is not None
        places = min(cap, max(0, digits - 1 - math.floor(math.log10(abs(v)))))
        text = f"{abs(v):.{places}f}".rstrip("0")
        if "." in text:
            dec = max(dec, len(text.split(".")[1]))
    return dec
