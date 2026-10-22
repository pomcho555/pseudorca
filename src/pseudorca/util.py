"""Small shared helpers: ids, hashes, time stamps and plain-text tables."""

from __future__ import annotations

import hashlib
import os
import re
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_RAND_MASK = (1 << 80) - 1
_last_ulid = [0, 0]


def new_ulid() -> str:
    """A 26-character ULID: 48 bits of milliseconds then 80 random bits, Crockford base32.

    Ids sort by creation time, and they stay strictly increasing within one process, so events written
    in the same millisecond keep their order.
    """
    ms = time.time_ns() // 1_000_000
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


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
IDEA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
HEX_RE = re.compile(r"^[0-9a-f]+$")


def slug(text: str, max_len: int = 40) -> str:
    """A file-name-safe version of free text (used for the actor part of event file names)."""
    s = _SLUG_RE.sub("-", text.strip()).strip("-.")
    return s[:max_len] or "unknown"


def short(sha: str | None, n: int = 10) -> str:
    return (sha or "")[:n]


def _char_width(c: str) -> int:
    return 2 if unicodedata.east_asian_width(c) in "WF" else 1


def display_width(text: str) -> int:
    """Terminal columns a string takes (wide East Asian characters count twice)."""
    return sum(_char_width(c) for c in text)


def clip(text: str, width: int) -> str:
    """Cut text to a display width, marking the cut with an ellipsis."""
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


def format_table(headers: list[str], rows: list[list[object]], max_width: int = 60) -> str:
    """A plain-text table. Cells are flattened to one line and clipped to `max_width` columns."""
    cells = [[clip(" ".join(("" if v is None else str(v)).split()), max_width) for v in row] for row in rows]
    heads = [str(h) for h in headers]
    widths = [display_width(h) for h in heads]
    for row in cells:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], display_width(v))

    def line(vals: list[str]) -> str:
        return "  ".join(v + " " * (widths[i] - display_width(v)) for i, v in enumerate(vals)).rstrip()

    out = [line(heads), line(["-" * w for w in widths])]
    out.extend(line(r) for r in cells)
    return "\n".join(out)
