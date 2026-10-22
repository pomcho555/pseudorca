"""Embedded payloads: text-encoded, compressed or archived data inside string and bytes constants.

Decoding is bounded by `Limits` and never runs anything: a pickle is recorded and never loaded, and archive
member names only ever become virtual tree paths.
"""

from __future__ import annotations

import ast
import base64
import binascii
import bisect
import bz2
import functools
import io
import json
import lzma
import re
import tarfile
import zipfile
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pseudorca.extract.writes import flatten_add, parse_python
from pseudorca.tree import decode_text, safe_path

if TYPE_CHECKING:
    from pseudorca.extract import Limits

MIN_DECODED = 32  # bytes a text decoding (base64, ...) must yield to count
LZMA_MEMLIMIT = 128 << 20  # xz/lzma decoder memory; enough for `xz -9`, refuses absurd dictionary sizes
MAX_STREAMS = 64  # concatenated compressed streams followed in one payload (`cat a.gz b.gz`, pbzip2)
DECOMPRESS_WINDOW = 64 << 10  # compressed input given to a decompressor at a time

# ----- candidates ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    line: int  # 1-based line in the scanned text
    col: int
    value: str | bytes
    node: int = 0  # id() of the constant node; 0 for joined or pattern-found values
    parts: frozenset[int] = frozenset()  # constants a joined value was built from


def _joined(nodes: list[ast.expr], sep: str | bytes) -> tuple[str | bytes, frozenset[int]] | None:
    """`sep.join` of two or more constants of the separator's type, with the constants' ids."""
    if len(nodes) < 2:
        return None
    value: str | bytes | None
    if isinstance(sep, bytes):
        data = [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, bytes)]
        value = sep.join(data) if len(data) == len(nodes) else None
    else:
        text = [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        value = sep.join(text) if len(text) == len(nodes) else None
    return None if value is None else (value, frozenset(id(n) for n in nodes))


def ast_candidates(module: ast.AST, min_chars: int) -> list[Candidate]:
    """String and bytes constants of at least `min_chars`, plus `+` chains and `"".join([...])` of
    constants, in source order. A joined value comes before the constants it was built from."""
    nodes = list(ast.walk(module))
    inner = {
        id(child)
        for n in nodes
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)
        for child in (n.left, n.right)
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Add)
    }
    out: list[Candidate] = []
    for n in nodes:
        joined = None
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)):
            if len(n.value) >= min_chars:
                out.append(Candidate(n.lineno, n.col_offset, n.value, node=id(n)))
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) and id(n) not in inner:
            parts = flatten_add(n)
            first = parts[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, (str, bytes)):
                joined = _joined(parts, first.value[:0])
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "join"
            and isinstance(n.func.value, ast.Constant)
            and isinstance(n.func.value.value, (str, bytes))
            and len(n.args) == 1
            and not n.keywords
            and isinstance(n.args[0], (ast.List, ast.Tuple))
        ):
            joined = _joined(n.args[0].elts, n.func.value.value)
        if joined is not None and len(joined[0]) >= min_chars:
            out.append(Candidate(n.lineno, n.col_offset, joined[0], parts=joined[1]))
    out.sort(key=lambda c: (c.line, c.col, -len(c.parts)))
    return out


_LINE_ESCAPES_RE = re.compile(r"\\(?:\r\n|\r|\n|[nrt])")


@functools.lru_cache(maxsize=8)
def _run_re(min_chars: int) -> re.Pattern[str]:
    return re.compile(rf"[A-Za-z0-9+/=_-]{{{max(min_chars, 1)},}}")


_LITERAL_START_RE = re.compile(r"""[#'"]""")
_LINE_END_RE = re.compile(r"\r\n|\r|\n")
# what can end a string's body: a backslash (it escapes the next character), its closing quote, and for a
# one-quote string a line end
_STRING_STOP_RE = {
    "'": re.compile(r"[\\'\r\n]"),
    '"': re.compile(r'[\\"\r\n]'),
    "'''": re.compile(r"\\|'''"),
    '"""': re.compile(r'\\|"""'),
}


def _string_end(text: str, start: int, quote: str) -> tuple[bool, int]:
    """(True, index of the closing quote) for the string whose body begins at `start`, or (False, where
    the scan stopped) when it is not closed: at a one-quote string's line end, or at the end of the text."""
    stops = _STRING_STOP_RE[quote]
    j = start
    while (m := stops.search(text, j)) is not None:
        k = m.start()
        c = text[k]
        if c == "\\":
            j = k + 3 if text.startswith("\r\n", k + 1) else k + 2
        else:
            return c not in "\r\n", k
    return False, len(text)


def _string_literals(text: str) -> Iterator[tuple[int, int, str]]:
    """(start, end, body) of each quoted string, skipping `#` comments; a best-effort scan in linear time.

    After an opening quote whose string is never closed, the scan goes on right after that quote. Any
    later opening quote of the same kind before the point where that string's scan stopped is skipped
    without a scan of its own: it was an escaped character of that string, so its scan would run in step
    with the first one and stop at the same point (scanning it again made `"` and many `\\"` quadratic).
    """
    stopped: dict[str, int] = {}  # quote -> where the scan of an unclosed string of that kind stopped
    i = 0
    while (m := _LITERAL_START_RE.search(text, i)) is not None:
        i = m.start()
        c = text[i]
        if c == "#":
            eol = _LINE_END_RE.search(text, i)
            if eol is None:
                return
            i = eol.end()
            continue
        quote = c * 3 if text.startswith(c * 3, i) else c
        start = i + len(quote)
        if i < stopped.get(quote, 0):
            i = start
            continue
        closed, end = _string_end(text, start, quote)
        if not closed:
            stopped[quote] = end
            i = start
            continue
        yield i, end + len(quote), text[start:end]
        i = end + len(quote)


def pattern_candidates(text: str, min_chars: int, *, bare_runs: bool = False) -> list[Candidate]:
    """Long quoted strings found without parsing, for code that does not parse.

    With `bare_runs` (shell and other non-Python text) long unquoted runs of base64 characters count too.
    """
    found: list[tuple[int, str]] = []
    spans: list[tuple[int, int]] = []
    for start, end, body in _string_literals(text):
        spans.append((start, end))
        body = _LINE_ESCAPES_RE.sub("", body)
        if len(body) >= min_chars:
            found.append((start, body))
    if bare_runs:
        starts = [s for s, _ in spans]
        for m in _run_re(min_chars).finditer(text):
            k = bisect.bisect_right(starts, m.start()) - 1
            if k >= 0 and spans[k][1] >= m.end():
                continue  # inside a quoted string, already a candidate
            found.append((m.start(), m.group()))
    if not found:
        return []
    line_starts = [0, *(m.end() for m in _LINE_END_RE.finditer(text))]  # lines as the tree counts them
    out = []
    for offset, value in sorted(found):
        line = bisect.bisect_right(line_starts, offset)
        out.append(Candidate(line, offset - line_starts[line - 1], value))
    return out


# ----- formats ---------------------------------------------------------------------------------------

COMPRESSED_EXT = {"zlib": "zlib", "gzip": "gz", "xz": "xz", "lzma": "lzma", "bz2": "bz2"}
STEP_NAMES = {"xz": "lzma"}  # origins say "lzma" for both lzma containers (what `lzma.compress` makes)
_COMPRESSED_SUFFIXES = (".gz", ".xz", ".bz2", ".lzma", ".zz", ".z")
_MAGIC_EXT = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"%PDF-", "pdf"),
    (b"\x93NUMPY", "npy"),
    (b"\x7fELF", "elf"),
    (b"\x00asm", "wasm"),
    (b"PAR1", "parquet"),
    (b"SQLite format 3\x00", "sqlite"),
    (b"\x28\xb5\x2f\xfd", "zst"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"Rar!\x1a\x07", "rar"),
    (b"\x89HDF\r\n\x1a\n", "h5"),
)


def sniff(data: bytes) -> str:
    """The container format magic bytes announce: a compression, "tar", "zip", "pickle" or ""."""
    head = data[:8]
    if len(head) >= 2 and head[0] == 0x78 and head[1] in (0x01, 0x5E, 0x9C, 0xDA):
        return "zlib"
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if head.startswith(b"\x5d\x00\x00"):
        return "lzma"
    if head.startswith(b"BZh") and head[3:4] in (b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"):
        return "bz2"
    if data[257:262] == b"ustar":
        return "tar"
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "zip"
    if len(head) >= 2 and head[0] == 0x80 and head[1] <= 5:
        return "pickle"
    return ""


def magic_ext(data: bytes) -> str:
    """A file extension for well-known binary formats (images, PDF, numpy, ...), else ""."""
    if data[:4] == b"RIFF" and data[8:12] in (b"WEBP", b"WAVE"):
        return "webp" if data[8:12] == b"WEBP" else "wav"
    return next((ext for magic, ext in _MAGIC_EXT if data.startswith(magic)), "")


_PY_LINE_RE = re.compile(r"^[ \t]*(?:(?:async[ \t]+)?def|class|import|from[ \t]+[\w.]+[ \t]+import)\b", re.M)


def guess_text_ext(text: str, max_parse_chars: int) -> str:
    """The extension for decoded text: "json", "py", "sh" or "txt"."""
    head = text.lstrip()
    if head[:1] in ("{", "["):
        try:
            json.loads(head)
            return "json"
        except (ValueError, RecursionError):
            pass
    first = head.split("\n", 1)[0].rstrip()
    if first.startswith("#!"):
        if "python" in first:
            return "py"
        if first.endswith("sh"):
            return "sh"
    if _PY_LINE_RE.search(text):
        return "py"
    if len(text) <= max_parse_chars:
        module, _ = parse_python(text)
        if module is not None and any(not _bare_word(s) for s in module.body):
            return "py"
    return "txt"


def _bare_word(stmt: ast.stmt) -> bool:
    """A statement that is only a name, a constant or `a.b` (plain words parse as Python too)."""
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Name, ast.Constant, ast.Attribute))


# ----- text decodings --------------------------------------------------------------------------------

_B64_RE = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_B64URL_RE = re.compile(r"[A-Za-z0-9_-]+={0,2}")
_HEX_RE = re.compile(r"(?:0[xX])?(?:[0-9a-fA-F]{2})+")
_B85_RE = re.compile(r"[0-9A-Za-z!#$%&()*+\-;<=>?@^_`{|}~]+")
_A85_RE = re.compile(r"[!-uz]+")
_DATA_URI_RE = re.compile(r"data:[\w.+-]+/[\w.+-]+(?:;[\w.+=-]+)*;base64,", re.I)


def _text_decodings(s: str) -> Iterator[tuple[str, Callable[[], bytes]]]:
    """(step, decoder) for each encoding whose alphabet fits `s` (whitespace already removed).

    base64 (standard or urlsafe, padding fixed), then base85 and ascii85, then hex. A string in the
    standard base64 alphabet is not tried as base85 or ascii85: real base85 almost always has a
    character base64 lacks, and the pure-Python base85 decoders are slow on long strings.
    """
    standard = _B64_RE.fullmatch(s) is not None
    if standard or _B64URL_RE.fullmatch(s):
        body = s.rstrip("=")
        if len(body) % 4 != 1:
            padded = body + "=" * (-len(body) % 4)
            if standard:
                yield "base64", lambda: binascii.a2b_base64(padded)
            else:
                yield "base64url", lambda: base64.urlsafe_b64decode(padded)
    if not standard:
        if _B85_RE.fullmatch(s):
            yield "base85", lambda: base64.b85decode(s)
        body85 = s[2:-2] if s.startswith("<~") and s.endswith("~>") else s
        if _A85_RE.fullmatch(body85):
            yield "ascii85", lambda: base64.a85decode(body85)
    if _HEX_RE.fullmatch(s):
        yield "hex", lambda: bytes.fromhex(s[2:] if s[:2] in ("0x", "0X") else s)


# ----- decoding --------------------------------------------------------------------------------------


@dataclass
class Budget:
    """What is left for one extraction (shared by every payload of a notebook)."""

    total_left: int
    members_left: int


@dataclass
class Item:
    """One decoded file: text, or a binary file kept without content."""

    origin: str
    text: str | None = None
    data: bytes = b""  # content of a binary file (only to hash it)
    ext: str = ""  # extension of a binary file
    rel: str = ""  # path inside an archive; "" for a single decoded file
    decompressed: bool = False


@dataclass
class Result:
    items: list[Item] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.items or self.warnings)

    def add_member(self, name: str, sub: Result) -> None:
        for item in sub.items:
            if item.rel:
                item.rel = f"{name}/{item.rel}"
            else:
                item.rel = _strip_compression(name) if item.decompressed else name
        self.items.extend(sub.items)
        self.warnings.extend(sub.warnings)


def _strip_compression(name: str) -> str:
    """`a.json.gz` -> `a.json` (for a member that was decompressed)."""
    base = name.rsplit("/", 1)[-1].lower()
    for suffix in _COMPRESSED_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            return name[: -len(suffix)]
    return name


class _NotDecodable(Exception):
    """The bytes only look like the format their magic bytes announce."""


def _decompressor(kind: str) -> Any:
    if kind == "zlib":
        return zlib.decompressobj()
    if kind == "gzip":
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if kind == "bz2":
        return bz2.BZ2Decompressor()
    return lzma.LZMADecompressor(format=lzma.FORMAT_AUTO, memlimit=LZMA_MEMLIMIT)


def decompress(kind: str, data: bytes, cap: int) -> tuple[bytes, str]:
    """At most `cap` bytes of decompressed output (never more than cap + 1 are produced), plus what went
    wrong: "", "truncated", "damaged stream", "stream ended early", "empty stream" (a stream that decodes
    to nothing ends the decoding) or "more than MAX_STREAMS streams". Concatenated streams are followed, up
    to MAX_STREAMS of them. Raises _NotDecodable when nothing decodes at all.

    The input is fed a window at a time, so the end of a stream copies at most a window of what follows,
    and the time stays linear in the input however many streams it holds.
    """
    view = memoryview(data)
    chunks: list[bytes] = []
    size = pos = 0
    problem = ""
    for _ in range(MAX_STREAMS):
        d = _decompressor(kind)
        before = size
        while not problem:
            window = view[pos : pos + DECOMPRESS_WINDOW]
            try:
                chunk = d.decompress(window, cap + 1 - size)  # max_length >= 1 (0 means "no limit" to zlib)
            except (zlib.error, lzma.LZMAError, OSError, EOFError, ValueError):
                problem = "damaged stream"
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > cap:
                chunks[-1] = chunk[: len(chunk) - (size - cap)]
                problem = "truncated"
            elif d.eof:
                pos += len(window) - len(d.unused_data)
                break
            else:
                pos += len(window)  # below max_length, a decompressor takes in all of its input
                if pos >= len(data):
                    problem = "stream ended early"
        if problem:
            break
        follows = pos < len(data) and sniff(data[pos : pos + 512]) == kind
        if size == before:  # an empty stream: hostile input repeats them, and they cost no output
            if follows:
                problem = "empty stream"
            break
        if not follows:
            break
    else:
        problem = f"more than {MAX_STREAMS} streams"
    if not size:
        raise _NotDecodable(kind)
    return b"".join(chunks), problem


def label(text: str, limit: int = 200) -> str:
    """Untrusted text (a path, a member name) made safe for one line of an origin or a warning."""
    text = text[: limit + 1]
    if not text.isprintable():
        text = "".join(c if c.isprintable() else "?" for c in text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def brief(e: BaseException) -> str:
    """An exception as one short, printable line."""
    text = label(" ".join(str(e).split()), 160)
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


class Decoder:
    """Decodes string and bytes constants into files, within one extraction's limits."""

    def __init__(self, limits: Limits, budget: Budget) -> None:
        self.limits = limits
        self.budget = budget

    def decode(self, value: str | bytes, origin: str) -> Result | None:
        """The files and warnings a constant decodes to, or None when it is not a payload."""
        if isinstance(value, bytes):
            res = self.expand(value, origin, 0, contained=False)
            if res:
                return res
            if not value.isascii():
                return None
            value = value.decode("ascii")
        return self._decode_text(value, origin)

    def _decode_text(self, value: str, origin: str) -> Result | None:
        compact = "".join(value.split())
        uri = _DATA_URI_RE.match(compact)
        if uri:
            compact = compact[uri.end() :]
        for step, decode in _text_decodings(compact):
            try:
                raw = decode()
            except (ValueError, OverflowError):
                continue
            if len(raw) < MIN_DECODED:
                continue
            cap, notes = self.limits.max_payload_bytes, []
            if len(raw) > cap:
                raw = raw[:cap]
                notes.append(f"{origin} > {step}: truncated to {cap} bytes (max_payload_bytes)")
            res = self.expand(raw, f"{origin} > {step}", 1, contained=False)
            if res:
                self.budget.total_left -= len(raw)
                res.warnings[:0] = notes
                return res
        return None

    def expand(self, data: bytes, origin: str, depth: int, *, contained: bool) -> Result:
        """Classify bytes by their magic and decode further (`depth` decode steps were applied so far).

        Text is kept once something was decoded; other binary data only when it came out of a
        decompression or an archive, or has a known magic. Anything else is dropped (an empty Result).
        """
        kind = sniff(data)
        res: Result | None = None
        if kind in COMPRESSED_EXT:
            res = self._inflate(kind, data, origin, depth)
        elif kind in ("tar", "zip"):
            res = self._unpack(kind, data, origin, depth)
        elif kind == "pickle":  # recorded, never loaded
            return Result([Item(origin, data=data, ext="pkl")])
        if res is not None:
            return res
        if depth:
            text = decode_text(data)
            if text is not None:
                return Result([Item(origin, text=text)])
        ext = magic_ext(data)
        if ext or contained:
            return Result([Item(origin, data=data, ext=ext or "bin")])
        return Result()

    def _inflate(self, kind: str, data: bytes, origin: str, depth: int) -> Result | None:
        step = f"{origin} > {STEP_NAMES.get(kind, kind)}"
        cap = min(self.limits.max_payload_bytes, self.budget.total_left)
        if depth >= self.limits.max_depth or cap <= 0:
            try:
                decompress(kind, data, 64)  # only keep it if it really is that format
            except _NotDecodable:
                return None
            why = (
                f"max_depth {self.limits.max_depth}" if depth >= self.limits.max_depth else "max_total_bytes"
            )
            item = Item(origin, data=data, ext=COMPRESSED_EXT[kind])
            return Result([item], [f"{step}: not decompressed ({why} reached)"])
        try:
            out, problem = decompress(kind, data, cap)
        except _NotDecodable:
            return None
        self.budget.total_left -= len(out)
        res = self.expand(out, step, depth + 1, contained=True)
        if problem == "truncated":
            limit = "max_payload_bytes" if cap == self.limits.max_payload_bytes else "max_total_bytes"
            res.warnings.insert(0, f"{step}: output truncated at {cap} bytes ({limit})")
        elif problem:
            res.warnings.insert(0, f"{step}: {problem} after {len(out)} bytes; kept what was decoded")
        if len(res.items) == 1 and not res.items[0].rel:
            res.items[0].decompressed = True
        return res

    def _unpack(self, kind: str, data: bytes, origin: str, depth: int) -> Result | None:
        step = f"{origin} > {kind}"
        if depth >= self.limits.max_depth:
            item = Item(origin, data=data, ext=kind)
            return Result([item], [f"{step}: members not read (max_depth {self.limits.max_depth} reached)"])
        if kind == "zip" and data.count(b"PK\x01\x02") > 4 * max(self.limits.max_members, 1):
            item = Item(origin, data=data, ext=kind)
            return Result([item], [f"{step}: members not read (far more entries than max_members)"])
        archive = _open_archive(kind, data)
        if archive is None:
            return None
        res = Result()
        cap = self.limits.max_payload_bytes
        with archive:
            try:
                for name, size, regular, read in _entries(archive):
                    if self.budget.members_left <= 0:
                        res.warnings.append(
                            f"{step}: max_members ({self.limits.max_members}) reached; later members were not read"
                        )
                        break
                    self.budget.members_left -= 1
                    if not regular:  # directories, links, devices
                        continue
                    name = safe_path(label(name, 512))
                    member = f"{origin} > {kind} member {name}"
                    if size > cap:
                        res.warnings.append(f"{member}: skipped ({size} bytes > max_payload_bytes {cap})")
                        continue
                    if size > self.budget.total_left:
                        res.warnings.append(f"{member}: max_total_bytes reached; later members were not read")
                        break
                    try:
                        content = read(size)
                    except Exception as e:  # damaged or encrypted member: hostile input must not raise
                        res.warnings.append(f"{member}: not read ({brief(e)})")
                        continue
                    self.budget.total_left -= len(content)
                    res.add_member(name, self.expand(content, member, depth + 1, contained=True))
            except _Damaged as e:
                res.warnings.append(f"{step}: damaged archive ({e}); later members were not read")
        return res


class _Damaged(Exception):
    """An archive broke off while its members were being listed."""


def _open_archive(kind: str, data: bytes) -> tarfile.TarFile | zipfile.ZipFile | None:
    try:
        if kind == "tar":
            return tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
        return zipfile.ZipFile(io.BytesIO(data))
    except Exception:  # anything unreadable is simply not an archive
        return None


Entry = tuple[str, int, bool, Callable[[int], bytes]]  # name, size, regular file, read(n)


def _entries(archive: tarfile.TarFile | zipfile.ZipFile) -> Iterator[Entry]:
    """Members one at a time (a tar is read lazily, so a hostile one cannot list millions up front)."""
    if isinstance(archive, zipfile.ZipFile):
        for info in archive.infolist():
            link = (info.external_attr >> 16) & 0o170000 == 0o120000
            yield info.filename, info.file_size, not (info.is_dir() or link), _zip_reader(archive, info)
        return
    while True:
        try:
            member = archive.next()
        except Exception as e:
            raise _Damaged(brief(e)) from None
        if member is None:
            return
        yield member.name, member.size, member.isreg(), _tar_reader(archive, member)


def _zip_reader(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Callable[[int], bytes]:
    def read(n: int) -> bytes:
        with archive.open(info) as f:
            return f.read(n)

    return read


def _tar_reader(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Callable[[int], bytes]:
    def read(n: int) -> bytes:
        f = archive.extractfile(member)
        return f.read(n) if f is not None else b""

    return read
