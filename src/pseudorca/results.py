"""Results tables of local evaluation runs: one row per (item, draw), as CSV or JSONL.

    item,draw,score,latency_s,answer
    q001,0,1,12.5,42
    q001,1,0,11.9,41

- `item` is required. `draw` is optional: without it, a repeated item gets draws 0, 1, 2... in file order.
- Every other column whose non-empty values all parse as numbers is a metric (true/false count as 1/0, and
  empty values, `NA`, `NaN`, `null` and `None` are missing). Other columns are listed in `Table.ignored`.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from pseudorca.model import Run

FORMATS = ("csv", "jsonl")
ITEM, DRAW = "item", "draw"
DEFAULT_METRIC = "score"
_MISSING = frozenset({"", "na", "n/a", "nan", "null", "none"})
_BOOLS = {"true": 1.0, "false": 0.0}
_SUFFIXES = {".csv": "csv", ".jsonl": "jsonl", ".ndjson": "jsonl"}
_FIELD_LIMIT = 1 << 30

Row = tuple[int, dict[str, Any]]  # (line number in the file, values by column)


@dataclass
class Table:
    """A results table. Row i is draw `draws[i]` of item `items[i]`; `metrics[name][i]` is its value (NaN if missing)."""

    items: list[str]
    draws: list[int]
    metrics: dict[str, np.ndarray]
    ignored: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def draw_counts(self) -> dict[str, int]:
        """Rows per item, in first-seen order."""
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item] = counts.get(item, 0) + 1
        return counts


def format_for(path: str | Path) -> str:
    """The table format implied by a file name: .csv, or .jsonl/.ndjson."""
    p = Path(path)
    fmt = _SUFFIXES.get(p.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"cannot tell the format of {p.name} from its suffix; pass --format csv or --format jsonl"
        )
    return fmt


def load_table(data: bytes, fmt: str) -> Table:
    """Parse and validate a results table. Raises ValueError with a one-line reason."""
    if fmt == "csv":
        columns, rows = _read_csv(data)
    elif fmt == "jsonl":
        columns, rows = _read_jsonl(data)
    else:
        raise ValueError(f"unknown results format {fmt!r} (use csv or jsonl)")
    return _build(columns, rows)


def per_item(table: Table, metric: str) -> dict[str, np.ndarray]:
    """Each item's values of one metric in draw order, without missing values (an item may end up empty)."""
    import numpy as np

    if metric not in table.metrics:
        raise ValueError(f"no metric {metric!r} (metrics: {', '.join(table.metrics) or 'none'})")
    values = table.metrics[metric]
    grouped: dict[str, list[tuple[int, float]]] = {}
    for i, (item, draw) in enumerate(zip(table.items, table.draws, strict=True)):
        grouped.setdefault(item, []).append((draw, float(values[i])))
    out: dict[str, np.ndarray] = {}
    for item, pairs in grouped.items():
        pairs.sort(key=lambda p: p[0])
        arr = np.array([v for _, v in pairs], dtype=float)
        out[item] = arr[~np.isnan(arr)]
    return out


def distinct_runs(runs: Sequence[Run]) -> tuple[list[Run], list[tuple[Run, Run]]]:
    """Each results table once: (the first run of each table, [(a later run, the earlier run it repeats)]).

    Two runs share a table when the same file was imported twice, for example by two teammates before they
    pulled. Pooling both would count every draw twice: a narrower interval, and an A/A check that compares
    a table with itself. Pass runs oldest first.
    """
    first: dict[str, Run] = {}
    repeats: list[tuple[Run, Run]] = []
    for run in runs:
        earlier = first.get(run.results)
        if earlier is None:
            first[run.results] = run
        else:
            repeats.append((run, earlier))
    return list(first.values()), repeats


def pool(tables: Sequence[Table]) -> Table:
    """Several runs' tables as one: rows in run order, draws renumbered 0, 1, 2... per item across runs.

    A metric missing from a run is NaN in that run's rows. Pass each table once (see `distinct_runs`).
    """
    import numpy as np

    if not tables:
        raise ValueError("nothing to pool")
    if len(tables) == 1:
        return tables[0]
    names = list(dict.fromkeys(m for t in tables for m in t.metrics))
    items: list[str] = []
    draws: list[int] = []
    columns: dict[str, list[np.ndarray]] = {m: [] for m in names}
    next_draw: dict[str, int] = {}
    for t in tables:
        order = sorted(range(len(t)), key=lambda i: t.draws[i])  # stable: each item's rows in draw order
        for i in order:
            item = t.items[i]
            items.append(item)
            draws.append(next_draw.get(item, 0))
            next_draw[item] = draws[-1] + 1
        for m in names:
            col = t.metrics.get(m)
            columns[m].append(col[order] if col is not None else np.full(len(t), np.nan))
    metrics = {m: np.concatenate(parts) for m, parts in columns.items()}
    ignored = [c for c in dict.fromkeys(c for t in tables for c in t.ignored) if c not in metrics]
    return Table(items, draws, metrics, ignored)


# ----- parsing -------------------------------------------------------------------------------------


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"the table is not UTF-8 text (bad byte at offset {e.start})") from None


def _named(columns: list[str]) -> list[str]:
    """Strip header names, give unnamed columns a placeholder, and reject duplicates."""
    out = [c.strip() or f"(column {i})" for i, c in enumerate(columns, 1)]
    dupes = sorted({c for c in out if out.count(c) > 1})
    if dupes:
        raise ValueError(f"duplicate column(s) in the header: {', '.join(dupes)}")
    return out


def _read_csv(data: bytes) -> tuple[list[str], list[Row]]:
    reader = csv.reader(io.StringIO(_decode(data), newline=""))
    rows: list[Row] = []
    old_limit = csv.field_size_limit(_FIELD_LIMIT)  # long model outputs exceed the 128 KiB default
    try:
        header = next(reader, None)
        if header is None or not any(h.strip() for h in header):
            raise ValueError("the CSV file has no header row")
        columns = _named(header)
        for record in reader:
            if not any(cell.strip() for cell in record):
                continue
            if len(record) > len(columns):
                raise ValueError(
                    f"line {reader.line_num}: {len(record)} fields, but the header has {len(columns)}"
                )
            rows.append((reader.line_num, dict(zip(columns, record, strict=False))))
    except csv.Error as e:
        raise ValueError(f"cannot parse the CSV (line {reader.line_num}): {e}") from None
    finally:
        csv.field_size_limit(old_limit)
    return columns, rows


def _read_jsonl(data: bytes) -> tuple[list[str], list[Row]]:
    seen: dict[str, None] = {}
    rows: list[Row] = []
    for no, line in enumerate(_decode(data).split("\n"), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {no}: not valid JSON ({e.msg})") from None
        if not isinstance(obj, dict):
            raise ValueError(f"line {no}: expected a JSON object, got {type(obj).__name__}")
        for key in obj:
            seen.setdefault(key, None)
        rows.append((no, obj))
    names = list(seen)
    renamed = _named(names)
    if renamed != names:  # stripped or placeholder names: rekey the rows to match
        mapping = dict(zip(names, renamed, strict=True))
        rows = [(no, {mapping[k]: v for k, v in obj.items()}) for no, obj in rows]
    return renamed, rows


def _build(columns: list[str], rows: list[Row]) -> Table:
    import numpy as np

    if ITEM not in columns:
        listed = ", ".join(columns) or "none"
        raise ValueError(f'the table has no "{ITEM}" column (columns: {listed})')
    if not rows:
        raise ValueError("the table has no rows")
    items = [_item(row.get(ITEM), no) for no, row in rows]
    if DRAW in columns:
        draws = [_draw(row.get(DRAW), no) for no, row in rows]
    else:
        counter: dict[str, int] = {}
        draws = []
        for item in items:
            draws.append(counter.get(item, 0))
            counter[item] = draws[-1] + 1
    first: dict[tuple[str, int], int] = {}
    for (no, _), item, draw in zip(rows, items, draws, strict=True):
        if (item, draw) in first:
            raise ValueError(f"item {item!r} has draw {draw} twice (lines {first[item, draw]} and {no})")
        first[item, draw] = no
    metrics: dict[str, np.ndarray] = {}
    ignored: list[str] = []
    for col in columns:
        if col in (ITEM, DRAW):
            continue
        values = [_number(row.get(col)) for _, row in rows]
        if col.startswith("(column ") or any(v is None for v in values):
            ignored.append(col)
        else:
            metrics[col] = np.array(values, dtype=float)
    return Table(items, draws, metrics, ignored)


def _item(value: Any, line: int) -> str:
    if value is None or isinstance(value, (list, dict)):
        raise ValueError(f"line {line}: missing {ITEM}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"line {line}: empty {ITEM}")
    return text


def _draw(value: Any, line: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    number = value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            number = _number(value)
    if isinstance(number, float) and number.is_integer():
        return int(number)
    raise ValueError(f"line {line}: {DRAW} must be an integer, not {value!r}")


def _number(value: Any) -> float | None:
    """A cell as a float (NaN when missing), or None when it is not a number."""
    if value is None:
        return math.nan
    if isinstance(value, (bool, int, float)):
        try:
            return float(value)
        except OverflowError:
            return None
    if isinstance(value, str):
        text = value.strip()
        low = text.lower()
        if low in _MISSING:
            return math.nan
        if low in _BOOLS:
            return _BOOLS[low]
        try:
            return float(text)
        except ValueError:
            return None
    return None
