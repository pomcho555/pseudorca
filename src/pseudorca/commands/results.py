"""`run`, `compare` and `lb`: local evaluation runs, paired comparisons, and leaderboard noise."""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from pseudorca import util
from pseudorca.commands._common import Ctx, fail, pass_ctx, show_warnings, user_errors, warn
from pseudorca.store import StoreError

if TYPE_CHECKING:
    from pseudorca.model import Run, State, Submission
    from pseudorca.results import Table
    from pseudorca.stats import CompareResult, LbNoise
    from pseudorca.store import Store

MINUS = "−"
RUN_PREFIX_MIN = 6
SUB_PREFIX_MIN = 4
Z_BEYOND = 2.0  # |z| at or above this counts as beyond the LB noise


# ----- run -----------------------------------------------------------------------------------------


@click.group("run")
def run_group() -> None:
    """Record and list local evaluation runs.

    A run is a results table with one row per (item, draw): CSV or JSONL with an `item` column, an
    optional `draw` column, and numeric metric columns such as `score`.
    """


@run_group.command("add")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--nb", "nb_ref", metavar="REF", help="The notebook this run evaluated.")
@click.option("--label", default="", help="A name to compare by; several runs may share one.")
@click.option(
    "--protocol", default="", help="Evaluation protocol (items, settings); compare runs only within one."
)
@click.option("--note", default="", help="Free text.")
@click.option(
    "--format", "fmt", type=click.Choice(["csv", "jsonl"]), help="Table format (default: from the suffix)."
)
@pass_ctx
def run_add(
    ctx: Ctx, file: Path, nb_ref: str | None, label: str, protocol: str, note: str, fmt: str | None
) -> None:
    """Record a results table (CSV or JSONL).

    The table needs an `item` column. `draw` is optional: without it, repeats of an item are numbered in
    file order. Numeric columns are metrics; other columns are ignored.
    """
    from pseudorca.results import DEFAULT_METRIC, format_for, load_table

    store = ctx.store()
    with user_errors():
        fmt = fmt or format_for(file)
        try:
            data = file.read_bytes()
        except OSError as e:
            fail(f"cannot read {file}: {e.strerror or e}")
        table = load_table(data, fmt)
        if not table.metrics:
            fail("the table has no metric columns (numeric columns besides item and draw)")
        st = store.state()
        nb = st.resolve_nb(nb_ref) if nb_ref else None
        sha = util.sha256_bytes(data)
        for other in sorted(st.runs.values(), key=lambda r: r.id):
            if other.results == sha:
                warn(f"this table was already added as run {_run_name(other)}; pooling both counts it twice")
        store.put_blob(data)
        run_id = util.new_ulid()
        store.append(
            "run.add",
            {
                "id": run_id,
                "results": sha,
                "format": fmt,
                "nb": nb.id if nb else None,
                "label": label,
                "protocol": protocol,
                "note": note,
                "n_rows": len(table),
                "file": file.name,
            },
        )
    tags = [f"label {label}" if label else "", f"notebook {nb.label}" if nb else "", _protocol_tag(protocol)]
    head = f"added run {run_id[:10]}"
    click.echo(head + (f" ({', '.join(t for t in tags if t)})" if any(tags) else ""))
    for line in _table_overview(table):
        click.echo("  " + line)
    if DEFAULT_METRIC not in table.metrics:
        warn(f"no {DEFAULT_METRIC!r} column: compare this run with --metric NAME")


@run_group.command("list")
@click.option("--nb", "nb_ref", metavar="REF", help="Only the runs of this notebook.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@pass_ctx
def run_list(ctx: Ctx, nb_ref: str | None, as_json: bool) -> None:
    """List runs, oldest first."""
    store = ctx.store()
    with user_errors():
        st = store.state()
        runs = sorted(st.runs.values(), key=lambda r: r.id)
        if nb_ref:
            nb = st.resolve_nb(nb_ref)
            runs = [r for r in runs if r.nb == nb.id]
    if as_json:
        click.echo(json.dumps([dataclasses.asdict(r) for r in runs], ensure_ascii=False, indent=1))
        return
    if not runs:
        click.echo(
            "no runs" + (" for that notebook" if nb_ref else "") + " (add one with `pseudorca run add FILE`)"
        )
    else:
        rows = [
            [r.id[:10], r.label, _nb_label(st, r.nb), r.protocol, r.n_rows, _when(r.added), r.added_by]
            for r in runs
        ]
        click.echo(util.format_table(["ID", "LABEL", "NOTEBOOK", "PROTOCOL", "ROWS", "ADDED", "BY"], rows))
    show_warnings(store)


@run_group.command("show")
@click.argument("run_ref", metavar="RUN")
@pass_ctx
def run_show(ctx: Ctx, run_ref: str) -> None:
    """Show a run and a summary of each metric.

    RUN is a run id, a unique id prefix (6+ characters), or a label only one run has.
    """
    import numpy as np

    from pseudorca.results import per_item

    store = ctx.store()
    with user_errors():
        st = store.state()
        run = _resolve_run(st, run_ref)
        table = _load(store, run)
    fields = [
        ("label", run.label),
        ("notebook", _nb_long(st, run.nb)),
        ("protocol", run.protocol),
        ("note", run.note),
        ("added", f"{_when(run.added)} by {run.added_by}"),
        ("table", f"{run.format}, {len(table)} rows, blob {util.short(run.results, 12)}"),
    ]
    click.echo(f"run {run.id}")
    for key, value in fields:
        if value:
            click.echo(f"{key:<9} {value}")
    lo, med, hi = _count_spread(table.draw_counts().values())
    click.echo(f"{'items':<9} {len(table.draw_counts())}, draws per item: min {lo}, median {med:g}, max {hi}")
    rows = []
    for name, column in table.metrics.items():
        values = column[~np.isnan(column)]
        means = [v.mean() for v in per_item(table, name).values() if v.size]
        rows.append(
            [
                name,
                len(means),
                values.size,
                column.size - values.size,
                _num(float(np.mean(means))) if means else "n/a",
                _num(float(np.std(means, ddof=1))) if len(means) > 1 else "n/a",
                _num(float(values.min())) if values.size else "n/a",
                _num(float(values.max())) if values.size else "n/a",
            ]
        )
    click.echo("")
    click.echo(util.format_table(["METRIC", "ITEMS", "VALUES", "MISSING", "MEAN", "SD", "MIN", "MAX"], rows))
    click.echo("MEAN is over items (each item's draws averaged first); SD is the SD of those item means.")
    if table.ignored:
        click.echo(f"ignored columns: {', '.join(table.ignored)}")
    show_warnings(store)


# ----- compare -------------------------------------------------------------------------------------


@dataclasses.dataclass
class _Arm:
    ref: str
    name: str  # how the output names the arm
    runs: list[Run]


@click.command("compare")
@click.argument("a")
@click.argument("b")
@click.option("--metric", default="score", show_default=True, help="Metric column to compare.")
@click.option(
    "--resamples",
    type=click.IntRange(min=100),
    default=10_000,
    show_default=True,
    help="Bootstrap resamples.",
)
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True, help="Random seed.")
@click.option(
    "--confidence",
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    default=0.95,
    show_default=True,
    help="Interval confidence level.",
)
@click.option("--lower-is-better", is_flag=True, help="Lower values are better (latency, cost, errors).")
@click.option("--protocol", metavar="P", help="Use only the runs with this protocol, in both arms.")
@click.option("--force", is_flag=True, help="Compare even when the runs' protocols differ.")
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
@pass_ctx
def compare_cmd(
    ctx: Ctx,
    a: str,
    b: str,
    metric: str,
    resamples: int,
    seed: int,
    confidence: float,
    lower_is_better: bool,
    protocol: str | None,
    force: bool,
    as_json: bool,
) -> None:
    """Does B beat A beyond the noise?

    A and B are each a run id (or a unique prefix of 6+ characters), else a run label (all its runs),
    else a notebook (all its runs). Runs in one arm are pooled. Items are paired across the arms, each
    item's draws are averaged, and the mean per-item difference B − A gets a two-stage bootstrap interval
    (items, then draws within items), a sign-flip permutation p-value, and the minimum detectable effect
    at 80% power (MDE80).
    """
    from pseudorca.results import per_item, pool
    from pseudorca.stats import compare

    store = ctx.store()
    with user_errors():
        st = store.state()
        arm_a, arm_b = _resolve_arm(st, a), _resolve_arm(st, b)
        if protocol is not None:
            arm_a, arm_b = _with_protocol(arm_a, protocol), _with_protocol(arm_b, protocol)
        _check_protocols(arm_a, arm_b, force)
        shared = {r.id for r in arm_a.runs} & {r.id for r in arm_b.runs}
        if shared:
            warn(f"the arms share {len(shared)} run(s)")
        tables_a = [_load(store, r) for r in arm_a.runs]
        tables_b = [_load(store, r) for r in arm_b.runs]
        _check_metric(
            metric, [*zip(arm_a.runs, tables_a, strict=True), *zip(arm_b.runs, tables_b, strict=True)]
        )
        values_a = per_item(pool(tables_a), metric)
        values_b = per_item(pool(tables_b), metric)
        result = compare(
            values_a,
            values_b,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            higher_is_better=not lower_is_better,
        )
    if as_json:
        payload = {
            "metric": metric,
            "arm_a": {"ref": arm_a.ref, "name": arm_a.name, "runs": [r.id for r in arm_a.runs]},
            "arm_b": {"ref": arm_b.ref, "name": arm_b.name, "runs": [r.id for r in arm_b.runs]},
            **dataclasses.asdict(result),
        }
        click.echo(json.dumps(_jsonable(payload), ensure_ascii=False, indent=1))
        return
    for line in _render_compare(result, metric, (arm_a, values_a), (arm_b, values_b)):
        click.echo(line)
    if result.n_items < 10:
        warn(f"only {result.n_items} item(s) compared: the interval and p-value are rough")
    show_warnings(store)


def _render_compare(
    res: CompareResult, metric: str, arm_a: tuple[_Arm, dict[str, Any]], arm_b: tuple[_Arm, dict[str, Any]]
) -> list[str]:
    dec = _decimals(res)
    conf = f"{res.confidence * 100:g}% CI"

    def interval(r: CompareResult) -> str:
        return f"{_signed(r.diff, dec)}  {conf} [{_signed(r.ci_low, dec)}, {_signed(r.ci_high, dec)}]"

    lines = [f"{side}  {_describe_arm(arm, values)}" for side, (arm, values) in (("A", arm_a), ("B", arm_b))]
    lines += [
        f"items compared: {res.n_items} (only in A: {len(res.only_a)}, only in B: {len(res.only_b)})",
        f"mean {metric}: A {_plain(res.mean_a, dec)}  B {_plain(res.mean_b, dec)}"
        + ("  (lower is better)" if not res.higher_is_better else ""),
        f"B − A: {interval(res)}",
        f"B wins/losses/ties: {res.wins}/{res.losses}/{res.ties}  p {_p(res.p_value)} (sign-flip permutation)",
        f"SE {_plain(res.se, dec)}  MDE80 {_plain(res.mde80, dec)}",
    ]
    for side, check in (("A", res.aa_a), ("B", res.aa_b)):
        if check is not None:
            ok = check.ci_low <= 0 <= check.ci_high
            note = "ok" if ok else "excludes 0: the draws may not be independent repeats"
            lines.append(f"A/A on {side} (draws split in halves): {interval(check)}  {note}")
    if res.aa_a is None and res.aa_b is None:
        lines.append("A/A check: needs an arm with 2+ draws on 5+ items")
    reason = "includes 0" if res.verdict == "within noise" else "excludes 0"
    lines.append(f"verdict: {res.verdict} (the {conf} {reason})")
    return lines


def _describe_arm(arm: _Arm, values: dict[str, Any]) -> str:
    counts = [v.size for v in values.values() if v.size]
    lo, med, hi = _count_spread(counts)
    if lo == hi:
        draws = f"{lo} draw{'s' if lo != 1 else ''} per item"
    else:
        draws = f"{lo}–{hi} draws per item (median {med:g})"
    runs = f"{len(arm.runs)} runs, " if len(arm.runs) > 1 else ""
    return f"{arm.name}: {runs}{len(counts)} items, {draws}"


def _check_protocols(arm_a: _Arm, arm_b: _Arm, force: bool) -> None:
    if len({r.protocol for r in [*arm_a.runs, *arm_b.runs]}) <= 1:
        return

    def summary(runs: list[Run]) -> str:
        counts = Counter(r.protocol for r in runs)
        return ", ".join(
            f"{repr(p) if p else 'no protocol'} ({_plural(n, 'run')})" for p, n in counts.items()
        )

    detail = f"A: {summary(arm_a.runs)}; B: {summary(arm_b.runs)}"
    if not force:
        raise ValueError(
            f"the runs have different protocols ({detail}); compare like with like "
            "(--protocol P keeps one), or pass --force"
        )
    warn(f"comparing runs with different protocols ({detail})")


def _with_protocol(arm: _Arm, protocol: str) -> _Arm:
    runs = [r for r in arm.runs if r.protocol == protocol]
    if not runs:
        have = ", ".join(sorted({repr(r.protocol) if r.protocol else "no protocol" for r in arm.runs}))
        raise LookupError(f"{arm.name} has no runs with protocol {protocol!r} (its protocols: {have})")
    return dataclasses.replace(arm, runs=runs)


def _check_metric(metric: str, loaded: list[tuple[Run, Table]]) -> None:
    missing = [run for run, table in loaded if metric not in table.metrics]
    if not missing:
        return
    everywhere = [m for m in loaded[0][1].metrics if all(m in t.metrics for _, t in loaded)]
    available = ", ".join(everywhere) or "none"
    if len(missing) == len(loaded):
        raise ValueError(f"no metric {metric!r} in these runs (metrics: {available})")
    names = ", ".join(_run_name(r) for r in missing[:3]) + (", …" if len(missing) > 3 else "")
    raise ValueError(
        f"metric {metric!r} is missing from {len(missing)} of {len(loaded)} runs ({names}); "
        f"metrics in every run: {available}"
    )


# ----- lb ------------------------------------------------------------------------------------------


@click.command("lb")
@click.option("--nb", "nb_ref", metavar="REF", help="Only this notebook's submissions.")
@click.option(
    "-n",
    "limit",
    type=click.IntRange(min=0),
    default=20,
    show_default=True,
    help="Show the newest N (0: all).",
)
@click.option(
    "--diff",
    "diff_refs",
    nargs=2,
    metavar="SUB_A SUB_B",
    help="Compare two submissions' public scores against the LB noise.",
)
@pass_ctx
def lb_cmd(ctx: Ctx, nb_ref: str | None, limit: int, diff_refs: tuple[str, str] | None) -> None:
    """Leaderboard submissions and the LB noise.

    Submissions are listed newest last. The LB noise is the pooled SD of the public scores of notebooks
    submitted more than once.
    """
    from pseudorca.stats import lb_noise

    store = ctx.store()
    with user_errors():
        st = store.state()
        subs = sorted(st.subs.values(), key=lambda s: (s.date or s.updated, s.id.zfill(20)))
        noise = lb_noise(subs)
        if diff_refs:
            sub_a, sub_b = (_resolve_sub(st, ref) for ref in diff_refs)
            for line in _render_diff(st, sub_a, sub_b, noise):
                click.echo(line)
            show_warnings(store)
            return
        shown = subs
        if nb_ref:
            nb = st.resolve_nb(nb_ref)
            shown = [s for s in subs if s.nb == nb.id]
    if not shown:
        click.echo(
            "no submissions of that notebook"
            if nb_ref
            else "no submissions yet (record them with `pseudorca sub` or `pseudorca kaggle sync`)"
        )
    else:
        total = len(shown)
        if limit:
            shown = shown[-limit:]
        rows = [
            [
                s.id,
                _when(s.date),
                _nb_label(st, s.nb),
                _score(s.public_score),
                _score(s.private_score),
                s.status,
            ]
            for s in shown
        ]
        click.echo(util.format_table(["ID", "DATE", "NOTEBOOK", "PUBLIC", "PRIVATE", "STATUS"], rows))
        if len(shown) < total:
            click.echo(f"(the newest {len(shown)} of {total}; -n 0 shows all)")
    click.echo("")
    if noise is None:
        click.echo(
            "LB noise: unknown. It needs a notebook submitted at least twice; "
            "resubmitting an unchanged notebook shows how much the public score moves by chance."
        )
    else:
        click.echo(
            f"LB noise: pooled SD {noise.sd:.3g} of the public score, from {_plural(noise.n_groups, 'notebook')} "
            f"and {_plural(noise.n_subs, 'submission')}."
        )
        click.echo(
            f"Two single submissions differ by chance with SD {noise.sd * math.sqrt(2):.3g}; "
            f"a gap under {Z_BEYOND * noise.sd * math.sqrt(2):.3g} is within noise (|z| < {Z_BEYOND:g})."
        )
    show_warnings(store)


def _render_diff(st: State, a: Submission, b: Submission, noise: LbNoise | None) -> list[str]:
    from pseudorca.stats import lb_diff

    score_a, score_b = a.public_score, b.public_score
    if score_a is None or score_b is None:
        raise ValueError(f"submission {(a if score_a is None else b).id} has no public score yet")
    lines = [
        f"{side}  {s.id}  {_nb_label(st, s.nb) or '(no notebook)'}  public {_score(s.public_score)}"
        + (f"  ({_when(s.date)})" if s.date else "")
        for side, s in (("A", a), ("B", b))
    ]
    if noise is None:
        lines.append(f"B − A: {_signed_g(score_b - score_a)} (public score)")
        lines.append("verdict: unknown, no LB noise estimate yet (submit a notebook at least twice)")
        return lines
    diff, z = lb_diff(score_a, score_b, noise.sd)
    lines.append(f"B − A: {_signed_g(diff)} (public score)")
    lines.append(
        f"z = {_signed(z, 2)} against the LB noise (pooled SD {noise.sd:.3g} from "
        f"{_plural(noise.n_groups, 'notebook')} and {_plural(noise.n_subs, 'submission')}; z = diff / (SD·√2))"
    )
    if abs(z) < Z_BEYOND:
        verdict = "within LB noise"
    else:
        verdict = f"{'B' if z > 0 else 'A'} higher, beyond LB noise"
    lines.append(f"verdict: {verdict}")
    return lines


# ----- lookups -------------------------------------------------------------------------------------


def _runs_by_id(st: State) -> list[Run]:
    return sorted(st.runs.values(), key=lambda r: r.id)


def _id_prefix_hits(st: State, ref: str) -> list[Run]:
    if len(ref) < RUN_PREFIX_MIN:
        return []
    up = ref.upper()
    return [r for r in _runs_by_id(st) if r.id.startswith(up)]


def _resolve_run(st: State, ref: str) -> Run:
    """One run: full id, unique id prefix (6+ characters), or a label only one run has."""
    ref = ref.strip()
    if ref in st.runs:
        return st.runs[ref]
    hits = _id_prefix_hits(st, ref)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise LookupError(f"{ref!r} matches {len(hits)} runs; use more characters")
    labelled = [r for r in _runs_by_id(st) if r.label == ref]
    if len(labelled) == 1:
        return labelled[0]
    if labelled:
        ids = ", ".join(r.id[:10] for r in labelled)
        raise LookupError(f"{len(labelled)} runs have the label {ref!r} ({ids}); give a run id")
    raise LookupError(f"no run matches {ref!r} (see `pseudorca run list`)")


def _resolve_arm(st: State, ref: str) -> _Arm:
    """A compare arm: a run id or unique id prefix (6+), else a run label, else a notebook reference."""
    ref = ref.strip()
    if ref in st.runs:
        run = st.runs[ref]
        return _Arm(ref, f"run {_run_name(run)}", [run])
    hits = _id_prefix_hits(st, ref)
    if len(hits) == 1:
        return _Arm(ref, f"run {_run_name(hits[0])}", hits)
    labelled = [r for r in _runs_by_id(st) if r.label == ref]
    if labelled:
        return _Arm(ref, f"label {ref}", labelled)
    try:
        nb = st.resolve_nb(ref) if st.notebooks else None
    except LookupError:
        nb = None
    if nb is not None:
        of_nb = [r for r in _runs_by_id(st) if r.nb == nb.id]
        if not of_nb:
            raise LookupError(
                f"notebook {nb.label} has no runs (add one with `pseudorca run add FILE --nb REF`)"
            )
        return _Arm(ref, f"notebook {nb.label}", of_nb)
    if len(hits) > 1:
        raise LookupError(f"{ref!r} matches {len(hits)} runs; use more characters")
    raise LookupError(f"no run, run label or notebook matches {ref!r} (see `pseudorca run list`)")


def _resolve_sub(st: State, ref: str) -> Submission:
    ref = ref.strip()
    if ref in st.subs:
        return st.subs[ref]
    hits = [s for s in st.subs.values() if s.id.startswith(ref)] if len(ref) >= SUB_PREFIX_MIN else []
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise LookupError(f"{ref!r} matches {len(hits)} submissions; use more characters")
    raise LookupError(f"no submission matches {ref!r} (see `pseudorca lb`)")


def _load(store: Store, run: Run) -> Table:
    from pseudorca.results import load_table

    if not store.has_blob(run.results):
        raise StoreError(
            f"run {_run_name(run)}: its results table ({util.short(run.results, 12)}) is not in this store; "
            "pull the latest .pseudorca from your repository"
        )
    try:
        return load_table(store.get_blob(run.results), run.format)
    except ValueError as e:
        raise ValueError(f"run {_run_name(run)}: {e}") from None


# ----- formatting ----------------------------------------------------------------------------------


def _table_overview(table: Table) -> list[str]:
    import numpy as np

    from pseudorca.results import per_item

    counts = table.draw_counts()
    lo, med, hi = _count_spread(counts.values())
    means = []
    for name in table.metrics:
        item_means = [v.mean() for v in per_item(table, name).values() if v.size]
        means.append(f"{name} {_num(float(np.mean(item_means))) if item_means else 'n/a'}")
    lines = [
        f"{len(table)} rows, {len(counts)} items, draws per item: min {lo}, median {med:g}, max {hi}",
        f"metrics (mean over items): {', '.join(means)}",
    ]
    if table.ignored:
        lines.append(f"ignored columns: {', '.join(table.ignored)}")
    return lines


def _count_spread(counts: Any) -> tuple[int, float, int]:
    from pseudorca.stats import spread

    return spread(counts)


def _run_name(run: Run) -> str:
    return f"{run.id[:10]} ({run.label})" if run.label else run.id[:10]


def _nb_label(st: State, nb_id: str | None) -> str:
    if not nb_id:
        return ""
    nb = st.notebooks.get(nb_id)
    return nb.label if nb else util.short(nb_id)


def _nb_long(st: State, nb_id: str | None) -> str:
    label = _nb_label(st, nb_id)
    return label if not nb_id or label == util.short(nb_id) else f"{label} ({util.short(nb_id)})"


def _protocol_tag(protocol: str) -> str:
    return f"protocol {protocol}" if protocol else ""


def _when(stamp: str) -> str:
    return (stamp or "")[:16].replace("T", " ")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _num(x: float) -> str:
    return f"{x:.4g}"


def _score(x: float | None) -> str:
    return "" if x is None else f"{x:.6g}"


def _decimals(res: CompareResult) -> int:
    """Decimals that show the interval to about two significant digits."""
    for base in (res.ci_high - res.ci_low, res.se, abs(res.diff)):
        if math.isfinite(base) and base > 0:
            return min(8, max(0, 1 - math.floor(math.log10(base))))
    return 3


def _plain(x: float, dec: int) -> str:
    if not math.isfinite(x):
        return "n/a"
    text = f"{abs(x):.{dec}f}"
    return MINUS + text if x < 0 and float(text) else text


def _signed(x: float, dec: int) -> str:
    if math.isnan(x):
        return "n/a"
    if math.isinf(x):
        return ("+" if x > 0 else MINUS) + "inf"
    text = f"{abs(x):.{dec}f}"
    if not float(text):
        return text
    return ("+" if x > 0 else MINUS) + text


def _signed_g(x: float) -> str:
    return f"{x:+.4g}".replace("-", MINUS)


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def _jsonable(value: Any) -> Any:
    """JSON-safe copy: non-finite floats become null, tuples become lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
