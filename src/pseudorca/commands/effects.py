"""`effects`: which ideas each submission has, and what each idea adds to the score."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import click

from pseudorca import util
from pseudorca.commands._common import SYMBOLS, Ctx, pass_ctx, show_warnings, user_errors
from pseudorca.util import interval_decimals as _decimals
from pseudorca.util import number as _num
from pseudorca.util import plain_number as _plain
from pseudorca.util import signed_number as _signed

if TYPE_CHECKING:
    from pseudorca.effects import Fit, Row, Term, Waterfall
    from pseudorca.model import State

MINUS = "−"
BAR = 21  # display columns of the interval sketch
PAD = "\u2800"  # a blank that format_table keeps (it is not whitespace); replaced by a space after formatting
NO_VERDICT = "·"
SOURCE_HINT = "record scores with `pseudorca sub add` or `pseudorca kaggle sync`"
LABEL_COLS = 40  # an IDEA label is cut to this many columns, and given in full under the table
NB_COLS = 24  # a notebook label is cut to this many columns, before the short sha that tells versions apart
TEXT_COLS = 400  # a table's last column (CHANGES, NOTE) is shown in full, short of an absurd ledger string
RULE_COLS = 60  # ...but the rule under its heading stops here
REASONS = 6  # reasons listed for the unused runs of a protocol without a model
SLOW = 3e8  # bootstrap draws (resamples × rows of the runs' tables) from which a hint says what is quicker


@click.command("effects")
@click.option("--idea", "idea_refs", multiple=True, metavar="ID", help="Model only these ideas (repeatable).")
@click.option("--all", "include_parked", is_flag=True, help="Include parked ideas.")
@click.option("--private", is_flag=True, help="Use private scores instead of public ones.")
@click.option(
    "--noise",
    type=click.FloatRange(min=0, min_open=True),
    metavar="SD",
    help="A known LB noise SD (default: [effects] noise in config.toml, else estimated from the model's "
    "residuals).",
)
@click.option("--sub", "target_ref", metavar="SUB", help="Break this submission down (default: the best).")
@click.option("--base", "base_ref", metavar="SUB", help="...starting from this one (default: the first).")
@click.option("--competition", metavar="SLUG", help="Only this competition's submissions.")
@click.option("--metric", default="score", show_default=True, help="Metric of the local runs.")
@click.option("--protocol", metavar="P", help="Only the local runs with this protocol.")
@click.option("--no-runs", is_flag=True, help="Leave out the local runs.")
@click.option(
    "--lower-is-better/--higher-is-better",
    default=None,
    help="Which scores are better, on the LB and in the runs (default: [effects] lower_is_better in "
    "config.toml, else higher).",
)
@click.option(
    "--confidence",
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    default=0.95,
    show_default=True,
    help="Interval confidence level.",
)
@click.option(
    "--resamples",
    type=click.IntRange(min=100),
    default=10_000,
    show_default=True,
    help="Bootstrap resamples (local runs); fewer is quicker.",
)
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True, help="Random seed.")
@click.option(
    "-n",
    "count",
    type=click.IntRange(min=0),
    default=20,
    show_default=True,
    metavar="N",
    help="Rows to list, also in --json: the latest N (0 = all). The model always uses all of them.",
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@pass_ctx
def effects_cmd(
    ctx: Ctx,
    idea_refs: tuple[str, ...],
    include_parked: bool,
    private: bool,
    noise: float | None,
    target_ref: str | None,
    base_ref: str | None,
    competition: str | None,
    metric: str,
    protocol: str | None,
    no_runs: bool,
    lower_is_better: bool | None,
    confidence: float,
    resamples: int,
    seed: int,
    count: int,
    as_json: bool,
) -> None:
    """Which ideas each submission has, and what each idea adds to the score.

    Each submission's ideas come from the verdicts on its notebook (parked ideas only with --all). The
    model is additive: score = baseline + the effects of the ideas in it + noise, fitted by least squares
    over the scored submissions, and separately over the local runs of each protocol (paired by item).
    Ideas that always come and go together are measured together, and what the submissions cannot
    separate is said so.

    The LB intervals use the residual SD (resubmit a notebook to make it reliable) or a known noise SD;
    the local ones a two-stage bootstrap over items and draws, as in `compare`, plus the run-to-run noise
    that runs with the same ideas show. The breakdown shows how the model gets from one submission to
    another, idea by idea, and names the ideas outside the model that differ between the two.

    \b
    Defaults in config.toml (the flags override them):
      [effects]
      lower_is_better = true   # a lower score is better
      noise = 0.012            # a known LB noise SD
    """
    from pseudorca import effects

    store = ctx.store()
    try:
        with user_errors():

            def setting(section: str, key: str, default: Any) -> Any:  # a flag overrides (and skips) a key
                if (key == "noise" and noise is not None) or (
                    key == "lower_is_better" and lower_is_better is not None
                ):
                    return default
                return store.setting(section, key, default)

            lower, configured = effects.config_defaults(setting)
            higher = not (lower if lower_is_better is None else lower_is_better)
            given = "--noise" if noise is not None else "[effects] noise in config.toml"
            st = store.state()
            ideas = effects.resolve_ideas(st, idea_refs) if idea_refs else None
            lb = effects.lb_fit(
                st,
                private=private,
                ideas=ideas,
                noise=configured if noise is None else noise,
                confidence=confidence,
                higher_is_better=higher,
                competition=competition,
                include_parked=include_parked,
            )
            breakdown = _breakdown(lb, base_ref, target_ref)
            fits: list[Fit] = []
            if not no_runs and (st.runs or protocol is not None):  # an unknown --protocol is an error
                from pseudorca.commands.results import _load

                hint = _slow_hint(st, protocol, resamples)
                if hint:
                    click.secho(hint, err=True, dim=True)
                fits = effects.run_fits(
                    st,
                    lambda run: _load(store, run),
                    metric=metric,
                    protocol=protocol,
                    ideas=ideas,
                    confidence=confidence,
                    higher_is_better=higher,
                    resamples=resamples,
                    seed=seed,
                    include_parked=include_parked,
                )
        if as_json:
            from pseudorca.commands.results import _echo_json

            _echo_json(
                {
                    "lb": _fit_json(lb, count),
                    "runs": [_fit_json(f, count) for f in fits],
                    "breakdown": effects.waterfall_json(breakdown) if breakdown else None,
                }
            )
            return
        hidden = 0 if include_parked else _hidden_parked(st, [lb, *fits])
        lines = _render(lb, fits, breakdown, count, given, hidden)
    except OverflowError as e:  # numbers near the limits of a float: a hand-edited score, an absurd --noise
        raise click.ClickException(
            f"a computation overflowed ({e}): are some scores, or the noise SD, huge?"
        ) from e
    for line in lines:  # ledger strings may hold control characters: every line is cleaned
        click.echo(util.clean(line.replace(PAD, " ")).rstrip())
    show_warnings(store)


def _breakdown(lb: Fit, base_ref: str | None, target_ref: str | None) -> Waterfall | None:
    """The breakdown to show: given submissions, else the best from the first (None when there is none, or
    when no idea is in the model)."""
    from pseudorca import effects

    used = lb.used
    if base_ref or target_ref:
        target = effects.pick_row(lb.rows, target_ref).key if target_ref else None
        base = effects.pick_row(lb.rows, base_ref).key if base_ref else None
        if target is None:
            best = lb.best()
            target = best.key if best else None
        if base is None:
            base = used[0].key if used else None
        if target is None or base is None:
            raise LookupError("no submission is in the model, so there is nothing to break down")
        return effects.waterfall(lb, base, target)
    best = lb.best()
    if lb._model is None or not lb._model.bundles or best is None or not used or best.key == used[0].key:
        return None
    return effects.waterfall(lb, used[0].key, best.key)


def _fit_json(fit: Fit, count: int) -> dict[str, Any]:
    """A fit as JSON with its latest `count` rows (0: all), and the total, as `lb --json` does."""
    from pseudorca import effects

    data = effects.fit_json(fit)
    data["total"] = len(data["rows"])
    if count:
        data["rows"] = data["rows"][-count:]
    return data


def _slow_hint(st: State, protocol: str | None, resamples: int) -> str:
    """A line to print before a long bootstrap of the local runs, else "". It draws about resamples × the
    rows (items × draws) of the tables of each protocol with two runs or more."""
    tables: dict[str, dict[str, int]] = {}  # protocol -> results table -> its rows (a table counts once)
    for run in st.runs.values():
        if run.nb and (protocol is None or run.protocol == protocol):
            rows = run.n_rows if isinstance(run.n_rows, int) and run.n_rows > 0 else 0
            tables.setdefault(run.protocol, {}).setdefault(run.results, rows)
    draws = resamples * sum(sum(t.values()) for t in tables.values() if len(t) > 1)
    if draws <= SLOW:
        return ""
    return (
        f"this may take a while: the bootstrap of the local runs draws about {draws:.2g} values "
        "(fewer --resamples is quicker, and --no-runs leaves the runs out)"
    )


def _hidden_parked(st: State, fits: Sequence[Fit]) -> int:
    """Parked ideas with a verdict on a notebook of the rows: left out unless --all."""
    nbs = {r.nb for fit in fits for r in fit.rows if r.nb}
    parked = [i.id for i in st.ideas.values() if i.status == "parked"]
    return sum(1 for iid in parked if any(st.effective_verdict(nb, iid) is not None for nb in nbs))


# ----- rendering -----------------------------------------------------------------------------------


def _render(
    lb: Fit, fits: list[Fit], breakdown: Waterfall | None, count: int, given: str, hidden: int
) -> list[str]:
    """The text output: the submissions, the protocols with a model, those without, the breakdown."""
    lines = _render_lb(lb, count, given, hidden)
    for fit in fits:
        if fit._model is not None:
            lines += ["", *_render_runs(fit, count)]
    idle = [f for f in fits if f._model is None]
    if idle:
        lines.append("")
        for fit in idle:
            lines += _render_idle(fit)
    if breakdown is not None:
        lines += ["", *_render_breakdown(lb, breakdown)]
    return lines


def _render_lb(fit: Fit, count: int, given: str, hidden: int) -> list[str]:
    from pseudorca import effects

    rows = fit.rows
    if not rows:
        return [f"no submissions yet ({SOURCE_HINT})"]
    first = len(rows) - count if 0 < count < len(rows) else 0
    ideas = effects.shown_ideas(rows)
    labels = _nb_labels(rows)
    dec, fdec = _score_columns(rows)
    moved, again = effects.all_changes(rows), effects.resubmits(rows)
    table = []
    for k in range(first, len(rows)):
        r = rows[k]
        notes = [] if r.used else [f"not used: {r.why_not}"]
        if again[k]:
            notes.append(f"resubmits {again[k]}")
        if moved[k]:
            notes.append(" ".join(moved[k]))
        table.append(
            [
                r.name,
                util.local_day(r.date),
                labels[k],
                _num(r.y, dec),
                _num(r.fitted, fdec),
                *(_symbol(r, i) for i in ideas),
                "; ".join(notes),
            ]
        )
    heads = ["SUB", "DATE", "NOTEBOOK", fit.target.split()[0].upper(), "FITTED"]
    heads += [str(n) for n in range(1, len(ideas) + 1)] + ["CHANGES"]
    lines = [f"Submissions, oldest first ({fit.target})", *_table(heads, table, {2: NB_COLS + 11})]
    if first:
        lines.append(
            f"(the latest {len(rows) - first} of {len(rows)}; -n 0 lists all, and the model uses all)"
        )
    if ideas:
        lines.append("ideas: " + "  ".join(f"{n} {i}" for n, i in enumerate(ideas, 1)))
        lines.append(
            "✓ present  ◐ partial  ✗ absent  ? unknown  ! methods disagree  · no verdict  * stale (re-audit)"
        )
    else:
        lines.append(
            "no verdicts on these submissions' notebooks yet: `pseudorca audit --submitted` audits them "
            "(`pseudorca kaggle pull` fetches Kaggle kernel versions)"
        )
    if hidden:
        lines.append(f"({hidden} parked idea{'' if hidden == 1 else 's'} hidden; use --all)")
    lines += ["", *_render_terms(fit, "SUBS", "submissions", given)]
    return lines


def _render_runs(fit: Fit, count: int) -> list[str]:
    from pseudorca import effects

    used = fit.used
    head = (
        f"Local runs ({_proto(fit)}), {fit.target}: {len(used)} of {len(fit.rows)} runs used, "
        f"{fit.n_items} items in every one"
    )
    first = len(fit.rows) - count if 0 < count < len(fit.rows) else 0
    ideas = effects.shown_ideas(fit.rows)
    labels = _nb_labels(fit.rows)
    dec, fdec = _score_columns(fit.rows, digits=4)  # means of a table
    table = [
        [
            r.name,
            labels[k],
            _num(r.y, dec),
            _num(r.fitted, fdec),
            *(_symbol(r, i) for i in ideas),
            "" if r.used else f"not used: {_why(r)}",
        ]
        for k, r in enumerate(fit.rows)
        if k >= first
    ]
    heads = ["RUN", "NOTEBOOK", "MEAN", "FITTED", *(str(n) for n in range(1, len(ideas) + 1)), "NOTE"]
    lines = [head, *_table(heads, table, {0: 28, 1: NB_COLS + 11})]
    if first:
        lines.append(
            f"(the latest {len(fit.rows) - first} of {len(fit.rows)}; -n 0 lists all, and the model uses all)"
        )
    if ideas:
        lines.append("ideas: " + "  ".join(f"{n} {i}" for n, i in enumerate(ideas, 1)))
    lines += ["", *_render_terms(fit, "RUNS", "runs")]
    return lines


def _render_idle(fit: Fit) -> list[str]:
    """A protocol whose runs give no model: one line, why runs are not used (grouped by reason), the notes."""
    need = next((n for n in fit.notes if n.startswith("nothing to compare")), "nothing to compare")
    lines = [f"local runs ({_proto(fit)}): {len(fit.used)} of {len(fit.rows)} usable, {need}"]
    groups: dict[str, list[str]] = {}
    for r in fit.rows:
        if not r.used:
            groups.setdefault(_why(r), []).append(r.name)
    for why, names in itertools.islice(groups.items(), REASONS):
        who = names[0] if len(names) == 1 else f"{len(names)} runs ({_some(names)})"
        lines.append(f"  not used: {who}: {why}")
    if len(groups) > REASONS:
        rest = sum(len(names) for names in list(groups.values())[REASONS:])
        lines.append(
            f"  not used: {rest} more run{'' if rest == 1 else 's'}, for other reasons (--json lists them)"
        )
    lines += [f"  {n}" for n in fit.notes if n is not need]
    return lines


def _render_terms(fit: Fit, count_head: str, what: str, given: str = "") -> list[str]:
    from pseudorca import effects

    used = fit.used
    if fit._model is None:
        return [*fit.notes] or [f"nothing to model yet: it needs {effects.MIN_ROWS} or more {what}"]
    conf = f"{fit.confidence * 100:g}% CI"
    title = f"What each idea adds to the {fit.target} (additive model over {len(used)} {what}"
    title += ")" if fit.higher_is_better else ", lower is better)"
    shown = [t for t in fit.terms if t.kind not in (effects.ALWAYS, effects.NEVER)]
    measured = fit.measured()
    dec = _decimals(measured)
    scale = max(
        (abs(v) for t in measured for v in (t.estimate, t.ci_low, t.ci_high) if _finite(v)),
        default=0.0,
    )
    table = []
    full: list[str] = []  # labels too long for the IDEA column, given in full under the table
    for t in shown:
        label = t.label
        if util.display_width(label) > LABEL_COLS:
            full.append(label)
            mark = f" [{len(full)}]"
            label = util.clip(label, LABEL_COLS - len(mark)) + mark
        if t.kind in effects.MEASURED:
            table.append(
                [
                    label,
                    _signed(t.estimate, dec),
                    _interval(t.ci_low, t.ci_high, dec),
                    f"{t.n_with}/{t.n_without}",
                    t.verdict,
                    _sketch(t, scale),
                    t.note,
                ]
            )
        else:
            table.append([label, "", "", f"{t.n_with}/{t.n_without}", "", "", t.note])
    lines = [title]
    if table:
        heads = ["IDEA", "EFFECT", conf, count_head, "VERDICT", _sketch_head(scale, dec), "NOTE"]
        lines += _table(heads, table, {0: LABEL_COLS})
        lines += [f"[{k}] {label}" for k, label in enumerate(full, 1)]
        foot = f"{count_head} = {what} with/without it"
        forms = sorted(
            {"“A instead of B”" if t.swapped else "“A vs B”" for t in shown if t.kind == effects.DIFFERENCE}
        )
        if forms:
            foot += f" (for {' and '.join(forms)}: with A/with B)"
        rule = f"the {conf} excludes 0"
        if fit.source == "local":
            rule += f" and the sign-flip p < {1 - fit.confidence:g}, as in `compare`"
        lines.append(f"{foot}. VERDICT is helps or hurts when {rule}.")
        lines += _verdict_notes(fit, {t.verdict for t in shown})
    else:
        lines.append(f"no idea is in some {what} but not all, so there is nothing to compare")
    always = [t.label for t in fit.terms if t.kind == effects.ALWAYS]
    never = [t.label for t in fit.terms if t.kind == effects.NEVER]
    if always:
        lines.append(f"in every one (cannot be measured here): {', '.join(always)}")
    if never:
        lines.append(f"in none (cannot be measured here): {', '.join(never)}")
    lines += _noise_lines(fit, given)
    lines += fit.notes
    return lines


def _verdict_notes(fit: Fit, verdicts: set[str]) -> list[str]:
    """What the verdicts besides helps, hurts and within noise mean, when the table has them."""
    from pseudorca import effects
    from pseudorca.stats import min_items

    out = []
    if effects.TOO_FEW in verdicts:
        out.append(
            f"too few items: {fit.n_items} items cannot reach p < {1 - fit.confidence:g} in the sign-flip "
            f"test (it needs {min_items(fit.confidence)} or more), so only the interval is informative"
        )
    if effects.UNKNOWN in verdicts:
        out.append("unknown: there is no interval, as there is no noise estimate")
    return out


def _noise_lines(fit: Fit, given: str) -> list[str]:
    out = []
    if fit.source == "lb":
        if fit.noise_from == "residuals":
            out.append(
                f"noise: residual SD {fit.sigma:.3g} with {fit.df:g} degrees of freedom; t intervals"
                + (
                    " (few degrees of freedom: wide intervals)"
                    if fit.df is not None and fit.df < 3 and (fit.sigma or 0) > 0
                    else ""
                )
            )
        elif fit.noise_from == "given":
            out.append(f"noise: SD {fit.sigma:.3g} as given ({given}); normal intervals")
        pure = fit.resubmission
        if pure is not None:
            out.append(
                f"resubmitting a notebook moved the score with SD {pure.sd:.3g} "
                f"({pure.n_groups} notebook(s), {pure.n_subs} submissions)"
            )
    else:
        out.append(
            "intervals: two-stage bootstrap over items and draws "
            f"({fit.resamples} resamples, seed {fit.seed}), the runs paired by item"
        )
        if fit.tau is None:
            out.append(
                "run-to-run SD: not measurable, as no two runs have the same ideas; the intervals assume "
                "none (run a notebook twice to measure it)"
            )
        elif fit.tau > 0:
            out.append(
                f"run-to-run SD {fit.tau:.3g}, from the runs with the same ideas ({fit.tau_df} degree"
                f"{'' if fit.tau_df == 1 else 's'} of freedom): in the intervals"
            )
        else:
            out.append("run-to-run SD 0: the runs with the same ideas agree within their item and draw noise")
    return out


def _render_breakdown(fit: Fit, w: Waterfall) -> list[str]:
    best = fit.best()
    used = fit.used
    what = " (best in the model)" if best is not None and w.target.key == best.key else ""
    start = " (first)" if used and w.base.key == used[0].key else ""
    lines = [f"Breakdown of {w.target.name}{what} from {w.base.name}{start}, by the model"]
    dec, fdec = _score_columns(fit.rows)  # as in the table above
    sdec = _decimals(w.steps, fallback=[w.start, w.end])  # the steps' intervals
    head, deltas, tail = _column(w, max(fdec, sdec))
    rows: list[tuple[str, str, str]] = [(f"{w.base.name} fitted", head, f"observed {_plain(w.base.y, dec)}")]
    for s, delta in zip(w.steps, deltas, strict=True):
        rows.append((s.label, delta, _interval(s.ci_low, s.ci_high, sdec)))
    if not w.steps:
        rows.append(("(no modelled idea differs)", "", ""))
    rows.append((f"= {w.target.name} fitted", tail, f"observed {_plain(w.target.y, dec)}"))
    width = max(util.display_width(a) for a, _, _ in rows)
    num = max(util.display_width(b) for _, b, _ in rows)
    for a, b, c in rows:
        pad = " " * (width - util.display_width(a))
        lines.append(f"  {a}{pad}  {' ' * (num - util.display_width(b))}{b}  {c}".rstrip())
    if w.unmodelled:
        lines.append(f"  not in the model, and different or unknown: {', '.join(w.unmodelled)}")
    return lines


# ----- formatting ----------------------------------------------------------------------------------


def _table(heads: list[str], rows: list[list[object]], widths: dict[int, int]) -> list[str]:
    """format_table's lines, with the last column (free text) in full. Its rule under the heading stops
    at RULE_COLS, so that one long note does not draw a rule that wraps."""
    lines = util.format_table(heads, rows, col_widths={**widths, len(heads) - 1: TEXT_COLS}).split("\n")
    rule = lines[1]
    last = rule.rfind(" ") + 1  # where the last column's rule starts (columns are two spaces apart)
    lines[1] = rule[: last + max(util.display_width(heads[-1]), RULE_COLS)]
    return lines


def _score_columns(rows: Sequence[Row], digits: int = 6) -> tuple[int, int]:
    """Decimals for the scores and for the fitted values (see util.score_columns)."""
    return util.score_columns([r.y for r in rows], [r.fitted for r in rows], digits)


def _nb_labels(rows: Sequence[Row]) -> list[str]:
    """Each row's notebook label (cut to NB_COLS), with the short sha when different notebooks among the
    rows would look the same (`kaggle pull` names every version of a kernel after its code file)."""
    labels = [util.clip(r.nb_label, NB_COLS) for r in rows]
    shas: dict[str, set[str]] = {}
    for r, label in zip(rows, labels, strict=True):
        if r.nb:
            shas.setdefault(label, set()).add(r.nb)
    return [
        f"{label} {util.short(r.nb)}" if r.nb and len(shas[label]) > 1 else label
        for r, label in zip(rows, labels, strict=True)
    ]


def _column(w: Waterfall, dec: int) -> tuple[str, list[str], str]:
    """The breakdown's start, steps and end at `dec` decimals; the printed column adds up."""
    return util.running_column(w.start, [s.delta for s in w.steps], w.end, dec)


def _interval(lo: float | None, hi: float | None, dec: int) -> str:
    """The interval as "[lo, hi]", or "" without one; a bound beyond the floats is −inf or +inf."""
    if lo is None or hi is None:
        return ""
    return f"[{_bound(lo, dec)}, {_bound(hi, dec)}]"


def _bound(x: float, dec: int) -> str:
    if math.isinf(x):
        return f"{MINUS if x < 0 else '+'}inf"
    return _signed(x, dec)


def _symbol(row: Row, idea: str) -> str:
    cell = row.cells[idea]
    if cell.status is None:
        return NO_VERDICT
    return SYMBOLS.get(cell.status, "?") + ("*" if cell.stale else "")


def _proto(fit: Fit) -> str:
    return f"protocol {fit.protocol}" if fit.protocol else "no protocol"


def _why(r: Row) -> str:
    """Why a run is not used, without the "run NAME: " that a table's load error starts with (its row, or
    its line, names it already)."""
    return r.why_not.removeprefix(f"run {r.name}: ")


def _some(names: Sequence[str], limit: int = 3) -> str:
    return ", ".join(names[:limit]) + (", …" if len(names) > limit else "")


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _sketch(t: Term, scale: float) -> str:
    """The estimate and interval on a fixed axis: `|` is 0, `[---*---]` the interval and the estimate.

    Rounding to columns never changes what the sketch says about 0: an interval that excludes it stays off
    its column, on its side, and one that includes it spans it. An infinite bound draws no interval."""
    if not _finite(t.estimate) or not _finite(scale) or not scale:
        return ""
    assert t.estimate is not None
    half = BAR // 2

    def pos(x: float) -> int:
        return max(0, min(BAR - 1, half + round(x / scale * half)))

    cells = [PAD] * BAR
    lo = hi = pos(t.estimate)
    if _finite(t.ci_low) and _finite(t.ci_high):
        assert t.ci_low is not None and t.ci_high is not None
        lo, hi = pos(t.ci_low), pos(t.ci_high)
        if t.ci_low > 0:
            lo = max(lo, half + 1)
            hi = max(hi, lo)
        elif t.ci_high < 0:
            hi = min(hi, half - 1)
            lo = min(lo, hi)
        else:
            lo, hi = min(lo, half - 1), max(hi, half + 1)
        for i in range(lo, hi + 1):
            cells[i] = "-"
        cells[lo], cells[hi] = "[", "]"
    cells[half] = "|"
    cells[min(max(pos(t.estimate), lo), hi)] = "*"
    return "".join(cells)


def _sketch_head(scale: float, dec: int) -> str:
    if not _finite(scale) or not scale:
        return ""
    text = _plain(scale, dec)
    # with its sign, each label must leave a blank beside the 0 in the middle, and never read as 0 itself
    if len(text) >= BAR // 2 - 1 or not float(text):
        text = f"{scale:.2g}"
    left, right = f"{MINUS}{text}", f"+{text}"
    cells = [PAD] * BAR
    for i, ch in enumerate(left):
        cells[i] = ch
    cells[BAR // 2] = "0"
    for i, ch in enumerate(reversed(right)):
        cells[BAR - 1 - i] = ch
    return "".join(cells)
