"""A single-file HTML report of the ledger: no scripts, no external resources. Open it locally.

The report holds the team's data. Keep it private: do not publish it (GitHub Pages sites are public even
when the repository is private).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pseudorca import __version__, charts, util
from pseudorca.model import Effective, Idea, Notebook, State

if TYPE_CHECKING:
    from pseudorca.effects import Fit, Row, Term, Waterfall

SYMBOLS = {"present": "✓", "partial": "◐", "absent": "✗", "unknown": "?", "conflict": "!"}
CELL_CLASS = {"present": "ok", "partial": "part", "absent": "no", "unknown": "unk", "conflict": "conf"}
MEANING = {"conflict": "methods disagree"}  # a status in the legends, when its name is not enough
CONFIG = ".pseudorca/config.toml"
# what drawing a part of the effects section may raise on odd numbers (hand-edited scores near the float
# limits, say); that part then becomes a note, and the rest of the report is still written
DRAW_ERRORS = (ValueError, LookupError, ArithmeticError, RecursionError)
_CODE_RE = re.compile(r"`([^`]+)`")

CSS = """
:root { --bg:#fff; --fg:#1d232a; --muted:#5f6b76; --line:#d9dee3; --head:#f4f6f8;
        --ok:#1f8a4c; --ok-bg:#e3f4ea; --part:#9a6700; --part-bg:#fdf3d8; --no:#b42318; --no-bg:#fde8e7;
        --unk:#5f6b76; --unk-bg:#eef1f4; --conf:#7a2bbf; --conf-bg:#f1e7fb; --fit:#2563c9; --fit-bg:#dde8f8;
        color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14181c; --fg:#e4e8ec; --muted:#9aa5b1; --line:#2c343c; --head:#1c2228;
          --ok:#5fd18f; --ok-bg:#16301f; --part:#e8b94a; --part-bg:#33290f; --no:#ff8a80; --no-bg:#3a1714;
          --unk:#9aa5b1; --unk-bg:#232a31; --conf:#c79cf0; --conf-bg:#2c1d3a; --fit:#4f8fe6; --fit-bg:#1b2738;
          color-scheme: dark; }
}
body { background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
       margin:0; padding:24px max(16px, 3vw); }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 8px; }
h3 { font-size:15px; margin:24px 0 6px; }
p.ct { font-weight:600; margin:18px 0 4px; }
.muted { color:var(--muted); } code { font-size:12.5px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; } th, td { border:1px solid var(--line); padding:4px 8px; vertical-align:top; }
th { background:var(--head); text-align:left; font-weight:600; }
td.c { text-align:center; font-weight:700; min-width:3.2em; }
td.ok { color:var(--ok); background:var(--ok-bg); } td.part { color:var(--part); background:var(--part-bg); }
td.no { color:var(--no); background:var(--no-bg); } td.unk { color:var(--unk); background:var(--unk-bg); }
td.conf { color:var(--conf); background:var(--conf-bg); } td.none { color:var(--muted); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
ul.tree { margin:0; padding-left:20px; } ul.tree li { margin:2px 0; } ul.flat { list-style:none; padding-left:0; }
.tag { display:inline-block; border:1px solid var(--line); border-radius:9px; padding:0 7px; font-size:12px; margin-right:3px; }
footer { margin-top:40px; color:var(--muted); font-size:12.5px; }
table.mx td.c { min-width:1.4em; padding:2px 5px; }
table.mx td.chg { box-shadow:inset 0 0 0 2px currentColor; }
table.mx th.v { vertical-align:bottom; padding:6px 3px; text-align:center; }
table.mx th.v span { writing-mode:vertical-rl; transform:rotate(180deg); white-space:nowrap;
                     font-weight:500; }
table.mx tr.off td { opacity:.55; } table.mx td { white-space:nowrap; }
table.mx td.chs { white-space:normal; min-width:24em; }
.lg { white-space:nowrap; margin-right:.8em; } p.muted code { white-space:nowrap; }
"""


_e = charts.escape


def _tags(tags: Iterable[str]) -> str:
    return "".join(f'<span class="tag">{_e(t)}</span>' for t in tags)


def _legend(keys: Iterable[tuple[str, str]]) -> str:
    """Symbol keys set apart by space, not by a character that could be read as a symbol; a narrow page
    wraps between them, never inside one."""
    return " ".join(f'<span class="lg">{_e(symbol)} {_e(meaning)}</span>' for symbol, meaning in keys) + " "


def _status_keys() -> list[tuple[str, str]]:
    return [(symbol, MEANING.get(status, status)) for status, symbol in SYMBOLS.items()]


def _cell(eff: Effective | None, claimed: bool) -> str:
    if eff is None:
        mark = "c" if claimed else "·"
        title = "claimed by the notebook, not audited yet" if claimed else "not audited"
        return f'<td class="c none" title="{_e(title)}">{mark}</td>'
    lines = [f"{eff.status} via {eff.method}" + (" (stale: the idea changed since)" if eff.stale else "")]
    for rec in eff.records[:3]:
        for ev in rec.evidence[:2]:
            where = f"{ev.get('path', '')}:{ev.get('line') or ''}"
            lines.append(f"{rec.method}: {where} {util.clip(str(ev.get('quote', '')), 120)}")
        if rec.note:
            lines.append(f"{rec.method}: {util.clip(rec.note, 160)}")
        lines.append(f"{util.local_time(rec.time)} by {rec.actor}")
    mark = SYMBOLS.get(eff.status, "?") + ("*" if eff.stale else "")
    return (
        f'<td class="c {CELL_CLASS.get(eff.status, "unk")}" title="{_e(chr(10).join(lines))}">{_e(mark)}</td>'
    )


def _board(state: State, ideas: list[Idea], notebooks: list[Notebook]) -> str:
    if not ideas or not notebooks:
        return '<p class="muted">Nothing to show yet: add ideas and notebooks, then run <code>pseudorca audit</code>.</p>'
    head = "".join(
        f'<th title="{_e(nb.id)}">{_e(util.clip(nb.label, 18))}<br><span class="muted"><code>{_e(util.short(nb.id, 8))}</code></span></th>'
        for nb in notebooks
    )
    rows = []
    for idea in ideas:
        cells = "".join(
            _cell(state.effective_verdict(nb.id, idea.id), idea.id in nb.applies) for nb in notebooks
        )
        rows.append(
            f'<tr><th><code>{_e(idea.id)}</code><br><span class="muted">{_e(util.clip(idea.title, 48))}</span></th>'
            f"<td>{_e(idea.status)}</td><td>{_e(idea.owner)}</td>{cells}</tr>"
        )
    legend = _legend([*_status_keys(), ("c", "claimed, not audited"), ("·", "none"), ("*", "stale")])
    return (
        f'<div class="scroll"><table><tr><th>idea</th><th>status</th><th>owner</th>{head}</tr>{"".join(rows)}</table></div>'
        f'<p class="muted">{legend}Hover a cell for the evidence.</p>'
    )


def _lineage(state: State) -> str:
    """Notebooks as a family tree. A single child continues at its parent's indent and only branches
    indent, so a long linear history stays flat; nothing is ever dropped."""
    ordered = state.notebooks_in_order()
    if not ordered:
        return '<p class="muted">No notebooks yet.</p>'
    children: dict[str | None, list[Notebook]] = {}
    for nb in ordered:
        parent = nb.parent if nb.parent in state.notebooks and nb.parent != nb.id else None
        children.setdefault(parent, []).append(nb)
    rows: list[tuple[int, Notebook]] = []
    seen: set[str] = set()
    stack = [(nb, 0) for nb in reversed(children.get(None, []))]
    while stack:
        nb, depth = stack.pop()
        if nb.id in seen:
            continue
        seen.add(nb.id)
        rows.append((depth, nb))
        kids = children.get(nb.id, [])
        for kid in reversed(kids):
            stack.append((kid, depth if len(kids) == 1 else depth + 1))
    rows.extend((0, nb) for nb in ordered if nb.id not in seen)  # notebooks caught in a parent loop
    items = []
    for depth, nb in rows:
        parent = state.notebooks.get(nb.parent or "")
        bits = [f"{_e(util.local_day(nb.added))} by {_e(nb.added_by)}"]
        if parent is not None:
            bits.append(f"from {_e(parent.label)}")
        if nb.applies:
            bits.append(f"applies {', '.join(_e(a) for a in nb.applies)}")
        items.append(
            f'<li style="margin-left:{min(depth, 12) * 18}px"><b>{_e(nb.label)}</b> <code>{_e(util.short(nb.id))}</code> '
            f'{_tags(nb.tags)}<span class="muted">{" · ".join(bits)}</span></li>'
        )
    return f'<ul class="tree flat">{"".join(items)}</ul>'


def _submissions(state: State, lb_noise: Any) -> str:
    subs = sorted(state.subs.values(), key=lambda s: (util.instant(s.date), s.id))
    if not subs:
        return '<p class="muted">No submissions yet (<code>pseudorca kaggle sync</code> or <code>pseudorca sub add</code>).</p>'
    rows = []
    for s in subs:
        nb = state.notebooks.get(s.nb or "")
        rows.append(
            f"<tr><td>{_e(util.local_time(s.date))}</td><td><code>{_e(s.id)}</code></td><td>{_e(nb.label if nb else '')}</td>"
            f'<td class="num">{_e(s.public_score if s.public_score is not None else "")}</td>'
            f'<td class="num">{_e(s.private_score if s.private_score is not None else "")}</td>'
            f"<td>{_e(s.status)}</td><td>{_e(util.clip(s.description, 80))}</td></tr>"
        )
    noise = ""
    if lb_noise is not None:
        noise = (
            f"<p>LB noise: pooled SD <b>{lb_noise.sd:.3g}</b> from {lb_noise.n_groups} notebook(s) submitted more than "
            f"once ({lb_noise.n_subs} submissions). Two single submissions differ by noise alone with SD "
            f"{lb_noise.sd * 2**0.5:.3g}.</p>"
        )
    else:
        noise = '<p class="muted">LB noise: submit the same notebook at least twice to estimate it.</p>'
    return (
        '<div class="scroll"><table><tr><th>date</th><th>id</th><th>notebook</th><th>public</th><th>private</th>'
        f"<th>status</th><th>description</th></tr>{''.join(rows)}</table></div>{noise}"
    )


@dataclass
class Effects:
    """What the "What each idea added" section shows. The caller computes it (`pseudorca report` does, as
    `pseudorca effects` would with the same options), so render() stays pure; `notes` say what could not be
    computed."""

    lb: Fit | None = None
    runs: list[Fit] = field(default_factory=list)
    breakdown: Waterfall | None = None  # the default breakdown of the LB fit, if there is one
    notes: list[str] = field(default_factory=list)


def _effects(state: State, data: Effects) -> str:
    """The section. Its parts (the leaderboard, then each protocol's runs) are drawn one by one: a part that
    fails on odd numbers becomes a note, and the others are still drawn."""
    fits = [f for f in (data.lb, *data.runs) if f is not None]
    intro = (
        "An additive model: score = baseline + what each idea in it adds + noise. A submission's ideas are "
        "the verdicts on its notebook; ideas that always come and go together are measured together. "
    )
    if fits:
        intro += _direction(fits[0].higher_is_better) + " "
    parts = [
        f'<p class="muted">{intro}<code>pseudorca effects</code> prints the same model in the terminal.</p>'
    ]
    parts += [_note([note]) for note in data.notes]
    if data.lb is not None:
        parts.append(
            _drawn("Leaderboard: this part", functools.partial(_lb_effects, state, data.lb, data.breakdown))
        )
    for fit in data.runs:
        parts.append(
            _drawn(f"Local runs ({_protocol(fit)}): this part", functools.partial(_run_effects, fit))
        )
    return "".join(parts)


def _drawn(what: str, draw: Callable[[], str]) -> str:
    """A chart or a part of the section; or, when it fails on odd numbers, a note saying so."""
    try:
        return draw()
    except DRAW_ERRORS as e:
        return _note([f"{what} cannot be drawn ({type(e).__name__}: {util.clip(str(e), 200)})"])


def _direction(higher_is_better: bool) -> str:
    """Which scores count as better, and where that is set (HTML)."""
    where = f"under <code>[effects]</code> in <code>{CONFIG}</code>"
    if higher_is_better:
        return (
            "Higher scores count as better; for an error metric, set <code>lower_is_better = true</code> "
            f"{where}."
        )
    return f"Lower scores count as better (<code>lower_is_better</code> {where})."


def _lower(fit: Fit) -> str:
    return "" if fit.higher_is_better else " (lower is better)"


def _protocol(fit: Fit) -> str:
    return f"protocol {fit.protocol}" if fit.protocol else "no protocol"


def _lb_effects(state: State, fit: Fit, breakdown: Waterfall | None) -> str:
    if not fit.rows:
        return (
            '<p class="muted">No submissions yet (<code>pseudorca kaggle sync</code> or '
            "<code>pseudorca sub add</code>).</p>"
        )
    parts = [f"<h3>Leaderboard: {_e(fit.target + _lower(fit))}</h3>", _matrix(state, fit)]
    chart = _drawn("the chart of the scores", functools.partial(charts.lb_chart, fit))
    if chart:
        parts.append(f'<p class="ct">{_e(fit.target.capitalize())} by submission, oldest first</p>{chart}')
    parts.append(_terms(fit, "submissions"))
    if breakdown is not None:
        parts.append(_breakdown(fit, breakdown))
    return "".join(parts)


def _breakdown(fit: Fit, w: Waterfall) -> str:
    """The waterfall from one submission to another, and the ideas outside the model that differ between
    the two (its steps cannot show them); or why there is nothing to break down."""
    base, target, outside = w.base.name, w.target.name, w.unmodelled
    one = len(outside) == 1
    if w.steps:
        lines = []
        if outside:
            lines.append(
                f"not in the model, so not in these steps: {_names(outside)} "
                f"({'it differs' if one else 'they differ'}, or may, between {base} and {target})"
            )
        title = f"From {base} to {target}, idea by idea (the model's fitted scores)"
        chart = _drawn("the breakdown", functools.partial(charts.waterfall, fit, w))
        return f'<p class="ct">{_e(title)}</p>{chart}{_note(lines)}'
    if outside:
        verb, be = ("differs", "is") if one else ("differ", "are")
        line = (
            f"no modelled idea differs between {base} and {target}, so there is nothing to break down; "
            f"{_names(outside)} {verb}, or may, but {be} not in the model (see above)"
        )
    elif w.base.nb is not None and w.base.nb == w.target.nb:
        line = f"{target} resubmits the notebook of {base}: nothing to break down"
    elif not any(c.status is not None for r in (w.base, w.target) for c in r.cells.values()):
        return ""  # no verdicts on either notebook: the matrix says so
    elif all(getattr(w.base.cells.get(i), "status", None) == c.status for i, c in w.target.cells.items()):
        line = f"{target} has the same ideas as {base}: nothing to break down"
    else:
        line = f"no modelled idea differs between {base} and {target}: nothing to break down"
    return _note([line])


def _names(ids: Sequence[str], cap: int = 6) -> str:
    return ", ".join(ids[:cap]) + (f" and {len(ids) - cap} more" if len(ids) > cap else "")


def _run_effects(fit: Fit) -> str:
    lines = [f"{len(fit.used)} of {len(fit.rows)} runs used, {fit.n_items} items in every one"]
    left_out = [f"{r.name} ({r.why_not})" for r in fit.rows if not r.used]
    if left_out:
        lines.append("not used: " + "; ".join(left_out[:6]) + ("; …" if len(left_out) > 6 else ""))
    if any(r.why_not.startswith("no metric ") for r in fit.rows):
        lines.append("`pseudorca report --metric NAME` uses another metric of the results tables")
    head = f"Local runs ({_protocol(fit)}): {fit.target}{_lower(fit)}"
    return f"<h3>{_e(head)}</h3>{_note(lines)}{_terms(fit, 'runs')}"


def _terms(fit: Fit, what: str) -> str:
    """The forest plot of what each idea adds, and what cannot be measured; or why there is no model."""
    from pseudorca import effects

    used = fit.used
    if not any(r.fitted is not None for r in used):
        return _note(fit.notes or [f"nothing to model yet: it needs {effects.MIN_ROWS} or more {what}"])
    chart = _drawn("the chart of the effects", functools.partial(charts.forest, fit, what))
    lines = [_rule(fit)] if chart else [_nothing_measured(fit, what)]
    lines += [f"{t.label}: {t.note}" for t in fit.measured() if _downgraded(t)]  # the tooltip is not enough
    always = [t.label for t in fit.terms if t.kind == effects.ALWAYS]
    never = [t.label for t in fit.terms if t.kind == effects.NEVER]
    if always:
        lines.append(f"in every one (cannot be measured here): {', '.join(always)}")
    if never:
        lines.append(f"in none (cannot be measured here): {', '.join(never)}")
    lines += [f"{t.label}: {t.note}" for t in fit.terms if t.kind in (effects.TANGLED, effects.EXCLUDED)]
    lines += _noise(fit) + fit.notes
    if not chart:
        return _note(lines)
    title = f"What each idea adds to the {fit.target} (additive model over {len(used)} {what})"
    return f'<p class="ct">{_e(title)}</p>{chart}{_note(lines)}'


def _rule(fit: Fit) -> str:
    """When a verdict is helps or hurts, as `pseudorca effects` says it."""
    from pseudorca import effects

    rule = f"helps or hurts when the {fit.confidence * 100:g}% interval excludes 0"
    if fit.source != "lb":
        rule += f" and the sign-flip p < {1 - fit.confidence:g}, as in `compare`"
    rule += "; within noise otherwise"
    if any(t.verdict == effects.UNKNOWN for t in fit.measured()):
        rule += ", and unknown without an interval"
    return rule


def _downgraded(t: Term) -> bool:
    """Whether the interval excludes 0 but the verdict is within noise (on local runs, the sign-flip test
    disagreed; the note says so)."""
    from pseudorca import effects

    lo, hi = t.ci_low, t.ci_high
    excludes = lo is not None and hi is not None and (lo > 0 or hi < 0)
    return excludes and t.verdict == effects.WITHIN_NOISE and bool(t.note)


def _nothing_measured(fit: Fit, what: str) -> str:
    """Why no idea is measured, by its cause."""
    from pseudorca import effects

    excluded = sum(t.kind == effects.EXCLUDED for t in fit.terms)
    tangled = sum(t.kind == effects.TANGLED for t in fit.terms)
    left_out = f"{excluded} {'idea is' if excluded == 1 else 'ideas are'} left out for missing verdicts"
    if excluded and tangled:
        return (
            f"no idea can be measured yet: {left_out}, and the others that vary cannot be separated "
            "(see below)"
        )
    if excluded:
        return f"no idea can be measured yet: {left_out} (see below)"
    if tangled:
        return "no idea can be measured on its own: the ideas that vary cannot be separated (see below)"
    if not fit.terms:
        return f"no idea has a verdict on the notebooks of these {what}, so there is nothing to compare"
    return f"no idea is in some {what} but not all, so there is nothing to compare"


def _noise(fit: Fit) -> list[str]:
    """Where the intervals come from, as `pseudorca effects` says it."""
    if fit.source != "lb":
        return [_bootstrap(fit)]
    out = []
    sd = charts.noise_sd(fit)
    if fit.noise_from == "residuals" and fit.df is not None:
        if sd == 0:
            out.append(
                f"noise: residual SD 0 with {fit.df:g} degrees of freedom: every submission in the model "
                "scored exactly its fitted value, so the intervals have no width"
            )
        else:
            few = " (few degrees of freedom: wide intervals)" if fit.df < 3 else ""
            out.append(
                f"noise: residual SD {charts.sig(sd)} with {fit.df:g} degrees of freedom; t intervals{few}"
            )
    elif fit.noise_from == "given":
        out.append(
            f"noise: SD {charts.sig(fit.sigma)} as given ([effects] noise in {CONFIG}); normal intervals"
        )
    pure = fit.resubmission
    if pure is not None:
        out.append(
            f"resubmitting a notebook moved the score with SD {charts.sig(pure.sd)} "
            f"({pure.n_groups} notebook(s), {pure.n_subs} submissions)"
        )
    return out


def _bootstrap(fit: Fit) -> str:
    """Where a local fit's intervals come from, with the run-to-run noise they include."""
    line = (
        f"intervals: two-stage bootstrap over items and draws ({fit.resamples} resamples, seed {fit.seed}), "
        "the runs paired by item"
    )
    if fit.tau is None:
        return line + (
            "; no two runs have the same ideas, so noise shared by a whole run cannot be measured, and the "
            "intervals leave it out"
        )
    if fit.tau > 0:
        return line + (
            f", plus a run-to-run SD of {charts.sig(fit.tau)} from runs with the same ideas "
            f"({fit.tau_df} degree{'' if fit.tau_df == 1 else 's'} of freedom)"
        )
    return line + "; runs with the same ideas show no run-to-run noise beyond that"


def _note(lines: Iterable[str]) -> str:
    """Muted lines; a `command` in backticks is shown as code, kept on one line."""
    text = "<br>".join(_CODE_RE.sub(r"<code>\1</code>", _e(line)) for line in lines)
    return f'<p class="muted">{text}</p>' if text else ""


def _matrix(state: State, fit: Fit) -> str:
    """Submissions × ideas: the verdicts on each submission's notebook, its score and fitted score, and what
    changed: each idea against the latest earlier submission with a present/absent verdict for it."""
    from pseudorca import effects

    rows = fit.rows
    ideas = effects.shown_ideas(rows)
    dec, fdec = util.score_columns([r.y for r in rows], [r.fitted for r in rows])  # as in the terminal
    heads = "".join(
        f'<th class="v" title="{_e(_idea_tip(state, iid))}"><span>{_e(util.clip(iid, 24))}</span></th>'
        for iid in ideas
    )
    body = []
    since: dict[str, str] = {}  # idea -> the latest submission with a present/absent verdict for it
    for r, moved, first in zip(rows, effects.all_changes(rows), effects.resubmits(rows), strict=True):
        notes = ([f"resubmits {first}"] if first else []) + moved
        cells = "".join(_matrix_cell(r, iid, moved, since) for iid in ideas)
        since.update({iid: r.name for iid, cell in r.cells.items() if cell.value is not None})
        why = "" if r.used else f' <span class="muted">not used: {_e(r.why_not)}</span>'
        attrs = "" if r.used else f' class="off" title="{_e("not used in the model: " + r.why_not)}"'
        body.append(
            f"<tr{attrs}><td><code>{_e(r.name)}</code></td><td>{_e(util.local_day(r.date))}</td>"
            f'<td>{_e(r.nb_label)}</td><td class="num">{_e(charts.num(r.y, dec))}</td>'
            f'<td class="num">{_e(charts.num(r.fitted, fdec))}</td>{cells}'
            f"<td class='chs'>{_e(' '.join(notes))}{why}</td></tr>"
        )
    table = (
        f'<div class="scroll"><table class="mx"><tr><th>submission</th><th>date</th><th>notebook</th>'
        f"<th>{_e(fit.target)}</th><th>fitted</th>{heads}<th>changes</th></tr>{''.join(body)}</table></div>"
    )
    if not ideas:
        return table + _note(
            [
                "No verdicts on these submissions' notebooks yet: run `pseudorca audit --submitted` "
                "(`pseudorca kaggle pull` fetches Kaggle kernel versions)."
            ]
        )
    legend = _legend([*_status_keys(), ("·", "no verdict"), ("*", "stale (re-audit)")])
    return table + (
        f'<p class="muted">{legend}<br>An outlined cell came in or left since the latest earlier submission '
        "with ✓ or ✗ for that idea; faded rows are not in the model (hover for why).</p>"
    )


def _matrix_cell(row: Row, iid: str, moved: Sequence[str], since: dict[str, str]) -> str:
    cell = row.cells[iid]
    if cell.status is None:
        return f'<td class="c none" title="{_e(iid)}: no verdict">·</td>'
    tip = f"{iid}: {cell.status}" + (" (stale: the idea changed since; re-audit)" if cell.stale else "")
    cls = CELL_CLASS.get(cell.status, "unk")
    came, left = ("+" + iid) in moved, (charts.MINUS + iid) in moved  # as effects.all_changes marks them
    if came or left:
        cls += " chg"
        tip += f"; {'came in' if came else 'left'} since {since.get(iid, '')}"
    mark = SYMBOLS.get(cell.status, "?") + ("*" if cell.stale else "")
    return f'<td class="c {cls}" title="{_e(tip)}">{_e(mark)}</td>'


def _idea_tip(state: State, iid: str) -> str:
    idea = state.ideas.get(iid)
    return f"{iid}: {idea.title}" if idea is not None and idea.title else iid


def _ideas(ideas: list[Idea]) -> str:
    if not ideas:
        return '<p class="muted">No ideas yet.</p>'
    rows = []
    for i in ideas:
        checks = [f"{len(i.fingerprints)} fingerprint(s)"] if i.fingerprints else []
        if i.patch:
            checks.append("patch")
        links = " ".join(
            f'<a href="{_e(u)}">{_e(util.clip(u, 50))}</a>'
            for u in i.links
            if u.startswith(("http://", "https://"))
        )
        rows.append(
            f"<tr><td><code>{_e(i.id)}</code></td><td>{_e(i.title)}</td><td>{_e(i.status)}</td><td>{_e(i.owner)}</td>"
            f"<td>{_tags(i.tags)}</td><td>{_e(', '.join(checks) or '-')}</td>"
            f"<td>{_e(util.clip(i.description, 160))} {links}</td></tr>"
        )
    return (
        '<div class="scroll"><table><tr><th>id</th><th>title</th><th>status</th><th>owner</th><th>tags</th>'
        f"<th>checks</th><th>description</th></tr>{''.join(rows)}</table></div>"
    )


def _activity(state: State, n: int = 30) -> str:
    rows = []
    for ev in reversed(state.events[-n:]):
        d = ev.data
        what = d.get("id") or f"{d.get('idea', '')} on {util.short(d.get('nb'), 8)}"
        extra = d.get("status", "")
        rows.append(
            f"<tr><td>{_e(util.local_time(ev.time))}</td><td>{_e(ev.actor)}</td><td>{_e(ev.type)}</td>"
            f"<td>{_e(util.clip(str(what), 60))} {_e(extra)}</td></tr>"
        )
    if not rows:
        return '<p class="muted">No activity yet.</p>'
    return f'<div class="scroll"><table><tr><th>time</th><th>who</th><th>event</th><th>what</th></tr>{"".join(rows)}</table></div>'


def render(
    state: State,
    *,
    store_name: str,
    notebooks: list[Notebook] | None = None,
    include_parked: bool = False,
    lb_noise: Any = None,
    generated: str | None = None,
    effects: Effects | None = None,
) -> str:
    """The whole report as an HTML string (the body of a page; it has its own <title> and <style>).

    With `effects`, a section after the submissions shows which ideas each submission has and what each
    idea adds to the score."""
    ideas = [i for i in state.ideas_in_order() if include_parked or i.status != "parked"]
    shown = notebooks if notebooks is not None else state.notebooks_in_order()[-8:]
    generated = generated or util.now_iso()
    counts = (
        f"{len(state.ideas)} ideas · {len(state.notebooks)} notebooks · {len(state.subs)} submissions · "
        f"{len(state.runs)} runs · generated {generated}"
    )
    added = f"<h2>What each idea added</h2>{_effects(state, effects)}" if effects is not None else ""
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(store_name)} · Pseudorca</title><style>{CSS}{charts.CSS}</style></head><body>"
        f"<h1>{_e(store_name)}</h1><div class='muted'>{_e(counts)}</div>"
        f"<h2>Which ideas are in which notebook</h2>{_board(state, ideas, shown)}"
        f"<h2>Notebooks</h2>{_lineage(state)}"
        f"<h2>Submissions</h2>{_submissions(state, lb_noise)}{added}"
        f"<h2>Ideas</h2>{_ideas(ideas)}"
        f"<h2>Recent activity</h2>{_activity(state)}"
        f"<footer>Pseudorca {_e(__version__)}. This file contains your team's data: keep it private and do not "
        f"publish it (GitHub Pages sites are public even for private repositories).</footer></body></html>"
    )
