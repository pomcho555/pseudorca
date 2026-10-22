"""Inline SVG charts for the HTML report: the scores over the submissions, what each idea adds (forest
plots), and a waterfall from one submission to another.

Each chart is one <svg> with a viewBox, scaled to the page: no scripts, no external resources, and no xmlns
(HTML does not need it). Marks are styled by the classes in CSS, which use the report's colour variables,
so the charts follow its light and dark themes. Hovering a submission or a row shows its details (an SVG
<title>). Every string from the ledger is escaped, with control characters and bidirectional overrides
shown as "?": it is shared through git and may be hand-edited.

Numbers follow `pseudorca effects`: a Unicode minus, decimals from the interval widths, and compact numbers
from 1e6 on. Odd values (hand-edited scores near the float limits) give a wrong axis, never an error.
"""

from __future__ import annotations

import html
import itertools
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pseudorca import util

if TYPE_CHECKING:
    from pseudorca.effects import Fit, Row, Term, Waterfall

MINUS = "−"
WIDTH = 900  # viewBox width; the page shows a chart at most this wide
FONT = 12  # text size in viewBox units (tick labels are one smaller)
ROW = 26  # one forest or waterfall row
BAR = 14  # waterfall bar thickness
PLOT_H = 220  # leaderboard plot height
LANES = 3  # rows of change labels above the leaderboard plot
LABEL_COLS = 30  # a change label is cut to about this many columns (the tooltip has all of it)
MIN_PLOT = 240  # the narrowest plot of a forest or waterfall: longer row labels are cut to keep it
LIMIT = 1e300  # an axis over values beyond this could overflow: it becomes 0..1 instead
TINY = 1e-12  # a noise SD below this share of the scores' scale is rounding: it is shown as 0
GOOD, BAD, NEUTRAL, ZERO = "ok", "no", "mu", "zr"  # mark colours: helps, hurts, neither, nothing

CSS = """
svg.chart { display:block; width:100%; max-width:900px; min-width:720px; height:auto; font-size:12px;
            font-variant-numeric:tabular-nums; }
svg.chart text { fill:var(--fg); } svg.chart text.m { fill:var(--muted); }
svg.chart .tick text { fill:var(--muted); font-size:11px; }
svg.chart .grid { stroke:var(--line); stroke-width:1; }
svg.chart .axis { stroke:var(--muted); stroke-width:1; }
svg.chart .zero { stroke:var(--muted); stroke-width:1.5; }
svg.chart .marks { pointer-events:none; }
svg.chart .hit { fill:transparent; } svg.chart .hit:hover { fill:var(--head); }
svg.chart .band { fill:var(--fit-bg); }
svg.chart .fit { fill:none; stroke:var(--fit); stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }
svg.chart .obs { fill:var(--fg); stroke:var(--bg); stroke-width:1.5; }
svg.chart .obs.off { fill:var(--bg); stroke:var(--muted); }
svg.chart .guide { stroke:var(--line); stroke-width:1; }
svg.chart .pole { stroke:var(--muted); stroke-width:1; }
svg.chart .chg { stroke:var(--fg); stroke-width:1.5; }
svg.chart .ci { stroke-width:2; stroke-linecap:round; }
svg.chart .ci.ok { stroke:var(--ok); } svg.chart .ci.no { stroke:var(--no); }
svg.chart .ci.mu { stroke:var(--muted); }
svg.chart .dot { stroke:var(--bg); stroke-width:2; }
svg.chart .ok { fill:var(--ok); } svg.chart .no { fill:var(--no); } svg.chart .mu { fill:var(--muted); }
svg.chart .zr { fill:var(--muted); opacity:.6; }
svg.chart .total { fill:var(--fit); }
svg.chart .wk { stroke:var(--fg); stroke-width:1.5; } svg.chart .link { stroke:var(--muted); stroke-width:1; }
svg.chart .cut { stroke:var(--bg); stroke-width:2; fill:none; }
.legend { color:var(--muted); font-size:12.5px; margin:4px 0 0; }
.legend .key { display:inline-block; vertical-align:middle; margin:0 5px 0 14px; }
.legend .key:first-child { margin-left:0; }
.key.obs { width:8px; height:8px; border-radius:50%; background:var(--fg); }
.key.off { width:5px; height:5px; border-radius:50%; border:1.5px solid var(--muted); }
.key.fit { width:16px; height:2px; background:var(--fit); }
.key.band { width:16px; height:10px; background:var(--fit-bg); }
.key.chg { width:2px; height:10px; background:var(--fg); }
.key.dot { width:8px; height:8px; border-radius:50%; } .key.bar { width:14px; height:8px; border-radius:2px; }
.key.total { background:var(--fit); } .key.ok { background:var(--ok); } .key.no { background:var(--no); }
.key.mu { background:var(--muted); } .key.zr { background:var(--muted); opacity:.6; } .key.wk { width:14px; height:1.5px; background:var(--fg); }
.key.cut { width:14px; height:8px; border-radius:2px;
           background:linear-gradient(115deg, var(--fit) 40%, var(--bg) 40% 60%, var(--fit) 60%); }
"""


# ----- numbers -------------------------------------------------------------------------------------


num = util.number
plain = util.plain_number
signed = util.signed_number
interval_decimals = util.interval_decimals
score_decimals = util.score_decimals


def noise_sd(fit: Fit) -> float | None:
    """The fit's noise SD as shown: 0 when it is below TINY of the scores' scale (rounding, not noise), None
    without an estimate."""
    if not _finite(fit.sigma):
        return None
    assert fit.sigma is not None
    scale = max((abs(float(r.y or 0)) for r in fit.used if _finite(r.y)), default=0.0)
    return 0.0 if fit.sigma <= TINY * scale else fit.sigma


def noise_halfwidth(fit: Fit) -> float | None:
    """t·sigma at the fit's confidence: how far a score may fall from its fitted value by noise alone (None
    without a noise estimate, or when it is 0)."""
    from pseudorca.stats import t_quantile

    sd = noise_sd(fit)
    if not sd or fit.df is None or not fit.df > 0:
        return None
    half = -t_quantile((1 - fit.confidence) / 2, fit.df) * sd  # the lower tail: exact for any level
    return half if math.isfinite(half) and half > 0 else None


def term_text(t: Term, dec: int) -> str:
    """A term's numbers, "+0.031 [+0.002, +0.060]"; the estimate alone when there is no interval."""
    if _finite(t.ci_low) and _finite(t.ci_high):
        return f"{signed(t.estimate, dec)} [{signed(t.ci_low, dec)}, {signed(t.ci_high, dec)}]"
    return f"{signed(t.estimate, dec)} (no interval)"


# ----- leaderboard ---------------------------------------------------------------------------------


@dataclass
class _Label:
    x0: float  # extent of the text
    x1: float
    at: float  # the boundary it belongs to
    anchor: str
    text: str
    title: str
    lane: int | None = None


def lb_chart(fit: Fit) -> str:
    """The scores over the submissions, oldest first: the observed scores; the fitted values as a step
    line, in a band of ± t·sigma (where the scores should fall if the fit were exact); and, above the plot,
    what changed where the ideas change. Returns "" when no submission has a score."""
    from pseudorca.effects import all_changes

    rows = [r for r in fit.rows if _finite(r.y)]
    if not rows:
        return ""
    n = len(rows)
    half = noise_halfwidth(fit)
    fitted = [r.fitted if r.used and _finite(r.fitted) else None for r in rows]
    known = [f for f in fitted if f is not None]
    ys = [float(r.y) for r in rows if r.y is not None] + known
    lo, hi = min(ys), max(ys)
    if half and known:
        reach = max(hi - lo, abs(hi) * 0.02, 1e-9)  # a wide band may widen the axis by one data span at most
        lo = min(lo, max(min(known) - half, lo - reach))
        hi = max(hi, min(max(known) + half, hi + reach))
    lo, hi, ticks, step = _axis(lo, hi, 4)
    tick_texts = _tick_texts(ticks, step)
    left = max(40.0, max(_width(s, FONT - 1) for s in tick_texts) + 14)
    right = 14.0
    slot = (WIDTH - left - right) / n
    xs = [left + (i + 0.5) * slot for i in range(n)]

    # each idea against the latest plotted submission with a present/absent verdict for it, so that a
    # submission without verdicts (no notebook linked, say) hides no change
    moved = all_changes(rows)
    labels = _change_labels(rows, [left + i * slot for i in range(n)], moved)
    lanes = max((lab.lane + 1 for lab in labels if lab.lane is not None), default=0)
    top = 10 + lanes * 15 + (6 if labels else 0)
    bottom = top + PLOT_H
    sy = _scale(lo, hi, bottom, top)
    height = bottom + 30

    out = [_open(height, f"{fit.target} of {n} submissions, oldest first, with the model's fitted values")]
    # hover targets first, so their highlight sits under the marks
    for i, r in enumerate(rows):
        title = _sub_title(fit, r, moved[i])
        out.append(
            f'<g><title>{_e(title)}</title><rect class="hit" x="{_f(left + i * slot)}" y="{_f(top)}" '
            f'width="{_f(slot)}" height="{PLOT_H}"/></g>'
        )
    for lab in labels:
        if lab.lane is not None:
            y = top - 6 - lab.lane * 15
            out.append(
                f'<g><title>{_e(lab.title)}</title><rect class="hit" x="{_f(lab.x0 - 2)}" y="{_f(y - 11)}" '
                f'width="{_f(lab.x1 - lab.x0 + 4)}" height="14"/></g>'
            )
    out.append('<g class="marks">')
    out += _y_grid(ticks, tick_texts, sy, left, WIDTH - right)
    if half and known:
        for run in _runs(fitted):
            top_pts = [(p, sy(fitted[i] + half)) for i in run for p in (xs[i] - slot / 2, xs[i] + slot / 2)]
            low_pts = [
                (p, sy(fitted[i] - half)) for i in reversed(run) for p in (xs[i] + slot / 2, xs[i] - slot / 2)
            ]
            out.append(f'<path class="band" d="{_poly(top_pts + low_pts)}"/>')
    for lab in labels:  # a tick at every change; a guide across the plot for those with a label
        if lab.lane is not None:
            out.append(
                f'<line class="guide" x1="{_f(lab.at)}" y1="{_f(top)}" x2="{_f(lab.at)}" y2="{_f(bottom)}"/>'
            )
        out.append(
            f'<line class="chg" x1="{_f(lab.at)}" y1="{_f(top - 4)}" x2="{_f(lab.at)}" y2="{_f(top + 3)}"/>'
        )
    for run in _runs(fitted):
        pts = [(p, sy(fitted[i])) for i in run for p in (xs[i] - slot / 2, xs[i] + slot / 2)]
        out.append(f'<path class="fit" d="{_poly(pts, close=False)}"/>')
    radius = min(4.0, max(1.8, slot * 0.32))
    for i, r in enumerate(rows):
        cls = "obs" if r.used else "obs off"
        out.append(
            f'<circle class="{cls}" cx="{_f(xs[i])}" cy="{_f(sy(float(r.y or 0)))}" r="{radius:.1f}"/>'
        )
    for lab in labels:
        if lab.lane is not None:
            y = top - 6 - lab.lane * 15
            x = lab.x0 if lab.anchor == "start" else lab.x1
            out.append(
                f'<line class="pole" x1="{_f(lab.at)}" y1="{_f(y - 10)}" x2="{_f(lab.at)}" y2="{_f(y + 3)}"/>'
            )
            out.append(
                f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{lab.anchor}" font-size="{FONT - 1}">'
                f"{_e(lab.text)}</text>"
            )
    out.append(
        f'<line class="axis" x1="{_f(left)}" y1="{_f(bottom)}" x2="{_f(WIDTH - right)}" y2="{_f(bottom)}"/>'
    )
    out += _date_ticks(rows, xs, bottom, left, WIDTH - right)
    out.append("</g></svg>")
    keys = [("obs", "observed score")]
    if any(not r.used for r in rows):
        keys.append(("off", "not in the model"))
    if known:
        keys.append(("fit", "fitted by the model"))
    if half and known:
        keys.append(("band", f"± {_pct(fit.confidence)} noise band (t·σ, σ = {sig(noise_sd(fit))})"))
    if labels:
        keys.append(("chg", "ideas changed (hover for the full list)"))
    return _figure("".join(out), keys)


def _change_labels(
    rows: Sequence[Row], bounds: Sequence[float], moved: Sequence[Sequence[str]]
) -> list[_Label]:
    """A label at the boundary before each submission whose ideas changed (`moved`, one list per row, as
    `effects.all_changes` gives them). Left to right, each goes in the first lane where it overlaps no
    other label, cut shorter if it fits nowhere whole; a label with no room at all keeps only its tick
    (and its tooltip)."""
    labels: list[_Label] = []
    lanes: list[list[tuple[float, float]]] = [[] for _ in range(LANES)]
    for i in range(1, len(rows)):
        if not moved[i]:
            continue
        lab = _Label(0, 0, bounds[i], "start", "", f"{rows[i].name}: {' '.join(moved[i])}")
        labels.append(lab)
        short = util.clip(moved[i][0], 14) + (" …" if len(moved[i]) > 1 else "")
        for text in dict.fromkeys([_cap([util.clip(m, LABEL_COLS) for m in moved[i]]), short]):
            w = _width(text, FONT - 1)
            start = lab.at + 3 if lab.at + 3 + w <= WIDTH - 2 else lab.at - 3 - w  # else left of the tick
            free = [k for k in range(LANES) if all(start > b + 8 or start + w < a - 8 for a, b in lanes[k])]
            if free:
                lab.x0, lab.x1, lab.text, lab.lane = start, start + w, text, free[0]
                lab.anchor = "start" if start > lab.at else "end"
                lanes[free[0]].append((start, start + w))
                break
    return labels


def _cap(moved: Sequence[str]) -> str:
    """As many changes as fit in about LABEL_COLS columns, then "and N more"."""
    shown: list[str] = []
    for m in moved:
        if shown and util.display_width(" ".join([*shown, m])) > LABEL_COLS:
            break
        shown.append(m)
    rest = len(moved) - len(shown)
    return " ".join(shown) + (f" and {rest} more" if rest else "")


def _sub_title(fit: Fit, r: Row, moved: Sequence[str]) -> str:
    bits = [r.name, util.local_day(r.date), r.nb_label]
    lines = [" · ".join(b for b in bits if b), f"{fit.target} {num(r.y)}"]
    if r.fitted is not None:
        lines[-1] += f", fitted {num(r.fitted)}"
    if moved:
        lines.append("changed: " + " ".join(moved))
    if not r.used:
        lines.append(f"not used in the model: {r.why_not}")
    return "\n".join(lines)


def _runs(values: Sequence[float | None]) -> list[list[int]]:
    """Index runs of consecutive values that are not None."""
    out: list[list[int]] = []
    for i, v in enumerate(values):
        if v is None:
            continue
        if out and out[-1][-1] == i - 1:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _date_ticks(rows: Sequence[Row], xs: Sequence[float], y: float, left: float, right: float) -> list[str]:
    """A few day labels under the plot, far enough apart not to overlap; a repeated day is not labelled."""
    room = _width("2026-01-01", FONT - 1) + 24
    every = max(1, math.ceil(len(rows) / max(1, int((right - left) // room))))
    out, last = ['<g class="tick">'], ""
    for i in range(0, len(rows), every):
        day = util.local_day(rows[i].date)
        if not day or day == last:
            continue
        last = day
        w = _width(day, FONT - 1)
        x, anchor = xs[i], "middle"
        if x - w / 2 < 2:
            x, anchor = 2, "start"
        elif x + w / 2 > WIDTH - 2:
            x, anchor = WIDTH - 2, "end"
        out.append(f'<line class="axis" x1="{_f(xs[i])}" y1="{_f(y)}" x2="{_f(xs[i])}" y2="{_f(y + 4)}"/>')
        out.append(f'<text x="{_f(x)}" y="{_f(y + 17)}" text-anchor="{anchor}">{_e(day)}</text>')
    out.append("</g>")
    return out


def _y_grid(
    ticks: Sequence[float], texts: Sequence[str], sy: Callable[[float], float], x0: float, x1: float
) -> list[str]:
    out = ['<g class="tick">']
    for v, text in zip(ticks, texts, strict=True):
        y = sy(v)
        out.append(f'<line class="grid" x1="{_f(x0)}" y1="{_f(y)}" x2="{_f(x1)}" y2="{_f(y)}"/>')
        out.append(f'<text x="{_f(x0 - 6)}" y="{_f(y + 4)}" text-anchor="end">{_e(text)}</text>')
    out.append("</g>")
    return out


# ----- forest --------------------------------------------------------------------------------------


def forest(fit: Fit, what: str) -> str:
    """One row per measured term (`fit.measured()`, best first): its label, the estimate as a dot and the
    interval as a line, coloured by verdict, around a zero line, with the numbers at the right. `what` names
    the rows ("submissions", "runs"). An interval far wider than the others runs off the axis with an arrow.
    Returns "" when nothing is measured."""
    terms = [t for t in fit.measured() if _finite(t.estimate)]
    if not terms:
        return ""
    dec = interval_decimals(terms)
    texts = [term_text(t, dec) for t in terms]
    text_w = max(_width(s) for s in texts) + 10
    verdict_w = max(_width(t.verdict) for t in terms) + 6
    x1 = WIDTH - text_w - verdict_w - 18
    labels, label_w = _row_labels([t.label for t in terms], x1)
    x0 = label_w + 10
    span = _forest_span(terms)
    lo, hi, ticks, step = _axis(-span, span, 4)
    sx = _scale(lo, hi, x0, x1)
    clamp = _clamp(lo, hi)
    top = 6.0
    bottom = top + ROW * len(terms)
    height = bottom + 40
    conf = _pct(fit.confidence)
    out = [_open(height, f"What each idea adds to the {fit.target}, with {conf} intervals")]
    for i, (t, text) in enumerate(zip(terms, texts, strict=True)):
        out.append(
            f"<g><title>{_e(_term_title(t, text, conf, what))}</title>"
            f'<rect class="hit" x="0" y="{_f(top + i * ROW)}" width="{WIDTH}" height="{ROW}"/></g>'
        )
    out.append('<g class="marks"><g class="tick">')
    for v, text in zip(ticks, _tick_texts(ticks, step, signed=True), strict=True):
        x = sx(v)
        cls = "zero" if abs(v) < step * 1e-9 else "grid"
        out.append(f'<line class="{cls}" x1="{_f(x)}" y1="{_f(top)}" x2="{_f(x)}" y2="{_f(bottom)}"/>')
        out.append(f'<text x="{_f(x)}" y="{_f(bottom + 15)}" text-anchor="middle">{_e(text)}</text>')
    out.append(
        f'<text x="{_f((x0 + x1) / 2)}" y="{_f(bottom + 32)}" text-anchor="middle">'
        f"{_e(f'effect on the {fit.target} ({conf} interval)')}</text></g>"
    )
    for i, (t, text, label) in enumerate(zip(terms, texts, labels, strict=True)):
        y = top + i * ROW + ROW / 2
        cls = _verdict_class(t.verdict)
        out.append(f'<text x="4" y="{_f(y + 4)}">{_e(label)}</text>')
        if _finite(t.ci_low) and _finite(t.ci_high):
            a, b = clamp(float(t.ci_low or 0)), clamp(float(t.ci_high or 0))
            out.append(
                f'<line class="ci {cls}" x1="{_f(sx(a))}" y1="{_f(y)}" x2="{_f(sx(b))}" y2="{_f(y)}"/>'
            )
            if a > float(t.ci_low or 0):
                out.append(_arrow(sx(lo), y, -1, cls))
            if b < float(t.ci_high or 0):
                out.append(_arrow(sx(hi), y, 1, cls))
        out.append(
            f'<circle class="dot {cls}" cx="{_f(sx(clamp(float(t.estimate or 0))))}" cy="{_f(y)}" r="4.5"/>'
        )
        out.append(f'<text x="{_f(x1 + 14)}" y="{_f(y + 4)}">{_e(text)}</text>')
        strong = ' font-weight="600"' if cls != NEUTRAL else ' class="m"'
        out.append(f'<text{strong} x="{_f(WIDTH - verdict_w)}" y="{_f(y + 4)}">{_e(t.verdict)}</text>')
    out.append("</g></svg>")
    shown = {_verdict_class(t.verdict) for t in terms}
    keys = [(GOOD, "helps"), (BAD, "hurts"), (NEUTRAL, "within noise, or too little data")]
    return _figure("".join(out), [(f"dot {cls}", text) for cls, text in keys if cls in shown])


def _forest_span(terms: Sequence[Term]) -> float:
    """Half the axis width: every estimate fits, and every interval, unless one is far wider than the next
    widest (it then runs off the axis instead of squeezing the others)."""
    est = max(abs(float(t.estimate or 0)) for t in terms)
    reach = sorted(
        (max(abs(v) for v in (t.estimate, t.ci_low, t.ci_high) if _finite(v)) for t in terms), reverse=True
    )
    span = reach[0]
    if len(reach) > 1 and reach[0] > 1.6 * reach[1]:
        span = 1.25 * reach[1]
    return max(span, est * 1.08, 1e-12)


def _term_title(t: Term, text: str, conf: str, what: str) -> str:
    from pseudorca.effects import DIFFERENCE

    interval = f" ({conf} interval)" if _finite(t.ci_low) and _finite(t.ci_high) else ""
    lines = [t.label, f"{text}{interval}: {t.verdict}"]
    if t.kind == DIFFERENCE:
        left, right = " + ".join(t.left), " + ".join(t.right)
        lines.append(f"a difference: what {left} adds minus what {right} adds")
        lines.append(f"{what} with {left}/with {right}: {t.n_with}/{t.n_without}")
    else:
        lines.append(f"{what} with/without it: {t.n_with}/{t.n_without}")
    if t.note:
        lines.append(t.note)
    return "\n".join(lines)


def _verdict_class(verdict: str) -> str:
    from pseudorca.effects import HELPS, HURTS

    return {HELPS: GOOD, HURTS: BAD}.get(verdict, NEUTRAL)


def _arrow(x: float, y: float, way: int, cls: str) -> str:
    """A small triangle at the end of the axis: the interval goes on past it."""
    tip = x + way * 7
    return f'<path class="{cls}" d="M{_f(tip)} {_f(y)} L{_f(x)} {_f(y - 4)} L{_f(x)} {_f(y + 4)} Z"/>'


# ----- waterfall -----------------------------------------------------------------------------------


@dataclass
class _Bar:
    label: str
    start: float | None  # None: from the baseline
    end: float
    cls: str
    title: str
    text: str
    whisker: tuple[float, float] | None = None
    observed: float | None = None


def waterfall(fit: Fit, w: Waterfall) -> str:
    """From the base's fitted score to the target's, one floating bar per step (what an idea, or ideas
    that cannot be separated, add), each with its interval as a whisker; the two fitted scores as bars and
    the observed scores as dots."""
    dec = interval_decimals(w.steps, fallback=[w.start, w.end])  # the steps' intervals
    scores, fitted = util.score_columns([r.y for r in fit.rows], [r.fitted for r in fit.rows])
    head, deltas, tail = util.running_column(
        w.start, [s.delta for s in w.steps], w.end, max(dec, fitted)
    )  # as `pseudorca effects` prints them: the column adds up
    used, best = fit.used, fit.best()
    conf = _pct(fit.confidence)
    first = " (first)" if used and w.base.key == used[0].key else ""
    top_one = " (best in the model)" if best is not None and best.key == w.target.key else ""
    bars = [
        _Bar(
            f"{w.base.name} fitted{first}",
            None,
            w.start,
            "total",
            f"{w.base.name} ({util.local_day(w.base.date)}, {w.base.nb_label}): fitted "
            f"{head}, observed {num(w.base.y)}",
            f"{head} · observed {plain(w.base.y, scores)}",
            observed=w.base.y,
        )
    ]
    level = w.start
    for s, shown in zip(w.steps, deltas, strict=True):
        whisker = (level + s.ci_low, level + s.ci_high) if _finite(s.ci_low) and _finite(s.ci_high) else None
        ci = f" [{signed(s.ci_low, dec)}, {signed(s.ci_high, dec)}]" if whisker else ""
        if s.delta == 0:  # it adds exactly nothing
            cls = ZERO
        elif s.sign == 0:  # a mix that cannot be split
            cls = NEUTRAL
        else:
            cls = GOOD if (s.delta > 0) == fit.higher_is_better else BAD
        title = f"{s.label}: {shown}" + (f", {conf} interval{ci}" if ci else "")
        bars.append(_Bar(s.label, level, level + s.delta, cls, title, shown + ci, whisker))
        level += s.delta
    bars.append(
        _Bar(
            f"= {w.target.name} fitted{top_one}",
            None,
            w.end,
            "total",
            f"{w.target.name} ({util.local_day(w.target.date)}, {w.target.nb_label}): fitted "
            f"{tail}, observed {num(w.target.y)}",
            f"{tail} · observed {plain(w.target.y, scores)}",
            observed=w.target.y,
        )
    )
    values = [b.end for b in bars] + [v for b in bars for v in (b.whisker or ())]
    values += [float(b.observed) for b in bars if b.observed is not None and _finite(b.observed)]
    lo, hi = min(values), max(values)
    span = hi - lo
    if lo > 0:  # start the fitted-score bars at zero when it is near, else leave them room to show
        lo = 0.0 if lo < span else max(0.0, lo - 0.3 * span)
    elif hi < 0:
        hi = 0.0 if -hi < span else min(0.0, hi + 0.3 * span)
    lo, hi, ticks, step = _axis(lo, hi, 4)
    base = 0.0 if lo <= 0 <= hi else (lo if lo > 0 else hi)
    cut = base != 0.0  # the fitted-score bars start at the end of the axis, not at zero
    texts = [b.text for b in bars]
    x1 = WIDTH - max(_width(s) for s in texts) - 22
    labels, label_w = _row_labels([b.label for b in bars], x1)
    x0 = label_w + 10
    sx = _scale(lo, hi, x0, x1)
    top = 6.0
    bottom = top + ROW * len(bars)
    height = bottom + 24
    out = [_open(height, f"Breakdown of {w.target.name} from {w.base.name}, by the model")]
    for i, b in enumerate(bars):
        out.append(
            f"<g><title>{_e(b.title)}</title>"
            f'<rect class="hit" x="0" y="{_f(top + i * ROW)}" width="{WIDTH}" height="{ROW}"/></g>'
        )
    out.append('<g class="marks"><g class="tick">')
    for v, text in zip(ticks, _tick_texts(ticks, step), strict=True):
        x = sx(v)
        out.append(f'<line class="grid" x1="{_f(x)}" y1="{_f(top)}" x2="{_f(x)}" y2="{_f(bottom)}"/>')
        out.append(f'<text x="{_f(x)}" y="{_f(bottom + 15)}" text-anchor="middle">{_e(text)}</text>')
    out.append("</g>")
    for i, (b, label) in enumerate(zip(bars, labels, strict=True)):
        y = top + i * ROW + ROW / 2
        a = base if b.start is None else b.start
        left, right = sorted((sx(a), sx(b.end)))
        out.append(f'<text x="4" y="{_f(y + 4)}">{_e(label)}</text>')
        out.append(
            f'<rect class="{b.cls}" x="{_f(left)}" y="{_f(y - BAR / 2)}" '
            f'width="{_f(max(right - left, 1.5))}" height="{BAR}" rx="2"/>'
        )
        if b.start is None and cut and right - left > 40:  # the bar is cut where the axis starts
            x = left + 9 if base == lo else right - 9
            out.append(
                f'<path class="cut" d="M{_f(x - 4)} {_f(y + BAR / 2 + 1)} L{_f(x)} {_f(y - BAR / 2 - 1)} '
                f'M{_f(x + 1)} {_f(y + BAR / 2 + 1)} L{_f(x + 5)} {_f(y - BAR / 2 - 1)}"/>'
            )
        if i + 1 < len(bars):  # a connector from this bar's end to the next row
            x = sx(b.end)
            out.append(
                f'<line class="link" x1="{_f(x)}" y1="{_f(y + BAR / 2)}" x2="{_f(x)}" '
                f'y2="{_f(y + ROW - BAR / 2)}"/>'
            )
        if b.whisker is not None:
            wa, wb = (sx(v) for v in b.whisker)
            out.append(
                f'<path class="wk" d="M{_f(wa)} {_f(y)} H{_f(wb)} M{_f(wa)} {_f(y - 4)} V{_f(y + 4)} '
                f'M{_f(wb)} {_f(y - 4)} V{_f(y + 4)}"/>'
            )
        if _finite(b.observed):
            out.append(f'<circle class="obs" cx="{_f(sx(float(b.observed or 0)))}" cy="{_f(y)}" r="4"/>')
        out.append(f'<text x="{_f(x1 + 12)}" y="{_f(y + 4)}">{_e(b.text)}</text>')
    out.append("</g></svg>")
    keys = [("bar total", "fitted score")]
    if fit.higher_is_better:
        names = ((GOOD, "adds"), (BAD, "takes away"))
    else:
        names = ((GOOD, "lowers it (better)"), (BAD, "raises it (worse)"))
    for cls, text in (*names, (NEUTRAL, "cannot be separated"), (ZERO, "adds nothing")):
        if any(b.cls == cls for b in bars):
            keys.append((f"bar {cls}", text))
    if any(b.whisker for b in bars):
        keys.append(("wk", f"{conf} interval"))
    keys.append(("obs", "observed score"))
    if cut:
        keys.append(("cut", "the axis does not start at 0"))
    return _figure("".join(out), keys)


# ----- helpers -------------------------------------------------------------------------------------


def _figure(svg: str, keys: Sequence[tuple[str, str]]) -> str:
    """The chart in a horizontally scrolling box, then its legend."""
    legend = "".join(f'<span class="key {k}"></span>{_e(text)}' for k, text in keys)
    return f'<div class="scroll">{svg}</div>' + (f'<div class="legend">{legend}</div>' if keys else "")


def _open(height: float, label: str) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {WIDTH} {_f(height)}" width="100%" role="img" '
        f'aria-label="{_e(label)}">'
    )


def _row_labels(labels: Sequence[str], x1: float) -> tuple[list[str], float]:
    """Row labels for a forest or waterfall whose plot ends at x1, and the width of their column: cut to
    44 columns, and shorter when they would leave the plot narrower than MIN_PLOT."""
    out = [util.clip(s, 44) for s in labels]
    width = min(330.0, max(120.0, max(_width(s) for s in out) + 16))
    room = max(120.0, x1 - 10 - MIN_PLOT)
    if width > room:
        out = [util.clip(s, max(8, int((room - 16) / (FONT * 0.6)))) for s in out]
        width = room
    return out, width


def _axis(lo: float, hi: float, count: int, pad: float = 0.05) -> tuple[float, float, list[float], float]:
    """An axis a little wider than [lo, hi] (by `pad` of its span on each side), about `count` round ticks
    inside it, and the tick step. Values that are not finite or beyond LIMIT (hand-edited scores near the
    float limits) get an axis of 0..1: the chart is then wrong, but it is drawn."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or max(abs(lo), abs(hi)) > LIMIT:
        lo, hi = 0.0, 1.0
    if hi - lo <= 1e-12 * max(1.0, abs(lo), abs(hi)):
        half = abs(lo) * 0.05 or 1.0
        lo, hi = lo - half, hi + half
    raw = (hi - lo) / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5) if m * mag >= raw * (1 - 1e-9)), 10 * mag)
    lo, hi = lo - pad * (hi - lo), hi + pad * (hi - lo)
    first, last = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    return lo, hi, [k * step for k in range(first, last + 1)], step


def _tick_texts(ticks: Sequence[float], step: float, signed: bool = False) -> list[str]:
    """The ticks' labels, down to the step's last digit: as decimals; or, when that needs more than 6 of
    them or a tick reaches 1e6, as significant digits."""
    top = max((abs(v) for v in ticks), default=0.0) or step
    last = math.floor(math.log10(step))  # the place of the step's last digit (a step of 2.5 has two)
    if abs(step / 10.0**last - round(step / 10.0**last)) > 1e-6:
        last -= 1
    if top < util.COMPACT and last >= -6:
        spec = f".{max(0, -last)}f"
    else:
        spec = f".{min(17, max(1, math.floor(math.log10(top)) - last + 1))}g"
    out = []
    for v in ticks:
        text = format(abs(v), spec)
        if float(text):
            text = (MINUS if v < 0 else "+" if signed else "") + text
        out.append(text)
    return out


def _scale(d0: float, d1: float, r0: float, r1: float) -> Callable[[float], float]:
    """[d0, d1] onto [r0, r1]. A value beyond the axis is held at its end: only an axis that could not
    cover the values has such values (see _axis)."""
    k = (r1 - r0) / (d1 - d0) if d1 != d0 else 0.0
    hold = _clamp(r0, r1)
    return lambda v: hold(r0 + (v - d0) * k)


def _clamp(a: float, b: float) -> Callable[[float], float]:
    lo, hi = min(a, b), max(a, b)
    return lambda v: min(hi, max(lo, v))


def _poly(points: Sequence[tuple[float, float]], close: bool = True) -> str:
    """Path data through the points; vertical and horizontal moves where they line up."""
    out = [f"M{_f(points[0][0])} {_f(points[0][1])}"]
    for (px, py), (x, y) in itertools.pairwise(points):
        if abs(y - py) < 1e-9:
            out.append(f"H{_f(x)}")
        elif abs(x - px) < 1e-9:
            out.append(f"V{_f(y)}")
        else:
            out.append(f"L{_f(x)} {_f(y)}")
    return " ".join(out) + (" Z" if close else "")


def _width(text: str, size: float = FONT) -> float:
    """About how wide `text` is at this size (a wide character counts twice)."""
    return util.display_width(text) * size * 0.6


def _pct(confidence: float) -> str:
    return f"{confidence * 100:g}%"


def sig(x: float | None) -> str:
    """Three significant digits, as the terminal shows a noise SD."""
    return f"{x:.3g}" if _finite(x) else "n/a"


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _f(x: float) -> str:
    """A coordinate. One that is not finite, or far off the chart (from values near the float limits), is
    pulled back, so that the SVG stays valid."""
    return f"{min(1e5, max(-1e5, x)) if math.isfinite(x) else 0.0:.1f}"


# characters a page should not show as they are: control characters (but tab and newline, which tooltips
# use) and bidirectional overrides, which could make ledger text read in another order than it is
_HIDDEN_RE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def escape(value: Any) -> str:
    """Ledger text for HTML: escaped, with the characters of _HIDDEN_RE as "?"."""
    return html.escape(_HIDDEN_RE.sub("?", "" if value is None else str(value)))


_e = escape
