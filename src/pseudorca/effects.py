"""Which ideas each submission has, and what each idea adds: an additive model over submissions or runs.

    score = baseline + (the effects of the ideas it has) + noise

Rows are scored submissions, whose notebooks' verdicts say which ideas they have, or the local runs of one
protocol, paired by item. An idea is in the model when every row has a present or absent verdict for it
and some rows have it but not all. Ideas that are always in or out together are measured together, and
what the rows cannot separate is reported as such, never guessed.

- Leaderboard: least squares over the submissions. The noise is the residual SD (resubmitting a notebook
  makes it reliable), or a known SD; intervals use Student's t.
- Local runs: least squares over each run's mean on the items every run has. Intervals come from a
  two-stage bootstrap (items, then draws within items) that keeps the runs paired, as in `compare`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pseudorca import util

if TYPE_CHECKING:
    import numpy as np

    from pseudorca.model import Run, State, Submission
    from pseudorca.results import Table
    from pseudorca.stats import LbNoise

PRESENT, ABSENT = "present", "absent"
MINUS = "−"

# what a Term is
EFFECT = "effect"  # one idea, measured
BUNDLE = "bundle"  # ideas that are always in or out together, measured together
DIFFERENCE = "difference"  # one idea (or bundle) instead of another: only the difference is measurable
TANGLED = "tangled"  # cannot be separated from other ideas at all
ALWAYS = "always"  # in every row: nothing to compare with
NEVER = "never"  # in no row
EXCLUDED = "excluded"  # some rows have no present/absent verdict for it, so it is left out of the model
MEASURED = (EFFECT, BUNDLE, DIFFERENCE)

HELPS, HURTS, WITHIN_NOISE, UNKNOWN = "helps", "hurts", "within noise", "unknown"
TOO_FEW = "too few items"
MIN_ROWS = 2
MAX_SCORE = 1e150  # larger scores would overflow the sums of squares
NO_NOISE = "no noise estimate: the model has as many unknowns as submissions"
ZERO_RESIDUALS = "no noise estimate: the residuals are 0"
TOL = 1e-8  # estimability and support tolerance on a 0/1 design
SNAP = 1e3 * 2.220446049250313e-16  # relative rounding: a value this small against its inputs is 0
POWER_Z = 0.8416212335729143  # z(0.80), for the minimum detectable effect at 80% power


@dataclass
class Cell:
    """An idea's verdict on a row's notebook."""

    status: str | None  # present | partial | absent | unknown | conflict, or None: no verdict
    stale: bool = False
    method: str = ""  # the method behind the verdict ("human", "fingerprint+patch", ...)

    @property
    def value(self) -> int | None:
        """1 present, 0 absent, None for anything else."""
        return 1 if self.status == PRESENT else 0 if self.status == ABSENT else None


@dataclass
class Row:
    """A submission or a run, with the verdicts of its notebook."""

    key: str  # submission id or run id
    name: str  # how to show it
    nb: str | None
    nb_label: str
    date: str
    y: float | None  # its score: the LB score, or the run's mean over the items every run has
    cells: dict[str, Cell]
    why_not: str = ""  # why it is not in the model; empty when it is
    fitted: float | None = None

    @property
    def used(self) -> bool:
        return not self.why_not


@dataclass
class Term:
    """One line of the effects table."""

    ideas: list[str]  # the idea, the bundle, or for a difference the ideas of `left` then `right`
    kind: str
    estimate: float | None = None
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_with: int = 0  # rows that have it (a difference: rows that have `left`)
    n_without: int = 0  # rows that do not (a difference: rows that have `right`)
    verdict: str = ""  # helps | hurts | within noise | too few items | unknown; empty when not measured
    note: str = ""
    left: list[str] = field(default_factory=list)  # a difference: `left` instead of (or vs) `right`
    right: list[str] = field(default_factory=list)
    swapped: bool = True  # a difference: no row has both; else they are also used together ("vs")
    mde80: float | None = None  # the smallest effect its test finds 80% of the time (see _mde)

    @property
    def label(self) -> str:
        if self.kind == DIFFERENCE:
            joiner = "instead of" if self.swapped else "vs"
            return f"{' + '.join(self.left)} {joiner} {' + '.join(self.right)}"
        return " + ".join(self.ideas)


@dataclass
class Fit:
    """The model over one set of rows: the leaderboard, or the local runs of one protocol."""

    source: str  # "lb" | "local"
    target: str  # "public score", "private score", or a metric name
    rows: list[Row]
    terms: list[Term]
    confidence: float
    higher_is_better: bool
    protocol: str | None = None  # local runs
    sigma: float | None = None  # LB: the noise SD used for the intervals
    df: float | None = None  # LB: its degrees of freedom (inf when given)
    noise_from: str = ""  # "residuals" | "given" | "bootstrap" | "" (no estimate)
    resubmission: LbNoise | None = None  # LB: the pooled SD of resubmitted notebooks, for comparison
    n_items: int = 0  # local: the items every run has
    n_items_any: int = 0  # local: the items any run has
    resamples: int = 0
    seed: int = 0
    lof_p: float | None = None  # p-value of the lack-of-fit test (residuals beyond the noise), if any
    tau: float | None = None  # local: the run-to-run SD added to the bootstrap (None: not measurable)
    tau_df: int = 0  # local: its degrees of freedom (runs with the same ideas, less one per set of them)
    notes: list[str] = field(default_factory=list)
    _model: _Model | None = field(default=None, repr=False)

    @property
    def used(self) -> list[Row]:
        return [r for r in self.rows if r.used]

    def measured(self) -> list[Term]:
        return [t for t in self.terms if t.kind in MEASURED]

    def best(self) -> Row | None:
        """The used row with the best score."""
        rows = [r for r in self.used if r.y is not None]
        if not rows:
            return None
        return (max if self.higher_is_better else min)(rows, key=lambda r: r.y)


@dataclass
class Step:
    """One bar of a waterfall: the ideas that differ, and what they add."""

    label: str
    ideas: list[str]
    sign: int  # +1 added, −1 removed, 0 a mix that cannot be separated
    delta: float
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass
class Waterfall:
    """How the model gets from one row's fitted score to another's."""

    base: Row
    target: Row
    start: float  # the base's fitted score
    end: float  # the target's fitted score
    steps: list[Step]
    unmodelled: list[str] = field(default_factory=list)  # ideas outside the model that differ (or may)


# ----- rows ----------------------------------------------------------------------------------------


def sub_order(s: Submission) -> tuple[Any, int, str]:
    """Submissions by date (as instants), then by numeric id."""
    numeric = s.id.isascii() and s.id.isdigit() and len(s.id) <= 30  # int() refuses thousands of digits
    return (util.instant(s.date or s.updated), int(s.id) if numeric else -1, s.id)


def _cells(state: State, nb_id: str | None, idea_ids: Sequence[str]) -> dict[str, Cell]:
    out: dict[str, Cell] = {}
    for iid in idea_ids:
        eff = state.effective_verdict(nb_id, iid) if nb_id else None
        out[iid] = Cell(eff.status, eff.stale, eff.method) if eff is not None else Cell(None)
    return out


def idea_ids(state: State, include_parked: bool = False) -> list[str]:
    """The ideas the rows carry, in ledger order: parked ones only when asked (as on the board)."""
    return [i.id for i in state.ideas_in_order() if include_parked or i.status != "parked"]


def submission_rows(
    state: State,
    *,
    private: bool = False,
    competition: str | None = None,
    include_parked: bool = False,
    also: Sequence[str] = (),
) -> list[Row]:
    """Every submission, oldest first, with its score and its notebook's verdicts.

    With several competitions in the ledger, `competition` must say which one (submissions without one
    are kept)."""
    from pseudorca.stats import finite

    subs = sorted(state.subs.values(), key=sub_order)
    names = sorted({s.competition for s in subs if s.competition})
    if competition:
        if names and competition not in names:
            raise LookupError(
                f"no submission is for competition {competition!r} (competitions: {', '.join(names)})"
            )
        subs = [s for s in subs if s.competition in ("", competition)]
    elif len(names) > 1:
        raise ValueError(
            f"the submissions are from {len(names)} competitions ({', '.join(names)}); choose one with "
            "--competition"
        )
    ideas = list(dict.fromkeys([*idea_ids(state, include_parked), *also]))  # `also`: ideas asked for by name
    board = "private" if private else "public"
    rows = []
    for s in subs:
        score = finite(s.private_score if private else s.public_score)
        nb_id = s.nb if isinstance(s.nb, str) and s.nb else None
        nb = state.notebooks.get(nb_id) if nb_id else None
        if score is None:
            why = f"no {board} score"
        elif nb_id is None:
            why = "no notebook (link one with `pseudorca sub link`)"
        elif nb is None:
            why = "its notebook is not registered"
        else:
            why = ""
        rows.append(
            Row(
                key=s.id,
                name=s.id,
                nb=nb_id,
                nb_label=nb.label if nb else util.short(nb_id),
                date=s.date,
                y=score,
                cells=_cells(state, nb.id if nb else None, ideas),
                why_not=why,
            )
        )
    return rows


def changes(prev: Row | None, row: Row) -> list[str]:
    """What changed from the previous row: "+idea" for an idea that came in, "−idea" for one that left.
    Changes to or from a verdict other than present/absent are not listed."""
    if prev is None:
        return []
    out = []
    for iid, cell in row.cells.items():
        before = prev.cells.get(iid)
        a, b = (before.value if before else None), cell.value
        if a == 0 and b == 1:
            out.append("+" + iid)
        elif a == 1 and b == 0:
            out.append(MINUS + iid)
    return out


def all_changes(rows: Sequence[Row]) -> list[list[str]]:
    """For each row, the ideas that came in ("+idea") or left ("−idea") since the latest earlier row with a
    present/absent verdict for them, so that a row without verdicts in between hides nothing."""
    last: dict[str, int] = {}
    out = []
    for row in rows:
        moved = []
        for iid, cell in row.cells.items():
            value = cell.value
            if value is None:
                continue
            before = last.get(iid)
            if before == 0 and value == 1:
                moved.append("+" + iid)
            elif before == 1 and value == 0:
                moved.append(MINUS + iid)
            last[iid] = value
        out.append(moved)
    return out


def resubmits(rows: Sequence[Row]) -> list[str | None]:
    """For each row, the earliest earlier row with the same notebook (its name), else None."""
    first: dict[str, str] = {}
    out: list[str | None] = []
    for row in rows:
        out.append(first.get(row.nb) if row.nb else None)
        if row.nb:
            first.setdefault(row.nb, row.name)
    return out


def shown_ideas(rows: Sequence[Row]) -> list[str]:
    """The ideas with a verdict on at least one of the rows' notebooks, in ledger order."""
    if not rows:
        return []
    return [iid for iid in rows[0].cells if any(r.cells[iid].status is not None for r in rows)]


# ----- the model -----------------------------------------------------------------------------------


class _Model:
    """Least squares on [1 | bundle columns], with what can and cannot be estimated from those rows."""

    def __init__(self, bundles: list[list[str]], z: np.ndarray, y: np.ndarray) -> None:
        import numpy as np

        n = len(y)
        x = np.column_stack([np.ones(n), z]) if z.size else np.ones((n, 1))
        # QR first, so that the SVD is of a small matrix: never an n × n one, however many rows
        q, r = np.linalg.qr(x)
        u_r, s, vt = np.linalg.svd(r, full_matrices=True)
        tol = (float(s.max()) if s.size else 0.0) * max(x.shape) * np.finfo(float).eps
        rank = int((s > tol).sum())
        u = q @ u_r
        self.bundles = bundles
        self.x, self.y, self.rank = x, y, rank
        self.null = vt[rank:].T  # columns span the coefficients the rows cannot see
        # Contrasts between bundles never involve the intercept: what they cannot see is the null space's
        # bundle part (independent columns: a null vector is never the intercept alone), made orthonormal
        part = self.null[1:, :]
        self.bundle_null = np.linalg.qr(part)[0] if part.size else part
        self.pinv = (vt[:rank].T / s[:rank]) @ u[:, :rank].T
        self.basis = u[
            :, :rank
        ]  # orthonormal columns spanning X's: the fitted values of v are basis·basisᵀ·v
        self.coef = self.pinv @ y
        self.fitted = x @ self.coef
        residuals = y - self.fitted
        rounding = SNAP * (np.abs(y) + np.abs(self.fitted))  # what rounding alone leaves of each residual
        self.rss = float((residuals**2).sum())
        if self.rss <= float((rounding**2).sum()):
            self.rss = 0.0  # an exact fit: the residuals are rounding
        self.df = n - rank
        self.interval: Callable[[np.ndarray], tuple[float, float | None, float | None, float | None]]
        self.interval = lambda w: (self.snap(w, float(w @ y)), None, None, None)
        self.p_value: Callable[[np.ndarray], float] | None = None  # local runs: the sign-flip test of w·y
        self.mde_quantiles: tuple[float, float] | None = None  # (two-sided level, 80% power) quantiles

    def snap(self, w: np.ndarray, value: float) -> float:
        """`value` (an estimate w·y, or a bound) as 0 when it is within rounding of 0 for these weights."""
        return 0.0 if abs(value) <= SNAP * float(abs(w * self.y).sum()) else value

    def unit(self, j: int, sign: float = 1.0) -> np.ndarray:
        """The contrast vector of bundle j (0-based; column j + 1)."""
        import numpy as np

        c = np.zeros(self.x.shape[1])
        c[j + 1] = sign
        return c

    def estimable(self, c: np.ndarray) -> bool:
        if not self.null.size:
            return True
        return float(abs(self.null.T @ c).max()) <= TOL * max(1.0, float(abs(c).max()))

    def weights(self, c: np.ndarray) -> np.ndarray:
        """w with c·coef = w·y, for an estimable c."""
        return self.pinv.T @ c

    def entangled(self, j: int) -> set[int]:
        """The bundles whose coefficients mix with bundle j's in what the rows cannot see: the support of
        e_j's projection onto the bundle part of the null space. Through the intercept two unrelated
        choices (each one of a set, always) would look mixed; contrasts never use it, so it is left out."""
        if not self.bundle_null.size:
            return set()
        proj = self.bundle_null @ self.bundle_null[j]
        return {k for k in range(len(proj)) if abs(proj[k]) > TOL}

    def residuals(self, v: np.ndarray) -> np.ndarray:
        """v (rows, or resamples × rows) less its least-squares fit on X."""
        return v - (v @ self.basis) @ self.basis.T


def _choose(
    rows: list[Row], chosen: Sequence[str] | None, what: str, hint: str = ""
) -> tuple[list[str], list[Term]]:
    """The ideas to model, and terms for ideas left out because some rows have no verdict for them.

    By default an idea is modelled when every usable row has a present/absent verdict for it. With
    `chosen` ideas, those are modelled and rows without such a verdict for one of them are not used.
    `hint` says how to fill the gaps."""
    cand = [r for r in rows if r.used]
    if chosen:
        for r in cand:
            missing = [i for i in chosen if r.cells[i].value is None]
            if missing:
                shown = ", ".join(f"{i} ({r.cells[i].status or 'no verdict'})" for i in missing[:3])
                r.why_not = f"not present/absent: {shown}" + (", …" if len(missing) > 3 else "")
        return list(chosen), []
    if not cand:
        return [], []
    model, left_out = [], []
    for iid in cand[0].cells:
        values = [r.cells[iid].value for r in cand]
        if all(v is not None for v in values):
            model.append(iid)
            continue
        if all(r.cells[iid].status is None for r in cand):
            continue  # never audited on these notebooks: not shown at all
        gaps = [r.name for r, v in zip(cand, values, strict=True) if v is None]
        known = [v for v in values if v is not None]
        note = (
            f"no present/absent verdict on {len(gaps)} of {len(cand)} {what} "
            f"({', '.join(gaps[:4])}{', …' if len(gaps) > 4 else ''})"
        )
        if 0 in known and 1 in known:
            note += "; it varies among the others, so leaving it out may bias their effects"
        statuses = {r.cells[iid].status for r, v in zip(cand, values, strict=True) if v is None}
        if None in statuses and hint:
            note += f"; {hint}"  # no verdict yet: an audit gives one
        if statuses - {None}:  # partial, unknown or conflict: another audit says the same
            note += f"; `pseudorca verdict NB {iid} present|absent` settles the inconclusive ones"
        left_out.append(Term([iid], EXCLUDED, n_with=known.count(1), n_without=known.count(0), note=note))
    return model, left_out


def _fit(
    fit: Fit,
    chosen: Sequence[str] | None,
    what: str,
    make_interval: Callable[[_Model], Callable[[np.ndarray], tuple[Any, Any, Any, Any]] | None],
    verdict_floor: str = "",
    need: str = "",
    hint: str = "",
) -> None:
    """Choose the ideas, build the design, and fill fit.terms and the rows' fitted values."""
    import numpy as np

    model_ideas, terms = _choose(fit.rows, chosen, what, hint)
    used = fit.used
    if len(used) < MIN_ROWS:
        fit.terms = terms
        if chosen:
            need += ", and a present/absent verdict for each idea asked for"
        fit.notes.append(f"nothing to compare: it needs at least {MIN_ROWS} {what}{need}")
        return
    varying: list[str] = []
    for iid in model_ideas:
        values = [r.cells[iid].value for r in used]
        if all(v == 1 for v in values):
            terms.append(Term([iid], ALWAYS, n_with=len(used), note=f"in every one of the {what}"))
        elif all(v == 0 for v in values):
            terms.append(Term([iid], NEVER, n_without=len(used), note=f"in none of the {what}"))
        else:
            varying.append(iid)
    columns: dict[tuple[int, ...], list[str]] = {}
    for iid in varying:
        columns.setdefault(tuple(int(r.cells[iid].value or 0) for r in used), []).append(iid)
    bundles = list(columns.values())
    z = np.array(list(columns), dtype=float).T if columns else np.zeros((len(used), 0))
    y = np.array([r.y for r in used], dtype=float)
    if not (np.abs(y) < MAX_SCORE).all():
        raise ValueError(f"a score is too large to model (beyond {MAX_SCORE:g})")
    m = _Model(bundles, z, y)
    fit._model = m
    interval = make_interval(m)
    if interval is not None:
        m.interval = interval
    for r, value in zip(used, m.fitted, strict=True):
        r.fitted = float(value)

    def counts(j: int) -> tuple[int, int]:
        have = int(z[:, j].sum())
        return have, len(used) - have

    def term(kind: str, c: np.ndarray, ideas: list[str], **kw: Any) -> Term:
        w = m.weights(c)
        est, se, lo, hi = m.interval(w)
        if lo is not None and hi is not None:
            lo, hi = m.snap(w, lo), m.snap(w, hi)
        t = Term(ideas, kind, est, se, lo, hi, **kw)
        if se is not None and m.mde_quantiles is not None:
            t.mde80 = sum(m.mde_quantiles) * se
        t.verdict = verdict_floor or _verdict(lo, hi, fit.higher_is_better)
        if t.verdict in (HELPS, HURTS) and m.p_value is not None:
            # as in `compare`: with few items every bootstrap mean can have one sign, so a winner also
            # needs the sign-flip test
            p, alpha = m.p_value(w), 1 - fit.confidence
            if not p < alpha:
                t.verdict = WITHIN_NOISE
                why = f"the interval excludes 0, but the sign-flip p = {p:.3g} is not below {alpha:g}"
                t.note = f"{t.note}; {why}" if t.note else why
        return t

    measured: list[Term] = []
    stuck: list[int] = []
    for j, members in enumerate(bundles):
        c = m.unit(j)
        if m.estimable(c):
            have, lack = counts(j)
            note = "always in or out together: measured together" if len(members) > 1 else ""
            kind = BUNDLE if len(members) > 1 else EFFECT
            measured.append(term(kind, c, list(members), n_with=have, n_without=lack, note=note))
        else:
            stuck.append(j)
    for group in _groups(m, stuck):
        picked: list[np.ndarray] = []
        # the latest rows' choices first, so that a choice among several is set against each alternative
        newest = sorted(group, key=lambda j: -int(np.flatnonzero(z[:, j])[-1]))
        for a_pos, a in enumerate(newest):
            for b in newest[a_pos + 1 :]:
                together = bool(((z[:, a] > 0) & (z[:, b] > 0)).any())
                left, right = _later_first(z, a, b) if together else _newer_first(z, a, b)
                c = m.unit(left) - m.unit(right)
                if not m.estimable(c) or _dependent(picked, c):
                    continue
                picked.append(c)
                lhave, rhave = int(z[:, left].sum()), int(z[:, right].sum())
                note = (
                    "also used together: only the difference between their effects can be measured"
                    if together
                    else "neither can be measured alone; only the difference between them can"
                )
                measured.append(
                    term(
                        DIFFERENCE,
                        c,
                        [*bundles[left], *bundles[right]],
                        n_with=lhave,
                        n_without=rhave,
                        left=list(bundles[left]),
                        right=list(bundles[right]),
                        swapped=not together,
                        note=note,
                    )
                )
        covered = {k for c in picked for k in range(len(bundles)) if abs(c[k + 1]) > TOL}
        for j in group:
            if j in covered:
                continue
            others = [i for k in group if k != j for i in bundles[k]]
            have, lack = counts(j)
            note = "cannot be separated from " + (", ".join(others) if others else "the baseline")
            terms.append(Term(list(bundles[j]), TANGLED, n_with=have, n_without=lack, note=note))
    sign = 1 if fit.higher_is_better else -1
    measured.sort(key=lambda t: -sign * (t.estimate or 0.0))
    order = {EXCLUDED: 3, ALWAYS: 1, NEVER: 2, TANGLED: 0}
    fit.terms = measured + sorted(terms, key=lambda t: order.get(t.kind, 4))
    if chosen:
        fit.notes += _left_out_by_choice(used, model_ideas, what)
    stale = {(r.nb, iid): r.cells[iid].method for r in used for iid in model_ideas if r.cells[iid].stale}
    if stale:
        humans = sum(1 for method in stale.values() if method == "human")
        machine = len(stale) - humans
        note = (
            f"{len(stale)} of the verdicts used {'is' if len(stale) == 1 else 'are'} stale (made for an older "
            "version of the idea)"
        )
        if machine:
            note += "; `pseudorca audit NB` on those notebooks refreshes " + (
                "it" if machine == 1 else "them"
            )
            if what == "submissions":
                note += " (`pseudorca audit --submitted` does every submitted one)"
        if humans:
            note += (
                f"; {'1 is a human verdict' if humans == 1 else f'{humans} are human verdicts'}: record new "
                "ones with `pseudorca verdict`"
            )
        fit.notes.append(note)


def _left_out_by_choice(used: list[Row], chosen: list[str], what: str) -> list[str]:
    """With --idea: the other ideas that vary among the rows, whose effects the model cannot see."""
    columns = {iid: tuple(r.cells[iid].value for r in used) for iid in used[0].cells}
    twins, varying = [], []
    for iid, column in columns.items():
        if iid in chosen or None in column or len(set(column)) < 2:
            continue
        same = [c for c in chosen if columns.get(c) == column]
        opposite = [c for c in chosen if columns.get(c) == tuple(1 - v for v in column)]
        if same:
            twins.append(f"{iid} always comes and goes with {same[0]}")
        elif opposite:
            twins.append(f"{iid} is in exactly the rows without {opposite[0]} (counted against it)")
        else:
            varying.append(iid)
    notes = []
    if twins:
        notes.append(f"not modelled, so its effect is counted as theirs: {'; '.join(twins)}")
    if varying:
        notes.append(
            f"not modelled but varying among these {what}: {', '.join(varying)} (their effects go into the "
            "residuals, and may bias the ideas asked for)"
        )
    return notes


def _groups(m: _Model, stuck: list[int]) -> list[list[int]]:
    """The bundles the rows cannot see one by one, grouped by which ones mix with each other."""
    parent = {j: j for j in stuck}

    def find(j: int) -> int:
        while parent[j] != j:
            parent[j] = parent[parent[j]]
            j = parent[j]
        return j

    for j in stuck:
        for k in m.entangled(j):
            if k in parent:
                parent[find(k)] = find(j)
    groups: dict[int, list[int]] = {}
    for j in stuck:
        groups.setdefault(find(j), []).append(j)
    return list(groups.values())


def _later_first(z: np.ndarray, a: int, b: int) -> tuple[int, int]:
    """Bundles a and b, the one that first appeared later first (for two that are also used together)."""
    import numpy as np

    first_a, first_b = int(np.flatnonzero(z[:, a])[0]), int(np.flatnonzero(z[:, b])[0])
    return (a, b) if first_a > first_b else (b, a)


def _newer_first(z: np.ndarray, a: int, b: int) -> tuple[int, int]:
    """Bundles a and b, the newer choice first: the one in the latest row that has one but not the other."""
    for row in z[::-1]:
        if row[a] != row[b]:
            return (a, b) if row[a] > row[b] else (b, a)
    return a, b


def _dependent(picked: list[np.ndarray], c: np.ndarray) -> bool:
    import numpy as np

    if not picked:
        return False
    a = np.array(picked)
    return int(np.linalg.matrix_rank(np.vstack([a, c]), tol=TOL)) == int(np.linalg.matrix_rank(a, tol=TOL))


def _verdict(lo: float | None, hi: float | None, higher_is_better: bool) -> str:
    if lo is None or hi is None:
        return UNKNOWN
    if lo > 0:
        return HELPS if higher_is_better else HURTS
    if hi < 0:
        return HURTS if higher_is_better else HELPS
    return WITHIN_NOISE


# ----- leaderboard ---------------------------------------------------------------------------------


def lb_fit(
    state: State,
    *,
    private: bool = False,
    ideas: Sequence[str] | None = None,
    noise: float | None = None,
    confidence: float = 0.95,
    higher_is_better: bool = True,
    competition: str | None = None,
    include_parked: bool = False,
) -> Fit:
    """The additive model over the scored submissions. `noise` is a known LB noise SD; without it the
    residual SD is used (it needs more submissions than the model has unknowns)."""
    from pseudorca.stats import lb_noise, t_quantile

    _check_confidence(confidence)
    if noise is not None and not (math.isfinite(noise) and noise > 0):
        raise ValueError("the noise SD must be a positive number")
    ideas = _known_ideas(state, ideas)
    rows = submission_rows(
        state, private=private, competition=competition, include_parked=include_parked, also=ideas or ()
    )
    fit = Fit("lb", f"{'private' if private else 'public'} score", rows, [], confidence, higher_is_better)
    alpha = 1 - confidence

    def make_interval(m: _Model) -> Callable[[np.ndarray], tuple[Any, Any, Any, Any]] | None:
        if noise is not None:
            fit.sigma, fit.df, fit.noise_from = noise, math.inf, "given"
        elif m.df >= 1 and m.rss > 0:
            fit.sigma, fit.df, fit.noise_from = math.sqrt(m.rss / m.df), float(m.df), "residuals"
        else:
            return None  # n = rank, or residuals of exactly 0 (resubmissions that repeat their scores)
        sigma, t = fit.sigma, -t_quantile(alpha / 2, fit.df)  # the lower tail: exact for any level
        m.mde_quantiles = (t, t_quantile(0.8, fit.df))

        def interval(w: np.ndarray) -> tuple[float, float, float, float]:
            est = m.snap(w, float(w @ m.y))
            se = sigma * float((w**2).sum()) ** 0.5
            return est, se, est - t * se, est + t * se

        return interval

    _fit(
        fit,
        ideas,
        "submissions",
        make_interval,
        need=" with a score and a registered notebook",
        hint="`pseudorca audit --submitted` fills the gaps",
    )
    if not private:  # the resubmissions among the rows in the model (lb_noise uses the public scores)
        used_ids = {r.key for r in fit.used}
        fit.resubmission = lb_noise([s for s in state.subs.values() if s.id in used_ids])
    m = fit._model
    if m is not None and fit.noise_from == "":
        if m.df >= 1:  # the residuals are exactly 0
            fit.notes.append(
                f"{ZERO_RESIDUALS}: every submission scored exactly its fitted value (resubmissions repeated "
                "their scores), which says nothing about the noise between notebooks. Give a known LB noise SD "
                "(`pseudorca effects --noise SD`, or `noise` under [effects] in config.toml) for intervals"
            )
        else:
            fit.notes.append(
                f"{NO_NOISE}. Resubmit a notebook, or give a known LB noise SD (`pseudorca effects --noise SD`, "
                "or `noise` under [effects] in config.toml)"
            )
    if m is not None and m.df >= 1:  # whatever the noise: it uses only the residuals and the resubmissions
        _lack_of_fit(fit, bool(ideas))
    return fit


def _lack_of_fit(fit: Fit, chosen: bool) -> None:
    """The lack-of-fit F test: do submissions of different notebooks stray from the model more than
    resubmissions of one notebook stray from each other? Sets fit.lof_p, and a note when p < 1 − confidence."""
    import numpy as np

    from pseudorca.stats import f_sf

    m = fit._model
    assert m is not None
    groups: dict[str | None, list[float]] = {}
    for r in fit.used:
        groups.setdefault(r.nb, []).append(float(r.y or 0.0))
    df_pure = sum(len(v) - 1 for v in groups.values())
    df_lof = m.df - df_pure
    if df_pure < 1 or df_lof < 1:
        return
    ss_pure = float(sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in groups.values() if len(v) > 1))
    ss_lof = m.rss - ss_pure
    if ss_lof <= (SNAP * float(np.abs(m.y).max())) ** 2 * len(m.y):
        ss_lof = 0.0  # rounding
    if ss_pure > 0:
        fit.lof_p = f_sf((ss_lof / df_lof) / (ss_pure / df_pure), df_lof, df_pure)
    else:
        fit.lof_p = 0.0 if ss_lof > 0 else 1.0
    if fit.lof_p < 1 - fit.confidence:
        fit.notes.append(
            "submissions of different notebooks stray from the model more than resubmissions of one notebook "
            f"stray from each other (lack-of-fit F test, p = {fit.lof_p:.2g}): ideas may interact, or a change "
            "no idea describes matters" + (", or ideas left out by --idea" if chosen else "")
        )


# ----- local runs ----------------------------------------------------------------------------------


def run_fits(
    state: State,
    load: Callable[[Run], Table],
    *,
    metric: str = "score",
    protocol: str | None = None,
    ideas: Sequence[str] | None = None,
    confidence: float = 0.95,
    higher_is_better: bool = True,
    resamples: int = 10_000,
    seed: int = 0,
    include_parked: bool = False,
) -> list[Fit]:
    """The additive model over local runs, one fit per protocol (or only `protocol`).

    `load` returns a run's results table; a run whose table cannot be loaded (ValueError, LookupError or
    OSError, or a store error) is listed as not used. A table recorded more than once counts once, through
    its first run that can be used."""
    from pseudorca.store import StoreError

    _check_confidence(confidence)
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    ideas = _known_ideas(state, ideas)
    runs = sorted(state.runs.values(), key=lambda r: r.id)
    protocols = sorted({r.protocol for r in runs})
    if protocol is not None:
        if protocol not in protocols:
            have = ", ".join(repr(p) if p else "no protocol" for p in protocols) or "none"
            raise LookupError(f"no runs have the protocol {protocol!r} (protocols: {have})")
        protocols = [protocol]
    ideas_all = list(dict.fromkeys([*idea_ids(state, include_parked), *(ideas or [])]))
    fits = []
    for proto in protocols:
        rows: list[Row] = []
        values: dict[str, dict[str, Any]] = {}
        kept: dict[str, Run] = {}  # results table -> the run that stands for it
        repeats = 0
        for run in (r for r in runs if r.protocol == proto):
            nb = state.notebooks.get(run.nb) if isinstance(run.nb, str) and run.nb else None
            name = f"{run.id[:10]} ({run.label})" if run.label else run.id[:10]
            row = Row(
                key=run.id,
                name=name,
                nb=run.nb if isinstance(run.nb, str) else None,
                nb_label=nb.label if nb else util.short(run.nb if isinstance(run.nb, str) else ""),
                date=run.added,
                y=None,
                cells=_cells(state, nb.id if nb else None, ideas_all),
            )
            if not run.nb:
                row.why_not = "no notebook (add its table again with `pseudorca run add FILE --nb NB`: that copy is used)"
            elif nb is None:
                row.why_not = "its notebook is not registered"
            elif run.results in kept:
                row.why_not = f"repeats the results table of run {kept[run.results].id[:10]}"
                repeats += 1
            else:
                try:
                    values[run.id] = _item_values(load(run), metric)
                    kept[run.results] = run
                except (ValueError, LookupError, OSError, StoreError) as e:
                    row.why_not = str(e) or type(e).__name__
            rows.append(row)
        fit = Fit("local", metric, rows, [], confidence, higher_is_better, protocol=proto)
        fit.resamples, fit.seed, fit.noise_from = resamples, seed, "bootstrap"
        _local(fit, values, ideas)
        if repeats:
            fit.notes.append(f"{repeats} run(s) repeat another run's results table and are left out")
        fits.append(fit)
    return fits


def _item_values(table: Table, metric: str) -> dict[str, Any]:
    """Each item's draws of the metric, for the items that have any. A table without any, or with an
    infinite value, is a ValueError (its run is then not used, rather than emptying or poisoning the rest)."""
    import numpy as np

    from pseudorca.results import per_item

    if metric not in table.metrics:
        raise ValueError(f"no metric {metric!r} (metrics: {', '.join(table.metrics) or 'none'})")
    values = {k: v for k, v in per_item(table, metric).items() if v.size}
    if not values:
        raise ValueError(f"no values of {metric!r}")
    for item, v in values.items():
        if not np.isfinite(v).all():
            raise ValueError(f"item {item!r} has an infinite {metric!r}")
    return values


def _local(fit: Fit, values: dict[str, dict[str, Any]], chosen: Sequence[str] | None) -> None:
    import numpy as np

    from pseudorca.stats import min_items

    if chosen:
        _choose(fit.rows, chosen, "runs")  # first drop the runs without a verdict for a chosen idea
    used = fit.used
    common: set[str] | None = None
    every: set[str] = set()
    for r in used:
        keys = set(values[r.key])
        every |= keys
        common = keys if common is None else common & keys
    items = sorted(common or ())
    fit.n_items, fit.n_items_any = len(items), len(every)
    if len(used) >= MIN_ROWS and not items:
        for r in used:
            r.why_not = "no item is in every run"  # (it may share items with some of them)
            r.y = None
        fit.notes.append("no item is in every run, so they cannot be paired")
    for r in fit.rows:
        if r.used:
            r.y = float(np.mean([values[r.key][i].mean() for i in items]))
        elif r.key in values and items:
            have = [values[r.key][i].mean() for i in items if i in values[r.key]]
            r.y = float(np.mean(have)) if have else None
        else:
            r.y = None
    if 0 < len(items) < fit.n_items_any:
        fit.notes.append(f"only the {len(items)} items every run has are used (of {fit.n_items_any})")
    floor = TOO_FEW if 0 < len(items) < min_items(fit.confidence) else ""

    def make_interval(m: _Model) -> Callable[[np.ndarray], tuple[Any, Any, Any, Any]]:
        import statistics

        from pseudorca.stats import _flips, _sign_flip_p

        draws = [[values[r.key][i] for i in items] for r in fit.used]
        boot_seq, perm_seq, run_seq = np.random.SeedSequence(fit.seed).spawn(3)  # the first two as `compare`
        boot = _bootstrap(draws, fit.resamples, np.random.default_rng(boot_seq))
        means = np.array([[x.mean() for x in run] for run in draws]).T  # items × runs
        boot = _run_to_run(fit, m, boot, means, np.random.default_rng(run_seq))
        alpha = 1 - fit.confidence
        m.mde_quantiles = (-statistics.NormalDist().inv_cdf(alpha / 2), POWER_Z)

        def interval(w: np.ndarray) -> tuple[float, float, float, float]:
            dist = boot @ w
            lo, hi = (float(q) for q in np.quantile(dist, [alpha / 2, 1 - alpha / 2]))
            return m.snap(w, float(w @ m.y)), float(dist.std(ddof=1)), lo, hi

        # w·y is the mean over items of means @ w, one paired difference per item: test it as `compare` does
        m.p_value = lambda w: _sign_flip_p(means @ w, np.random.default_rng(perm_seq), _flips(alpha))
        return interval

    _fit(
        fit,
        chosen,
        "runs",
        make_interval,
        floor,
        need=" of registered notebooks with this metric",
        hint="`pseudorca audit NB` on those notebooks fills the gaps",
    )


def _run_to_run(
    fit: Fit, m: _Model, boot: np.ndarray, means: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """The bootstrap with run-to-run noise added, when runs with the same ideas show some; and a check
    that the runs follow the additive model.

    The item/draw bootstrap knows nothing of noise shared by a whole run (another machine, a cache, a
    nondeterministic server). Runs with the same modelled ideas reveal it: their spread around their own
    mean, SS_rep with df_rep degrees of freedom. Its expectation without run noise comes from resampling
    the items alone (each item keeps its mean over draws: resampling the draws as well would count their
    noise twice), and the part beyond it is τ²·df_rep. N(0, (k·τ)²) per run is added to every resample of
    the intervals, with k = t(1 − α/2, df_rep) / z(1 − α/2), since τ is estimated from df_rep degrees of
    freedom. The rest of the residuals (df − df_rep degrees of freedom) is how far the runs stray from the
    additive model; it is tested against the same resamples plus the run noise (fit.lof_p)."""
    import statistics

    import numpy as np

    from pseudorca.stats import _CHUNK, t_quantile

    if m.df < 1:
        return boot  # every run has its own ideas, and the model fits them exactly: nothing to measure
    groups: dict[tuple[float, ...], list[int]] = {}
    for i, row in enumerate(m.x):
        groups.setdefault(tuple(row), []).append(i)
    sets = [np.array(g) for g in groups.values() if len(g) > 1]
    df_rep = sum(len(g) - 1 for g in sets)
    df_lof = m.df - df_rep

    def within(v: np.ndarray) -> np.ndarray:  # v less the mean of each set of runs with the same ideas
        out = np.zeros_like(v)
        for g in sets:
            part = v[..., g]
            out[..., g] = part - part.mean(axis=-1, keepdims=True)
        return out

    n_items, n_runs = means.shape
    reference = np.empty((len(boot), n_runs))  # run means over resampled items, draws kept
    chunk = max(1, _CHUNK // (n_items * n_runs))
    for start in range(0, len(boot), chunk):
        size = min(chunk, len(boot) - start)
        reference[start : start + size] = means[rng.integers(0, n_items, size=(size, n_items))].mean(axis=1)
    reference -= reference.mean(axis=0)
    alpha = 1 - fit.confidence
    tau = 0.0
    if df_rep:
        observed = float((within(m.y) ** 2).sum()) if m.rss > 0 else 0.0
        sums = (within(reference) ** 2).sum(axis=1)
        tau = math.sqrt(max(0.0, (observed - float(sums.mean())) / df_rep))
        fit.tau, fit.tau_df = tau, df_rep
        p = float((1 + (sums >= observed).sum()) / (1 + len(sums)))
        if p < alpha:
            fit.notes.append(
                f"runs with the same ideas differ beyond their item and draw noise (p = {p:.2g}): the intervals "
                f"include a run-to-run SD of {tau:.3g} from {df_rep} degree(s) of freedom; more runs of the "
                "same notebooks pin it down"
            )
    if df_lof:
        noise = tau * rng.standard_normal(reference.shape)
        rest = m.residuals(m.y) - within(m.y) if m.rss > 0 else np.zeros_like(m.y)
        resampled = m.residuals(reference + noise)
        resampled -= within(reference + noise)
        observed = float((rest**2).sum())
        fit.lof_p = float((1 + ((resampled**2).sum(axis=1) >= observed).sum()) / (1 + len(reference)))
        if fit.lof_p < alpha:
            fit.notes.append(
                "the runs stray from the additive model beyond their noise (lack-of-fit p = "
                f"{fit.lof_p:.2g}): ideas may interact, a change no idea describes may matter"
                + ("" if df_rep else ", or runs have noise of their own (run a notebook twice to tell)")
            )
    if tau <= 0:
        return boot
    k = t_quantile(1 - alpha / 2, df_rep) / statistics.NormalDist().inv_cdf(1 - alpha / 2)
    return boot + k * tau * rng.standard_normal(boot.shape)


def _bootstrap(runs: Sequence[Sequence[np.ndarray]], resamples: int, rng: np.random.Generator) -> np.ndarray:
    """(resamples × runs) means over resampled items, then resampled draws within each item and run. The
    same items are drawn for every run, so the runs stay paired. It draws as `stats._bootstrap` does, so
    two runs get `compare`'s resamples."""
    import numpy as np

    from pseudorca.stats import _CHUNK, _pad, _resampled_means

    padded = [_pad(xs) for xs in runs]
    n = len(runs[0])
    width = sum(p.shape[1] for p, _ in padded) + 1
    chunk = max(1, _CHUNK // (n * width))
    out = np.empty((resamples, len(runs)))
    for start in range(0, resamples, chunk):
        m = min(chunk, resamples - start)
        idx = rng.integers(0, n, size=(m, n))
        for j, (p, c) in enumerate(padded):
            out[start : start + m, j] = _resampled_means(rng, p, c, idx).mean(axis=1)
    return out


# ----- waterfall -----------------------------------------------------------------------------------


def waterfall(fit: Fit, base: str, target: str) -> Waterfall:
    """From `base`'s fitted score to `target`'s (both used rows), one step per idea or bundle that differs.
    Differences the rows cannot separate are one step per group of bundles that mix with each other."""
    import numpy as np

    m = fit._model
    rows = {r.key: r for r in fit.used}
    for key in (base, target):
        if key not in rows:
            raise LookupError(f"{key} is not in the model ({_why(fit, key)})")
    if m is None:
        raise LookupError("there is no model to break down")
    b, t = rows[base], rows[target]
    index = {r.key: i for i, r in enumerate(fit.used)}
    d = m.x[index[target]] - m.x[index[base]]
    d[0] = 0.0
    steps: list[Step] = []
    stuck: list[int] = []
    for j, members in enumerate(m.bundles):
        if abs(d[j + 1]) < TOL:
            continue
        c = m.unit(j, float(d[j + 1]))
        if m.estimable(c):
            est, se, lo, hi = _snapped(m, c)
            sign = 1 if d[j + 1] > 0 else -1
            label = ("+ " if sign > 0 else f"{MINUS} ") + _group_label(members)
            steps.append(Step(label, list(members), sign, est, se, lo, hi))
        else:
            stuck.append(j)
    # d is estimable (a difference of two rows), and so is its part on each group: the parts' projections
    # onto the null space are orthogonal to each other and add up to d's, which is 0, so each one is 0
    for group in _groups(m, stuck):
        c = np.zeros_like(d)
        for j in group:
            c += m.unit(j, float(d[j + 1]))
        came = [i for j in group if d[j + 1] > 0 for i in m.bundles[j]]
        left = [i for j in group if d[j + 1] < 0 for i in m.bundles[j]]
        if came and left:
            label = f"{_group_label(came)} instead of {_group_label(left)}"
        else:
            label = ("+ " if came else f"{MINUS} ") + _group_label(came or left) + " (not separable)"
        est, se, lo, hi = _snapped(m, c)
        steps.append(Step(label, came + left, 0, est, se, lo, hi))
    sign = 1 if fit.higher_is_better else -1
    steps.sort(key=lambda s: -sign * s.delta)
    assert b.fitted is not None and t.fitted is not None
    modelled = {i for members in m.bundles for i in members}
    unmodelled = [  # outside the model, and not known to be the same in both (no verdict on either: no news)
        iid
        for iid, cell in t.cells.items()
        if b.nb != t.nb  # one notebook is the same as itself
        and iid not in modelled
        and (b.cells[iid].status is not None or cell.status is not None)
        and not (cell.value is not None and b.cells[iid].value == cell.value)
    ]
    return Waterfall(b, t, b.fitted, t.fitted, steps, unmodelled)


def _group_label(ideas: Sequence[str]) -> str:
    """Ideas that move together: "a", or "(a + b)" so that a sign in front applies to all of them."""
    return ideas[0] if len(ideas) == 1 else f"({' + '.join(ideas)})"


def _snapped(m: _Model, c: np.ndarray) -> tuple[float, float | None, float | None, float | None]:
    w = m.weights(c)
    est, se, lo, hi = m.interval(w)
    if lo is not None and hi is not None:
        lo, hi = m.snap(w, lo), m.snap(w, hi)
    return est, se, lo, hi


def _why(fit: Fit, key: str) -> str:
    for r in fit.rows:
        if r.key == key:
            return r.why_not or "used"
    return "no such row"


# ----- JSON ----------------------------------------------------------------------------------------


def fit_json(fit: Fit) -> dict[str, Any]:
    import dataclasses

    return {
        "source": fit.source,
        "target": fit.target,
        "protocol": fit.protocol,
        "confidence": fit.confidence,
        "higher_is_better": fit.higher_is_better,
        "noise": {
            "from": fit.noise_from or None,
            "sd": fit.sigma,
            "df": fit.df,
            "resubmission": dataclasses.asdict(fit.resubmission) if fit.resubmission else None,
        },
        "items": {"used": fit.n_items, "any": fit.n_items_any} if fit.source == "local" else None,
        "resamples": fit.resamples or None,
        "seed": fit.seed if fit.source == "local" else None,
        "lack_of_fit_p": fit.lof_p,
        "run_to_run_sd": fit.tau,
        "rows": [
            {
                "key": r.key,
                "name": r.name,
                "nb": r.nb,
                "nb_label": r.nb_label,
                "date": r.date,
                "y": r.y,
                "fitted": r.fitted,
                "used": r.used,
                "why_not": r.why_not or None,
                "changes": moved,
                "resubmits": again,
                "cells": {
                    k: {"status": c.status, "stale": c.stale, "method": c.method}
                    for k, c in r.cells.items()
                    if c.status
                },
            }
            for r, moved, again in zip(fit.rows, all_changes(fit.rows), resubmits(fit.rows), strict=True)
        ],
        "terms": [{**dataclasses.asdict(t), "label": t.label} for t in fit.terms],
        "notes": list(fit.notes),
    }


def waterfall_json(w: Waterfall) -> dict[str, Any]:
    import dataclasses

    return {
        "base": w.base.key,
        "target": w.target.key,
        "start": w.start,
        "end": w.end,
        "observed_base": w.base.y,
        "observed_target": w.target.y,
        "steps": [dataclasses.asdict(s) for s in w.steps],
        "unmodelled": list(w.unmodelled),
    }


def config_defaults(setting: Callable[[str, str, Any], Any]) -> tuple[bool, float | None]:
    """(lower_is_better, noise) from config.toml's [effects] table, through `store.setting`:

        [effects]
        lower_is_better = true   # the competition's score: lower is better (default false)
        noise = 0.012            # a known LB noise SD (default: from the residuals)

    A value of the wrong type is a ValueError that names it."""
    lower = setting("effects", "lower_is_better", False)
    if not isinstance(lower, bool):
        raise ValueError("[effects] lower_is_better in config.toml must be true or false")
    noise = setting("effects", "noise", None)
    if noise is not None:
        if (
            isinstance(noise, bool)
            or not isinstance(noise, (int, float))
            or not (math.isfinite(noise) and noise > 0)
        ):
            raise ValueError("[effects] noise in config.toml must be a positive number (the LB noise SD)")
        noise = float(noise)
    return lower, noise


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")


def _known_ideas(state: State, ideas: Sequence[str] | None) -> list[str] | None:
    """`ideas` once each, in order (None for none); an id that is not in the ledger is a LookupError."""
    if not ideas:
        return None
    for iid in ideas:
        if iid not in state.ideas:
            raise LookupError(f"no idea matches {iid!r}")
    return list(dict.fromkeys(ideas))


def resolve_ideas(state: State, refs: Sequence[str]) -> list[str]:
    """Idea ids for --idea values (ids or unique prefixes), each once, in the given order."""
    out: list[str] = []
    for ref in refs:
        iid = state.resolve_idea(ref).id
        if iid not in out:
            out.append(iid)
    return out


def pick_row(rows: Sequence[Row], ref: str) -> Row:
    """A row by key, or by a unique key prefix (4+ characters)."""
    ref = ref.strip()
    for r in rows:
        if r.key == ref:
            return r
    hits = [r for r in rows if len(ref) >= 4 and r.key.startswith(ref)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise LookupError(f"{ref!r} matches {len(hits)} submissions; use more characters")
    raise LookupError(f"no submission matches {ref!r} (see `pseudorca lb`)")
