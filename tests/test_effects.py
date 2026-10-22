"""`effects`: which ideas each submission has, and what each idea adds (an additive model over the rows)."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from pseudorca import effects, stats, util
from pseudorca.cli import main
from pseudorca.model import Idea, Notebook, Run, State, Submission, VerdictRecord
from pseudorca.notebooks import register_bytes
from pseudorca.results import load_table, per_item
from pseudorca.store import Store, StoreError

STATUS = {"1": "present", "0": "absent", "p": "partial", "u": "unknown"}

# ----- building ledgers ----------------------------------------------------------------------------


def _ideas(store: Store, ids: str) -> dict[str, str]:
    """Add the ideas (space-separated ids); returns {id: rev}."""
    for iid in ids.split():
        store.append("idea.add", {"id": iid, "title": f"idea {iid}"})
    st = store.state()
    return {iid: st.ideas[iid].rev for iid in ids.split()}


def _nb(store: Store, name: str) -> str:
    return register_bytes(store, json.dumps({"cells": [name]}).encode(), name=name)[0].id


def _record(store: Store, nb_id: str, idea: str, rev: str, code: str) -> None:
    """One verdict code: 1 present, 0 absent, p partial, u unknown, x conflict (fingerprint and patch
    disagree), - none."""
    if code == "-":
        return
    if code == "x":
        for method, status in (("fingerprint", "present"), ("patch", "absent")):
            data = {"nb": nb_id, "idea": idea, "idea_rev": rev, "status": status, "method": method}
            store.append("audit.verdict", data)
        return
    data = {"nb": nb_id, "idea": idea, "idea_rev": rev, "status": STATUS[code], "method": "human"}
    store.append("audit.verdict", data)


def _day(k: int) -> str:
    return (date(2026, 3, 1) + timedelta(days=k)).isoformat()


def _ledger(store: Store, ideas: str, subs: Sequence[tuple[str, str | None, float | None, str]]) -> State:
    """The ideas (space-separated ids) and one submission per entry: (id, notebook name or None, public score,
    verdict codes, one per idea; see `_record`). A notebook's verdicts are recorded where it first appears.
    The submissions are a day apart, in the order given."""
    revs = _ideas(store, ideas)
    nbs: dict[str, str] = {}
    for k, (sid, name, score, codes) in enumerate(subs):
        if name is not None and name not in nbs:
            nbs[name] = _nb(store, name)
            for iid, code in zip(revs, codes, strict=True):
                _record(store, nbs[name], iid, revs[iid], code)
        nb_id = nbs.get(name) if name else None
        store.append("sub.upsert", {"id": sid, "nb": nb_id, "public_score": score, "date": _day(k)})
    return store.state()


def _state(cells: np.ndarray, y: Sequence[float]) -> State:
    """An in-memory ledger, for many designs: idea i0, i1... per column of `cells` (1 present, 0 absent),
    and one notebook and one scored submission per row, a minute apart."""
    st = State()
    names = [f"i{j}" for j in range(cells.shape[1])]
    for iid in names:
        st.ideas[iid] = Idea(id=iid, title=iid)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for k, (row, score) in enumerate(zip(cells, y, strict=True)):
        nb = util.sha256_bytes(f"nb{k}".encode())
        st.notebooks[nb] = Notebook(id=nb, name=f"nb{k}", seq=k + 1)
        for iid, v in zip(names, row, strict=True):
            status = "present" if v else "absent"
            st.verdicts[(nb, iid)] = [VerdictRecord(f"e{k}", nb, iid, st.ideas[iid].rev, status, "human")]
        when = (start + timedelta(minutes=k)).isoformat()
        st.subs[str(k)] = Submission(id=str(k), nb=nb, public_score=float(score), date=when)
    return st


def _terms(fit: effects.Fit) -> dict[str, effects.Term]:
    return {t.label: t for t in fit.terms}


def _cli(store: Store, *args: str) -> Any:
    return CliRunner().invoke(main, ["--store", str(store.root), *args])


# ----- least squares, by hand ----------------------------------------------------------------------


def _design(cells: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(cells)), cells]).astype(float)


def _contrast(names: list[str], plus: Sequence[str], minus: Sequence[str] = ()) -> np.ndarray:
    """A contrast on [intercept | one column per idea]."""
    c = np.zeros(len(names) + 1)
    for i in plus:
        c[names.index(i) + 1] += 1
    for i in minus:
        c[names.index(i) + 1] -= 1
    return c


def _estimable(x: np.ndarray, c: np.ndarray) -> bool:
    return int(np.linalg.matrix_rank(np.vstack([x, c]))) == int(np.linalg.matrix_rank(x))


def _term_contrast(names: list[str], t: effects.Term) -> np.ndarray:
    if t.kind == effects.DIFFERENCE:
        return _contrast(names, t.left, t.right)
    return _contrast(names, t.ideas)


# ----- the model on noiseless and noisy designs ----------------------------------------------------


def test_noiseless_scores_give_the_true_effects(store: Store):
    truth = {"a": 0.10, "b": -0.03, "c": 0.02}
    patterns = ["000", "100", "010", "001", "110", "101", "011", "111"]
    subs = []
    for k, codes in enumerate(patterns):
        score = 0.5 + sum(v for v, code in zip(truth.values(), codes, strict=True) if code == "1")
        subs.append((str(100 + k), f"nb{k}", score, codes))
    st = _ledger(store, "a b c", subs)
    fit = effects.lb_fit(st)
    terms = _terms(fit)
    assert [t.kind for t in fit.terms] == [effects.EFFECT] * 3
    assert [t.label for t in fit.terms] == ["a", "c", "b"]  # best first
    for iid, value in truth.items():
        t = terms[iid]
        assert t.estimate == pytest.approx(value, abs=1e-12)
        assert t.ci_low is None and t.ci_high is None  # residuals of exactly 0 are no noise estimate
        assert (t.n_with, t.n_without) == (4, 4)
    assert {t.verdict for t in fit.terms} == {"unknown"} and fit.sigma is None
    assert fit.notes[0].startswith("no noise estimate: the residuals are 0: every submission scored exactly")
    given = _terms(effects.lb_fit(st, noise=0.001))
    assert (given["a"].verdict, given["c"].verdict, given["b"].verdict) == ("helps", "helps", "hurts")
    assert fit.df is None and fit._model.df == 4 and fit.noise_from == ""
    for r in fit.rows:
        assert r.used and r.fitted == pytest.approx(r.y, abs=1e-12)
    assert fit.best() is not None and fit.best().key == "105"  # a and c: 0.62, the highest
    assert len(fit.notes) == 1  # only that there is no noise estimate


def test_a_noisy_full_rank_design_is_ordinary_least_squares(store: Store):
    rng = np.random.default_rng(7)
    cells = np.array([[int(c) for c in f"{k:04b}"] for k in range(16)])[rng.permutation(16)][:12]
    y = 0.4 + cells @ np.array([0.05, -0.02, 0.0, 0.03]) + rng.normal(0, 0.01, 12)
    subs = [
        (str(k), f"nb{k}", float(v), "".join(map(str, row)))
        for k, (row, v) in enumerate(zip(cells, y, strict=True))
    ]
    st = _ledger(store, "a b c d", subs)
    fit = effects.lb_fit(st, confidence=0.9)
    x = _design(cells)
    beta, rss, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    assert rank == 5 and fit.df == 12 - 5
    sigma = math.sqrt(float(rss[0]) / 7)
    assert fit.sigma == pytest.approx(sigma, rel=1e-10)
    cov = np.linalg.inv(x.T @ x)
    t90 = stats.t_quantile(0.95, 7)
    for j, iid in enumerate("abcd"):
        term = _terms(fit)[iid]
        se = sigma * math.sqrt(cov[j + 1, j + 1])
        assert term.kind == effects.EFFECT
        assert term.estimate == pytest.approx(beta[j + 1], abs=1e-12)
        assert term.se == pytest.approx(se, rel=1e-10)
        assert term.ci_low == pytest.approx(beta[j + 1] - t90 * se, rel=1e-9, abs=1e-12)
        assert term.ci_high == pytest.approx(beta[j + 1] + t90 * se, rel=1e-9, abs=1e-12)
        assert term.mde80 == pytest.approx((t90 + stats.t_quantile(0.8, 7)) * se)  # the t test it uses
        expected = "helps" if term.ci_low > 0 else "hurts" if term.ci_high < 0 else "within noise"
        assert term.verdict == expected
    assert [_terms(fit)[i].verdict for i in "abcd"] == ["helps", "hurts", "within noise", "helps"]
    fitted = x @ beta
    assert [r.fitted for r in fit.rows] == pytest.approx(list(fitted), abs=1e-12)
    # a contrast of two ideas, through the breakdown: c'b and sigma·sqrt(c'(X'X)^-1 c)
    base = next(r for r in fit.rows if r.cells["a"].value == 0 and r.cells["b"].value == 1)
    target = next(
        r
        for r in fit.rows
        if r.cells["a"].value == 1
        and r.cells["b"].value == 0
        and all(r.cells[i].value == base.cells[i].value for i in "cd")
    )
    w = effects.waterfall(fit, base.key, target.key)
    assert [s.label for s in w.steps] == ["+ a", "− b"]
    assert [s.delta for s in w.steps] == pytest.approx([beta[1], -beta[2]], abs=1e-12)
    assert w.steps[1].se == pytest.approx(sigma * math.sqrt(cov[2, 2]), rel=1e-10)
    assert w.end - w.start == pytest.approx(fitted[fit.rows.index(target)] - fitted[fit.rows.index(base)])


def _random_cells(rng: np.random.Generator) -> np.ndarray:
    """A random 0/1 design with the structures that make ideas hard to tell apart: copies, complements,
    disjoint pairs and their union, and constant columns."""
    n = int(rng.integers(3, 10))
    cols = [rng.random(n) < rng.uniform(0.2, 0.8) for _ in range(int(rng.integers(1, 5)))]
    for _ in range(int(rng.integers(0, 4))):
        a = cols[int(rng.integers(len(cols)))]
        kind = int(rng.integers(5))
        if kind == 0:
            cols.append(a.copy())
        elif kind == 1:
            cols.append(~a)
        elif kind == 2:
            cols.append(np.ones(n, bool) if rng.random() < 0.5 else np.zeros(n, bool))
        else:  # a split of `a` into two disjoint ideas, kept beside it
            part = a & (rng.random(n) < 0.5)
            cols += [part, a & ~part]
    return np.column_stack(cols).astype(int)


@pytest.mark.parametrize("seed", range(60))
def test_random_designs_agree_with_least_squares_by_hand(seed: int):
    """Whatever the design, every number is an estimable function computed as least squares would, and
    nothing that is not estimable gets one. Checked on [1 | one column per idea], not the model's bundles."""
    rng = np.random.default_rng(seed)
    cells = _random_cells(rng)
    n, k = cells.shape
    names = [f"i{j}" for j in range(k)]
    y = 0.5 + cells @ rng.normal(0, 0.05, k) + rng.normal(0, 0.01, n)
    fit = effects.lb_fit(_state(cells, y))
    x = _design(cells)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    rank = int(np.linalg.matrix_rank(x))
    assert [r.fitted for r in fit.rows] == pytest.approx(list(x @ beta), abs=1e-9)
    df = n - rank
    if df:
        rss = float(((y - x @ beta) ** 2).sum())
        assert (fit.noise_from, fit.df) == ("residuals", df)
        assert fit.sigma == pytest.approx(math.sqrt(rss / df), rel=1e-6, abs=1e-12)
        t = stats.t_quantile(0.975, df)
    else:
        assert fit.noise_from == "" and fit.sigma is None and fit.notes[0].startswith("no noise estimate")
    gram = np.linalg.pinv(x.T @ x)
    varying = [i for j, i in enumerate(names) if 0 < cells[:, j].sum() < n]
    single: dict[str, str] = {}  # idea -> the kind of the one non-difference term it is in
    in_difference: set[str] = set()
    differences = []
    for term in fit.terms:
        c = _term_contrast(names, term)
        if term.kind in effects.MEASURED:
            assert _estimable(x, c), term.label
            assert term.estimate == pytest.approx(float(c @ beta), abs=1e-9), term.label
            if df:
                se = fit.sigma * math.sqrt(float(c @ gram @ c))
                assert term.se == pytest.approx(se, rel=1e-6, abs=1e-12), term.label
                assert term.ci_low == pytest.approx(term.estimate - t * term.se, abs=1e-9)
                assert term.ci_high == pytest.approx(term.estimate + t * term.se, abs=1e-9)
            else:
                assert (term.se, term.ci_low, term.verdict) == (None, None, effects.UNKNOWN)
        if term.kind == effects.DIFFERENCE:
            in_difference.update(term.ideas)
            differences.append(c)
            assert set(term.left).isdisjoint(term.right) and term.ideas == [*term.left, *term.right]
            continue
        for iid in term.ideas:
            assert iid not in single, f"{iid} is in two terms"
            single[iid] = term.kind
        col = cells[:, names.index(term.ideas[0])]
        if term.kind in (effects.ALWAYS, effects.NEVER):
            assert len(term.ideas) == 1 and (col.all() if term.kind == effects.ALWAYS else not col.any())
        else:
            assert 0 < col.sum() < n
            # a bundle: ideas with identical columns, and every such idea is in it
            same = [i for i in varying if (cells[:, names.index(i)] == col).all()]
            assert term.ideas == same, term.label
            assert (term.n_with, term.n_without) == (int(col.sum()), n - int(col.sum()))
        if term.kind == effects.TANGLED:
            assert not _estimable(x, c), term.label
            # no difference with any other bundle is estimable, or the greedy pick would have covered it
            for other in varying:
                if other not in term.ideas:
                    assert not _estimable(x, c - _contrast(names, [other])), (term.label, other)
    # every idea is in exactly one term, or covered by differences
    assert set(single) | in_difference == set(names)
    assert not set(single) & in_difference
    if differences:  # none of the differences is a combination of the others
        assert int(np.linalg.matrix_rank(np.array(differences))) == len(differences)
    # the breakdown from the first row to every other row adds up to the change in the fitted score
    first = fit.used[0]
    for row in fit.used[1:]:
        w = effects.waterfall(fit, first.key, row.key)
        assert sum(s.delta for s in w.steps) == pytest.approx(w.end - w.start, abs=1e-9)
        assert (w.start, w.end) == (first.fitted, row.fitted)
        for s in w.steps:
            assert (" instead of " in s.label) == (s.sign == 0 and not s.label.endswith("(not separable)"))


# ----- what the rows can and cannot separate -------------------------------------------------------


def test_ideas_always_in_or_out_together_are_one_bundle(store: Store):
    st = _ledger(
        store,
        "a b c",
        [
            ("1", "n1", 0.50, "000"),
            ("2", "n2", 0.58, "110"),
            ("3", "n3", 0.51, "001"),
            ("4", "n4", 0.60, "111"),
            ("5", "n1", 0.52, ""),
        ],
    )
    fit = effects.lb_fit(st)
    bundle = _terms(fit)["a + b"]
    assert bundle.kind == effects.BUNDLE and bundle.ideas == ["a", "b"]
    assert bundle.note == "always in or out together: measured together"
    assert (bundle.n_with, bundle.n_without) == (2, 3)
    y = np.array([0.50, 0.58, 0.51, 0.60, 0.52])
    beta = np.linalg.lstsq(_design(np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0]])), y, rcond=None)[0]
    assert bundle.estimate == pytest.approx(beta[1]) and _terms(fit)["c"].estimate == pytest.approx(beta[2])
    assert [t.kind for t in fit.terms] == [effects.BUNDLE, effects.EFFECT]


def test_complementary_ideas_are_one_difference_with_the_latest_choice_first(store: Store):
    rows = [
        ("1", "n1", 0.50, "100"),
        ("2", "n2", 0.56, "011"),
        ("3", "n3", 0.55, "010"),
        ("4", "n4", 0.52, "101"),
        ("5", "n3", 0.57, ""),  # the latest resubmits n3, which has `new`
    ]
    st = _ledger(store, "old new x", rows)
    fit = effects.lb_fit(st)
    (diff,) = [t for t in fit.terms if t.kind == effects.DIFFERENCE]
    assert diff.label == "new instead of old" and (diff.left, diff.right) == (["new"], ["old"])
    assert diff.ideas == ["new", "old"] and (diff.n_with, diff.n_without) == (3, 2)
    assert diff.note == "neither can be measured alone; only the difference between them can"
    # old = 1 − new, so [1, old, new, x] is [1, new, x] in disguise: the difference is new's coefficient there
    y = np.array([0.50, 0.56, 0.55, 0.52, 0.57])
    beta = np.linalg.lstsq(_design(np.array([[0, 0], [1, 1], [1, 0], [0, 1], [1, 0]])), y, rcond=None)[0]
    assert diff.estimate == pytest.approx(beta[1]) and _terms(fit)["x"].estimate == pytest.approx(beta[2])
    assert {t.kind for t in fit.terms} == {effects.DIFFERENCE, effects.EFFECT}  # neither is tangled

    # the same submissions with one that has `old` last: the other way round, the same number negated
    cells = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0]])
    flipped = effects.lb_fit(_state(cells, [0.50, 0.56, 0.55, 0.52, 0.57, 0.50]))
    (other,) = [t for t in flipped.terms if t.kind == effects.DIFFERENCE]
    assert other.label == "i0 instead of i1" and other.estimate < 0


def test_a_choice_among_several_is_set_against_each_alternative(store: Store):
    # one-hot: every notebook uses exactly one of k1, k2, k3, and the latest uses k3
    st = _ledger(
        store,
        "k1 k2 k3",
        [
            ("1", "n1", 0.50, "100"),
            ("2", "n2", 0.54, "010"),
            ("3", "n3", 0.60, "001"),
            ("4", "n1", 0.51, ""),
            ("5", "n2", 0.53, ""),
            ("6", "n3", 0.61, ""),
        ],
    )
    fit = effects.lb_fit(st)
    assert [t.label for t in fit.terms] == ["k3 instead of k1", "k3 instead of k2"]
    assert [t.estimate for t in fit.terms] == pytest.approx([0.605 - 0.505, 0.605 - 0.535])
    assert all(t.kind == effects.DIFFERENCE and t.verdict == "helps" for t in fit.terms)


def test_ideas_in_every_row_or_in_none_cannot_be_measured(store: Store):
    st = _ledger(
        store,
        "base-fix never-tried x",
        [("1", "n1", 0.50, "100"), ("2", "n2", 0.60, "101"), ("3", "n3", 0.55, "100")],
    )
    fit = effects.lb_fit(st)
    terms = _terms(fit)
    always, never = terms["base-fix"], terms["never-tried"]
    assert always.kind == effects.ALWAYS and always.note == "in every one of the submissions"
    assert never.kind == effects.NEVER and never.note == "in none of the submissions"
    assert (always.n_with, always.n_without, never.n_with, never.n_without) == (3, 0, 0, 3)
    assert always.estimate is None and always.verdict == "" and never.estimate is None
    assert [t.label for t in fit.measured()] == ["x"]
    assert [t.kind for t in fit.terms] == [effects.EFFECT, effects.ALWAYS, effects.NEVER]


def test_an_idea_that_comes_with_either_of_two_others_is_tangled(store: Store):
    # c comes with a or with b, and a and b never come together: c = a + b, so c cannot be told apart
    # from them, but a instead of b can be measured
    truth = {"a": 0.04, "b": 0.01, "c": 0.02}
    patterns = ["000", "011", "101", "000", "011", "101"]
    subs = [
        (str(k), f"nb{k}", 0.5 + sum(truth[i] for i, code in zip("abc", p, strict=True) if code == "1"), p)
        for k, p in enumerate(patterns)
    ]
    fit = effects.lb_fit(_ledger(store, "a b c", subs))
    diff, tangled = fit.terms
    assert (diff.kind, diff.label) == (effects.DIFFERENCE, "a instead of b")  # the latest has a
    assert diff.estimate == pytest.approx(truth["a"] - truth["b"], abs=1e-12)
    assert tangled.kind == effects.TANGLED and tangled.ideas == ["c"]
    assert tangled.note == "cannot be separated from a, b"
    assert tangled.estimate is None and tangled.ci_low is None and tangled.verdict == ""
    assert (tangled.n_with, tangled.n_without) == (4, 2)


def test_several_independent_groups_at_once(store: Store):
    # a/b/c tangled as above; p/q and r/s two complementary pairs; t and u always together; x on its own
    names = "a b c p q r s t u x"
    patterns = [
        "0001010000",
        "1010110001",
        "0111001001",
        "0000101111",
        "1011001110",
        "0110110110",
        "0001010001",
        "1011001110",
        "0110110001",
        "1010101000",
    ]
    truth = dict(zip(names.split(), [0.03, -0.01, 0.02, 0.05, 0.0, 0.04, 0.0, -0.02, 0.0, 0.01], strict=True))
    subs = []
    for k, p in enumerate(patterns):
        score = 0.5 + sum(truth[i] for i, code in zip(names.split(), p, strict=True) if code == "1")
        subs.append((str(k), f"nb{k}", score, p))
    fit = effects.lb_fit(_ledger(store, names, subs))
    terms = _terms(fit)
    assert {label: t.kind for label, t in terms.items()} == {
        "a instead of b": effects.DIFFERENCE,
        "c": effects.TANGLED,
        "q instead of p": effects.DIFFERENCE,
        "s instead of r": effects.DIFFERENCE,
        "t + u": effects.BUNDLE,
        "x": effects.EFFECT,
    }
    assert terms["a instead of b"].estimate == pytest.approx(0.04, abs=1e-12)
    assert terms["q instead of p"].estimate == pytest.approx(-0.05, abs=1e-12)
    assert terms["s instead of r"].estimate == pytest.approx(-0.04, abs=1e-12)
    assert terms["t + u"].estimate == pytest.approx(-0.02, abs=1e-12)
    assert terms["x"].estimate == pytest.approx(0.01, abs=1e-12)
    assert terms["c"].note == "cannot be separated from a, b"
    # each group's part of a breakdown is its own step, never lumped with another group's
    w = effects.waterfall(fit, "3", "1")
    assert {s.label: s.delta for s in w.steps} == pytest.approx(
        {"+ (a + c) (not separable)": 0.05, "r instead of s": 0.04, "− (t + u)": 0.02}
    )
    assert [s.sign for s in w.steps] == [0, 0, -1]
    assert sum(s.delta for s in w.steps) == pytest.approx(w.end - w.start)


# ----- missing, partial and stale verdicts ---------------------------------------------------------


def test_an_idea_without_a_verdict_on_every_submission_is_left_out(store: Store):
    st = _ledger(
        store,
        "a gap flat unseen",
        [
            ("1", "n1", 0.50, "001-"),
            ("2", "n2", 0.55, "1---"),
            ("3", "n3", 0.60, "111-"),
            ("4", "n4", 0.52, "0p1-"),
            ("5", "n5", 0.66, "101-"),
        ],
    )
    fit = effects.lb_fit(st)
    terms = _terms(fit)
    assert list(terms) == ["a", "flat", "gap"]  # in ledger order; `unseen` has no verdict anywhere
    assert terms["a"].kind == effects.EFFECT and all(r.used for r in fit.rows)
    gap, flat = terms["gap"], terms["flat"]
    assert gap.kind == flat.kind == effects.EXCLUDED
    assert gap.note == (
        "no present/absent verdict on 2 of 5 submissions (2, 4); it varies among the others, so leaving it "
        "out may bias their effects; `pseudorca audit --submitted` fills the gaps; `pseudorca verdict NB gap "
        "present|absent` settles the inconclusive ones"  # 2 has none, 4 a partial one
    )
    assert (gap.n_with, gap.n_without) == (1, 2)
    assert flat.note == (  # the same on all the others: no bias note
        "no present/absent verdict on 1 of 5 submissions (2); `pseudorca audit --submitted` fills the gaps"
    )
    assert (flat.n_with, flat.n_without) == (4, 0) and flat.estimate is None


def test_partial_unknown_and_conflicting_verdicts_count_as_missing(store: Store):
    st = _ledger(
        store,
        "a b",
        [("1", "n1", 0.5, "00"), ("2", "n2", 0.6, "1p"), ("3", "n3", 0.7, "1u"), ("4", "n4", 0.8, "1x")],
    )
    fit = effects.lb_fit(st)
    assert [fit.rows[k].cells["b"].status for k in range(4)] == ["absent", "partial", "unknown", "conflict"]
    assert [fit.rows[k].cells["b"].value for k in range(4)] == [0, None, None, None]
    b = _terms(fit)["b"]
    assert (
        b.kind == effects.EXCLUDED
        and b.note
        == "no present/absent verdict on 3 of 4 submissions (2, 3, 4); `pseudorca verdict NB b present|absent` "
        "settles the inconclusive ones"  # another audit would say the same
    )
    asked = effects.lb_fit(st, ideas=["b", "a"])
    assert [r.why_not for r in asked.rows] == [
        "",
        "not present/absent: b (partial)",
        "not present/absent: b (unknown)",
        "not present/absent: b (conflict)",
    ]
    assert asked._model is None and asked.notes == [
        "nothing to compare: it needs at least 2 submissions with a score and a registered notebook, and a "
        "present/absent verdict for each idea asked for"
    ]


def test_asking_for_ideas_models_exactly_those_and_drops_the_rows_without_a_verdict(store: Store):
    st = _ledger(
        store,
        "a gap other",
        [
            ("1", "n1", 0.50, "000"),
            ("2", "n2", 0.55, "1-1"),
            ("3", "n3", 0.60, "11-"),
            ("4", "n4", 0.52, "0p0"),
            ("5", "n5", 0.66, "101"),
            ("6", "n3", 0.62, ""),
        ],
    )
    fit = effects.lb_fit(st, ideas=["gap", "a", "gap"])
    assert [r.why_not for r in fit.rows] == [
        "",
        "not present/absent: gap (no verdict)",
        "",
        "not present/absent: gap (partial)",
        "",
        "",
    ]
    assert [r.fitted is None for r in fit.rows] == [False, True, False, True, False, False]
    assert sorted(t.label for t in fit.terms) == ["a", "gap"]  # `other` is neither modelled nor listed
    x = _design(np.array([[0, 0], [1, 1], [1, 0], [1, 1]]))  # rows 1, 3, 5, 6: [a, gap]
    beta = np.linalg.lstsq(x, np.array([0.50, 0.60, 0.66, 0.62]), rcond=None)[0]
    assert _terms(fit)["gap"].estimate == pytest.approx(beta[2])
    # the resubmission SD comes from the submissions in the model only (n3 twice; not n2, n4)
    assert fit.resubmission is not None and (fit.resubmission.n_groups, fit.resubmission.n_subs) == (1, 2)
    with pytest.raises(LookupError, match="no idea matches 'nope'"):
        effects.lb_fit(st, ideas=["a", "nope"])
    assert effects.resolve_ideas(st, ["ga", "a", "gap"]) == ["gap", "a"]
    with pytest.raises(LookupError, match="no idea matches"):
        effects.resolve_ideas(st, ["zzz"])


def test_stale_verdicts_are_used_and_counted(store: Store):
    subs = [("1", "n1", 0.5, "00"), ("2", "n2", 0.6, "10"), ("3", "n3", 0.58, "11"), ("4", "n1", 0.51, "")]
    _ledger(store, "a b", subs)
    store.append("idea.update", {"id": "b", "description": "now with a fallback"})  # b's verdicts are old
    fit = effects.lb_fit(store.state())
    assert [r.cells["b"].stale for r in fit.rows] == [True] * 4 and not fit.rows[0].cells["a"].stale
    assert _terms(fit)["b"].kind == effects.EFFECT and _terms(fit)["b"].estimate == pytest.approx(-0.02)
    assert fit.notes == [  # counted once per notebook (1 and 4 share n1), and human ones need `verdict`
        "3 of the verdicts used are stale (made for an older version of the idea); 3 are human verdicts: "
        "record new ones with `pseudorca verdict`"
    ]


def test_changes_and_shown_ideas(store: Store):
    st = _ledger(store, "a b c d", [("1", "n1", 0.5, "10p-"), ("2", "n2", 0.6, "011-"), ("3", None, 0.7, "")])
    rows = effects.submission_rows(st)
    assert effects.changes(None, rows[0]) == []
    assert effects.changes(rows[0], rows[1]) == ["−a", "+b"]  # c went from partial to present: not a change
    assert effects.changes(rows[1], rows[2]) == []  # no verdicts on the third row
    assert effects.shown_ideas(rows) == ["a", "b", "c"] and effects.shown_ideas([]) == []


# ----- which submissions are used ------------------------------------------------------------------


def test_which_submissions_are_used_and_in_what_order(store: Store):
    st = _ledger(store, "a", [("7", "n1", 0.5, "0"), ("8", "n2", 0.6, "1")])
    stray = util.sha256_bytes(b"a notebook nobody registered")
    store.append("sub.upsert", {"id": "10", "nb": stray, "public_score": 0.7, "date": "2026-04-01"})
    store.append("sub.upsert", {"id": "9", "nb": None, "public_score": 0.8, "date": "2026-04-01"})
    nb1 = st.resolve_nb("n1").id
    store.append(
        "sub.upsert", {"id": "11", "nb": nb1, "public_score": None, "date": "2026-04-02T10:00:00+00:00"}
    )
    store.append(
        "sub.upsert", {"id": "12", "nb": nb1, "public_score": 0.52, "date": "2026-04-02T18:00:00+09:00"}
    )
    fit = effects.lb_fit(store.state())
    assert [r.key for r in fit.rows] == ["7", "8", "9", "10", "12", "11"]  # by instant, then numeric id
    why = {r.key: r.why_not for r in fit.rows}
    assert why == {
        "7": "",
        "8": "",
        "9": "no notebook (link one with `pseudorca sub link`)",
        "10": "its notebook is not registered",
        "11": "no public score",
        "12": "",
    }
    assert [r.key for r in fit.used] == ["7", "8", "12"]
    unused = {r.key: r for r in fit.rows if not r.used}
    assert all(r.fitted is None for r in unused.values()) and unused["11"].y is None
    assert unused["10"].nb_label == stray[:10] and unused["9"].nb_label == "" and unused["9"].nb is None
    assert all(c.status is None for c in unused["10"].cells.values())


def test_private_scores(store: Store):
    _ideas(store, "a")
    st = store.state()
    rev = st.ideas["a"].rev
    for k, (code, public, private) in enumerate([("0", 0.5, 0.40), ("1", 0.6, 0.47), ("1", 0.62, None)]):
        nb = _nb(store, f"n{k}")
        _record(store, nb, "a", rev, code)
        data = {"id": str(k), "nb": nb, "public_score": public, "private_score": private, "date": _day(k)}
        store.append("sub.upsert", data)
    fit = effects.lb_fit(store.state(), private=True)
    assert fit.target == "private score" and fit.resubmission is None
    assert [r.y for r in fit.rows] == [0.40, 0.47, None] and fit.rows[2].why_not == "no private score"
    assert _terms(fit)["a"].estimate == pytest.approx(0.07)


def test_several_competitions_need_a_choice(store: Store):
    st = _ledger(store, "a", [("1", "n1", 0.5, "0"), ("2", "n2", 0.6, "1"), ("3", "n1", 0.4, "")])
    for sid, comp in (("1", "comp-a"), ("2", "comp-b")):
        store.append("sub.upsert", {"id": sid, "competition": comp})  # the notebooks and scores stay
    data = {
        "id": "4",
        "nb": st.resolve_nb("n2").id,
        "public_score": 0.62,
        "date": _day(4),
        "competition": "comp-a",
    }
    store.append("sub.upsert", data)
    st = store.state()
    with pytest.raises(
        ValueError, match=r"from 2 competitions \(comp-a, comp-b\); choose one with --competition"
    ):
        effects.lb_fit(st)
    fit = effects.lb_fit(st, competition="comp-a")
    assert [r.key for r in fit.used] == ["1", "3", "4"]  # 3 has no competition: kept
    assert _terms(fit)["a"].estimate == pytest.approx(0.62 - 0.45)
    assert [r.key for r in effects.lb_fit(st, competition="comp-b").used] == ["2", "3"]


# ----- noise ---------------------------------------------------------------------------------------


def _two_groups(store: Store) -> State:
    """Three submissions without the idea and two with it."""
    return _ledger(
        store,
        "a",
        [
            ("1", "n1", 0.50, "0"),
            ("2", "n2", 0.53, "0"),
            ("3", "n3", 0.51, "0"),
            ("4", "n4", 0.58, "1"),
            ("5", "n5", 0.61, "1"),
        ],
    )


def test_residual_noise_uses_students_t(store: Store):
    fit = effects.lb_fit(_two_groups(store))
    term = _terms(fit)["a"]
    rss = sum((v - 0.513333333) ** 2 for v in (0.50, 0.53, 0.51)) + sum(
        (v - 0.595) ** 2 for v in (0.58, 0.61)
    )
    sigma = math.sqrt(rss / 3)
    se = sigma * math.sqrt(1 / 3 + 1 / 2)
    assert (fit.noise_from, fit.df) == ("residuals", 3) and fit.sigma == pytest.approx(sigma, rel=1e-6)
    assert term.estimate == pytest.approx(0.595 - 0.51333333333)
    assert term.se == pytest.approx(se, rel=1e-6)
    t = 3.182446305284263  # t(0.975, 3)
    assert (term.ci_low, term.ci_high) == pytest.approx(
        (term.estimate - t * se, term.estimate + t * se), rel=1e-6
    )
    assert term.verdict == "helps"
    assert fit.resubmission is None  # no notebook was submitted twice


def test_a_known_noise_sd_uses_the_normal_distribution(store: Store):
    fit = effects.lb_fit(_two_groups(store), noise=0.01, confidence=0.9)
    term = _terms(fit)["a"]
    se = 0.01 * math.sqrt(1 / 3 + 1 / 2)
    z = statistics.NormalDist().inv_cdf(0.95)
    assert (fit.noise_from, fit.sigma, fit.df) == ("given", 0.01, math.inf)
    assert term.se == pytest.approx(se) and term.ci_high - term.estimate == pytest.approx(z * se)
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="noise SD must be a positive number"):
            effects.lb_fit(_two_groups_state(), noise=bad)
    for conf in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            effects.lb_fit(_two_groups_state(), confidence=conf)
    # the largest confidence below 1: 1 − α/2 rounds to 1, so the quantile comes from the lower tail
    sure = _terms(effects.lb_fit(_two_groups_state(), confidence=1 - 2**-53))["i0"]
    assert sure.ci_low < -1e3 < 1e3 < sure.ci_high < math.inf and sure.verdict == "within noise"


def _two_groups_state() -> State:
    return _state(np.array([[0], [0], [0], [1], [1]]), [0.50, 0.53, 0.51, 0.58, 0.61])


def test_no_noise_estimate_when_the_model_has_as_many_unknowns_as_submissions(store: Store):
    st = _ledger(store, "a b", [("1", "n1", 0.50, "00"), ("2", "n2", 0.56, "10"), ("3", "n3", 0.54, "01")])
    fit = effects.lb_fit(st)
    assert fit.noise_from == "" and fit.sigma is None and fit.df is None
    assert [(t.label, t.verdict, t.se, t.ci_low) for t in fit.terms] == [
        ("a", "unknown", None, None),
        ("b", "unknown", None, None),
    ]
    assert [t.estimate for t in fit.terms] == pytest.approx([0.06, 0.04])
    assert fit.notes == [
        "no noise estimate: the model has as many unknowns as submissions. Resubmit a notebook, or give a known "
        "LB noise SD (`pseudorca effects --noise SD`, or `noise` under [effects] in config.toml)"
    ]
    given = effects.lb_fit(st, noise=0.02)
    assert given.notes == [] and [t.verdict for t in given.terms] == ["helps", "within noise"]


def test_the_residual_sd_is_compared_with_resubmissions(store: Store):
    # n1 resubmitted three times scores within 0.001; the notebooks score far apart for reasons no idea
    # describes, so the residual SD is many times the resubmission SD
    subs = [
        ("1", "n1", 0.500, "0"),
        ("2", "n1", 0.501, ""),
        ("3", "n1", 0.499, ""),
        ("4", "n2", 0.60, "1"),
        ("5", "n3", 0.50, "1"),
        ("6", "n4", 0.60, "0"),
        ("7", "n5", 0.40, "0"),
    ]
    fit = effects.lb_fit(_ledger(store, "a", subs))
    pure = fit.resubmission
    assert pure is not None and (pure.n_groups, pure.n_subs) == (1, 3) and pure.sd == pytest.approx(0.001)
    assert fit.df == 5 and fit.sigma > 2 * pure.sd
    # lack of fit: F = (SS_lof / 3) / (SS_pure / 2), with SS_pure from n1's three scores
    ss_pure = 2 * 0.001**2
    ss_lof = fit._model.rss - ss_pure
    assert fit.lof_p == pytest.approx(stats.f_sf((ss_lof / 3) / (ss_pure / 2), 3, 2)) and fit.lof_p < 0.001
    assert fit.notes == [
        "submissions of different notebooks stray from the model more than resubmissions of one notebook stray "
        f"from each other (lack-of-fit F test, p = {fit.lof_p:.2g}): ideas may interact, or a change no idea "
        "describes matters"
    ]


def test_no_resubmission_note_when_the_residuals_match_it(store: Store):
    subs = [
        ("1", "n1", 0.500, "0"),
        ("2", "n1", 0.510, ""),
        ("3", "n1", 0.490, ""),
        ("4", "n2", 0.605, "1"),
        ("5", "n2", 0.595, ""),
        ("6", "n3", 0.51, "0"),
    ]
    fit = effects.lb_fit(_ledger(store, "a", subs))
    assert fit.resubmission is not None and fit.sigma <= 2 * fit.resubmission.sd
    assert fit.notes == []


# ----- the breakdown -------------------------------------------------------------------------------


def _three_ideas(store: Store) -> State:
    truth = {"a": 0.05, "b": -0.03, "c": 0.01}
    patterns = ["000", "100", "010", "001", "110", "011", "101", "111", "000"]
    rng = np.random.default_rng(3)
    subs = []
    for k, p in enumerate(patterns):
        score = (
            0.5
            + sum(truth[i] for i, code in zip("abc", p, strict=True) if code == "1")
            + rng.normal(0, 0.004)
        )
        subs.append((str(k + 1), f"nb{k % 8}", round(float(score), 5), p))
    return _ledger(store, "a b c", subs)


def test_the_breakdown_adds_up_and_is_sorted_by_contribution(store: Store):
    st = _three_ideas(store)
    fit = effects.lb_fit(st)
    w = effects.waterfall(fit, "1", "8")
    rows = {r.key: r for r in fit.rows}
    assert (w.base, w.target, w.start, w.end) == (rows["1"], rows["8"], rows["1"].fitted, rows["8"].fitted)
    assert [s.label for s in w.steps] == ["+ a", "+ c", "+ b"] and [s.sign for s in w.steps] == [1, 1, 1]
    assert sum(s.delta for s in w.steps) == pytest.approx(w.end - w.start, abs=1e-12)
    terms = _terms(fit)
    for s in w.steps:
        t = terms[s.ideas[0]]
        assert (s.delta, s.se, s.ci_low, s.ci_high) == pytest.approx((t.estimate, t.se, t.ci_low, t.ci_high))
    back = effects.waterfall(fit, "8", "1")
    assert [s.label for s in back.steps] == ["− b", "− c", "− a"] and [s.sign for s in back.steps] == [-1] * 3
    assert [s.delta for s in back.steps] == pytest.approx([-s.delta for s in reversed(w.steps)])
    lower = effects.waterfall(effects.lb_fit(st, higher_is_better=False), "1", "8")
    assert [s.label for s in lower.steps] == ["+ b", "+ c", "+ a"]  # best (lowest) first
    same = effects.waterfall(fit, "1", "9")  # the same ideas: no step at all
    assert same.steps == [] and same.start == pytest.approx(same.end)


def test_a_complementary_swap_is_one_step(store: Store):
    st = _ledger(
        store,
        "old new x",
        [
            ("1", "n1", 0.50, "100"),
            ("2", "n2", 0.56, "011"),
            ("3", "n3", 0.55, "010"),
            ("4", "n4", 0.52, "101"),
            ("5", "n3", 0.57, ""),
        ],
    )
    fit = effects.lb_fit(st)
    w = effects.waterfall(fit, "1", "2")
    assert [(s.label, s.sign) for s in w.steps] == [("new instead of old", 0), ("+ x", 1)]
    swap = w.steps[0]
    assert swap.ideas == ["new", "old"] and swap.delta == pytest.approx(0.05142857)
    diff = _terms(fit)["new instead of old"]
    assert (swap.delta, swap.se, swap.ci_low) == pytest.approx((diff.estimate, diff.se, diff.ci_low))
    assert sum(s.delta for s in w.steps) == pytest.approx(w.end - w.start)
    back = effects.waterfall(fit, "2", "1")
    assert {s.label for s in back.steps} == {"old instead of new", "− x"}


def test_the_breakdown_needs_rows_in_the_model(store: Store):
    st = _ledger(store, "a", [("1", "n1", 0.5, "0"), ("2", "n2", 0.6, "1"), ("3", None, 0.7, "")])
    fit = effects.lb_fit(st)
    with pytest.raises(LookupError, match=r"^3 is not in the model \(no notebook \(link one with"):
        effects.waterfall(fit, "1", "3")
    with pytest.raises(LookupError, match=r"^nope is not in the model \(no such row\)"):
        effects.waterfall(fit, "nope", "2")
    lonely = effects.lb_fit(st, ideas=["a"], private=True)  # no private scores: no row is in the model
    with pytest.raises(LookupError, match=r"^1 is not in the model \(no private score\)"):
        effects.waterfall(lonely, "1", "2")
    one = _state(np.array([[1]]), [0.5])
    single = effects.lb_fit(one)
    assert single._model is None and len(single.used) == 1
    with pytest.raises(LookupError, match="there is no model to break down"):
        effects.waterfall(single, "0", "0")
    assert effects.pick_row(fit.rows, "2").key == "2"
    with pytest.raises(LookupError, match="no submission matches '77'"):
        effects.pick_row(fit.rows, "77")


# ----- local runs ----------------------------------------------------------------------------------


def _scores(rows: Sequence[tuple[str, Any]], metric: str = "score") -> bytes:
    """A CSV results table with one (item, value) per line."""
    return (f"item,{metric}\n" + "".join(f"{item},{value}\n" for item, value in rows)).encode()


def _levels(
    shift: float, draws: int, seed: int, *, items: int = 30, noise: float = 0.05, first: int = 0
) -> bytes:
    """Items q000, q001... at levels 0.2, 0.22... (the same in every run), plus `shift`, plus noise per draw."""
    rng = np.random.default_rng(seed)
    lines = ["item,draw,score"]
    for i in range(first, first + items):
        for d in range(draws):
            lines.append(f"q{i:03d},{d},{0.2 + 0.02 * i + shift + rng.normal(0, noise):.6f}")
    return ("\n".join(lines) + "\n").encode()


def _run(store: Store, data: bytes, nb: str | None, *, label: str = "", protocol: str = "") -> str:
    sha = store.put_blob(data)
    run_id = util.new_ulid()
    store.append(
        "run.add",
        {"id": run_id, "results": sha, "format": "csv", "nb": nb, "label": label, "protocol": protocol},
    )
    return run_id


def _table_of(store: Store) -> Callable[[Run], Any]:
    return lambda run: load_table(store.get_blob(run.results), run.format)


def _notebooks(store: Store, ideas: str, codes: dict[str, str]) -> dict[str, str]:
    """Notebooks by name with their verdict codes (one per idea); returns {name: id}."""
    revs = _ideas(store, ideas)
    out = {}
    for name, row in codes.items():
        out[name] = _nb(store, name)
        for iid, code in zip(revs, row, strict=True):
            _record(store, out[name], iid, revs[iid], code)
    return out


def _arms(*tables: bytes) -> list[dict[str, np.ndarray]]:
    return [per_item(load_table(t, "csv"), "score") for t in tables]


def test_two_runs_give_compares_estimate_interval_and_verdict(store: Store):
    nbs = _notebooks(store, "retry", {"base": "0", "new": "1"})
    a = _levels(0.0, 3, seed=1)
    b = _levels(0.03, 2, seed=2, items=32)  # ragged: two draws, and two items the other run lacks
    head, *lines = b.decode().splitlines()
    b = "\n".join([head, *reversed(lines)]).encode() + b"\n"  # in another order: paired by item, not by line
    _run(store, a, nbs["base"], label="base", protocol="p30")
    _run(store, b, nbs["new"], label="new", protocol="p30")
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=3000, seed=4)
    (term,) = fit.measured()
    res = stats.compare(*_arms(a, b), resamples=3000, seed=4)
    assert res.verdict == "B better" and res.only_b == ["q030", "q031"]
    assert term.estimate == pytest.approx(res.diff, abs=1e-12)
    assert (term.ci_low, term.ci_high) == pytest.approx((res.ci_low, res.ci_high), abs=1e-12)
    assert term.se == pytest.approx(res.se, rel=1e-9)
    assert term.verdict == "helps" and term.label == "retry" and (term.n_with, term.n_without) == (1, 1)
    # paired by item: the item levels (SD 0.18 over items) would swamp a 0.03 shift without the pairing
    assert term.se < 0.015 < statistics.stdev(0.2 + 0.02 * i for i in range(30)) / math.sqrt(30)
    assert [r.y for r in fit.rows] == pytest.approx([res.mean_a, res.mean_b])
    assert [r.fitted for r in fit.rows] == pytest.approx([res.mean_a, res.mean_b])
    assert (fit.n_items, fit.n_items_any) == (30, 32)
    assert fit.notes == ["only the 30 items every run has are used (of 32)"]
    assert (fit.source, fit.target, fit.protocol, fit.noise_from) == ("local", "score", "p30", "bootstrap")
    assert (fit.resamples, fit.seed, fit.sigma, fit.df) == (3000, 4, None, None)


def test_a_winner_on_the_runs_also_needs_the_sign_flip_test(store: Store):
    # five items favour the idea by 0.2 and one goes against it by 0.02: every bootstrap mean is positive,
    # but 4 of the 64 sign flips are as extreme as the data (p = 0.0625), so `compare` names no winner
    nbs = _notebooks(store, "x", {"without": "0", "with": "1"})
    a = _scores([(f"q{i}", 0.5) for i in range(6)])
    b = _scores([*((f"q{i}", 0.7) for i in range(5)), ("q5", 0.48)])
    _run(store, a, nbs["without"])
    _run(store, b, nbs["with"])
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=2000)
    (term,) = fit.measured()
    res = stats.compare(*_arms(a, b), resamples=2000)
    assert res.verdict == "within noise" and res.ci_low > 0 and res.p_value == pytest.approx(4 / 64)
    assert term.ci_low > 0 and term.verdict == "within noise"
    assert term.note == "the interval excludes 0, but the sign-flip p = 0.0625 is not below 0.05"
    (loose,) = effects.run_fits(store.state(), _table_of(store), resamples=2000, confidence=0.9)
    assert loose.measured()[0].verdict == "helps" and loose.measured()[0].note == ""
    assert stats.compare(*_arms(a, b), resamples=2000, confidence=0.9).verdict == "B better"


def test_fewer_items_than_the_sign_flip_test_needs(store: Store):
    nbs = _notebooks(store, "x", {"without": "0", "with": "1"})
    _run(store, _scores([(f"q{i}", 0.5) for i in range(5)]), nbs["without"])
    _run(store, _scores([(f"q{i}", 0.6 + i / 100) for i in range(5)]), nbs["with"])
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=500)
    (term,) = fit.measured()
    assert term.ci_low > 0 and term.verdict == effects.TOO_FEW == stats.TOO_FEW


def _factorial(store: Store, *, noise: float, draws: int = 2) -> tuple[dict[str, str], list[str]]:
    """Four notebooks, without and with a (+0.05) and b (−0.03), each with a run on the same 30 items."""
    nbs = _notebooks(store, "a b", {"n00": "00", "n10": "10", "n01": "01", "n11": "11"})
    shifts = {"n00": 0.0, "n10": 0.05, "n01": -0.03, "n11": 0.02}
    runs = [
        _run(store, _levels(shifts[n], draws, seed=k, noise=noise), nbs[n], label=n)
        for k, n in enumerate(shifts)
    ]
    return nbs, runs


def test_runs_are_fitted_by_least_squares_on_their_means(store: Store):
    _factorial(store, noise=0.05)
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=2000, seed=1)
    y = np.array([r.y for r in fit.used])
    beta = np.linalg.lstsq(_design(np.array([[0, 0], [1, 0], [0, 1], [1, 1]])), y, rcond=None)[0]
    assert [t.label for t in fit.terms] == ["a", "b"]
    assert [t.estimate for t in fit.terms] == pytest.approx([beta[1], beta[2]], abs=1e-12)
    assert [t.verdict for t in fit.terms] == ["helps", "hurts"]
    assert [r.fitted for r in fit.used] == pytest.approx(
        list(_design(np.array([[0, 0], [1, 0], [0, 1], [1, 1]])) @ beta)
    )
    for t in fit.terms:
        assert t.ci_low < t.estimate < t.ci_high and t.se > 0
    lower = effects.run_fits(store.state(), _table_of(store), resamples=2000, seed=1, higher_is_better=False)
    assert [(t.label, t.verdict) for t in lower[0].terms] == [("b", "helps"), ("a", "hurts")]


def test_the_bootstrap_keeps_the_runs_paired(store: Store):
    # every run is the same item levels plus its own shift, with no other noise: when every resample draws
    # the same items for every run, each contrast comes out the same in every resample
    _factorial(store, noise=0.0, draws=1)
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=500)
    for t, value in zip(fit.terms, [0.05, -0.03], strict=True):
        assert t.estimate == pytest.approx(value, abs=1e-6)
        assert t.se == pytest.approx(0, abs=1e-12)
        assert (t.ci_low, t.ci_high) == pytest.approx((value, value), abs=1e-6)


def test_each_protocol_is_fitted_on_its_own(store: Store):
    nbs = _notebooks(store, "a", {"n0": "0", "n1": "1"})
    _run(store, _levels(0.0, 1, seed=1), nbs["n0"], protocol="dev")
    _run(store, _levels(0.05, 1, seed=2), nbs["n1"], protocol="dev")
    _run(store, _levels(0.0, 1, seed=3), nbs["n0"], protocol="full")
    _run(store, _levels(-0.05, 1, seed=4), nbs["n1"], protocol="full")
    _run(store, _levels(0.0, 1, seed=5), nbs["n0"])
    fits = effects.run_fits(store.state(), _table_of(store), resamples=500)
    assert [f.protocol for f in fits] == ["", "dev", "full"]
    assert fits[0]._model is None and fits[0].notes == [
        "nothing to compare: it needs at least 2 runs of registered notebooks with this metric"
    ]
    assert fits[1].measured()[0].estimate == pytest.approx(0.05, abs=0.03)
    assert fits[2].measured()[0].estimate == pytest.approx(-0.05, abs=0.03)
    (dev,) = effects.run_fits(store.state(), _table_of(store), protocol="dev", resamples=500)
    assert dev.protocol == "dev" and len(dev.rows) == 2
    with pytest.raises(
        LookupError, match=r"no runs have the protocol 'nope' \(protocols: no protocol, 'dev', 'full'\)"
    ):
        effects.run_fits(store.state(), _table_of(store), protocol="nope")
    with pytest.raises(ValueError, match="resamples must be at least 100"):
        effects.run_fits(store.state(), _table_of(store), resamples=99)
    assert effects.run_fits(State(), _table_of(store)) == []


def test_runs_that_cannot_be_used_say_why(store: Store):
    nbs = _notebooks(store, "a", {"n0": "0", "n1": "1"})
    first = _levels(0.0, 2, seed=1)
    good = [_run(store, first, nbs["n0"]), _run(store, _levels(0.05, 2, seed=2), nbs["n1"])]
    stray = util.sha256_bytes(b"never registered")
    why = {
        _run(
            store, _levels(0.0, 1, seed=3), None
        ): "no notebook (add its table again with `pseudorca run add FILE --nb NB`: that copy is used)",
        _run(store, _levels(0.0, 1, seed=4), stray): "its notebook is not registered",
        _run(store, _scores([("q000", 1)], "acc"), nbs["n1"]): "no metric 'score' (metrics: acc)",
        _run(store, first, nbs["n1"]): f"repeats the results table of run {good[0][:10]}",
        _run(
            store, _scores([("q000", "inf"), ("q001", 0.5)]), nbs["n1"]
        ): "item 'q000' has an infinite 'score'",
        _run(store, _scores([("q000", "NA"), ("q001", "")]), nbs["n1"]): "no values of 'score'",
    }
    failing = {
        _run(store, _levels(0.0, 1, seed=5), nbs["n0"]): OSError("the disk is gone"),
        _run(store, _levels(0.0, 1, seed=6), nbs["n0"]): StoreError("its results table is not in this store"),
        _run(store, _levels(0.0, 1, seed=7), nbs["n0"]): ValueError(),
    }
    why.update({rid: str(e) or "ValueError" for rid, e in failing.items()})
    load = _table_of(store)

    def loader(run: Run) -> Any:
        if run.id in failing:
            raise failing[run.id]
        return load(run)

    (fit,) = effects.run_fits(store.state(), loader, resamples=500)
    assert [r.key for r in fit.used] == good
    assert {r.key: r.why_not for r in fit.rows if not r.used} == why
    assert fit.notes == ["1 run(s) repeat another run's results table and are left out"]
    assert fit.measured()[0].estimate == pytest.approx(0.05, abs=0.03)
    assert all(r.fitted is None for r in fit.rows if not r.used)


def test_only_items_every_run_has_are_used(store: Store):
    nbs = _notebooks(store, "a", {"n0": "0", "n1": "1", "n2": "1"})
    _run(store, _levels(0.0, 1, seed=1, items=10), nbs["n0"])
    _run(store, _levels(0.05, 1, seed=2, items=10, first=10), nbs["n1"])  # q010...: nothing in common
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=500)
    assert [r.why_not for r in fit.rows] == ["no item is in every run"] * 2
    assert fit._model is None and fit.notes[0] == "no item is in every run, so they cannot be paired"
    assert (fit.n_items, fit.n_items_any) == (0, 20) and [r.y for r in fit.rows] == [None, None]
    # a third run on q004...q015 shares items with each of them, but no item is in all three
    _run(store, _levels(0.05, 1, seed=3, items=12, first=4), nbs["n2"])
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=500)
    assert [r.why_not for r in fit.rows] == ["no item is in every run"] * 3 and fit.n_items == 0


def test_asking_for_ideas_drops_the_runs_without_a_verdict(store: Store):
    nbs = _notebooks(store, "a b", {"n0": "00", "n1": "1p", "n2": "11"})
    for k, name in enumerate(("n0", "n1", "n2")):
        _run(store, _levels(0.02 * k, 1, seed=k), nbs[name])
    (fit,) = effects.run_fits(store.state(), _table_of(store), ideas=["b"], resamples=500)
    assert [r.why_not for r in fit.rows] == ["", "not present/absent: b (partial)", ""]
    assert [t.label for t in fit.terms] == ["b"]
    (default,) = effects.run_fits(store.state(), _table_of(store), resamples=500)
    assert {t.label: t.kind for t in default.terms} == {"a": effects.EFFECT, "b": effects.EXCLUDED}
    with pytest.raises(LookupError, match="no idea matches 'c'"):
        effects.run_fits(store.state(), _table_of(store), ideas=["c"])


# ----- the command ---------------------------------------------------------------------------------


def _rich(store: Store) -> dict[str, str]:
    """Submissions over two versions of a choice (old, new), an idea x, an idea in all of them, an idea
    with gaps, resubmissions and one unscored submission; runs of two notebooks plus one without a
    notebook. Returns the notebook ids by name."""
    _ledger(
        store,
        "old new x always gap",
        [
            ("101", "base.ipynb", 0.500, "10011"),
            ("102", "base.ipynb", 0.506, ""),
            ("103", "cand.ipynb", 0.548, "01011"),
            ("104", "cand-x.ipynb", 0.561, "0111p"),
            ("105", "cand-x.ipynb", 0.566, ""),
            ("106", "base-x.ipynb", 0.511, "1011-"),
            ("107", "cand.ipynb", None, ""),
        ],
    )
    nbs = {n.name: n.id for n in store.state().notebooks.values()}
    _run(store, _levels(0.0, 2, seed=1), nbs["base.ipynb"], label="base", protocol="p30")
    _run(store, _levels(0.04, 2, seed=2), nbs["cand.ipynb"], label="cand", protocol="p30")
    _run(store, _levels(0.0, 1, seed=3), None, label="loose", protocol="p30")
    return nbs


def test_the_command_shows_the_matrix_the_effects_and_the_breakdown(store: Store):
    _rich(store)
    result = _cli(store, "effects", "--resamples", "1000")
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "Submissions, oldest first (public score)"
    assert " ".join(lines[1].split()) == "SUB DATE NOTEBOOK PUBLIC FITTED 1 2 3 4 5 CHANGES"
    rows = {line.split()[0]: line for line in lines[3:10]}
    # a fitted value gets the scores' decimals and up to 2 more (it is a mean)
    assert " ".join(rows["101"].split()) == "101 2026-03-01 base.ipynb 0.500 0.50175 ✓ ✓ ✗ ✓ ✗"
    assert rows["102"].endswith("resubmits 101") and rows["103"].endswith("+new −old")
    assert rows["104"].split()[5:11] == ["✓", "◐", "✓", "✗", "✓", "+x"]
    assert rows["106"].split()[5:10] == ["✓", "·", "✗", "✓", "✓"]
    # the parts of CHANGES apart, why the row is not used first
    assert rows["107"].endswith("not used: no public score; resubmits 103; +new −old −x")
    assert lines[10] == "ideas: 1 always  2 gap  3 new  4 old  5 x"
    assert lines[11].startswith("✓ present  ◐ partial  ✗ absent  ? unknown  ! methods disagree  · no verdict")
    out = result.output
    assert "What each idea adds to the public score (additive model over 6 submissions)" in out
    head = next(line for line in lines if line.startswith("IDEA "))
    assert head.split() == ["IDEA", "EFFECT", "95%", "CI", "SUBS", "VERDICT", "−0.060", "0", "+0.060", "NOTE"]
    x = next(line for line in lines if line.startswith("x "))
    assert x.split()[:6] == ["x", "+0.012", "[+0.001,", "+0.023]", "3/3", "helps"]
    sketch_at = head.index("−0.060")
    # from the 0 in the middle rightwards: +0.001 rounds onto the 0, but an interval that excludes 0 stays
    # off it
    assert x[sketch_at:] == "          |[*-]"
    old = next(line for line in lines if line.startswith("old instead of new"))
    assert "hurts" in old and old[sketch_at : sketch_at + 21].rstrip().endswith("|")
    assert "neither can be measured alone" in old
    assert "no present/absent verdict on 3 of 6 submissions (104, 105, 106)" in out
    assert (
        "SUBS = submissions with/without it (for “A instead of B”: with A/with B). VERDICT is helps or hurts "
        "when the 95% CI excludes 0." in out
    )
    assert "in every one (cannot be measured here): always" in out
    assert "noise: residual SD 0.00405 with 3 degrees of freedom; t intervals" in out
    assert "resubmitting a notebook moved the score with SD 0.00391 (2 notebook(s), 4 submissions)" in out
    # the local runs: a section of their own
    assert "Local runs (protocol p30), score: 2 of 3 runs used, 30 items in every one" in out
    assert (
        "not used: no notebook (add its table again with `pseudorca run add FILE --nb NB`: that copy is used)"
        in out
    )
    assert "What each idea adds to the score (additive model over 2 runs)" in out
    assert (
        "VERDICT is helps or hurts when the 95% CI excludes 0 and the sign-flip p < 0.05, as in `compare`"
        in out
    )
    assert "in none (cannot be measured here): x" in out
    assert (
        "intervals: two-stage bootstrap over items and draws (1000 resamples, seed 0), the runs paired" in out
    )
    assert (
        "run-to-run SD: not measurable, as no two runs have the same ideas; the intervals assume none" in out
    )
    # the breakdown, from the first submission to the best, at the fitted values' decimals
    tail = lines[lines.index("Breakdown of 105 (best in the model) from 101 (first), by the model") :]
    assert [line.split()[:3] for line in tail[1:]] == [
        ["101", "fitted", "0.50175"],
        ["new", "instead", "of"],
        ["+", "x", "+0.01175"],
        ["=", "105", "fitted"],
        ["not", "in", "the"],
    ]
    assert tail[1].endswith("observed 0.500") and tail[4].endswith("observed 0.566")  # as in the table
    assert tail[2].split()[4:] == ["+0.04875", "[+0.038,", "+0.060]"]
    assert tail[5] == "  not in the model, and different or unknown: gap"  # 101 has it, 105 has it partly


def test_the_json_output(store: Store):
    _rich(store)
    data = json.loads(_cli(store, "effects", "--json", "--resamples", "500", "--seed", "3").stdout)
    assert set(data) == {"lb", "runs", "breakdown"}
    lb = data["lb"]
    fit = effects.lb_fit(store.state())
    assert (lb["source"], lb["target"], lb["confidence"], lb["higher_is_better"]) == (
        "lb",
        "public score",
        0.95,
        True,
    )
    assert lb["noise"]["from"] == "residuals" and lb["noise"]["df"] == 3
    assert lb["noise"]["sd"] == pytest.approx(fit.sigma)
    assert lb["noise"]["resubmission"]["n_groups"] == 2 and lb["items"] is None and lb["seed"] is None
    assert [r["key"] for r in lb["rows"]] == ["101", "102", "103", "104", "105", "106", "107"]
    last = lb["rows"][-1]
    assert (last["used"], last["why_not"], last["y"], last["fitted"]) == (
        False,
        "no public score",
        None,
        None,
    )
    assert lb["rows"][5]["cells"] == {
        "always": {"status": "present", "stale": False, "method": "human"},
        "new": {"status": "absent", "stale": False, "method": "human"},
        "old": {"status": "present", "stale": False, "method": "human"},
        "x": {"status": "present", "stale": False, "method": "human"},
    }  # no "gap": no verdict
    assert [t["label"] for t in lb["terms"]] == [t.label for t in fit.terms]
    for got, t in zip(lb["terms"], fit.terms, strict=True):
        assert set(got) == {
            "ideas",
            "kind",
            "estimate",
            "se",
            "ci_low",
            "ci_high",
            "n_with",
            "n_without",
            "verdict",
            "note",
            "left",
            "right",
            "swapped",
            "label",
            "mde80",
        }
        assert (got["kind"], got["verdict"], got["ideas"]) == (t.kind, t.verdict, t.ideas)
        assert got["estimate"] == pytest.approx(t.estimate) and got["mde80"] == pytest.approx(t.mde80)
    diff = next(t for t in lb["terms"] if t["kind"] == "difference")
    assert (diff["left"], diff["right"]) == (["old"], ["new"])
    (local,) = data["runs"]
    assert (local["source"], local["protocol"], local["resamples"], local["seed"]) == ("local", "p30", 500, 3)
    assert local["items"] == {"used": 30, "any": 30} and local["noise"]["from"] == "bootstrap"
    breakdown = data["breakdown"]
    assert (breakdown["base"], breakdown["target"]) == ("101", "105")
    assert breakdown["observed_base"] == 0.5 and breakdown["observed_target"] == 0.566
    assert sum(s["delta"] for s in breakdown["steps"]) == pytest.approx(breakdown["end"] - breakdown["start"])
    assert [set(s) for s in breakdown["steps"]] == [
        {"label", "ideas", "sign", "delta", "se", "ci_low", "ci_high"}
    ] * 2
    given = json.loads(_cli(store, "effects", "--json", "--no-runs", "--noise", "0.01").stdout)
    assert given["lb"]["noise"]["from"] == "given" and given["lb"]["noise"]["df"] is None  # inf: not JSON
    assert given["runs"] == []


def test_choosing_the_breakdown(store: Store):
    _rich(store)
    picked = _cli(store, "effects", "--no-runs", "--sub", "103", "--base", "106")
    assert picked.exit_code == 0, picked.output
    tail = picked.output[picked.output.index("Breakdown of 103 from 106, by the model") :].splitlines()
    assert [line.split()[:2] for line in tail[1:]] == [
        ["106", "fitted"],
        ["new", "instead"],
        ["−", "x"],
        ["=", "103"],
        ["not", "in"],
    ]
    assert tail[5] == "  not in the model, and different or unknown: gap"  # no verdict on 106's notebook
    to_best = _cli(store, "effects", "--no-runs", "--base", "103").output
    assert "Breakdown of 105 (best in the model) from 103, by the model" in to_best
    from_first = _cli(store, "effects", "--no-runs", "--sub", "106").output
    assert "Breakdown of 106 from 101 (first), by the model" in from_first
    for args, message in (
        (("--sub", "999"), "Error: no submission matches '999' (see `pseudorca lb`)"),
        (("--base", "107"), "Error: 107 is not in the model (no public score)"),
        (("--sub", "10"), "Error: no submission matches '10'"),  # prefixes need 4 characters
    ):
        bad = _cli(store, "effects", "--no-runs", *args)
        assert bad.exit_code == 1 and bad.output.startswith(message) and "Traceback" not in bad.output
    for sid in ("20001", "20002", "30001"):
        store.append("sub.upsert", {"id": sid, "public_score": 0.4, "date": "2026-03-08"})
    ambiguous = _cli(store, "effects", "--no-runs", "--sub", "2000")
    assert (
        ambiguous.exit_code == 1 and "'2000' matches 2 submissions; use more characters" in ambiguous.output
    )
    prefix = _cli(store, "effects", "--no-runs", "--base", "3000")  # a unique prefix: 30001
    assert "Error: 30001 is not in the model (no notebook (link one with" in prefix.output
    same = _cli(store, "effects", "--no-runs", "--base", "105", "--sub", "104")  # the same notebook
    assert "Breakdown of 104 from 105, by the model" in same.output
    assert "(no modelled idea differs)" in same.output and "not in the model" not in same.output


def test_choosing_ideas_rows_and_runs(store: Store):
    _rich(store)
    asked = _cli(store, "effects", "--no-runs", "--idea", "x", "--idea", "ne")  # ids or unique prefixes
    assert asked.exit_code == 0, asked.output
    assert "not used: not present/absent: x (no verdict)" not in asked.output
    assert "What each idea adds to the public score (additive model over 6 submissions)" in asked.output
    terms = asked.output.split("What each idea adds")[1].split("Breakdown")[0]
    assert "gap" not in terms  # only the ideas asked for
    assert "not in the model, and different or unknown: gap, old" in asked.output  # in the breakdown
    assert "old is in exactly the rows without new (counted against it)" in asked.output  # the model's note
    gap = _cli(store, "effects", "--no-runs", "--idea", "gap").output
    assert "104  2026-03-04" in gap and "not used: not present/absent: gap (partial)" in gap
    assert "not used: not present/absent: gap (no verdict)" in gap
    assert "(additive model over 3 submissions)" in gap
    for args, message in (
        (("--idea", "zzz"), "Error: no idea matches 'zzz'\n"),
        (("--protocol", "nope"), "Error: no runs have the protocol 'nope' (protocols: 'p30')\n"),
    ):
        bad = _cli(store, "effects", *args)
        assert (bad.exit_code, bad.output) == (1, message)
    limited = _cli(store, "effects", "-n", "2", "--no-runs").output
    listed = [line.split()[0] for line in limited.splitlines()[3:6]]
    assert (
        listed == ["106", "107", "(the"]
        and "(the latest 2 of 7; -n 0 lists all, and the model uses all)" in limited
    )
    assert "(additive model over 6 submissions)" in limited  # the model still uses all of them
    every = _cli(store, "effects", "-n", "0", "--no-runs").output
    assert "(the latest" not in every and "101  2026-03-01" in every
    assert "Local runs" not in every
    runs_only_some = _cli(store, "effects", "-n", "1", "--resamples", "500").output
    local = runs_only_some[runs_only_some.index("Local runs") :]
    assert "(the latest 1 of 3; -n 0 lists all, and the model uses all)" in local and "(loose)" in local


def test_the_command_on_an_empty_or_unaudited_ledger(store: Store):
    empty = _cli(store, "effects")
    assert empty.exit_code == 0
    assert (
        empty.output
        == "no submissions yet (record scores with `pseudorca sub add` or `pseudorca kaggle sync`)\n"
    )
    data = json.loads(_cli(store, "effects", "--json").stdout)
    assert data["lb"]["rows"] == [] and data["runs"] == [] and data["breakdown"] is None
    assert data["lb"]["notes"] == [
        "nothing to compare: it needs at least 2 submissions with a score and a registered notebook"
    ]
    no_runs = _cli(store, "effects", "--protocol", "p30")  # checked even when there are no runs at all
    assert (no_runs.exit_code, no_runs.output) == (
        1,
        "Error: no runs have the protocol 'p30' (protocols: none)\n",
    )
    one = _cli(store, "effects", "--sub", "1")
    assert one.exit_code == 1 and "no submission matches '1'" in one.output

    _ledger(store, "a", [("1", "n1", 0.5, "-"), ("2", "n2", 0.6, "-")])
    out = _cli(store, "effects").output
    assert "no verdicts on these submissions' notebooks yet: `pseudorca audit --submitted` audits them" in out
    assert "no idea is in some submissions but not all, so there is nothing to compare" in out
    assert out.splitlines()[3].split() == ["1", "2026-03-01", "n1", "0.5", "0.55"]  # the mean keeps its digit
    assert (
        "noise: residual SD 0.0707 with 1 degrees of freedom; t intervals (few degrees of freedom: wide"
        in out
    )


def test_huge_and_tiny_scores_render(store: Store):
    _ledger(
        store,
        "a",
        [
            ("1", "n1", 1.0e30, "0"),
            ("2", "n2", 3.0e30, "1"),
            ("3", "n1", 1.2e30, ""),
            ("4", "n2", 2.8e30, ""),
        ],
    )
    result = _cli(store, "effects")
    assert result.exit_code == 0, result.output
    head = next(line for line in result.output.splitlines() if line.startswith("IDEA "))
    left, zero, right = head.split()[6:9]  # labels too long for the sketch are shortened
    assert re.fullmatch(r"−\d\.\de\+30", left) and zero == "0" and right == "+" + left[1:]
    tiny = Store.init(store.root.parent.parent / "tiny", name="tiny")
    _ledger(
        tiny,
        "a",
        [("1", "n1", 1e-7, "0"), ("2", "n2", 3e-7, "1"), ("3", "n1", 1.2e-7, ""), ("4", "n2", 2.9e-7, "")],
    )
    assert _cli(tiny, "effects").exit_code == 0


# ----- robustness ----------------------------------------------------------------------------------


def test_many_ideas_on_few_submissions(store: Store):
    # 32 ideas on 5 submissions: at most 5 numbers can be told apart, so nearly everything is tangled
    rng = np.random.default_rng(11)
    names = [f"idea{j:02d}" for j in range(32)]
    cells = (rng.random((5, 32)) < 0.5).astype(int)
    cells[:, 0], cells[:, 1] = 1, 0  # one in every submission, one in none
    subs = [
        (str(k + 1), f"nb{k}", round(0.5 + float(rng.normal(0, 0.02)), 4), "".join(map(str, row)))
        for k, row in enumerate(cells)
    ]
    subs.append(("6", "nb0", 0.51, ""))  # a resubmission: one degree of freedom for the noise
    st = _ledger(store, " ".join(names), subs)
    fit = effects.lb_fit(st)
    kinds = {t.kind for t in fit.terms}
    assert effects.TANGLED in kinds and {effects.ALWAYS, effects.NEVER} <= kinds
    assert _terms(fit)["idea00"].kind == effects.ALWAYS and _terms(fit)["idea01"].kind == effects.NEVER
    covered = [i for t in fit.terms for i in t.ideas]
    assert set(covered) == set(names)
    x = _design(np.vstack([cells, cells[:1]]))
    y = np.array([s[2] for s in subs])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for t in fit.measured():
        c = _term_contrast(names, t)
        assert _estimable(x, c) and t.estimate == pytest.approx(float(c @ beta), abs=1e-9)
    for t in fit.terms:
        if t.kind == effects.TANGLED:
            assert not _estimable(x, _term_contrast(names, t))
    assert len(fit.measured()) <= int(np.linalg.matrix_rank(x)) - 1  # never more numbers than the rows hold
    assert fit.df == 1
    w = effects.waterfall(fit, "1", "5")
    assert sum(s.delta for s in w.steps) == pytest.approx(w.end - w.start)
    result = _cli(store, "effects", "--json")
    assert result.exit_code == 0 and len(json.loads(result.stdout)["lb"]["terms"]) == len(fit.terms)
    shown = _cli(store, "effects")
    assert shown.exit_code == 0 and "cannot be separated from" in shown.output


def test_scores_that_are_not_finite_numbers_are_skipped(store: Store):
    st = _ledger(store, "a", [("1", "n1", 0.5, "0"), ("2", "n2", 0.6, "1"), ("3", "n1", 0.52, "")])
    nb = st.resolve_nb("n2").id
    for k, bad in enumerate(("NaN", "inf", "-Infinity", "pending", [0.7])):
        # hand-edited events, as a teammate might push them
        store.append("sub.upsert", {"id": f"9{k}", "nb": nb, "public_score": bad, "date": "2026-04-01"})
    fit = effects.lb_fit(store.state())
    assert [r.key for r in fit.used] == ["1", "2", "3"]
    assert {r.why_not for r in fit.rows if not r.used} == {"no public score"}
    assert _terms(fit)["a"].estimate == pytest.approx(0.09)
    # scores that never went through the event checks (an in-memory ledger) are skipped the same way
    raw = _state(np.array([[0], [1], [0], [1], [1]]), [0.5, 0.6, 0.52, 0.0, 0.0])
    raw.subs["3"].public_score = math.nan
    raw.subs["4"].public_score = math.inf
    rows = effects.submission_rows(raw)
    assert [(r.key, r.y, r.why_not) for r in rows[3:]] == [
        ("3", None, "no public score"),
        ("4", None, "no public score"),
    ]
    assert [r.key for r in effects.lb_fit(raw).used] == ["0", "1", "2"]


def test_a_run_whose_table_is_missing_is_listed_as_not_used(store: Store):
    nbs = _notebooks(store, "a", {"n0": "0", "n1": "1"})
    _run(store, _levels(0.0, 2, seed=1), nbs["n0"], label="base")
    _run(store, _levels(0.05, 2, seed=2), nbs["n1"], label="new")
    gone = _run(store, _levels(0.05, 2, seed=3), nbs["n1"], label="gone")
    store.blob_path(store.state().runs[gone].results).unlink()  # a teammate forgot to push it
    result = _cli(store, "effects", "--resamples", "500")
    assert result.exit_code == 0, result.output
    assert "Local runs (no protocol), score: 2 of 3 runs used, 30 items in every one" in result.output
    line = next(x for x in result.output.splitlines() if "(gone)" in x)
    assert "not used: its results table" in line and line.endswith(
        "pull the latest .pseudorca from your repository"
    )
    acc = _run(store, _scores([("q000", 1)], "acc"), nbs["n0"], label="acc", protocol="other")
    only = _cli(store, "effects", "--protocol", "other").output.splitlines()
    assert only[-2:] == [
        "local runs (protocol other): 0 of 1 usable, nothing to compare: it needs at least 2 runs of "
        "registered notebooks with this metric",
        f"  not used: {acc[:10]} (acc): no metric 'score' (metrics: acc)",
    ]


# ----- review fixes --------------------------------------------------------------------------------


def test_an_exact_fit_gives_exact_effects_but_no_noise_estimate(store: Store):
    # a pure speed-up: the same score with or without it; resubmissions score the same again
    subs = [
        ("1", "base", 0.310, "00"),
        ("2", "base", 0.310, ""),
        ("3", "cache", 0.310, "10"),
        ("4", "prompt", 0.321, "01"),
        ("5", "both", 0.321, "11"),
    ]
    st = _ledger(store, "cache prompt", subs)
    fit = effects.lb_fit(st)
    assert fit.sigma is None and fit.noise_from == "" and fit._model.df == 2 and fit.lof_p == 1.0
    cache, prompt = _terms(fit)["cache"], _terms(fit)["prompt"]
    assert (cache.estimate, cache.ci_low, cache.verdict) == (0.0, None, "unknown")  # 0, not rounding
    assert prompt.estimate == pytest.approx(0.011) and prompt.verdict == "unknown"
    assert fit.notes[0].startswith(effects.ZERO_RESIDUALS)
    given = _terms(effects.lb_fit(st, noise=0.002))
    assert (given["cache"].estimate, given["cache"].verdict) == (0.0, "within noise")
    assert given["prompt"].verdict == "helps"


def test_two_independent_swaps_are_measured_and_broken_down_apart(store: Store):
    subs = [
        ("1", "a-x", 0.500, "1010"),
        ("2", "b-x", 0.531, "0110"),
        ("3", "a-y", 0.519, "1001"),
        ("4", "b-y", 0.552, "0101"),
        ("5", "a-x", 0.503, ""),
        ("6", "b-y", 0.548, ""),
    ]
    fit = effects.lb_fit(_ledger(store, "model-a model-b prompt-x prompt-y", subs))
    terms = _terms(fit)
    model, prompt = terms["model-b instead of model-a"], terms["prompt-y instead of prompt-x"]
    w = effects.waterfall(fit, "1", "4")
    assert {s.label: s.delta for s in w.steps} == pytest.approx(
        {"model-b instead of model-a": model.estimate, "prompt-y instead of prompt-x": prompt.estimate}
    )
    assert sum(s.delta for s in w.steps) == pytest.approx(w.end - w.start) and w.unmodelled == []


def test_a_tangled_note_names_only_the_ideas_it_mixes_with():
    # (a, b, c) in {(0,0,1), (1,0,0), (1,1,1)}: c = 1 − a + b; and p + q = 1 on every row. Both mix with the
    # intercept, but not with each other
    cells = np.array([[*abc, *pq] for abc in ((0, 0, 1), (1, 0, 0), (1, 1, 1)) for pq in ((1, 0), (0, 1))])
    y = [0.50, 0.53, 0.56, 0.58, 0.61, 0.64]
    fit = effects.lb_fit(_state(np.vstack([cells, cells[:2]]), [*y, 0.51, 0.52]))
    terms = {t.label: t for t in fit.terms}
    assert terms["i1"].kind == effects.TANGLED and terms["i1"].note == "cannot be separated from i0, i2"
    assert terms["i4 instead of i3"].swapped  # never together; the latest row has i4
    assert not terms["i0 vs i2"].swapped  # also used together; i0 first appeared later


def test_a_difference_of_ideas_used_together_says_vs(store: Store):
    subs = [
        ("1", "v1", 0.50, "0000"),
        ("2", "v2", 0.53, "1100"),
        ("3", "v3", 0.52, "1010"),
        ("4", "v4", 0.56, "1111"),
        ("5", "v1", 0.505, ""),
        ("6", "v4", 0.555, ""),
    ]
    fit = effects.lb_fit(_ledger(store, "a b c d", subs))
    diffs = [t for t in fit.terms if t.kind == effects.DIFFERENCE]
    assert diffs and all(" vs " in t.label and not t.swapped for t in diffs)
    assert all(t.note.startswith("also used together") for t in diffs)
    assert "d vs a" in {t.label for t in diffs}  # d came later than a


def test_runs_with_the_same_ideas_that_disagree_widen_the_intervals(store: Store):
    nbs = _notebooks(store, "a", {"base": "0", "with-a": "1", "with-a-v2": "1"})
    _run(store, _levels(0.0, 3, seed=1, noise=0.01), nbs["base"])
    _run(store, _levels(0.0, 3, seed=2, noise=0.01), nbs["with-a"])
    two = effects.run_fits(store.state(), _table_of(store), resamples=2000)[0]
    assert two.tau is None and two.lof_p is None  # no two runs share their ideas: nothing to measure
    assert _terms(two)["a"].verdict == "within noise"
    _run(store, _levels(0.20, 3, seed=3, noise=0.01), nbs["with-a-v2"])  # a change no idea describes
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=2000)
    a = _terms(fit)["a"]
    assert fit.tau == pytest.approx(0.14, abs=0.02) and fit.tau_df == 1
    assert fit.lof_p is None  # every residual degree of freedom comes from the two runs with a
    assert a.estimate == pytest.approx(0.10, abs=0.01) and a.ci_low < 0 < a.ci_high
    assert a.verdict == "within noise"
    assert any(
        n.startswith("runs with the same ideas differ beyond their item and draw noise") for n in fit.notes
    )


def test_repeated_runs_that_agree_add_no_run_to_run_noise_note(store: Store):
    nbs = _notebooks(store, "a", {"base": "0", "with-a": "1", "again": "1"})
    _run(store, _levels(0.0, 3, seed=1, noise=0.05), nbs["base"])
    _run(store, _levels(0.1, 3, seed=2, noise=0.05), nbs["with-a"])
    _run(store, _levels(0.1, 3, seed=3, noise=0.05), nbs["again"])
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=2000)
    assert (fit.tau, fit.tau_df, fit.lof_p) == (0.0, 1, None)
    assert not any("run-to-run" in n for n in fit.notes)
    assert _terms(fit)["a"].verdict == "helps"


def test_runs_that_do_not_add_up_are_a_lack_of_fit_not_run_noise(store: Store):
    # a 2 x 2 factorial, one run each, with an interaction: no two runs share their ideas
    nbs = _notebooks(store, "a b", {"n00": "00", "n10": "10", "n01": "01", "n11": "11"})
    for k, (name, shift) in enumerate((("n00", 0.0), ("n10", 0.05), ("n01", 0.05), ("n11", 0.30))):
        _run(store, _levels(shift, 3, seed=k, noise=0.01), nbs[name])
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=2000)
    assert fit.tau is None and fit.tau_df == 0  # not measurable without runs that share their ideas
    assert fit.lof_p is not None and fit.lof_p < 0.01
    assert any(n.startswith("the runs stray from the additive model beyond their noise") for n in fit.notes)
    assert not any("run-to-run" in n for n in fit.notes)


def test_a_repeated_table_counts_through_its_first_usable_run(store: Store):
    nbs = _notebooks(store, "a", {"n0": "0", "n1": "1"})
    table = _levels(0.05, 2, seed=1)
    lost = _run(store, table, None)  # recorded without --nb, then again with it
    kept = _run(store, table, nbs["n1"])
    _run(store, _levels(0.0, 2, seed=2), nbs["n0"])
    (fit,) = effects.run_fits(store.state(), _table_of(store), resamples=500)
    rows = {r.key: r for r in fit.rows}
    assert rows[kept].used and rows[lost].why_not.startswith(
        "no notebook (add its table again with `pseudorca"
    )
    assert not any("repeat" in n for n in fit.notes)


def test_parked_ideas_are_left_out_unless_asked(store: Store):
    subs = [("1", "n1", 0.5, "0-"), ("2", "n2", 0.6, "11"), ("3", "n1", 0.51, "")]
    _ledger(store, "a old", subs)
    store.append("idea.update", {"id": "old", "status": "parked"})
    fit = effects.lb_fit(store.state())
    assert "old" not in fit.rows[0].cells and [t.label for t in fit.terms] == ["a"]
    every = effects.lb_fit(store.state(), include_parked=True)
    assert _terms(every)["old"].kind == effects.EXCLUDED


def test_ideas_left_out_by_choice_are_named(store: Store):
    subs = [
        ("1", "n1", 0.50, "000"),
        ("2", "n2", 0.55, "110"),
        ("3", "n3", 0.52, "111"),
        ("4", "n1", 0.51, ""),
    ]
    fit = effects.lb_fit(_ledger(store, "short cache temp", subs), ideas=["short"])
    assert fit.notes[:2] == [
        "not modelled, so its effect is counted as theirs: cache always comes and goes with short",
        "not modelled but varying among these submissions: temp (their effects go into the residuals, and may "
        "bias the ideas asked for)",
    ]


def test_changes_look_past_rows_without_verdicts(store: Store):
    subs = [("1", "n1", 0.5, "0"), ("2", None, 0.6, ""), ("3", "n3", 0.7, "1"), ("4", "n4", 0.7, "-")]
    rows = effects.lb_fit(_ledger(store, "x", subs)).rows
    assert effects.all_changes(rows) == [[], [], ["+x"], []]
    assert effects.resubmits(rows) == [None, None, None, None]


def test_the_breakdown_names_ideas_outside_the_model_that_differ(store: Store):
    subs = [("1", "n1", 0.50, "00"), ("2", "n2", 0.55, "1-"), ("3", "n3", 0.56, "11"), ("4", "n1", 0.51, "")]
    fit = effects.lb_fit(_ledger(store, "a gap", subs))
    assert _terms(fit)["gap"].kind == effects.EXCLUDED
    w = effects.waterfall(fit, "1", "3")
    assert [s.label for s in w.steps] == ["+ a"] and w.unmodelled == ["gap"]
    same = effects.waterfall(fit, "2", "3")
    assert same.steps == [] and same.unmodelled == ["gap"]  # gap is unknown on 2: it may differ


def test_an_unknown_competition_is_an_error(store: Store):
    _ledger(store, "a", [("1", "n1", 0.5, "0"), ("2", "n2", 0.6, "1")])
    for sid in ("1", "2"):
        store.append("sub.upsert", {"id": sid, "competition": "some-competition"})
    with pytest.raises(
        LookupError, match=r"no submission is for competition 'other' \(competitions: some-competition\)"
    ):
        effects.lb_fit(store.state(), competition="other")


def test_a_huge_numeric_submission_id_sorts(store: Store):
    store.append("sub.upsert", {"id": "9" * 5000, "public_score": 0.5, "date": "2026-01-01"})
    assert effects.lb_fit(store.state()).rows[0].why_not.startswith("no notebook")


def test_config_defaults():
    def setting(table: dict[str, Any]) -> Callable[[str, str, Any], Any]:
        return lambda section, key, default: table.get(key, default)

    assert effects.config_defaults(setting({})) == (False, None)
    assert effects.config_defaults(setting({"lower_is_better": True, "noise": 2})) == (True, 2.0)
    for bad in ({"lower_is_better": "yes"}, {"noise": 0}, {"noise": True}, {"noise": "1"}):
        with pytest.raises(ValueError, match=r"\[effects\]"):
            effects.config_defaults(setting(bad))


def test_numbers_for_display():
    terms = [effects.Term(["a"], effects.EFFECT, 0.004, 0.001, 0.0004, 0.0076)]
    assert util.interval_decimals(terms) == 4  # 0.0004 excludes 0: it must not print as 0.000
    assert util.signed_number(0.0004, 4) == "+0.0004"
    assert util.score_decimals([0.412, 45.11, 0.12345]) == 5 and util.score_decimals([1e-5]) == 5
    assert util.score_decimals([0.123456789]) == 6 and util.score_decimals([1e30, None, math.nan]) == 0
    assert util.number(1e30, 3) == "1e+30" and util.plain_number(-2.5e30, 3) == "−2.5e+30"
    assert util.signed_number(2.5e30, 3) == "+2.5e+30" and util.signed_number(-0.0001, 2) == "0.00"
    assert util.number(None) == "" and util.plain_number(math.inf, 2) == "n/a"


def test_terminal_text_is_cleaned():
    hostile = "a\x1b]0;pwned\x07b\nc\u202ed\te"
    assert util.clean(hostile) == "a?]0;pwned?b?c?d\te"
    assert util.clean("日本語 ✓ −0.5") == "日本語 ✓ −0.5"
    table = util.format_table(["X\x1b[31m"], [[hostile]])
    assert "\x1b" not in table and "\x07" not in table and "\u202e" not in table


# ----- the command: review fixes -------------------------------------------------------------------


def _set_effects_config(store: Store, lines: str) -> None:
    """The [effects] table of config.toml set to `lines`; the other tables are kept."""
    config = store.root / "config.toml"
    text = re.sub(r"(?ms)^\[effects\]\n.*?(?=^\[|\Z)", "", config.read_text(encoding="utf-8"))
    config.write_text(text.rstrip("\n") + f"\n\n[effects]\n{lines}\n", encoding="utf-8")


def _section(output: str, head: str) -> list[str]:
    """The output's lines from `head` on."""
    lines = output.splitlines()
    return lines[lines.index(head) :]


_COLUMN_RE = re.compile(r"^  (?P<label>.+?) +(?P<num>[+−]?\d[\d.]*)(?:  (?P<rest>.*))?$")


def _column_adds_up(tail: list[str]) -> None:
    """The breakdown's numbers (the lines after its title) add up exactly as printed."""
    nums = [m["num"] for line in tail[1:] if (m := _COLUMN_RE.match(line))]
    values = [Decimal(n.replace("−", "-")) for n in nums]
    assert len(values) >= 2 and values[0] + sum(values[1:-1]) == values[-1], nums


def test_the_breakdown_says_which_ideas_are_outside_the_model(store: Store):
    subs = [("1", "n1", 0.50, "00"), ("2", "n2", 0.55, "1-"), ("3", "n3", 0.56, "11"), ("4", "n1", 0.51, "")]
    _ledger(store, "a gap", subs)  # gap has no verdict on 2: it is left out of the model
    out = _cli(store, "effects", "--sub", "3", "--base", "2").output
    tail = _section(out, "Breakdown of 3 (best in the model) from 2, by the model")
    assert [" ".join(line.split()) for line in tail[1:]] == [
        "2 fitted 0.555 observed 0.55",
        "(no modelled idea differs)",
        "= 3 fitted 0.555 observed 0.56",
        "not in the model, and different or unknown: gap",
    ]
    # no idea in the model at all: no breakdown by default, and an asked-for one says so
    bare = Store.init(store.root.parent.parent / "bare", name="bare")
    _ledger(bare, "a", [("1", "n1", 0.5, "-"), ("2", "n2", 0.6, "-"), ("3", "n2", 0.62, "")])
    assert "Breakdown" not in _cli(bare, "effects").output
    asked = _cli(bare, "effects", "--sub", "2").output
    assert "(no modelled idea differs)" in asked and "not in the model" not in asked  # no verdicts at all


def test_a_removed_bundle_reads_as_removed(store: Store):
    subs = [("1", "n0", 0.50, "00"), ("2", "n1", 0.55, "11"), ("3", "n0", 0.51, ""), ("4", "n1", 0.56, "")]
    _ledger(store, "p q", subs)
    out = _cli(store, "effects", "--sub", "1", "--base", "2").output
    tail = _section(out, "Breakdown of 1 from 2, by the model")
    assert tail[2].split()[:4] == ["−", "(p", "+", "q)"] and "−0.050" in tail[2]


def test_the_breakdown_keeps_the_scores_digits_and_adds_up(store: Store):
    # exactly one of x, y, z: z instead of x is +0.045 from a fitted 0.405, which 2 decimals cannot show
    subs = [
        ("1", "n0", 0.40, "100"),
        ("2", "n1", 0.42, "010"),
        ("3", "n2", 0.45, "001"),
        ("4", "n3", 0.41, "100"),
    ]
    _ledger(store, "x y z", subs)
    tail = _section(
        _cli(store, "effects").output, "Breakdown of 3 (best in the model) from 1 (first), by the model"
    )
    assert [line.split()[:4] for line in tail[1:]] == [
        ["1", "fitted", "0.405", "observed"],
        ["z", "instead", "of", "x"],
        ["=", "3", "fitted", "0.450"],
    ]
    assert tail[1].endswith("observed 0.40") and tail[2].split()[4] == "+0.045"
    _column_adds_up(tail)
    # three steps whose own roundings would not add up: the printed column still does
    store3 = Store.init(store.root.parent.parent / "three", name="three")
    _three_ideas(store3)
    for args in (
        (),
        ("--sub", "8", "--base", "1"),
        ("--sub", "1", "--base", "8"),
        ("--sub", "6", "--base", "2"),
    ):
        out = _cli(store3, "effects", *args).output
        _column_adds_up(_section(out, next(x for x in out.splitlines() if x.startswith("Breakdown of"))))


def test_unused_runs_of_a_protocol_without_a_model_say_why(store: Store):
    nbs = _notebooks(store, "a", {"n0": "0", "n1": "1"})
    typo = [
        _run(store, _levels(0.01 * k, 1, seed=k), nbs["n0" if k % 2 else "n1"], label=f"r{k}")
        for k in range(3)
    ]
    loose = _run(store, _levels(0.0, 1, seed=9), None, label="loose")
    gone = _run(store, _levels(0.05, 1, seed=8), nbs["n1"], label="gone")
    store.blob_path(store.state().runs[gone].results).unlink()
    out = _cli(store, "effects", "--metric", "scroe").output.splitlines()
    names = [f"{rid[:10]} (r{k})" for k, rid in enumerate(typo)]
    assert out[out.index("") + 1 :][-4:] == [
        "local runs (no protocol): 0 of 5 usable, nothing to compare: it needs at least 2 runs of registered "
        "notebooks with this metric",
        f"  not used: 3 runs ({', '.join(names)}): no metric 'scroe' (metrics: score)",
        f"  not used: {loose[:10]} (loose): no notebook (add its table again with `pseudorca run add FILE "
        "--nb NB`: that copy is used)",
        f"  not used: {gone[:10]} (gone): its results table ({store.state().runs[gone].results[:12]}) is not "
        "in this store; pull the latest .pseudorca from your repository",
    ]
    # many reasons: the first few, then how many runs are left
    for k in range(7):
        _run(store, _scores([("q000", 1)], f"m{k}"), nbs["n0"], label=f"m{k}", protocol="odd")
    odd = _cli(store, "effects", "--protocol", "odd", "--metric", "scroe").output.splitlines()
    assert sum(line.startswith("  not used: ") for line in odd) == 7
    assert odd[-1] == "  not used: 1 more run, for other reasons (--json lists them)"


def test_long_labels_and_notes_are_shown_in_full(store: Store):
    long_a, long_b = "shorter-chain-of-thought-prompt", "vllm-prefix-caching-enabled"
    subs = [
        ("1", "n0", 0.40, "00-"),
        ("2", "n0", 0.41, ""),
        ("3", "n1", 0.45, "111"),
        ("4", "n2", 0.44, "11-"),
        ("5", "n3", 0.45, "110"),
    ]
    out = _cli(store, "effects", "-n", "0").output
    assert out.startswith("no submissions yet")
    _ledger(store, f"{long_a} {long_b} gap", subs)
    lines = _cli(store, "effects").output.splitlines()
    bundle = next(line for line in lines if line.startswith(long_a[:20]))
    assert bundle.split()[:4] == ["shorter-chain-of-thought-prompt", "+", "v…", "[1]"]
    assert f"[1] {long_a} + {long_b}" in lines  # cut in the table, in full under it
    gap = next(line for line in lines if line.startswith("gap "))
    assert gap.endswith(
        "no present/absent verdict on 3 of 5 submissions (1, 2, 4); it varies among the others, so leaving "
        "it out may bias their effects; `pseudorca audit --submitted` fills the gaps"
    )
    rule = lines[lines.index(next(line for line in lines if line.startswith("IDEA "))) + 1]
    assert rule.endswith("  " + "-" * 60)  # one long note does not draw a rule that wraps


def test_the_sketch_never_draws_a_significant_effect_on_zero():
    def sketch(est: float, lo: float, hi: float, scale: float) -> str:
        from pseudorca.commands.effects import PAD, _sketch

        term = effects.Term(["a"], effects.EFFECT, est, 0.001, lo, hi)
        return _sketch(term, scale).replace(PAD, " ")

    assert sketch(0.01, 0.0072, 0.0128, 0.3028) == " " * 10 + "|*" + " " * 9
    assert sketch(-0.01, -0.0128, -0.0072, 0.3028) == " " * 9 + "*|" + " " * 10
    assert sketch(0.001, -0.0005, 0.0025, 0.3).strip() == "[*]"  # spans 0: the estimate sits on it
    assert sketch(0.1, -math.inf, math.inf, 0.2) == " " * 10 + "|    *     "  # no interval to draw
    assert sketch(math.nan, 0.0, 1.0, 1.0) == "" and sketch(0.1, 0.0, 0.2, math.inf) == ""


def test_the_sketch_heading_keeps_its_labels_apart():
    from pseudorca.commands.effects import PAD, _sketch_head

    for scale, dec in (
        (12345.678, 3),
        (123456789.0, 0),
        (12.345678, 6),
        (2.408e30, 3),
        (2.4e-7, 7),
        (0.066, 3),
    ):
        head = _sketch_head(scale, dec).replace(PAD, " ")
        assert len(head) == 21 and head[9:12] == " 0 ", head
        left, _, right = head.split()
        assert left == "−" + right[1:] and right.startswith("+")
    assert _sketch_head(1e-300, 3).replace(PAD, " ").split() == ["−1e-300", "0", "+1e-300"]  # never "0.000"
    assert _sketch_head(math.inf, 3) == "" and _sketch_head(0.0, 3) == ""


def test_absurd_noise_and_overflow_give_no_traceback(store: Store, monkeypatch: pytest.MonkeyPatch):
    _two_groups(store)
    wide = _cli(store, "effects", "--noise", "1.5e308")  # the bounds overflow to infinity
    assert wide.exit_code == 0, wide.output
    assert "[−inf, +inf]" in wide.output and "Traceback" not in wide.output

    def overflow(*args: Any, **kwargs: Any) -> Any:
        raise OverflowError("integer division result too large for a float")

    monkeypatch.setattr(effects, "lb_fit", overflow)
    huge = _cli(store, "effects")
    assert (huge.exit_code, huge.output) == (
        1,
        "Error: a computation overflowed (integer division result too large for a float): are some scores, "
        "or the noise SD, huge?\n",
    )


def test_json_lists_the_latest_rows_and_the_total(store: Store):
    _rich(store)
    data = json.loads(_cli(store, "effects", "--json", "-n", "2", "--resamples", "500").stdout)
    assert [r["key"] for r in data["lb"]["rows"]] == ["106", "107"] and data["lb"]["total"] == 7
    assert data["lb"]["rows"][-1]["changes"] == ["+new", "−old", "−x"]
    assert data["lb"]["rows"][-1]["resubmits"] == "103"
    assert [(len(f["rows"]), f["total"]) for f in data["runs"]] == [(2, 3)]
    every = json.loads(_cli(store, "effects", "--json", "-n", "0", "--no-runs").stdout)
    assert len(every["lb"]["rows"]) == every["lb"]["total"] == 7


def test_versions_with_the_same_name_are_told_apart(store: Store):
    revs = _ideas(store, "a")
    versions = [json.dumps({"cells": [f"version {k}"]}).encode() for k in range(3)]
    ids = [register_bytes(store, data, name="kernel.ipynb")[0].id for data in versions]
    other = _nb(store, "other.ipynb")
    for nb, code in zip([*ids, other], "0100", strict=True):
        _record(store, nb, "a", revs["a"], code)
    for k, nb in enumerate([ids[0], ids[1], ids[2], ids[0], other]):
        store.append(
            "sub.upsert", {"id": str(k + 1), "nb": nb, "public_score": 0.4 + k / 100, "date": _day(k)}
        )
    lines = _cli(store, "effects", "--no-runs").output.splitlines()
    shown = [line.split()[2:4] for line in lines[3:8]]
    assert shown[:4] == [["kernel.ipynb", util.short(nb)] for nb in (ids[0], ids[1], ids[2], ids[0])]
    assert shown[4][0] == "other.ipynb" and shown[4][1] != util.short(other)  # unique: the name alone


def test_tiny_scores_keep_their_digits(store: Store):
    _ledger(
        store,
        "a",
        [("1", "n1", 1e-7, "0"), ("2", "n2", 3e-7, "1"), ("3", "n1", 1.2e-7, ""), ("4", "n2", 2.9e-7, "")],
    )
    out = _cli(store, "effects").output
    lines = out.splitlines()
    assert lines[3].split()[3:5] == ["0.00000010", "0.00000011"]  # not "0"
    head = next(line for line in lines if line.startswith("IDEA "))
    assert head.split()[6:9] == ["−2.3e-07", "0", "+2.3e-07"]
    _column_adds_up(_section(out, "Breakdown of 2 (best in the model) from 1 (first), by the model"))


def test_a_long_bootstrap_says_what_is_quicker(store: Store):
    nbs = _notebooks(store, "a", {"n0": "0", "n1": "1"})
    for k, name in enumerate(("n0", "n1")):
        data = _levels(0.02 * k, 1, seed=k, items=10)
        store.append(
            "run.add",
            {
                "id": util.new_ulid(),
                "results": store.put_blob(data),
                "format": "csv",
                "nb": nbs[name],
                "n_rows": 10**6,
            },
        )
    slow = _cli(store, "effects", "--resamples", "200")
    assert slow.exit_code == 0, slow.output
    assert slow.stderr == (
        "this may take a while: the bootstrap of the local runs draws about 4e+08 values (fewer --resamples "
        "is quicker, and --no-runs leaves the runs out)\n"
    )
    assert _cli(store, "effects", "--resamples", "100").stderr == ""  # 2e8 draws: quick enough
    assert _cli(store, "effects", "--resamples", "200", "--no-runs").stderr == ""
    assert _cli(store, "effects", "--resamples", "200", "--protocol", "").stderr != ""  # the same protocol


def test_ledger_strings_cannot_drive_the_terminal(store: Store):
    hostile = "EVIL\x1b]0;pwned\x07RED\u202e"
    revs = _ideas(store, "a")
    store.append("idea.add", {"id": "b\nc\x1b[31m", "title": "a hand-edited event"})
    bad_rev = store.state().ideas["b\nc\x1b[31m"].rev
    nb0 = register_bytes(store, b"zero", name=hostile)[0].id
    nb1 = _nb(store, "one")
    for nb, code in ((nb0, "0"), (nb1, "1")):
        _record(store, nb, "a", revs["a"], code)
        data = {
            "nb": nb,
            "idea": "b\nc\x1b[31m",
            "idea_rev": bad_rev,
            "status": STATUS[code],
            "method": "human",
        }
        store.append("audit.verdict", data)
    for k, (sid, nb, score) in enumerate(
        (("1", nb0, 0.5), ("2", nb1, 0.6), ("3", nb0, 0.51), ("4\x1b[2J", nb1, 0.61))
    ):
        store.append("sub.upsert", {"id": sid, "nb": nb, "public_score": score, "date": _day(k)})
    _run(store, _levels(0.0, 1, seed=1), nb0, label=hostile, protocol="p\x1b[2J")
    out = _cli(store, "effects").output
    assert "Breakdown of 4?[2J (best in the model) from 1 (first), by the model" in out
    assert "local runs (protocol p?[2J)" in out and "EVIL?]0;pwned?RED?" in out
    unsafe = [c for c in out if (ord(c) < 32 and c not in "\n\t") or 0x7F <= ord(c) <= 0x9F or c == "\u202e"]
    assert unsafe == []


def test_the_direction_and_the_noise_come_from_the_config(store: Store):
    _two_groups(store)
    _set_effects_config(store, "lower_is_better = true\nnoise = 0.01")
    out = _cli(store, "effects").output
    assert (
        "What each idea adds to the public score (additive model over 5 submissions, lower is better)" in out
    )
    assert next(line for line in out.splitlines() if line.startswith("a ")).split()[5] == "hurts"
    assert "noise: SD 0.01 as given ([effects] noise in config.toml); normal intervals" in out
    assert "Breakdown of 1 (best in the model) from 1" not in out  # the best is the lowest: the first
    flags = _cli(store, "effects", "--higher-is-better", "--noise", "0.02").output
    assert "lower is better" not in flags and "noise: SD 0.02 as given (--noise); normal intervals" in flags
    assert next(line for line in flags.splitlines() if line.startswith("a ")).split()[5] == "helps"
    _set_effects_config(store, 'noise = "high"')
    assert _cli(store, "effects", "--noise", "0.02").exit_code == 0  # the flag overrides the bad value
    bad = _cli(store, "effects")
    assert (bad.exit_code, bad.output) == (
        1,
        "Error: [effects] noise in config.toml must be a positive number (the LB noise SD)\n",
    )


def test_parked_ideas_are_shown_with_all(store: Store):
    _ledger(store, "a old", [("1", "n1", 0.5, "00"), ("2", "n2", 0.6, "11"), ("3", "n1", 0.51, "")])
    store.append("idea.update", {"id": "old", "status": "parked"})
    hidden = _cli(store, "effects", "--no-runs").output
    assert "ideas: 1 a\n" in hidden and "(1 parked idea hidden; use --all)" in hidden
    every = _cli(store, "effects", "--no-runs", "--all").output
    assert "ideas: 1 a  2 old" in every and "parked idea" not in every and "a + old" in every


def test_the_run_to_run_noise_is_reported(store: Store):
    nbs = _notebooks(store, "a", {"base": "0", "with-a": "1", "with-a-v2": "1"})
    _run(store, _levels(0.0, 3, seed=1, noise=0.01), nbs["base"])
    _run(store, _levels(0.0, 3, seed=2, noise=0.01), nbs["with-a"])
    _run(store, _levels(0.20, 3, seed=3, noise=0.01), nbs["with-a-v2"])
    out = _cli(store, "effects", "--resamples", "500").output
    (line,) = [x for x in out.splitlines() if x.startswith("run-to-run SD")]
    assert re.fullmatch(
        r"run-to-run SD 0\.1\d+, from the runs with the same ideas \(1 degree of freedom\): in the intervals",
        line,
    )
    calm = Store.init(store.root.parent.parent / "calm", name="calm")
    nbs = _notebooks(calm, "a", {"base": "0", "with-a": "1", "again": "1"})
    for k, (name, shift) in enumerate((("base", 0.0), ("with-a", 0.1), ("again", 0.1))):
        _run(calm, _levels(shift, 3, seed=k + 1, noise=0.05), nbs[name])
    out = _cli(calm, "effects", "--resamples", "500").output
    assert "run-to-run SD 0: the runs with the same ideas agree within their item and draw noise" in out


def test_the_verdicts_that_need_explaining_are_explained(store: Store):
    nbs = _notebooks(store, "x", {"without": "0", "with": "1"})
    _run(store, _scores([(f"q{i}", 0.5) for i in range(5)]), nbs["without"])
    _run(store, _scores([(f"q{i}", 0.6 + i / 100) for i in range(5)]), nbs["with"])
    out = _cli(store, "effects", "--resamples", "500").output
    assert (
        "too few items: 5 items cannot reach p < 0.05 in the sign-flip test (it needs 6 or more), so only "
        "the interval is informative" in out
    )
    _ledger(store, "a b", [("1", "n1", 0.50, "00"), ("2", "n2", 0.56, "10"), ("3", "n3", 0.54, "01")])
    out = _cli(store, "effects", "--no-runs").output
    assert "unknown: there is no interval, as there is no noise estimate" in out


def test_an_unknown_competition_is_a_one_line_error(store: Store):
    _ledger(store, "a", [("1", "n1", 0.5, "0"), ("2", "n2", 0.6, "1")])
    for sid in ("1", "2"):
        store.append("sub.upsert", {"id": sid, "competition": "word-problems"})
    bad = _cli(store, "effects", "--competition", "word-problem")
    assert (bad.exit_code, bad.output) == (
        1,
        "Error: no submission is for competition 'word-problem' (competitions: word-problems)\n",
    )
