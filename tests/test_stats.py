from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
import pytest

from pseudorca import stats
from pseudorca.model import Submission


def _arms(n_items: int, draws: int, shift: float, *, noise: float = 0.1, seed: int = 0):
    """Two arms on the same items: B = A's item level + shift, each draw with independent noise."""
    rng = np.random.default_rng(seed)
    level = rng.uniform(0.2, 0.8, n_items)
    a = {f"q{i:02d}": level[i] + rng.normal(0, noise, draws) for i in range(n_items)}
    b = {f"q{i:02d}": level[i] + shift + rng.normal(0, noise, draws) for i in range(n_items)}
    return a, b


def test_clear_effect_excludes_zero():
    a, b = _arms(30, 3, 0.2, seed=1)
    res = stats.compare(a, b, resamples=4000, seed=7)
    assert res.n_items == 30 and res.only_a == [] and res.only_b == []
    assert res.ci_low > 0 and res.verdict == "B better"
    assert res.ci_low < res.diff < res.ci_high
    assert res.diff == pytest.approx(0.2, abs=0.05)
    assert res.diff == pytest.approx(res.mean_b - res.mean_a)
    assert res.wins > 25 and res.p_value < 0.001
    assert res.mde80 == pytest.approx(2.8 * res.se) and res.se > 0
    assert res.draws_a == (3, 3.0, 3) and (res.resamples, res.seed) == (4000, 7)

    flipped = stats.compare(b, a, resamples=4000, seed=7)
    assert flipped.verdict == "A better" and flipped.ci_high < 0
    assert (flipped.wins, flipped.losses) == (res.losses, res.wins)


def test_lower_is_better_flips_the_verdict_not_the_sign():
    a, b = _arms(30, 2, 0.2, seed=2)
    res = stats.compare(a, b, resamples=2000, higher_is_better=False)
    assert res.diff > 0 and res.verdict == "A better" and not res.higher_is_better
    assert res.losses > res.wins


def test_identical_arms_are_within_noise():
    a, _ = _arms(20, 3, 0.0, seed=3)
    res = stats.compare(a, a, resamples=2000)
    assert res.diff == 0 and res.verdict == "within noise"
    assert res.ci_low <= 0 <= res.ci_high
    assert res.ties == 20 and res.wins == res.losses == 0
    assert res.p_value == 1.0


def test_ragged_draws_are_averaged_within_items_first():
    a = {"x": np.array([0.0, 0.0, 0.0, 1.0]), "y": np.array([1.0]), "z": np.array([0.5, 0.5])}
    b = {"x": np.array([1.0, 1.0]), "y": np.array([1.0, 1.0, 0.0]), "z": np.array([1.0])}
    res = stats.compare(a, b, resamples=3000, seed=1)
    assert res.mean_a == pytest.approx((0.25 + 1.0 + 0.5) / 3)  # not the row mean 2/7
    assert res.mean_b == pytest.approx((1.0 + 2 / 3 + 1.0) / 3)
    assert res.diff == pytest.approx(((1 - 0.25) + (2 / 3 - 1) + (1 - 0.5)) / 3)
    assert res.draws_a == (1, 2.0, 4) and res.draws_b == (1, 2.0, 3)
    assert res.ci_low <= res.diff <= res.ci_high
    again = stats.compare(a, b, resamples=3000, seed=1)
    assert (again.ci_low, again.ci_high, again.p_value) == (res.ci_low, res.ci_high, res.p_value)
    other_seed = stats.compare(a, b, resamples=3000, seed=2)
    assert (other_seed.ci_low, other_seed.ci_high) != (res.ci_low, res.ci_high)


def test_only_items_in_both_arms_count():
    a = {"x": [1.0], "y": [1.0, 2.0], "only-a": [5.0], "empty-in-b": [3.0]}
    b = {"x": [2.0], "y": [2.0], "only-b": [9.0], "empty-in-b": [], "nan-only": [math.nan]}
    res = stats.compare(a, b, resamples=500)
    assert res.n_items == 2
    assert res.only_a == ["empty-in-b", "only-a"] and res.only_b == ["only-b"]
    assert res.diff == pytest.approx((1.0 + 0.5) / 2)
    with pytest.raises(ValueError, match="no item"):
        stats.compare({"x": [1.0]}, {"y": [1.0]})


def test_bad_inputs_raise_value_errors():
    a, b = {"x": [1.0, 2.0]}, {"x": [1.5, math.inf]}
    with pytest.raises(ValueError, match="infinite"):
        stats.compare(a, b)
    with pytest.raises(ValueError, match="confidence"):
        stats.compare(a, a, confidence=1.5)
    with pytest.raises(ValueError, match="resamples"):
        stats.compare(a, a, resamples=1)
    with pytest.raises(ValueError, match="seed"):
        stats.compare(a, a, seed=-1)


def test_interval_covers_zero_under_the_null():
    """200 null simulations with ragged draws: the 95% interval should cover 0 most of the time."""
    rng = np.random.default_rng(2024)
    covered = 0
    for sim in range(200):
        a, b = {}, {}
        for i in range(15):
            level = rng.normal(0, 1)
            a[f"q{i}"] = level + rng.normal(0, 1, int(rng.integers(1, 5)))
            b[f"q{i}"] = level + rng.normal(0, 1, int(rng.integers(1, 5)))
        res = stats.compare(a, b, resamples=1000, seed=sim)
        covered += res.ci_low <= 0 <= res.ci_high
    assert 0.88 <= covered / 200 <= 0.99


def test_exact_sign_flip_p_for_a_tiny_case():
    # d = (1, 2, 3): of the 8 sign flips only (+,+,+) and (−,−,−) reach |sum| = 6, so p = 2/8
    a = {k: [0.0] for k in "xyz"}
    b = {"x": [1.0], "y": [2.0], "z": [3.0]}
    assert stats.compare(a, b, resamples=200).p_value == pytest.approx(0.25)
    # d = (2, 1): |±2 ± 1| = 3, 1, 1, 3, so p = 2/4
    assert stats.compare({"x": [0.0], "y": [0.0]}, {"x": [2.0], "y": [1.0]}, resamples=200).p_value == 0.5


def test_random_sign_flips_above_sixteen_items():
    a = {f"q{i}": [0.0] for i in range(30)}
    b = {f"q{i}": [1.0 + i / 100] for i in range(30)}
    res = stats.compare(a, b, resamples=200)
    assert res.p_value == pytest.approx(1 / (stats.RANDOM_FLIPS + 1))  # only the identity flip reaches it


def _all_favour_b(n: int) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    return {f"q{i}": [0.5] for i in range(n)}, {f"q{i}": [0.51 + i / 100] for i in range(n)}


def test_too_few_items_never_name_a_winner():
    # one item: the bootstrap interval is a single point that excludes 0, but p = 1
    one = stats.compare({"x": [0.50]}, {"x": [0.51]}, resamples=500)
    assert one.ci_low > 0 and one.p_value == 1.0 and one.se == 0
    assert one.verdict == "too few items" and one.small_n and one.min_items == 6
    # 2, 3 and 5 items that all favour B: the interval excludes 0, but the exact p is 0.5, 0.25, 0.0625
    for n, p in ((2, 0.5), (3, 0.25), (5, 0.0625)):
        res = stats.compare(*_all_favour_b(n), resamples=500)
        assert res.ci_low > 0 and res.p_value == pytest.approx(p)
        assert res.verdict == "too few items" and res.small_n
        lower = stats.compare(*_all_favour_b(n), resamples=500, higher_is_better=False)
        assert lower.verdict == "too few items"
    # six items is the fewest that can reach p < 0.05: 2 of the 64 sign flips are as extreme as the data
    six = stats.compare(*_all_favour_b(6), resamples=500)
    assert six.p_value == pytest.approx(2 / 64) and six.verdict == "B better" and six.small_n
    # at 90% confidence five items are enough
    five = stats.compare(*_all_favour_b(5), resamples=500, confidence=0.9)
    assert five.min_items == 5 and five.verdict == "B better"


def test_a_winner_needs_the_permutation_test_to_agree():
    # Five items favour B by 0.2 and one favours A by 0.02. Every bootstrap mean is positive, but 4 of the
    # 64 sign flips are as extreme as the data (p = 0.0625), so the difference is within noise.
    a = {f"q{i}": [0.5] for i in range(6)}
    b = {**{f"q{i}": [0.7] for i in range(5)}, "q5": [0.48]}
    res = stats.compare(a, b, resamples=2000)
    assert res.ci_low > 0 and res.p_value == pytest.approx(4 / 64)
    assert res.verdict == "within noise" and res.min_items == 6
    assert stats.compare(b, a, resamples=2000).verdict == "within noise"


def test_min_items_and_small_n():
    assert [stats.min_items(c) for c in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)] == [3, 4, 5, 6, 8, 11]
    nine, ten = _arms(9, 2, 0.2, seed=5), _arms(10, 2, 0.2, seed=5)
    assert stats.compare(*nine, resamples=500).small_n
    res = stats.compare(*ten, resamples=500)
    assert not res.small_n and res.verdict == "B better"
    strict = stats.compare(*ten, resamples=500, confidence=0.999)  # 99.9% needs 11 items
    assert strict.small_n and strict.min_items == 11 and strict.verdict == "too few items"


def test_random_flips_resolve_a_high_confidence():
    # 30 items that all favour B: the exact p is 2/2^30, so even a 99.999% verdict can name a winner
    a = {f"q{i}": [0.0] for i in range(30)}
    b = {f"q{i}": [1.0 + i / 100] for i in range(30)}
    res = stats.compare(a, b, resamples=200, confidence=0.99999)
    assert res.p_value < 1e-5 and res.verdict == "B better"


def test_aa_needs_two_draws_on_five_items():
    four = {f"q{i}": np.array([0.0, 1.0]) for i in range(4)}
    assert stats.aa(four, resamples=200) is None
    with_single = {**four, "single": np.array([1.0]), "nan-pair": np.array([1.0, math.nan])}
    assert stats.aa(with_single, resamples=200) is None  # 1 draw (after dropping NaN) does not count
    five = {**four, "q4": np.array([0.0, 1.0, 0.0, 1.0])}
    check = stats.aa(five, resamples=200)
    assert check is not None and check.n_items == 5
    # alternate split: draws 0, 2 (all 0.0) versus draws 1, 3 (all 1.0)
    assert check.diff == pytest.approx(1.0) and check.draws_a == (1, 1.0, 2)
    assert check.aa_a is None and check.aa_b is None
    assert check.verdict == "too few items"  # five items cannot reach p < 0.05


def test_compare_attaches_aa_checks_on_the_compared_items():
    a, b = _arms(6, 2, 0.0, seed=4)
    b_single = {k: v[:1] for k, v in b.items()}
    res = stats.compare(a, b_single, resamples=500)
    assert res.aa_a is not None and res.aa_b is None
    assert res.aa_a.n_items == 6 and res.aa_a.ci_low <= res.aa_a.ci_high
    # dropping two items from B leaves A with only 4 compared items with 2+ draws: no A/A check
    fewer = {k: v for k, v in b.items() if k not in ("q00", "q01")}
    assert stats.compare(a, fewer, resamples=500).aa_a is None


def _sub(sid: str, nb: Any, score: Any) -> Submission:
    return Submission(id=sid, nb=nb, public_score=score)


def test_lb_noise_pooled_sd_by_hand():
    subs = [
        _sub("1", "nb1", 0.70),
        _sub("2", "nb1", 0.72),  # nb1: mean 0.71, s^2 = 0.0002 (1 dof)
        _sub("3", "nb2", 0.50),
        _sub("4", "nb2", 0.53),
        _sub("5", "nb2", 0.56),  # nb2: mean 0.53, s^2 = 0.0009 (2 dof)
        _sub("6", "nb3", 0.90),  # a single submission: no spread to learn from
        _sub("7", None, 0.80),  # no notebook
        _sub("8", "nb1", None),  # not scored yet
    ]
    noise = stats.lb_noise(subs)
    assert noise is not None
    assert noise.sd == pytest.approx(math.sqrt((1 * 0.0002 + 2 * 0.0009) / 3))
    assert noise.sd == pytest.approx(0.0258199, abs=1e-6)
    assert (noise.n_groups, noise.n_subs) == (2, 5)
    assert noise.groups == {"nb1": [0.70, 0.72], "nb2": [0.50, 0.53, 0.56]}


def test_lb_noise_needs_a_repeat():
    assert stats.lb_noise([]) is None
    assert stats.lb_noise([_sub("1", "a", 0.5), _sub("2", "b", 0.6), _sub("3", None, 0.7)]) is None


def test_lb_noise_skips_scores_that_are_not_finite_numbers():
    subs = [
        _sub("1", "nb1", 0.70),
        _sub("2", "nb1", 0.72),
        _sub("3", "nb1", "pending"),  # a hand-edited or foreign event
        _sub("4", "nb1", "0.9"),
        _sub("5", "nb1", math.nan),
        _sub("6", "nb1", -math.inf),
        _sub("7", "nb1", True),
        _sub("8", "nb1", [0.7]),
        _sub("9", ["nb1"], 0.9),  # not a notebook id
        _sub("10", "nb2", np.float64(0.5)),
        _sub("11", "nb2", np.int64(1)),
    ]
    noise = stats.lb_noise(subs)
    assert noise is not None
    assert noise.groups == {"nb1": [0.70, 0.72], "nb2": [0.5, 1.0]}
    assert all(type(v) is float for scores in noise.groups.values() for v in scores)
    assert (noise.n_groups, noise.n_subs) == (2, 4)
    assert stats.lb_noise([_sub("1", "a", "x"), _sub("2", "a", "y")]) is None
    assert stats.finite(0.5) == 0.5 and stats.finite(np.float32(0.25)) == 0.25
    assert [stats.finite(v) for v in (None, "1", True, math.nan, math.inf, 10**400)] == [None] * 6


def test_lb_diff():
    diff, z = stats.lb_diff(0.70, 0.75, 0.025)
    assert diff == pytest.approx(0.05) and z == pytest.approx(0.05 / (0.025 * math.sqrt(2)))
    assert stats.lb_diff(0.75, 0.70, 0.025)[1] == pytest.approx(-z)
    assert stats.lb_diff(0.5, 0.6, 0.0)[1] == math.inf
    assert stats.lb_diff(0.5, 0.5, 0.0) == (0.0, 0.0)


def test_spread():
    assert stats.spread([3, 1, 2, 4]) == (1, 2.5, 4)
    assert stats.spread([]) == (0, 0.0, 0)


# ----- Student's t quantiles -----------------------------------------------------------------------

T_TABLE = {  # (p, df): the quantile, as tables print it (12 significant digits)
    (0.975, 1): 12.7062047362,
    (0.975, 2): 4.30265272975,
    (0.975, 3): 3.18244630528,
    (0.975, 4): 2.77644510520,
    (0.975, 5): 2.57058183564,
    (0.975, 10): 2.22813885199,
    (0.975, 30): 2.04227245630,
    (0.975, 100): 1.98397151852,
    (0.95, 1): 6.31375151468,
    (0.95, 5): 2.01504837333,
    (0.95, 10): 1.81246112281,
    (0.9, 3): 1.63774435370,
    (0.99, 20): 2.52797700274,
    (0.995, 5): 4.03214298356,
    (0.995, 10): 3.16927267262,
    (0.75, 7): 0.711141778082,
}


@pytest.mark.parametrize(("p", "df"), list(T_TABLE))
def test_t_quantile_table_values(p: float, df: int):
    assert stats.t_quantile(p, df) == pytest.approx(T_TABLE[p, df], rel=1e-11)
    assert stats.t_quantile(1 - p, df) == pytest.approx(-T_TABLE[p, df], rel=1e-11)
    assert stats.t_quantile(p, float(df)) == stats.t_quantile(p, df)


def test_t_quantile_closed_forms():
    for p in (0.51, 0.6, 0.9, 0.975, 0.999):
        assert stats.t_quantile(p, 1) == pytest.approx(math.tan(math.pi * (p - 0.5)), rel=1e-12)
        assert stats.t_quantile(p, 2) == pytest.approx((2 * p - 1) / math.sqrt(2 * p * (1 - p)), rel=1e-12)
        # 4 degrees of freedom: t = 2·sqrt(cos(acos(√a)/3)/√a − 1) with a = 4p(1 − p)
        a = 4 * p * (1 - p)
        closed = 2 * math.sqrt(math.cos(math.acos(math.sqrt(a)) / 3) / math.sqrt(a) - 1)
        assert stats.t_quantile(p, 4) == pytest.approx(closed, rel=1e-11)
        # 3 degrees of freedom: the CDF is 1/2 + (θ + sin θ cos θ)/π with θ = atan(t/√3)
        theta = math.atan(stats.t_quantile(p, 3) / math.sqrt(3))
        assert 0.5 + (theta + math.sin(theta) * math.cos(theta)) / math.pi == pytest.approx(p, rel=1e-12)
        # 5: 1/2 + (θ + sin θ cos θ (1 + 2/3 cos² θ))/π with θ = atan(t/√5)
        theta = math.atan(stats.t_quantile(p, 5) / math.sqrt(5))
        cdf = 0.5 + (theta + math.sin(theta) * math.cos(theta) * (1 + 2 / 3 * math.cos(theta) ** 2)) / math.pi
        assert cdf == pytest.approx(p, rel=1e-12)


def test_t_quantile_is_symmetric_and_keeps_its_tails():
    for df in (1, 2, 3, 4.5, 7, 50, 2999, 3000, 1e6, math.inf):
        for p in (0.25, 0.125, 0.0625, 1 / 1024):  # 1 − p is exact for these
            assert stats.t_quantile(1 - p, df) == -stats.t_quantile(p, df) > 0
        assert stats.t_quantile(0.5, df) == 0.0
    # far out in the tails, where 1 − p would round to 1 (this used to raise ValueError)
    for p in (1e-10, 1e-17, 1e-20, 1e-300):
        assert stats.t_quantile(p, 1) == pytest.approx(-1 / math.tan(math.pi * p), rel=1e-12)
        assert stats.t_quantile(p, 2) == pytest.approx(-(1 - 2 * p) / math.sqrt(2 * p * (1 - p)), rel=1e-12)
        a = 4 * p * (1 - p)
        closed = 2 * math.sqrt(math.cos(math.acos(math.sqrt(a)) / 3) / math.sqrt(a) - 1)
        assert stats.t_quantile(p, 4) == pytest.approx(-closed, rel=1e-10)
        assert stats.t_quantile(p, 3) < stats.t_quantile(p, 30) < stats.t_quantile(p, math.inf) < 0
    # just above 1/2 the quantile is tiny but not 0: P(T < t) ≈ 1/2 + t·f(0)
    density = math.gamma(4) / (math.sqrt(7 * math.pi) * math.gamma(3.5))
    assert stats.t_quantile(0.5 + 1e-6, 7) == pytest.approx(1e-6 / density, rel=1e-6)


def test_t_quantile_approaches_the_normal(monkeypatch: pytest.MonkeyPatch):
    for p in (0.6, 0.975, 0.9999):
        z = statistics.NormalDist().inv_cdf(p)
        assert stats.t_quantile(p, math.inf) == z
        values = [stats.t_quantile(p, df) for df in (3, 10, 100, 1000, 2999, 3000, 1e5, 1e9, 1e300)]
        assert values == sorted(values, reverse=True) and values[-1] >= z  # down towards the normal
        assert stats.t_quantile(p, 1e9) - z == pytest.approx((z**3 + z) / 4e9, rel=1e-6)  # the 1/df term
    # from T_EXPANSION_DF on, the expansion: where the two methods meet, they agree
    assert stats.T_EXPANSION_DF == 3000
    expanded = [stats.t_quantile(p, df) for p in (0.6, 0.975, 1 - 1e-9) for df in (3000, 5000)]
    monkeypatch.setattr(stats, "T_EXPANSION_DF", math.inf)
    bisected = [stats.t_quantile(p, df) for p in (0.6, 0.975, 1 - 1e-9) for df in (3000, 5000)]
    assert expanded == pytest.approx(bisected, rel=1e-11)


def test_t_quantile_rejects_bad_input():
    for p in (0, 1, -0.5, 1.5, math.nan):
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            stats.t_quantile(p, 5)
    for df in (0, -1, math.nan, -math.inf):
        with pytest.raises(ValueError, match="degrees of freedom must be positive"):
            stats.t_quantile(0.9, df)


def test_t_quantile_below_one_degree_of_freedom():
    # tails heavier than the Cauchy's: larger quantiles, and beyond the float range ±inf
    assert stats.t_quantile(0.975, 1) < stats.t_quantile(0.975, 0.5) < math.inf
    assert stats.t_quantile(0.975, 1e-3) == math.inf and stats.t_quantile(0.025, 1e-3) == -math.inf
