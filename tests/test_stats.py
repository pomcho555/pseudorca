from __future__ import annotations

import math

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


def test_compare_attaches_aa_checks_on_the_compared_items():
    a, b = _arms(6, 2, 0.0, seed=4)
    b_single = {k: v[:1] for k, v in b.items()}
    res = stats.compare(a, b_single, resamples=500)
    assert res.aa_a is not None and res.aa_b is None
    assert res.aa_a.n_items == 6 and res.aa_a.ci_low <= res.aa_a.ci_high
    # dropping two items from B leaves A with only 4 compared items with 2+ draws: no A/A check
    fewer = {k: v for k, v in b.items() if k not in ("q00", "q01")}
    assert stats.compare(a, fewer, resamples=500).aa_a is None


def _sub(sid: str, nb: str | None, score: float | None) -> Submission:
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


def test_lb_diff():
    diff, z = stats.lb_diff(0.70, 0.75, 0.025)
    assert diff == pytest.approx(0.05) and z == pytest.approx(0.05 / (0.025 * math.sqrt(2)))
    assert stats.lb_diff(0.75, 0.70, 0.025)[1] == pytest.approx(-z)
    assert stats.lb_diff(0.5, 0.6, 0.0)[1] == math.inf
    assert stats.lb_diff(0.5, 0.5, 0.0) == (0.0, 0.0)


def test_spread():
    assert stats.spread([3, 1, 2, 4]) == (1, 2.5, 4)
    assert stats.spread([]) == (0, 0.0, 0)
