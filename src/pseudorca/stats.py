"""Noise-aware statistics: paired per-item comparisons, A/A checks and leaderboard noise (numpy only).

A comparison pairs the two arms by item, averages each item's draws, and reports the mean per-item
difference with a two-stage bootstrap interval (resample items, then draws within each item), a sign-flip
permutation p-value, and the minimum detectable effect at 80% power. It names a winner only when both the
interval and the permutation test say so.
"""

from __future__ import annotations

import math
import numbers
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from pseudorca.model import Submission

TIE = 1e-12  # |d_i| below this is a tie
EXACT_MAX_ITEMS = 16  # the permutation test enumerates all 2^n sign flips up to this many items
RANDOM_FLIPS = 20_000  # random sign flips above it (more for a confidence above 99.9%, see _flips)
FLIPS_PER_ALPHA = 20  # enough random flips for this many expected hits at p = 1 − confidence...
MAX_FLIPS = 1_000_000  # ...up to this many (a p-value floor of 1e-6)
MDE_Z = 2.8  # z(0.975) + z(0.80): an effect of 2.8 SE is detected 80% of the time at 95% confidence
AA_MIN_ITEMS = 5  # an A/A check needs this many items with 2+ draws
SMALL_N = 10  # with fewer items the interval and p-value are rough
WITHIN_NOISE = "within noise"
TOO_FEW = "too few items"  # fewer than min_items(confidence): no outcome could be significant
_CHUNK = 2_000_000  # array elements per vectorised batch, to bound memory


@dataclass
class CompareResult:
    """B versus A on the items both arms have. `diff` and the interval are B − A."""

    n_items: int
    only_a: list[str]
    only_b: list[str]
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    confidence: float
    se: float
    mde80: float
    p_value: float
    wins: int  # items where B is better (by `higher_is_better`)
    losses: int
    ties: int
    draws_a: tuple[int, float, int]  # draws per item: min, median, max
    draws_b: tuple[int, float, int]
    aa_a: CompareResult | None
    aa_b: CompareResult | None
    resamples: int
    seed: int
    verdict: str  # "B better" | "A better" | "within noise" | "too few items"
    higher_is_better: bool = True
    min_items: int = 1  # the fewest items that can show a difference at this confidence (see min_items)
    small_n: bool = False  # fewer than SMALL_N items (or min_items): the interval and p-value are rough


@dataclass
class LbNoise:
    """Leaderboard noise: the pooled SD of public scores of notebooks submitted more than once."""

    sd: float
    n_groups: int
    n_subs: int
    groups: dict[str, list[float]]


def compare(
    a: Mapping[str, np.ndarray],
    b: Mapping[str, np.ndarray],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    higher_is_better: bool = True,
) -> CompareResult:
    """Compare two arms given as {item: draws}. Only items with at least one value in both arms count.

    The verdict is "B better" or "A better" when the interval excludes 0 and the sign-flip p-value is below
    1 − confidence. With fewer than `min_items(confidence)` items no outcome can pass that test, and the
    verdict is "too few items": over a handful of items that all favour one side, every bootstrap mean has
    that sign, so the interval alone would always name a winner. Otherwise it is "within noise". Each arm
    also gets an A/A check (see `aa`) on the compared items.
    """
    result = _compare(
        a, b, resamples=resamples, seed=seed, confidence=confidence, higher_is_better=higher_is_better
    )
    ca, cb = _clean(a), _clean(b)
    compared = {k for k, v in ca.items() if v.size and k in cb and cb[k].size}
    for side, arm in (("a", ca), ("b", cb)):
        check = aa(
            {k: v for k, v in arm.items() if k in compared},
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            higher_is_better=higher_is_better,
        )
        setattr(result, f"aa_{side}", check)
    return result


def aa(
    arm: Mapping[str, np.ndarray],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    higher_is_better: bool = True,
) -> CompareResult | None:
    """An A/A check: split each item's draws alternately (0, 2, 4... vs 1, 3, 5...) and compare the halves.

    Uses the items with 2+ draws, and needs at least 5 of them (else None). Its verdict should be "within
    noise"; a winner means the halves differ, so the draws are probably not independent repeats. Below
    `min_items(confidence)` items (6 at 95%) the verdict is "too few items": only the interval is informative.
    """
    multi = {k: v for k, v in _clean(arm).items() if v.size >= 2}
    if len(multi) < AA_MIN_ITEMS:
        return None
    return _compare(
        {k: v[0::2] for k, v in multi.items()},
        {k: v[1::2] for k, v in multi.items()},
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        higher_is_better=higher_is_better,
    )


def min_items(confidence: float = 0.95) -> int:
    """The fewest items with which the sign-flip test can reach p < 1 − confidence.

    The smallest p-value n items can give is 2/2^n (every difference on one side), so with fewer items no
    outcome is significant at that confidence: 6 items at 95%, 5 at 90%, 8 at 99%.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    alpha = 1 - confidence
    n = 1
    while 2.0 ** (1 - n) >= alpha:
        n += 1
    return n


def spread(counts: Iterable[int]) -> tuple[int, float, int]:
    """(min, median, max) of draw counts; (0, 0.0, 0) when empty."""
    values = sorted(int(c) for c in counts)
    if not values:
        return (0, 0.0, 0)
    return (values[0], float(statistics.median(values)), values[-1])


def finite(value: object) -> float | None:
    """`value` as a float if it is a finite real number (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def lb_noise(subs: Iterable[Submission]) -> LbNoise | None:
    """Pooled SD of public scores over notebooks with 2+ scored submissions; None if there are none.

    sd = sqrt(sum((n_g - 1) * s_g^2) / sum(n_g - 1)), with s_g each notebook's sample SD. A score that is not
    a finite number (say, from a hand-edited event) is skipped.
    """
    by_nb: dict[str, list[float]] = {}
    for s in subs:
        score = finite(s.public_score)
        if score is None or not isinstance(s.nb, str) or not s.nb:
            continue
        by_nb.setdefault(s.nb, []).append(score)
    groups = {nb: scores for nb, scores in by_nb.items() if len(scores) >= 2}
    if not groups:
        return None
    dof = sum(len(v) - 1 for v in groups.values())
    pooled = sum((len(v) - 1) * statistics.variance(v) for v in groups.values()) / dof
    return LbNoise(math.sqrt(pooled), len(groups), sum(len(v) for v in groups.values()), groups)


def lb_diff(x: float, y: float, sd: float) -> tuple[float, float]:
    """(y − x, z): z measures the difference against two single submissions' noise, sd·√2."""
    diff = y - x
    if sd > 0:
        return diff, diff / (sd * math.sqrt(2))
    return diff, (math.copysign(math.inf, diff) if diff else 0.0)


# ----- internals -----------------------------------------------------------------------------------


def _clean(arm: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    import numpy as np

    out = {}
    for item, values in arm.items():
        x = np.asarray(values, dtype=float).ravel()
        out[str(item)] = x[~np.isnan(x)]
    return out


def _compare(
    a: Mapping[str, np.ndarray],
    b: Mapping[str, np.ndarray],
    *,
    resamples: int,
    seed: int,
    confidence: float,
    higher_is_better: bool,
) -> CompareResult:
    import numpy as np

    if resamples < 2:
        raise ValueError("resamples must be at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if seed < 0:
        raise ValueError("seed must not be negative")
    ca, cb = _clean(a), _clean(b)
    has_a = {k for k, v in ca.items() if v.size}
    has_b = {k for k, v in cb.items() if v.size}
    items = sorted(has_a & has_b)
    if not items:
        raise ValueError("no item has values in both arms")
    xa, xb = [ca[k] for k in items], [cb[k] for k in items]
    for side, xs in (("A", xa), ("B", xb)):
        for item, x in zip(items, xs, strict=True):
            if not np.isfinite(x).all():
                raise ValueError(f"item {item!r} has an infinite value in {side}")

    mean_a = np.array([x.mean() for x in xa])
    mean_b = np.array([x.mean() for x in xb])
    d = mean_b - mean_a
    boot_seq, perm_seq = np.random.SeedSequence(seed).spawn(2)
    boot = _bootstrap(xa, xb, resamples, np.random.default_rng(boot_seq))
    alpha = 1 - confidence
    lo, hi = (float(q) for q in np.quantile(boot, [alpha / 2, 1 - alpha / 2]))
    se = float(boot.std(ddof=1))
    p = _sign_flip_p(d, np.random.default_rng(perm_seq), _flips(alpha))

    better = d if higher_is_better else -d
    tie = np.abs(d) < TIE
    wins, losses = int((~tie & (better > 0)).sum()), int((~tie & (better < 0)).sum())
    need = min_items(confidence)
    if len(items) < need:
        verdict = TOO_FEW
    elif (lo > 0 or hi < 0) and p < alpha:  # both the interval and the permutation test see a difference
        verdict = "B better" if (lo > 0) == higher_is_better else "A better"
    else:
        verdict = WITHIN_NOISE
    return CompareResult(
        n_items=len(items),
        only_a=sorted(has_a - has_b),
        only_b=sorted(has_b - has_a),
        mean_a=float(mean_a.mean()),
        mean_b=float(mean_b.mean()),
        diff=float(d.mean()),
        ci_low=lo,
        ci_high=hi,
        confidence=float(confidence),
        se=se,
        mde80=MDE_Z * se,
        p_value=p,
        wins=wins,
        losses=losses,
        ties=int(tie.sum()),
        draws_a=spread(x.size for x in xa),
        draws_b=spread(x.size for x in xb),
        aa_a=None,
        aa_b=None,
        resamples=int(resamples),
        seed=int(seed),
        verdict=verdict,
        higher_is_better=higher_is_better,
        min_items=need,
        small_n=len(items) < max(SMALL_N, need),
    )


def _pad(xs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Ragged draws as a zero-padded (items, max draws) matrix plus the draw counts."""
    import numpy as np

    counts = np.array([x.size for x in xs], dtype=np.int64)
    padded = np.zeros((len(xs), int(counts.max())))
    for i, x in enumerate(xs):
        padded[i, : x.size] = x
    return padded, counts


def _resampled_means(
    rng: np.random.Generator, padded: np.ndarray, counts: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    """For each sampled item idx[r, j], the mean of its draws resampled with replacement."""
    import numpy as np

    width = padded.shape[1]
    if width == 1:
        return padded[idx, 0]
    c = counts[idx][..., None]
    pos = (rng.random((*idx.shape, width)) * c).astype(np.int64)
    np.minimum(pos, c - 1, out=pos)
    values = padded[idx[..., None], pos]
    if (counts == width).all():
        return values.mean(axis=-1)
    values[np.arange(width) >= c] = 0.0  # slots past an item's draw count
    return values.sum(axis=-1) / c[..., 0]


def _bootstrap(
    xa: Sequence[np.ndarray], xb: Sequence[np.ndarray], resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Two-stage bootstrap of the mean per-item difference B − A."""
    import numpy as np

    pa, ca = _pad(xa)
    pb, cb = _pad(xb)
    n = len(xa)
    chunk = max(1, _CHUNK // (n * (pa.shape[1] + pb.shape[1] + 1)))
    out = np.empty(resamples)
    for start in range(0, resamples, chunk):
        m = min(chunk, resamples - start)
        idx = rng.integers(0, n, size=(m, n))
        means_a = _resampled_means(rng, pa, ca, idx)
        means_b = _resampled_means(rng, pb, cb, idx)
        out[start : start + m] = (means_b - means_a).mean(axis=1)
    return out


def _flips(alpha: float) -> int:
    """Random sign flips for a p-value that can resolve `alpha`: RANDOM_FLIPS, or more when alpha is so
    small that they would give fewer than FLIPS_PER_ALPHA expected hits at p = alpha (up to MAX_FLIPS).

    A random-flip p-value is never below 1/(flips + 1), so too few flips would hide every difference."""
    return min(MAX_FLIPS, max(RANDOM_FLIPS, math.ceil(FLIPS_PER_ALPHA / alpha)))


def _sign_flip_p(d: np.ndarray, rng: np.random.Generator, flips: int = RANDOM_FLIPS) -> float:
    """Two-sided sign-flip permutation p-value of mean(d) = 0: exact up to 16 items, else `flips` random
    sign flips (the p-value is then at least 1/(flips + 1))."""
    import numpy as np

    n = d.size
    observed = abs(float(d.sum()))
    threshold = observed - 1e-9 * float(np.abs(d).sum())  # tolerate rounding in the sums
    if n <= EXACT_MAX_ITEMS:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
        return float((np.abs(signs @ d) >= threshold).mean())
    hits = 0
    chunk = max(1, _CHUNK // n)
    for start in range(0, flips, chunk):
        m = min(chunk, flips - start)
        signs = 1 - 2 * rng.integers(0, 2, size=(m, n), dtype=np.int8)
        hits += int((np.abs(signs @ d) >= threshold).sum())
    return (hits + 1) / (flips + 1)
