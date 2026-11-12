"""Confidence intervals for a share measured on a sample.

The Wilson score interval stays inside 0 to 1 and behaves at small samples and at shares near
the ends, where the normal approximation does not: a sample of about 25 judged claims, nearly
all faithful, is exactly that case."""

import math
from collections.abc import Sequence

import numpy as np

Z_95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The interval for the share successes / n; the whole range when nothing was sampled."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261111


def weighted_share(
    weights: Sequence[float],
    hits: Sequence[bool],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, tuple[float, float]]:
    """The self-normalized share, weighted hits over total weight, which stays inside 0 to 1,
    and its 95% percentile interval from a seeded bootstrap over the units."""
    w = np.asarray(weights, dtype=np.float64)
    y = np.asarray(hits, dtype=np.float64)
    if w.size == 0:
        return None, (0.0, 1.0)
    share = float((w * y).sum() / w.sum())
    picks = np.random.default_rng(seed).integers(0, w.size, size=(draws, w.size))
    shares = (w[picks] * y[picks]).sum(axis=1) / w[picks].sum(axis=1)
    lo, hi = np.quantile(shares, [0.025, 0.975]).tolist()
    return share, (float(lo), float(hi))
