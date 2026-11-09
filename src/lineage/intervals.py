"""Confidence intervals for a share measured on a sample.

The Wilson score interval stays inside 0 to 1 and behaves at small samples and at shares near
the ends, where the normal approximation does not: a sample of about 25 judged claims, nearly
all faithful, is exactly that case."""

import math

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
