"""Statistics shared by the analyses. Pure Python, no dependencies."""

from __future__ import annotations

import math
from collections import Counter
from typing import Hashable, Iterable


def entropy(values: Iterable[Hashable]) -> float:
    """Shannon entropy in bits of the empirical distribution of `values`."""
    values = list(values)
    if not values:
        raise ValueError("entropy requires at least one value")
    n = len(values)
    return -sum((c / n) * math.log2(c / n) for c in Counter(values).values())


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))

    def cdf(kk, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))

    def bisect(f):
        lo, hi = 0.0, 1.0
        inc = f(hi) > f(lo)
        for _ in range(80):
            mid = (lo + hi) / 2
            if (f(mid) < 0) == inc:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lo = 0.0 if k == 0 else bisect(lambda p: (1 - cdf(k - 1, p)) - alpha / 2)
    hi = 1.0 if k == n else bisect(lambda p: cdf(k, p) - alpha / 2)
    return lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
