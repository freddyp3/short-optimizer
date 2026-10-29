"""Statistics for the hook test, numpy only (no scipy)."""

from __future__ import annotations

import math

import numpy as np


def rankdata(a) -> np.ndarray:
    """1-based ranks with ties given their average rank."""
    a = np.asarray(a, dtype=float)
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2
    return avg_rank[inverse]


def mann_whitney_u(x, y) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test, normal approximation with tie correction. Returns (U_x, p)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    u1 = rankdata(combined)[:n1].sum() - n1 * (n1 + 1) / 2
    n = n1 + n2
    _, counts = np.unique(combined, return_counts=True)
    var = n1 * n2 / 12 * ((n + 1) - (counts**3 - counts).sum() / (n * (n - 1)))
    if var <= 0:
        return float(u1), 1.0
    z = (u1 - n1 * n2 / 2) / math.sqrt(var)
    return float(u1), math.erfc(abs(z) / math.sqrt(2))
