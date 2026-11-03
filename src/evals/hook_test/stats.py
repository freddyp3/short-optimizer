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


def binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) under Binomial(n, p), exact."""
    return float(
        sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def auc(labels, scores) -> float:
    """Mann-Whitney AUC: P(score_strong > score_weak) with ties counting 0.5."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def auc_permutation_p(labels, scores, n_perms: int = 10_000, seed: int = 0) -> float:
    """One-sided permutation p for AUC: shuffle labels, P(auc_null >= auc_obs)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    obs = auc(labels, scores)
    rng = np.random.default_rng(seed)
    count = 0
    lab = labels.copy()
    for _ in range(n_perms):
        rng.shuffle(lab)
        count += auc(lab, scores) >= obs
    return float((1 + count) / (1 + n_perms))


def auc_bootstrap_ci(labels, scores, n_boot: int = 10_000, seed: int = 0,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for AUC, resampling videos."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc(labels[idx], scores[idx])
        if a == a:  # skip degenerate resamples (all one class)
            vals.append(a)
    return tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def spearman(x, y) -> float:
    """Spearman rank correlation (ties -> average ranks)."""
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p_greater(obs: float, nulls) -> float:
    """(1 + #{null >= obs}) / (1 + n_nulls)."""
    nulls = np.asarray(nulls, dtype=float)
    return float((1 + np.sum(nulls >= obs)) / (1 + len(nulls)))
