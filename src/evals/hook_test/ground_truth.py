"""Ground-truth drop-off from a retention curve (plan/02-hook-test.md section 6).

Input: a video's curve ``a_i = audienceWatchRatio`` at ratios ``r_i`` (~i/100),
and the export ``duration_seconds`` D. Produces per-second loss windows, the
unrestricted (t*_all) and post-1.5s (t*_post) argmax drop-offs, a flat-curve
flag, and ``rank_score`` for scoring a predicted timestamp. Pure functions.

    from src.evals.hook_test.ground_truth import curve_gt, rank_score
"""

from __future__ import annotations

import numpy as np

from src.evals.hook_test.config import DROPOFF_MIN_T_S, FLAT_CURVE_MIN_RANGE
from src.evals.hook_test.stats import rankdata

EPS = 1e-9


def smooth3(a: np.ndarray) -> np.ndarray:
    """Centred 3-point moving average; edges use the shrunk window."""
    a = np.asarray(a, dtype=float)
    out = a.copy()
    for i in range(len(a)):
        lo, hi = max(0, i - 1), min(len(a), i + 2)
        out[i] = a[lo:hi].mean()
    return out


def loss_windows(ratios, values, duration_s: float):
    """Per-window relative loss over the next ~1 s.

    Returns (t_starts, losses, w): ``loss_i = (s_i - s_{i+w}) / max(s_i, eps)``
    over windows starting at ``t_i = r_i * D``. Negative loss means the curve
    rose (loops/rewatch); the sign is kept.
    """
    ratios = np.asarray(ratios, dtype=float)
    s = smooth3(np.asarray(values, dtype=float))
    t = ratios * duration_s
    w = max(1, round(len(s) / duration_s)) if duration_s > 0 else 1
    w = min(w, len(s) - 1) if len(s) > 1 else 1
    n = len(s) - w
    loss = (s[:n] - s[w:w + n]) / np.maximum(s[:n], EPS)
    return t[:n], loss, w


def window_index(t_starts: np.ndarray, p: float) -> int:
    """Index of the window containing prediction time p (clamped to range)."""
    i = int(np.searchsorted(t_starts, p, side="right")) - 1
    return int(np.clip(i, 0, len(t_starts) - 1))


def rank_score(p: float, t_starts, losses) -> float:
    """Percentile rank of the loss at the window containing p.

    1.0 = steepest-loss window, ~0.5 under a uniformly random p.
    """
    losses = np.asarray(losses, dtype=float)
    if len(losses) == 0:
        return float("nan")
    if len(losses) == 1:
        return 0.5
    i = window_index(np.asarray(t_starts, dtype=float), p)
    ranks = rankdata(losses)  # 1..n ascending; steepest loss gets n
    return float((ranks[i] - 1) / (len(losses) - 1))


def curve_gt(ratios, values, duration_s: float,
             min_t: float = DROPOFF_MIN_T_S,
             flat_range: float = FLAT_CURVE_MIN_RANGE) -> dict:
    """Full ground-truth record for one video's retention curve."""
    t, loss, w = loss_windows(ratios, values, duration_s)
    if len(loss) == 0:
        return {"flat_curve": True, "t_star_all": None, "t_star_post": None,
                "t": [], "loss": [], "w": w}
    i_all = int(np.argmax(loss))
    post = np.where(t >= min_t)[0]
    i_post = int(post[np.argmax(loss[post])]) if len(post) else i_all
    flat = float(loss.max() - np.median(loss)) < flat_range
    return {
        "flat_curve": flat,
        "t_star_all": float(t[i_all]),
        "t_star_post": float(t[i_post]),
        "t": t.tolist(),
        "loss": loss.tolist(),
        "w": int(w),
    }
