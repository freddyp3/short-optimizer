"""Metadata + constant-timestamp baselines (plan sections 7.3, 7.4).

The confound check: if a dumb baseline (duration / publish date / title
features) also reaches >=26/40 on GATE, the selection is confounded and
Gemini's pass proves nothing. Each baseline is a leave-one-out
threshold classifier. Constant baselines for criterion B predict a fixed
ratio of the video length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def loo_threshold_predict(x, y) -> np.ndarray:
    """Leave-one-out predictions from the best single threshold on x.

    For each held-out point, scan candidate thresholds at midpoints of sorted
    unique train values in both directions and keep the one with max train
    accuracy (ties -> the lowest threshold wins deterministically).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    pred = np.empty(len(x), dtype=object)
    for i in range(len(x)):
        xi = np.delete(x, i)
        yi = np.delete(y, i)
        uniq = np.unique(xi)
        cands = (uniq[:-1] + uniq[1:]) / 2 if len(uniq) > 1 else uniq
        best_acc, best_pred = -1.0, "weak"
        for t in cands:
            for hi_is_strong in (True, False):
                p = np.where(xi > t, "strong" if hi_is_strong else "weak",
                             "weak" if hi_is_strong else "strong")
                acc = float(np.mean(p == yi))
                if acc > best_acc:
                    best_acc = acc
                    best_pred = ("strong" if hi_is_strong else "weak") if x[i] > t \
                        else ("weak" if hi_is_strong else "strong")
        pred[i] = best_pred
    return pred


def metadata_baselines(df: pd.DataFrame) -> dict:
    """LOO threshold baselines on duration, publish date, title length, caps ratio.

    df needs columns: label, duration_seconds, published_at, title.
    Returns {name: {correct, accuracy, predictions}}.
    """
    y = df["label"].to_numpy()
    feats = {
        "duration_only": df["duration_seconds"].astype(float).to_numpy(),
        "date_only": pd.to_datetime(df["published_at"]).astype("int64").to_numpy() / 1e9,
        "title_length": df["title"].fillna("").str.len().to_numpy(),
        "title_caps_ratio": df["title"].fillna("").map(
            lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s))
        ).to_numpy(),
    }
    out = {}
    for name, x in feats.items():
        if np.all(np.isnan(x)) or len(np.unique(x)) < 2:
            out[name] = {"skipped": "constant feature"}
            continue
        pred = loo_threshold_predict(np.nan_to_num(x, nan=np.nanmedian(x)), y)
        correct = int(np.sum(pred == y))
        out[name] = {"correct": correct, "n": len(y),
                     "accuracy": correct / len(y)}
    return out
