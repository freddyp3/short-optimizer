"""Per-second retention model on train + DEV (plan/03c section 16.1). GATE/HELDOUT are never loaded.

One row per video per 1 % of runtime; target = share of current viewers lost in that segment.

    ./venv/bin/python -m src.models.hazard            # CV table
    ./venv/bin/python -m src.models.hazard --perm     # plus the permutation tests (slow for the trees)

Why: 145 videos become 14,355 rows, and the payoff timeline Gemini extracted can be used second by second.
Folds are over videos, so a video's seconds never sit on both sides of a split. The permutation test gives every
video another video's covariates and keeps its own curve.

Results (2026-11-11, train + DEV, 145 videos; covariates added to a position + length baseline)
----------------------------------------------------------------------------------------------
                    per-second R2 gain   within-video rank corr. (baseline)   per-video late_hold   permutation p
    ridge           +0.082               0.43 (0.28)                          0.46                  0.002  (500 shuffles)
    boosted trees   +0.078               0.46 (0.32)                          0.41                  0.0099 (100 shuffles)

Both p values are the smallest their shuffle counts allow. The payoff timeline does say where in a video viewers
leave. This does not contradict the failed drop-off test in targets.py: that one asked for the single steepest
second, this one scores the whole profile. Per video it is better than ridge on late_hold (0.40) and below the
random forest (0.52); it adds nothing for early_hold (-0.01) and little for mid_hold (0.14). Its use is the
per-second curve itself: where in a draft the loss is expected.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evals.hook_test.ground_truth import smooth3
from src.evals.hook_test.stats import spearman
from src.models import cv, targets as T
from src.models.dataset import GEMINI_EVAL, GEMINI_TRAIN

RAW_DIR = Path("data/features/gemini/raw")
N_BINS, CAP_S, NEAR_S = 20, 10.0, 1.0
REPEATS = 3
PERMS = {"ridge": 500, "trees": 100}
P_MAX = 0.05 / len(PERMS)
SPANS = {"early_hold": (0.02, 0.25), "mid_hold": (0.25, 0.75), "late_hold": (0.75, 0.95)}
COVARIATES = ("payoff_here", "payoff_last_1s", "since_last_payoff_s", "until_next_payoff_s", "before_first_payoff",
              "after_last_payoff", "after_main_payoff", "tail_elapsed_s", "before_premise", "near_loudness_peak")


def segment_covariates(t: np.ndarray, dt: float, payoffs: list[dict], t_premise: float, t_loud: float) -> np.ndarray:
    """Covariates for segments starting at ``t`` (seconds) of length ``dt``; columns in COVARIATES order."""
    times = np.array(sorted(p["t_s"] for p in payoffs), dtype=float)
    out = np.zeros((len(t), len(COVARIATES)))
    out[:, 2] = out[:, 3] = CAP_S
    if len(times):
        weight = {p["t_s"]: p["weight"] for p in sorted(payoffs, key=lambda p: p["weight"])}
        main = min((p for p in payoffs if p["weight"] == max(q["weight"] for q in payoffs)), key=lambda p: p["t_s"])
        for i, ti in enumerate(t):
            here = [weight[x] for x in times if ti <= x < ti + dt]
            prev = [weight[x] for x in times if ti - 1.0 <= x < ti]
            past, future = times[times < ti], times[times >= ti]
            out[i, 0] = max(here, default=0)
            out[i, 1] = max(prev, default=0)
            if len(past):
                out[i, 2] = min(CAP_S, ti - past[-1])
            if len(future):
                out[i, 3] = min(CAP_S, future[0] - ti)
            out[i, 4] = not len(past)
            out[i, 5] = not len(future)
            out[i, 6] = ti > main["t_s"]
            out[i, 7] = min(CAP_S, ti - times[-1]) if not len(future) else 0.0
    else:
        out[:, 4] = 1.0
    out[:, 8] = t < t_premise
    out[:, 9] = np.abs(t - t_loud) <= NEAR_S if t_loud == t_loud else 0.0
    return out


def baseline_features(r: np.ndarray, duration_s: float, is_short: float) -> np.ndarray:
    bins = np.minimum((r * N_BINS).astype(int), N_BINS - 1)
    onehot = np.eye(N_BINS)[bins]
    return np.column_stack([onehot, onehot * is_short, np.full(len(r), np.log(duration_s))])


@dataclass
class Segments:
    video: np.ndarray       # row -> video position
    r: np.ndarray           # segment start as a share of runtime
    y: np.ndarray
    base: np.ndarray
    cov: np.ndarray
    n_videos: int
    per_video: int


def load_payoffs(video_ids) -> dict[str, list[dict]]:
    """Payoff lists from the raw record each video's frozen features came from. Reads only the ids asked for."""
    src = pd.concat([pd.read_csv(p, usecols=["video_id", "blind_id", "source_run_id"])
                     for p in (GEMINI_TRAIN, GEMINI_EVAL)]).set_index("video_id")
    return {v: json.loads((RAW_DIR / src.at[v, "source_run_id"] / f"{src.at[v, 'blind_id']}__r1.json")
                          .read_text())["result"]["payoffs"] for v in video_ids}


def build(df: pd.DataFrame, retention: pd.DataFrame, payoffs: dict[str, list[dict]]) -> Segments:
    curves = {v: g.sort_values("elapsedVideoTimeRatio") for v, g in retention.groupby("video_id")}
    vid, rs, ys, base, cov = [], [], [], [], []
    for k, row in enumerate(df.itertuples()):
        c = curves[row.video_id]
        r, s = c["elapsedVideoTimeRatio"].to_numpy()[:-1], smooth3(c["audienceWatchRatio"].to_numpy())
        t = r * row.duration_s
        vid.append(np.full(len(r), k))
        rs.append(r)
        ys.append((s[:-1] - s[1:]) / np.maximum(s[:-1], 1e-9))
        base.append(baseline_features(r, row.duration_s, row.is_short))
        cov.append(segment_covariates(t, row.duration_s / len(s), payoffs[row.video_id], row.t_premise_clear_s,
                                      row.t_loudness_peak_s))
    per = {len(x) for x in rs}
    if len(per) != 1:
        raise ValueError(f"curves have different lengths: {per}")
    return Segments(np.concatenate(vid), np.concatenate(rs), np.concatenate(ys), np.vstack(base), np.vstack(cov),
                    len(df), per.pop())


def fit_predict(model: str, Xtr, ytr, Xte) -> np.ndarray:
    if model == "ridge":
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        return cv.ridge_path((Xtr - mu) / sd, ytr, (Xte - mu) / sd, np.array([1.0]))[0][:, 0]
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=50,
                                         random_state=cv.SEED).fit(Xtr, ytr).predict(Xte)


def oof(seg: Segments, X: np.ndarray, model: str, folds) -> np.ndarray:
    """Out-of-fold predictions with folds over videos, so a video's seconds never straddle train and test."""
    pred = np.empty(len(seg.y))
    for f in folds:
        te = np.isin(seg.video, f)
        pred[te] = fit_predict(model, X[~te], seg.y[~te], X[te])
    return pred


def implied_holds(seg: Segments, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per-video hold over each span, rebuilt from per-segment losses: product of (1 - loss)."""
    keep = (1 - pred).reshape(seg.n_videos, seg.per_video)
    r = seg.r[: seg.per_video]
    return {name: keep[:, (r >= a - 1e-9) & (r < b - 1e-9)].prod(axis=1) for name, (a, b) in SPANS.items()}


def within_video_spearman(seg: Segments, pred: np.ndarray) -> float:
    P, Y = pred.reshape(seg.n_videos, -1), seg.y.reshape(seg.n_videos, -1)
    return float(np.nanmean([spearman(p, y) for p, y in zip(P, Y)]))


def shuffled_cov(seg: Segments, rng: np.random.Generator) -> np.ndarray:
    block = seg.cov.reshape(seg.n_videos, seg.per_video, -1)
    return block[rng.permutation(seg.n_videos)].reshape(seg.cov.shape)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--perm", action="store_true")
    args = p.parse_args(argv)
    df, _, _, export = T.load()
    seg = build(df, export.retention, load_payoffs(df["video_id"]))
    full = np.column_stack([seg.base, seg.cov])
    mse = lambda pr: float(((pr - seg.y) ** 2).mean())
    print(f"{seg.n_videos} videos, {len(seg.y)} segments")
    print(f"{'model':8}{'R2 gain':>9}{'gain>0':>8}{'within-video Sp':>17}{'(baseline)':>11}"
          + "".join(f"{k:>12}" for k in SPANS) + "   (baseline late)")
    for model in PERMS:
        gains, wv, wb, holds, hb = [], [], [], {k: [] for k in SPANS}, []
        for rep in range(REPEATS):
            folds = cv.kfold(seg.n_videos, cv.K, np.random.default_rng(cv.SEED + rep))
            pb, pf = oof(seg, seg.base, model, folds), oof(seg, full, model, folds)
            gains.append(1 - mse(pf) / mse(pb))
            wv.append(within_video_spearman(seg, pf))
            wb.append(within_video_spearman(seg, pb))
            for k, v in implied_holds(seg, pf).items():
                holds[k].append(spearman(v, df[k].to_numpy(dtype=float)))
            hb.append(spearman(implied_holds(seg, pb)["late_hold"], df["late_hold"].to_numpy(dtype=float)))
        rec = {"hazard_model": model, "r2_gain": float(np.mean(gains)), "repeats_gain_positive": int(np.sum(np.array(gains) > 0)),
               "within_video_spearman": float(np.mean(wv)), "within_video_spearman_baseline": float(np.mean(wb)),
               "late_hold_spearman_baseline": float(np.mean(hb))} | {f"{k}_spearman": float(np.mean(v)) for k, v in holds.items()}
        print(f"{model:8}{rec['r2_gain']:>9.4f}{rec['repeats_gain_positive']:>6}/{REPEATS}{rec['within_video_spearman']:>17.3f}"
              f"{rec['within_video_spearman_baseline']:>11.3f}" + "".join(f"{rec[f'{k}_spearman']:>12.3f}" for k in SPANS)
              + f"{rec['late_hold_spearman_baseline']:>12.3f}", flush=True)
        if args.perm:
            folds = cv.kfold(seg.n_videos, cv.K, np.random.default_rng(cv.SEED))
            late = df["late_hold"].to_numpy(dtype=float)
            base_mse = mse(oof(seg, seg.base, model, folds))
            def stat(c):
                pf = oof(seg, np.column_stack([seg.base, c]), model, folds)
                return 1 - mse(pf) / base_mse, spearman(implied_holds(seg, pf)["late_hold"], late)
            obs = stat(seg.cov)
            rng = np.random.default_rng(cv.SEED + 1)
            null = np.array([stat(shuffled_cov(seg, rng)) for _ in range(PERMS[model])])
            ps = (1 + (null >= np.array(obs)[None, :]).sum(axis=0)) / (1 + len(null))
            rec |= {"p_r2_gain": float(ps[0]), "p_late_hold": float(ps[1]), "n_perm": len(null),
                    "pass": bool(rec["repeats_gain_positive"] == REPEATS and ps[0] <= P_MAX)}
            print(f"{'':8}permutation ({len(null)} shuffles): p(R2 gain) = {ps[0]:.4f}, p(late_hold) = {ps[1]:.4f}, "
                  f"pass = {rec['pass']}", flush=True)
        T.append_log(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
