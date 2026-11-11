"""Other targets and drop-off timing on train + DEV (plan/03c section 14). GATE/HELDOUT are never loaded.

    ./venv/bin/python -m src.models.targets run        # nested CV, 8 targets x 4 recipes
    ./venv/bin/python -m src.models.targets perm       # one permutation test over all 32 pairs
    ./venv/bin/python -m src.models.targets dropoff    # do extracted timestamps mark where viewers leave

The list of 8 targets x 4 recipes was fixed before anything was computed, and one permutation test covers all 32
pairs (each shuffle moves the eight labels together and keeps the best of the 32).

Results (2026-11-10, train + DEV, 145 videos, ridge; nested-CV R2, adjusted p)
------------------------------------------------------------------------------
Acceptable (beats every baseline in >= 9/10 repeats, adjusted p <= 0.05, holds on the views-corrected label):

    late_hold      M1 core 0.087 (p 0.001)   M3 full 0.151 (p 0.001)
    share_rate     M1 core 0.103 (p 0.005)   M3 full 0.214 (p 0.001)   format only 0.179 (p 0.002)
    early_hold     M1 core 0.089 (p 0.007)
    replay_level   M1 core 0.058 (p 0.032)   (nearly the same quantity as early_hold: Spearman -0.77)

Not acceptable: mid_hold, mean_rrp, like_rate, comment_rate, and the code-only features (M4) on every target.

late_hold coefficients have the hypothesised signs: later main payoff +, more payoffs +, longer tail after the
last payoff -. share_rate is mostly format (gameplay clips and loot openings are shared less than edits).
A flexible duration + date baseline explains none of these targets, so this is not a leftover length effect.
The overall score fails although its parts work because features push parts of the curve in opposite directions
(a late payoff lowers early_hold and raises late_hold).

Drop-off timing fails (mean rank of the loss at the predicted second; 0.5 = chance, pass >= 0.60 and both nulls):
last_payoff 0.46, main_payoff 0.46, loudness_peak 0.52. premise_clear scores 0.80 only because it is always in the
first seconds, where every video loses viewers (p 0.43 against other videos' premise times).

Limits: CV on videos that were visible, small effects, and the extractor's accuracy was never checked by hand.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.data import load_export
from src.evals.hook_test.score import criterion_b, ground_truth_for, residualize
from src.models import cv
from src.models.dataset import load_split
from src.models.experiments import MIN_REPEATS_BEATING, OUT_DIR, P_MAX, all_recipes, git_sha
from src.models.features import Recipe, prepare

LOG = OUT_DIR / "log_targets.jsonl"
CURVE_TARGETS = ("early_hold", "mid_hold", "late_hold", "replay_level", "mean_rrp")
RATE_TARGETS = ("like_rate", "share_rate", "comment_rate")
TARGETS = CURVE_TARGETS + RATE_TARGETS
RECIPES = ("M1_core_mechanics", "M3_full_static", "M4_code_only", "c06_format_only")
BASELINES = ("mean_only", "date_only", "duration_only")
DROPOFF_RULES = ("last_payoff", "main_payoff", "premise_clear", "loudness_peak")
DROPOFF_P_MAX = 0.05 / len(DROPOFF_RULES)


def curve_targets(retention: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vid, g in retention.sort_values("elapsedVideoTimeRatio").groupby("video_id"):
        x, y = g["elapsedVideoTimeRatio"].to_numpy(), g["audienceWatchRatio"].to_numpy()
        w = lambda r: float(np.interp(r, x, y))
        rows.append({"video_id": vid, "early_hold": w(0.25) / w(0.02), "mid_hold": w(0.75) / w(0.25),
                     "late_hold": w(0.95) / w(0.75), "replay_level": w(0.02),
                     "mean_rrp": float(g["relativeRetentionPerformance"].mean())})
    return pd.DataFrame(rows)


def rate_targets(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics[metrics["creatorContentType"] == "shorts"]
    out = m[["video_id"]].copy()
    for name, col in zip(RATE_TARGETS, ("likes", "shares", "comments")):
        out[name] = np.log((m[col] + 1) / m["engagedViews"])
    return out


def residual_on_views(y: np.ndarray, views: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(y)), np.log(views)])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def add_targets(df: pd.DataFrame, export) -> pd.DataFrame:
    """Raw targets joined on video_id, then residualised on log(duration) and era using these rows only."""
    df = df.drop(columns=["mean_rrp"], errors="ignore")  # same definition, recomputed below
    df = df.merge(curve_targets(export.retention), on="video_id", how="left", validate="one_to_one")
    df = df.merge(rate_targets(export.metrics), on="video_id", how="left", validate="one_to_one")
    if df[list(TARGETS)].isna().any().any():
        raise ValueError("some videos have no retention curve or metrics row")
    for t in TARGETS:
        df[f"{t}_raw"] = df[t]
        df[t] = residualize(df[t], df["duration_seconds"], df["published_at"]).to_numpy()
        df[f"{t}_views"] = residual_on_views(df[t].to_numpy(), df["engagedViews"].to_numpy(dtype=float))
    return df


def load():
    df = pd.concat([load_split("train"), load_split("dev")], ignore_index=True)
    export = load_export(Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"]))
    df = add_targets(df, export)
    data = prepare(df)
    recipes = all_recipes(data)
    m3 = recipes["M3_full"]
    recipes["M3_full_static"] = Recipe("M3_full_static", cols=m3.cols)
    return df, data, recipes, export


def append_log(rec: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": git_sha()}
                           | rec) + "\n")


def run() -> int:
    df, data, recipes, _ = load()
    print(f"n = {len(df)}")
    print(f"{'target':14}{'recipe':22}{'params':>7}{'R2':>8}{'Spearman':>10}{'beats':>7}{'Sp(views)':>11}")
    for t in TARGETS:
        y, yv = df[t].to_numpy(dtype=float), df[f"{t}_views"].to_numpy(dtype=float)
        base = {b: cv.nested_cv(data, y, recipes[b]) for b in BASELINES}
        for n in RECIPES:
            r, rv = cv.nested_cv(data, y, recipes[n]), cv.nested_cv(data, yv, recipes[n])
            r2 = float((1 - r["mse"] / base["mean_only"]["mse"]).mean())
            beats = min(int((r["mse"] < base[b]["mse"]).sum()) for b in BASELINES)
            rec = {"target": t, "recipe": n, "n_params": recipes[n].n_params, "r2": r2,
                   "spearman": float(r["spearman"].mean()), "repeats_beating_every_baseline": beats,
                   "spearman_views": float(rv["spearman"].mean()), "alpha_median": r["alpha_median"]}
            append_log(rec)
            print(f"{t:14}{n:22}{rec['n_params']:>7}{r2:>8.3f}{rec['spearman']:>10.3f}{beats:>5}/10"
                  f"{rec['spearman_views']:>11.3f}")
    return 0


def perm(n_perm: int) -> int:
    df, data, recipes, _ = load()
    Y = df[list(TARGETS)].to_numpy(dtype=float)
    folds = cv.kfold(len(data), cv.K, np.random.default_rng(cv.SEED))
    stat = lambda Ym: np.array([[cv.perm_stat(data, Ym[:, j], recipes[n], folds) for n in RECIPES]
                                for j in range(Ym.shape[1])])
    obs = stat(Y)
    rng = np.random.default_rng(cv.SEED + 1)
    null = np.array([stat(Y[rng.permutation(len(Y))]) for _ in range(n_perm)])
    p = (1 + (null >= obs[None]).sum(axis=0)) / (1 + n_perm)
    p_adj = (1 + (null.max(axis=(1, 2))[:, None, None] >= obs[None]).sum(axis=0)) / (1 + n_perm)
    latest = {(r["target"], r["recipe"]): r for line in LOG.read_text().splitlines()
              if "recipe" in (r := json.loads(line))}
    print(f"{len(TARGETS) * len(RECIPES)} target x recipe pairs, {n_perm} shuffles")
    print(f"{'target':14}{'recipe':22}{'Sp(1x10)':>9}{'p':>8}{'p_adj':>8}{'beats':>7}{'Sp(views)':>11}  acceptable")
    out = []
    for j, t in enumerate(TARGETS):
        for i, n in enumerate(RECIPES):
            r = latest[(t, n)]
            ok = bool(r["repeats_beating_every_baseline"] >= MIN_REPEATS_BEATING and p_adj[j, i] <= P_MAX
                      and r["spearman_views"] > 0)
            out.append({"target": t, "recipe": n, "spearman_1x10": float(obs[j, i]), "p": float(p[j, i]),
                        "p_adjusted": float(p_adj[j, i]), "acceptable": ok})
            print(f"{t:14}{n:22}{obs[j, i]:>9.3f}{p[j, i]:>8.3f}{p_adj[j, i]:>8.3f}"
                  f"{r['repeats_beating_every_baseline']:>5}/10{r['spearman_views']:>11.3f}  {'YES' if ok else 'no'}")
    append_log({"permutation": out, "n_perm": n_perm})
    return 0


def dropoff_predictions(df: pd.DataFrame) -> dict[str, pd.Series]:
    never_clear = df["t_premise_clear_s"] >= df["duration_s"]
    return {"last_payoff": df["duration_s"] - df["tail_all"], "main_payoff": df["t_main_payoff_s"],
            "premise_clear": df["t_premise_clear_s"].where(~never_clear), "loudness_peak": df["t_loudness_peak_s"]}


def dropoff() -> int:
    df, _, _, export = load()
    gts = ground_truth_for(df, export.retention)
    print(f"n = {len(df)}; pass needs T >= {C.T_OBS_MIN} and both p < {DROPOFF_P_MAX:.4f}")
    print(f"{'rule':16}{'scored':>7}{'T_obs':>7}{'p random':>10}{'p shuffled':>12}{'hit<=1s':>9}  pass")
    for rule, pred in dropoff_predictions(df).items():
        b = criterion_b(df.assign(dropoff_pred_s=pred), gts)
        ok = bool(b["T_obs"] >= C.T_OBS_MIN and b["null1_uniform_p"] < DROPOFF_P_MAX
                  and b["null2_permutation_p"] < DROPOFF_P_MAX)
        append_log({"dropoff_rule": rule, "pass": ok} | {k: v for k, v in b.items() if k != "flat_ids"})
        print(f"{rule:16}{b['n_scored']:>7}{b['T_obs']:>7.3f}{b['null1_uniform_p']:>10.4f}"
              f"{b['null2_permutation_p']:>12.4f}{b['hit_1s']:>9.2f}  {'YES' if ok else 'no'}")
    c = b["constant_baselines"]
    print(f"constants: 0.3D {c['0.3D']:.3f}, 0.5D {c['0.5D']:.3f}, median ratio ({c['median_ratio']:.2f}) "
          f"{c['median_ratio_score']:.3f}; flat curves skipped: {b['n_flat']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("perm").add_argument("--n-perm", type=int, default=cv.N_PERM)
    sub.add_parser("dropoff")
    a = p.parse_args(argv)
    return run() if a.cmd == "run" else perm(a.n_perm) if a.cmd == "perm" else dropoff()


if __name__ == "__main__":
    raise SystemExit(main())
