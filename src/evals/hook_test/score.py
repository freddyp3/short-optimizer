"""Score a hook-test run: criteria A/B, baselines, stability, decision (plan sections 7, 12).

Reads ``data/evals/hook_test/raw/<run_id>/`` (or ``--raw-dir``), joins with
selection.csv labels and the export's retention curves, writes
``results/<run_id>/scores.json`` + ``report.md``. DEV runs also append a line to
``dev_log.jsonl``. For HELDOUT pass ``--pool-with <gate_run_id>`` to apply the
pooled rule from the decision tree.

    ./venv/bin/python -m src.evals.hook_test.score --run <run_id>
    ./venv/bin/python -m src.evals.hook_test.score --run <heldout_id> --pool-with <gate_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.baselines import metadata_baselines
from src.evals.hook_test.data import load_export
from src.evals.hook_test.ground_truth import curve_gt, rank_score
from src.evals.hook_test.stats import (
    auc, auc_bootstrap_ci, auc_permutation_p, binom_sf, permutation_p_greater,
    spearman, wilson_ci)

log = logging.getLogger(__name__)

CAVEATS = """\
- n=40 is small. Power of A (>=26/40) if true accuracy is 0.60/0.65/0.70/0.75/0.80: 32% / 57% / 81% / 95% / 99%. A true 65% model fails the gate 43% of the time, so a no-go at 22-25/40 is "no evidence", not "evidence of none".
- Labels are noisy (finite views, retention estimates, algorithmic reach differences), so the accuracy ceiling is below 100%; the extreme-quartile design raises the ceiling but also makes the task easier than the real distribution. The held-out number on the full 25/75 range at Step 3 is the one to quote externally, not this one.
- One channel, one game, one creator's style: result does not generalise to other channels (relevant to the "500 creators" goal; Phase 8 deals with it).
- Videos are not exchangeable in time; stratification reduces but does not remove era confounds.
- 3 runs per video reduce sampling noise from the model but do not add independent videos; n stays 40 for the binomial test. Do not multiply n by the number of runs.
- Picking best of several prompt versions inflates results; this is why DEV tuning is separated from GATE and GATE runs once.
- The gate is a one-sided test at ~0.04-0.05, not a measure of usefulness. Usefulness is Phase 3's job.
- A and B are two shots at the same question, so "A or B" has a false-positive rate of about 8.8% under the null."""


# ------------------------------------------------------------------ loading

def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Flatten raw/<run_id>/*.json into one row per (blind_id, run_k)."""
    rows = []
    for f in sorted(Path(raw_dir).glob("*__r*.json")):
        rec = json.loads(f.read_text())
        r = rec.get("result") or {}
        rows.append({
            "blind_id": rec["blind_id"],
            "run_k": rec.get("run_k", int(f.stem.split("__r")[-1])),
            "status": rec["status"],
            "invalid_reason": rec.get("invalid_reason"),
            "p_strong": r.get("p_strong"),
            "pred_avp": r.get("predicted_avg_view_pct"),
            "schema_version": rec.get("schema_version", "v1"),
            "group": r.get("group"),
            "dropoff": r.get("dropoff_timestamp_s"),
            "length_est": r.get("video_length_s_estimate"),
            "model": rec.get("model"),
            "prompt_version": rec.get("prompt_version"),
            "fps": rec.get("fps"),
            "media_resolution": rec.get("media_resolution"),
            "total_tokens": (rec.get("usage") or {}).get("total_tokens", 0),
            "cost_usd_paid_equiv": rec.get("cost_usd_paid_equiv", 0),
            "cost_usd_actual": rec.get("cost_usd_actual", 0),
            "latency_s": rec.get("latency_s"),
        })
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-video: n_valid, p_strong_mean, dropoff median, group list (plan 7.1)."""
    ok = raw[raw["status"] == "ok"]
    out = ok.groupby("blind_id").agg(
        n_valid=("status", "size"),
        p_strong_mean=("p_strong", "mean"),
        pred_avp_mean=("pred_avp", "mean"),
        dropoff_pred_s=("dropoff", "median"),
        length_est_mean=("length_est", "mean"),
        dropoff_std=("dropoff", "std"),
    )
    grp = ok.groupby("blind_id")["group"].apply(list)
    out["groups"] = grp
    return out.reset_index()


def ground_truth_for(videos: pd.DataFrame, retention: pd.DataFrame) -> dict:
    """blind_id -> curve_gt record, using export duration_seconds as D."""
    gts = {}
    ret = retention.groupby("video_id")
    for _, v in videos.iterrows():
        if v["video_id"] not in ret.groups:
            continue
        cur = ret.get_group(v["video_id"]).sort_values("elapsedVideoTimeRatio")
        gts[v["blind_id"]] = curve_gt(
            cur["elapsedVideoTimeRatio"].to_numpy(),
            cur["audienceWatchRatio"].to_numpy(),
            float(v["duration_seconds"]))
    return gts


# ------------------------------------------------------------------ criteria

def residualize(y: pd.Series, duration_s: pd.Series, published_at) -> pd.Series:
    """OLS residual of y on [1, log(duration), is_late_era] (ERA_CUTOFF),
    fit on the given rows. Drops the era column when it is constant."""
    import numpy as np
    from src.evals.hook_test.categorize import ERA_CUTOFF
    y, dur = np.asarray(y, float), np.asarray(duration_s, float)
    late = (pd.to_datetime(pd.Series(published_at)) >= ERA_CUTOFF).astype(float).to_numpy()
    cols = [np.ones(len(y)), np.log(dur)]
    if np.unique(late).size > 1:
        cols.append(late)
    X = np.column_stack(cols)
    ok = ~(np.isnan(y) | np.isnan(dur))
    out = np.full(len(y), np.nan)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    out[ok] = y[ok] - X[ok] @ coef
    return pd.Series(out)


def criterion_a(videos: pd.DataFrame, seed: int = C.SEED,
                n_perms: int = C.PERMS, n_boot: int = C.PERMS,
                score_col: str = "score") -> dict:
    """Rank-based group call + binomial + AUC + Spearman (plan 7.2)."""
    df = videos.dropna(subset=[score_col]).copy()
    n = len(df)
    if n == 0:
        return {"n": 0}
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["pred"] = np.where(df.index < n / 2, "strong", "weak")
    correct = int((df["pred"] == df["label"]).sum())
    labels01 = (df["label"] == "strong").astype(int).to_numpy()
    scores = df[score_col].to_numpy()
    lo, hi = wilson_ci(correct, n)
    blo, bhi = auc_bootstrap_ci(labels01, scores, n_boot=n_boot, seed=seed)
    hard = df.explode("groups").dropna(subset=["groups"])
    hard_acc = float((hard["groups"] == hard["label"]).mean()) if len(hard) else float("nan")
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "p_binom": binom_sf(correct, n),
        "wilson_ci": [lo, hi],
        "pass": correct >= math.ceil(n * C.GATE_PASS / C.GATE_N),
        "hard_group_accuracy": hard_acc,
        "auc": auc(labels01, scores),
        "auc_perm_p": auc_permutation_p(labels01, scores, n_perms, seed),
        "auc_boot_ci": [blo, bhi],
        "spearman_hold_resid": spearman(scores, df["hold_resid"].to_numpy()),
    }


def criterion_b(videos: pd.DataFrame, gts: dict, seed: int = C.SEED,
                n_perms: int = C.PERMS) -> dict:
    """T_obs = mean rank_score of the predicted drop-off vs both nulls (plan 7.4)."""
    scored, flat = [], []
    for _, v in videos.iterrows():
        gt = gts.get(v["blind_id"])
        if gt is None:
            continue
        if gt["flat_curve"]:
            flat.append(v["blind_id"])
            continue
        if pd.isna(v["dropoff_pred_s"]):
            continue
        D = float(v["duration_seconds"])
        scored.append({
            "blind_id": v["blind_id"], "D": D,
            "p": float(v["dropoff_pred_s"]),
            "rank_score": rank_score(v["dropoff_pred_s"], gt["t"], gt["loss"]),
            "err": abs(v["dropoff_pred_s"] - gt["t_star_post"]) / D,
            "hit_1s": abs(v["dropoff_pred_s"] - gt["t_star_post"]) <= 1.0,
            "t": gt["t"], "loss": gt["loss"], "t_star_post": gt["t_star_post"],
        })
    if not scored:
        return {"n_scored": 0, "n_flat": len(flat), "flat_ids": flat}
    rng = np.random.default_rng(seed)
    t_obs = float(np.mean([s["rank_score"] for s in scored]))

    null1 = np.empty(n_perms)
    for b in range(n_perms):
        null1[b] = np.mean([
            rank_score(rng.uniform(C.DROPOFF_MIN_T_S, s["D"]), s["t"], s["loss"])
            for s in scored])
    ratios = np.array([s["p"] / s["D"] for s in scored])
    null2 = np.empty(n_perms)
    for b in range(n_perms):
        perm = rng.permutation(len(scored))
        null2[b] = np.mean([
            rank_score(ratios[perm[i]] * s["D"], s["t"], s["loss"])
            for i, s in enumerate(scored)])

    med_ratio = float(np.median([s["t_star_post"] / s["D"] for s in scored]))
    const = {
        "0.3D": float(np.mean([rank_score(0.3 * s["D"], s["t"], s["loss"]) for s in scored])),
        "0.5D": float(np.mean([rank_score(0.5 * s["D"], s["t"], s["loss"]) for s in scored])),
        "median_ratio": med_ratio,
        "median_ratio_score": float(np.mean(
            [rank_score(med_ratio * s["D"], s["t"], s["loss"]) for s in scored])),
    }
    p1 = permutation_p_greater(t_obs, null1)
    p2 = permutation_p_greater(t_obs, null2)
    return {
        "n_scored": len(scored),
        "n_flat": len(flat),
        "flat_ids": flat,
        "T_obs": t_obs,
        "null1_uniform_p": p1,
        "null2_permutation_p": p2,
        "pass": t_obs >= C.T_OBS_MIN and p1 < C.P_MAX and p2 < C.P_MAX,
        "mean_err": float(np.mean([s["err"] for s in scored])),
        "hit_1s": float(np.mean([s["hit_1s"] for s in scored])),
        "constant_baselines": const,
    }


def stability(raw: pd.DataFrame, schema_version: str = "v1",
              meta: pd.DataFrame | None = None) -> dict:
    """Run-to-run stability: pairwise Spearman of the per-run ranking score
    (p_strong for v1, residualized pred_avp for v2), group flips, dropoff std.
    meta (needed for v2): per-blind_id duration_seconds + published_at."""
    ok = raw[raw["status"] == "ok"]
    if schema_version == "v2" and meta is not None:
        piv_raw = ok.pivot_table(index="blind_id", columns="run_k",
                                 values="pred_avp")
        m = meta.set_index("blind_id")
        piv = pd.DataFrame(index=piv_raw.index)
        for k in piv_raw.columns:
            # residualize this run's pred_avp with the same fit form
            col = piv_raw[k]
            piv[k] = residualize(col, m.loc[col.index, "duration_seconds"],
                                 m.loc[col.index, "published_at"])
    elif schema_version == "v2":
        piv = ok.pivot_table(index="blind_id", columns="run_k", values="pred_avp")
    else:
        piv = ok.pivot_table(index="blind_id", columns="run_k", values="p_strong")
    ks = list(piv.columns)
    sps = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            pair = piv[[ks[i], ks[j]]].dropna()
            if len(pair) >= 5:
                sps.append(spearman(pair[ks[i]], pair[ks[j]]))
    grp = ok.groupby("blind_id")["group"].nunique()
    dp = ok.groupby("blind_id")["dropoff"].std()
    return {
        "mean_pairwise_spearman": float(np.nanmean(sps)) if sps else float("nan"),
        "pairwise_spearmans": sps,
        "group_flip_share": float((grp > 1).mean()) if len(grp) else float("nan"),
        "mean_dropoff_std_s": float(dp.mean()) if len(dp) else float("nan"),
    }


# ------------------------------------------------------------------ decision

def decide(A: int, B_pass: bool, C_confounded: bool) -> str:
    """Section 12 decision tree for the GATE split (pooled rule lives in score_run)."""
    if C_confounded:
        return "INVALID_CONFOUNDED"
    if A >= C.GATE_PASS:
        return "GO"
    if A >= C.AMBIGUOUS_LO:
        return "CONDITIONAL_GO" if B_pass else "AMBIGUOUS_RUN_HELDOUT"
    if B_pass:
        return "CONDITIONAL_GO_SCOPED"
    return "NO_GO"


# ------------------------------------------------------------------ report

def score_run(raw_dir: Path, run_id: str, selection: pd.DataFrame | None = None,
              export_dir: Path | None = None, titles: dict | None = None,
              retention: pd.DataFrame | None = None,
              pool_raw: pd.DataFrame | None = None,
              out_dir: Path | None = None,
              n_perms: int = C.PERMS) -> dict:
    sel = selection if selection is not None else pd.read_csv(C.SELECTION_CSV_PATH)
    raw = load_raw(raw_dir)
    split = None
    settings_p = Path(raw_dir) / "settings.json"
    if settings_p.exists():
        split = json.loads(settings_p.read_text()).get("split")
    vids_b = sel.merge(raw[["blind_id"]].drop_duplicates(), on="blind_id")
    if split is None:
        split = vids_b["split"].mode().iat[0]
    vids = sel[sel["split"] == split].copy()

    exp = load_export(export_dir) if export_dir or (C.EXPORT_BASE).exists() else None
    if retention is None:
        retention = exp.retention if exp else pd.DataFrame()
    if titles is None and exp is not None:
        titles = exp.videos.set_index("video_id")["title"].to_dict()
    vids["title"] = vids["video_id"].map(titles or {})

    agg = aggregate(raw)
    vids = vids.merge(agg, on="blind_id", how="left")
    vids["n_valid"] = vids["n_valid"].fillna(0).astype(int)
    n_invalid = int((vids["n_valid"] == 0).sum())

    schema_version = raw["schema_version"].mode().iat[0] \
        if "schema_version" in raw.columns and raw["schema_version"].notna().any() else "v1"
    if schema_version == "v2":
        # ranking score = pred_avp residualized on [1, log(dur), is_late_era],
        # fit on this split's predictions only (never labels)
        vids["score"] = residualize(
            vids["pred_avp_mean"], vids["duration_seconds"], vids["published_at"])
    else:
        vids["score"] = vids["p_strong_mean"]

    gts = ground_truth_for(vids, retention) if len(retention) else {}
    A = criterion_a(vids, n_perms=n_perms, n_boot=n_perms, score_col="score")
    if schema_version == "v2":
        A["hard_group_accuracy"] = None  # v2 has no hard group call
    B = criterion_b(vids, gts, n_perms=n_perms)
    stab = stability(raw, schema_version,
                     meta=vids[["blind_id", "duration_seconds", "published_at"]])
    base = metadata_baselines(vids.dropna(subset=["title"]))
    confounded = any(
        b.get("correct", 0) >= C.GATE_PASS * len(vids) / C.GATE_N
        for b in base.values() if isinstance(b, dict))

    usage = {
        "total_tokens": int(raw["total_tokens"].sum()),
        "calls": len(raw),
        "n_ok": int((raw["status"] == "ok").sum()),
        "n_invalid_calls": int((raw["status"] == "invalid").sum()),
        "cost_usd_paid_equiv": round(float(raw["cost_usd_paid_equiv"].sum()), 4),
        "cost_usd_actual": round(float(raw["cost_usd_actual"].sum()), 4),
        "models": sorted(raw["model"].dropna().unique().tolist()),
        "prompt_versions": sorted(raw["prompt_version"].dropna().unique().tolist()),
    }
    length_err = vids.dropna(subset=["length_est_mean"])
    length_est = {
        "mean_abs_err_s": float((length_err["length_est_mean"] - length_err["duration_seconds"]).abs().mean())
        if len(length_err) else float("nan"),
    }

    v2_diag = None
    if schema_version == "v2":
        d = vids.dropna(subset=["pred_avp_mean"])
        v2_diag = {
            "spearman_pred_avp_hold_score": spearman(d["pred_avp_mean"], d["hold_score"]),
            "mae_pred_avp_hold_score": float((d["pred_avp_mean"] - d["hold_score"]).abs().mean()),
            "spearman_pred_avp_duration": spearman(d["pred_avp_mean"], d["duration_seconds"]),
            "spearman_score_duration": spearman(d["score"], d["duration_seconds"]),
            "pred_avp_summary": {
                "min": float(d["pred_avp_mean"].min()),
                "median": float(d["pred_avp_mean"].median()),
                "mean": float(d["pred_avp_mean"].mean()),
                "max": float(d["pred_avp_mean"].max()),
                "std": float(d["pred_avp_mean"].std()),
            },
        }

    scores = {
        "run_id": run_id, "split": split, "n_videos": len(vids),
        "schema_version": schema_version,
        "v2_diagnostics": v2_diag,
        "n_invalid_videos": n_invalid,
        "criterion_A": A, "criterion_B": B, "stability": stab,
        "metadata_baselines": base, "confounded": confounded,
        "usage": usage, "video_length_s_estimate": length_est,
    }

    if split in ("gate", "heldout") and n_invalid > 2:
        scores["decision"] = "INVALID_INFRA"
        scores["decision_note"] = f"{n_invalid} invalid videos (>2); fix infra and re-run only those."
    elif split == "gate":
        scores["decision"] = decide(A.get("correct", 0), B.get("pass", False), confounded)
    elif split == "heldout":
        if pool_raw is not None:
            pvids = sel.merge(pool_raw[["blind_id"]].drop_duplicates(), on="blind_id")
            pvids = pvids.merge(aggregate(pool_raw), on="blind_id", how="left")
            pA = criterion_a(pvids)
            pooled = {"heldout_correct": int(A.get("correct", 0)),
                      "gate_correct": int(pA.get("correct", 0)),
                      "correct": int(A.get("correct", 0)) + pA.get("correct", 0),
                      "n": len(vids) + len(pvids),
                      "B_pass_heldout": B.get("pass", False)}
            need = math.ceil(pooled["n"] * C.POOLED_PASS / C.POOLED_N)
            scores["pooled"] = pooled
            scores["decision"] = ("GO_POOLED"
                                  if pooled["correct"] >= need or pooled["B_pass_heldout"]
                                  else "NO_GO_POOLED")
            scores["pooled"]["need"] = need
        else:
            scores["decision"] = None
    else:
        scores["decision"] = None  # dev: metrics only

    out_dir = out_dir or (C.RESULTS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.json").write_text(
        json.dumps(scores, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    (out_dir / "report.md").write_text(render_report(scores))
    return scores


def render_report(s: dict) -> str:
    A, B, st = s["criterion_A"], s["criterion_B"], s["stability"]
    lines = [
        f"# Hook test report — {s['run_id']} ({s['split']})",
        "",
        f"- Videos: {s['n_videos']} ({s['n_invalid_videos']} invalid)",
        f"- Criterion A: {A.get('correct')}/{A.get('n')} correct "
        f"(need {math.ceil(A.get('n', 0) * C.GATE_PASS / C.GATE_N)}), "
        f"p={A.get('p_binom', float('nan')):.4f}, Wilson {A.get('wilson_ci')}, "
        f"AUC={A.get('auc'):.3f} (perm p={A.get('auc_perm_p'):.4f}, CI {A.get('auc_boot_ci')}), "
        f"Spearman vs hold_resid={A.get('spearman_hold_resid'):.3f}, "
        f"hard-group acc={A.get('hard_group_accuracy')}",
        f"- Criterion B: T_obs={B.get('T_obs')} (need >= {C.T_OBS_MIN}), "
        f"null1 p={B.get('null1_uniform_p')}, null2 p={B.get('null2_permutation_p')}, "
        f"pass={B.get('pass')}; err={B.get('mean_err')}, hit@1s={B.get('hit_1s')}, "
        f"flat excluded={B.get('n_flat')}",
        f"- Constants: {B.get('constant_baselines')}",
        f"- Metadata baselines: {s['metadata_baselines']} (confounded={s['confounded']})",
        f"- Stability: mean pairwise Spearman={st.get('mean_pairwise_spearman')}, "
        f"group flips={st.get('group_flip_share')}, dropoff std={st.get('mean_dropoff_std_s')}",
        f"- Usage: {s['usage']}",
        f"- Length estimate: {s['video_length_s_estimate']}",
        f"- Schema: {s.get('schema_version')}; v2 diagnostics: {s.get('v2_diagnostics')}",
        "",
        f"## Decision: {s.get('decision')}",
        "",
        "## Caveats",
        CAVEATS,
        "",
    ]
    return "\n".join(lines)


def append_dev_log(scores: dict) -> None:
    A, B, st = scores["criterion_A"], scores["criterion_B"], scores["stability"]
    rec = {
        "run_id": scores["run_id"],
        "schema_version": scores.get("schema_version"),
        "version": (scores["usage"]["prompt_versions"] or [None])[0],
        "model": (scores["usage"]["models"] or [None])[0],
        "auc": A.get("auc"), "acc": A.get("accuracy"), "correct": A.get("correct"),
        "T_obs": B.get("T_obs"), "mean_rank_score": B.get("T_obs"),
        "n_valid": A.get("n"), "tokens": scores["usage"]["total_tokens"],
        "fps": None, "media_resolution": None,
        "v2_diagnostics": scores.get("v2_diagnostics"),
    }
    C.DEV_LOG.parent.mkdir(parents=True, exist_ok=True)
    with C.DEV_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", dest="run_id", required=True)
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--export", type=Path, default=None)
    p.add_argument("--pool-with", dest="pool_id", default=None)
    p.add_argument("--no-dev-log", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    raw_dir = args.raw_dir or (C.RAW_DIR / args.run_id)
    pool_raw = load_raw(C.RAW_DIR / args.pool_id) if args.pool_id else None
    scores = score_run(raw_dir, args.run_id, export_dir=args.export,
                       pool_raw=pool_raw, out_dir=args.out_dir)
    if scores["split"] == "dev" and not args.no_dev_log:
        append_dev_log(scores)
    print(json.dumps({k: scores[k] for k in
                      ["split", "n_videos", "n_invalid_videos", "decision"]},
                     indent=2))
    print("criterion_A:", {k: v for k, v in scores["criterion_A"].items()
                          if not isinstance(v, list)})
    print("criterion_B:", {k: v for k, v in scores["criterion_B"].items()
                          if not isinstance(v, (list, dict))})
    print("report:", args.out_dir or C.RESULTS_DIR / args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
