"""Pick the strong/weak Shorts for the hook test and split them into DEV / GATE / HELDOUT (plan section 2).

Pipeline: per-video features -> eligible pool (funnel) -> hold_resid label -> strong/weak
candidates (top/bottom quartile) -> cell-matched strong/weak pairs -> splits -> balance check
(re-draw with seed+1 on failure). Deterministic, no network. Writes selection.json + selection.csv.

Deviation from the plan: no relativeRetentionPerformance sanity filter. On this channel the pool
median RRP is 0.65 and RRP correlates -0.67 with duration, so the filter removed half the weak
candidates and reintroduced the length confound. mean_rrp is kept as a secondary label.

    ./venv/bin/python -m src.evals.hook_test.select [--export DIR] [--available video_map.csv]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.categorize import categorize_video, hold_resid
from src.evals.hook_test.data import ExportTables, load_export
from src.evals.hook_test.stats import mann_whitney_u

log = logging.getLogger(__name__)

RECORD_COLS = [
    "video_id", "split", "label", "blind_id", "hold_score", "hold_resid", "duration_seconds",
    "published_at", "mean_rrp", "hook3s", "engagedViews", "dur_tertile", "date_tertile", "pair_id", "pair_level",
]


class SelectionError(RuntimeError):
    """Raised when the pool is too small or no seed yields a balanced draw."""


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def build_frame(t: ExportTables) -> pd.DataFrame:
    """One row per Short with every column the funnel and label need."""
    v = t.videos[t.videos["content_type"] == "shorts"][
        ["video_id", "title", "published_at", "duration_seconds"]
    ].copy()
    v["category"] = [categorize_video(i, t) for i, t in zip(v["video_id"], v["title"])]
    m = t.metrics[t.metrics["creatorContentType"] == "shorts"][
        ["video_id", "engagedViews", "averageViewPercentage"]
    ]
    v = v.merge(m, on="video_id", how="left")

    total = t.daily.groupby("video_id")["engagedViews"].sum()
    post = t.daily[t.daily["day"] >= C.VIEW_CHANGE_DATE].groupby("video_id")["engagedViews"].sum()
    share = post.reindex(total.index, fill_value=0) / total.where(total > 0)
    v["post_change_share"] = v["video_id"].map(share)

    r = t.retention.sort_values(["video_id", "elapsedVideoTimeRatio"])
    curves = r.groupby("video_id").agg(
        curve_points=("audienceWatchRatio", "count"),
        mean_rrp=("relativeRetentionPerformance", "mean"),
    )
    v = v.merge(curves, left_on="video_id", right_index=True, how="left")
    v["curve_points"] = v["curve_points"].fillna(0).astype(int)

    xy = {vid: (g["elapsedVideoTimeRatio"].to_numpy(), g["audienceWatchRatio"].to_numpy())
          for vid, g in r.groupby("video_id")}

    def hook3s(row) -> float:
        if row.video_id not in xy or not row.duration_seconds:
            return np.nan
        x, y = xy[row.video_id]
        return float(np.interp(min(1.0, C.HOOK_SECONDS / row.duration_seconds), x, y))

    v["hook3s"] = v.apply(hook3s, axis=1)
    return v.reset_index(drop=True)


def eligible_pool(
    frame: pd.DataFrame, available: set[str] | None, max_post_share: float
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply rules 2.1 in order; return the pool and per-rule exclusion counts."""
    rules = [
        ("category_" + C.REFERENCE_CATEGORY, frame["category"] == C.REFERENCE_CATEGORY),
        ("duration_8_60s", frame["duration_seconds"].between(C.MIN_DURATION_S, C.MAX_DURATION_S)),
        ("published_before_" + C.PUBLISHED_BEFORE, _ts(frame["published_at"]) < pd.Timestamp(C.PUBLISHED_BEFORE, tz="UTC")),
        (f"post_change_share_le_{max_post_share:.2f}", frame["post_change_share"] <= max_post_share),
        (f"engaged_views_ge_{C.MIN_ENGAGED_VIEWS}", frame["engagedViews"] >= C.MIN_ENGAGED_VIEWS),
        ("complete_retention_curve", frame["curve_points"] == C.CURVE_POINTS),
    ]
    if available is not None:
        rules.append(("video_file_available", frame["video_id"].isin(available)))
    keep = pd.Series(True, index=frame.index)
    funnel = [{"rule": "all_shorts", "excluded": 0, "remaining": len(frame)}]
    for name, mask in rules:
        new = keep & mask.fillna(False).astype(bool)
        funnel.append({"rule": name, "excluded": int((keep & ~new).sum()), "remaining": int(new.sum())})
        keep = new
    if available is None:
        funnel.append({"rule": "video_file_available", "excluded": 0, "remaining": int(keep.sum()), "note": "not applied (no --available)"})
    return frame[keep].copy(), funnel


def label_candidates(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """hold_resid, top/bottom quartile labels, and the 3x3 duration x date grid."""
    pool = pool.copy()
    pool["hold_score"] = pool["averageViewPercentage"]
    pool["hold_resid"] = hold_resid(pool, pd.Series(True, index=pool.index))
    lo, hi = pool["hold_resid"].quantile([C.CANDIDATE_QUANTILE, 1 - C.CANDIDATE_QUANTILE])
    pool["label"] = None
    pool.loc[pool["hold_resid"] >= hi, "label"] = "strong"
    pool.loc[pool["hold_resid"] <= lo, "label"] = "weak"
    pool["dur_tertile"] = pd.qcut(pool["duration_seconds"].rank(method="first"), 3, labels=False)
    pool["date_tertile"] = pd.qcut(_ts(pool["published_at"]).rank(method="first"), 3, labels=False)
    info = {
        "hold_resid_cutoffs": {"weak_le": float(lo), "strong_ge": float(hi)},
        "candidates": {"strong": int((pool["label"] == "strong").sum()), "weak": int((pool["label"] == "weak").sum())},
    }
    return pool, info


def make_pairs(cands: pd.DataFrame, rng: np.random.Generator) -> list[tuple[str, str, str]]:
    """Pair each strong with a weak from the same grid cell; leftovers pair within the duration row."""
    pairs: list[tuple[str, str, str]] = []
    leftover: dict[int, dict[str, list[str]]] = {}
    for (dur_t, _), cell in cands.groupby(["dur_tertile", "date_tertile"]):
        s = list(rng.permutation(sorted(cell.loc[cell["label"] == "strong", "video_id"])))
        w = list(rng.permutation(sorted(cell.loc[cell["label"] == "weak", "video_id"])))
        n = min(len(s), len(w))
        pairs += [(s[i], w[i], "cell") for i in range(n)]
        row = leftover.setdefault(int(dur_t), {"strong": [], "weak": []})
        row["strong"] += s[n:]
        row["weak"] += w[n:]
    for dur_t in sorted(leftover):
        s = list(rng.permutation(sorted(leftover[dur_t]["strong"])))
        w = list(rng.permutation(sorted(leftover[dur_t]["weak"])))
        pairs += [(s[i], w[i], "row") for i in range(min(len(s), len(w)))]
    return pairs


def split_sizes(n_pairs: int) -> dict[str, int]:
    """10/20/rest pairs; shrink DEV to 6 first, then accept HELDOUT down to 10; GATE never shrinks."""
    dev = C.DEV_PAIRS if n_pairs - C.DEV_PAIRS - C.GATE_PAIRS >= C.HELDOUT_MIN_PAIRS else C.DEV_PAIRS_SHRUNK
    heldout = n_pairs - dev - C.GATE_PAIRS
    if heldout < C.HELDOUT_MIN_PAIRS_SHRUNK:
        raise SelectionError(
            f"only {n_pairs} strong/weak pairs; need >= {C.DEV_PAIRS_SHRUNK + C.GATE_PAIRS + C.HELDOUT_MIN_PAIRS_SHRUNK}"
        )
    return {"gate": C.GATE_PAIRS, "dev": dev, "heldout": heldout}


def balance(rows: pd.DataFrame) -> dict:
    """Strong vs weak on duration and publish date (Mann-Whitney) plus median duration gap."""
    s, w = rows[rows["label"] == "strong"], rows[rows["label"] == "weak"]
    as_num = lambda df: _ts(df["published_at"]).astype("int64") / 1e9
    _, p_dur = mann_whitney_u(s["duration_seconds"], w["duration_seconds"])
    _, p_date = mann_whitney_u(as_num(s), as_num(w))
    diff = abs(float(s["duration_seconds"].median() - w["duration_seconds"].median()))
    return {
        "n_strong": len(s), "n_weak": len(w),
        "duration_p": p_dur, "date_p": p_date, "median_duration_diff_s": diff,
        "passed": p_dur > C.BALANCE_MIN_P and p_date > C.BALANCE_MIN_P
        and diff <= C.BALANCE_MAX_MEDIAN_DURATION_DIFF_S,
    }


def blind_ids(video_ids: list[str], seed: int) -> dict[str, str]:
    """'v' + 8 hex chars, seeded so reruns reproduce them; unrelated to label or order."""
    rnd, out, used = random.Random(seed), {}, set()
    for vid in sorted(video_ids):
        while (b := f"v{rnd.getrandbits(32):08x}") in used:
            pass
        used.add(b)
        out[vid] = b
    return out


def draw(cands: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One seeded draw: pairs -> shuffled -> GATE, DEV, HELDOUT."""
    rng = np.random.default_rng(seed)
    pairs = make_pairs(cands[cands["label"].notna()], rng)
    sizes = split_sizes(len(pairs))
    order = rng.permutation(len(pairs))
    bounds = {"gate": (0, sizes["gate"]), "dev": (sizes["gate"], sizes["gate"] + sizes["dev"]),
              "heldout": (sizes["gate"] + sizes["dev"], len(pairs))}
    rows = []
    for split, (a, b) in bounds.items():
        for k in order[a:b]:
            strong, weak, level = pairs[k]
            rows += [{"video_id": strong, "split": split, "pair_id": int(k), "pair_level": level},
                     {"video_id": weak, "split": split, "pair_id": int(k), "pair_level": level}]
    sel = pd.DataFrame(rows).merge(cands, on="video_id")
    sel["blind_id"] = sel["video_id"].map(blind_ids(sel["video_id"].tolist(), seed))
    return sel


def select(frame: pd.DataFrame, seed: int = C.SEED, available: set[str] | None = None) -> dict:
    """Full selection; retries seed+1.. until GATE and HELDOUT are balanced."""
    pool, funnel = eligible_pool(frame, available, C.MAX_POST_CHANGE_SHARE)
    relaxed = len(pool) < C.MIN_POOL_BEFORE_RELAX
    if relaxed:
        pool, funnel = eligible_pool(frame, available, C.RELAXED_POST_CHANGE_SHARE)
    if len(pool) < 40:
        raise SelectionError(f"eligible pool has only {len(pool)} videos")
    cands, label_info = label_candidates(pool)

    tries = []
    for attempt in range(C.MAX_SEED_TRIES):
        sel = draw(cands, seed + attempt)
        checks = {split: balance(sel[sel["split"] == split]) for split in ("dev", "gate", "heldout")}
        checks["all"] = balance(sel)
        tries.append({"seed": seed + attempt, "gate_passed": checks["gate"]["passed"], "heldout_passed": checks["heldout"]["passed"]})
        if checks["gate"]["passed"] and checks["heldout"]["passed"]:
            break
    else:
        raise SelectionError(f"no balanced draw in {C.MAX_SEED_TRIES} seeds: {tries}")

    sel = sel.sort_values(["split", "label", "video_id"]).reset_index(drop=True)
    return {
        "seed": seed,
        "seed_used": seed + attempt,
        "attempt": attempt + 1,
        "attempts": tries,
        "pool_size": len(pool),
        "post_change_share_relaxed": relaxed,
        "pool_funnel": funnel,
        "era_cutoff": C.ERA_CUTOFF,
        "pool_median_published_at": str(_ts(pool["published_at"]).median()),
        "labels": label_info,
        "pairs": {"cell": int((sel["pair_level"] == "cell").sum() // 2), "row": int((sel["pair_level"] == "row").sum() // 2)},
        "splits": {s: {"strong": int(((sel["split"] == s) & (sel["label"] == "strong")).sum()),
                       "weak": int(((sel["split"] == s) & (sel["label"] == "weak")).sum())}
                   for s in ("dev", "gate", "heldout")},
        "balance": checks,
        "prompt_sha256": None,
        "videos": sel[RECORD_COLS],
    }


def write_outputs(result: dict, out_dir: Path, export_dir: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    videos: pd.DataFrame = result["videos"]
    videos.to_csv(out_dir / "selection.csv", index=False)
    payload = {**result, "export_dir": str(export_dir) if export_dir else None,
               "videos": json.loads(videos.to_json(orient="records"))}
    (out_dir / "selection.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, help="export dir (default: newest under data/channel)")
    parser.add_argument("--available", type=Path, help="CSV with a video_id column of videos whose file was obtained (rule 5)")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    tables = load_export(args.export)
    available = set(pd.read_csv(args.available)["video_id"]) if args.available else None
    result = select(build_frame(tables), args.seed, available)
    write_outputs(result, args.out_dir, tables.export_dir)

    log.info("export: %s", tables.export_dir)
    for step in result["pool_funnel"]:
        log.info("  %-34s -%-4d -> %d %s", step["rule"], step["excluded"], step["remaining"], step.get("note", ""))
    li = result["labels"]
    log.info("quartiles: strong if hold_resid>=%.2f, weak if <=%.2f; candidates %s",
             li["hold_resid_cutoffs"]["strong_ge"], li["hold_resid_cutoffs"]["weak_le"], li["candidates"])
    log.info("pairs: %s; seed used %d (attempt %d)", result["pairs"], result["seed_used"], result["attempt"])
    for split, b in result["balance"].items():
        log.info("  %-8s %2d/%-2d  duration p=%.3f  date p=%.3f  |median dur diff|=%.1fs  %s", split, b["n_strong"],
                 b["n_weak"], b["duration_p"], b["date_p"], b["median_duration_diff_s"], "ok" if b["passed"] else "FAIL")
    log.info("wrote %s/selection.json and selection.csv", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
