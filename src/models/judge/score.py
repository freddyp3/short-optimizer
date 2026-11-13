"""Score judging runs against the real audience (plan/03c sections 20.4-20.5).

    ./venv/bin/python -m src.models.judge.score TAG [TAG ...] [--full]

Hold measures are scored after residualising predictions on log(duration) and era, like the real targets, because
raw curves mostly reflect video length. Baselines are computed on exactly the videos each run covers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.data import load_export
from src.evals.hook_test.score import criterion_b, ground_truth_for, residualize
from src.evals.hook_test.stats import spearman
from src.models.judge.library import JUDGE_DIR, SEED, TIERS, Library, build
from src.models.judge.run import RAW

HOLDS = ("early_hold", "mid_hold", "late_hold")
N_BOOT, N_PERM = 2000, 2000
SCORES = JUDGE_DIR / "scores"


def load_run(tag: str, raw: Path = RAW) -> pd.DataFrame:
    rows = []
    for p in sorted((raw / tag).glob("*.json")):
        r = json.loads(p.read_text())
        row = {"video_id": r["video_id"], "status": r["status"], "cost_usd": r["cost_usd"], "neighbours": r["neighbours"]}
        if r["status"] == "ok":
            res = r["result"]
            row |= {f"p{k[3:]}": v for k, v in res["retention"].items()}
            row |= {"drop1": res["drop_moments"][0]["t_s"], "drop2": res["drop_moments"][1]["t_s"],
                    "rewatch": res["rewatch_level"], "like": res["like_rate_tier"]}
        rows.append(row)
    return pd.DataFrame(rows)


def holds_from_points(P: np.ndarray) -> np.ndarray:
    """(n, 3): early = at 25 %, mid = 75 % / 25 %, late = 95 % / 75 %. Columns of P: 10, 25, 50, 75, 95."""
    return np.column_stack([P[:, 1], P[:, 3] / P[:, 1], P[:, 4] / P[:, 3]])


def residual_holds(P: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    H = holds_from_points(P)
    return np.column_stack([residualize(pd.Series(H[:, k]), rows["duration_seconds"].reset_index(drop=True),
                                        rows["published_at"].reset_index(drop=True)).to_numpy() for k in range(3)])


def hold_scores(R: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rank correlations for early, mid, late, then their mean."""
    s = np.array([spearman(R[:, k], Y[:, k]) for k in range(3)])
    return np.append(s, s.mean())


def baselines(lib: Library, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Point predictions (n, 5) of the cheap alternatives, for the videos at positions ``idx``."""
    nb = lambda arm: np.vstack([lib.points[[j for j, _ in lib.neighbours(i, arm)]].mean(axis=0) for i in idx])
    return {"length only": np.vstack([lib.anchor(i) for i in idx]), "per-second model (hint)": lib.hint_points[idx],
            "neighbour mean, arm E": nb("E"), "neighbour mean, arm S": nb("S")}


def score_run(tag: str, lib: Library, export, full: bool = False) -> dict:
    run = load_run(tag)
    ok = run[run["status"] == "ok"].reset_index(drop=True)
    idx = np.array([lib.index_of(v) for v in ok["video_id"]])
    rows = lib.df.iloc[idx]
    Y = rows[list(HOLDS)].to_numpy(dtype=float)
    P = ok[["p10", "p25", "p50", "p75", "p95"]].to_numpy(dtype=float)
    R = residual_holds(P, rows)
    out = {"tag": tag, "n": len(run), "n_ok": len(ok), "cost_usd": float(run["cost_usd"].sum()),
           "holds": dict(zip(HOLDS + ("mean",), hold_scores(R, Y).round(4).tolist()))}
    order = {t: k for k, t in enumerate(TIERS)}
    out["rewatch"] = spearman(ok["rewatch"].map(order), rows["replay_level"].to_numpy(dtype=float))
    out["like"] = spearman(ok["like"].map(order), rows["like_rate"].to_numpy(dtype=float))
    out["raw_point_mae"] = float(np.abs(P - lib.points[idx]).mean())
    gts = ground_truth_for(rows, export.retention)
    b = criterion_b(rows.assign(dropoff_pred_s=ok["drop1"].to_numpy()), gts, n_perms=N_PERM)
    out["drop"] = {k: b[k] for k in ("T_obs", "null1_uniform_p", "null2_permutation_p", "n_scored")}
    base = {name: residual_holds(B, rows) for name, B in baselines(lib, idx).items()}
    rf = base["per-second model (hint)"].copy()
    rf[:, 2] = lib.hint_late_pct[idx]
    base["models (per-second + forest for late)"] = rf
    out["baselines"] = {name: dict(zip(HOLDS + ("mean",), hold_scores(B, Y).round(4).tolist())) for name, B in base.items()}
    out["echo"] = {"late: corr(Gemini, forest hint)": spearman(R[:, 2], lib.hint_late_pct[idx]),
                   "early: corr(Gemini, per-second hint)": spearman(R[:, 0], base["per-second model (hint)"][:, 0])}
    if full:
        rng = np.random.default_rng(SEED)
        n = len(ok)
        boots = [rng.integers(0, n, n) for _ in range(N_BOOT)]
        g = np.array([hold_scores(R[b], Y[b]) for b in boots])
        out["ci95"] = {k: np.nanpercentile(g[:, j], [2.5, 97.5]).round(3).tolist() for j, k in enumerate(HOLDS + ("mean",))}
        null = np.array([hold_scores(R[rng.permutation(n)], Y) for _ in range(N_PERM)])
        obs = hold_scores(R, Y)
        out["perm_p"] = {k: float((1 + (null[:, j] >= obs[j]).sum()) / (1 + N_PERM)) for j, k in enumerate(HOLDS + ("mean",))}
        out["p_better_than"] = {name: float(np.mean(g[:, 3] > np.array([hold_scores(B[b], Y[b])[3] for b in boots])))
                                for name, B in base.items()}
        for item, col in (("rewatch", "replay_level"), ("like", "like_rate")):
            x, y = ok[item].map(order).to_numpy(dtype=float), rows[col].to_numpy(dtype=float)
            nullx = [spearman(x[rng.permutation(n)], y) for _ in range(N_PERM)]
            out[f"{item}_perm_p"] = float((1 + np.sum(np.array(nullx) >= out[item])) / (1 + N_PERM))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("tags", nargs="+")
    p.add_argument("--full", action="store_true", help="bootstrap intervals, permutation p, P(better than baseline)")
    p.add_argument("--hazard-extra", action="store_true")
    args = p.parse_args(argv)
    lib = build(hazard_extra=args.hazard_extra)
    export = load_export(Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"]))
    SCORES.mkdir(parents=True, exist_ok=True)
    print(f"{'run':46}{'ok':>7}{'early':>7}{'mid':>7}{'late':>7}{'MEAN':>7}{'drop':>6}{'rewatch':>8}{'like':>6}{'cost':>7}")
    res = []
    for tag in args.tags:
        s = score_run(tag, lib, export, args.full)
        res.append(s)
        (SCORES / f"{tag}.json").write_text(json.dumps(s, indent=1))
        h = s["holds"]
        print(f"{tag:46}{s['n_ok']:>4}/{s['n']:<3}{h['early_hold']:>6.2f}{h['mid_hold']:>7.2f}{h['late_hold']:>7.2f}"
              f"{h['mean']:>7.2f}{s['drop']['T_obs']:>6.2f}{s['rewatch']:>8.2f}{s['like']:>6.2f}{s['cost_usd']:>7.2f}")
    print("\nbaselines on the videos of the last run:")
    for name, h in res[-1]["baselines"].items():
        print(f"  {name:44}{'':7}{h['early_hold']:>6.2f}{h['mid_hold']:>7.2f}{h['late_hold']:>7.2f}{h['mean']:>7.2f}")
    if args.full:
        for s in res:
            print(f"\n{s['tag']}\n  95% intervals: {s['ci95']}\n  permutation p: {s['perm_p']}"
                  f"\n  drop moments: {s['drop']}\n  rewatch p {s['rewatch_perm_p']:.3f}, like p {s['like_perm_p']:.3f}"
                  f"\n  P(Gemini mean hold > baseline): {s['p_better_than']}\n  echo: {s['echo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
