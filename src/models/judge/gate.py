"""GATE: every source on every item, one specialist per item (plan/03c section 21). Needs src/models/FROZEN.

    ./venv/bin/python -m src.models.judge.gate extract [--reverse]   # one-second timelines for the 40 GATE videos
    ./venv/bin/python -m src.models.judge.gate check                 # exit checks before any judging call
    ./venv/bin/python -m src.models.judge.gate score                 # once

GATE videos are scored against a library of the 145 train + DEV videos. In the library used for prompts, every
GATE outcome is blanked (NaN), so it cannot reach a prompt, an example list or a model fit.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.categorize import ERA_CUTOFF
from src.evals.hook_test.score import criterion_b, ground_truth_for
from src.evals.hook_test.stats import spearman
from src.models import hazard as H
from src.models import targets as T
from src.models.dataset import load_split
from src.models.features import M1_COLS, design, prepare
from src.models.judge import score as S
from src.models.judge.library import (JUDGE_DIR, K, RAW_DIR, SEED, TIERS, V3_RUNS, Library, curve_points,
                                      embed_timelines, implied_points, load_timelines, tiers)
from src.models.judge.prompt import build_prompt

log = logging.getLogger("gate")
GATE_RUN = "20261113T000000Z_gate_v3"
ITEMS = ("early_hold", "mid_hold", "late_hold", "replay_level")
GEMINI = {"Gemini, no examples": "j1_L0_E_3.8-flash_gate", "Gemini, examples >= 0.75": "j1_L2_T75_3.8-flash_gate",
          "Gemini, 5 similar": "j1_L2_E_3.8-flash_gate"}
SPECIALIST = {"drop moments": "Gemini, no examples", "mid_hold": "Gemini, no examples", "late_hold": "Random forest",
              "rewatch": "Gemini, examples >= 0.75", "early_hold": "Structure neighbours"}
N_PERM = 10_000
BLANKED = ["early_hold", "mid_hold", "late_hold", "replay_level", "like_rate", "share_rate", "comment_rate",
           "hold_resid", "hold_resid_views", "hold_score", "hook3s", "engagedViews"]


def trend(rows: pd.DataFrame) -> np.ndarray:
    late = (pd.to_datetime(pd.Series(rows["published_at"].to_numpy())) >= ERA_CUTOFF).astype(float).to_numpy()
    return np.column_stack([np.ones(len(rows)), np.log(rows["duration_seconds"].to_numpy(dtype=float)), late])


def residual_from_fit(y_fit, rows_fit: pd.DataFrame, y_new, rows_new: pd.DataFrame) -> np.ndarray:
    """Residual of ``y_new`` from the length-and-era trend fitted on other rows (nothing about the new rows is fitted)."""
    coef, *_ = np.linalg.lstsq(trend(rows_fit), np.asarray(y_fit, dtype=float), rcond=None)
    return np.asarray(y_new, dtype=float) - trend(rows_new) @ coef


@dataclass
class Gate:
    lib: Library                 # 145 library rows first, then the GATE rows with blanked outcomes
    gate_idx: np.ndarray
    truth: pd.DataFrame          # residualised early / mid / late / replay for the GATE rows (scoring only)
    rows: pd.DataFrame           # GATE rows with real metadata (scoring only)
    forest: np.ndarray           # (n_gate, 4) random-forest predictions for ITEMS
    per_second: np.ndarray       # (n_gate, 5) checkpoint predictions


def build_gate() -> Gate:
    from sklearn.ensemble import RandomForestRegressor

    df, _, recipes, export = T.load()
    n = len(df)
    g = load_split("gate").drop(columns=["mean_rrp"], errors="ignore")
    raw = T.curve_targets(export.retention[export.retention["video_id"].isin(set(g["video_id"]))])
    g = g.merge(raw, on="video_id", how="left", validate="one_to_one")
    truth = pd.DataFrame({t: residual_from_fit(df[f"{t}_raw"], df, g[t], g) for t in ITEMS})
    for t in ITEMS:   # the same function must reproduce the library's own residuals
        assert np.allclose(residual_from_fit(df[f"{t}_raw"], df, df[f"{t}_raw"], df), df[t].to_numpy(dtype=float), atol=1e-8)
    blank = g.copy()
    for c in BLANKED:
        if c in blank:
            blank[c] = np.nan
    all_df = pd.concat([df, blank], ignore_index=True)
    gate_idx = np.arange(n, n + len(g))
    timelines = load_timelines(all_df, runs=V3_RUNS + (GATE_RUN,))
    missing = sorted(set(all_df["video_id"]) - set(timelines))
    if missing:
        raise RuntimeError(f"{len(missing)} videos have no valid one-second timeline")
    emb = embed_timelines(all_df, timelines, JUDGE_DIR / "timeline_embeddings_gate.npz")
    data = prepare(all_df)
    X = data.matrix(M1_COLS)
    Z = (X - X[:n].mean(axis=0)) / X[:n].std(axis=0)
    payoffs = H.load_payoffs(all_df["video_id"])
    seg, segg = H.build(df, export.retention, payoffs), H.build(g, export.retention, payoffs)
    per_second = implied_points(segg, H.fit_predict("ridge", np.column_stack([seg.base, seg.cov]), seg.y,
                                                     np.column_stack([segg.base, segg.cov])))
    tr = np.arange(n)
    forest = []
    for t in ITEMS:
        Xtr, Xte, ytr, _ = design(recipes["M1_core_mechanics"], data, tr, df[t].to_numpy(dtype=float), te=gate_idx)
        forest.append(RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33, random_state=SEED,
                                            n_jobs=-1).fit(Xtr, ytr).predict(Xte))
    nan_rows = np.full((len(g), 5), np.nan)
    lib = Library(
        df=all_df, timelines=timelines, payoffs=payoffs,
        points=np.vstack([curve_points(export.retention, df["video_id"]).to_numpy(), nan_rows]),
        rewatch=np.concatenate([tiers(df["replay_level"]).to_numpy(), np.full(len(g), "nan")]),
        likes=np.concatenate([tiers(df["like_rate"]).to_numpy(), np.full(len(g), "nan")]),
        late=np.concatenate([df["late_hold"].to_numpy(dtype=float), np.full(len(g), np.nan)]),
        sim_e=emb @ emb.T, dist_s=np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)),
        hint_points=np.vstack([nan_rows[:0].reshape(0, 5), np.full((n, 5), np.nan), per_second]),
        hint_late_pct=np.full(n + len(g), np.nan), split={"gate": g["video_id"].tolist()}, pool=tr)
    return Gate(lib=lib, gate_idx=gate_idx, truth=truth, rows=g.reset_index(drop=True),
                forest=np.column_stack(forest), per_second=per_second)


def extract(reverse: bool) -> int:
    from dotenv import load_dotenv

    from src.evals.hook_test.gemini_client import sha256_file
    from src.tools.analyze_video import extract as E
    from src.tools.analyze_video.schema import render_prompt

    load_dotenv(".env")
    if not (Path("src/models") / "FROZEN").exists():
        raise SystemExit("src/models/FROZEN missing: GATE stays locked")
    vids = E.pick_eval_videos()
    vids = vids[vids["eval_split"] == "gate"].sort_values("blind_id")
    vids = vids.iloc[::-1] if reverse else vids
    run_dir = RAW_DIR / GATE_RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    pf = E.PROMPTS_DIR / "v3.md"
    template, sha = pf.read_text(), sha256_file(pf)
    ext = E.GeminiFeatureExtractor(model=C.DEFAULT_MODEL, version="v3")
    spent = 0.0
    for _, v in vids.iterrows():
        out = run_dir / f"{v['blind_id']}__r1.json"
        if out.exists():
            continue
        if spent > 3.0:
            raise SystemExit("extraction cap reached")
        bp = Path(v["blind_path"])
        dur = E.blind_duration(bp)
        rec = ext.analyze(v["blind_id"], bp, render_prompt(template, dur), dur, fps=C.FPS,
                          media_resolution=C.MEDIA_RESOLUTION, prompt_version="v3", prompt_sha256=sha)
        rec.update({"run_id": GATE_RUN, "run_k": 1, "split": "gate", "video_id": v["video_id"], "band": None,
                    "duration_s_file": dur, "duration_s_export": float(v["duration_s_export"])})
        if not out.exists():
            out.write_text(json.dumps(rec, indent=2))
        spent += rec.get("cost_usd_paid_equiv", 0)
        log.info("%s -> %s ($%.4f)", v["blind_id"], rec["status"], rec.get("cost_usd_paid_equiv", 0))
    print(f"EXTRACT DONE ${spent:.2f}")
    return 0


def check() -> int:
    """Exit checks that read no GATE outcome: timelines valid, prompts free of blanked values and GATE examples."""
    from src.tools.analyze_video.schema import windows

    recs = [json.loads(p.read_text()) for p in sorted((RAW_DIR / GATE_RUN).glob("*__r1.json"))]
    ok = [r for r in recs if r["status"] == "ok"]
    bad_w = sum(len(r["result"]["timeline"]) != len(windows(r["duration_s_file"])) for r in ok)
    print(f"GATE timelines: {len(recs)} records, {len(ok)} valid, window count wrong: {bad_w}, "
          f"cost ${sum(r.get('cost_usd_paid_equiv', 0) for r in recs):.2f}")
    gate = build_gate()
    lib, gset = gate.lib, set(gate.gate_idx.tolist())
    problems = 0
    shown_counts = {}
    for variant, arm in (("L0", "E"), ("L2", "T75"), ("L2", "E")):
        counts = []
        for i in gate.gate_idx:
            text, shown = build_prompt(lib, int(i), variant, arm)
            problems += bool(re.search(r"(?<![A-Za-z])nan(?![A-Za-z])", text)) + bool(gset & set(shown))
            row = lib.df.iloc[int(i)]
            problems += any(s in text for s in (row.video_id, row.blind_id))
            counts.append(len(shown))
        shown_counts[f"{variant}/{arm}"] = (float(np.mean(counts)), int(np.sum(np.array(counts) == 0)))
    print(f"prompts checked: {3 * len(gate.gate_idx)}; problems: {problems}; examples shown (mean, drafts with none): {shown_counts}")
    fails = len(ok) != 40 or bad_w or problems
    print("EXIT CHECK:", "FAIL" if fails else "PASS")
    return int(bool(fails))


def perm_p(x: np.ndarray, y: np.ndarray, rng) -> float:
    obs = spearman(x, y)
    null = np.array([spearman(x[rng.permutation(len(x))], y) for _ in range(N_PERM)])
    return float((1 + np.sum(null >= obs)) / (1 + N_PERM))


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for k, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - k) * ps[name]))
        out[name] = running
    return out


def score() -> int:
    gate = build_gate()
    lib, idx, rows = gate.lib, gate.gate_idx, gate.rows
    from src.evals.hook_test.data import load_export
    export = load_export(Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"]))
    Y = gate.truth[["early_hold", "mid_hold", "late_hold"]].to_numpy()
    replay = gate.truth["replay_level"].to_numpy()
    ids = rows["video_id"].tolist()
    order = {t: k for k, t in enumerate(TIERS)}
    runs = {name: S.load_run(tag).set_index("video_id").loc[ids] for name, tag in GEMINI.items()}
    n_bad = {name: int((f["status"] != "ok").sum()) for name, f in runs.items()}
    pts = lambda f: f[["p10", "p25", "p50", "p75", "p95"]].to_numpy(dtype=float)
    nb = [[j for j, _ in lib.neighbours(int(i), "S")] for i in idx]
    holds = {name: S.residual_holds(pts(f), rows) for name, f in runs.items()}
    holds["Random forest"] = gate.forest[:, :3]
    holds["Per-second model"] = S.residual_holds(gate.per_second, rows)
    holds["Structure neighbours"] = S.residual_holds(np.vstack([lib.points[j].mean(axis=0) for j in nb]), rows)
    holds["Length only"] = S.residual_holds(np.vstack([lib.anchor(int(i)) for i in idx]), rows)
    rewatch = {name: f["rewatch"].map(order).to_numpy(dtype=float) for name, f in runs.items()}
    rewatch["Random forest"] = gate.forest[:, 3]
    lib_replay = lib.df["replay_level"].to_numpy(dtype=float)
    rewatch["Structure neighbours"] = np.array([lib_replay[j].mean() for j in nb])
    gts = ground_truth_for(rows, export.retention)
    drops = {name: criterion_b(rows.assign(dropoff_pred_s=f["drop1"].to_numpy()), gts, n_perms=N_PERM) for name, f in runs.items()}

    rng = np.random.default_rng(SEED)
    out = {"n": len(ids), "invalid_outputs": n_bad, "holds": {}, "rewatch": {}, "drops": {}, "tests": {}}
    print(f"GATE: {len(ids)} videos; invalid Gemini outputs: {n_bad}")
    print(f"\n{'source':28}{'early':>7}{'mid':>7}{'late':>7}{'mean':>7}{'rewatch':>9}{'drop':>7}")
    for name, R in holds.items():
        s = S.hold_scores(R, Y)
        out["holds"][name] = s.round(4).tolist()
        rw = spearman(rewatch[name], replay) if name in rewatch else float("nan")
        if name in rewatch:
            out["rewatch"][name] = rw
        dr = drops[name]["T_obs"] if name in drops else float("nan")
        if name in drops:
            out["drops"][name] = {k: drops[name][k] for k in ("T_obs", "null1_uniform_p", "null2_permutation_p", "n_scored")}
        fmt = lambda x: f"{x:>7.2f}" if x == x else f"{'':>7}"
        print(f"{name:28}" + "".join(fmt(x) for x in s) + (f"{rw:>9.2f}" if rw == rw else f"{'':>9}") + fmt(dr))

    col = {"early_hold": 0, "mid_hold": 1, "late_hold": 2}
    ps = {}
    for item, src in SPECIALIST.items():
        if item == "drop moments":
            d = drops[src]
            ps[item] = max(d["null1_uniform_p"], d["null2_permutation_p"])
            out["tests"][item] = {"source": src, "score": d["T_obs"], "p": ps[item],
                                  "pass": bool(d["T_obs"] >= C.T_OBS_MIN and ps[item] < 0.05)}
            continue
        x, y = (rewatch[src], replay) if item == "rewatch" else (holds[src][:, col[item]], Y[:, col[item]])
        r = spearman(x, y)
        ps[item] = perm_p(x, y, rng)
        b = [spearman(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
        out["tests"][item] = {"source": src, "score": r, "p": ps[item], "ci95": np.nanpercentile(b, [2.5, 97.5]).round(3).tolist(),
                              "pass": bool(r > 0 and ps[item] <= 0.05)}
    adj = holm(ps)
    print(f"\n{'item':18}{'specialist':28}{'score':>7}{'p':>9}{'Holm p':>9}  result")
    for item, t in out["tests"].items():
        t["holm_p"] = adj[item]
        ci = f"  (95% {t['ci95'][0]:.2f} to {t['ci95'][1]:.2f})" if "ci95" in t else ""
        print(f"{item:18}{t['source']:28}{t['score']:>7.2f}{t['p']:>9.4f}{adj[item]:>9.4f}  {'PASS' if t['pass'] else 'fail'}{ci}")
    S.SCORES.mkdir(parents=True, exist_ok=True)
    (S.SCORES / "gate.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract").add_argument("--reverse", action="store_true")
    sub.add_parser("check")
    sub.add_parser("score")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return extract(a.reverse) if a.cmd == "extract" else check() if a.cmd == "check" else score()


if __name__ == "__main__":
    raise SystemExit(main())
