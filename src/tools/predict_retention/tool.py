"""predict_retention: how the channel's audience is expected to watch a draft Short (plan/03d sections 3.3, 12).

    ./venv/bin/python -m src.tools.predict_retention.tool path/to/draft.mp4 [--out result.json]

Five specialists, each used only for the item it was confirmed on (GATE and HELDOUT):
  drop moments      Gemini reading the draft's one-second timeline
  mid-video hold    mean of the two Gemini calls (draft only; with similar past Shorts)
  end-of-video hold random forest on the structure features
  early hold        random forest on the structure features
  rewatch level     Gemini shown similar past Shorts (similarity >= 0.75) with their real outcomes
The library is every labelled Short of the channel (235). The curve is a drawing of those statements on the
typical curve for the draft's length; it is not a number-for-number forecast.
Costs four Gemini calls per draft (two extraction, two judging) on GEMINI_API_KEY_PAID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.data import load_export
from src.evals.hook_test.gemini_client import new_genai_client, sha256_file
from src.models import cv, hazard as H, targets as T
from src.models.dataset import load_split
from src.models.experiments import all_recipes
from src.models.features import M1_COLS, design, prepare
from src.models.judge import score as S, simulator as M
from src.models.judge.gate import RUNS
from src.models.judge.library import (JUDGE_DIR, SEED, TIERS, V3_RUNS, Library, build_from, embed_timelines,
                                      load_timelines, oof_folds, rf_late_oof, timeline_lines, chunked_embeddings, ENCODER)
from src.models.judge.run import judge_one
from src.tools.analyze_video.ab_score import ARC_ORD
from src.tools.analyze_video.derive import derive_features
from src.tools.analyze_video.measure import measure

SPLITS = ("train", "dev", "gate", "heldout")
SETS = ("pilot", "validation", "gate", "heldout")
SEGMENT_TARGET = {"early": "early_hold", "late": "late_hold"}
EVIDENCE = {"early": "confirmed once (HELDOUT 0.56)", "mid": "confirmed twice (GATE 0.56, HELDOUT 0.63)",
            "late": "confirmed twice (GATE 0.48, HELDOUT 0.60)", "drop_moments": "confirmed twice (GATE 0.65, HELDOUT 0.66; 0.5 = chance)",
            "rewatch_level": "confirmed twice, weakest item (GATE 0.43, HELDOUT 0.27)"}
NO_EXAMPLES_NOTE = ("no past Short was similar enough (0.75) to show as an example; without examples this item was "
                    "not confirmed (0.15-0.30), so treat it as a guess")
CAVEATS = [
    "Statements are relative to this channel's own Shorts of the same length; they rank drafts, they do not forecast exact percentages.",
    "The curve is the typical curve for this length, bent by the statements; the band shows how far real videos fell from it.",
    "Based on 235 past Shorts of one channel. The feature extraction was never checked against hand labels.",
    "Confirmation sets were strong and weak videos; on everyday drafts the predictions separate less sharply.",
]
FEATURE_WORDS = {"t_premise_clear_s": "time until the premise is clear", "payoff_count_all": "number of payoffs",
                 "main_payoff_frac": "how late the main payoff comes", "tail_all": "seconds left after the last payoff",
                 "dead_time_s": "dead time before the last payoff", "arc_ord": "build-up towards the biggest payoff",
                 "first_last_dhash_dist": "difference between first and last frame", "dhash_x_short": "first/last frame difference (short video)",
                 "cuts_per_s": "cuts per second"}


@dataclass
class State:
    lib: Library
    emb: np.ndarray
    export: object
    curves: np.ndarray              # (n, 100) real normalised curves
    ref: np.ndarray                 # (n, 100) leave-one-out reference curves
    signal: dict[str, np.ndarray]   # segment -> library signals made without each video's own outcome
    calib: dict[str, tuple[float, float]]
    band: np.ndarray                # (2, 5) residual percentiles at the checkpoints
    ps_loss: np.ndarray
    seg: object


_STATE: State | None = None


def library_frame() -> tuple[pd.DataFrame, object]:
    df = pd.concat([load_split(s) for s in SPLITS], ignore_index=True)
    export = load_export(Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"]))
    return T.add_targets(df, export), export


def gemini_points(ids: list[str], tag: str) -> np.ndarray:
    g = pd.concat([S.load_run(f"{tag}_{s}") for s in SETS]).set_index("video_id").loc[ids]
    return g[["p10", "p25", "p50", "p75", "p95"]].to_numpy(dtype=float)


def mid_signal(p_a: np.ndarray, p_b: np.ndarray, ref: np.ndarray) -> np.ndarray:
    m = (p_a + p_b) / 2
    return m[:, 3] / m[:, 1] - ref[:, M.I75] / ref[:, M.I25]


def state() -> State:
    """Library, models' out-of-fold signals, calibration and band for the 235 videos. Built once per process."""
    global _STATE
    if _STATE is not None:
        return _STATE
    df, export = library_frame()
    timelines = load_timelines(df, runs=V3_RUNS + tuple(RUNS.values()))
    lib = build_from(df, export, JUDGE_DIR / "timeline_embeddings_all.npz", timelines)
    n, ids = len(lib), lib.df["video_id"].tolist()
    emb = embed_timelines(lib.df, timelines, JUDGE_DIR / "timeline_embeddings_all.npz")
    curves = M.normalise(M.raw_curves(export.retention, ids))
    dur = lib.df["duration_s"].to_numpy(dtype=float)
    ref = np.vstack([M.reference(curves, dur, np.arange(n), dur[i], exclude=[i]) for i in range(n)])
    data, folds = prepare(lib.df), oof_folds(n)
    recipe = all_recipes(data)["M1_core_mechanics"]
    signal = {s: rf_late_oof(data, lib.df[t].to_numpy(dtype=float), recipe, folds) for s, t in SEGMENT_TARGET.items()}
    signal["mid"] = mid_signal(gemini_points(ids, "j1_L0_E_3.8-flash"), gemini_points(ids, "j1_L2_T75_3.8-flash"), ref)
    seg = H.build(lib.df, export.retention, lib.payoffs)
    ps = H.oof(seg, np.column_stack([seg.base, seg.cov]), "ridge", folds).reshape(n, -1)
    pieces = M.Pieces(ids, dur, curves, ref, signal, ps, lib.df)
    pred = M.oof_predict(pieces, tuple(M.SEGMENTS), True)
    _STATE = State(lib, emb, export, curves, ref, signal, M.calibrate(pieces, np.arange(n)), M.band_coverage(pred, pieces)[1], ps, seg)
    return _STATE


def blind_copy(path: Path, out_dir: Path) -> Path:
    """Copy without container metadata, so no title or date travels with the file."""
    out = out_dir / f"draft_{sha256_file(path)[:10]}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-map_metadata", "-1", "-c", "copy", str(out)], check=True)
    return out


def extract(path: Path, blind_id: str, dur: float, version: str) -> dict:
    from src.tools.analyze_video import extract as E
    from src.tools.analyze_video.schema import render_prompt

    pf = E.PROMPTS_DIR / f"{version}.md"
    rec = E.GeminiFeatureExtractor(model=C.DEFAULT_MODEL, version=version).analyze(
        blind_id, path, render_prompt(pf.read_text(), dur), dur, fps=C.FPS, media_resolution=C.MEDIA_RESOLUTION,
        prompt_version=version, prompt_sha256=sha256_file(pf))
    if rec["status"] != "ok":
        raise RuntimeError(f"extraction {version} failed: {rec.get('invalid_reason')}")
    return rec


def draft_row(v2: dict, code: dict, dur: float, template: pd.Series) -> pd.Series:
    """One library-shaped row for the draft: structure features from the v2 extraction and the code measurements."""
    d = derive_features(v2["result"], dur, [])
    row = template.copy()
    row[:] = np.nan
    for k, v in {**d, **code}.items():
        if k in row.index:
            row[k] = v
    row["video_id"], row["blind_id"], row["split"] = "draft", "draft", "draft"
    row["duration_s"], row["duration_seconds"] = dur, float(np.ceil(dur))
    row["published_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row["t_premise_clear_s"] = d["t_premise_clear_s"] if d["t_premise_clear_s"] is not None else dur
    row["arc_ord"] = ARC_ORD[d["arc"]]
    row["is_short"] = float(dur <= 20.0)
    return row


def with_draft(st: State, row: pd.Series, timeline: list[dict], payoffs: list[dict]) -> Library:
    """Library plus the draft as the last row. The draft is outside the pool and has no outcomes."""
    from sentence_transformers import SentenceTransformer

    lib, n = st.lib, len(st.lib)
    df = pd.concat([lib.df, row.to_frame().T], ignore_index=True)
    for c in lib.df.columns:
        if pd.api.types.is_numeric_dtype(lib.df[c]):
            df[c] = pd.to_numeric(df[c])
    model = SentenceTransformer(ENCODER)
    e = chunked_embeddings([timeline_lines(timeline)], lambda p: model.encode(p, normalize_embeddings=True))
    emb = np.vstack([st.emb, e])
    X = prepare(df).matrix(M1_COLS)
    Z = (X - X[:n].mean(axis=0)) / X[:n].std(axis=0)
    pad = lambda a, v: np.concatenate([a, np.full((1,) + a.shape[1:], v)])
    return Library(df=df, timelines={**lib.timelines, "draft": timeline}, payoffs={**lib.payoffs, "draft": payoffs},
                   points=pad(lib.points, np.nan), rewatch=np.append(lib.rewatch, "nan"), likes=np.append(lib.likes, "nan"),
                   late=np.append(lib.late, np.nan), sim_e=emb @ emb.T,
                   dist_s=np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)),
                   hint_points=pad(lib.hint_points, np.nan), hint_late_pct=np.append(lib.hint_late_pct, np.nan),
                   split={"draft": ["draft"]}, pool=np.arange(n))


def label(pct: float) -> str:
    return "below" if pct < 100 / 3 else "above" if pct > 200 / 3 else "typical"


def percentile(x: float, ref: np.ndarray) -> float:
    return float(100 * np.mean(ref < x))


def why_from_ridge(lib236: Library, target: str, n: int, top: int = 2) -> list[str]:
    """Largest ridge contributions (coefficient x standardised feature) for the draft, in words."""
    data = prepare(lib236.df)
    recipe = all_recipes(data)["M1_core_mechanics"]
    y = lib236.df[target].to_numpy(dtype=float)[:n]
    Xtr, Xte, ytr, names = design(recipe, data, np.arange(n), y, te=np.array([n]))
    alpha, _ = cv.choose_alpha(prepare(lib236.df.iloc[:n]), y, recipe)
    coef = cv.ridge_path(Xtr, ytr, Xte, np.array([alpha]))[1][:, 0]
    contrib = coef * Xte[0]
    out = []
    for j in np.argsort(-np.abs(contrib))[:top]:
        side = "higher" if Xte[0, j] > 0 else "lower"
        effect = "helps" if contrib[j] > 0 else "costs viewers"
        out.append(f"{FEATURE_WORDS.get(names[j], names[j])} is {side} than usual for this channel, which {effect} here")
    return out


def predict_retention(video_path: str | Path, client=None) -> dict:
    video_path = Path(video_path)
    st = state()
    n = len(st.lib)
    key = os.environ.get("GEMINI_API_KEY_PAID")
    if client is None and not key:
        raise RuntimeError("GEMINI_API_KEY_PAID not set")
    client = client or new_genai_client(key)
    cost = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        blind = blind_copy(video_path, Path(tmp))
        code = measure(blind)
        dur = float(code["duration_s"])
        v2 = extract(blind, blind.stem + "_v2", dur, "v2")
        v3 = extract(blind, blind.stem + "_v3", dur, "v3")
    cost += v2["cost_usd_paid_equiv"] + v3["cost_usd_paid_equiv"]
    row = draft_row(v2, code, dur, st.lib.df.iloc[0])
    lib = with_draft(st, row, v3["result"]["timeline"], v2["result"]["payoffs"])
    a = judge_one(lib, n, "L0", "E", C.DEFAULT_MODEL, client)
    b = judge_one(lib, n, "L2", "T75", C.DEFAULT_MODEL, client)
    cost += a["cost_usd"] + b["cost_usd"]
    if a["status"] != "ok" or b["status"] != "ok":
        raise RuntimeError(f"judging failed: {a.get('invalid_reason')} / {b.get('invalid_reason')}")
    pts = lambda r: np.array([[r["result"]["retention"][f"at_{k}"] for k in (10, 25, 50, 75, 95)]], dtype=float)

    from sklearn.ensemble import RandomForestRegressor
    data = prepare(lib.df)
    recipe = all_recipes(data)["M1_core_mechanics"]
    tr, te = np.arange(n), np.array([n])
    ref = M.reference(st.curves, st.lib.df["duration_s"].to_numpy(dtype=float), tr, dur)
    signal = {}
    for s, t in SEGMENT_TARGET.items():
        Xtr, Xte, ytr, _ = design(recipe, data, tr, st.lib.df[t].to_numpy(dtype=float), te=te)
        signal[s] = float(RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33, random_state=SEED,
                                                n_jobs=-1).fit(Xtr, ytr).predict(Xte)[0])
    signal["mid"] = float(mid_signal(pts(a), pts(b), ref[None, :])[0])

    r = M.GRID[:-1]
    cov = H.segment_covariates(r * dur, dur / 100, v2["result"]["payoffs"], float(row["t_premise_clear_s"]),
                               float(row["t_loudness_peak_s"]) if row["t_loudness_peak_s"] == row["t_loudness_peak_s"] else float("nan"))
    Xd = np.column_stack([H.baseline_features(r, dur, float(row["is_short"])), cov])
    loss = H.fit_predict("ridge", np.column_stack([st.seg.base, st.seg.cov]), st.seg.y, Xd)
    ref_ratio = M.ratios(ref)
    targets = {s: ref_ratio[k] + st.calib[s][0] + st.calib[s][1] * signal[s] for k, s in enumerate(M.SEGMENTS)}
    curve = M.assemble(ref, targets, loss)
    x = np.array([0] + M.CHECK_IDX + [99])
    lo = np.interp(np.arange(100), x, np.concatenate([[st.band[0, 0]], st.band[0], [st.band[0, -1]]]))
    hi = np.interp(np.arange(100), x, np.concatenate([[st.band[1, 0]], st.band[1], [st.band[1, -1]]]))

    why = {"early": why_from_ridge(lib, "early_hold", n), "late": why_from_ridge(lib, "late_hold", n),
           "mid": [a["result"]["reasoning"]]}
    source = {"early": "random forest on structure features", "mid": "mean of two Gemini readings", "late": "random forest on structure features"}
    segments = {}
    for s, (i0, i1) in M.SEGMENTS.items():
        pct = percentile(signal[s], st.signal[s])
        segments[s] = {"from_pct": int(M.GRID[i0] * 100), "to_pct": int(M.GRID[i1] * 100), "verdict": label(pct),
                       "percentile_among_channel": round(pct), "source": source[s], "why": why[s], "evidence": EVIDENCE[s]}
    return {
        "duration_s": round(dur, 1),
        "curve": {"pct_of_runtime": [int(v * 100) for v in M.GRID], "viewers_per_100": curve.round(1).tolist(),
                  "lower": np.maximum(curve + lo, 0).round(1).tolist(), "upper": (curve + hi).round(1).tolist(),
                  "band": "80 % of past videos fell inside"},
        "reference": {"viewers_per_100": ref.round(1).tolist(), "meaning": "typical Short of this length on this channel"},
        "segments": segments,
        "drop_moments": [{"t_s": d["t_s"], "reason": d["reason"]} for d in a["result"]["drop_moments"]],
        "drop_moments_evidence": EVIDENCE["drop_moments"],
        "rewatch_level": b["result"]["rewatch_level"],
        "rewatch_evidence": EVIDENCE["rewatch_level"] if b["neighbours"] else NO_EXAMPLES_NOTE,
        "advice": a["result"]["advice"],
        "similar_past_shorts_used": b["neighbours"],
        "caveats": CAVEATS,
        "cost_usd": round(cost, 4), "library_size": n, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv=None) -> int:
    from dotenv import load_dotenv

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    load_dotenv(".env")
    res = predict_retention(a.video)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text)
    print(json.dumps({k: v for k, v in res.items() if k not in ("curve", "reference")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
