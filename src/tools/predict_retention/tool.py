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

``analyze_draft`` is the entry point for the server (plan/10a section 6): the same prediction plus the artifacts the
audience tools need, with an optional caller-supplied key, cost and step callbacks, and cleanup of everything made
for the draft (temporary copy, upload cache, files in Gemini's store). ``predict_retention`` wraps it for the CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
import threading
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
from src.tools.analyze_video.measure import ff_env, ff_input, measure

log = logging.getLogger(__name__)
SPLITS = ("train", "dev", "gate", "heldout")
SETS = ("pilot", "validation", "gate", "heldout")
SEGMENT_TARGET = {"early": "early_hold", "late": "late_hold"}
EVIDENCE = {
    "early": "Moderate evidence: confirmed on one set of 50 unseen videos (rank correlation 0.56). Predicts direction, not exact numbers.",
    "mid": "Good evidence: confirmed on two separate sets of unseen videos (rank correlations 0.56 and 0.63). Predicts direction, not exact numbers.",
    "late": "Good evidence: confirmed on two separate sets of unseen videos (rank correlations 0.48 and 0.60). Predicts direction, not exact numbers.",
    "drop_moments": "Good evidence: on two separate sets of unseen videos the predicted moments were steeper drops than chance "
                    "(score 0.65 and 0.66, where 0.50 is chance and 1.00 is always the steepest second).",
    "rewatch_level": "Weak evidence: confirmed on two sets of unseen videos but with small effect (rank correlations 0.43 and 0.27). Treat as a hint.",
}
NO_EXAMPLES_NOTE = ("Guess only: no past Short was similar enough to compare with, and without examples this item was not "
                    "reliable in testing.")
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
STEPS = ("measuring", "describing", "judging", "modelling")
STRUCTURE_FIELDS = ("t_premise_clear_s", "tail_all", "dead_time_s", "cuts_per_s", "arc", "payoff_count_all", "main_payoff_frac")
REDACTED = "[redacted key]"
BLIND_COPY_TIMEOUT_S = 60


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


@dataclass
class DraftAnalysis:
    result: dict       # the predict_retention output; library videos appear only as neutral labels
    artifacts: dict    # timeline, payoffs, structure, embedding, duration_s (JSON-serialisable; input of audience/brief.py)
    cost_usd: float


_STATE: State | None = None
_STATE_LOCK = threading.Lock()
_ENCODER = None
_ENCODER_LOCK = threading.Lock()
_SECRETS: set[str] = set()
_SECRETS_LOCK = threading.Lock()
_FACTORY_INSTALLED = False


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
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = build_state()
    return _STATE


def build_state() -> State:
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
    return State(lib, emb, export, curves, ref, signal, M.calibrate(pieces, np.arange(n)), M.band_coverage(pred, pieces)[1], ps, seg)


def blind_copy(path: Path, out_dir: Path) -> Path:
    """Copy without container metadata, so no title or date travels with the file. Same restrictions as every
    ffmpeg call on a draft: local file only, accepted demuxers only, PATH-only environment, 60 s limit."""
    out = out_dir / f"draft_{sha256_file(path)[:10]}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *ff_input(path), "-map_metadata", "-1", "-c", "copy", str(out)],
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=BLIND_COPY_TIMEOUT_S,
                   shell=False, check=True, env=ff_env())
    return out


def extract(path: Path, blind_id: str, dur: float, version: str, client=None, uploads_path: Path | None = None) -> dict:
    """One extraction record (status ok or invalid). ``client`` None means the paid key from the environment;
    ``uploads_path`` None means the research upload cache."""
    from src.tools.analyze_video import extract as E
    from src.tools.analyze_video.schema import render_prompt

    pf = E.PROMPTS_DIR / f"{version}.md"
    kw = {} if uploads_path is None else {"uploads_path": uploads_path}
    return E.GeminiFeatureExtractor(client=client, model=C.DEFAULT_MODEL, version=version, **kw).analyze(
        blind_id, path, render_prompt(pf.read_text(), dur), dur, fps=C.FPS, media_resolution=C.MEDIA_RESOLUTION,
        prompt_version=version, prompt_sha256=sha256_file(pf))


def extraction_cost(rec: dict) -> float:
    """Cost of the record's call plus the invalid first attempt that preceded it, if any."""
    first = rec.get("first_attempt_invalid") or rec.get("first_attempt") or {}
    return float(rec.get("cost_usd_paid_equiv", 0.0)) + float(first.get("cost_usd_paid_equiv", 0.0))


def delete_uploads(client, uploads_path: Path) -> None:
    """Remove from Gemini's file store every file recorded in this job's upload cache."""
    if not uploads_path.exists():
        return
    for ent in json.loads(uploads_path.read_text()).values():
        try:
            client.files.delete(name=ent["file_name"])
        except Exception as e:
            log.warning("could not delete an uploaded draft file (%s); Gemini removes it after 48 h", type(e).__name__)


def encoder():
    """The sentence encoder, loaded once per process."""
    global _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None:
            from sentence_transformers import SentenceTransformer
            _ENCODER = SentenceTransformer(ENCODER)
    return _ENCODER


def draft_embedding(timeline: list[dict]) -> np.ndarray:
    """(1, d) unit vector for the draft's one-second timeline, made the same way as the library's."""
    model = encoder()
    with _ENCODER_LOCK:
        return chunked_embeddings([timeline_lines(timeline)], lambda p: model.encode(p, normalize_embeddings=True))


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


def with_draft(st: State, row: pd.Series, timeline: list[dict], payoffs: list[dict], e: np.ndarray | None = None) -> Library:
    """Library plus the draft as the last row. The draft is outside the pool and has no outcomes."""
    lib, n = st.lib, len(st.lib)
    df = pd.concat([lib.df, row.to_frame().T], ignore_index=True)
    for c in lib.df.columns:
        if pd.api.types.is_numeric_dtype(lib.df[c]):
            df[c] = pd.to_numeric(df[c])
    emb = np.vstack([st.emb, draft_embedding(timeline) if e is None else e])
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


def runtime_pct() -> list[int]:
    """The curve's grid as whole percents of the runtime (rounded: truncating repeated 28 and skipped 29)."""
    return [round(v * 100) for v in M.GRID]


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


def build_result(st: State, lib: Library, row: pd.Series, v2: dict, a: dict, b: dict, dur: float, cost: float) -> dict:
    """The publish-plan numbers from the four records. ``similar_past_shorts_used`` still holds library video ids
    here; ``analyze_draft`` replaces them before anything leaves this module."""
    n = len(st.lib)
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
        "curve": {"pct_of_runtime": runtime_pct(), "viewers_per_100": curve.round(1).tolist(),
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

def structure_of(row: pd.Series) -> dict:
    out = {}
    for k in STRUCTURE_FIELDS:
        v = row.get(k)
        out[k] = v if isinstance(v, str) else None if v is None or v != v else float(v)
    return out


def _install_redaction() -> None:
    """Every log record made while a job runs has that job's key replaced, whichever logger or thread wrote it."""
    global _FACTORY_INSTALLED
    with _SECRETS_LOCK:
        if _FACTORY_INSTALLED:
            return
        inner = logging.getLogRecordFactory()

        def factory(*args, **kw):
            rec = inner(*args, **kw)
            if not _SECRETS:
                return rec
            secrets = tuple(_SECRETS)
            try:
                msg = rec.getMessage()
            except Exception:
                msg = str(rec.msg)
            if any(k in msg for k in secrets):
                rec.msg, rec.args = scrub(msg, secrets), None
            if rec.exc_info:
                text = logging.Formatter().formatException(rec.exc_info)
                if any(k in text for k in secrets):
                    rec.exc_info, rec.exc_text = None, scrub(text, secrets)
            return rec

        logging.setLogRecordFactory(factory)
        _FACTORY_INSTALLED = True


def scrub(text: str, secrets) -> str:
    for k in secrets:
        text = text.replace(k, REDACTED)
    return text


def safe_error(e: BaseException, secret: str | None) -> Exception | None:
    """A replacement for ``e`` when its text, or that of any exception it was raised from, contains the key."""
    if not secret:
        return None
    chain, x = [], e
    while x is not None and x not in chain:
        chain.append(x)
        x = x.__cause__ or x.__context__
    if not any(secret in f"{c} {c!r}" for c in chain):
        return None
    return RuntimeError(f"{type(e).__name__}: {scrub(str(e), (secret,))}")


def analyze_draft(video_path: str | Path, api_key: str | None = None, on_cost=None, on_step=None, *, client=None) -> DraftAnalysis:
    """Full analysis of one draft. ``api_key`` None means GEMINI_API_KEY_PAID; a given key is used for all four
    Gemini calls of this job and nothing else, and never reaches a log, a file, the output or an exception text.
    ``on_cost(usd)`` runs after every paid call (an exception from it stops the job), ``on_step(name)`` at each of
    ``STEPS``. ``client`` (tests, CLI) replaces the client built from the key."""
    if client is not None and api_key is not None:
        raise ValueError("give a client or an api_key, not both")
    secret = None
    if client is None:
        secret = os.environ.get("GEMINI_API_KEY_PAID") if api_key is None else api_key
        if not secret or not secret.strip():
            raise RuntimeError("GEMINI_API_KEY_PAID not set" if api_key is None else "the supplied Gemini key is empty")
        _install_redaction()
        with _SECRETS_LOCK:
            _SECRETS.add(secret)
    err = None
    try:
        return _analyze(Path(video_path), client or new_genai_client(secret), on_cost, on_step)
    except Exception as e:
        err = safe_error(e, secret)
        if err is None:
            raise
    finally:
        if secret:
            with _SECRETS_LOCK:
                _SECRETS.discard(secret)
    raise err          # raised outside the handler, so the original exception is not attached as context


def _analyze(video_path: Path, client, on_cost, on_step) -> DraftAnalysis:
    from src.tools.audience.brief import library_label

    spent = [0.0]

    def paid(usd: float) -> None:
        spent[0] += usd
        if on_cost:
            on_cost(usd)

    step = on_step or (lambda name: None)
    step("measuring")
    st = state()
    n = len(st.lib)
    with tempfile.TemporaryDirectory(prefix="draft_") as tmp:
        uploads = Path(tmp) / "uploads.json"
        try:
            blind = blind_copy(video_path, Path(tmp))
            code = measure(blind)
            dur = float(code["duration_s"])
            step("describing")
            recs = {}
            for version in ("v2", "v3"):
                rec = recs[version] = extract(blind, f"{blind.stem}_{version}", dur, version, client=client, uploads_path=uploads)
                paid(extraction_cost(rec))
                if rec["status"] != "ok":
                    raise RuntimeError(f"extraction {version} failed: {rec.get('invalid_reason')}")
        finally:
            delete_uploads(client, uploads)
    v2, v3 = recs["v2"], recs["v3"]
    timeline, payoffs = v3["result"]["timeline"], v2["result"]["payoffs"]
    row = draft_row(v2, code, dur, st.lib.df.iloc[0])
    e = draft_embedding(timeline)
    lib = with_draft(st, row, timeline, payoffs, e)
    step("judging")
    judged = []
    for variant, arm in (("L0", "E"), ("L2", "T75")):
        rec = judge_one(lib, n, variant, arm, C.DEFAULT_MODEL, client)
        paid(rec["cost_usd"])
        judged.append(rec)
    a, b = judged
    if a["status"] != "ok" or b["status"] != "ok":
        raise RuntimeError(f"judging failed: {a.get('invalid_reason')} / {b.get('invalid_reason')}")
    step("modelling")
    result = build_result(st, lib, row, v2, a, b, dur, spent[0])
    result["similar_past_shorts_used"] = [library_label(st.lib.index_of(v)) for v in result["similar_past_shorts_used"]]
    artifacts = {"timeline": timeline, "payoffs": payoffs, "structure": structure_of(row),
                 "embedding": [float(x) for x in np.asarray(e).ravel()], "duration_s": dur}
    return DraftAnalysis(result=result, artifacts=json.loads(json.dumps(artifacts)), cost_usd=spent[0])


def predict_retention(video_path: str | Path, client=None) -> dict:
    return analyze_draft(video_path, client=client).result


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
