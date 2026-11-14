"""The retention simulator: one predicted audience curve per draft (plan/03d).

Three segment ratios (early 2-25 %, mid 25-75 %, late 75-95 % of the runtime) come from three specialists, each
calibrated as  ratio = reference ratio + a + b * signal.  The shape inside a segment comes from the per-second
model. With no specialist and no shape the output is exactly the reference curve (what a Short of that length
does on this channel).

    ./venv/bin/python -m src.models.judge.simulator onecall    # S2: can one v3 extraction feed the models?
    ./venv/bin/python -m src.models.judge.simulator build      # S3 + S4 on the 145, everything out of fold
    ./venv/bin/python -m src.models.judge.simulator gate       # S6: check of the frozen configuration on GATE
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.data import load_export
from src.evals.hook_test.stats import spearman
from src.models import hazard as H
from src.models import targets as T
from src.models.features import M1_COLS, prepare
from src.models.judge import score as S
from src.models.judge.library import ANCHOR_N, JUDGE_DIR, K, SEED, Library, build, oof_folds, rf_late_oof

GRID = np.arange(1, 101) / 100                       # r = 0.01 ... 1.00
I2, I25, I75, I95 = 1, 24, 74, 94                    # indices of 2 %, 25 %, 75 %, 95 %
SEGMENTS = {"early": (I2, I25), "mid": (I25, I75), "late": (I75, I95)}
CHECK_IDX = [9, 24, 49, 74, 94]                      # 10, 25, 50, 75, 95 %
RATIO_CLIP = (0.2, 1.6)
BAND = (10, 90)
CONFIG_JSON = JUDGE_DIR / "simulator_config.json"
N_FLIPS = 10_000


def normalise(curves: np.ndarray) -> np.ndarray:
    """Viewers per 100 present at 2 % of the runtime."""
    return 100 * curves / curves[:, [I2]]


def raw_curves(retention: pd.DataFrame, ids) -> np.ndarray:
    g = retention[retention["video_id"].isin(set(ids))].sort_values(["video_id", "elapsedVideoTimeRatio"])
    by = {v: x["audienceWatchRatio"].to_numpy() for v, x in g.groupby("video_id")}
    return np.vstack([by[v] for v in ids])


def reference(curves: np.ndarray, durations: np.ndarray, pool: np.ndarray, d: float, exclude=()) -> np.ndarray:
    """Mean normalised curve of the ANCHOR_N pool videos closest in length (never the excluded ones)."""
    p = np.setdiff1d(pool, list(exclude))
    return curves[p[np.argsort(np.abs(durations[p] - d), kind="stable")][:ANCHOR_N]].mean(axis=0)


def ratios(curve: np.ndarray) -> np.ndarray:
    """(..., 3): early, mid, late ratios of a curve or a stack of curves."""
    return np.stack([curve[..., b] / curve[..., a] for a, b in SEGMENTS.values()], axis=-1)


def assemble(ref: np.ndarray, targets: dict[str, float] | None = None, shape_loss: np.ndarray | None = None) -> np.ndarray:
    """100-point curve. ``targets``: segment -> ratio (missing segments keep the reference's ratio).
    ``shape_loss``: per-step relative losses (99) from the per-second model; None keeps the reference's shape."""
    targets = targets or {}
    g_ref = np.log(ref[:-1]) - np.log(ref[1:])
    g = g_ref.copy() if shape_loss is None else -np.log(1 - np.clip(shape_loss, -0.5, 0.9))
    for name, (a, b) in SEGMENTS.items():
        t = float(np.clip(targets.get(name, ref[b] / ref[a]), *RATIO_CLIP))
        g[a:b] += (-np.log(t) - g[a:b].sum()) / (b - a)
    if shape_loss is not None:                       # outside the three segments the shape is not anchored: keep the reference
        g[:I2], g[I95:] = g_ref[:I2], g_ref[I95:]
    logw = np.empty(100)
    logw[I2] = np.log(100.0)
    logw[I2 + 1:] = logw[I2] - np.cumsum(g[I2:])
    logw[0] = logw[I2] + g[0]
    return np.exp(logw)


def fit_calibration(signal: np.ndarray, deviation: np.ndarray) -> tuple[float, float]:
    """ratio deviation = a + b * signal by least squares; a negative slope is not trusted and becomes 0."""
    X = np.column_stack([np.ones(len(signal)), signal])
    a, b = np.linalg.lstsq(X, deviation, rcond=None)[0]
    return (float(deviation.mean()), 0.0) if b < 0 else (float(a), float(b))


@dataclass
class Pieces:
    ids: list[str]
    duration: np.ndarray
    truth: np.ndarray            # (n, 100) real normalised curves
    ref: np.ndarray              # (n, 100) reference curves (never include the video itself)
    signal: dict[str, np.ndarray]   # segment -> (n,) specialist signal
    ps_loss: np.ndarray          # (n, 99) per-second model losses
    rows: pd.DataFrame


def gemini_mid_signal(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Gemini's 75/25 ratio minus the reference's: the prompt showed Gemini that same reference."""
    return points[:, 3] / points[:, 1] - ref[:, I75] / ref[:, I25]


def neighbour_early_signal(curves: np.ndarray, durations: np.ndarray, pool: np.ndarray, nb: list[list[int]],
                           self_idx: list[int | None]) -> np.ndarray:
    """Mean over the structure neighbours of (their early ratio - their own reference's early ratio). A neighbour's
    reference never contains the neighbour or the video being predicted."""
    out = []
    for i, js in zip(self_idx, nb):
        dev = []
        for j in js:
            r = reference(curves, durations, pool, durations[j], exclude=[j] + ([i] if i is not None else []))
            dev.append(curves[j, I25] / curves[j, I2] - r[I25] / r[I2])
        out.append(np.mean(dev))
    return np.array(out)


def library_pieces(lib: Library, export, feature_df: pd.DataFrame | None = None) -> Pieces:
    """Everything for the 145, each video's inputs made without its own outcome."""
    df = lib.df if feature_df is None else feature_df
    n = len(lib)
    ids = lib.df["video_id"].tolist()
    curves = normalise(raw_curves(export.retention, ids))
    dur = lib.df["duration_s"].to_numpy(dtype=float)
    pool = np.arange(n)
    ref = np.vstack([reference(curves, dur, pool, dur[i], exclude=[i]) for i in range(n)])
    data = prepare(df)
    folds = oof_folds(n)
    X = data.matrix(M1_COLS)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    nb = [list(np.setdiff1d(pool, [i])[np.argsort(dist[i, np.setdiff1d(pool, [i])], kind="stable")][:K]) for i in range(n)]
    g = pd.concat([S.load_run(f"j1_L0_E_3.8-flash_{s}") for s in ("pilot", "validation")]).set_index("video_id").loc[ids]
    seg = H.build(df, export.retention, lib.payoffs if feature_df is None else payoffs_from(feature_df))
    ps = H.oof(seg, np.column_stack([seg.base, seg.cov]), "ridge", folds).reshape(n, -1)
    recipe = T.all_recipes(data)["M1_core_mechanics"]
    signal = {"early": neighbour_early_signal(curves, dur, pool, nb, list(range(n))),
              "mid": gemini_mid_signal(g[["p10", "p25", "p50", "p75", "p95"]].to_numpy(dtype=float), ref),
              "late": rf_late_oof(data, lib.df["late_hold"].to_numpy(dtype=float), recipe, folds)}
    return Pieces(ids, dur, curves, ref, signal, ps, lib.df)


def payoffs_from(df: pd.DataFrame) -> dict:
    return dict(zip(df["video_id"], df["_payoffs"]))


def predict(p: Pieces, calib: dict[str, tuple[float, float]], use: tuple[str, ...], shape: bool, rows=None) -> np.ndarray:
    rows = np.arange(len(p.ids)) if rows is None else rows
    ref_ratio = ratios(p.ref)
    out = []
    for i in rows:
        targets = {s: ref_ratio[i, k] + calib[s][0] + calib[s][1] * p.signal[s][i]
                   for k, s in enumerate(SEGMENTS) if s in use}
        out.append(assemble(p.ref[i], targets, p.ps_loss[i] if shape else None))
    return np.vstack(out)


def calibrate(p: Pieces, rows: np.ndarray) -> dict[str, tuple[float, float]]:
    dev = ratios(p.truth) - ratios(p.ref)
    return {s: fit_calibration(p.signal[s][rows], dev[rows, k]) for k, s in enumerate(SEGMENTS)}


def oof_predict(p: Pieces, use: tuple[str, ...], shape: bool) -> np.ndarray:
    """Out-of-fold assembled curves: the calibration for a video is fitted on the other folds."""
    n = len(p.ids)
    out = np.empty((n, 100))
    for f in oof_folds(n):
        out[f] = predict(p, calibrate(p, np.setdiff1d(np.arange(n), f)), use, shape, f)
    return out


def step_loss(curves: np.ndarray) -> np.ndarray:
    return 1 - curves[:, 1:] / curves[:, :-1]


def measures(pred: np.ndarray, p: Pieces, rng=None) -> dict:
    rng = rng or np.random.default_rng(SEED)
    e_model = np.abs(pred[:, CHECK_IDX] - p.truth[:, CHECK_IDX]).mean(axis=1)
    e_ref = np.abs(p.ref[:, CHECK_IDX] - p.truth[:, CHECK_IDX]).mean(axis=1)
    d = e_ref - e_model
    flips = rng.choice([-1.0, 1.0], size=(N_FLIPS, len(d)))
    out = {"curve_error": float(e_model.mean()), "reference_error": float(e_ref.mean()),
           "skill": float(1 - e_model.mean() / e_ref.mean()),
           "skill_p": float((1 + np.sum((flips * d).mean(axis=1) >= d.mean())) / (1 + N_FLIPS)),
           "share_of_videos_improved": float(np.mean(d > 0)),
           "full_curve_skill": float(1 - np.abs(pred - p.truth).mean() / np.abs(p.ref - p.truth).mean())}
    with np.errstate(invalid="ignore"):
        out["deviation_corr"] = [spearman(pred[:, k] - p.ref[:, k], p.truth[:, k] - p.ref[:, k]) for k in CHECK_IDX]
        lt, lp, lr = step_loss(p.truth), step_loss(pred), step_loss(p.ref)
        out["shape_corr"] = float(np.nanmean([spearman(a, b) for a, b in zip(lp, lt)]))
        out["shape_corr_reference"] = float(np.nanmean([spearman(a, b) for a, b in zip(lr, lt)]))
    seg_err = lambda c: np.abs(ratios(c) - ratios(p.truth)).mean(axis=0)
    out["segment_skill"] = dict(zip(SEGMENTS, (1 - seg_err(pred) / seg_err(p.ref)).round(4).tolist()))
    return out


def band_coverage(pred: np.ndarray, p: Pieces) -> tuple[float, np.ndarray]:
    """80 % band per checkpoint from other folds' out-of-fold residuals; returns coverage and the pooled band."""
    n = len(p.ids)
    res = p.truth[:, CHECK_IDX] - pred[:, CHECK_IDX]
    inside = np.zeros_like(res, dtype=bool)
    for f in oof_folds(n):
        lo, hi = np.percentile(res[np.setdiff1d(np.arange(n), f)], BAND, axis=0)
        inside[f] = (res[f] >= lo) & (res[f] <= hi)
    return float(inside.mean()), np.percentile(res, BAND, axis=0)


def show(name: str, m: dict) -> None:
    print(f"{name:44}{m['curve_error']:>7.2f}{m['skill']:>8.3f}{m['skill_p']:>8.4f}{m['share_of_videos_improved']:>8.2f}"
          f"{m['full_curve_skill']:>8.3f}{m['shape_corr']:>7.2f}   " + " ".join(f"{x:5.2f}" for x in m["deviation_corr"])
          + "   " + " ".join(f"{k[0]}:{v:+.2f}" for k, v in m["segment_skill"].items()))


HEADER = (f"{'configuration':44}{'error':>7}{'skill':>8}{'p':>8}{'better':>8}{'full':>8}{'shape':>7}"
          "   deviation corr at 10/25/50/75/95    segment skill")

LADDER = [("reference only (length)", (), False), ("+ late (forest)", ("late",), False),
          ("+ mid (Gemini)", ("late", "mid"), False), ("+ early (structure neighbours)", ("late", "mid", "early"), False),
          ("late + mid + per-second shape", ("late", "mid"), True),
          ("late + mid + early + per-second shape", ("late", "mid", "early"), True),
          ("per-second shape only", (), True)]


def cmd_build() -> int:
    lib = build()
    export = load_export(Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"]))
    p = library_pieces(lib, export)
    full = calibrate(p, np.arange(len(p.ids)))
    print("calibration on all 145 (a, b):", {k: (round(a, 4), round(b, 3)) for k, (a, b) in full.items()})
    print("signal vs real deviation (rank corr):", {s: round(spearman(p.signal[s], (ratios(p.truth) - ratios(p.ref))[:, k]), 2)
                                                    for k, s in enumerate(SEGMENTS)})
    print("\n" + HEADER)
    res = {}
    for name, use, shape in LADDER:
        pred = oof_predict(p, use, shape)
        res[name] = measures(pred, p)
        show(name, res[name])
        res[name]["coverage"] = band_coverage(pred, p)[0]
    (S.SCORES / "simulator_build.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n80 % band coverage (out of fold):", {k: round(v["coverage"], 3) for k, v in res.items()})
    return 0


def cmd_gate() -> int:
    from src.models.judge import gate as G

    cfg = json.loads(CONFIG_JSON.read_text())
    lib = build()
    export = load_export(Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"]))
    p = library_pieces(lib, export)
    calib = {k: tuple(v) for k, v in cfg["calibration"].items()}
    gate = G.build_gate()
    glib, idx, rows = gate.lib, gate.gate_idx, gate.rows
    n = len(p.ids)
    gids = rows["video_id"].tolist()
    gdur = rows["duration_s"].to_numpy(dtype=float)
    truth = normalise(raw_curves(export.retention, gids))
    pool = np.arange(n)
    ref = np.vstack([reference(p.truth, p.duration, pool, d) for d in gdur])
    nb = [[j for j, _ in glib.neighbours(int(i), "S")] for i in idx]
    run = S.load_run("j1_L0_E_3.8-flash_gate").set_index("video_id").loc[gids]
    seg = H.build(lib.df, export.retention, lib.payoffs)
    segg = H.build(rows, export.retention, glib.payoffs)
    ps = H.fit_predict("ridge", np.column_stack([seg.base, seg.cov]), seg.y, np.column_stack([segg.base, segg.cov])).reshape(len(gids), -1)
    signal = {"early": neighbour_early_signal(p.truth, p.duration, pool, nb, [None] * len(gids)),
              "mid": gemini_mid_signal(run[["p10", "p25", "p50", "p75", "p95"]].to_numpy(dtype=float), ref),
              "late": gate.forest[:, 2]}
    gp = Pieces(gids, gdur, truth, ref, signal, ps, rows)
    print("GATE check of the frozen simulator (not a clean test: GATE's source table has been seen)\n" + HEADER)
    out = {}
    for name, use, shape in [("reference only (length)", (), False), (f"FROZEN: {cfg['name']}", tuple(cfg["use"]), cfg["shape"])]:
        pred = predict(gp, calib, use, shape)
        out[name] = measures(pred, gp)
        show(name, out[name])
    res = truth[:, CHECK_IDX] - pred[:, CHECK_IDX]
    lo, hi = np.array(cfg["band"])
    out["coverage"] = float(((res >= lo) & (res <= hi)).mean())
    print(f"80 % band coverage on GATE: {out['coverage']:.3f}")
    (S.SCORES / "simulator_gate.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


def v3_feature_frame(lib: Library) -> pd.DataFrame:
    """The library table with the Gemini-derived M1 features recomputed from each video's v3 record."""
    from src.models.judge.verify import v3_records
    from src.tools.analyze_video.ab_score import ARC_ORD
    from src.tools.analyze_video.derive import derive_features

    df = lib.df.copy()
    recs = v3_records(df)
    pay = []
    for i, row in enumerate(df.itertuples()):
        r = recs[row.video_id]
        d = derive_features(r["result"], r["duration_s_file"], [])
        df.loc[i, "t_premise_clear_s"] = d["t_premise_clear_s"] if d["t_premise_clear_s"] is not None else row.duration_s
        for c in ("payoff_count_all", "main_payoff_frac", "tail_all", "dead_time_s"):
            df.loc[i, c] = d[c]
        df.loc[i, "arc_ord"] = ARC_ORD[d["arc"]]
        pay.append(r["result"]["payoffs"])
    df["_payoffs"] = pay
    return df


def cmd_onecall() -> int:
    lib = build()
    export = load_export(Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"]))
    a = library_pieces(lib, export)
    b = library_pieces(lib, export, v3_feature_frame(lib))
    dev = ratios(a.truth) - ratios(a.ref)
    print(f"{'signal (rank corr with real deviation)':42}{'v2 features':>12}{'v3 features':>12}")
    worst = 0.0
    for k, s in enumerate(SEGMENTS):
        x, y = spearman(a.signal[s], dev[:, k]), spearman(b.signal[s], dev[:, k])
        if s != "mid":
            worst = max(worst, x - y)
        print(f"{s + (' (Gemini; unchanged)' if s == 'mid' else ''):42}{x:>12.3f}{y:>12.3f}")
    wa = float(np.nanmean([spearman(u, v) for u, v in zip(a.ps_loss, step_loss(a.truth))]))
    wb = float(np.nanmean([spearman(u, v) for u, v in zip(b.ps_loss, step_loss(b.truth))]))
    print(f"{'per-second shape (within-video corr)':42}{wa:>12.3f}{wb:>12.3f}")
    worst = max(worst, wa - wb)
    print(f"largest loss from using one v3 call: {worst:.3f} -> {'ONE CALL IS ENOUGH' if worst <= 0.05 else 'KEEP TWO CALLS'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["onecall", "build", "gate"])
    a = ap.parse_args(argv)
    return {"onecall": cmd_onecall, "build": cmd_build, "gate": cmd_gate}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------- candidate C2: direct calibration of the anchor levels

ANCHORS = (I25, I75, I95)


def direct_features(p: Pieces) -> np.ndarray:
    """(n, 4 + 3): the three specialist signals, a constant, and the per-second model's own deviation at each anchor."""
    ps = np.vstack([assemble(p.ref[i], None, p.ps_loss[i]) for i in range(len(p.ids))])
    return np.column_stack([np.ones(len(p.ids)), p.signal["early"], p.signal["mid"], p.signal["late"],
                            np.log(ps[:, ANCHORS]) - np.log(p.ref[:, ANCHORS])])


def fit_direct(F: np.ndarray, p: Pieces, rows: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """(7, 3) coefficients: log deviation of the real level at each anchor on the features (light ridge, not on the constant)."""
    Y = np.log(p.truth[rows][:, ANCHORS]) - np.log(p.ref[rows][:, ANCHORS])
    X = F[rows]
    mu, sd = X[:, 1:].mean(axis=0), X[:, 1:].std(axis=0)
    Z = np.column_stack([np.ones(len(rows)), (X[:, 1:] - mu) / sd])
    pen = np.diag([0.0] + [ridge] * (Z.shape[1] - 1))
    return np.linalg.solve(Z.T @ Z + pen, Z.T @ Y), mu, sd


def predict_direct(F: np.ndarray, p: Pieces, model, rows: np.ndarray, shape: bool = True) -> np.ndarray:
    coef, mu, sd = model
    Z = np.column_stack([np.ones(len(rows)), (F[rows][:, 1:] - mu) / sd])
    lvl = p.ref[rows][:, ANCHORS] * np.exp(Z @ coef)
    out = []
    for k, i in enumerate(rows):
        t = {"early": lvl[k, 0] / 100, "mid": lvl[k, 1] / lvl[k, 0], "late": lvl[k, 2] / lvl[k, 1]}
        out.append(assemble(p.ref[i], t, p.ps_loss[i] if shape else None))
    return np.vstack(out)


def oof_direct(p: Pieces, ridge: float = 1.0, shape: bool = True) -> np.ndarray:
    n = len(p.ids)
    F = direct_features(p)
    out = np.empty((n, 100))
    for f in oof_folds(n):
        out[f] = predict_direct(F, p, fit_direct(F, p, np.setdiff1d(np.arange(n), f), ridge), f, shape)
    return out
