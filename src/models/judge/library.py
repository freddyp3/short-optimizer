"""Everything the judging prompts are built from (plan/03c sections 19-20). Train + DEV only.

One ``Library`` holds, for the 145 videos: one-second timelines, structure, real outcomes, similarities for both
neighbour arms, out-of-fold model hints and the pilot / validation split. GATE and HELDOUT are never loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import cv, targets as T
from src.models.features import M1_COLS, design

JUDGE_DIR = Path("data/models/judge")
SPLIT_JSON = JUDGE_DIR / "split.json"
EMB_NPZ = JUDGE_DIR / "timeline_embeddings.npz"
RAW_DIR = Path("data/features/gemini/raw")
V3_RUNS = ("20261104T231214Z_fa305e9", "20261112T215720Z_6ca2cb5", "20261112T215722Z_6ca2cb5")
ENCODER = "sentence-transformers/all-mpnet-base-v2"

CHECKPOINTS = (0.10, 0.25, 0.50, 0.75, 0.95)
C_COLS = [f"c{int(r * 100)}" for r in CHECKPOINTS]
START = 0.02
TIERS = ("low", "medium", "high")
N_PILOT, SEED = 40, 340
POOL_E, K, CHUNK = 15, 5, 8
ANCHOR_N = 20


def load_timelines(df: pd.DataFrame, runs=V3_RUNS, raw_dir: Path = RAW_DIR) -> dict[str, list[dict]]:
    """video_id -> one-second timeline, from the v3 record (run 1) of whichever run holds it. Reads only ``df`` ids."""
    out = {}
    for v, b in zip(df["video_id"], df["blind_id"]):
        for run in runs:
            p = raw_dir / run / f"{b}__r1.json"
            if p.exists():
                rec = json.loads(p.read_text())
                if rec["status"] == "ok":
                    out[v] = rec["result"]["timeline"]
                    break
    return out


def timeline_lines(timeline: list[dict]) -> list[str]:
    return [f"{e['start_s']:g}-{e['end_s']:g} s | seen: {e['seen']} | heard: {e['heard']}" for e in timeline]


def chunked_embeddings(texts_per_video: list[list[str]], encode, chunk: int = CHUNK) -> np.ndarray:
    """One unit vector per video: mean of the embeddings of consecutive ``chunk``-line pieces (the encoder
    truncates long inputs, so a whole 60-second timeline cannot go in at once)."""
    out = []
    for lines in texts_per_video:
        pieces = ["\n".join(lines[i:i + chunk]) for i in range(0, len(lines), chunk)]
        e = np.asarray(encode(pieces)).mean(axis=0)
        out.append(e / np.linalg.norm(e))
    return np.vstack(out)


def curve_points(retention: pd.DataFrame, ids) -> pd.DataFrame:
    """Viewers still watching at each checkpoint per 100 present at the start (2 % of runtime)."""
    rows = {}
    for vid, g in retention[retention["video_id"].isin(set(ids))].sort_values("elapsedVideoTimeRatio").groupby("video_id"):
        x, y = g["elapsedVideoTimeRatio"].to_numpy(), g["audienceWatchRatio"].to_numpy()
        base = np.interp(START, x, y)
        rows[vid] = [100 * np.interp(r, x, y) / base for r in CHECKPOINTS]
    return pd.DataFrame.from_dict(rows, orient="index", columns=C_COLS).loc[list(ids)]


def tiers(values: pd.Series) -> pd.Series:
    return pd.qcut(values, 3, labels=list(TIERS)).astype(str)


def make_split(df: pd.DataFrame, n_pilot: int = N_PILOT, seed: int = SEED) -> dict:
    """Pilot videos stratified by duration third; the rest is validation."""
    third = pd.qcut(df["duration_s"], 3, labels=False)
    sizes = [n_pilot // 3 + (i < n_pilot % 3) for i in range(3)]
    pilot = [v for i in range(3) for v in
             df.loc[third == i, "video_id"].sort_values().sample(n=sizes[i], random_state=seed + i)]
    return {"seed": seed, "pilot": sorted(pilot), "validation": sorted(set(df["video_id"]) - set(pilot))}


def load_split(df: pd.DataFrame, path: Path = SPLIT_JSON) -> dict:
    """The split is written once and never regenerated."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_split(df), indent=2))
    s = json.loads(path.read_text())
    if set(s["pilot"]) & set(s["validation"]) or set(s["pilot"]) | set(s["validation"]) != set(df["video_id"]):
        raise ValueError("split.json does not partition the current pool")
    return s


def pick_contrast(order: np.ndarray, late: np.ndarray, pool: int = POOL_E) -> list[int]:
    """Arm E: from the ``pool`` most similar (``order`` is most similar first), the 2 best and 2 worst on
    late hold and the one closest to the pool's median; returned most similar first."""
    cand = list(order[:pool])
    by_late = sorted(cand, key=lambda j: late[j])
    chosen = by_late[:2] + by_late[-2:]
    rest = [j for j in cand if j not in chosen]
    med = np.median(late[cand])
    chosen.append(min(rest, key=lambda j: abs(late[j] - med)))
    return [j for j in cand if j in chosen]


def oof_folds(n: int) -> list[np.ndarray]:
    return cv.kfold(n, cv.K, np.random.default_rng(SEED))


def rf_late_oof(data, y: np.ndarray, recipe, folds) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    pred = np.empty(len(y))
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        Xtr, Xte, ytr, _ = design(recipe, data, tr, y[tr], te=f)
        pred[f] = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33,
                                        random_state=SEED, n_jobs=-1).fit(Xtr, ytr).predict(Xte)
    return pred


def implied_points(seg, pred: np.ndarray) -> np.ndarray:
    """(n_videos, 5) checkpoint values rebuilt from per-segment losses, per 100 viewers at the start."""
    keep = (1 - pred).reshape(seg.n_videos, seg.per_video)
    r = seg.r[: seg.per_video]
    return np.column_stack([100 * keep[:, (r >= START - 1e-9) & (r < c - 1e-9)].prod(axis=1) for c in CHECKPOINTS])


@dataclass
class Library:
    df: pd.DataFrame                 # one row per video, index = position
    timelines: dict[str, list[dict]]
    payoffs: dict[str, list[dict]]
    points: np.ndarray               # (n, 5) real checkpoint values
    rewatch: np.ndarray              # tier strings
    likes: np.ndarray
    late: np.ndarray                 # residualised late_hold, used only to pick contrast among neighbours
    sim_e: np.ndarray                # cosine similarity of timeline embeddings
    dist_s: np.ndarray               # Euclidean distance on z-scored M1 features
    hint_points: np.ndarray          # (n, 5) out-of-fold per-second model
    hint_late_pct: np.ndarray        # out-of-fold random forest late hold, as a percentile 0-100
    split: dict

    def __len__(self) -> int:
        return len(self.df)

    def index_of(self, video_id: str) -> int:
        return int(np.flatnonzero(self.df["video_id"].to_numpy() == video_id)[0])

    def others(self, i: int) -> np.ndarray:
        return np.setdiff1d(np.arange(len(self)), [i])

    def neighbours(self, i: int, arm: str, rng: np.random.Generator | None = None) -> list[tuple[int, float]]:
        """[(index, similarity 0-1)], most similar first. Never ``i`` itself."""
        o = self.others(i)
        if arm == "E":
            order = o[np.argsort(-self.sim_e[i, o], kind="stable")]
            return [(int(j), float(self.sim_e[i, j])) for j in pick_contrast(order, self.late)]
        if arm == "S":
            order = o[np.argsort(self.dist_s[i, o], kind="stable")][:K]
            return [(int(j), float(1 / (1 + self.dist_s[i, j]))) for j in order]
        if arm.startswith("T"):          # threshold arm: only videos at least this similar, at most K, may be none
            order = o[np.argsort(-self.sim_e[i, o], kind="stable")][:K]
            return [(int(j), float(self.sim_e[i, j])) for j in order if self.sim_e[i, j] >= int(arm[1:]) / 100]
        if arm == "random":
            return [(int(j), float(self.sim_e[i, j])) for j in rng.choice(o, K, replace=False)]
        raise ValueError(arm)

    def anchor(self, i: int) -> np.ndarray:
        """Mean real curve of the ``ANCHOR_N`` other videos closest in length."""
        o = self.others(i)
        d = self.df["duration_s"].to_numpy()
        return self.points[o[np.argsort(np.abs(d[o] - d[i]), kind="stable")][:ANCHOR_N]].mean(axis=0)


def embed_timelines(df: pd.DataFrame, timelines: dict, path: Path = EMB_NPZ) -> np.ndarray:
    ids = df["video_id"].tolist()
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if z["ids"].tolist() == ids:
            return z["emb"]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER)
    emb = chunked_embeddings([timeline_lines(timelines[v]) for v in ids],
                             lambda p: model.encode(p, normalize_embeddings=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), emb=emb)
    return emb


def build(hazard_extra: bool = False) -> Library:
    from src.models import hazard as H

    df, data, recipes, export = T.load()
    timelines = load_timelines(df)
    missing = sorted(set(df["video_id"]) - set(timelines))
    if missing:
        raise RuntimeError(f"{len(missing)} videos have no valid one-second timeline, e.g. {missing[:3]}")
    emb = embed_timelines(df, timelines)
    X = data.matrix(M1_COLS)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    folds = oof_folds(len(df))
    payoffs = H.load_payoffs(df["video_id"])
    seg = H.build(df, export.retention, payoffs)
    Xseg = np.column_stack([seg.base, seg.cov] + ([H.timeline_covariates(df, timelines, seg)] if hazard_extra else []))
    rf = rf_late_oof(data, df["late_hold"].to_numpy(dtype=float), recipes["M1_core_mechanics"], folds)
    return Library(
        df=df.reset_index(drop=True), timelines=timelines, payoffs=payoffs,
        points=curve_points(export.retention, df["video_id"]).to_numpy(),
        rewatch=tiers(df["replay_level"]).to_numpy(), likes=tiers(df["like_rate"]).to_numpy(),
        late=df["late_hold"].to_numpy(dtype=float), sim_e=emb @ emb.T,
        dist_s=np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)),
        hint_points=implied_points(seg, H.oof(seg, Xseg, "ridge", folds)),
        hint_late_pct=100 * (pd.Series(rf).rank(pct=True).to_numpy()),
        split=load_split(df))
