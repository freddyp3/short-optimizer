"""Do similar past Shorts predict the outcome? Free check before nearest-example prompting (plan/03c section 17).

    ./venv/bin/python -m src.models.neighbours           # theme / mechanics neighbours (section 17)
    ./venv/bin/python -m src.models.neighbours --text    # title / description embeddings (section 18)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.evals.hook_test.stats import spearman
from src.models import cv, targets as T
from src.models.dataset import GEMINI_EVAL, GEMINI_TRAIN
from src.models.features import M1_COLS, history_features
from src.models.hazard import RAW_DIR

TARGETS = ("late_hold", "share_rate", "early_hold")
VARIANTS = ("theme_past_k5", "theme_any_k5", "theme_any_k10", "mech_any_k10")
TEXT_VARIANTS = ("title_any_k1", "title_any_k10", "desc_any_k1", "desc_any_k10")
ENCODER = "sentence-transformers/all-mpnet-base-v2"
REPEATS, N_PERM = 3, 1000


def timelines(video_ids) -> list[str]:
    """Gemini's timeline text from the raw record each video's frozen features came from."""
    src = pd.concat([pd.read_csv(p, usecols=["video_id", "blind_id", "source_run_id"])
                     for p in (GEMINI_TRAIN, GEMINI_EVAL)]).set_index("video_id")
    return [json.loads((RAW_DIR / src.at[v, "source_run_id"] / f"{src.at[v, 'blind_id']}__r1.json")
                       .read_text())["result"]["timeline"] for v in video_ids]


def text_distances(df, export) -> dict[str, np.ndarray]:
    """Cosine distance between MPNet embeddings of titles and of timeline descriptions (local model, no API)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER)
    titles = df["video_id"].map(export.videos.set_index("video_id")["title"]).fillna("").tolist()
    out = {}
    for name, texts in (("title", titles), ("desc", timelines(df["video_id"]))):
        E = model.encode(texts, normalize_embeddings=True)
        out[name] = 1 - E @ E.T
    return out


def distances(data) -> dict[str, np.ndarray]:
    theme = (data.codes[:, None, :] != data.codes[None, :, :]).mean(axis=2)
    X = data.matrix(M1_COLS)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    return {"theme": theme, "mech": np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))}


def knn_mean(dist: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, k: int) -> np.ndarray:
    """Mean label of the k nearest training rows (ties broken by row order)."""
    near = np.argsort(dist[np.ix_(te, tr)], axis=1, kind="stable")[:, :k]
    return y[tr][near].mean(axis=1)


def oof(variant: str, data, dist: dict, y: np.ndarray, folds) -> np.ndarray:
    pred = np.empty(len(y))
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        if variant == "theme_past_k5":
            pred[f] = y[tr].mean() + history_features(data, tr, y[tr] - y[tr].mean(), f)[:, 0]
        else:
            kind, _, k = variant.split("_")
            pred[f] = knn_mean(dist[kind], y, tr, f, int(k[1:]))
    return pred


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text", action="store_true")
    args = ap.parse_args(argv)
    df, data, _, export = T.load()
    dist = text_distances(df, export) if args.text else distances(data)
    VARIANTS = TEXT_VARIANTS if args.text else globals()["VARIANTS"]
    Y = df[list(TARGETS)].to_numpy(dtype=float)
    print(f"{'target':12}{'variant':16}{'Spearman':>10}{'R2':>8}")
    for j, t in enumerate(TARGETS):
        for v in VARIANTS:
            rho, r2 = [], []
            for rep in range(REPEATS):
                folds = cv.kfold(len(Y), cv.K, np.random.default_rng(cv.SEED + rep))
                p = oof(v, data, dist, Y[:, j], folds)
                base = oof_mean(Y[:, j], folds)
                rho.append(spearman(p, Y[:, j]))
                r2.append(1 - ((p - Y[:, j]) ** 2).mean() / ((base - Y[:, j]) ** 2).mean())
            T.append_log({"neighbour_variant": v, "target": t, "spearman": float(np.mean(rho)), "r2": float(np.mean(r2))})
            print(f"{t:12}{v:16}{np.mean(rho):>10.3f}{np.mean(r2):>8.3f}")
    folds = cv.kfold(len(Y), cv.K, np.random.default_rng(cv.SEED))
    stat = lambda Ym: np.array([[spearman(oof(v, data, dist, Ym[:, j], folds), Ym[:, j]) for v in VARIANTS]
                                for j in range(len(TARGETS))])
    obs = stat(Y)
    rng = np.random.default_rng(cv.SEED + 1)
    null = np.nan_to_num(np.array([stat(Y[rng.permutation(len(Y))]) for _ in range(N_PERM)]), nan=0.0)
    p = (1 + (null >= obs[None]).sum(axis=0)) / (1 + N_PERM)
    p_adj = (1 + (null.max(axis=(1, 2))[:, None, None] >= obs[None]).sum(axis=0)) / (1 + N_PERM)
    print(f"\npermutation, {N_PERM} shuffles, adjusted over {obs.size} pairs")
    print(f"{'target':12}{'variant':16}{'Sp(1x10)':>10}{'p':>8}{'p_adj':>8}")
    for j, t in enumerate(TARGETS):
        for i, v in enumerate(VARIANTS):
            print(f"{t:12}{v:16}{obs[j, i]:>10.3f}{p[j, i]:>8.3f}{p_adj[j, i]:>8.3f}")
    T.append_log({"neighbour_permutation": [{"target": t, "variant": v, "spearman_1x10": float(obs[j, i]),
                                             "p": float(p[j, i]), "p_adjusted": float(p_adj[j, i])}
                                            for j, t in enumerate(TARGETS) for i, v in enumerate(VARIANTS)]})
    return 0


def oof_mean(y: np.ndarray, folds) -> np.ndarray:
    pred = np.empty(len(y))
    for f in folds:
        pred[f] = np.delete(y, f).mean()
    return pred


if __name__ == "__main__":
    raise SystemExit(main())
