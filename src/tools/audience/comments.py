"""Own-channel comments as evidence for the audience reaction (plan/09 step R1).

    ./venv/bin/python -m src.tools.audience.comments          # the free E1 test

Top-level comments only; the creator's own comments are dropped; author names and ids never leave this module.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C

ENCODER = "sentence-transformers/all-mpnet-base-v2"
EMB_NPZ = Path("data/models/judge/comment_embeddings.npz")
MIN_COMMENTS = 5
K = 5
MENTION = re.compile(r"@[\w.\-]+")
N_PERM = 5000


def export_dir() -> Path:
    return Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"])


def clean_comments(comments: pd.DataFrame, own_channel_id: str) -> pd.DataFrame:
    """video_id, text, like_count for top-level comments by viewers, most liked first; @mentions removed."""
    c = comments[~comments["is_reply"].astype(bool) & (comments["author_channel_id"] != own_channel_id)]
    text = c["text_original"].astype(str).map(lambda t: MENTION.sub("", t).strip())
    out = pd.DataFrame({"video_id": c["video_id"].to_numpy(), "text": text.to_numpy(),
                        "like_count": c["like_count"].fillna(0).astype(int).to_numpy()})
    out = out[out["text"].str.len() > 0].drop_duplicates(["video_id", "text"])
    return out.sort_values(["video_id", "like_count"], ascending=[True, False], kind="stable").reset_index(drop=True)


def load_comments() -> pd.DataFrame:
    d = export_dir()
    return clean_comments(pd.read_csv(d / "comments.csv"), pd.read_csv(d / "channel.csv")["id"].iloc[0])


def top_comments(comments: pd.DataFrame, video_id: str, n: int = 8, max_chars: int = 200) -> list[dict]:
    c = comments[comments["video_id"] == video_id].head(n)
    return [{"text": t[:max_chars], "likes": int(l)} for t, l in zip(c["text"], c["like_count"])]


def similar_past_shorts(lib, i: int, comments: pd.DataFrame, k: int = K, n_comments: int = 8) -> list[dict]:
    """The k past Shorts whose one-second timelines are most like video ``i`` (never ``i`` itself), each with its
    real outcomes and real top comments. No titles, names or ids of commenters."""
    o = lib.others(i)
    order = o[np.argsort(-lib.sim_e[i, o], kind="stable")][:k]
    out = []
    for j in order:
        row = lib.df.iloc[int(j)]
        out.append({"video_id": row.video_id, "similarity": round(float(lib.sim_e[i, j]), 2), "duration_s": round(float(row.duration_s), 1),
                    "viewers_per_100_at_10_25_50_75_95_pct": [round(float(x)) for x in lib.points[j]],
                    "rewatch_level": lib.rewatch[j], "like_rate_tier": lib.likes[j],
                    "top_comments": top_comments(comments, row.video_id, n_comments)})
    return out


def centroids(comments: pd.DataFrame, ids: list[str], path: Path = EMB_NPZ) -> tuple[np.ndarray, np.ndarray]:
    """(n, d) mean comment embedding per video (unit length) and the number of comments used."""
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if z["ids"].tolist() == ids:
            return z["cent"], z["n"]
    from sentence_transformers import SentenceTransformer

    c = comments[comments["video_id"].isin(set(ids))]
    E = SentenceTransformer(ENCODER).encode(c["text"].tolist(), normalize_embeddings=True, batch_size=128)
    pos = {v: np.flatnonzero(c["video_id"].to_numpy() == v) for v in ids}
    cent = np.vstack([E[pos[v]].mean(axis=0) if len(pos[v]) else np.zeros(E.shape[1]) for v in ids])
    norm = np.linalg.norm(cent, axis=1, keepdims=True)
    cent = cent / np.where(norm > 0, norm, 1)
    n = np.array([len(pos[v]) for v in ids])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), cent=cent, n=n)
    return cent, n


def neighbour_test(sim: np.ndarray, cent: np.ndarray, ok: np.ndarray, k: int = K, seed: int = 340) -> dict:
    """For each eligible video: similarity of its comment centroid to the mean centroid of its k timeline neighbours,
    against k random eligible videos. Leave-one-out; one-sided sign-flip test on the paired differences."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    near, rand = [], []
    for i in idx:
        o = idx[idx != i]
        nb = o[np.argsort(-sim[i, o], kind="stable")][:k]
        unit = lambda m: m / np.linalg.norm(m)
        near.append(float(cent[i] @ unit(cent[nb].mean(axis=0))))
        rand.append(float(np.mean([cent[i] @ unit(cent[rng.choice(o, k, replace=False)].mean(axis=0)) for _ in range(20)])))
    d = np.array(near) - np.array(rand)
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    return {"n": int(len(idx)), "neighbours": float(np.mean(near)), "random": float(np.mean(rand)), "difference": float(d.mean()),
            "share_of_videos_closer": float(np.mean(d > 0)), "p": float((1 + np.sum((flips * d).mean(axis=1) >= d.mean())) / (1 + N_PERM))}


def main() -> int:
    from src.tools.predict_retention.tool import state

    st = state()
    lib, comments = st.lib, load_comments()
    ids = lib.df["video_id"].tolist()
    per = comments.groupby("video_id").size().reindex(ids, fill_value=0)
    print(f"viewer top-level comments: {len(comments)} on {comments['video_id'].nunique()} videos; library videos with >= {MIN_COMMENTS}: {(per >= MIN_COMMENTS).sum()} of {len(ids)}")
    cent, n = centroids(comments, ids)
    ok = n >= MIN_COMMENTS
    X = np.nan_to_num(lib.df[["t_premise_clear_s", "payoff_count_all", "main_payoff_frac", "tail_all", "dead_time_s", "arc_ord",
                              "first_last_dhash_dist", "cuts_per_s"]].to_numpy(dtype=float))
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    struct = -np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    for name, sim in (("timeline embedding (what the video is)", lib.sim_e), ("structure (how it is built)", struct)):
        print(f"{name}: {neighbour_test(sim, cent, ok)}")
    i = int(np.flatnonzero(ok)[0])
    print(json.dumps(similar_past_shorts(lib, i, comments, k=2, n_comments=3), indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
