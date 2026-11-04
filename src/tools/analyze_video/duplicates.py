"""Duplicate / near-duplicate scan of the whole eligible pool (plan/03b "Duplicates").

Hashes every video's frames at 1 fps (gray 72x128, dhash), scores candidate pairs by
the best median hamming over -2..+2 s offsets, and writes ``duplicates.csv`` plus
``duplicates_exclude.json`` (the fixed, outcome-blind exclusion rule). Pair contact
sheets for the visual check go to ``data/features/duplicate_frames/``.

    ./venv/bin/python -m src.tools.analyze_video.duplicates [--workers 2]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test.fetch_videos import extract_frame, label_strip
from src.evals.hook_test.fetch_videos import FRAME_W, FRAME_H
from src.models.pool import labelled_pool
from src.tools.analyze_video.measure import (
    DEFAULT_PARAMS, MeasureParams, dhash, hamming)

log = logging.getLogger(__name__)

VIDEO_DIR = Path("data/videos")
HASH_CACHE = Path("data/features/frame_hashes.json")
OUT_CSV = Path("data/features/duplicates.csv")
OUT_JSON = Path("data/features/duplicates_exclude.json")
PAIR_FRAMES_DIR = Path("data/features/duplicate_frames")

SAFE_COLS = ["video_id", "split", "duration_seconds", "published_at"]

HASH_FPS = 1.0
DUR_DIFF_MAX_S = 2.0
FIRST_FRAME_HAMM = 10
OFFSETS_S = (-2, -1, 0, 1, 2)
MATCH_HAMM = 10          # an aligned frame counts as matching below this
DUP_MED, DUP_FRAC = 8, 0.8
NEAR_MED, NEAR_FRAC = 16, 0.5


# ----------------------------------------------------------------- hashing

def frame_hashes(path: Path, p: MeasureParams = DEFAULT_PARAMS) -> list[int]:
    """dhash per frame, decoded at HASH_FPS gray 72x128."""
    cp = subprocess.run(
        ["ffmpeg", "-nostdin", "-nostats", "-hide_banner", "-loglevel", "error",
         "-i", str(path), "-an",
         "-vf", f"fps={HASH_FPS},scale={p.frame_w}:{p.frame_h}:flags=area,format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=120)
    if cp.returncode != 0:
        raise RuntimeError(f"decode failed {path.name}: {cp.stderr.decode(errors='replace')[-200:]}")
    buf = np.frombuffer(cp.stdout, dtype=np.uint8)
    n = len(buf) // (p.frame_w * p.frame_h)
    frames = buf[: n * p.frame_w * p.frame_h].reshape(n, p.frame_h, p.frame_w)
    return [dhash(f) for f in frames]


def hash_pool(ids: list[str], workers: int = 2,
              cache_path: Path = HASH_CACHE) -> dict[str, list[int]]:
    """Hashes for every id, cached to disk (reruns are cheap)."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [v for v in ids if v not in cache and (VIDEO_DIR / f"{v}.mp4").exists()]
    with ThreadPoolExecutor(max_workers=min(workers, 2)) as ex:
        for i, (v, h) in enumerate(
                zip(missing, ex.map(lambda v: frame_hashes(VIDEO_DIR / f"{v}.mp4"), missing)), 1):
            cache[v] = h
            if i % 50 == 0 or i == len(missing):
                log.info("hashed %d/%d new", i, len(missing))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return {v: cache[v] for v in ids if v in cache}


# ----------------------------------------------------------------- scoring

def score_pair(ha: list[int], hb: list[int],
               offsets: tuple = OFFSETS_S) -> dict | None:
    """Best offset by lowest median hamming; ties -> smallest |offset|."""
    best = None
    for off in offsets:
        pairs = [(a, hb[i + off]) for i, a in enumerate(ha)
                 if 0 <= i + off < len(hb)]
        if not pairs:
            continue
        hs = [hamming(a, b) for a, b in pairs]
        med = float(np.median(hs))
        frac = float(np.mean([h <= MATCH_HAMM for h in hs]))
        cand = {"best_offset_s": off, "median_hamming": med,
                "match_frac": round(frac, 3), "n_aligned": len(hs)}
        if best is None or (med, abs(off), off) < (
                best["median_hamming"], abs(best["best_offset_s"]), best["best_offset_s"]):
            best = cand
    return best


def verdict(score: dict) -> str | None:
    if score["median_hamming"] <= DUP_MED and score["match_frac"] >= DUP_FRAC:
        return "duplicate"
    if score["median_hamming"] <= NEAR_MED or score["match_frac"] >= NEAR_FRAC:
        return "near"
    return None


# ----------------------------------------------------------------- exclusion

def groups_from_pairs(pairs: list[tuple[str, str]]) -> list[set[str]]:
    """Connected components over duplicate pairs."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for a, b in pairs:
        for v in (a, b):
            groups.setdefault(find(v), set()).add(v)
    return list(groups.values())


def exclusion_plan(dup_pairs: list[tuple[str, str]],
                   info: dict[str, dict]) -> dict:
    """Fixed rule: never exclude eval-split videos; a train member of a group that
    touches an eval split is excluded; a train-only group keeps the earliest
    published_at. Groups with 2+ eval members are reported only."""
    exclude, groups, eval_dups = set(), [], []
    for g in groups_from_pairs(dup_pairs):
        members = sorted(g)
        ev = [v for v in members if info[v]["split"] != "train"]
        tr = [v for v in members if info[v]["split"] == "train"]
        rec = {"members": members, "eval_members": ev}
        if len(ev) >= 2:
            rec["note"] = "eval-eval duplicate: reported only, never excluded"
            eval_dups.append(members)
        if ev:
            rec["excluded"] = tr
            exclude.update(tr)
        else:
            keep = min(tr, key=lambda v: info[v]["published_at"])
            rec["keep"] = keep
            rec["excluded"] = [v for v in tr if v != keep]
            exclude.update(rec["excluded"])
        groups.append(rec)
    return {
        "rule": ("connected components of 'duplicate' pairs; never exclude dev/gate/"
                 "heldout; group touching an eval split drops its train members; "
                 "train-only group keeps earliest published_at"),
        "exclude": sorted(exclude), "groups": groups,
        "eval_duplicates": eval_dups,
    }


# ----------------------------------------------------------------- contact sheets

def pair_sheet(a: dict, b: dict, v: str, out_dir: Path, scratch: Path) -> Path:
    """3 frames of each video at 20/50/80 % duration, stacked, labelled."""
    scratch.mkdir(parents=True, exist_ok=True)
    imgs = []
    for row, m in enumerate((a, b)):
        frames = []
        for i, frac in enumerate((0.2, 0.5, 0.8)):
            fr = scratch / f"_{a['video_id']}_{b['video_id']}_{row}{i}.jpg"
            extract_frame(VIDEO_DIR / f"{m['video_id']}.mp4",
                          max(0.5, m["duration_seconds"] * frac), fr)
            frames.append(fr)
        lab = scratch / f"_{a['video_id']}_{b['video_id']}_{row}lab.png"
        lab.write_bytes(label_strip(m["video_id"], FRAME_W * 3))
        strip = scratch / f"_{a['video_id']}_{b['video_id']}_{row}strip.jpg"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(frames[0]), "-i", str(frames[1]),
             "-i", str(frames[2]), "-i", str(lab),
             "-filter_complex",
             (";".join(f"[{i}]scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease,"
                       f"pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2:black[{x}]"
                       for i, x in enumerate("abc"))
              + ";[a][b][c]hstack=3[frames];[3][frames]vstack=2"),
             str(strip)], capture_output=True, check=True)
        imgs.append(strip)
    lab2 = scratch / f"_{a['video_id']}_{b['video_id']}_title.png"
    lab2.write_bytes(label_strip(f"{a['video_id']} vs {b['video_id']}: {v}", FRAME_W * 3))
    out = out_dir / f"{a['video_id']}__{b['video_id']}.jpg"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(imgs[0]), "-i", str(imgs[1]), "-i", str(lab2),
         "-filter_complex", "vstack=inputs=3", "-frames:v", "1", str(out)],
        capture_output=True, check=True)
    return out


# ----------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pool = labelled_pool()[SAFE_COLS].copy()     # outcome columns dropped immediately
    log.info("pool: %d eligible videos", len(pool))
    info = {r.video_id: {"video_id": r.video_id, "split": r.split,
                         "published_at": r.published_at,
                         "duration_seconds": r.duration_seconds}
            for r in pool.itertuples()}
    hashes = hash_pool(pool["video_id"].tolist(), workers=args.workers)

    rows = []
    for a, b in combinations(pool["video_id"], 2):
        if a not in hashes or b not in hashes:
            continue
        ra, rb = info[a], info[b]
        if not (abs(ra["duration_seconds"] - rb["duration_seconds"]) <= DUR_DIFF_MAX_S
                or hamming(hashes[a][0], hashes[b][0]) <= FIRST_FRAME_HAMM):
            continue
        s = score_pair(hashes[a], hashes[b])
        if s is None:
            continue
        v = verdict(s)
        if v is None:
            continue
        rows.append({"video_id_a": a, "split_a": ra["split"], "published_a": ra["published_at"],
                     "video_id_b": b, "split_b": rb["split"], "published_b": rb["published_at"],
                     "dur_a": ra["duration_seconds"], "dur_b": rb["duration_seconds"],
                     **s, "verdict": v})
    df = pd.DataFrame(rows, columns=["video_id_a", "split_a", "published_a", "video_id_b",
                                     "split_b", "published_b", "dur_a", "dur_b",
                                     "best_offset_s", "median_hamming", "match_frac",
                                     "n_aligned", "verdict"])
    df.to_csv(OUT_CSV, index=False)
    log.info("%d candidate rows (duplicate+near) -> %s", len(df), OUT_CSV)

    dup_pairs = [(r.video_id_a, r.video_id_b) for r in df.itertuples()
                 if r.verdict == "duplicate"]
    plan = exclusion_plan(dup_pairs, info)
    OUT_JSON.write_text(json.dumps(plan, indent=2))
    log.info("exclusions: %d video_ids; eval-eval groups: %d -> %s",
             len(plan["exclude"]), len(plan["eval_duplicates"]), OUT_JSON)

    PAIR_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    scratch = PAIR_FRAMES_DIR / "_scratch"
    for r in df.itertuples():
        p = pair_sheet(info[r.video_id_a], info[r.video_id_b], r.verdict,
                       PAIR_FRAMES_DIR, scratch)
        log.info("pair sheet: %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
