"""Consistency A/B selection + blinding (plan/03b section 3, step 5). $0, no Gemini.

Picks 30 train-pool videos stratified into duration thirds (10 each, seed 340), gives
them fresh blind ids (seed 1340), makes metadata-stripped copies under
``data/features/blind/``, verifies no title/comment/creation_time tags survive,
ffprobes each blinded file, writes ``consistency_selection.csv`` + ``.json``, renders
contact sheets into ``data/features/blind_frames/`` and scene cuts into
``data/features/cuts/``. Deterministic: rerunning reproduces the same selection.

Outcome columns never enter this module: ``train_rows()`` is projected down to
SAFE_COLS the moment it returns, and nothing else is read, printed or written.

    ./venv/bin/python -m src.tools.analyze_video.consistency_select
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.fetch_videos import (
    blind_video, contact_sheets, format_tags)
from src.evals.hook_test.select import blind_ids
from src.models.pool import train_rows
from src.tools.analyze_video.cuts import write_cut_file
from src.tools.analyze_video.measure import probe

log = logging.getLogger(__name__)

SAFE_COLS = ["video_id", "split", "duration_seconds", "published_at"]
SAMPLE_SEED = C.SEED          # 340
BLIND_SEED = 1340             # different space from the hook test's blind ids
N_PER_BAND = 10
BANDS = ("short", "middle", "long")

VIDEO_DIR = Path("data/videos")
BLIND_DIR = Path("data/features/blind")
FRAMES_DIR = Path("data/features/blind_frames")
OUT_CSV = Path("data/features/gemini/consistency_selection.csv")
OUT_JSON = OUT_CSV.with_suffix(".json")
LEAKY_TAGS = {"title", "comment", "creation_time"}


def load_train_safe() -> pd.DataFrame:
    """Train rows projected to SAFE_COLS; asserts disjointness from the hook-test selection."""
    df = train_rows()[SAFE_COLS].copy()
    assert (df["split"] == "train").all()
    hook = pd.read_csv(C.SELECTION_CSV_PATH)
    overlap = set(df["video_id"]) & set(hook["video_id"])
    assert not overlap, f"train rows overlap the hook-test selection: {sorted(overlap)}"
    return df


def assign_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Duration tertiles on rank(method='first') so the split is exact thirds."""
    df = df.copy()
    df["band"] = pd.qcut(df["duration_seconds"].rank(method="first"), 3,
                         labels=list(BANDS))
    return df


def select_videos(df: pd.DataFrame, n_per_band: int = N_PER_BAND,
                  seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """10 videos per band, seeded; reproducible."""
    df = assign_bands(df)
    parts = [df[df["band"] == b].sample(n=n_per_band, random_state=seed)
             for b in BANDS]
    return pd.concat(parts).reset_index(drop=True)


def band_edges(df: pd.DataFrame) -> dict:
    """(min, max) duration_seconds per assigned band."""
    return {b: [float(df[df["band"] == b]["duration_seconds"].min()),
                float(df[df["band"] == b]["duration_seconds"].max())]
            for b in BANDS}


def check_blind_collisions(bids: dict, existing: set) -> None:
    hit = set(bids.values()) & existing
    if hit:
        raise SystemExit(f"blind_id collision with hook-test selection: {sorted(hit)}; "
                         "do not silently re-seed")


def leaked_tags(tags: dict) -> dict:
    return {k: v for k, v in tags.items() if k.lower() in LEAKY_TAGS}


BLIND_SEED_ALL = 2340
TRAIN_CSV = Path("data/features/gemini/train_selection.csv")


def build_train_selection(workers: int = 2) -> int:
    """Full train pool: keep the 30 A/B rows, blind the other 95 (seed 2340)."""
    df = load_train_safe()
    ab = pd.read_csv(OUT_CSV)
    keep = ab[["video_id", "blind_id", "duration_s_export",
               "duration_s_file", "blind_path"]].assign(source="ab30")
    rest = df[~df["video_id"].isin(keep["video_id"])]
    bids = blind_ids(rest["video_id"].tolist(), seed=BLIND_SEED_ALL)
    hook = pd.read_csv(C.SELECTION_CSV_PATH)
    check_blind_collisions(bids, set(ab["blind_id"]) | set(hook["blind_id"]))

    jobs = [(bids[r.video_id], VIDEO_DIR / f"{r.video_id}.mp4",
             BLIND_DIR / f"{bids[r.video_id]}.mp4") for r in rest.itertuples()]
    missing = [s for _, s, _ in jobs if not s.exists()]
    if missing:
        log.error("%d source videos missing", len(missing))
        return 1
    # rebuild any blind file whose id was remapped since a previous selection file
    prev_map = {}
    if TRAIN_CSV.exists():
        prev = pd.read_csv(TRAIN_CSV)
        prev_map = dict(zip(prev["blind_id"], prev["video_id"]))
    for bid, src, dst in jobs:
        if dst.exists() and prev_map.get(bid) not in (None, src.stem):
            dst.unlink()
    with ThreadPoolExecutor(max_workers=min(workers, 2)) as ex:
        list(ex.map(lambda j: blind_video(j[1], j[2]), jobs))

    leaky, durations = [], {}
    for bid, _, dst in jobs:
        if (leaked := leaked_tags(format_tags(dst))):
            leaky.append((bid, leaked))
        durations[bid] = probe(dst)["duration_s"]
    if leaky:
        log.error("metadata survived blinding: %s", leaky)
        return 1

    new = pd.DataFrame({
        "video_id": rest["video_id"],
        "blind_id": rest["video_id"].map(bids),
        "duration_s_export": rest["duration_seconds"],
        "duration_s_file": rest["video_id"].map(lambda v: durations[bids[v]]),
        "blind_path": rest["video_id"].map(lambda v: str(BLIND_DIR / f"{bids[v]}.mp4")),
        "source": "new",
    })
    out = pd.concat([keep, new], ignore_index=True)
    out.to_csv(TRAIN_CSV, index=False)
    log.info("wrote %s: %d ab30 + %d new = %d", TRAIN_CSV, len(keep), len(new), len(out))

    blind_videos = [(r.blind_id, "train", Path(r.blind_path), r.duration_s_file)
                    for r in new.itertuples()]
    sheets = contact_sheets(sorted(blind_videos), FRAMES_DIR, FRAMES_DIR / "_scratch")
    for s in sheets:
        log.info("contact sheet: %s", s)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--all", action="store_true", dest="all_train",
                   help="full train pool: keep the 30 A/B rows, blind the other 95")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.all_train:
        return build_train_selection(workers=args.workers)

    n_before = len(train_rows(exclude_duplicates=False)[SAFE_COLS])
    df = load_train_safe()
    n_after = len(df)
    log.info("train rows: %d before dedup, %d after (%d excluded)",
             n_before, n_after, n_before - n_after)

    # remember the previous draw: stale blind/cut/frame files are removed, and any
    # blind_id that now maps to a different video must be rebuilt (blind_video only
    # checks mtime — a recycled id would silently keep the old video's content)
    stale_blind_ids, old_map = set(), {}
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        stale_blind_ids = set(prev["blind_id"])
        old_map = dict(zip(prev["blind_id"], prev["video_id"]))

    sel = select_videos(df)
    log.info("selected %d train videos; band edges %s", len(sel), band_edges(assign_bands(df)))

    bids = blind_ids(sel["video_id"].tolist(), seed=BLIND_SEED)
    hook = pd.read_csv(C.SELECTION_CSV_PATH)
    check_blind_collisions(bids, set(hook["blind_id"]))

    # blind copies + tag check + probe (<=2 workers; blind_video is stream-copy, cheap)
    jobs = [(bids[r.video_id], VIDEO_DIR / f"{r.video_id}.mp4",
             BLIND_DIR / f"{bids[r.video_id]}.mp4") for r in sel.itertuples()]
    missing = [s for _, s, _ in jobs if not s.exists()]
    if missing:
        log.error("%d source videos missing under %s (e.g. %s)", len(missing), VIDEO_DIR,
                  missing[0])
        return 1
    rebuilt = 0
    for bid, src, dst in jobs:
        if dst.exists() and old_map.get(bid) != src.stem:
            dst.unlink()
            rebuilt += 1
    if rebuilt:
        log.info("%d blind files rebuilt (blind_id remapped to a different video)", rebuilt)
    with ThreadPoolExecutor(max_workers=min(args.workers, 2)) as ex:
        list(ex.map(lambda j: blind_video(j[1], j[2]), jobs))

    leaky = []
    durations = {}
    for bid, _, dst in jobs:
        leaked = leaked_tags(format_tags(dst))
        if leaked:
            leaky.append((bid, leaked))
        durations[bid] = probe(dst)["duration_s"]
    if leaky:
        log.error("metadata survived blinding: %s", leaky)
        return 1

    sel = sel.assign(blind_id=sel["video_id"].map(bids),
                     duration_s_file=sel["video_id"].map(
                         lambda v: durations[bids[v]]),
                     blind_path=sel["video_id"].map(
                         lambda v: str(BLIND_DIR / f"{bids[v]}.mp4")))
    out = sel.rename(columns={"duration_seconds": "duration_s_export"})[
        ["video_id", "blind_id", "band", "duration_s_export",
         "duration_s_file", "blind_path"]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        "sample_seed": SAMPLE_SEED, "blind_seed": BLIND_SEED,
        "n_per_band": N_PER_BAND, "band_edges_s": band_edges(assign_bands(df)),
        "counts": out["band"].value_counts().to_dict(),
        "n_train_before": n_before, "n_excluded": n_before - n_after,
        "n_train_after": n_after,
    }, indent=2))

    # remove files from the previous draw that are not part of the new 30
    stale_blind_ids -= set(out["blind_id"])
    for bid in sorted(stale_blind_ids):
        for p in [BLIND_DIR / f"{bid}.mp4", Path("data/features/cuts") / f"{bid}.json",
                  *FRAMES_DIR.glob(f"**/{bid}*")]:
            if p.exists():
                p.unlink()
                log.info("removed stale %s", p)
    log.info("wrote %s and %s", OUT_CSV, OUT_JSON)

    # scene cuts on the blinded files (same file Gemini will see)
    with ThreadPoolExecutor(max_workers=min(args.workers, 2)) as ex:
        for obj in ex.map(lambda b: write_cut_file(b, blind_dir=BLIND_DIR),
                          out["blind_id"].tolist()):
            log.info("cuts %s: %d, %.2f s", obj["blind_id"], len(obj["cuts"]),
                     obj["duration_s"])

    # contact sheets for the leakage eyeball check (split label = band)
    blind_videos = [(r.blind_id, r.band, Path(r.blind_path), r.duration_s_file)
                    for r in out.itertuples()]
    sheets = contact_sheets(sorted(blind_videos), FRAMES_DIR, FRAMES_DIR / "_scratch")
    for s in sheets:
        log.info("contact sheet: %s", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
