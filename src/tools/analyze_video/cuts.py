"""Scene-cut timestamps per blinded file (plan/03b rows 1.5, 1.11).

Writes ``data/features/cuts/<blind_id>.json`` = {blind_id, threshold, duration_s, cuts}
using the same ffmpeg scene threshold as G2 and the ffprobe duration of the blinded
file (the same file Gemini sees). Used by derive.py's payoff snapping.

    ./venv/bin/python -m src.tools.analyze_video.cuts b1,b2,... [--workers 2]
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.evals.hook_test import config as C
from src.tools.analyze_video.measure import (
    MeasureParams, probe, scene_cuts, DEFAULT_PARAMS)

log = logging.getLogger(__name__)

OUT_DIR = Path("data/features/cuts")


def write_cut_file(blind_id: str, blind_dir: Path = C.BLIND_DIR,
                   out_dir: Path = OUT_DIR,
                   p: MeasureParams = DEFAULT_PARAMS) -> dict:
    path = blind_dir / f"{blind_id}.mp4"
    obj = {
        "blind_id": blind_id,
        "threshold": p.scene_threshold,
        "duration_s": probe(path)["duration_s"],
        "cuts": scene_cuts(path, p.scene_threshold),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{blind_id}.json").write_text(json.dumps(obj, indent=2))
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("blind_ids", help="comma-separated blind_ids")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ids = [s for s in args.blind_ids.split(",") if s]
    with ThreadPoolExecutor(max_workers=min(args.workers, 2)) as ex:
        for obj in ex.map(write_cut_file, ids):
            log.info("%s: %d cuts, %.2f s", obj["blind_id"], len(obj["cuts"]),
                     obj["duration_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
