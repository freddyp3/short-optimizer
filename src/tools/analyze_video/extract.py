"""Gemini feature-extraction runner for ``analyze_video`` (plan/03a section 6 step 1).

Runs the current feature checklist prompt on DEV videos with the PAID key
(``GEMINI_API_KEY_PAID``).
Only ``--split dev`` is allowed: gate/heldout stay untouched until the checklist is frozen.
Each video writes ``data/features/gemini/raw/<run_id>/<blind_id>__r1.json``; rerunning with
the same ``--run-id`` skips files that already exist. Budget guard: $3 nominal cap.

    ./venv/bin/python -m src.tools.analyze_video.extract --split dev --n 10 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.evals.hook_test import config as C
from src.evals.hook_test.gemini_client import (
    DailyQuotaExceeded, GeminiHookTester, estimate_call_cost, sha256_file)
from src.evals.hook_test.run import default_run_id
from src.tools.analyze_video.schema import (
    PROMPT_VERSION, VideoFeatures, validate_features)

log = logging.getLogger(__name__)

OUT_DIR = Path("data/features/gemini")
RAW_DIR = OUT_DIR / "raw"
UPLOADS_JSON = OUT_DIR / "uploads_paid.json"  # Files API uploads are per project
PROMPT_FILE = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md"
BUDGET_CAP_USD = 3.0


class GeminiFeatureExtractor(GeminiHookTester):
    """Hook-test client with the feature schema/validator swapped in."""

    def __init__(self, client=None, api_key=None, **kw):
        kw.setdefault("tier", "paid")
        kw.setdefault("uploads_path", UPLOADS_JSON)
        if client is None and not api_key:
            api_key = os.environ.get("GEMINI_API_KEY_PAID")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY_PAID not set; the feature extractor runs on the "
                    "paid project only and never falls back to GEMINI_API_KEY")
        super().__init__(client=client, api_key=api_key, **kw)

    def _schema(self, schema_version: str):
        return VideoFeatures

    def _validate(self, data, duration_s: float, schema_version: str = "v1"):
        return validate_features(data, duration_s)

    def analyze(self, *args, **kwargs) -> dict:
        rec = super().analyze(*args, **kwargs)
        rec["schema_version"] = PROMPT_VERSION
        return rec


def pick_dev_videos(n: int, seed: int = C.SEED) -> pd.DataFrame:
    """Stratified sample from the dev split: n/2 strong + n/2 weak."""
    sel = pd.read_csv(C.SELECTION_CSV_PATH)
    dev = sel[sel["split"] == "dev"]
    parts = []
    for label, k in (("strong", (n + 1) // 2), ("weak", n // 2)):
        pool = dev[dev["label"] == label]
        if k > len(pool):
            raise SystemExit(f"only {len(pool)} {label} dev videos for n={n}")
        parts.append(pool.sample(n=k, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def summarize(run_dir: Path) -> float:
    """Print the per-video summary table from the raw files; returns total cost."""
    recs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("*__r1.json"))]
    total = 0.0
    print(f"\n{'blind_id':<11} {'status':<8} {'invalid_reason':<28} "
          f"{'quality_flags':<24} {'uncertain':<20} {'po':<3} {'cost':>7} {'lat_s':>6}")
    for r in recs:
        res = r.get("result") or {}
        cost = r.get("cost_usd_paid_equiv", 0.0)
        total += cost
        print(f"{r['blind_id']:<11} {r['status']:<8} "
              f"{str(r.get('invalid_reason')):<28} "
              f"{','.join(r.get('quality_flags') or []):<24} "
              f"{','.join(res.get('uncertain_fields') or []):<20} "
              f"{len(res.get('payoffs') or []):<3} "
              f"{cost:>7.4f} {r.get('latency_s', 0):>6.1f}")
    print(f"\ntotal nominal cost: ${total:.3f}  ({len(recs)} videos, {run_dir})")
    return total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", required=True, choices=["dev"],
                   help="only dev is allowed before the checklist is frozen")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--model", default=C.DEFAULT_MODEL)
    p.add_argument("--fps", type=float, default=C.FPS)
    p.add_argument("--resolution", default=C.MEDIA_RESOLUTION)
    p.add_argument("--prompt", type=Path, default=PROMPT_FILE)
    p.add_argument("--run-id", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="everything except upload/generate")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    template = args.prompt.read_text()
    prompt_sha = sha256_file(args.prompt)
    vids = pick_dev_videos(args.n)

    run_id = args.run_id or default_run_id()
    run_dir = RAW_DIR / run_id
    settings = {
        "run_id": run_id, "split": args.split, "n": args.n,
        "model": args.model, "fps": args.fps, "media_resolution": args.resolution,
        "prompt_file": str(args.prompt), "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha, "tier": "paid", "seed": C.SEED,
    }

    total_est = 0.0
    for _, v in vids.iterrows():
        blind = C.BLIND_DIR / f"{v['blind_id']}.mp4"
        est = estimate_call_cost(args.model, float(v["duration_seconds"]), args.fps)
        total_est += est
        log.info("  %-12s %-6s %-36s dur=%5.1fs %s est=$%.3f",
                 v["blind_id"], v["label"], blind, v["duration_seconds"],
                 "OK" if blind.exists() else "MISSING", est)
    log.info("split=%s n=%d; nominal cost ~$%.2f (paid tier)",
             args.split, len(vids), total_est)
    if args.dry_run:
        log.info("dry run: settings %s", json.dumps(settings, indent=None))
        return 0
    if total_est > BUDGET_CAP_USD:
        log.error("estimated nominal cost $%.2f exceeds $%.0f cap; aborting",
                  total_est, BUDGET_CAP_USD)
        return 1

    extractor = GeminiFeatureExtractor(model=args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    spent, done, skipped = 0.0, 0, 0
    try:
        for _, v in vids.iterrows():
            out = run_dir / f"{v['blind_id']}__r1.json"
            if out.exists():
                skipped += 1
                continue
            dur = float(v["duration_seconds"])
            if spent + estimate_call_cost(args.model, dur, args.fps) > BUDGET_CAP_USD:
                log.error("budget cap would be exceeded (spent $%.2f); aborting", spent)
                return 1
            rec = extractor.analyze(
                v["blind_id"], C.BLIND_DIR / f"{v['blind_id']}.mp4",
                template.format(duration_s=dur), dur,
                fps=args.fps, media_resolution=args.resolution,
                prompt_version=PROMPT_VERSION, prompt_sha256=prompt_sha)
            rec.update({"run_id": run_id, "split": args.split,
                        "video_id": v["video_id"]})
            out.write_text(json.dumps(rec, indent=2))
            spent += rec.get("cost_usd_paid_equiv", 0)
            done += 1
            log.info("%s -> %s (%d tokens, $%.4f)",
                     v["blind_id"], rec["status"],
                     rec.get("usage", {}).get("total_tokens", 0),
                     rec.get("cost_usd_paid_equiv", 0))
    except DailyQuotaExceeded as e:
        log.error("%s", e)
        log.error("wrote %d new + %d existing raw files; rerun with --run-id %s to resume",
                  done, skipped, run_id)
        return 2

    log.info("done: %d new + %d resumed calls; nominal total $%.2f", done, skipped, spent)
    summarize(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
