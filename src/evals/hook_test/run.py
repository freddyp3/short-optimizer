"""Run the hook test against Gemini: --split dev|gate|heldout, --runs K, resumable.

Each (blind_id, run k) writes ``data/evals/hook_test/raw/<run_id>/<blind_id>__r<k>.json``.
Rerunning with the same ``--run-id`` skips files that already exist (daily-quota
resume). Video order is shuffled per run with seed = SEED + k. Default model is
gemini-3.8-flash on the free tier; the $30 budget guard applies to nominal cost
only when GEMINI_TIER != free.

    ./venv/bin/python -m src.evals.hook_test.run --split dev --runs 1 [--dry-run]
    ./venv/bin/python -m src.evals.hook_test.run --split gate --runs 3 --run-id 20261030T000000Z_abc1234
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.evals.hook_test import config as C
from src.evals.hook_test.gemini_client import (
    DailyQuotaExceeded, GeminiHookTester, estimate_call_cost, sha256_file)

log = logging.getLogger(__name__)


def default_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        sha = "nogit"
    return f"{ts}_{sha}"


def raw_path(run_dir: Path, blind_id: str, k: int) -> Path:
    return run_dir / f"{blind_id}__r{k}.json"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", required=True, choices=["dev", "gate", "heldout"])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--model", default=C.DEFAULT_MODEL)
    p.add_argument("--fps", type=float, default=C.FPS)
    p.add_argument("--resolution", default=C.MEDIA_RESOLUTION)
    p.add_argument("--prompt", type=Path, default=C.DEFAULT_PROMPT_FILE)
    p.add_argument("--run-id", default=None)
    p.add_argument("--rpm", type=float, default=0.0, help="sleep 60/rpm s between calls")
    p.add_argument("--dry-run", action="store_true",
                   help="everything except upload/generate")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    prompt_text = args.prompt.read_text()
    prompt_sha = sha256_file(args.prompt)
    prompt_version = args.prompt.stem

    sel = pd.read_csv(C.SELECTION_CSV_PATH)
    vids = sel[sel["split"] == args.split].reset_index(drop=True)
    if not len(vids):
        log.error("no videos for split %s in %s", args.split, C.SELECTION_CSV_PATH)
        return 1

    run_id = args.run_id or default_run_id()
    run_dir = C.RAW_DIR / run_id
    settings = {
        "run_id": run_id, "split": args.split, "runs": args.runs,
        "model": args.model, "fps": args.fps, "media_resolution": args.resolution,
        "prompt_file": str(args.prompt), "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha, "tier": os.environ.get("GEMINI_TIER", C.GEMINI_TIER),
        "seed_base": C.SEED,
    }

    # dry-run / preflight listing
    missing = []
    total_est = 0.0
    for _, v in vids.iterrows():
        blind = C.BLIND_DIR / f"{v['blind_id']}.mp4"
        ok = blind.exists()
        if not ok:
            missing.append(str(blind))
        est = estimate_call_cost(args.model, float(v["duration_seconds"]), args.fps)
        total_est += est * args.runs
        log.info("  %-12s %-9s %-36s dur=%5.1fs %s est=$%.3f/call",
                 v["blind_id"], v["label"], blind, v["duration_seconds"],
                 "OK" if ok else "MISSING", est)
    log.info("split=%s n=%d runs=%d -> %d calls; nominal cost ~$%.2f (%s tier)",
             args.split, len(vids), args.runs, len(vids) * args.runs,
             total_est, settings["tier"])
    log.info("prompt %s sha256=%s", args.prompt.name, prompt_sha[:16])
    if missing:
        log.error("%d blind files missing (run fetch_videos.py first)", len(missing))
        return 1
    if args.dry_run:
        log.info("dry run: settings %s", json.dumps(settings, indent=None))
        return 0

    tier = settings["tier"]
    if tier != "free" and total_est > C.BUDGET_CAP_USD:
        log.error("estimated nominal cost $%.2f exceeds $%.0f cap; aborting",
                  total_est, C.BUDGET_CAP_USD)
        return 1

    tester = GeminiHookTester(api_key=os.environ.get("GEMINI_API_KEY"),
                              model=args.model, tier=tier)
    return run_calls(tester, vids, settings, prompt_text, run_dir,
                     runs=args.runs, rpm=args.rpm)


def run_calls(tester, vids: pd.DataFrame, settings: dict, prompt_text: str,
              run_dir: Path, runs: int, rpm: float = 0.0,
              blind_dir: Path = C.BLIND_DIR) -> int:
    """The (video, run) loop: resumable via existing raw files, budget-guarded."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    tier, model = settings["tier"], settings["model"]
    run_id, split = settings["run_id"], settings["split"]
    fps, res = settings["fps"], settings["media_resolution"]

    spent = 0.0
    done = skipped = 0
    inter_call_sleep = 60.0 / rpm if rpm > 0 else 0.0
    try:
        for k in range(1, runs + 1):
            order = vids.sample(frac=1.0, random_state=C.SEED + k)
            for _, v in order.iterrows():
                out = raw_path(run_dir, v["blind_id"], k)
                if out.exists():
                    skipped += 1
                    continue
                if tier != "free" and spent + estimate_call_cost(
                        model, float(v["duration_seconds"]), fps) > C.BUDGET_CAP_USD:
                    log.error("budget cap would be exceeded (spent $%.2f); aborting", spent)
                    return 1
                rec = tester.analyze(
                    v["blind_id"], blind_dir / f"{v['blind_id']}.mp4",
                    prompt_text, float(v["duration_seconds"]),
                    fps=fps, media_resolution=res,
                    prompt_version=settings["prompt_version"],
                    prompt_sha256=settings["prompt_sha256"])
                rec.update({"run_id": run_id, "run_k": k, "split": split,
                            "video_id": v["video_id"]})
                out.write_text(json.dumps(rec, indent=2))
                spent += rec.get("cost_usd_paid_equiv", 0)
                done += 1
                log.info("%s r%d -> %s (%d tokens, $%.4f nominal)",
                         v["blind_id"], k, rec["status"],
                         rec.get("usage", {}).get("total_tokens", 0),
                         rec.get("cost_usd_paid_equiv", 0))
                if inter_call_sleep:
                    time.sleep(inter_call_sleep)
    except DailyQuotaExceeded as e:
        log.error("%s", e)
        log.error("wrote %d new + %d existing raw files under %s; "
                  "daily quota hit, rerun the same command with --run-id %s to resume",
                  done, skipped, run_dir, run_id)
        return 2

    log.info("done: %d new + %d resumed calls; nominal total $%.2f; raw -> %s",
             done, skipped, spent, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
