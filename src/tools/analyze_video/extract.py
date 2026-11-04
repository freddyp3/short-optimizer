"""Gemini feature-extraction runner for ``analyze_video`` (plan/03a §6, plan/03b step 6).

Runs a checklist prompt on DEV videos (stratified from the hook-test selection) or on
the 30 train videos of the consistency A/B (``--split train``, reads
``consistency_selection.csv``). Paid key only (``GEMINI_API_KEY_PAID``); gate/heldout
are not selectable. Each (video, run k) writes
``data/features/gemini/raw/<run_id>/<blind_id>__r<k>.json``; rerunning with the same
``--run-id`` skips existing files. Budget guard via ``--budget-usd``.

    ./venv/bin/python -m src.tools.analyze_video.extract --split train \
        --prompt-version v3 --runs 2 [--dry-run] [--budget-usd 6]
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
from src.tools.analyze_video.measure import probe
from src.tools.analyze_video.schema import (
    PROMPT_VERSION, SCHEMAS, render_prompt, validate_features)

log = logging.getLogger(__name__)

OUT_DIR = Path("data/features/gemini")
RAW_DIR = OUT_DIR / "raw"
UPLOADS_JSON = OUT_DIR / "uploads_paid.json"  # Files API uploads are per project
CONSISTENCY_CSV = OUT_DIR / "consistency_selection.csv"
PROMPTS_DIR = Path(__file__).parent / "prompts"


class GeminiFeatureExtractor(GeminiHookTester):
    """Hook-test client with the feature schema/validator swapped in."""

    def __init__(self, client=None, api_key=None, version: str = PROMPT_VERSION, **kw):
        kw.setdefault("tier", "paid")
        kw.setdefault("uploads_path", UPLOADS_JSON)
        if client is None and not api_key:
            api_key = os.environ.get("GEMINI_API_KEY_PAID")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY_PAID not set; the feature extractor runs on the "
                    "paid project only and never falls back to GEMINI_API_KEY")
        super().__init__(client=client, api_key=api_key, **kw)
        self.version = version

    def _thinking_config(self):
        """Pin thinking level to medium (03b row 1.9); hook test keeps its config."""
        from google.genai import types
        return types.ThinkingConfig(
            include_thoughts=True, thinking_level=types.ThinkingLevel.MEDIUM)

    def _schema(self, schema_version: str):
        return SCHEMAS[self.version]

    def _validate(self, data, duration_s: float, schema_version: str = "v1"):
        return validate_features(data, duration_s, self.version)

    def analyze(self, *args, **kwargs) -> dict:
        rec = super().analyze(*args, **kwargs)
        rec["schema_version"] = self.version
        return rec


def blind_duration(path: Path) -> float:
    """ffprobe duration of the blinded file Gemini sees (03b row 1.11)."""
    return float(probe(path)["duration_s"])


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
    vids = pd.concat(parts).reset_index(drop=True)
    vids["band"] = None
    vids["duration_s_export"] = vids["duration_seconds"].astype(float)
    vids["blind_path"] = vids["blind_id"].map(lambda b: str(C.BLIND_DIR / f"{b}.mp4"))
    return vids[["video_id", "blind_id", "band", "duration_s_export", "blind_path"]]


def pick_train_videos() -> pd.DataFrame:
    """All 30 rows of the consistency A/B selection (--n is ignored)."""
    df = pd.read_csv(CONSISTENCY_CSV)
    return df[["video_id", "blind_id", "band", "duration_s_export", "blind_path"]].copy()


def summarize(run_dir: Path) -> float:
    """Per-video summary table + per-run invalid counts; returns total cost."""
    recs = sorted((json.loads(p.read_text()) for p in sorted(run_dir.glob("*__r*.json"))),
                  key=lambda r: (r.get("run_k", 1), r["blind_id"]))
    total = 0.0
    print(f"\n{'blind_id':<11} {'k':<2} {'status':<8} {'invalid_reason':<28} "
          f"{'quality_flags':<24} {'uncertain':<20} {'po':<3} {'cost':>7} {'lat_s':>6}")
    for r in recs:
        res = r.get("result") or {}
        cost = r.get("cost_usd_paid_equiv", 0.0)
        total += cost
        print(f"{r['blind_id']:<11} {r.get('run_k', 1):<2} {r['status']:<8} "
              f"{str(r.get('invalid_reason')):<28} "
              f"{','.join(r.get('quality_flags') or []):<24} "
              f"{','.join(res.get('uncertain_fields') or []):<20} "
              f"{len(res.get('payoffs') or []):<3} "
              f"{cost:>7.4f} {r.get('latency_s', 0):>6.1f}")
    runs = {}
    for r in recs:
        k = r.get("run_k", 1)
        runs.setdefault(k, [0, 0])
        runs[k][1] += 1
        if r["status"] == "invalid":
            runs[k][0] += 1
    for k in sorted(runs):
        print(f"run {k}: {runs[k][0]} invalid / {runs[k][1]}")
    print(f"\ntotal nominal cost: ${total:.3f}  ({len(recs)} records, {run_dir})")
    return total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", required=True, choices=["dev", "train"],
                   help="dev: stratified hook-test dev pick; train: the 30 consistency videos")
    p.add_argument("--n", type=int, default=10, help="dev only; train uses all 30")
    p.add_argument("--prompt-version", default=PROMPT_VERSION,
                   choices=sorted(SCHEMAS), help="prompt + schema version")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--budget-usd", type=float, default=3.0)
    p.add_argument("--model", default=C.DEFAULT_MODEL)
    p.add_argument("--fps", type=float, default=C.FPS)
    p.add_argument("--resolution", default=C.MEDIA_RESOLUTION)
    p.add_argument("--run-id", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="everything except upload/generate")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    prompt_file = PROMPTS_DIR / f"{args.prompt_version}.md"
    template = prompt_file.read_text()
    prompt_sha = sha256_file(prompt_file)
    vids = pick_train_videos() if args.split == "train" else pick_dev_videos(args.n)

    run_id = args.run_id or default_run_id()
    run_dir = RAW_DIR / run_id
    settings = {
        "run_id": run_id, "split": args.split, "n": len(vids), "runs": args.runs,
        "prompt_version": args.prompt_version, "budget_usd": args.budget_usd,
        "model": args.model, "fps": args.fps, "media_resolution": args.resolution,
        "prompt_file": str(prompt_file), "prompt_sha256": prompt_sha,
        "tier": "paid", "seed": C.SEED,
    }

    total_est = 0.0
    for _, v in vids.iterrows():
        blind = Path(v["blind_path"])
        if not blind.exists():
            log.error("blind file missing: %s", blind)
            return 1
        dur = blind_duration(blind)
        est = estimate_call_cost(args.model, dur, args.fps) * args.runs
        total_est += est
        log.info("  %-12s %-6s dur=%5.1fs est=$%.3f x%d",
                 v["blind_id"], v["band"] or "-", dur, est, args.runs)
    log.info("split=%s n=%d runs=%d -> %d calls; nominal cost ~$%.2f",
             args.split, len(vids), args.runs, len(vids) * args.runs, total_est)
    if args.dry_run:
        log.info("dry run: settings %s", json.dumps(settings, indent=None))
        return 0
    if total_est > args.budget_usd:
        log.error("estimated nominal cost $%.2f exceeds $%.2f budget; aborting",
                  total_est, args.budget_usd)
        return 1

    extractor = GeminiFeatureExtractor(model=args.model, version=args.prompt_version)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    spent, done, skipped = 0.0, 0, 0
    try:
        for k in range(1, args.runs + 1):
            order = vids.sample(frac=1.0, random_state=C.SEED + k)
            for _, v in order.iterrows():
                out = run_dir / f"{v['blind_id']}__r{k}.json"
                if out.exists():
                    skipped += 1
                    continue
                blind_path = Path(v["blind_path"])
                dur = blind_duration(blind_path)
                if spent + estimate_call_cost(args.model, dur, args.fps) > args.budget_usd:
                    log.error("budget would be exceeded (spent $%.2f); aborting", spent)
                    return 1
                rec = extractor.analyze(
                    v["blind_id"], blind_path,
                    render_prompt(template, dur), dur,
                    fps=args.fps, media_resolution=args.resolution,
                    prompt_version=args.prompt_version, prompt_sha256=prompt_sha)
                rec.update({"run_id": run_id, "run_k": k, "split": args.split,
                            "video_id": v["video_id"], "band": v["band"],
                            "duration_s_file": dur,
                            "duration_s_export": float(v["duration_s_export"])})
                out.write_text(json.dumps(rec, indent=2))
                spent += rec.get("cost_usd_paid_equiv", 0)
                done += 1
                log.info("%s r%d -> %s (%d tokens, $%.4f)",
                         v["blind_id"], k, rec["status"],
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
