"""Run the judging call on the pilot or validation videos (plan/03c section 20.4-20.5). Paid key, resumable.

    ./venv/bin/python -m src.models.judge.run --variant L4 --set pilot --stage C2 --stage-cap 3 [--limit 3]
        [--arm E|S|random] [--model gemini-3.8-flash] [--video] [--suffix _r2] [--ids-file path] [--dry-run]

One JSON record per video in data/models/judge/raw/<tag>/. Every call is checked against the ledger caps first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.evals.hook_test import config as C
from src.evals.hook_test.gemini_client import (extract_text, extract_thought_summary, extract_usage,
                                               is_network_error, is_rate_limit, new_genai_client, nominal_cost)
from src.models.judge import ledger
from src.models.judge.library import JUDGE_DIR, SEED, Library, build
from src.models.judge.prompt import PROMPT_VERSION, VARIANTS, build_prompt, schema, template_sha256, validate

log = logging.getLogger("judge")
RAW = JUDGE_DIR / "raw"
WORKERS = 2
RETRIES = 4
DEFAULT_CALL_EST = 0.06


def tag_for(variant: str, arm: str, model: str, video: bool, set_name: str, suffix: str = "") -> str:
    return f"{PROMPT_VERSION}_{variant}_{arm}_{model.replace('gemini-', '')}{'_video' if video else ''}_{set_name}{suffix}"


def generate(client, model: str, contents, sleep=time.sleep):
    from google.genai import types

    cfg = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=schema(),
        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level=types.ThinkingLevel.MEDIUM))
    for attempt in range(RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=cfg)
        except Exception as e:
            code = getattr(e, "code", None) or 0
            transient = is_network_error(e) or is_rate_limit(e) or (str(code).isdigit() and 500 <= int(code) < 600)
            if not transient or attempt >= RETRIES:
                raise
            d = 10.0 * 3 ** attempt
            log.warning("%r, retrying in %.0fs", e, d)
            sleep(d)


def judge_one(lib: Library, i: int, variant: str, arm: str, model: str, client, video_contents=None,
              seed: int = SEED) -> dict:
    """One video: build the prompt, call, validate, retry once if invalid. Returns the record."""
    prompt, shown = build_prompt(lib, i, variant, arm, np.random.default_rng(seed + i))
    row = lib.df.iloc[i]
    contents = video_contents(row, prompt) if video_contents else [prompt]
    rec = {"video_id": row.video_id, "variant": variant, "arm": arm, "model": model,
           "prompt_version": PROMPT_VERSION, "template_sha256": template_sha256(),
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt": prompt,
           "neighbours": [lib.df.iloc[j].video_id for j in shown], "cost_usd": 0.0, "attempts": 0}
    for _ in range(2):
        t0 = time.time()
        resp = generate(client, model, contents)
        usage = extract_usage(resp)
        rec["cost_usd"] += nominal_cost(model, usage.get("prompt_tokens", 0),
                                        usage.get("response_tokens", 0) + usage.get("thinking_tokens", 0))
        rec.update(attempts=rec["attempts"] + 1, usage=usage, latency_s=round(time.time() - t0, 1),
                   model_version=getattr(resp, "model_version", None), thought_summary=extract_thought_summary(resp),
                   raw_text=extract_text(resp))
        try:
            data = json.loads(rec["raw_text"])
        except (json.JSONDecodeError, TypeError):
            rec.update(status="invalid", invalid_reason="unparseable JSON", result=None)
            continue
        j, reason = validate(data, float(row.duration_s), len(shown))
        if reason:
            rec.update(status="invalid", invalid_reason=reason, result=data)
            continue
        rec.update(status="ok", invalid_reason=None, result=j.model_dump())
        break
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--set", dest="set_name", required=True, choices=["pilot", "validation", "gate", "heldout"])
    p.add_argument("--stage", required=True)
    p.add_argument("--stage-cap", type=float, required=True)
    p.add_argument("--arm", default="E", help="E, S, random, or T<percent> for a similarity threshold (e.g. T80)")
    p.add_argument("--model", default=C.DEFAULT_MODEL)
    p.add_argument("--video", action="store_true", help="send the draft's blinded video as well as its timeline")
    p.add_argument("--limit", type=int)
    p.add_argument("--ids-file", type=Path, help="JSON list of video ids; restricts the set")
    p.add_argument("--suffix", default="")
    p.add_argument("--hazard-extra", action="store_true", help="hints from the per-second model with timeline covariates")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()

    if args.set_name in ("gate", "heldout"):
        from src.models.judge.gate import build_gate
        lib = build_gate(args.set_name).lib
    else:
        lib = build(hazard_extra=args.hazard_extra)
    ids = lib.split[args.set_name]
    if args.ids_file:
        keep = set(json.loads(args.ids_file.read_text()))
        ids = [v for v in ids if v in keep]
    ids = ids[: args.limit] if args.limit else ids
    out_dir = RAW / tag_for(args.variant, args.arm, args.model, args.video, args.set_name, args.suffix)
    todo = [v for v in ids if not (out_dir / f"{v}.json").exists()]
    log.info("%s: %d videos, %d to do; ledger: stage %s $%.2f of $%.2f, total $%.2f of $%.2f", out_dir.name, len(ids),
             len(todo), args.stage, ledger.total(stage=args.stage), args.stage_cap, ledger.total(), ledger.HARD_CAP_USD)
    if args.dry_run:
        prompt, shown = build_prompt(lib, lib.index_of(ids[0]), args.variant, args.arm, np.random.default_rng(SEED))
        print(prompt)
        log.info("prompt: %d characters (about %d tokens), %d examples", len(prompt), len(prompt) // 4, len(shown))
        return 0
    key = os.environ.get("GEMINI_API_KEY_PAID")
    if not key:
        raise SystemExit("GEMINI_API_KEY_PAID not set")
    client = new_genai_client(key)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_contents = None
    if args.video:
        from src.models.dataset import blind_paths
        from src.tools.analyze_video.extract import GeminiFeatureExtractor

        ext = GeminiFeatureExtractor(client=client)
        paths = blind_paths().set_index("video_id")["blind_path"]
        video_contents = lambda row, prompt: ext._build_contents(
            ext.upload_video(row.blind_id, Path(paths[row.video_id])), prompt, C.FPS, C.MEDIA_RESOLUTION)

    costs: list[float] = []

    def work(v: str) -> str:
        est = float(np.mean(costs)) * 1.5 if costs else DEFAULT_CALL_EST * (5 if args.video else 1)
        ledger.check(est, args.stage, args.stage_cap)
        i = lib.index_of(v)
        base = RAW / tag_for("L0", "E", args.model, args.video, args.set_name) / f"{v}.json"
        if args.variant != "L0" and base.exists() and not build_prompt(lib, i, args.variant, args.arm)[1]:
            rec = json.loads(base.read_text())   # no example passed the threshold: the prompt is the L0 prompt
            if rec["prompt"] == build_prompt(lib, i, args.variant, args.arm)[0]:
                rec.update(variant=args.variant, arm=args.arm, reused_from=base.parent.name, cost_usd=0.0)
                (out_dir / f"{v}.json").write_text(json.dumps(rec, indent=1))
                return rec["status"]
        rec = judge_one(lib, i, args.variant, args.arm, args.model, client, video_contents)
        rec.update(set=args.set_name, stage=args.stage)
        ledger.record(rec["cost_usd"], args.stage, f"{out_dir.name}/{v}")
        costs.append(rec["cost_usd"])
        (out_dir / f"{v}.json").write_text(json.dumps(rec, indent=1))
        log.info("%s -> %s ($%.4f, %ss)", v, rec["status"], rec["cost_usd"], rec.get("latency_s"))
        return rec["status"]

    try:
        with ThreadPoolExecutor(WORKERS) as ex:
            statuses = list(ex.map(work, todo))
    except ledger.BudgetExceeded as e:
        log.error("stopped: %s", e)
        return 3
    bad = sum(s != "ok" for s in statuses)
    log.info("done: %d calls, %d invalid, $%.3f this run; ledger total $%.2f", len(statuses), bad, sum(costs), ledger.total())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
