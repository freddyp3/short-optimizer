"""The review of one draft as a Cloud Run job: ``python -m src.cloud.review_job``.

Cloud Tasks starts one execution per review and may start it more than once (its deliveries are at-least-once,
and Cloud Run retries a task that crashed). The review is keyed on the hash of its video, ``<sha256>/<draft id>``
in ``SO_REVIEW_KEY``, and everything it knows lives in Cloud Storage under ``reviews/<key>/``:

    video           the upload, put there by the server; checked against the hash before anything runs
    lease           who is working on the review (the execution's name) and when it last made progress
    stages/<name>   the output of each finished Gemini call
    status          the current step and what the review has cost so far
    result          the outcome; written once, by whoever finishes first

So a delivery does exactly one of three things. If ``result`` exists it stops: the work is done. If another
execution holds a fresh ``lease`` it stops: the work is being done. Otherwise it takes the lease and runs, and
every Gemini call whose output is already under ``stages/`` is skipped. No paid call is made twice.

A failure that a retry could cure (anything unexpected) ends the attempt with exit code 1 while Cloud Run has
retries left; the last attempt, and failures a retry cannot cure (a bad file, the budget), write a failed result.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from src.server.jobs import failure
from src.server.media import Rejected
from src.server.quota import BudgetExceeded

log = logging.getLogger("so.review_job")
LEASE_STALE_S = 600.0
DONE, DUPLICATE, RETRY = "done", "duplicate", "retry"


def prefix(key: str) -> str:
    return f"reviews/{key}/"


class Stages:
    """The review's checkpoint in the bucket: ``get(stage)`` / ``put(stage, value)``; the first write wins."""

    def __init__(self, bucket, base: str):
        self.bucket, self.base = bucket, base

    def get(self, stage: str):
        return self.bucket.get_json(f"{self.base}stages/{stage}")

    def put(self, stage: str, value) -> None:
        try:
            self.bucket.put_json(f"{self.base}stages/{stage}", value, only_if_absent=True)
        except Exception as inner:
            log.warning("could not keep a stage class=%s", type(inner).__name__)


def take_lease(bucket, base: str, owner: str, now: float) -> bool:
    """True if this execution may work on the review: nobody holds the lease, it holds it itself (a retry of the
    same execution), or the holder made no progress for ``LEASE_STALE_S``."""
    mine = {"owner": owner, "at": now}
    if bucket.put_json(base + "lease", mine, only_if_absent=True):
        return True
    meta, held = bucket.stat(base + "lease"), bucket.get_json(base + "lease")
    if meta is None or held is None:
        return bucket.put_json(base + "lease", mine, only_if_absent=True)
    if held.get("owner") != owner and now - float(held.get("at", 0)) < LEASE_STALE_S:
        return False
    return bucket.put_json(base + "lease", mine, if_generation=meta["generation"])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(bucket, key: str, analyze: Callable, owner: str, budget_usd: float | None = None, last_attempt: bool = True,
        now: Callable[[], float] = time.time, workdir: Path | None = None) -> str:
    """One delivery of the review ``key``. Returns ``done`` (a result was written, good or failed), ``duplicate``
    (nothing to do) or ``retry`` (this attempt failed and another may cure it)."""
    sha, _, draft_id = key.partition("/")
    base = prefix(key)
    if bucket.stat(base + "result") is not None or not take_lease(bucket, base, owner, now()):
        return DUPLICATE
    spent = [float((bucket.get_json(base + "status") or {}).get("cost_usd", 0.0))]
    step = [None]

    def progress() -> None:
        bucket.put_json(base + "status", {"step": step[0], "cost_usd": round(spent[0], 6), "at": now()})
        bucket.put_json(base + "lease", {"owner": owner, "at": now()})

    def on_step(name: str) -> None:
        step[0] = str(name)
        progress()

    def on_cost(usd: float) -> None:
        spent[0] += float(usd)
        progress()
        if budget_usd is not None and spent[0] > budget_usd + 1e-9:
            raise BudgetExceeded(f"review cost ${spent[0]:.2f} passed its budget")

    def finish(body: dict) -> str:
        bucket.put_json(base + "result", {**body, "cost_usd": round(spent[0], 6)}, only_if_absent=True)
        for name in [base + "video", base + "lease", *bucket.list(base + "stages/")]:
            bucket.delete(name)
        return DONE

    with tempfile.TemporaryDirectory(prefix="review_", dir=workdir) as tmp:
        video = Path(tmp) / "video"
        try:
            if not bucket.download(base + "video", video) or sha256_of(video) != sha:
                raise Rejected("unreadable", "The uploaded video did not arrive intact. Please upload it again.")
            analysis = analyze(video, on_cost=on_cost, on_step=on_step, checkpoint=Stages(bucket, base))
            if not isinstance(analysis.result, dict) or not isinstance(analysis.artifacts, dict):
                raise TypeError("analysis returned no result")
            return finish({"ok": True, "result": analysis.result, "artifacts": analysis.artifacts})
        except Exception as exc:
            code, message = failure(exc, "server")
            log.warning("review failed class=%s code=%s", type(exc).__name__, code)
            if last_attempt or isinstance(exc, (Rejected, BudgetExceeded)):
                return finish({"ok": False, "code": code, "message": message})
            return RETRY


def canned(video_path, on_cost=None, on_step=None, checkpoint=None):
    """Stand-in analysis for testing the plumbing (``SO_FAKE_ANALYSIS=1``): two paid stages that honour the
    checkpoint, no Gemini call. ``SO_FAKE_CRASH=1`` kills the first attempt between the stages."""
    from types import SimpleNamespace

    from src.server.app import fake_analyze

    for stage, usd in (("extract_v2", 0.01), ("judge_L0", 0.01)):
        on_step(stage)
        if checkpoint.get(stage) is None:
            on_cost(usd)
            checkpoint.put(stage, {"status": "ok", "stage": stage})
        if stage == "extract_v2" and os.environ.get("SO_FAKE_CRASH") == "1" and os.environ.get("CLOUD_RUN_TASK_ATTEMPT", "0") == "0":
            os._exit(3)
    done = fake_analyze(video_path, delay_s=0.0)
    return SimpleNamespace(result=done.result, artifacts=done.artifacts, cost_usd=0.0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s", stream=sys.stderr)
    from src.cloud.storage import Bucket, session

    key = os.environ["SO_REVIEW_KEY"]
    attempt, retries = int(os.environ.get("CLOUD_RUN_TASK_ATTEMPT", "0")), int(os.environ.get("SO_MAX_RETRIES", "1"))
    budget = os.environ.get("SO_BUDGET_USD", "").strip()
    if os.environ.get("SO_FAKE_ANALYSIS") == "1":
        analyze = canned
    else:
        from src.tools.predict_retention.tool import analyze_draft as analyze
    outcome = run(Bucket(os.environ["SO_GCP_BUCKET"], session()), key, analyze,
                  owner=os.environ.get("CLOUD_RUN_EXECUTION", f"local-{os.getpid()}"),
                  budget_usd=float(budget) if budget else None, last_attempt=attempt >= retries)
    log.info("review %s attempt %d: %s", key[:12], attempt, outcome)
    return 1 if outcome == RETRY else 0


if __name__ == "__main__":
    raise SystemExit(main())
