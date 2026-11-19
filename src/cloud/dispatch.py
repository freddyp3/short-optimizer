"""The server's side of a review that runs on Google Cloud.

``CloudReviews.analyze`` has the signature of ``analyze_draft``, so the job runner uses it in place of the local
analysis. For one review it:

1. copies the video to Cloud Storage under the review's key, ``reviews/<sha256 of the video>/<draft id>/``;
2. adds a Cloud Tasks task **named after that key**, whose request starts one execution of the Cloud Run job
   with the key as an override. Cloud Tasks refuses a second task of the same name, and retries the request
   until Cloud Run has accepted it;
3. waits, passing on the step and the cost the job reports, until the job has written the result.

Every step can be repeated without repeating work: the upload and the task are create-if-absent, and the job
itself stops when the review is finished or already running (``review_job``). That is what lets the server
restart, or Cloud Tasks deliver twice, in the middle of a review.

A review paid with the user's own Gemini key never goes through here: the key would have to be written into the
task. It runs in the local analysis when there is one, and is refused otherwise.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

from src.cloud.review_job import prefix
from src.cloud.storage import Bucket, session
from src.server.jobs import RemoteFailure, file_sha256

TASKS_API = "https://cloudtasks.googleapis.com/v2"
RUN_API = "https://run.googleapis.com/v2"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
POLL_S = 3.0
TIMEOUT_S = 1500.0
MSG_OWN_KEY_OFF = "Uploads with your own Gemini key are switched off on this server. Use a free upload."
REQUIRED = ("SO_GCP_PROJECT", "SO_GCP_REGION", "SO_GCP_BUCKET", "SO_GCP_QUEUE", "SO_GCP_JOB", "SO_GCP_TASKS_SA")


class CloudReviews:
    def __init__(self, bucket: Bucket, http, project: str, region: str, queue: str, job: str, tasks_sa: str,
                 local_analyze: Callable | None = None, budget_usd: float | None = None, poll_s: float = POLL_S,
                 timeout_s: float = TIMEOUT_S, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.bucket, self.http, self.local_analyze, self.budget_usd = bucket, http, local_analyze, budget_usd
        self.queue_path = f"projects/{project}/locations/{region}/queues/{queue}"
        self.job_url = f"{RUN_API}/projects/{project}/locations/{region}/jobs/{job}:run"
        self.tasks_sa, self.poll_s, self.timeout_s, self.sleep, self.clock = tasks_sa, poll_s, timeout_s, sleep, clock

    def enqueue(self, key: str) -> bool:
        """Add the review's task. True if it was created, False if a task of that name already exists (the
        review was enqueued before: nothing to do)."""
        env = [{"name": "SO_REVIEW_KEY", "value": key}]
        if self.budget_usd is not None:
            env.append({"name": "SO_BUDGET_USD", "value": f"{self.budget_usd:.4f}"})
        body = json.dumps({"overrides": {"containerOverrides": [{"env": env}]}}).encode()
        task = {"name": f"{self.queue_path}/tasks/{key.replace('/', '-')}",
                "httpRequest": {"httpMethod": "POST", "url": self.job_url, "headers": {"Content-Type": "application/json"},
                                "body": base64.b64encode(body).decode(),
                                "oauthToken": {"serviceAccountEmail": self.tasks_sa, "scope": SCOPE}}}
        r = self.http.post(f"{TASKS_API}/{self.queue_path}/tasks", json={"task": task}, timeout=60)
        if r.status_code == 409:
            return False
        if r.status_code != 200:
            raise RuntimeError(f"Cloud Tasks answered {r.status_code}")
        return True

    def analyze(self, video_path, api_key=None, on_cost=None, on_step=None, checkpoint=None):
        if api_key is not None:
            if self.local_analyze is None:
                raise RemoteFailure("own_key_unavailable", MSG_OWN_KEY_OFF)
            return self.local_analyze(video_path, api_key=api_key, on_cost=on_cost, on_step=on_step)
        if checkpoint is None:
            raise RuntimeError("a cloud review needs the review's key")
        sha, draft_id = checkpoint.video_sha256, checkpoint.draft_id
        if file_sha256(video_path) != sha:
            raise RuntimeError("the video is not the one the review is keyed on")
        key = f"{sha}/{draft_id}"
        base = prefix(key)
        before = reported = float(getattr(checkpoint, "already_spent", 0.0))
        if self.bucket.stat(base + "result") is None:
            self.bucket.put_file(base + "video", Path(video_path), only_if_absent=True)
            self.enqueue(key)
        step, deadline = None, self.clock() + self.timeout_s
        while True:
            status = self.bucket.get_json(base + "status") or {}
            result = self.bucket.get_json(base + "result")
            total = float((result or status).get("cost_usd", 0.0))
            if status.get("step") and status["step"] != step and result is None:
                step = status["step"]
                if on_step:
                    on_step(step)
            if total > reported + 1e-9:
                if on_cost:
                    on_cost(total - reported)
                reported = total
            if result is not None:
                break
            if self.clock() >= deadline:
                raise TimeoutError("the review did not finish in time")
            self.sleep(self.poll_s)
        if not result.get("ok"):
            raise RemoteFailure(result.get("code", "analysis_failed"), result.get("message", ""))
        return SimpleNamespace(result=result["result"], artifacts=result["artifacts"], cost_usd=max(0.0, reported - before))


def from_env(env: Mapping[str, str], local_analyze: Callable | None = None, budget_usd: float | None = None) -> CloudReviews | None:
    """Reviews run on Google Cloud when ``SO_GCP_JOB`` is set; then the other ``SO_GCP_*`` settings are required.
    ``SO_GCP_CREDENTIALS`` names a service-account key file; without it the environment's own credentials are used."""
    if not env.get("SO_GCP_JOB", "").strip():
        return None
    missing = [name for name in REQUIRED if not env.get(name, "").strip()]
    if missing:
        raise ValueError(f"{', '.join(missing)} must be set when SO_GCP_JOB is")
    http = session(env.get("SO_GCP_CREDENTIALS", "").strip() or None)
    return CloudReviews(Bucket(env["SO_GCP_BUCKET"].strip(), http), http, env["SO_GCP_PROJECT"].strip(),
                        env["SO_GCP_REGION"].strip(), env["SO_GCP_QUEUE"].strip(), env["SO_GCP_JOB"].strip(),
                        env["SO_GCP_TASKS_SA"].strip(), local_analyze=local_analyze, budget_usd=budget_usd)
