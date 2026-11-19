"""Reviews on Google Cloud: the job's exactly-once rules and the server's dispatch, against an in-memory bucket."""

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cloud import review_job as J
from src.cloud.dispatch import MSG_OWN_KEY_OFF, CloudReviews
from src.server.jobs import RemoteFailure, failure
from src.server.media import Rejected

VIDEO = b"the video bytes"
SHA = hashlib.sha256(VIDEO).hexdigest()
KEY = f"{SHA}/draft-1"
BASE = f"reviews/{KEY}/"
RESULT, ARTIFACTS = {"advice": "end on the payoff"}, {"embedding": [1.0, 0.0]}


class FakeBucket:
    """The calls of ``storage.Bucket`` over a dict, with generations and the two preconditions."""

    def __init__(self):
        self.objects: dict[str, tuple[bytes, int]] = {}
        self.gen = 0
        self.writes: list[str] = []

    def stat(self, key):
        return {"generation": str(self.objects[key][1])} if key in self.objects else None

    def get(self, key):
        return self.objects[key][0] if key in self.objects else None

    def get_json(self, key):
        return json.loads(self.objects[key][0]) if key in self.objects else None

    def download(self, key, dest):
        if key not in self.objects:
            return False
        Path(dest).write_bytes(self.objects[key][0])
        return True

    def put(self, key, data, only_if_absent=False, if_generation=None, content_type=""):
        if only_if_absent and key in self.objects:
            return False
        if if_generation is not None and (key not in self.objects or str(self.objects[key][1]) != str(if_generation)):
            return False
        self.gen += 1
        self.objects[key] = (data if isinstance(data, bytes) else data.read(), self.gen)
        self.writes.append(key)
        return True

    def put_json(self, key, value, **kw):
        return self.put(key, json.dumps(value).encode(), **kw)

    def put_file(self, key, path, only_if_absent=False):
        return self.put(key, Path(path).read_bytes(), only_if_absent=only_if_absent)

    def delete(self, key):
        return self.objects.pop(key, None) is not None

    def list(self, prefix):
        return sorted(k for k in self.objects if k.startswith(prefix))


@pytest.fixture
def bucket():
    b = FakeBucket()
    b.put(BASE + "video", VIDEO)
    return b


def staged(calls: list, fail_after: str | None = None, error: Exception | None = None):
    """Two paid stages that honour the checkpoint; optionally dies after one of them."""
    def analyze(path, on_cost=None, on_step=None, checkpoint=None):
        assert Path(path).read_bytes() == VIDEO
        for stage, usd in (("extract_v2", 0.03), ("judge_L0", 0.01)):
            on_step(stage)
            if checkpoint.get(stage) is None:
                calls.append(stage)
                on_cost(usd)
                checkpoint.put(stage, {"status": "ok"})
            if stage == fail_after:
                raise error or RuntimeError("boom at /secret/path")
        return SimpleNamespace(result=RESULT, artifacts=ARTIFACTS, cost_usd=0.0)
    return analyze


# ------------------------------------------------------------------------------------------------------------ the job

def test_a_review_runs_writes_its_result_and_cleans_up(bucket, tmp_path):
    calls = []
    assert J.run(bucket, KEY, staged(calls), owner="exec-1", workdir=tmp_path) == J.DONE
    assert calls == ["extract_v2", "judge_L0"]
    assert bucket.get_json(BASE + "result") == {"ok": True, "result": RESULT, "artifacts": ARTIFACTS, "cost_usd": 0.04}
    assert bucket.get_json(BASE + "status")["cost_usd"] == 0.04
    assert bucket.list(BASE) == [BASE + "result", BASE + "status"]              # video, lease and stages are gone
    assert list(tmp_path.iterdir()) == []


def test_a_second_delivery_of_a_finished_review_does_nothing(bucket):
    calls = []
    J.run(bucket, KEY, staged(calls), owner="exec-1")
    writes = len(bucket.writes)
    assert J.run(bucket, KEY, staged(calls), owner="exec-2") == J.DUPLICATE
    assert calls == ["extract_v2", "judge_L0"] and len(bucket.writes) == writes


def test_a_delivery_while_another_execution_works_does_nothing(bucket):
    calls = []
    assert J.take_lease(bucket, BASE, "exec-1", now=1000.0)
    assert J.run(bucket, KEY, staged(calls), owner="exec-2", now=lambda: 1000.0 + J.LEASE_STALE_S - 1) == J.DUPLICATE
    assert calls == [] and bucket.get_json(BASE + "lease")["owner"] == "exec-1"
    assert J.run(bucket, KEY, staged(calls), owner="exec-2", now=lambda: 1000.0 + J.LEASE_STALE_S + 1) == J.DONE   # it went quiet
    assert calls == ["extract_v2", "judge_L0"]


def test_a_retry_skips_the_stage_that_finished_and_pays_only_for_the_rest(bucket):
    calls = []
    assert J.run(bucket, KEY, staged(calls, fail_after="extract_v2"), owner="exec-1", last_attempt=False) == J.RETRY
    assert bucket.stat(BASE + "result") is None and bucket.get_json(BASE + "status")["cost_usd"] == 0.03
    assert bucket.stat(BASE + "video") is not None                                 # kept for the retry
    assert J.run(bucket, KEY, staged(calls), owner="exec-1") == J.DONE            # same execution, next attempt
    assert calls == ["extract_v2", "judge_L0"]                                     # each paid call ran once
    assert bucket.get_json(BASE + "result")["cost_usd"] == 0.04


def test_the_last_attempt_writes_a_failed_result_with_a_safe_sentence(bucket):
    assert J.run(bucket, KEY, staged([], fail_after="extract_v2"), owner="exec-1", last_attempt=True) == J.DONE
    result = bucket.get_json(BASE + "result")
    assert result["ok"] is False and result["code"] == "analysis_failed" and result["cost_usd"] == 0.03
    assert "/secret/path" not in json.dumps(result) and bucket.stat(BASE + "video") is None


def test_failures_a_retry_cannot_cure_end_the_review_at_once(bucket):
    over = J.run(bucket, KEY, staged([]), owner="exec-1", budget_usd=0.02, last_attempt=False)
    assert over == J.DONE and bucket.get_json(BASE + "result")["code"] == "over_budget"
    other = FakeBucket()
    other.put(BASE + "video", b"not the video the key names")
    calls = []
    assert J.run(other, KEY, staged(calls), owner="exec-1", last_attempt=False) == J.DONE
    assert other.get_json(BASE + "result")["code"] == "unreadable" and calls == []
    missing = FakeBucket()
    assert J.run(missing, KEY, staged(calls), owner="exec-1", last_attempt=False) == J.DONE and calls == []


def test_a_rejected_file_keeps_its_own_sentence(bucket):
    J.run(bucket, KEY, staged([], fail_after="extract_v2", error=Rejected("too_short")), owner="exec-1", last_attempt=False)
    result = bucket.get_json(BASE + "result")
    assert result["code"] == "too_short" and "seconds" in result["message"]


# ------------------------------------------------------------------------------------------------------- the dispatch

class FakeHttp:
    """Cloud Tasks: remembers task names, refuses a second task of the same name, runs the job when asked."""

    def __init__(self, on_task=None, status=200):
        self.names, self.posts, self.on_task, self.status = set(), [], on_task, status

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        name = json["task"]["name"]
        if self.status != 200:
            return SimpleNamespace(status_code=self.status)
        if name in self.names:
            return SimpleNamespace(status_code=409)
        self.names.add(name)
        if self.on_task:
            self.on_task(json["task"])
        return SimpleNamespace(status_code=200)


def reviews(bucket, http, **kw):
    ticks = iter(range(10_000))
    return CloudReviews(bucket, http, "proj", "us-west1", "reviews", "review", "so-tasks@proj.iam.gserviceaccount.com",
                        **{"sleep": lambda s: None, "clock": lambda: float(next(ticks)), "timeout_s": 50, **kw})


def checkpoint(already_spent: float = 0.0):
    return SimpleNamespace(draft_id="draft-1", video_sha256=SHA, already_spent=already_spent)


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "upload"
    p.write_bytes(VIDEO)
    return p


def test_dispatch_uploads_enqueues_a_task_named_after_the_key_and_returns_the_result(video):
    bucket, calls = FakeBucket(), []
    http = FakeHttp(on_task=lambda task: J.run(bucket, KEY, staged(calls), owner="exec-1"))
    costs, steps = [], []
    out = reviews(bucket, http, budget_usd=0.24).analyze(video, on_cost=costs.append, on_step=steps.append, checkpoint=checkpoint())
    assert out.result == RESULT and out.artifacts == ARTIFACTS and out.cost_usd == pytest.approx(0.04)
    assert sum(costs) == pytest.approx(0.04) and calls == ["extract_v2", "judge_L0"]
    (url, body), = http.posts
    task = body["task"]
    assert url == "https://cloudtasks.googleapis.com/v2/projects/proj/locations/us-west1/queues/reviews/tasks"
    assert task["name"].endswith(f"/queues/reviews/tasks/{SHA}-draft-1")
    request = task["httpRequest"]
    assert request["url"] == "https://run.googleapis.com/v2/projects/proj/locations/us-west1/jobs/review:run"
    assert request["oauthToken"]["serviceAccountEmail"] == "so-tasks@proj.iam.gserviceaccount.com"
    env = json.loads(base64.b64decode(request["body"]))["overrides"]["containerOverrides"][0]["env"]
    assert env == [{"name": "SO_REVIEW_KEY", "value": KEY}, {"name": "SO_BUDGET_USD", "value": "0.2400"}]


def test_dispatching_again_after_a_restart_repeats_nothing(video):
    bucket, calls = FakeBucket(), []
    http = FakeHttp(on_task=lambda task: J.run(bucket, KEY, staged(calls), owner="exec-1"))
    r = reviews(bucket, http)
    first = r.analyze(video, checkpoint=checkpoint())
    costs = []
    again = r.analyze(video, on_cost=costs.append, checkpoint=checkpoint(already_spent=0.04))   # the server came back
    assert again.result == first.result and again.cost_usd == 0 and costs == []
    assert calls == ["extract_v2", "judge_L0"] and len(http.posts) == 1                          # no second task, no second run


def test_a_task_that_already_exists_is_not_an_error_and_the_wait_goes_on(video):
    bucket, calls = FakeBucket(), []
    http = FakeHttp()
    http.names.add(f"projects/proj/locations/us-west1/queues/reviews/tasks/{SHA}-draft-1")      # enqueued before the restart
    polls = []

    def sleep(s):
        polls.append(s)
        if len(polls) == 3:
            J.run(bucket, KEY, staged(calls), owner="exec-1")                                    # the job finishes meanwhile

    costs = []
    out = reviews(bucket, http, sleep=sleep).analyze(video, on_cost=costs.append, checkpoint=checkpoint(already_spent=0.03))
    assert out.result == RESULT and costs == [pytest.approx(0.01)] and out.cost_usd == pytest.approx(0.01)


def test_a_failed_review_surfaces_with_the_jobs_sentence(video):
    bucket = FakeBucket()
    http = FakeHttp(on_task=lambda task: J.run(bucket, KEY, staged([], fail_after="extract_v2"), owner="exec-1"))
    costs = []
    with pytest.raises(RemoteFailure) as e:
        reviews(bucket, http).analyze(video, on_cost=costs.append, checkpoint=checkpoint())
    assert failure(e.value, "server") == ("analysis_failed", bucket.get_json(BASE + "result")["message"])
    assert costs == [pytest.approx(0.03)]                                                        # what it cost is still recorded


def test_dispatch_gives_up_after_its_timeout_and_refuses_what_it_cannot_key(video, tmp_path):
    with pytest.raises(TimeoutError):
        reviews(FakeBucket(), FakeHttp()).analyze(video, checkpoint=checkpoint())
    with pytest.raises(RuntimeError):
        reviews(FakeBucket(), FakeHttp(status=403)).analyze(video, checkpoint=checkpoint())
    with pytest.raises(RuntimeError):
        reviews(FakeBucket(), FakeHttp()).analyze(video, checkpoint=None)
    other = tmp_path / "other"
    other.write_bytes(b"another video")
    with pytest.raises(RuntimeError):
        reviews(FakeBucket(), FakeHttp()).analyze(other, checkpoint=checkpoint())


def test_an_own_key_review_never_reaches_the_cloud(video):
    bucket, http, seen = FakeBucket(), FakeHttp(), []

    def local(path, api_key=None, on_cost=None, on_step=None):
        seen.append(api_key)
        return SimpleNamespace(result=RESULT, artifacts=ARTIFACTS, cost_usd=0.0)

    assert reviews(bucket, http, local_analyze=local).analyze(video, api_key="FAKE-KEY").result == RESULT
    with pytest.raises(RemoteFailure) as e:
        reviews(bucket, http).analyze(video, api_key="FAKE-KEY")
    assert e.value.user_message == MSG_OWN_KEY_OFF and seen == ["FAKE-KEY"]
    assert bucket.objects == {} and http.posts == [] and "FAKE-KEY" not in json.dumps(http.posts)
