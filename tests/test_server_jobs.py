"""Job runner of the Claude MCP server: outcomes, cleanup, limits and secrecy (plan/10a sections 7 and 9)."""

import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.server import jobs
from src.server.config import Config
from src.server.jobs import JobRefused, JobRunner
from src.server.media import Rejected
from src.server.quota import Quota
from src.server.store import DraftStateError, Store

CFG = Config(DRAFTS_PER_DAY=50, OWN_KEY_DRAFTS_PER_DAY=50, DAY_CAP_USD=100.0, MONTH_CAP_USD=100.0,
             EST_COST_PER_DRAFT_USD=0.12)
FAKE_KEY = "FAKE-KEY-123"
FAKE_PATH = "/Users/somebody/secret-folder/clip.mp4"
RESULT = {"advice": "end on the payoff", "segments": {"mid": {"why": ["x"]}}}
ARTIFACTS = {"timeline": [{"start_s": 0.0, "end_s": 1.0, "seen": "a", "heard": "b"}], "embedding": [1.0, 0.0]}


class Env:
    def __init__(self, tmp_path: Path):
        self.t = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc).timestamp()
        self.db, self.uploads = tmp_path / "server.sqlite3", tmp_path / "uploads"
        self.uploads.mkdir()
        self.store, self.quota = Store(self.db, lambda: self.t), Quota(self.db, lambda: self.t, CFG)
        self.runners: list[JobRunner] = []

    def runner(self, analyze, **kw) -> JobRunner:
        self.runners.append(JobRunner(self.store, self.quota, analyze, **{"max_total": 2, "max_per_user": 1, **kw}))
        return self.runners[-1]

    def claimed(self, token_id: str = "tok-a", paid_by: str = "server"):
        draft = self.store.create_draft(token_id)
        self.store.claim_upload(draft.upload_id)
        self.quota.check_and_reserve(token_id, draft.id, paid_by)
        path = self.uploads / draft.id
        path.write_bytes(b"video bytes")
        return draft, path

    def db_bytes(self) -> bytes:
        return b"".join(p.read_bytes() for p in self.db.parent.iterdir() if p.name.startswith(self.db.name))


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    for r in e.runners:
        r.shutdown(wait=True)


def ok(steps=("measuring", "describing"), costs=(0.02, 0.03), seen=None):
    def analyze(path, api_key=None, on_cost=None, on_step=None):
        if seen is not None:
            seen.append({"path": Path(path), "api_key": api_key, "existed": Path(path).exists()})
        for s in steps:
            on_step(s)
        for c in costs:
            on_cost(c)
        return SimpleNamespace(result=RESULT, artifacts=ARTIFACTS, cost_usd=sum(costs))
    return analyze


def test_success_stores_result_artifacts_cost_and_deletes_the_file(env):
    seen = []
    runner = env.runner(ok(seen=seen))
    draft, path = env.claimed()
    runner.submit(draft, path)
    assert runner.wait_idle(10)
    got = env.store.get_draft(draft.id, "tok-a")
    assert (got.status, got.paid_by, got.step) == ("done", "server", None)
    assert got.result == RESULT and got.artifacts == ARTIFACTS
    assert got.cost_usd == pytest.approx(0.05)
    assert seen == [{"path": path, "api_key": None, "existed": True}]
    assert not path.exists() and runner.active_ids() == set()
    assert env.quota.summary()["settled_usd"] == pytest.approx(0.05)
    assert env.quota.usage("tok-a")["drafts_today"] == 1


def test_steps_are_written_while_the_job_runs(env):
    at_step, go = threading.Event(), threading.Event()

    def analyze(path, api_key=None, on_cost=None, on_step=None):
        on_step("describing")
        at_step.set()
        go.wait(10)
        return SimpleNamespace(result=RESULT, artifacts=ARTIFACTS, cost_usd=0.0)

    runner = env.runner(analyze)
    draft, path = env.claimed()
    runner.submit(draft, path)
    assert at_step.wait(10)
    got = env.store.get_draft(draft.id, "tok-a")
    assert (got.status, got.step) == ("processing", "describing")
    assert runner.active_ids() == {draft.id}
    go.set()
    assert runner.wait_idle(10)


def test_job_reports_starting_until_the_library_is_ready(env):
    ready, called = threading.Event(), threading.Event()

    def analyze(path, api_key=None, on_cost=None, on_step=None):
        called.set()
        return SimpleNamespace(result=RESULT, artifacts=ARTIFACTS, cost_usd=0.0)

    runner = env.runner(analyze, ready=ready)
    draft, path = env.claimed()
    runner.submit(draft, path)
    for _ in range(200):
        if env.store.get_draft(draft.id, "tok-a").step == "starting":
            break
        threading.Event().wait(0.01)
    assert env.store.get_draft(draft.id, "tok-a").step == "starting" and not called.is_set()
    ready.set()
    assert runner.wait_idle(10)
    assert env.store.get_draft(draft.id, "tok-a").status == "done"


def test_failure_stores_a_fixed_sentence_and_leaks_nothing(env, caplog):
    def analyze(path, api_key=None, on_cost=None, on_step=None):
        on_cost(0.04)
        raise RuntimeError(f"ffmpeg exploded reading {FAKE_PATH} and {path} with key {api_key}")

    runner = env.runner(analyze)
    draft, path = env.claimed(paid_by="own_key")
    with caplog.at_level(logging.DEBUG):
        runner.submit(draft, path, FAKE_KEY)
        assert runner.wait_idle(10)
    got = env.store.get_draft(draft.id, "tok-a")
    assert (got.status, got.fail_code, got.fail_message) == ("failed", "analysis_failed", jobs.MSG_FAILED)
    assert got.cost_usd == pytest.approx(0.04) and got.result is None
    assert not path.exists()
    assert env.quota.summary()["own_key_usd"] == pytest.approx(0.04)
    for where in (repr(got), caplog.text, env.db_bytes().decode("latin-1")):
        for secret in (FAKE_KEY, FAKE_PATH, "ffmpeg exploded", str(path), str(env.uploads)):
            assert secret not in where
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize("reported, paid_by, settled, own", [
    ((), "server", 0.12, 0.0),              # no cost reported: the estimate is charged
    ((0.04,), "server", 0.12, 0.0),         # below the estimate: the estimate is charged
    ((0.10, 0.08), "server", 0.18, 0.0),    # above it: what was reported
    ((), "own_key", 0.0, 0.0),              # the server paid nothing
    ((0.04,), "own_key", 0.0, 0.04),        # informational only
])
def test_a_failed_analysis_is_charged_at_least_the_estimate_when_the_server_pays(env, reported, paid_by, settled, own):
    def analyze(path, api_key=None, on_cost=None, on_step=None):
        for usd in reported:
            on_cost(usd)
        raise RuntimeError("a billed call may have gone out without reporting its cost")

    runner = env.runner(analyze)
    draft, path = env.claimed(paid_by=paid_by)
    runner.submit(draft, path, FAKE_KEY if paid_by == "own_key" else None)
    assert runner.wait_idle(10)
    got, summary = env.store.get_draft(draft.id, "tok-a"), env.quota.summary()
    assert got.status == "failed" and got.cost_usd == pytest.approx(sum(reported))      # the draft keeps what was reported
    assert summary["settled_usd"] == pytest.approx(settled) == pytest.approx(summary["spend_usd"])
    assert summary["own_key_usd"] == pytest.approx(own)
    assert runner.failed_charge(0.0, "server") == 0.12 and runner.failed_charge(0.3, "server") == 0.3


def test_a_job_that_fails_after_the_analysis_returned_is_settled_at_its_real_cost(env):
    def analyze(path, api_key=None, on_cost=None, on_step=None):
        on_cost(0.03)
        return SimpleNamespace(result="not a dict", artifacts=ARTIFACTS, cost_usd=0.03)

    runner = env.runner(analyze)
    draft, path = env.claimed()
    runner.submit(draft, path)
    assert runner.wait_idle(10)
    assert env.store.get_draft(draft.id, "tok-a").status == "failed"
    assert env.quota.summary()["settled_usd"] == pytest.approx(0.03)


def test_budget_exceeded_from_on_cost_stops_the_job(env):
    calls = []

    def analyze(path, api_key=None, on_cost=None, on_step=None):
        for _ in range(10):
            calls.append(1)
            on_cost(0.20)
        return SimpleNamespace(result=RESULT, artifacts=ARTIFACTS, cost_usd=2.0)

    runner = env.runner(analyze)
    draft, path = env.claimed()
    runner.submit(draft, path)
    assert runner.wait_idle(10)
    got = env.store.get_draft(draft.id, "tok-a")
    assert len(calls) == 2
    assert (got.status, got.fail_code, got.fail_message) == ("failed", "over_budget", jobs.MSG_BUDGET)
    assert got.cost_usd == pytest.approx(0.40) and not path.exists()
    assert env.quota.summary()["settled_usd"] == pytest.approx(0.40)


class ApiError(Exception):
    def __init__(self, code: int, text: str = "refused"):
        super().__init__(text)
        self.code = code


@pytest.mark.parametrize("exc, own_key, message", [
    (ApiError(403), FAKE_KEY, jobs.MSG_OWN_KEY),
    (ApiError(429), FAKE_KEY, jobs.MSG_OWN_KEY),
    (RuntimeError("ClientError: 400 INVALID_ARGUMENT API key not valid [redacted key]"), FAKE_KEY, jobs.MSG_OWN_KEY),
    (RuntimeError("DailyQuotaExceeded: daily quota hit"), FAKE_KEY, jobs.MSG_OWN_KEY),
    (ApiError(403), None, jobs.MSG_FAILED),
    (ApiError(500), FAKE_KEY, jobs.MSG_FAILED),
    (ValueError("bad json"), FAKE_KEY, jobs.MSG_FAILED),
    (Rejected("too_long"), None, Rejected("too_long").user_message),
])
def test_failure_sentence_is_chosen_by_error_class(env, exc, own_key, message):
    def analyze(path, api_key=None, on_cost=None, on_step=None):
        raise exc

    runner = env.runner(analyze)
    draft, path = env.claimed(paid_by="own_key" if own_key else "server")
    runner.submit(draft, path, own_key)
    assert runner.wait_idle(10)
    got = env.store.get_draft(draft.id, "tok-a")
    assert (got.status, got.fail_message) == ("failed", message)
    assert not path.exists()


def test_key_problem_is_found_through_the_cause_chain():
    try:
        try:
            raise ApiError(401)
        except ApiError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as exc:
        assert jobs.key_problem(exc)
    assert not jobs.key_problem(RuntimeError("extraction v2 failed: schema"))


def test_unusable_result_fails_the_job_and_still_deletes_the_file(env):
    def analyze(path, api_key=None, on_cost=None, on_step=None):
        return SimpleNamespace(result={"x": float("nan")}, artifacts=ARTIFACTS, cost_usd=0.01)

    runner = env.runner(analyze)
    draft, path = env.claimed()
    runner.submit(draft, path)
    assert runner.wait_idle(10)
    got = env.store.get_draft(draft.id, "tok-a")
    assert (got.status, got.fail_message) == ("failed", jobs.MSG_FAILED)
    assert not path.exists() and env.quota.summary()["settled_usd"] == pytest.approx(0.01)


def test_own_key_reaches_analyze_and_nothing_else(env, caplog):
    seen = []
    runner = env.runner(ok(seen=seen))
    draft, path = env.claimed(paid_by="own_key")
    with caplog.at_level(logging.DEBUG):
        runner.submit(draft, path, FAKE_KEY)
        assert runner.wait_idle(10)
    assert seen[0]["api_key"] == FAKE_KEY
    got = env.store.get_draft(draft.id, "tok-a")
    assert (got.status, got.paid_by) == ("done", "own_key")
    assert FAKE_KEY not in repr(got) and FAKE_KEY not in caplog.text and FAKE_KEY.encode() not in env.db_bytes()
    assert FAKE_KEY not in repr(vars(runner))
    assert env.quota.usage("tok-a")["drafts_today"] == 0 and env.quota.usage("tok-a")["own_key_drafts_today"] == 1


def _blocking(release: threading.Event):
    def analyze(path, api_key=None, on_cost=None, on_step=None):
        release.wait(20)
        return SimpleNamespace(result=RESULT, artifacts=ARTIFACTS, cost_usd=0.0)
    return analyze


def _submit_all(runner: JobRunner, items) -> list[str]:
    barrier, out, lock = threading.Barrier(len(items)), [], threading.Lock()

    def go(draft, path):
        barrier.wait(10)
        try:
            runner.submit(draft, path)
            got = "started"
        except JobRefused as exc:
            got = exc.kind
        with lock:
            out.append(got)

    threads = [threading.Thread(target=go, args=item) for item in items]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    return sorted(out)


def test_total_job_limit_under_concurrent_submits(env):
    release = threading.Event()
    runner = env.runner(_blocking(release), max_total=2, max_per_user=1)
    items = [env.claimed(f"tok-{i}") for i in range(6)]
    assert _submit_all(runner, items) == ["server_busy"] * 4 + ["started"] * 2
    assert env.store.running_jobs() == 2 and len(runner.active_ids()) == 2
    refused = [(d, p) for d, p in items if d.id not in runner.active_ids()]
    assert all(p.exists() and env.store.get_draft(d.id, d.token_id).status == "waiting_for_upload" for d, p in refused)
    release.set()
    assert runner.wait_idle(10)
    assert env.store.running_jobs() == 0
    runner.submit(*refused[0])
    assert runner.wait_idle(10)
    assert env.store.get_draft(refused[0][0].id, refused[0][0].token_id).status == "done"


def test_per_user_job_limit_under_concurrent_submits(env):
    release = threading.Event()
    runner = env.runner(_blocking(release), max_total=5, max_per_user=1)
    items = [env.claimed("tok-a") for _ in range(4)]
    con = sqlite3.connect(env.db)      # create_draft expires the older ones; bring them back to test the race itself
    con.execute("UPDATE drafts SET status = 'waiting_for_upload', finished_at = NULL WHERE token_id = 'tok-a'")
    con.commit()
    con.close()
    assert _submit_all(runner, items) == ["started"] + ["user_busy"] * 3
    assert env.store.running_jobs("tok-a") == 1
    other = env.claimed("tok-b")
    runner.submit(*other)
    assert env.store.running_jobs() == 2
    release.set()
    assert runner.wait_idle(10)


def test_submit_of_an_unclaimed_draft_raises_and_starts_nothing(env):
    runner = env.runner(ok())
    draft = env.store.create_draft("tok-a")
    path = env.uploads / draft.id
    path.write_bytes(b"x")
    with pytest.raises(DraftStateError):
        runner.submit(draft, path)
    assert runner.active_ids() == set() and path.exists()



# ------------------------------------------------------------------------------------- one review, delivered many times

def cut_off(env, stage: dict | None = None, cost: float = 0.0):
    """A server-paid review whose process died: processing, video on disk, maybe a kept stage and recorded cost."""
    draft, path = env.claimed()
    sha = jobs.file_sha256(path)
    env.store.start_processing(draft.id, "server", sha)
    if cost:
        env.quota.add_cost(draft.id, cost)
    for name, value in (stage or {}).items():
        env.store.put_stage(draft.id, sha, name, value)
    assert [d.id for d in env.store.redeliver_interrupted()[0]] == [draft.id]
    return draft, path, sha


def staged(calls: list, costs: dict):
    """An analysis of two stages that honours the checkpoint: a kept stage is not run and not paid again."""
    def analyze(path, api_key=None, on_cost=None, on_step=None, checkpoint=None):
        spent = 0.0
        for stage in ("extract_v2", "judge_L0"):
            if checkpoint.get(stage) is None:
                calls.append(stage)
                on_cost(costs[stage])
                spent += costs[stage]
                checkpoint.put(stage, {"status": "ok", "stage": stage})
        return SimpleNamespace(result=RESULT, artifacts=ARTIFACTS, cost_usd=spent)
    return analyze


def test_file_sha256_is_the_hash_of_the_bytes(tmp_path):
    p = tmp_path / "v"
    p.write_bytes(b"video bytes")
    assert jobs.file_sha256(p) == hashlib.sha256(b"video bytes").hexdigest()


def test_submit_keys_the_review_on_the_video_and_hands_the_job_a_checkpoint(env):
    calls = []
    runner = env.runner(staged(calls, {"extract_v2": 0.03, "judge_L0": 0.01}))
    draft, path = env.claimed()
    sha = jobs.file_sha256(path)
    runner.submit(draft, path)
    assert runner.wait_idle(10)
    got = env.store.get_draft(draft.id, "tok-a")
    assert (got.status, got.video_sha256, got.attempts) == ("done", sha, 1) and calls == ["extract_v2", "judge_L0"]
    assert env.store.stages(draft.id, sha) == {}                                # nothing is kept once it is over


def test_a_redelivered_review_skips_its_finished_stage_and_pays_only_for_the_rest(env):
    calls = []
    draft, path, sha = cut_off(env, {"extract_v2": {"status": "ok", "stage": "extract_v2"}}, cost=0.03)
    runner = env.runner(staged(calls, {"extract_v2": 0.03, "judge_L0": 0.01}))
    assert runner.resume(draft.id, "tok-a", path) == "resumed"
    assert runner.wait_idle(10)
    got = env.store.get_draft(draft.id, "tok-a")
    assert (got.status, got.attempts) == ("done", 2) and got.result == RESULT
    assert calls == ["judge_L0"]                                                 # the kept stage did not run again
    assert got.cost_usd == pytest.approx(0.04) and env.quota.summary()["settled_usd"] == pytest.approx(0.04)
    assert env.quota.usage("tok-a")["drafts_today"] == 1
    assert not path.exists() and env.store.stages(draft.id, sha) == {}


def test_a_duplicate_delivery_does_nothing(env):
    calls, started, go = [], threading.Event(), threading.Event()

    def analyze(path, api_key=None, on_cost=None, on_step=None):
        calls.append(1)
        started.set()
        go.wait(10)
        return SimpleNamespace(result=RESULT, artifacts=ARTIFACTS, cost_usd=0.0)

    draft, path, _ = cut_off(env)
    runner = env.runner(analyze)
    assert runner.resume(draft.id, "tok-a", path) == "resumed"
    assert started.wait(10)
    assert runner.resume(draft.id, "tok-a", path) == "duplicate"                 # while it runs
    go.set()
    assert runner.wait_idle(10)
    assert runner.resume(draft.id, "tok-a", path) == "duplicate"                 # after it finished
    assert runner.resume(draft.id, "tok-b", path) == "duplicate"                 # somebody else's token
    assert runner.resume("missing", "tok-a", path) == "duplicate"
    assert runner.wait_idle(10) and calls == [1]
    assert env.store.get_draft(draft.id, "tok-a").status == "done"


@pytest.mark.parametrize("damage", ["gone", "changed"])
def test_a_review_whose_video_is_gone_or_changed_is_failed_not_run(env, damage):
    calls = []
    draft, path, _ = cut_off(env, cost=0.03)
    if damage == "gone":
        path.unlink()
    else:
        path.write_bytes(b"another video")
    runner = env.runner(staged(calls, {"extract_v2": 0.03, "judge_L0": 0.01}))
    assert runner.resume(draft.id, "tok-a", path) == "lost"
    got = env.store.get_draft(draft.id, "tok-a")
    assert (got.status, got.fail_code) == ("failed", "interrupted") and calls == [] and not path.exists()
    assert got.cost_usd == pytest.approx(0.03) and env.quota.summary()["settled_usd"] == pytest.approx(0.03)


def test_an_analysis_without_a_checkpoint_parameter_still_runs_and_resumes(env):
    seen = []
    draft, path, _ = cut_off(env)
    runner = env.runner(ok(seen=seen))
    assert runner.resume(draft.id, "tok-a", path) == "resumed" and runner.wait_idle(10)
    assert len(seen) == 1 and env.store.get_draft(draft.id, "tok-a").status == "done"
