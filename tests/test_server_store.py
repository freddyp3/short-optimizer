"""Draft store: ownership, the one-time upload link, expiry, state changes and purge (plan/10a section 3)."""

import base64
import sqlite3
import threading

import pytest

from src.server.store import (DONE, EXPIRED, FAILED, PROCESSING, WAITING, DraftStateError, Store, UploadUnavailable,
                              connect)

TTL, KEEP = 1800, 7 * 86400


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return Store(tmp_path / "server.sqlite3", clock, upload_ttl_s=TTL, result_ttl_s=KEEP)


def _processing(store: Store, token_id: str = "tok-a"):
    draft = store.create_draft(token_id)
    store.claim_upload(draft.upload_id)
    store.start_processing(draft.id, "server")
    return draft


def test_create_draft(store, clock):
    draft = store.create_draft("tok-a")
    assert draft.status == WAITING and draft.token_id == "tok-a"
    assert draft.created_at == clock.t and draft.upload_expires_at == clock.t + TTL
    assert len(base64.urlsafe_b64decode(draft.id + "==")) == 16
    assert len(base64.urlsafe_b64decode(draft.upload_id + "=")) == 32
    assert draft.result is None and draft.artifacts is None and draft.cost_usd == 0 and draft.claimed_at is None
    assert store.get_draft(draft.id, "tok-a") == draft


def test_database_uses_wal_and_is_private(store):
    con = connect(store.db_path)
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        con.close()
    assert store.db_path.stat().st_mode & 0o077 == 0


def test_new_draft_expires_the_older_waiting_one(store):
    first = store.create_draft("tok-a")
    other = store.create_draft("tok-b")
    second = store.create_draft("tok-a")
    assert store.get_draft(first.id, "tok-a").status == EXPIRED
    assert store.get_draft(second.id, "tok-a").status == WAITING
    assert store.get_draft(other.id, "tok-b").status == WAITING
    with pytest.raises(UploadUnavailable):
        store.claim_upload(first.upload_id)


def test_new_draft_leaves_running_and_finished_drafts_alone(store):
    running = _processing(store)
    store.create_draft("tok-a")
    assert store.get_draft(running.id, "tok-a").status == PROCESSING


def test_foreign_draft_is_invisible(store):
    draft = store.create_draft("tok-a")
    assert store.get_draft(draft.id, "tok-b") is None
    assert store.get_draft("does-not-exist", "tok-b") is None
    assert store.get_draft(draft.id, "") is None
    assert store.get_draft(None, "tok-a") is None and store.get_draft(draft.id, None) is None
    assert store.get_draft(draft.upload_id, "tok-a") is None, "an upload id is not a draft id"


def test_claim_is_one_time(store, clock):
    draft = store.create_draft("tok-a")
    assert store.upload_open(draft.upload_id)
    claimed = store.claim_upload(draft.upload_id)
    assert claimed.id == draft.id and claimed.claimed_at == clock.t and claimed.status == WAITING
    assert not store.upload_open(draft.upload_id)
    with pytest.raises(UploadUnavailable):
        store.claim_upload(draft.upload_id)


@pytest.mark.parametrize("bad", ["", "unknown", None, 7, "' OR 1=1 --"])
def test_claim_unknown_link(store, bad):
    store.create_draft("tok-a")
    with pytest.raises(UploadUnavailable) as err:
        store.claim_upload(bad)
    assert str(err.value) == ""
    assert not store.upload_open(bad)


def test_draft_id_is_not_an_upload_link(store):
    draft = store.create_draft("tok-a")
    with pytest.raises(UploadUnavailable):
        store.claim_upload(draft.id)


def test_claim_races_have_one_winner(store):
    for _ in range(10):
        draft = store.create_draft("tok-a")
        barrier, outcomes = threading.Barrier(2), []

        def race():
            barrier.wait()
            try:
                outcomes.append(store.claim_upload(draft.upload_id).id)
            except UploadUnavailable:
                outcomes.append(None)

        threads = [threading.Thread(target=race) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert sorted(outcomes, key=str) == sorted([draft.id, None], key=str)


def test_expired_link_cannot_be_claimed(store, clock):
    draft = store.create_draft("tok-a")
    clock.t += TTL - 1
    assert store.upload_open(draft.upload_id)
    clock.t += 1
    assert not store.upload_open(draft.upload_id)
    with pytest.raises(UploadUnavailable):
        store.claim_upload(draft.upload_id)
    assert store.get_draft(draft.id, "tok-a").status == EXPIRED


def test_release_reopens_the_link_until_it_expires(store, clock):
    draft = store.create_draft("tok-a")
    store.claim_upload(draft.upload_id)
    assert store.release_upload(draft.id) is True
    assert store.release_upload(draft.id) is False
    assert store.claim_upload(draft.upload_id).id == draft.id
    assert store.release_upload(draft.id) is True
    clock.t += TTL
    with pytest.raises(UploadUnavailable):
        store.claim_upload(draft.upload_id)


def test_release_does_not_touch_a_running_draft(store):
    draft = _processing(store)
    assert store.release_upload(draft.id) is False
    assert store.get_draft(draft.id, "tok-a").status == PROCESSING
    with pytest.raises(UploadUnavailable):
        store.claim_upload(draft.upload_id)


def test_start_processing_needs_a_claimed_waiting_draft(store):
    draft = store.create_draft("tok-a")
    with pytest.raises(DraftStateError):
        store.start_processing(draft.id, "server")
    store.claim_upload(draft.upload_id)
    with pytest.raises(ValueError):
        store.start_processing(draft.id, "somebody")
    store.start_processing(draft.id, "own_key")
    got = store.get_draft(draft.id, "tok-a")
    assert (got.status, got.paid_by) == (PROCESSING, "own_key")
    with pytest.raises(DraftStateError):
        store.start_processing(draft.id, "server")
    with pytest.raises(DraftStateError):
        store.start_processing("missing", "server")


def test_steps_and_finish(store, clock):
    draft = _processing(store)
    assert store.set_step(draft.id, "describing") is True
    assert store.get_draft(draft.id, "tok-a").step == "describing"
    clock.t += 60
    result, artifacts = {"mid_hold": {"level": "high"}}, {"timeline": ["a"], "embedding": [0.1, 0.2], "duration_s": 12.5}
    assert store.finish(draft.id, result, artifacts, 0.11) is True
    got = store.get_draft(draft.id, "tok-a")
    assert (got.status, got.step, got.cost_usd, got.finished_at) == (DONE, None, 0.11, clock.t)
    assert got.result == result and got.artifacts == artifacts
    assert store.set_step(draft.id, "late") is False
    assert store.finish(draft.id, {}, {}, 9.0) is False and store.fail(draft.id, "x", "y") is False
    assert store.get_draft(draft.id, "tok-a").result == result


def test_fail(store, clock):
    draft = _processing(store)
    store.set_step(draft.id, "judging")
    assert store.fail(draft.id, "analysis_failed", "The analysis failed. Please try again.", 0.05) is True
    got = store.get_draft(draft.id, "tok-a")
    assert (got.status, got.fail_code, got.cost_usd, got.step) == (FAILED, "analysis_failed", 0.05, None)
    assert got.fail_message == "The analysis failed. Please try again." and got.finished_at == clock.t
    assert store.finish(draft.id, {}, {}, 0.0) is False

    waiting = store.create_draft("tok-b")
    assert store.fail(waiting.id, "limit", "x" * 2000) is True
    got = store.get_draft(waiting.id, "tok-b")
    assert got.status == FAILED and len(got.fail_message) == 500 and got.cost_usd == 0


def test_finish_rejects_unserialisable_results(store):
    draft = _processing(store)
    with pytest.raises((TypeError, ValueError)):
        store.finish(draft.id, {"x": object()}, {}, 0.0)
    with pytest.raises(ValueError):
        store.finish(draft.id, {"x": float("nan")}, {}, 0.0)
    assert store.get_draft(draft.id, "tok-a").status == PROCESSING


def test_running_jobs(store):
    assert store.running_jobs() == 0
    a = _processing(store, "tok-a")
    _processing(store, "tok-b")
    store.create_draft("tok-c")
    assert store.running_jobs() == 2
    assert (store.running_jobs("tok-a"), store.running_jobs("tok-c")) == (1, 0)
    store.finish(a.id, {}, {}, 0.0)
    assert (store.running_jobs(), store.running_jobs("tok-a")) == (1, 0)


def test_purge_expires_stale_links(store, clock):
    stale = store.create_draft("tok-a")
    clock.t += TTL
    fresh = store.create_draft("tok-b")
    report = store.purge()
    assert report == {"expired": [stale.id], "interrupted": [], "deleted": []}
    assert store.get_draft(fresh.id, "tok-b").status == WAITING
    assert store.purge()["expired"] == []


def test_purge_spares_an_upload_in_flight(store, clock):
    draft = store.create_draft("tok-a")
    clock.t += TTL - 1
    store.claim_upload(draft.upload_id)
    clock.t += 60
    assert store.purge()["expired"] == []
    store.start_processing(draft.id, "server")
    assert store.get_draft(draft.id, "tok-a").status == PROCESSING

    stuck = store.create_draft("tok-b")
    store.claim_upload(stuck.upload_id)
    clock.t += 2 * TTL
    assert store.purge()["expired"] == [stuck.id]


def test_purge_deletes_old_results_and_reports_them(store, clock):
    done = _processing(store, "tok-a")
    store.finish(done.id, {"r": 1}, {"frames": "drafts/abc/frames"}, 0.1)
    failed = _processing(store, "tok-b")
    store.fail(failed.id, "x", "y")
    expired = store.create_draft("tok-c")
    store.create_draft("tok-c")
    clock.t += KEEP - 1
    recent = _processing(store, "tok-d")
    store.finish(recent.id, {}, {}, 0.0)
    assert store.purge()["deleted"] == []
    clock.t += 1
    report = store.purge()
    assert {d["id"]: d for d in report["deleted"]} == {
        done.id: {"id": done.id, "token_id": "tok-a", "artifacts": {"frames": "drafts/abc/frames"}},
        failed.id: {"id": failed.id, "token_id": "tok-b", "artifacts": {}},
        expired.id: {"id": expired.id, "token_id": "tok-c", "artifacts": {}}}
    assert store.get_draft(done.id, "tok-a") is None and store.get_draft(failed.id, "tok-b") is None
    assert store.get_draft(recent.id, "tok-d").status == DONE
    con = sqlite3.connect(store.db_path)
    try:
        assert done.upload_id not in repr(con.execute("SELECT * FROM drafts").fetchall())
    finally:
        con.close()


def test_purge_takes_an_explicit_time(store, clock):
    draft = store.create_draft("tok-a")
    assert store.purge(now=clock.t + TTL)["expired"] == [draft.id]


def test_stuck_jobs_are_failed(store, clock):
    old = _processing(store, "tok-a")
    clock.t += 3000
    young = _processing(store, "tok-b")
    clock.t += 600
    assert store.purge()["interrupted"] == [old.id]
    got = store.get_draft(old.id, "tok-a")
    assert (got.status, got.fail_code) == (FAILED, "interrupted")
    assert store.running_jobs() == 1
    assert store.fail_interrupted() == [young.id]
    assert store.running_jobs() == 0


def test_two_stores_share_one_file(store, clock):
    other = Store(store.db_path, clock, upload_ttl_s=TTL, result_ttl_s=KEEP)
    draft = store.create_draft("tok-a")
    assert other.get_draft(draft.id, "tok-a") == draft


def test_many_threads_can_write(store):
    errors, ids = [], []

    def work(i: int):
        try:
            for _ in range(5):
                draft = store.create_draft(f"tok-{i}")
                store.claim_upload(draft.upload_id)
                store.start_processing(draft.id, "server")
                store.finish(draft.id, {"i": i}, {}, 0.01)
                ids.append(draft.id)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == [] and len(set(ids)) == 30 and store.running_jobs() == 0


def test_a_new_draft_does_not_expire_an_upload_in_flight(tmp_path):
    from src.server import store as S
    st = S.Store(tmp_path / "db.sqlite3")
    first = st.create_draft("tok")
    st.claim_upload(first.upload_id)
    st.create_draft("tok")
    assert st.get_draft(first.id, "tok").status == S.WAITING
    st.start_processing(first.id, "server")
    assert st.get_draft(first.id, "tok").status == S.PROCESSING


# ------------------------------------------------------------------------------------------- reviews keyed on the video

SHA = "a" * 64


def _review(store: Store, token_id: str, paid_by: str = "server", sha: str | None = SHA):
    draft = store.create_draft(token_id)
    store.claim_upload(draft.upload_id)
    store.start_processing(draft.id, paid_by, sha)
    return store.get_draft(draft.id, token_id)


def test_start_processing_stores_the_video_hash_as_the_first_delivery(store):
    got = _review(store, "tok-a")
    assert (got.video_sha256, got.attempts) == (SHA, 1)
    assert _review(store, "tok-b", sha=None).video_sha256 is None


def test_interrupted_reviews_are_delivered_again_until_their_deliveries_run_out(store, clock):
    server, own, unkeyed = _review(store, "tok-a"), _review(store, "tok-b", "own_key"), _review(store, "tok-c", sha=None)
    waiting = store.create_draft("tok-d")
    clock.t += 50
    again, failed = store.redeliver_interrupted(max_attempts=3)
    assert [d.id for d in again] == [server.id] and sorted(failed) == sorted([own.id, unkeyed.id])
    got = store.get_draft(server.id, "tok-a")
    assert (got.status, got.attempts, got.started_at, got.step) == (PROCESSING, 2, clock.t, None)
    for draft, token in ((own, "tok-b"), (unkeyed, "tok-c")):
        lost = store.get_draft(draft.id, token)
        assert (lost.status, lost.fail_code, lost.finished_at) == (FAILED, "interrupted", clock.t)
    assert store.get_draft(waiting.id, "tok-d").status == WAITING
    assert [d.attempts for d in store.redeliver_interrupted(3)[0]] == [3]
    assert store.redeliver_interrupted(3) == ([], [server.id])                  # third delivery was the last
    assert store.get_draft(server.id, "tok-a").fail_code == "interrupted"
    assert store.redeliver_interrupted(3) == ([], [])


def test_stages_are_kept_once_per_review_and_dropped_when_it_is_over(store):
    a, b = _review(store, "tok-a"), _review(store, "tok-b", sha="b" * 64)
    assert store.stages(a.id, SHA) == {}
    assert store.put_stage(a.id, SHA, "extract_v2", {"status": "ok", "n": 1})
    assert not store.put_stage(a.id, SHA, "extract_v2", {"status": "ok", "n": 2})          # the first write wins
    assert store.put_stage(b.id, "b" * 64, "extract_v2", {"n": 3})
    assert store.stages(a.id, SHA) == {"extract_v2": {"status": "ok", "n": 1}}
    assert store.stages(a.id, "b" * 64) == {} and store.stages(b.id, SHA) == {}             # other video, other draft
    with pytest.raises(ValueError):
        store.put_stage(a.id, SHA, "bad", {"x": float("nan")})
    assert store.drop_stages() == 0
    store.finish(a.id, {"r": 1}, {}, 0.0)
    assert store.drop_stages() == 1
    assert store.stages(a.id, SHA) == {} and store.stages(b.id, "b" * 64) == {"extract_v2": {"n": 3}}


def test_a_database_from_before_video_hashes_gains_the_columns(tmp_path, clock):
    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE drafts (id TEXT PRIMARY KEY, token_id TEXT NOT NULL, upload_id TEXT NOT NULL UNIQUE, "
                "created_at REAL NOT NULL, upload_expires_at REAL NOT NULL, claimed_at REAL, status TEXT NOT NULL, step TEXT, "
                "fail_code TEXT, fail_message TEXT, paid_by TEXT, cost_usd REAL NOT NULL DEFAULT 0, result_json TEXT, "
                "artifacts_json TEXT, started_at REAL, finished_at REAL)")
    con.execute("INSERT INTO drafts (id, token_id, upload_id, created_at, upload_expires_at, status) "
                "VALUES ('d1', 'tok-a', 'u1', 0, 0, 'done')")
    con.commit()
    con.close()
    for _ in range(2):                                                          # opening twice changes nothing more
        old = Store(path, clock).get_draft("d1", "tok-a")
        assert (old.status, old.video_sha256, old.attempts) == (DONE, None, 0)
    assert _review(Store(path, clock), "tok-b").video_sha256 == SHA
