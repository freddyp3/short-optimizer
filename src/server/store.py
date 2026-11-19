"""Draft records of the Claude MCP server (plan/10a section 3; rules in plan/10 section 12.4).

One SQLite file (WAL) shared with ``auth`` and ``quota``. Every operation opens its own connection, so the
store can be called from any thread; multi-step operations run inside ``BEGIN IMMEDIATE``.

A draft goes ``waiting_for_upload`` -> ``processing`` -> ``done`` | ``failed``, or to ``expired``. The upload
link is a one-time capability: ``claim_upload`` marks it used (``claimed_at``) without changing the status,
``release_upload`` re-opens it after a refused file, ``start_processing`` needs a claimed draft.

A review is keyed on the hash of its video (``video_sha256``, set by ``start_processing``). A review cut off by
a restart is delivered again (``redeliver_interrupted``) instead of failed, and what its finished stages produced
is kept in ``review_stages`` under that key, so a retry never repeats a stage that already ran.

The store never touches files. ``artifacts`` is JSON (any file reference in it must be relative to the server's
data directory) and ``purge`` reports what it removed so the caller can delete the matching files.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src.server import config

WAITING, PROCESSING, DONE, FAILED, EXPIRED = "waiting_for_upload", "processing", "done", "failed", "expired"
STATUSES = (WAITING, PROCESSING, DONE, FAILED, EXPIRED)
PAID_BY = ("server", "own_key")
JOB_STALE_S = 3600
MAX_STEP_CHARS, MAX_CODE_CHARS, MAX_MESSAGE_CHARS = 40, 40, 500
BUSY_TIMEOUT_S = 10.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY,
    token_id TEXT NOT NULL,
    upload_id TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    upload_expires_at REAL NOT NULL,
    claimed_at REAL,
    status TEXT NOT NULL,
    step TEXT,
    fail_code TEXT,
    fail_message TEXT,
    paid_by TEXT,
    cost_usd REAL NOT NULL DEFAULT 0,
    result_json TEXT,
    artifacts_json TEXT,
    started_at REAL,
    finished_at REAL,
    video_sha256 TEXT,
    attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS drafts_token_status ON drafts (token_id, status);
CREATE TABLE IF NOT EXISTS review_stages (
    video_sha256 TEXT NOT NULL,
    draft_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY (video_sha256, draft_id, stage)
);
"""
_COLUMNS = ("id, token_id, upload_id, created_at, upload_expires_at, claimed_at, status, step, fail_code, "
            "fail_message, paid_by, cost_usd, result_json, artifacts_json, started_at, finished_at, video_sha256, attempts")
MSG_INTERRUPTED = "The analysis was interrupted. Please start a new draft."


class UploadUnavailable(Exception):
    """The upload link is unknown, already used or expired (deliberately not said which)."""


class DraftStateError(RuntimeError):
    """The draft is not in the state the operation needs."""


@dataclass(frozen=True)
class Draft:
    id: str
    token_id: str
    upload_id: str
    created_at: float
    upload_expires_at: float
    claimed_at: float | None
    status: str
    step: str | None
    fail_code: str | None
    fail_message: str | None
    paid_by: str | None
    cost_usd: float
    result: dict | None
    artifacts: dict | None
    started_at: float | None
    finished_at: float | None
    video_sha256: str | None = None
    attempts: int = 0


def connect(db_path: Path | str) -> sqlite3.Connection:
    """A fresh autocommit connection in WAL mode; the file and its directory are private to the server user."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
    con = sqlite3.connect(path, timeout=BUSY_TIMEOUT_S, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


@contextmanager
def transaction(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` ... ``COMMIT``; rolled back on any exception. Writers queue on the busy timeout."""
    con = connect(db_path)
    try:
        con.execute("BEGIN IMMEDIATE")
        yield con
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def init_schema(db_path: Path | str, schema: str) -> None:
    with transaction(db_path) as con:
        for statement in filter(None, (part.strip() for part in schema.split(";"))):
            con.execute(statement)


def _draft(row: tuple) -> Draft:
    values = list(row)
    values[12] = json.loads(values[12]) if values[12] is not None else None
    values[13] = json.loads(values[13]) if values[13] is not None else None
    return Draft(*values)


class Store:
    def __init__(self, db_path: Path | str, clock: Callable[[], float] = time.time,
                 upload_ttl_s: float = config.UPLOAD_TTL_S, result_ttl_s: float = config.RESULT_TTL_S,
                 job_stale_s: float = JOB_STALE_S):
        self.db_path, self.clock = Path(db_path), clock
        self.upload_ttl_s, self.result_ttl_s, self.job_stale_s = upload_ttl_s, result_ttl_s, job_stale_s
        init_schema(self.db_path, SCHEMA)
        with transaction(self.db_path) as con:          # databases created before reviews were keyed on the video
            have = {row[1] for row in con.execute("PRAGMA table_info(drafts)")}
            for column, kind in (("video_sha256", "TEXT"), ("attempts", "INTEGER NOT NULL DEFAULT 0")):
                if column not in have:
                    con.execute(f"ALTER TABLE drafts ADD COLUMN {column} {kind}")

    def _one(self, con: sqlite3.Connection, where: str, args: tuple) -> Draft | None:
        row = con.execute(f"SELECT {_COLUMNS} FROM drafts WHERE {where}", args).fetchone()
        return _draft(row) if row else None

    def _update(self, sql: str, args: tuple) -> bool:
        with transaction(self.db_path) as con:
            return con.execute(sql, args).rowcount == 1

    def create_draft(self, token_id: str) -> Draft:
        """A new draft with a fresh upload link; an older waiting draft of the same token is expired first, unless its
        upload is in flight (claimed), which must not be pulled from under the user."""
        now = self.clock()
        draft_id, upload_id = secrets.token_urlsafe(16), secrets.token_urlsafe(32)
        with transaction(self.db_path) as con:
            con.execute("UPDATE drafts SET status = ?, finished_at = ? WHERE token_id = ? AND status = ? "
                        "AND claimed_at IS NULL", (EXPIRED, now, token_id, WAITING))
            con.execute("INSERT INTO drafts (id, token_id, upload_id, created_at, upload_expires_at, status) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (draft_id, token_id, upload_id, now, now + self.upload_ttl_s, WAITING))
            return self._one(con, "id = ?", (draft_id,))

    def get_draft(self, draft_id: str, token_id: str) -> Draft | None:
        """The draft, or None when it does not exist or belongs to another token (indistinguishable)."""
        if not isinstance(draft_id, str) or not isinstance(token_id, str):
            return None
        now = self.clock()
        with transaction(self.db_path) as con:
            con.execute("UPDATE drafts SET status = ?, finished_at = ? WHERE id = ? AND token_id = ? AND status = ? "
                        "AND claimed_at IS NULL AND upload_expires_at <= ?", (EXPIRED, now, draft_id, token_id, WAITING, now))
            return self._one(con, "id = ? AND token_id = ?", (draft_id, token_id))

    def upload_open(self, upload_id: str) -> bool:
        """Whether ``claim_upload`` would succeed right now (for showing the upload form); claims nothing."""
        if not isinstance(upload_id, str):
            return False
        con = connect(self.db_path)
        try:
            return con.execute("SELECT 1 FROM drafts WHERE upload_id = ? AND status = ? AND claimed_at IS NULL "
                               "AND upload_expires_at > ?", (upload_id, WAITING, self.clock())).fetchone() is not None
        finally:
            con.close()

    def claim_upload(self, upload_id: str) -> Draft:
        """Mark the link used and return its draft. Exactly one caller wins; everyone else gets UploadUnavailable."""
        if not isinstance(upload_id, str):
            raise UploadUnavailable()
        now = self.clock()
        with transaction(self.db_path) as con:
            claimed = con.execute("UPDATE drafts SET claimed_at = ? WHERE upload_id = ? AND status = ? "
                                  "AND claimed_at IS NULL AND upload_expires_at > ?", (now, upload_id, WAITING, now)).rowcount
            if claimed != 1:
                raise UploadUnavailable()
            return self._one(con, "upload_id = ?", (upload_id,))

    def release_upload(self, draft_id: str) -> bool:
        """Re-open the link after a refused file. It keeps its original expiry; False if the draft moved on."""
        return self._update("UPDATE drafts SET claimed_at = NULL WHERE id = ? AND status = ? AND claimed_at IS NOT NULL",
                            (draft_id, WAITING))

    def start_processing(self, draft_id: str, paid_by: str, video_sha256: str | None = None) -> None:
        """Claimed draft -> processing, as the first delivery of its review. ``video_sha256`` is the review's key;
        without it the review cannot be delivered again. Raises DraftStateError if the draft was not claimed or is
        no longer waiting."""
        if paid_by not in PAID_BY:
            raise ValueError(f"paid_by must be one of {PAID_BY}")
        if not self._update("UPDATE drafts SET status = ?, paid_by = ?, started_at = ?, step = NULL, video_sha256 = ?, "
                            "attempts = 1 WHERE id = ? AND status = ? AND claimed_at IS NOT NULL",
                            (PROCESSING, paid_by, self.clock(), video_sha256, draft_id, WAITING)):
            raise DraftStateError("draft is not waiting with a claimed upload")

    def redeliver_interrupted(self, max_attempts: int = config.MAX_JOB_ATTEMPTS) -> tuple[list[Draft], list[str]]:
        """Start-up, before any job runs: every draft a dead process left in ``processing`` is either delivered
        again or failed. Returns ``(drafts to run again, ids that were failed)``.

        Delivered again: a server-paid review with a video hash that has deliveries left; its attempt count goes up
        and its clock restarts. Failed as ``interrupted``: an own-key review (the key was never stored, so nobody
        can run it again), a review without a hash, and one that has used all its deliveries."""
        now = self.clock()
        with transaction(self.db_path) as con:
            rows = con.execute("SELECT id, paid_by, video_sha256, attempts FROM drafts WHERE status = ?", (PROCESSING,)).fetchall()
            again = [i for i, paid_by, sha, attempts in rows if paid_by == "server" and sha and attempts < max_attempts]
            failed = [r[0] for r in rows if r[0] not in again]
            con.executemany("UPDATE drafts SET attempts = attempts + 1, started_at = ?, step = NULL WHERE id = ?",
                            [(now, i) for i in again])
            con.executemany("UPDATE drafts SET status = ?, step = NULL, fail_code = 'interrupted', fail_message = ?, "
                            "finished_at = ? WHERE id = ?", [(FAILED, MSG_INTERRUPTED, now, i) for i in failed])
            return [self._one(con, "id = ?", (i,)) for i in again], failed

    def stages(self, draft_id: str, video_sha256: str) -> dict:
        """What the finished stages of this review produced, by stage name."""
        con = connect(self.db_path)
        try:
            return {stage: json.loads(value) for stage, value in con.execute(
                "SELECT stage, value_json FROM review_stages WHERE video_sha256 = ? AND draft_id = ?", (video_sha256, draft_id))}
        finally:
            con.close()

    def put_stage(self, draft_id: str, video_sha256: str, stage: str, value) -> bool:
        """Keep the output of a finished stage. The first write wins; False if the stage was already kept."""
        value_json = json.dumps(value, allow_nan=False)
        with transaction(self.db_path) as con:
            return con.execute("INSERT OR IGNORE INTO review_stages (video_sha256, draft_id, stage, value_json) "
                               "VALUES (?, ?, ?, ?)", (video_sha256, draft_id, str(stage), value_json)).rowcount == 1

    def drop_stages(self) -> int:
        """Delete the kept stages of every review that is not running any more. Returns how many rows went."""
        with transaction(self.db_path) as con:
            return con.execute("DELETE FROM review_stages WHERE draft_id NOT IN (SELECT id FROM drafts WHERE status = ?)",
                               (PROCESSING,)).rowcount

    def set_step(self, draft_id: str, step: str) -> bool:
        return self._update("UPDATE drafts SET step = ? WHERE id = ? AND status = ?",
                            (str(step)[:MAX_STEP_CHARS], draft_id, PROCESSING))

    def finish(self, draft_id: str, result: dict, artifacts: dict, cost_usd: float) -> bool:
        """processing -> done. False (nothing stored) if the draft is no longer processing."""
        result_json, artifacts_json = json.dumps(result, allow_nan=False), json.dumps(artifacts, allow_nan=False)
        return self._update("UPDATE drafts SET status = ?, step = NULL, result_json = ?, artifacts_json = ?, cost_usd = ?, "
                            "finished_at = ? WHERE id = ? AND status = ?",
                            (DONE, result_json, artifacts_json, float(cost_usd), self.clock(), draft_id, PROCESSING))

    def fail(self, draft_id: str, code: str, message: str, cost_usd: float = 0.0) -> bool:
        """waiting or processing -> failed. ``message`` is shown to the user: the caller must pass a safe sentence."""
        return self._update("UPDATE drafts SET status = ?, step = NULL, fail_code = ?, fail_message = ?, cost_usd = ?, "
                            "finished_at = ? WHERE id = ? AND status IN (?, ?)",
                            (FAILED, str(code)[:MAX_CODE_CHARS], str(message)[:MAX_MESSAGE_CHARS], float(cost_usd),
                             self.clock(), draft_id, WAITING, PROCESSING))

    def running_jobs(self, token_id: str | None = None) -> int:
        con = connect(self.db_path)
        try:
            if token_id is None:
                return con.execute("SELECT COUNT(*) FROM drafts WHERE status = ?", (PROCESSING,)).fetchone()[0]
            return con.execute("SELECT COUNT(*) FROM drafts WHERE status = ? AND token_id = ?",
                               (PROCESSING, token_id)).fetchone()[0]
        finally:
            con.close()

    def fail_interrupted(self, older_than_s: float = 0.0, now: float | None = None) -> list[str]:
        """Fail drafts stuck in processing (a job that died with the server). Call with 0 at start-up, before any
        job runs. Returns their ids so the caller can settle the quota and delete the files."""
        now = self.clock() if now is None else now
        with transaction(self.db_path) as con:
            ids = [r[0] for r in con.execute("SELECT id FROM drafts WHERE status = ? AND started_at <= ?",
                                             (PROCESSING, now - older_than_s))]
            con.executemany("UPDATE drafts SET status = ?, step = NULL, fail_code = 'interrupted', fail_message = ?, "
                            "finished_at = ? WHERE id = ?",
                            [(FAILED, MSG_INTERRUPTED, now, i) for i in ids])
            return ids

    def purge(self, now: float | None = None) -> dict:
        """Expire stale links, fail jobs stuck longer than ``job_stale_s`` and delete rows that finished more than
        ``result_ttl_s`` ago.

        Returns ``{"expired": [draft_id], "interrupted": [draft_id], "deleted": [{"id", "token_id", "artifacts"}]}``.
        A claimed link (an upload in flight) is only expired once it is a full TTL past its expiry.
        """
        now = self.clock() if now is None else now
        with transaction(self.db_path) as con:
            stale = "status = ? AND upload_expires_at <= ? AND (claimed_at IS NULL OR upload_expires_at <= ?)"
            args = (WAITING, now, now - self.upload_ttl_s)
            expired = [r[0] for r in con.execute(f"SELECT id FROM drafts WHERE {stale}", args)]
            con.execute(f"UPDATE drafts SET status = ?, finished_at = ? WHERE {stale}", (EXPIRED, now, *args))
            cutoff = now - self.result_ttl_s
            old = "finished_at IS NOT NULL AND finished_at <= ? AND status IN (?, ?, ?)"
            deleted = [{"id": i, "token_id": t, "artifacts": json.loads(a) if a else {}} for i, t, a in con.execute(
                f"SELECT id, token_id, artifacts_json FROM drafts WHERE {old}", (cutoff, DONE, FAILED, EXPIRED))]
            con.execute(f"DELETE FROM drafts WHERE {old}", (cutoff, DONE, FAILED, EXPIRED))
        return {"expired": expired, "interrupted": self.fail_interrupted(self.job_stale_s, now), "deleted": deleted}
