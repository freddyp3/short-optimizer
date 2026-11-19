"""Analysis jobs of the Claude MCP server (plan/10a sections 7 and 9).

``JobRunner.submit`` starts one worker thread per accepted upload from a small pool. The limits (jobs in total, jobs
per user) are checked and the draft moved to ``processing`` under the runner's own lock, because the store's count is
not atomic with the start. A job writes its steps and costs as it goes, ends in ``store.finish`` or ``store.fail``,
settles the quota with what was really spent (a server-paid job whose analysis raised is settled at no less than
the estimate, see ``failed_charge``), and deletes the uploaded file in every case.

Each review runs exactly once. Its key is the SHA-256 of the uploaded video (``file_sha256``), stored on the draft
when the job starts. Deliveries are at-least-once: ``submit`` is the first delivery, ``resume`` every later one (a
restart re-delivers the reviews it cut off; a hosted queue such as Cloud Tasks would call it for each retry). A
delivery for a review that is already running or already finished does nothing, and a delivery that does run picks
up the outputs of the stages that finished before (``Checkpoint``), so no paid call is made twice. A review paid
with the user's own key cannot be delivered again, because the key is never stored.

What a user can see of a failure is one of a few fixed sentences chosen by the class of the error; the error's own
text (which may hold a path) is never stored or logged. A user's own Gemini key exists only as an argument of that
one job's function call.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

from src.server import config
from src.server.media import Rejected
from src.server.quota import BudgetExceeded, Quota
from src.server.store import MSG_INTERRUPTED, PROCESSING, Draft, Store

log = logging.getLogger("so.server.jobs")

MSG_BUDGET = "This draft cost more than allowed and was stopped."
MSG_OWN_KEY = "Your Gemini key was rejected or is out of quota."
MSG_FAILED = "The analysis failed. Nothing was charged to your daily limit beyond this attempt."
MSG_USER_BUSY = "Another draft of yours is still being analysed. Wait until it is finished, then upload again."
MSG_SERVER_BUSY = "The server is busy with other drafts right now. Please try again in a minute."
STARTING = "starting"
WARM_WAIT_S = 900.0
KEY_ERROR_CODES = (401, 403, 429)
KEY_ERROR_CLASSES = ("DailyQuotaExceeded", "PermissionDenied", "Unauthenticated", "ResourceExhausted")
KEY_ERROR_MARKERS = ("API_KEY_INVALID", "API key not valid", "API key expired", "PERMISSION_DENIED", "UNAUTHENTICATED",
                     "RESOURCE_EXHAUSTED")


class JobRefused(Exception):
    """The job limits do not allow another job now. ``kind`` is ``user_busy`` or ``server_busy``."""

    def __init__(self, kind: str):
        self.kind, self.user_message = kind, MSG_USER_BUSY if kind == "user_busy" else MSG_SERVER_BUSY
        super().__init__(kind)


class RemoteFailure(Exception):
    """A review that ran somewhere else (the Cloud Run job) and failed there. ``code`` and ``user_message`` were
    chosen there by ``failure``, so the message is one of the fixed, safe sentences."""

    def __init__(self, code: str, user_message: str):
        self.code, self.user_message = str(code), str(user_message)
        super().__init__(self.code)


def _chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    while exc is not None and exc not in out and len(out) < 10:
        out.append(exc)
        exc = exc.__cause__ or exc.__context__
    return out


def key_problem(exc: BaseException) -> bool:
    """Whether the error (or one it was raised from) is Gemini refusing the key: bad key, no permission, no quota."""
    for e in _chain(exc):
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        text = str(e)
        if code in KEY_ERROR_CODES or type(e).__name__ in KEY_ERROR_CLASSES:
            return True
        if any(text.startswith(name + ":") for name in KEY_ERROR_CLASSES) or any(m in text for m in KEY_ERROR_MARKERS):
            return True
    return False


def failure(exc: BaseException, paid_by: str) -> tuple[str, str]:
    """(code, sentence) stored for a failed job. Chosen by error class only; never the error's own text."""
    if isinstance(exc, (Rejected, RemoteFailure)):
        return exc.code, exc.user_message
    if isinstance(exc, BudgetExceeded):
        return "over_budget", MSG_BUDGET
    if paid_by == "own_key" and key_problem(exc):
        return "own_key_rejected", MSG_OWN_KEY
    return "analysis_failed", MSG_FAILED


def file_sha256(path: Path | str) -> str:
    """SHA-256 of the file's bytes: the key of the review of this video."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Checkpoint:
    """The finished stages of one review, kept under its video hash. ``get(stage)`` is the kept output or None;
    ``put(stage, value)`` keeps a JSON value. A stage that cannot be kept is only run again on a retry, so a
    failing ``put`` never fails the job. It also carries the review's identity (``draft_id``, ``video_sha256``)
    and what earlier deliveries recorded as spent, for an analysis that runs the review somewhere else."""

    def __init__(self, store: Store, draft_id: str, video_sha256: str, already_spent: float = 0.0):
        self.store, self.draft_id, self.video_sha256, self.already_spent = store, draft_id, video_sha256, already_spent
        self._kept = store.stages(draft_id, video_sha256)

    def get(self, stage: str):
        return self._kept.get(stage)

    def put(self, stage: str, value) -> None:
        try:
            if self.store.put_stage(self.draft_id, self.video_sha256, stage, value):
                self._kept[stage] = value
        except Exception as inner:
            log.warning("could not keep a stage class=%s", type(inner).__name__)


def _default_analyze(video_path, api_key=None, on_cost=None, on_step=None, checkpoint=None):
    from src.tools.predict_retention import tool

    return tool.analyze_draft(video_path, api_key=api_key, on_cost=on_cost, on_step=on_step, checkpoint=checkpoint)


class JobRunner:
    """``analyze(path, api_key=, on_cost=, on_step=)`` returns an object with ``result``, ``artifacts``, ``cost_usd``
    (default: ``predict_retention.tool.analyze_draft``, imported on first use). If it also takes ``checkpoint=``
    it is handed the review's ``Checkpoint``. ``ready`` is an event set once the library is loaded; until then a
    job reports the step "starting" and waits."""

    def __init__(self, store: Store, quota: Quota, analyze: Callable | None = None,
                 max_total: int = config.MAX_JOBS_TOTAL, max_per_user: int = config.MAX_JOBS_PER_USER,
                 ready: threading.Event | None = None, on_done: Callable[[str, str, float], None] | None = None):
        self.store, self.quota, self.analyze = store, quota, analyze or _default_analyze
        self.max_total, self.max_per_user, self.ready, self.on_done = int(max_total), int(max_per_user), ready, on_done
        self._lock = threading.Lock()
        self._active: dict[str, str] = {}
        self._futures: set[Future] = set()
        self._pool = ThreadPoolExecutor(max_workers=max(1, self.max_total), thread_name_prefix="so-job")
        try:
            self._checkpointed = "checkpoint" in inspect.signature(self.analyze).parameters
        except (TypeError, ValueError):
            self._checkpointed = False

    def active_ids(self) -> set[str]:
        with self._lock:
            return set(self._active)

    def submit(self, draft: Draft, path: Path, own_key: str | None = None, video_sha256: str | None = None) -> None:
        """First delivery: start the analysis of a claimed draft whose file passed validation and whose quota is
        reserved. ``video_sha256`` (default: computed from the file) becomes the review's key.
        Raises JobRefused at the job limits and DraftStateError if the draft cannot start; in both cases nothing
        was started and the caller cleans up (file, reservation, link)."""
        paid_by = "own_key" if own_key else "server"
        if not video_sha256:
            try:
                video_sha256 = file_sha256(path)
            except OSError:
                video_sha256 = None          # no key: the review runs once and cannot be delivered again
        with self._lock:
            mine = sum(1 for t in self._active.values() if t == draft.token_id)
            if len(self._active) >= self.max_total or self.store.running_jobs() >= self.max_total:
                raise JobRefused("server_busy")
            if mine >= self.max_per_user or self.store.running_jobs(draft.token_id) >= self.max_per_user:
                raise JobRefused("user_busy")
            self.store.start_processing(draft.id, paid_by, video_sha256)
            self._active[draft.id] = draft.token_id
            try:
                future = self._pool.submit(self._run, draft.id, draft.token_id, Path(path), own_key, paid_by, video_sha256)
            except RuntimeError:
                del self._active[draft.id]
                self.store.fail(draft.id, "analysis_failed", MSG_FAILED)
                raise JobRefused("server_busy") from None
            self._futures.add(future)
            future.add_done_callback(self._futures.discard)

    def resume(self, draft_id: str, token_id: str, path: Path) -> str:
        """A later delivery of a review (after a restart, or a queue retry). Returns what it did:

        ``duplicate``  the review is running in this process, or is not ``processing`` any more: nothing to do
        ``lost``       the video is gone or is not the video the review is keyed on: the draft is failed
        ``resumed``    the review runs again, starting after its last finished stage

        The job limits do not apply: the review already held its place when it was first delivered."""
        path = Path(path)
        with self._lock:
            if draft_id in self._active:
                return "duplicate"
            draft = self.store.get_draft(draft_id, token_id)
            if draft is None or draft.status != PROCESSING or draft.paid_by != "server":
                return "duplicate"
            try:
                same = bool(draft.video_sha256) and path.is_file() and file_sha256(path) == draft.video_sha256
            except OSError:
                same = False
            if same:
                self._active[draft_id] = token_id
                try:
                    future = self._pool.submit(self._run, draft_id, token_id, path, None, "server", draft.video_sha256,
                                               self.quota.recorded_cost(draft_id))
                except RuntimeError:
                    del self._active[draft_id]
                    same = False
                else:
                    self._futures.add(future)
                    future.add_done_callback(self._futures.discard)
                    return "resumed"
        spent = self.quota.recorded_cost(draft_id)
        self.store.fail(draft_id, "interrupted", MSG_INTERRUPTED, spent)
        self._settle(draft_id, spent)
        try:
            path.unlink(missing_ok=True)
        except OSError as inner:
            log.error("could not delete an uploaded file class=%s", type(inner).__name__)
        return "lost"

    def _run(self, draft_id: str, token_id: str, path: Path, own_key: str | None, paid_by: str,
             video_sha256: str | None = None, already_spent: float = 0.0) -> None:
        spent, outcome, analysed = [float(already_spent)], "failed", False
        extra = {}
        if self._checkpointed and video_sha256 and paid_by == "server":
            try:
                extra["checkpoint"] = Checkpoint(self.store, draft_id, video_sha256, float(already_spent))
            except Exception as inner:
                log.warning("could not read kept stages class=%s", type(inner).__name__)

        def on_cost(usd: float) -> None:
            usd = float(usd)
            if not 0 <= usd < math.inf:
                raise ValueError("invalid cost")
            spent[0] += usd
            self.quota.add_cost(draft_id, usd)

        def on_step(name: str) -> None:
            try:
                self.store.set_step(draft_id, str(name))
            except Exception as inner:
                log.warning("could not record a step class=%s", type(inner).__name__)

        try:
            try:
                if self.ready is not None and not self.ready.is_set():
                    on_step(STARTING)
                    self.ready.wait(WARM_WAIT_S)
                analysis = self.analyze(path, api_key=own_key, on_cost=on_cost, on_step=on_step, **extra)
                analysed = True
            finally:        # the file and the key are gone before the outcome becomes visible
                own_key = None
                try:
                    path.unlink(missing_ok=True)
                except OSError as inner:
                    log.error("could not delete an uploaded file class=%s", type(inner).__name__)
            cost = float(analysis.cost_usd) + float(already_spent)
            spent[0] = max(spent[0], cost) if 0 <= cost < math.inf else spent[0]
            if not isinstance(analysis.result, dict) or not isinstance(analysis.artifacts, dict):
                raise TypeError("analysis returned no result")
            finished = self.store.finish(draft_id, analysis.result, analysis.artifacts, spent[0])
            outcome = "done" if finished else "discarded"
            self._settle(draft_id, spent[0])
        except Exception as exc:
            code, message = failure(exc, paid_by)
            outcome = f"failed:{code}"
            log.warning("job error class=%s", type(exc).__name__)
            try:
                self.store.fail(draft_id, code, message, spent[0])
            except Exception as inner:
                log.error("could not record a failed job class=%s", type(inner).__name__)
            self._settle(draft_id, spent[0] if analysed else self.failed_charge(spent[0], paid_by))
        finally:
            try:
                self.store.drop_stages()
            except Exception as inner:
                log.warning("could not drop kept stages class=%s", type(inner).__name__)
            with self._lock:
                self._active.pop(draft_id, None)
            if self.on_done is not None:
                try:
                    self.on_done(token_id, outcome, spent[0])
                except Exception:
                    pass

    def failed_charge(self, spent: float, paid_by: str) -> float:
        """What a draft whose analysis raised is settled at: for a server-paid draft at least the estimate, because
        a failed job may have made billed calls that reported no cost. An own-key draft costs the server nothing
        and keeps what it reported (informational, never counted against the caps)."""
        return spent if paid_by == "own_key" else max(spent, self.quota.cfg.EST_COST_PER_DRAFT_USD)

    def _settle(self, draft_id: str, usd: float) -> None:
        try:
            self.quota.settle(draft_id, usd)
        except Exception as inner:
            log.error("could not settle a job class=%s", type(inner).__name__)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until every submitted job has ended (tests, shutdown). False on timeout."""
        with self._lock:
            pending = set(self._futures)
        return not wait(pending, timeout=timeout).not_done

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)
