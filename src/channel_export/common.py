"""Shared plumbing: retrying request execution, call/quota accounting, raw JSON storage."""

from __future__ import annotations

import http.client
import json
import logging
import random
import re
import socket
import ssl
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from googleapiclient.errors import HttpError

log = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
# 403s that are rate limits (retry) as opposed to real permission/quota problems (don't).
RETRYABLE_403_REASONS = frozenset(
    {"rateLimitExceeded", "userRateLimitExceeded", "backendError", "internalError"}
)
QUOTA_EXHAUSTED_REASONS = frozenset({"quotaExceeded", "dailyLimitExceeded"})
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    socket.timeout,
    ssl.SSLError,
    http.client.HTTPException,
)

# YouTube Data API v3 cost per call, from
# https://developers.google.com/youtube/v3/determine_quota_cost (all list calls we use cost 1).
# Every request, including invalid ones and retries, is charged.
DATA_API_COSTS: dict[str, int] = {
    "channels.list": 1,
    "playlistItems.list": 1,
    "videos.list": 1,
    "commentThreads.list": 1,
    "comments.list": 1,
}


def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield consecutive slices of ``items`` with at most ``size`` elements."""
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def http_status(err: BaseException) -> int | None:
    """HTTP status code of an ``HttpError`` (None for other exceptions)."""
    if not isinstance(err, HttpError):
        return None
    try:
        return int(err.resp.status)
    except (AttributeError, TypeError, ValueError):
        return None


def _error_payload(err: HttpError) -> dict[str, Any]:
    content = getattr(err, "content", b"") or b""
    try:
        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        payload = json.loads(text)
    except (ValueError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def error_reason(err: BaseException) -> str | None:
    """First ``reason`` in a Google API error body, e.g. ``quotaExceeded`` or ``commentsDisabled``."""
    if not isinstance(err, HttpError):
        return None
    error = _error_payload(err).get("error")
    if not isinstance(error, dict):
        return None
    for item in error.get("errors") or []:
        if isinstance(item, dict) and item.get("reason"):
            return str(item["reason"])
    status = error.get("status")
    return str(status) if status else None


def error_message(err: BaseException) -> str:
    """Human-readable message for any exception, preferring the API's own message."""
    if isinstance(err, HttpError):
        error = _error_payload(err).get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        reason = getattr(err, "reason", None)
        if reason:
            return str(reason)
    return f"{type(err).__name__}: {err}"


def describe_error(err: BaseException) -> dict[str, Any]:
    """Compact, JSON-serialisable description of an exception for the manifest."""
    return {
        "type": type(err).__name__,
        "http_status": http_status(err),
        "reason": error_reason(err),
        "message": error_message(err),
    }


def is_retryable(err: BaseException) -> bool:
    """True for rate limits, 5xx responses and transient network failures."""
    if isinstance(err, HttpError):
        status = http_status(err)
        if status in RETRYABLE_STATUSES:
            return True
        return status == 403 and error_reason(err) in RETRYABLE_403_REASONS
    return isinstance(err, TRANSIENT_ERRORS)


def is_quota_exhausted(err: BaseException) -> bool:
    """True when the daily project quota is used up (retrying won't help until it resets)."""
    return isinstance(err, HttpError) and error_reason(err) in QUOTA_EXHAUSTED_REASONS


def execute_with_retry(
    request: Any,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 64.0,
    sleep: Callable[[float], None] = time.sleep,
    on_attempt: Callable[[], None] | None = None,
) -> Any:
    """Call ``request.execute()`` with exponential backoff (plus jitter) on 429/5xx/network errors."""
    attempt = 0
    while True:
        if on_attempt is not None:
            on_attempt()
        try:
            return request.execute()
        except Exception as err:  # noqa: BLE001 - filtered by is_retryable below
            if attempt >= max_retries or not is_retryable(err):
                raise
            delay = min(max_delay, base_delay * (2**attempt)) + random.uniform(0, base_delay)
            log.warning(
                "Transient error (%s); retry %d/%d in %.1fs",
                error_message(err),
                attempt + 1,
                max_retries,
                delay,
            )
            sleep(delay)
            attempt += 1


@dataclass
class CallStats:
    """Counts request attempts per API method and derives Data API quota units."""

    calls: Counter[str] = field(default_factory=Counter)

    def record(self, api: str, method: str) -> None:
        """Count one attempt (retries count too; Google charges for them)."""
        self.calls[f"{api}:{method}"] += 1

    def _for(self, api: str) -> dict[str, int]:
        prefix = f"{api}:"
        return {k[len(prefix) :]: v for k, v in self.calls.items() if k.startswith(prefix)}

    @property
    def data_api_units(self) -> int:
        """Estimated YouTube Data API v3 quota units consumed."""
        return sum(DATA_API_COSTS.get(m, 1) * n for m, n in self._for("youtube").items())

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable summary for the manifest."""
        return {
            "data_api_units_used": self.data_api_units,
            "data_api_calls": self._for("youtube"),
            "analytics_api_calls": sum(self._for("analytics").values()),
            "reporting_api_calls": self._for("reporting"),
            "note": (
                "Data API units per https://developers.google.com/youtube/v3/determine_quota_cost "
                "(default 10,000/day). Analytics API: 1 unit per request, separate quota."
            ),
        }


_UNSAFE = re.compile(r"[^A-Za-z0-9._=@-]+")


class RawStore:
    """Writes untouched API responses as pretty JSON under a root folder (no-op if root is None)."""

    def __init__(self, root: Path | None) -> None:
        self.root = root
        self.files_written = 0

    def path_for(self, name: str) -> Path | None:
        """Resolve a slash-separated name into a sanitised path under the root."""
        if self.root is None:
            return None
        parts = [_UNSAFE.sub("_", p) for p in name.split("/") if p]
        return self.root.joinpath(*parts)

    def save(self, name: str, payload: Any) -> Path | None:
        """Serialise ``payload`` to ``<root>/<name>``; returns the path written."""
        path = self.path_for(name)
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), "utf-8")
        self.files_written += 1
        return path


class ApiCaller:
    """Executes googleapiclient requests with retries, counts calls and stores raw responses."""

    def __init__(
        self,
        raw: RawStore | None = None,
        stats: CallStats | None = None,
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.raw = raw or RawStore(None)
        self.stats = stats or CallStats()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.sleep = sleep

    def execute(self, api: str, method: str, request: Any, raw_name: str | None = None) -> Any:
        """Run ``request``; save the JSON response (or the error body) under ``raw_name``."""
        try:
            response = execute_with_retry(
                request,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                sleep=self.sleep,
                on_attempt=lambda: self.stats.record(api, method),
            )
        except HttpError as err:
            if raw_name:
                stem = raw_name[:-5] if raw_name.endswith(".json") else raw_name
                self.raw.save(f"{stem}.error.json", {**describe_error(err), "body": _error_payload(err)})
            raise
        if raw_name:
            self.raw.save(raw_name, response)
        return response


def paginate_tokens(fetch_page: Callable[[str | None, int], dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield pages from a pageToken-style list call until ``nextPageToken`` runs out."""
    token: str | None = None
    page = 0
    while True:
        page += 1
        response = fetch_page(token, page)
        yield response
        token = response.get("nextPageToken")
        if not token:
            return
