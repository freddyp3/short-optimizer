"""YouTube Reporting API (bulk CSV reports): create reporting jobs and download their files.

A new job only backfills about 30 days of history, and generated files expire after 30-60 days,
so jobs should be created early and downloads run regularly. Downloads go to a persistent
archive folder and already-downloaded report files are skipped.
Docs: https://developers.google.com/youtube/reporting/v1/reports/channel_reports
"""

from __future__ import annotations

import gzip
import io
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from googleapiclient.errors import HttpError

from .common import ApiCaller, describe_error

log = logging.getLogger(__name__)

# Channel report types useful for a Shorts creator (latest versions as of 2026-09-25).
# Reach reports (added 2026-01-15) are the only source of thumbnail impressions / CTR.
DEFAULT_REPORT_TYPES: tuple[str, ...] = (
    "channel_reach_basic_a1",
    "channel_reach_combined_a1",
    "channel_basic_a3",
    "channel_combined_a3",
    "channel_traffic_source_a3",
    "channel_device_os_a3",
    "channel_playback_location_a3",
    "channel_demographics_a1",
    "channel_sharing_service_a2",
)

JOB_NAME_PREFIX = "short-optimizer"


def _paginate(
    caller: ApiCaller, method: str, make_request: Callable[[str | None], Any], key: str, raw_prefix: str
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    token: str | None = None
    page = 0
    while True:
        page += 1
        response = caller.execute(
            "reporting", method, make_request(token), f"{raw_prefix}.page_{page:03d}.json"
        )
        items.extend(response.get(key) or [])
        token = response.get("nextPageToken")
        if not token:
            return items


def list_report_types(reporting: Any, caller: ApiCaller) -> list[dict[str, Any]]:
    """Report types available to this channel (reportTypes.list)."""
    return _paginate(
        caller,
        "reportTypes.list",
        lambda t: reporting.reportTypes().list(**({"pageToken": t} if t else {})),
        "reportTypes",
        "reporting/reportTypes.list",
    )


def list_jobs(reporting: Any, caller: ApiCaller) -> list[dict[str, Any]]:
    """Existing reporting jobs for this channel (jobs.list)."""
    return _paginate(
        caller,
        "jobs.list",
        lambda t: reporting.jobs().list(**({"pageToken": t} if t else {})),
        "jobs",
        "reporting/jobs.list",
    )


def list_reports(reporting: Any, caller: ApiCaller, job_id: str) -> list[dict[str, Any]]:
    """All currently downloadable report files for one job (jobs.reports.list)."""
    return _paginate(
        caller,
        "jobs.reports.list",
        lambda t: reporting.jobs().reports().list(jobId=job_id, **({"pageToken": t} if t else {})),
        "reports",
        f"reporting/jobs.reports.list/{job_id}",
    )


def ensure_jobs(
    reporting: Any, caller: ApiCaller, wanted: Sequence[str] = DEFAULT_REPORT_TYPES
) -> dict[str, Any]:
    """Create a job for each wanted report type that has none yet. Returns a summary dict."""
    summary: dict[str, Any] = {"existing": [], "created": [], "unavailable": [], "errors": []}
    available = {rt.get("id") for rt in list_report_types(reporting, caller)}
    existing = {job.get("reportTypeId"): job for job in list_jobs(reporting, caller)}
    for report_type in wanted:
        if report_type in existing:
            summary["existing"].append(
                {"reportTypeId": report_type, "jobId": existing[report_type].get("id")}
            )
            continue
        if report_type not in available:
            log.warning("Report type %s is not offered for this channel; skipping.", report_type)
            summary["unavailable"].append(report_type)
            continue
        body = {"reportTypeId": report_type, "name": f"{JOB_NAME_PREFIX} {report_type}"}
        try:
            job = caller.execute(
                "reporting",
                "jobs.create",
                reporting.jobs().create(body=body),
                f"reporting/jobs.create/{report_type}.json",
            )
        except HttpError as err:
            log.error("jobs.create failed for %s: %s", report_type, describe_error(err)["message"])
            summary["errors"].append({"reportTypeId": report_type, **describe_error(err)})
            continue
        log.info("Created reporting job %s for %s", job.get("id"), report_type)
        summary["created"].append({"reportTypeId": report_type, "jobId": job.get("id")})
    return summary


def _media_fetcher(reporting: Any) -> Callable[[str], bytes]:
    """Download a report's downloadUrl with the authorized client (pattern from the official sample)."""
    from googleapiclient.http import MediaIoBaseDownload

    def fetch(url: str) -> bytes:
        request = reporting.media().download_media(resourceName=" ")
        request.uri = url
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request, chunksize=-1)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=5)
        return buffer.getvalue()

    return fetch


_TS = re.compile(r"[^0-9TZ]")


def report_filename(report: dict[str, Any]) -> str:
    """Stable file name: <start>_<end>_<reportId>.csv (backfills get a new report id)."""
    start = _TS.sub("", str(report.get("startTime", "")))
    end = _TS.sub("", str(report.get("endTime", "")))
    return f"{start}_{end}_{report.get('id', 'unknown')}.csv"


def download_all(
    reporting: Any,
    caller: ApiCaller,
    dest: Path,
    fetch: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Download every available report file of every job into ``dest/<reportTypeId>/``."""
    fetch = fetch or _media_fetcher(reporting)
    summary: dict[str, Any] = {
        "archive_dir": str(dest),
        "jobs": {},
        "downloaded": 0,
        "skipped_existing": 0,
        "errors": [],
    }
    for job in list_jobs(reporting, caller):
        job_id, report_type = job.get("id"), job.get("reportTypeId") or "unknown"
        job_summary = {"jobId": job_id, "available": 0, "downloaded": 0, "skipped_existing": 0}
        summary["jobs"][report_type] = job_summary
        try:
            reports = list_reports(reporting, caller, job_id)
        except HttpError as err:
            summary["errors"].append({"reportTypeId": report_type, "jobId": job_id, **describe_error(err)})
            log.error("jobs.reports.list failed for %s: %s", report_type, describe_error(err)["message"])
            continue
        job_summary["available"] = len(reports)
        folder = dest / report_type
        for report in reports:
            path = folder / report_filename(report)
            if path.exists():
                job_summary["skipped_existing"] += 1
                continue
            try:
                caller.stats.record("reporting", "media.download")
                data = fetch(report["downloadUrl"])
            except Exception as err:  # noqa: BLE001 - one bad file must not stop the rest
                summary["errors"].append(
                    {"reportTypeId": report_type, "reportId": report.get("id"), **describe_error(err)}
                )
                log.error("Download failed for %s %s: %s", report_type, report.get("id"), err)
                continue
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            job_summary["downloaded"] += 1
        summary["downloaded"] += job_summary["downloaded"]
        summary["skipped_existing"] += job_summary["skipped_existing"]
        log.info(
            "Reporting %s: %d available, %d new, %d already archived",
            report_type,
            job_summary["available"],
            job_summary["downloaded"],
            job_summary["skipped_existing"],
        )
    return summary
