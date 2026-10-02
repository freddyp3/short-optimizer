"""Orchestrates one export run: Data API + Analytics + Reporting, then CSVs and manifest.json.

Every step and every Analytics report runs independently: a failure is logged, recorded in the
manifest (with the exact query), and the export moves on.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from . import __version__, analytics, data_api, reporting
from .common import ApiCaller, RawStore, describe_error, is_quota_exhausted
from .flatten import (
    combine,
    comments_to_df,
    content_types_by_video,
    flatten_resources,
    result_table_to_df,
    videos_to_df,
    write_csv,
)

log = logging.getLogger(__name__)

T = TypeVar("T")

FALLBACK_START = date(2005, 4, 23)  # used only if the channel resource can't be read
FAIL_FAST_AFTER = 5  # stop a per-video report family after this many failures with no success

# CSVs written on every full export (empty if no data), in addition to any extra tables.
CORE_TABLES = (
    "channel",
    "videos",
    "video_metrics",
    "video_daily",
    "retention",
    "traffic_sources",
    "demographics",
    "geography",
    "devices",
)


@dataclass
class ExportOptions:
    """User-facing switches (mirrors the CLI flags)."""

    out_base: Path = Path("data/channel")
    since: date | None = None
    until: date | None = None
    skip_comments: bool = False
    max_videos: int | None = None
    monetary: bool = False
    create_reporting_jobs: bool = False
    download_reports: bool = False
    reporting_only: bool = False


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


class Exporter:
    """Runs the export against already-built googleapiclient service objects."""

    def __init__(
        self,
        youtube: Any,
        analytics_service: Any,
        reporting_service: Any,
        options: ExportOptions,
        *,
        run_dir: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 5,
    ) -> None:
        self.youtube = youtube
        self.analytics = analytics_service
        self.reporting = reporting_service
        self.options = options
        self.started_at = datetime.now(UTC)
        self.run_dir = run_dir or options.out_base / self.started_at.strftime("%Y%m%dT%H%M%SZ")
        self.raw = RawStore(self.run_dir / "raw")
        self.caller = ApiCaller(self.raw, sleep=sleep, max_retries=max_retries)
        self.tables: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self.errors: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.report_stats: dict[str, Counter[str]] = defaultdict(Counter)
        self.info: dict[str, Any] = {}
        self.analytics_quota_exhausted = False
        self.start: date = options.since or FALLBACK_START
        self.end: date = options.until or date.today()

    # ------------------------------------------------------------------ helpers

    def _step(self, name: str, fn: Callable[..., T], *args: Any) -> T | None:
        """Run one step; log and record any exception instead of propagating it."""
        log.info("=== %s ===", name)
        started = time.monotonic()
        status = "ok"
        result: T | None = None
        try:
            result = fn(*args)
        except Exception as err:  # noqa: BLE001 - one failing step must not stop the export
            status = "failed"
            self.errors.append({"step": name, **describe_error(err)})
            log.error(
                "Step '%s' failed: %s",
                name,
                describe_error(err)["message"],
                exc_info=log.isEnabledFor(logging.DEBUG),
            )
        self.steps.append({"name": name, "status": status, "seconds": round(time.monotonic() - started, 2)})
        return result

    def _report(
        self,
        spec: analytics.ReportSpec,
        start: date,
        end: date,
        raw_name: str,
        labels: dict[str, Any] | None = None,
        record_error: bool = True,
    ) -> pd.DataFrame | None:
        """Run one Analytics query; returns a DataFrame, or None (error logged) on failure."""
        try:
            response = analytics.run_query(self.analytics, self.caller, spec, start, end, raw_name)
        except Exception as err:  # noqa: BLE001
            self.report_stats[spec.name]["failed"] += 1
            if is_quota_exhausted(err):
                self.analytics_quota_exhausted = True
            query = spec.params(start, end)
            if record_error:
                self.errors.append({"step": f"analytics:{spec.name}", "query": query, **describe_error(err)})
                log.error("Report %s failed: %s | query=%s", spec.name, describe_error(err)["message"], query)
            else:
                log.warning("Report %s failed (%s); falling back.", spec.name, describe_error(err)["message"])
            return None
        df = result_table_to_df(response)
        if "video" in df.columns:
            df = df.rename(columns={"video": "video_id"})
        inserted = 0
        for key, value in (labels or {}).items():
            if key not in df.columns:
                df.insert(inserted, key, value)
                inserted += 1
        self.report_stats[spec.name]["ok"] += 1
        self.report_stats[spec.name]["rows"] += len(df)
        if spec.table:
            self.tables[spec.table].append(df)
        return df

    def _per_video(
        self,
        family: str,
        video_starts: Sequence[tuple[str, date]],
        make_spec: Callable[[str], analytics.ReportSpec],
    ) -> None:
        """Run one report per video, with progress logs and a fail-fast guard."""
        failures, successes = 0, 0
        total = len(video_starts)
        for n, (video_id, start) in enumerate(video_starts, 1):
            if self.analytics_quota_exhausted:
                log.error("Analytics quota exhausted; skipping remaining %s queries.", family)
                break
            df = self._report(
                make_spec(video_id),
                start,
                self.end,
                f"analytics/{family}/{video_id}",
                labels={"video_id": video_id},
            )
            if df is None:
                failures += 1
                if successes == 0 and failures >= FAIL_FAST_AFTER:
                    msg = f"{family}: first {failures} videos all failed; skipping the rest"
                    self.errors.append({"step": f"analytics:{family}", "message": msg})
                    log.error(msg)
                    break
            else:
                successes += 1
            if n % 25 == 0 or n == total:
                log.info("%s: %d/%d videos", family, n, total)

    # ------------------------------------------------------------------ steps

    def _fetch_channel(self) -> dict[str, Any]:
        channel = data_api.get_my_channel(self.youtube, self.caller)
        self.tables["channel"].append(flatten_resources([channel]))
        snippet = channel.get("snippet") or {}
        self.info["channel"] = {
            "id": channel.get("id"),
            "title": snippet.get("title"),
            "customUrl": snippet.get("customUrl"),
            "publishedAt": snippet.get("publishedAt"),
        }
        log.info("Channel: %s (%s)", snippet.get("title"), channel.get("id"))
        return channel

    def _video_starts(
        self, playlist_items: Sequence[dict[str, Any]], videos: Sequence[dict[str, Any]]
    ) -> list[tuple[str, date]]:
        """(video_id, first date to query) for each upload, from its publish date (minus a day for PT)."""
        published: dict[str, date | None] = {}
        for video in videos:
            published[video["id"]] = _parse_date((video.get("snippet") or {}).get("publishedAt"))
        starts: list[tuple[str, date]] = []
        for item in playlist_items:
            details = item.get("contentDetails") or {}
            video_id = details.get("videoId")
            if not video_id:
                continue
            when = (
                published.get(video_id)
                or _parse_date(details.get("videoPublishedAt"))
                or _parse_date((item.get("snippet") or {}).get("publishedAt"))
            )
            start = max(self.start, (when - timedelta(days=1)) if when else self.start)
            if start <= self.end:
                starts.append((video_id, start))
        return starts

    def _traffic_sources(self, video_starts: Sequence[tuple[str, date]]) -> None:
        """Batched traffic-source report (falls back per video), then per-video referrer details."""
        start_of = dict(video_starts)
        batches = analytics.plan_traffic_batches(video_starts, self.end)
        log.info("Traffic sources: %d videos in %d batched queries", len(video_starts), len(batches))
        frames: list[pd.DataFrame] = []
        for i, (ids, batch_start) in enumerate(batches, 1):
            if self.analytics_quota_exhausted:
                break
            df = self._report(
                analytics.traffic_batch_spec(ids),
                batch_start,
                self.end,
                f"analytics/traffic_sources/batch_{i:03d}",
                record_error=False,
            )
            if df is not None:
                frames.append(df)
                continue
            if self.analytics_quota_exhausted:
                break
            for video_id in ids:
                single = self._report(
                    analytics.traffic_single_spec(video_id),
                    start_of[video_id],
                    self.end,
                    f"analytics/traffic_sources/{video_id}",
                    labels={"video_id": video_id},
                )
                if single is not None:
                    frames.append(single)
        detail_targets: list[tuple[str, str]] = []
        for df in frames:
            if df.empty or "insightTrafficSourceType" not in df:
                continue
            hits = df[
                (df["views"].fillna(0) > 0)
                & df["insightTrafficSourceType"].isin(analytics.TRAFFIC_DETAIL_TYPES)
            ]
            detail_targets += list(
                zip(hits["video_id"].astype(str), hits["insightTrafficSourceType"].astype(str), strict=True)
            )
        log.info("Traffic source details: %d (video, source) queries", len(detail_targets))
        for n, (video_id, source) in enumerate(detail_targets, 1):
            if self.analytics_quota_exhausted:
                break
            self._report(
                analytics.traffic_detail_spec(video_id, source),
                start_of.get(video_id, self.start),
                self.end,
                f"analytics/traffic_detail/{video_id}.{source}",
                labels={
                    "report": "video_traffic_detail",
                    "video_id": video_id,
                    "insightTrafficSourceType": source,
                },
            )
            if n % 25 == 0 or n == len(detail_targets):
                log.info("Traffic source details: %d/%d", n, len(detail_targets))

    def _comments(self, video_ids: Sequence[str]) -> None:
        errors: list[dict[str, Any]] = []
        records = data_api.fetch_comments(self.youtube, self.caller, video_ids, errors)
        self.errors.extend(errors)
        self.tables["comments"].append(comments_to_df(records))

    def _export_all(self) -> None:
        opts = self.options
        channel = self._step("channel", self._fetch_channel)
        if opts.since is None and channel:
            self.start = _parse_date((channel.get("snippet") or {}).get("publishedAt")) or FALLBACK_START
        if self.start > self.end:
            raise ValueError(f"Start date {self.start} is after end date {self.end}")
        self.info["date_range"] = {"start": self.start.isoformat(), "end": self.end.isoformat()}
        log.info("Analytics date range: %s .. %s", self.start, self.end)

        playlist_items: list[dict[str, Any]] = []
        if channel:
            playlist_items = (
                self._step(
                    "uploads",
                    data_api.list_uploads,
                    self.youtube,
                    self.caller,
                    data_api.uploads_playlist_id(channel),
                    opts.max_videos,
                )
                or []
            )
        video_ids = data_api.upload_video_ids(playlist_items)
        fetched = self._step("videos", data_api.get_videos, self.youtube, self.caller, video_ids)
        videos, dropped_parts = fetched if fetched else ([], [])
        self.info["videos"] = {
            "uploads_found": len(video_ids),
            "exported": len(videos),
            "dropped_video_parts": dropped_parts,
        }

        log.info("=== analytics: per-video lifetime metrics ===")
        lifetime = self._report(
            analytics.video_lifetime_spec(), self.start, self.end, "analytics/video_lifetime"
        )
        if opts.monetary:
            self._report(analytics.video_revenue_spec(), self.start, self.end, "analytics/video_revenue")
        content_types = content_types_by_video(lifetime) if lifetime is not None else {}
        self.tables["videos"].append(videos_to_df(videos, content_types))

        log.info("=== analytics: channel-level reports ===")
        for spec in analytics.channel_report_specs(opts.monetary):
            labels: dict[str, Any] = {"report": spec.name}
            if spec.filters and spec.filters.startswith("insightTrafficSourceType=="):
                labels.update({"video_id": None, "insightTrafficSourceType": spec.filters.split("==", 1)[1]})
            self._report(spec, self.start, self.end, f"analytics/channel/{spec.name}", labels=labels)

        video_starts = self._video_starts(playlist_items, videos)
        self._step("video_daily", self._per_video, "video_daily", video_starts, analytics.video_daily_spec)
        self._step("retention", self._per_video, "retention", video_starts, analytics.retention_spec)
        self._step("traffic_sources", self._traffic_sources, video_starts)
        if opts.skip_comments:
            log.info("Skipping comments (--skip-comments)")
        else:
            self._step("comments", self._comments, video_ids)

    def _create_jobs(self) -> None:
        self.info["reporting_jobs"] = reporting.ensure_jobs(self.reporting, self.caller)

    def _download_reports(self) -> None:
        summary = reporting.download_all(self.reporting, self.caller, self.options.out_base / "reporting")
        self.errors.extend({"step": "reporting_download", **e} for e in summary.get("errors", []))
        self.info["reporting_downloads"] = summary

    # ------------------------------------------------------------------ output

    def _write_tables(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        names = list(
            dict.fromkeys(([] if self.options.reporting_only else list(CORE_TABLES)) + list(self.tables))
        )
        for name in names:
            df = combine(self.tables.get(name, []))
            counts[f"{name}.csv"] = write_csv(df, self.run_dir / f"{name}.csv")
        return counts

    def run(self) -> dict[str, Any]:
        """Execute the export and return the manifest (also written to manifest.json)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log.info("Writing to %s", self.run_dir)
        if not self.options.reporting_only:
            self._step("export", self._export_all)
        if self.options.create_reporting_jobs:
            self._step("reporting_create_jobs", self._create_jobs)
        if self.options.download_reports:
            self._step("reporting_download", self._download_reports)
        csv_counts = self._write_tables()
        finished = datetime.now(UTC)
        options = {k: (str(v) if isinstance(v, (Path, date)) else v) for k, v in asdict(self.options).items()}
        manifest: dict[str, Any] = {
            "tool": "src.channel_export",
            "version": __version__,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_seconds": round((finished - self.started_at).total_seconds(), 2),
            "output_dir": str(self.run_dir),
            "options": options,
            **self.info,
            "steps": self.steps,
            "analytics_reports": {k: dict(v) for k, v in sorted(self.report_stats.items())},
            "csv_row_counts": csv_counts,
            "error_count": len(self.errors),
            "errors": self.errors,
            "quota": self.caller.stats.summary(),
            "raw_files_written": self.raw.files_written,
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), "utf-8")
        return manifest
