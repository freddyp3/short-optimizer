"""YouTube Analytics API v2 (reports.query) report definitions for an owned channel.

Every dimension/metric/filter combination below was checked on 2026-09-25 against
https://developers.google.com/youtube/analytics/channel_reports (updated 2026-09-14),
/metrics, /dimensions and /revision_history. Notes:

* ``views`` for Shorts counts every start/replay (since 2025-04-30); ``engagedViews`` keeps
  the old definition. Both are requested everywhere they are allowed.
* ``creatorContentType`` is documented only as an optional *dimension* (never as a filter), so
  Shorts splits are produced by adding it as a dimension instead of ``creatorContentType==SHORTS``.
* Card/annotation metrics are omitted: cards and annotations don't exist on Shorts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from .common import ApiCaller

log = logging.getLogger(__name__)

IDS = "channel==MINE"

VIEW_METRICS: tuple[str, ...] = (
    "engagedViews",
    "views",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
)
PREMIUM_METRICS: tuple[str, ...] = ("redViews", "estimatedRedMinutesWatched")
ENGAGEMENT_METRICS: tuple[str, ...] = (
    "likes",
    "dislikes",
    "comments",
    "shares",
    "subscribersGained",
    "subscribersLost",
    "videosAddedToPlaylists",
    "videosRemovedFromPlaylists",
)
# Valid for: Top videos, Time-based, User activity by country (channel_reports).
BASIC_METRICS: tuple[str, ...] = VIEW_METRICS + PREMIUM_METRICS + ENGAGEMENT_METRICS
# "User activity by subscribed status" does not list comments/subscribersGained/Lost.
SUBSCRIBED_STATUS_METRICS: tuple[str, ...] = VIEW_METRICS + (
    "likes",
    "dislikes",
    "shares",
    "videosAddedToPlaylists",
    "videosRemovedFromPlaylists",
)
# Traffic source, traffic source detail, device/OS and playback location reports.
INSIGHT_METRICS: tuple[str, ...] = ("engagedViews", "views", "estimatedMinutesWatched")
RETENTION_METRICS: tuple[str, ...] = (
    "audienceWatchRatio",
    "relativeRetentionPerformance",
    "startedWatching",
    "stoppedWatching",
    "totalSegmentImpressions",
)
# Need the yt-analytics-monetary.readonly scope and a YPP channel (403 otherwise).
MONETARY_METRICS: tuple[str, ...] = (
    "estimatedRevenue",
    "estimatedAdRevenue",
    "grossRevenue",
    "estimatedRedPartnerRevenue",
    "monetizedPlaybacks",
    "playbackBasedCpm",
    "adImpressions",
    "cpm",
)
AD_TYPE_METRICS: tuple[str, ...] = ("grossRevenue", "adImpressions", "cpm")

# insightTrafficSourceDetail is only available for these types. The channel_reports note lists
# VIDEO_REMIXES, NOTIFICATION, END_SCREEN, CAMPAIGN_CARD and NO_LINK_EMBEDDED as unsupported, and
# SHORTS (the swipe feed) has no detail values.
TRAFFIC_DETAIL_TYPES: tuple[str, ...] = (
    "YT_SEARCH",
    "HASHTAGS",
    "SOUND_PAGE",
    "RELATED_VIDEO",
    "YT_CHANNEL",
    "SUBSCRIBER",
    "EXT_URL",
    "YT_OTHER_PAGE",
    "ADVERTISING",
    "WATCH_WITH",
)

TOP_VIDEOS_MAX_RESULTS = 200  # "Top videos" reports: maxResults <= 200 and sort is required
DETAIL_MAX_RESULTS = 25  # traffic source detail: maxResults <= 25 and sort is required
CITY_MAX_RESULTS = 250  # user activity by city: maxResults <= 250 and sort is required
CITY_DATA_START = date(2022, 1, 1)  # city data exists from 2022-01-01
TRAFFIC_VIDEO_DAY_LIMIT = 50_000  # traffic source: (#videos x #days) must not exceed 50,000
MAX_FILTER_IDS = 200  # docs allow 500 ids per filter; 200 keeps request URLs short


@dataclass(frozen=True)
class ReportSpec:
    """One reports.query call (optionally paginated with startIndex/maxResults)."""

    name: str
    metrics: tuple[str, ...]
    dimensions: tuple[str, ...] = ()
    filters: str | None = None
    sort: str | None = None
    max_results: int | None = None
    paginate: bool = False
    table: str | None = None
    min_start: date | None = None

    def params(self, start: date, end: date, start_index: int | None = None) -> dict[str, Any]:
        """Keyword arguments for ``youtubeAnalytics.reports().query``."""
        if self.min_start and start < self.min_start:
            start = self.min_start
        params: dict[str, Any] = {
            "ids": IDS,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "metrics": ",".join(self.metrics),
        }
        if self.dimensions:
            params["dimensions"] = ",".join(self.dimensions)
        if self.filters:
            params["filters"] = self.filters
        if self.sort:
            params["sort"] = self.sort
        if self.max_results:
            params["maxResults"] = self.max_results
        if start_index is not None:
            params["startIndex"] = start_index
        return params


def run_query(
    analytics: Any,
    caller: ApiCaller,
    spec: ReportSpec,
    start: date,
    end: date,
    raw_name: str,
) -> dict[str, Any]:
    """Execute a report, following startIndex pages when ``spec.paginate``; returns one merged table."""
    if not spec.paginate or not spec.max_results:
        params = spec.params(start, end)
        return caller.execute(
            "analytics", "reports.query", analytics.reports().query(**params), f"{raw_name}.json"
        )

    merged: dict[str, Any] | None = None
    rows: list[list[Any]] = []
    seen: set[tuple[Any, ...]] = set()
    n_dims = len(spec.dimensions)
    start_index = 1
    page = 0
    while True:
        page += 1
        params = spec.params(start, end, start_index=start_index)
        response = caller.execute(
            "analytics",
            "reports.query",
            analytics.reports().query(**params),
            f"{raw_name}.page_{page:03d}.json",
        )
        if merged is None:
            merged = {k: v for k, v in response.items() if k != "rows"}
        page_rows = response.get("rows") or []
        new_rows = [r for r in page_rows if tuple(r[:n_dims]) not in seen]
        for r in new_rows:
            seen.add(tuple(r[:n_dims]))
        rows.extend(new_rows)
        # Stop on a short page, or if the API ignored startIndex and repeated rows.
        if len(page_rows) < spec.max_results or not new_rows:
            break
        start_index += spec.max_results
    assert merged is not None
    if rows:
        merged["rows"] = rows
    return merged


def _with_content_type(
    name: str, table: str, metrics: tuple[str, ...], dims: tuple[str, ...], **kw: Any
) -> list[ReportSpec]:
    """A report both for the whole channel and split by creatorContentType (SHORTS, VIDEO_ON_DEMAND...)."""
    return [
        ReportSpec(name=name, metrics=metrics, dimensions=dims, table=table, **kw),
        ReportSpec(
            name=f"{name}_by_content_type",
            metrics=metrics,
            dimensions=dims + ("creatorContentType",),
            table=table,
            **kw,
        ),
    ]


def channel_report_specs(monetary: bool = False) -> list[ReportSpec]:
    """Channel-wide reports (one call each)."""
    specs: list[ReportSpec] = []
    specs += _with_content_type("channel_daily", "channel_daily", BASIC_METRICS, ("day",))
    specs += _with_content_type("demographics", "demographics", ("viewerPercentage",), ("ageGroup", "gender"))
    specs += _with_content_type("geography", "geography", BASIC_METRICS, ("country",))
    specs += _with_content_type("devices", "devices", INSIGHT_METRICS, ("deviceType", "operatingSystem"))
    specs += _with_content_type(
        "subscribed_status", "subscribed_status", SUBSCRIBED_STATUS_METRICS, ("subscribedStatus",)
    )
    specs += _with_content_type("sharing_services", "sharing_services", ("shares",), ("sharingService",))
    specs += _with_content_type(
        "playback_locations", "playback_locations", INSIGHT_METRICS, ("insightPlaybackLocationType",)
    )
    specs += _with_content_type(
        "channel_traffic_sources", "channel_traffic_sources", INSIGHT_METRICS, ("insightTrafficSourceType",)
    )
    for source in TRAFFIC_DETAIL_TYPES:
        specs.append(
            ReportSpec(
                name=f"channel_traffic_detail_{source}",
                metrics=INSIGHT_METRICS,
                dimensions=("insightTrafficSourceDetail",),
                filters=f"insightTrafficSourceType=={source}",
                sort="-views",
                max_results=DETAIL_MAX_RESULTS,
                table="traffic_source_details",
            )
        )
    specs.append(
        ReportSpec(
            name="cities",
            metrics=VIEW_METRICS,
            dimensions=("city", "country"),
            sort="-views",
            max_results=CITY_MAX_RESULTS,
            table="cities",
            min_start=CITY_DATA_START,
        )
    )
    if monetary:
        specs.append(
            ReportSpec(
                name="revenue_daily", metrics=MONETARY_METRICS, dimensions=("day",), table="revenue_daily"
            )
        )
        specs.append(
            ReportSpec(name="ad_types", metrics=AD_TYPE_METRICS, dimensions=("adType",), table="ad_types")
        )
    return specs


def video_lifetime_spec() -> ReportSpec:
    """Per-video totals plus content type ("Top videos" report, paginated 200 rows at a time)."""
    return ReportSpec(
        name="video_lifetime",
        metrics=BASIC_METRICS,
        dimensions=("video", "creatorContentType"),
        sort="-views",
        max_results=TOP_VIDEOS_MAX_RESULTS,
        paginate=True,
        table="video_metrics",
    )


def video_revenue_spec() -> ReportSpec:
    """Per-video monetary totals (only with --monetary)."""
    return ReportSpec(
        name="video_revenue",
        metrics=MONETARY_METRICS,
        dimensions=("video",),
        sort="-estimatedRevenue",
        max_results=TOP_VIDEOS_MAX_RESULTS,
        paginate=True,
        table="video_revenue",
    )


def video_daily_spec(video_id: str) -> ReportSpec:
    """Daily time series for one video ("Time-based" report with a video filter)."""
    return ReportSpec(
        name="video_daily",
        metrics=BASIC_METRICS,
        dimensions=("day",),
        filters=f"video=={video_id}",
        table="video_daily",
    )


def retention_spec(video_id: str) -> ReportSpec:
    """Audience retention curve (100 points) for one video; only single-video filters are allowed."""
    return ReportSpec(
        name="retention",
        metrics=RETENTION_METRICS,
        dimensions=("elapsedVideoTimeRatio",),
        filters=f"video=={video_id}",
        table="retention",
    )


def traffic_batch_spec(video_ids: Sequence[str]) -> ReportSpec:
    """Traffic sources for several videos in one call (multi-value video filter + video dimension)."""
    return ReportSpec(
        name="traffic_sources",
        metrics=INSIGHT_METRICS,
        dimensions=("video", "insightTrafficSourceType"),
        filters="video==" + ",".join(video_ids),
        table="traffic_sources",
    )


def traffic_single_spec(video_id: str) -> ReportSpec:
    """Traffic sources for one video (fallback if a batched call is rejected)."""
    return ReportSpec(
        name="traffic_sources",
        metrics=INSIGHT_METRICS,
        dimensions=("insightTrafficSourceType",),
        filters=f"video=={video_id}",
        table="traffic_sources",
    )


def traffic_detail_spec(video_id: str, source_type: str) -> ReportSpec:
    """Top 25 referrers (search terms, hashtags, sounds, ...) of one traffic source type for one video."""
    return ReportSpec(
        name="video_traffic_detail",
        metrics=INSIGHT_METRICS,
        dimensions=("insightTrafficSourceDetail",),
        filters=f"video=={video_id};insightTrafficSourceType=={source_type}",
        sort="-views",
        max_results=DETAIL_MAX_RESULTS,
        table="traffic_source_details",
    )


def plan_traffic_batches(
    videos: Sequence[tuple[str, date]],
    end: date,
    limit: int = TRAFFIC_VIDEO_DAY_LIMIT,
    max_ids: int = MAX_FILTER_IDS,
) -> list[tuple[list[str], date]]:
    """Group (video_id, start_date) pairs so each batch keeps #videos x #days <= ``limit``.

    Each batch is queried from its oldest start date, so newest videos are grouped together.
    """
    batches: list[tuple[list[str], date]] = []
    current: list[str] = []
    current_start: date | None = None
    for video_id, start in sorted(videos, key=lambda v: v[1], reverse=True):
        start = min(start, end)
        candidate_start = start if current_start is None else min(current_start, start)
        days = (end - candidate_start).days + 1
        if current and ((len(current) + 1) * days > limit or len(current) >= max_ids):
            batches.append((current, current_start))  # type: ignore[arg-type]
            current, candidate_start = [], start
        current.append(video_id)
        current_start = candidate_start
    if current:
        batches.append((current, current_start))  # type: ignore[arg-type]
    return batches
