"""Tests for src.channel_export using mocked googleapiclient service objects (no network)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httplib2
import pandas as pd
import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from src.channel_export import analytics, auth, data_api, reporting
from src.channel_export.cli import build_parser
from src.channel_export.common import ApiCaller, CallStats, RawStore, execute_with_retry
from src.channel_export.export import Exporter, ExportOptions
from src.channel_export.flatten import (
    combine,
    comments_to_df,
    content_types_by_video,
    parse_iso_duration,
    result_table_to_df,
    videos_to_df,
    write_csv,
)

# ---------------------------------------------------------------- fakes


class FakeRequest:
    """Stands in for a googleapiclient HttpRequest."""

    def __init__(self, response: Any = None, errors: list[Exception] | None = None) -> None:
        self.response = response
        self.errors = list(errors or [])
        self.calls = 0

    def execute(self, num_retries: int = 0) -> Any:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.response


def http_error(
    status: int, reason: str = "badRequest", message: str = "The query is not supported."
) -> HttpError:
    resp = httplib2.Response({"status": status})
    resp.reason = message
    body = {"error": {"code": status, "message": message, "errors": [{"reason": reason, "message": message}]}}
    return HttpError(resp, json.dumps(body).encode())


def no_sleep_caller(tmp_path: Path | None = None) -> ApiCaller:
    return ApiCaller(RawStore(tmp_path / "raw" if tmp_path else None), sleep=lambda _s: None)


# ---------------------------------------------------------------- retries / quota


def test_execute_with_retry_backs_off_on_5xx_and_429() -> None:
    request = FakeRequest(
        {"ok": True}, errors=[http_error(503, "backendError"), http_error(429, "rateLimitExceeded")]
    )
    sleeps: list[float] = []
    assert execute_with_retry(request, sleep=sleeps.append, base_delay=1.0) == {"ok": True}
    assert request.calls == 3
    assert len(sleeps) == 2 and 1.0 <= sleeps[0] < 2.0 and 2.0 <= sleeps[1] < 3.0


def test_execute_with_retry_does_not_retry_400_or_quota() -> None:
    for err in (http_error(400), http_error(403, "quotaExceeded")):
        request = FakeRequest(errors=[err])
        with pytest.raises(HttpError):
            execute_with_retry(request, sleep=lambda _s: None)
        assert request.calls == 1


def test_call_stats_counts_retries_as_quota_units(tmp_path: Path) -> None:
    caller = no_sleep_caller(tmp_path)
    caller.execute(
        "youtube",
        "videos.list",
        FakeRequest({"items": []}, errors=[http_error(500, "backendError")]),
        "x.json",
    )
    caller.execute("analytics", "reports.query", FakeRequest({}), None)
    summary = caller.stats.summary()
    assert summary["data_api_units_used"] == 2
    assert summary["analytics_api_calls"] == 1
    assert (tmp_path / "raw" / "x.json").exists()


# ---------------------------------------------------------------- Data API


def test_list_uploads_follows_page_tokens(tmp_path: Path) -> None:
    pages = {
        None: {"items": [{"contentDetails": {"videoId": f"v{i}"}} for i in range(50)], "nextPageToken": "p2"},
        "p2": {
            "items": [{"contentDetails": {"videoId": f"v{i}"}} for i in range(50, 100)],
            "nextPageToken": "p3",
        },
        "p3": {"items": [{"contentDetails": {"videoId": "v100"}}]},
    }
    youtube = MagicMock()
    youtube.playlistItems.return_value.list.side_effect = lambda **kw: FakeRequest(pages[kw.get("pageToken")])
    items = data_api.list_uploads(youtube, no_sleep_caller(tmp_path), "UUabc")
    ids = data_api.upload_video_ids(items)
    assert ids == [f"v{i}" for i in range(101)]
    calls = youtube.playlistItems.return_value.list.call_args_list
    assert [c.kwargs.get("pageToken") for c in calls] == [None, "p2", "p3"]
    assert all(c.kwargs["maxResults"] == 50 and c.kwargs["playlistId"] == "UUabc" for c in calls)
    assert len(list((tmp_path / "raw" / "data_api" / "playlistItems.list").glob("page_*.json"))) == 3


def test_list_uploads_stops_at_max_videos() -> None:
    pages = {
        None: {"items": [{"contentDetails": {"videoId": f"v{i}"}} for i in range(50)], "nextPageToken": "p2"}
    }
    youtube = MagicMock()
    youtube.playlistItems.return_value.list.side_effect = lambda **kw: FakeRequest(pages[kw.get("pageToken")])
    items = data_api.list_uploads(youtube, no_sleep_caller(), "UUabc", max_videos=7)
    assert len(items) == 7
    assert youtube.playlistItems.return_value.list.call_count == 1


def test_get_videos_batches_ids_in_fifties() -> None:
    youtube = MagicMock()

    def videos_list(**kw: Any) -> FakeRequest:
        return FakeRequest({"items": [{"id": vid} for vid in kw["id"].split(",")]})

    youtube.videos.return_value.list.side_effect = videos_list
    ids = [f"v{i}" for i in range(120)]
    videos, dropped = data_api.get_videos(youtube, no_sleep_caller(), ids)
    calls = youtube.videos.return_value.list.call_args_list
    assert [len(c.kwargs["id"].split(",")) for c in calls] == [50, 50, 20]
    assert all(c.kwargs["part"] == ",".join(data_api.VIDEO_PARTS) for c in calls)
    assert [v["id"] for v in videos] == ids
    assert dropped == []


def test_get_videos_drops_parts_that_error() -> None:
    youtube = MagicMock()

    def videos_list(**kw: Any) -> FakeRequest:
        if "brandPartner" in kw["part"].split(","):
            return FakeRequest(errors=[http_error(400, "invalidPart")])
        return FakeRequest({"items": [{"id": vid} for vid in kw["id"].split(",")]})

    youtube.videos.return_value.list.side_effect = videos_list
    videos, dropped = data_api.get_videos(youtube, no_sleep_caller(), ["a", "b"])
    assert dropped == ["brandPartner"]
    assert [v["id"] for v in videos] == ["a", "b"]
    assert "brandPartner" not in youtube.videos.return_value.list.call_args_list[-1].kwargs["part"]


def _comment(cid: str, text: str, parent: str | None = None) -> dict[str, Any]:
    snippet: dict[str, Any] = {"textDisplay": text, "authorDisplayName": "x", "likeCount": 1}
    if parent:
        snippet["parentId"] = parent
    return {"id": cid, "snippet": snippet}


def test_comments_paginate_and_fetch_missing_replies() -> None:
    thread_pages = {
        None: {
            "items": [
                {
                    "id": "t1",
                    "snippet": {"topLevelComment": _comment("t1", "first"), "totalReplyCount": 3},
                    "replies": {"comments": [_comment("t1.r1", "r1", "t1")]},
                }
            ],
            "nextPageToken": "n2",
        },
        "n2": {
            "items": [
                {"id": "t2", "snippet": {"topLevelComment": _comment("t2", "second"), "totalReplyCount": 0}}
            ]
        },
    }
    reply_pages = {
        None: {
            "items": [_comment("t1.r1", "r1", "t1"), _comment("t1.r2", "r2", "t1")],
            "nextPageToken": "r2",
        },
        "r2": {"items": [_comment("t1.r3", "r3", "t1")]},
    }
    youtube = MagicMock()

    def threads_list(**kw: Any) -> FakeRequest:
        if kw["videoId"] == "disabled":
            return FakeRequest(errors=[http_error(403, "commentsDisabled", "Comments disabled")])
        return FakeRequest(thread_pages[kw.get("pageToken")])

    youtube.commentThreads.return_value.list.side_effect = threads_list
    youtube.comments.return_value.list.side_effect = lambda **kw: FakeRequest(
        reply_pages[kw.get("pageToken")]
    )

    errors: list[dict[str, Any]] = []
    records = data_api.fetch_comments(youtube, no_sleep_caller(), ["vid1", "disabled"], errors)

    assert errors == []  # commentsDisabled is not an error
    thread_calls = youtube.commentThreads.return_value.list.call_args_list
    assert [c.kwargs.get("pageToken") for c in thread_calls if c.kwargs["videoId"] == "vid1"] == [None, "n2"]
    assert all(c.kwargs["maxResults"] == 100 and c.kwargs["part"] == "snippet,replies" for c in thread_calls)
    reply_calls = youtube.comments.return_value.list.call_args_list
    assert [c.kwargs["parentId"] for c in reply_calls] == ["t1", "t1"]  # only t1 needed the full fetch

    df = comments_to_df(records)
    assert len(df) == 5  # 2 top-level + 3 replies
    assert df["is_reply"].sum() == 3
    assert set(df["video_id"]) == {"vid1"}
    assert df.loc[df["comment_id"] == "t1.r3", "thread_id"].item() == "t1"


# ---------------------------------------------------------------- Analytics flattening + pagination


def test_result_table_to_df_and_csv_round_trip(tmp_path: Path) -> None:
    response = {
        "kind": "youtubeAnalytics#resultTable",
        "columnHeaders": [
            {"name": "day", "columnType": "DIMENSION", "dataType": "STRING"},
            {"name": "views", "columnType": "METRIC", "dataType": "INTEGER"},
            {"name": "averageViewPercentage", "columnType": "METRIC", "dataType": "FLOAT"},
        ],
        "rows": [["2026-09-01", 120, 88.5], ["2026-09-02", 7, 101.25]],
    }
    df = result_table_to_df(response)
    assert list(df.columns) == ["day", "views", "averageViewPercentage"]
    assert str(df["views"].dtype) == "Int64" and df["views"].sum() == 127
    assert df["averageViewPercentage"].dtype == "float64"
    assert write_csv(df, tmp_path / "out.csv") == 2
    back = pd.read_csv(tmp_path / "out.csv")
    assert back.to_dict("records")[1] == {"day": "2026-09-02", "views": 7, "averageViewPercentage": 101.25}


def test_result_table_without_rows_keeps_columns() -> None:
    df = result_table_to_df(
        {
            "columnHeaders": [
                {"name": "country", "dataType": "STRING"},
                {"name": "views", "dataType": "INTEGER"},
            ]
        }
    )
    assert df.empty and list(df.columns) == ["country", "views"]


def test_run_query_paginates_top_videos_with_start_index() -> None:
    headers = [{"name": "video", "dataType": "STRING"}, {"name": "views", "dataType": "INTEGER"}]
    rows = [[f"v{i}", 1000 - i] for i in range(250)]
    service = MagicMock()

    def query(**params: Any) -> FakeRequest:
        start = params["startIndex"] - 1
        return FakeRequest({"columnHeaders": headers, "rows": rows[start : start + params["maxResults"]]})

    service.reports.return_value.query.side_effect = query
    spec = analytics.ReportSpec(
        name="top", metrics=("views",), dimensions=("video",), sort="-views", max_results=200, paginate=True
    )
    merged = analytics.run_query(
        service, no_sleep_caller(), spec, date(2025, 1, 1), date(2026, 1, 1), "analytics/top"
    )
    assert len(merged["rows"]) == 250
    calls = service.reports.return_value.query.call_args_list
    assert [c.kwargs["startIndex"] for c in calls] == [1, 201]
    assert all(c.kwargs["sort"] == "-views" and c.kwargs["ids"] == "channel==MINE" for c in calls)


def test_plan_traffic_batches_respects_video_day_limit() -> None:
    end = date(2026, 9, 25)
    videos = [(f"v{i}", date(2026, 1, 1)) for i in range(10)] + [("old", date(2020, 1, 1))]
    batches = analytics.plan_traffic_batches(videos, end, limit=1000)
    for ids, start in batches:
        assert len(ids) * ((end - start).days + 1) <= 1000 or len(ids) == 1
    assert sorted(v for ids, _ in batches for v in ids) == sorted(v for v, _ in videos)


def test_videos_to_df_labels_shorts() -> None:
    metrics = pd.DataFrame(
        {
            "video_id": ["a", "b", "b"],
            "creatorContentType": ["SHORTS", "VIDEO_ON_DEMAND", "LIVE_STREAM"],
            "views": [5, 9, 1],
        }
    )
    types = content_types_by_video(metrics)
    assert types == {"a": "SHORTS", "b": "VIDEO_ON_DEMAND"}
    videos = [
        {"id": "a", "snippet": {"title": "A", "tags": ["x", "y"]}, "contentDetails": {"duration": "PT45S"}},
        {"id": "b", "snippet": {"title": "B"}, "contentDetails": {"duration": "PT1H2M3S"}},
        {"id": "c", "snippet": {"title": "C"}, "contentDetails": {"duration": "P0D"}},
    ]
    df = videos_to_df(videos, types)
    assert df["is_short"].tolist()[:2] == [True, False] and pd.isna(df["is_short"].iloc[2])
    assert df["duration_seconds"].tolist() == [45.0, 3723.0, 0.0]
    assert json.loads(df.loc[0, "snippet.tags"]) == ["x", "y"]
    assert parse_iso_duration("bogus") is None


# ---------------------------------------------------------------- end-to-end export with a failing report


def _analytics_response(params: dict[str, Any]) -> dict[str, Any]:
    dims = params["dimensions"].split(",") if params.get("dimensions") else []
    metrics = params["metrics"].split(",")
    filters = dict(f.split("==", 1) for f in params.get("filters", "").split(";") if f)
    videos = filters.get("video", "v1,v2").split(",")
    samples = {
        "day": ["2026-09-01"],
        "creatorContentType": ["SHORTS"],
        "ageGroup": ["age18-24"],
        "gender": ["male"],
        "country": ["US"],
        "city": ["London"],
        "deviceType": ["MOBILE"],
        "operatingSystem": ["ANDROID"],
        "subscribedStatus": ["UNSUBSCRIBED"],
        "sharingService": ["WHATSAPP"],
        "insightPlaybackLocationType": ["BROWSE"],
        "insightTrafficSourceType": ["SHORTS", "YT_SEARCH"],
        "insightTrafficSourceDetail": ["brawl stars"],
        "elapsedVideoTimeRatio": [0.01, 0.02],
        "video": videos,
    }
    rows: list[list[Any]] = [[]]
    for dim in dims:
        rows = [row + [value] for row in rows for value in samples[dim]]
    rows = [row + [10] * len(metrics) for row in rows]
    headers = [
        {
            "name": d,
            "columnType": "DIMENSION",
            "dataType": "FLOAT" if d == "elapsedVideoTimeRatio" else "STRING",
        }
        for d in dims
    ]
    headers += [{"name": m, "columnType": "METRIC", "dataType": "INTEGER"} for m in metrics]
    return {"kind": "youtubeAnalytics#resultTable", "columnHeaders": headers, "rows": rows}


def _fake_youtube() -> MagicMock:
    youtube = MagicMock()
    channel = {
        "id": "UC123",
        "snippet": {"title": "alfie", "publishedAt": "2024-01-01T00:00:00Z"},
        "contentDetails": {"relatedPlaylists": {"uploads": "UU123"}},
        "statistics": {"viewCount": "100"},
    }
    youtube.channels.return_value.list.return_value = FakeRequest({"items": [channel]})
    items = [
        {"contentDetails": {"videoId": "v1", "videoPublishedAt": "2026-09-01T10:00:00Z"}},
        {"contentDetails": {"videoId": "v2", "videoPublishedAt": "2026-08-01T10:00:00Z"}},
    ]
    youtube.playlistItems.return_value.list.return_value = FakeRequest({"items": items})
    youtube.videos.return_value.list.side_effect = lambda **kw: FakeRequest(
        {
            "items": [
                {
                    "id": v,
                    "snippet": {"title": v, "publishedAt": "2026-08-01T10:00:00Z"},
                    "contentDetails": {"duration": "PT30S"},
                }
                for v in kw["id"].split(",")
            ]
        }
    )

    def threads(**kw: Any) -> FakeRequest:
        if kw["videoId"] == "v2":
            return FakeRequest(errors=[http_error(403, "commentsDisabled", "disabled")])
        return FakeRequest(
            {
                "items": [
                    {"id": "t1", "snippet": {"topLevelComment": _comment("t1", "gg"), "totalReplyCount": 0}}
                ]
            }
        )

    youtube.commentThreads.return_value.list.side_effect = threads
    return youtube


def test_one_failing_report_does_not_stop_the_export(tmp_path: Path) -> None:
    service = MagicMock()
    queries: list[dict[str, Any]] = []

    def query(**params: Any) -> FakeRequest:
        queries.append(params)
        if params.get("dimensions") == "ageGroup,gender":
            return FakeRequest(errors=[http_error(400)])
        return FakeRequest(_analytics_response(params))

    service.reports.return_value.query.side_effect = query
    options = ExportOptions(out_base=tmp_path, until=date(2026, 9, 25))
    exporter = Exporter(
        _fake_youtube(), service, MagicMock(), options, run_dir=tmp_path / "run", sleep=lambda _s: None
    )
    manifest = exporter.run()
    run = tmp_path / "run"

    failed = [e for e in manifest["errors"] if e["step"] == "analytics:demographics"]
    assert (
        len(failed) == 1
        and failed[0]["query"]["dimensions"] == "ageGroup,gender"
        and failed[0]["http_status"] == 400
    )
    assert manifest["error_count"] == 1
    assert all(step["status"] == "ok" for step in manifest["steps"])
    assert manifest["date_range"] == {"start": "2024-01-01", "end": "2026-09-25"}

    demographics = pd.read_csv(run / "demographics.csv")
    assert set(demographics["report"]) == {
        "demographics_by_content_type"
    }  # the failing half is missing, rest kept
    videos = pd.read_csv(run / "videos.csv")
    assert videos["is_short"].tolist() == [True, True]
    assert set(pd.read_csv(run / "video_daily.csv")["video_id"]) == {"v1", "v2"}
    assert len(pd.read_csv(run / "retention.csv")) == 4
    traffic = pd.read_csv(run / "traffic_sources.csv")
    assert set(traffic["insightTrafficSourceType"]) == {"SHORTS", "YT_SEARCH"}
    details = pd.read_csv(run / "traffic_source_details.csv")
    assert {"video_traffic_detail", "channel_traffic_detail_YT_SEARCH"} <= set(details["report"])
    assert len(pd.read_csv(run / "comments.csv")) == 1
    for name in ("channel", "geography", "devices", "video_metrics", "channel_daily", "cities"):
        assert len(pd.read_csv(run / f"{name}.csv")) > 0, name

    saved = json.loads((run / "manifest.json").read_text())
    assert saved["csv_row_counts"]["geography.csv"] == 2
    assert (
        saved["quota"]["data_api_units_used"] == 1 + 1 + 1 + 2
    )  # channels, playlistItems, videos, 2x commentThreads
    assert (run / "raw" / "analytics" / "channel" / "demographics.error.json").exists()
    assert (run / "raw" / "analytics" / "channel" / "geography.json").exists()
    # per-video queries start the day before publication, not at channel creation
    daily = [q for q in queries if q.get("filters") == "video==v1" and q.get("dimensions") == "day"]
    assert daily[0]["startDate"] == "2026-07-31"


# ---------------------------------------------------------------- Reporting API


def test_ensure_jobs_creates_only_missing_available_types() -> None:
    service = MagicMock()
    available = [{"id": t} for t in reporting.DEFAULT_REPORT_TYPES if t != "channel_sharing_service_a2"]
    service.reportTypes.return_value.list.return_value = FakeRequest({"reportTypes": available})
    service.jobs.return_value.list.return_value = FakeRequest(
        {"jobs": [{"id": "j1", "reportTypeId": "channel_basic_a3"}]}
    )
    service.jobs.return_value.create.side_effect = lambda body: FakeRequest(
        {"id": f"job-{body['reportTypeId']}", **body}
    )
    summary = reporting.ensure_jobs(service, no_sleep_caller())
    created = [c["reportTypeId"] for c in summary["created"]]
    assert "channel_reach_basic_a1" in created and "channel_reach_combined_a1" in created
    assert "channel_basic_a3" not in created
    assert summary["unavailable"] == ["channel_sharing_service_a2"]
    assert len(created) == len(reporting.DEFAULT_REPORT_TYPES) - 2


def test_download_all_writes_new_files_and_skips_existing(tmp_path: Path) -> None:
    service = MagicMock()
    service.jobs.return_value.list.return_value = FakeRequest(
        {"jobs": [{"id": "j1", "reportTypeId": "channel_reach_basic_a1"}]}
    )
    reports = [
        {
            "id": "r1",
            "startTime": "2026-09-01T07:00:00Z",
            "endTime": "2026-09-02T07:00:00Z",
            "downloadUrl": "https://x/r1",
        },
        {
            "id": "r2",
            "startTime": "2026-09-02T07:00:00Z",
            "endTime": "2026-09-03T07:00:00Z",
            "downloadUrl": "https://x/r2",
        },
    ]
    service.jobs.return_value.reports.return_value.list.return_value = FakeRequest({"reports": reports})
    existing = tmp_path / "channel_reach_basic_a1" / reporting.report_filename(reports[0])
    existing.parent.mkdir(parents=True)
    existing.write_text("old")
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return b"date,video_id,video_thumbnail_impressions\n"

    summary = reporting.download_all(service, no_sleep_caller(), tmp_path, fetch=fetch)
    assert fetched == ["https://x/r2"]
    assert summary["downloaded"] == 1 and summary["skipped_existing"] == 1
    assert (
        (tmp_path / "channel_reach_basic_a1" / "20260902T070000Z_20260903T070000Z_r2.csv")
        .read_bytes()
        .startswith(b"date,")
    )


# ---------------------------------------------------------------- auth + CLI


def test_refresh_failure_falls_back_to_browser_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"token": "old", "refresh_token": "r", "scopes": list(auth.BASE_SCOPES)}))
    secret = tmp_path / "client_secret.json"
    secret.write_text("{}")

    stale = MagicMock(valid=False, refresh_token="r")
    stale.refresh.side_effect = RefreshError("invalid_grant: Token has been expired or revoked.")
    monkeypatch.setattr(auth.Credentials, "from_authorized_user_file", staticmethod(lambda *a, **k: stale))
    fresh = MagicMock(valid=True)
    fresh.to_json.return_value = json.dumps({"token": "new", "scopes": list(auth.BASE_SCOPES)})
    flow = MagicMock()
    flow.run_local_server.return_value = fresh
    from_secrets = MagicMock(return_value=flow)
    monkeypatch.setattr(auth.InstalledAppFlow, "from_client_secrets_file", from_secrets)

    creds = auth.get_credentials(client_secret=secret, token_path=token, open_browser=False)
    assert creds is fresh
    assert from_secrets.call_args.kwargs["scopes"] == list(auth.BASE_SCOPES)
    assert json.loads(token.read_text())["token"] == "new"


def test_missing_client_secret_gives_setup_error(tmp_path: Path) -> None:
    with pytest.raises(auth.AuthSetupError):
        auth.get_credentials(client_secret=tmp_path / "nope.json", token_path=tmp_path / "token.json")


def test_client_secret_discovery_picks_the_single_desktop_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(auth.CLIENT_SECRET_ENV, raising=False)
    default = tmp_path / "client_secret.json"
    (tmp_path / "client_secret_1-web.apps.googleusercontent.com.json").write_text('{"web": {}}')
    desktop = tmp_path / "client_secret_2-desk.apps.googleusercontent.com.json"
    desktop.write_text('{"installed": {}}')
    assert auth.client_secret_path(default=default) == desktop
    (tmp_path / "client_secret_3-desk.apps.googleusercontent.com.json").write_text('{"installed": {}}')
    with pytest.raises(auth.AuthSetupError, match="Several Desktop-app"):
        auth.client_secret_path(default=default)
    default.write_text('{"installed": {}}')  # an explicit client_secret.json always wins
    assert auth.client_secret_path(default=default) == default


def test_monetary_scope_only_when_requested() -> None:
    assert auth.ANALYTICS_MONETARY_READONLY not in auth.scopes_for(False)
    assert auth.ANALYTICS_MONETARY_READONLY in auth.scopes_for(True)


def test_cli_flags_parse() -> None:
    args = build_parser().parse_args(
        ["--since", "2025-01-01", "--max-videos", "3", "--skip-comments", "--monetary", "-v"]
    )
    assert (
        args.since == date(2025, 1, 1)
        and args.max_videos == 3
        and args.skip_comments
        and args.monetary
        and args.verbose
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--since", "not-a-date"])


def test_call_stats_summary_shape() -> None:
    stats = CallStats()
    stats.record("youtube", "channels.list")
    stats.record("reporting", "jobs.list")
    summary = stats.summary()
    assert summary["data_api_units_used"] == 1 and summary["reporting_api_calls"] == {"jobs.list": 1}


# ---------------------------------------------------------------- video_lifetime batching


def _lifetime_response(params: dict[str, Any]) -> dict[str, Any]:
    """Analytics response for the batched video_lifetime report (video,creatorContentType)."""
    dims = params.get("dimensions", "").split(",")
    metrics = params["metrics"].split(",")
    filters = dict(f.split("==", 1) for f in params.get("filters", "").split(";") if f)
    ids = [v for v in filters.get("video", "").split(",") if v]
    rows: list[list[Any]] = []
    for vid in ids:
        rows.append([vid, "SHORTS"] + [10] * len(metrics))
    headers = [{"name": d, "columnType": "DIMENSION", "dataType": "STRING"} for d in dims]
    headers += [{"name": m, "columnType": "METRIC", "dataType": "INTEGER"} for m in metrics]
    return {"kind": "youtubeAnalytics#resultTable", "columnHeaders": headers, "rows": rows}


def test_video_lifetime_specs_batches_ids() -> None:
    ids = [f"v{i}" for i in range(250)]
    specs = analytics.video_lifetime_specs(ids, max_ids=200)
    assert len(specs) == 2
    assert specs[0].name == "video_lifetime_batch_1"
    assert specs[1].name == "video_lifetime_batch_2"
    assert specs[0].filters == "video==" + ",".join(ids[:200])
    assert specs[1].filters == "video==" + ",".join(ids[200:])
    assert specs[0].dimensions == ("video", "creatorContentType")


def test_watch_with_is_not_requested() -> None:
    assert "WATCH_WITH" not in analytics.TRAFFIC_DETAIL_TYPES
    names = {s.name for s in analytics.channel_report_specs()}
    assert "channel_traffic_detail_WATCH_WITH" not in names


def test_video_lifetime_uses_video_filter(tmp_path: Path) -> None:
    service = MagicMock()
    queries: list[dict[str, Any]] = []

    def query(**params: Any) -> FakeRequest:
        queries.append(params)
        if "video" in params.get("dimensions", "") and "creatorContentType" in params.get("dimensions", ""):
            return FakeRequest(_lifetime_response(params))
        if params.get("dimensions") == "ageGroup,gender":
            return FakeRequest(errors=[http_error(400)])
        return FakeRequest(_analytics_response(params))

    service.reports.return_value.query.side_effect = query
    options = ExportOptions(out_base=tmp_path, until=date(2026, 9, 25))
    exporter = Exporter(
        _fake_youtube(), service, MagicMock(), options, run_dir=tmp_path / "run", sleep=lambda _s: None
    )
    manifest = exporter.run()
    run = tmp_path / "run"

    lifetime_queries = [
        q
        for q in queries
        if "video" in q.get("dimensions", "") and "creatorContentType" in q.get("dimensions", "")
    ]
    assert len(lifetime_queries) == 1
    assert lifetime_queries[0]["filters"] == "video==v1,v2"
    assert manifest["error_count"] == 1  # demographics still fails

    video_metrics = pd.read_csv(run / "video_metrics.csv")
    assert len(video_metrics) == 2
    assert set(video_metrics["video_id"]) == {"v1", "v2"}
    assert set(video_metrics["creatorContentType"]) == {"SHORTS"}

    videos = pd.read_csv(run / "videos.csv")
    assert videos["is_short"].tolist() == [True, True]


def _fake_youtube_with_n(n: int, extra_playlist_ids: list[str] | None = None) -> MagicMock:
    youtube = MagicMock()
    channel = {
        "id": "UC123",
        "snippet": {"title": "alfie", "publishedAt": "2024-01-01T00:00:00Z"},
        "contentDetails": {"relatedPlaylists": {"uploads": "UU123"}},
        "statistics": {"viewCount": "100"},
    }
    youtube.channels.return_value.list.return_value = FakeRequest({"items": [channel]})
    valid_ids = {f"v{i}" for i in range(n)}
    item_ids = list(valid_ids)
    if extra_playlist_ids:
        item_ids += extra_playlist_ids
    items = [
        {"contentDetails": {"videoId": vid, "videoPublishedAt": "2026-08-01T10:00:00Z"}}
        for vid in item_ids
    ]
    youtube.playlistItems.return_value.list.return_value = FakeRequest({"items": items})

    def videos_list(**kw: Any) -> FakeRequest:
        # videos.list silently drops invalid/deleted IDs, so only return valid ones
        ids = [v for v in kw["id"].split(",") if v in valid_ids]
        return FakeRequest(
            {
                "items": [
                    {
                        "id": v,
                        "snippet": {"title": v, "publishedAt": "2026-08-01T10:00:00Z"},
                        "contentDetails": {"duration": "PT30S"},
                    }
                    for v in ids
                ]
            }
        )

    youtube.videos.return_value.list.side_effect = videos_list
    youtube.commentThreads.return_value.list.side_effect = lambda **kw: FakeRequest(
        {"items": [{"id": "t1", "snippet": {"topLevelComment": _comment("t1", "x"), "totalReplyCount": 0}}]}
    )
    return youtube


def test_video_lifetime_batches_large_channel(tmp_path: Path) -> None:
    n = 250
    service = MagicMock()
    queries: list[dict[str, Any]] = []

    def query(**params: Any) -> FakeRequest:
        queries.append(params)
        if "video" in params.get("dimensions", "") and "creatorContentType" in params.get("dimensions", ""):
            return FakeRequest(_lifetime_response(params))
        if params.get("dimensions") == "ageGroup,gender":
            return FakeRequest(errors=[http_error(400)])
        return FakeRequest(_analytics_response(params))

    service.reports.return_value.query.side_effect = query
    options = ExportOptions(out_base=tmp_path, until=date(2026, 9, 25))
    exporter = Exporter(
        _fake_youtube_with_n(n), service, MagicMock(), options, run_dir=tmp_path / "run", sleep=lambda _s: None
    )
    exporter.run()
    run = tmp_path / "run"

    lifetime_queries = [
        q
        for q in queries
        if "video" in q.get("dimensions", "") and "creatorContentType" in q.get("dimensions", "")
    ]
    assert len(lifetime_queries) == 2
    filters_0 = lifetime_queries[0]["filters"].replace("video==", "").split(",")
    filters_1 = lifetime_queries[1]["filters"].replace("video==", "").split(",")
    assert len(filters_0) == 200
    assert len(filters_1) == 50
    assert set(filters_0) & set(filters_1) == set()

    video_metrics = pd.read_csv(run / "video_metrics.csv")
    assert len(video_metrics) == n
    assert set(video_metrics["video_id"]) == {f"v{i}" for i in range(n)}


def test_video_lifetime_skips_deleted_videos(tmp_path: Path) -> None:
    service = MagicMock()
    queries: list[dict[str, Any]] = []

    def query(**params: Any) -> FakeRequest:
        queries.append(params)
        if "video" in params.get("dimensions", "") and "creatorContentType" in params.get("dimensions", ""):
            return FakeRequest(_lifetime_response(params))
        if params.get("dimensions") == "ageGroup,gender":
            return FakeRequest(errors=[http_error(400)])
        return FakeRequest(_analytics_response(params))

    service.reports.return_value.query.side_effect = query
    options = ExportOptions(out_base=tmp_path, until=date(2026, 9, 25))
    # deleted is in uploads playlist but not returned by videos.list
    youtube = _fake_youtube_with_n(2, extra_playlist_ids=["deleted"])
    exporter = Exporter(youtube, service, MagicMock(), options, run_dir=tmp_path / "run", sleep=lambda _s: None)
    exporter.run()
    run = tmp_path / "run"

    lifetime_queries = [
        q
        for q in queries
        if "video" in q.get("dimensions", "") and "creatorContentType" in q.get("dimensions", "")
    ]
    assert len(lifetime_queries) == 1
    assert "deleted" not in lifetime_queries[0]["filters"]
    ids_in_filter = lifetime_queries[0]["filters"].replace("video==", "").split(",")
    assert set(ids_in_filter) == {"v0", "v1"}

    videos = pd.read_csv(run / "videos.csv")
    assert set(videos["video_id"]) == {"v0", "v1"}


def test_video_lifetime_deduplicates_ids(tmp_path: Path) -> None:
    service = MagicMock()
    queries: list[dict[str, Any]] = []

    def query(**params: Any) -> FakeRequest:
        queries.append(params)
        if "video" in params.get("dimensions", "") and "creatorContentType" in params.get("dimensions", ""):
            return FakeRequest(_lifetime_response(params))
        if params.get("dimensions") == "ageGroup,gender":
            return FakeRequest(errors=[http_error(400)])
        return FakeRequest(_analytics_response(params))

    service.reports.return_value.query.side_effect = query
    options = ExportOptions(out_base=tmp_path, until=date(2026, 9, 25))

    youtube = MagicMock()
    channel = {
        "id": "UC123",
        "snippet": {"title": "alfie", "publishedAt": "2024-01-01T00:00:00Z"},
        "contentDetails": {"relatedPlaylists": {"uploads": "UU123"}},
        "statistics": {"viewCount": "100"},
    }
    youtube.channels.return_value.list.return_value = FakeRequest({"items": [channel]})
    # same ID twice in uploads playlist
    items = [
        {"contentDetails": {"videoId": "v1", "videoPublishedAt": "2026-08-01T10:00:00Z"}},
        {"contentDetails": {"videoId": "v1", "videoPublishedAt": "2026-08-01T10:00:00Z"}},
    ]
    youtube.playlistItems.return_value.list.return_value = FakeRequest({"items": items})
    youtube.videos.return_value.list.side_effect = lambda **kw: FakeRequest(
        {
            "items": [
                {
                    "id": v,
                    "snippet": {"title": v, "publishedAt": "2026-08-01T10:00:00Z"},
                    "contentDetails": {"duration": "PT30S"},
                }
                for v in kw["id"].split(",")
            ]
        }
    )
    youtube.commentThreads.return_value.list.side_effect = lambda **kw: FakeRequest(
        {"items": [{"id": "t1", "snippet": {"topLevelComment": _comment("t1", "x"), "totalReplyCount": 0}}]}
    )

    exporter = Exporter(youtube, service, MagicMock(), options, run_dir=tmp_path / "run", sleep=lambda _s: None)
    exporter.run()
    run = tmp_path / "run"

    lifetime_queries = [
        q
        for q in queries
        if "video" in q.get("dimensions", "") and "creatorContentType" in q.get("dimensions", "")
    ]
    assert len(lifetime_queries) == 1
    ids_in_filter = lifetime_queries[0]["filters"].replace("video==", "").split(",")
    assert ids_in_filter == ["v1"]

    video_metrics = pd.read_csv(run / "video_metrics.csv")
    assert len(video_metrics) == 1
    assert video_metrics["video_id"].iloc[0] == "v1"


def test_video_lifetime_missing_analytics_row(tmp_path: Path) -> None:
    service = MagicMock()
    queries: list[dict[str, Any]] = []

    def query(**params: Any) -> FakeRequest:
        queries.append(params)
        dims = params.get("dimensions", "")
        if "video" in dims and "creatorContentType" in dims:
            filters = dict(f.split("==", 1) for f in params.get("filters", "").split(";") if f)
            ids = [v for v in filters.get("video", "").split(",") if v]
            # drop v2 from the lifetime response to simulate missing data
            ids = [v for v in ids if v != "v2"]
            params = {**params, "filters": "video==" + ",".join(ids) if ids else "video=="}
            return FakeRequest(_lifetime_response(params))
        if params.get("dimensions") == "ageGroup,gender":
            return FakeRequest(errors=[http_error(400)])
        return FakeRequest(_analytics_response(params))

    service.reports.return_value.query.side_effect = query
    options = ExportOptions(out_base=tmp_path, until=date(2026, 9, 25))
    exporter = Exporter(
        _fake_youtube(), service, MagicMock(), options, run_dir=tmp_path / "run", sleep=lambda _s: None
    )
    exporter.run()
    run = tmp_path / "run"

    video_metrics = pd.read_csv(run / "video_metrics.csv")
    assert set(video_metrics["video_id"]) == {"v1"}
    assert "v2" not in video_metrics["video_id"].values

    videos = pd.read_csv(run / "videos.csv")
    row = videos.set_index("video_id").loc["v2"]
    assert pd.isna(row["content_type"])
    assert pd.isna(row["is_short"])


def test_combine_fills_missing_content_type() -> None:
    df1 = pd.DataFrame({"video_id": ["a"], "creatorContentType": ["SHORTS"], "views": [10]})
    df2 = pd.DataFrame({"video_id": ["b"], "creatorContentType": [None], "views": [5]})

    # default: aggregate tables use "ALL"
    combined = combine([df1, df2])
    assert combined["creatorContentType"].tolist() == ["SHORTS", "ALL"]

    # video_metrics uses "MISSING"
    combined = combine([df1, df2], missing_content_type="MISSING")
    assert combined["creatorContentType"].tolist() == ["SHORTS", "MISSING"]
