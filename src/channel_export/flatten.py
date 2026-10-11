"""Turn raw API responses into flat pandas DataFrames / CSV files."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

NUMERIC_TYPES = {"INTEGER", "FLOAT", "CURRENCY", "DOUBLE", "NUMBER"}

_DURATION = re.compile(
    r"^P(?:(?P<w>\d+)W)?(?:(?P<d>\d+)D)?"
    r"(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)?$"
)


def parse_iso_duration(value: str | None) -> float | None:
    """ISO-8601 duration (``PT1M5S``) to seconds; None if missing or unparseable."""
    if not value:
        return None
    match = _DURATION.match(value)
    if not match:
        return None
    parts = {k: float(v) if v else 0.0 for k, v in match.groupdict().items()}
    return parts["w"] * 604800 + parts["d"] * 86400 + parts["h"] * 3600 + parts["m"] * 60 + parts["s"]


def result_table_to_df(response: Mapping[str, Any]) -> pd.DataFrame:
    """Analytics ``resultTable`` (columnHeaders + rows) to a typed DataFrame.

    Column order follows ``columnHeaders``; a response without ``rows`` (no data) gives an
    empty frame that still has the columns.
    """
    headers = list(response.get("columnHeaders") or [])
    columns = [h["name"] for h in headers]
    df = pd.DataFrame(list(response.get("rows") or []), columns=columns)
    for header in headers:
        name, data_type = header["name"], str(header.get("dataType", "")).upper()
        if data_type == "INTEGER":
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
        elif data_type in NUMERIC_TYPES:
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("float64")
    return df


def _cell(value: Any) -> Any:
    """JSON-encode nested lists/dicts so they survive a CSV round trip."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def flatten_resources(resources: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Nested API resources to one row each, with dotted column names (``snippet.title``)."""
    if not resources:
        return pd.DataFrame()
    df = pd.json_normalize(list(resources), sep=".")
    for column in df.columns:
        if df[column].map(lambda v: isinstance(v, (list, dict))).any():
            df[column] = df[column].map(_cell)
    return df


def content_types_by_video(video_metrics: pd.DataFrame) -> dict[str, str]:
    """Pick each video's creatorContentType (the one with the most views if there are several)."""
    if video_metrics.empty or "creatorContentType" not in video_metrics:
        return {}
    id_col = "video_id" if "video_id" in video_metrics else "video"
    ordered = video_metrics.sort_values("views", ascending=False, na_position="last")
    first = ordered.drop_duplicates(subset=[id_col], keep="first")
    return dict(zip(first[id_col].astype(str), first["creatorContentType"].astype(str), strict=True))


def videos_to_df(videos: Sequence[Mapping[str, Any]], content_types: Mapping[str, str]) -> pd.DataFrame:
    """One row per video with its content type label from YouTube Analytics.

    ``is_short`` is True/False when Analytics reported a creatorContentType for the video, and
    empty when it did not (for example, videos with no views in the date range).
    """
    df = flatten_resources(videos)
    if df.empty:
        return pd.DataFrame(
            columns=["video_id", "title", "published_at", "content_type", "is_short", "duration_seconds"]
        )
    df.insert(0, "video_id", df["id"].astype(str))
    df.insert(1, "title", df.get("snippet.title"))
    df.insert(2, "published_at", df.get("snippet.publishedAt"))
    labels = df["video_id"].map(lambda v: content_types.get(v))
    df.insert(3, "content_type", labels)
    df.insert(
        4,
        "is_short",
        labels.map(lambda v: pd.NA if v is None or pd.isna(v) else v == "SHORTS").astype("boolean"),
    )
    durations = df.get("contentDetails.duration")
    df.insert(
        5,
        "duration_seconds",
        durations.map(parse_iso_duration) if durations is not None else pd.Series([None] * len(df)),
    )
    return df


def _comment_row(
    comment: Mapping[str, Any], *, video_id: str, thread_id: str, is_reply: bool, extra: Mapping[str, Any]
) -> dict[str, Any]:
    snippet = comment.get("snippet") or {}
    return {
        "comment_id": comment.get("id"),
        "video_id": video_id,
        "thread_id": thread_id,
        "parent_id": snippet.get("parentId"),
        "is_reply": is_reply,
        "author_display_name": snippet.get("authorDisplayName"),
        "author_channel_id": (snippet.get("authorChannelId") or {}).get("value"),
        "text_display": snippet.get("textDisplay"),
        "text_original": snippet.get("textOriginal"),
        "like_count": snippet.get("likeCount"),
        "published_at": snippet.get("publishedAt"),
        "updated_at": snippet.get("updatedAt"),
        "viewer_rating": snippet.get("viewerRating"),
        "can_rate": snippet.get("canRate"),
        **extra,
    }


def comments_to_df(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Records from ``data_api.fetch_comments`` to one row per comment (top-level and replies)."""
    rows: list[dict[str, Any]] = []
    for record in records:
        thread = record["thread"]
        snippet = thread.get("snippet") or {}
        video_id = record.get("videoId") or snippet.get("videoId")
        top = snippet.get("topLevelComment") or {}
        rows.append(
            _comment_row(
                top,
                video_id=video_id,
                thread_id=thread.get("id"),
                is_reply=False,
                extra={
                    "total_reply_count": snippet.get("totalReplyCount"),
                    "is_public": snippet.get("isPublic"),
                    "can_reply": snippet.get("canReply"),
                },
            )
        )
        for reply in record.get("replies") or []:
            rows.append(
                _comment_row(
                    reply,
                    video_id=video_id,
                    thread_id=thread.get("id"),
                    is_reply=True,
                    extra={
                        "total_reply_count": None,
                        "is_public": snippet.get("isPublic"),
                        "can_reply": None,
                    },
                )
            )
    return pd.DataFrame(rows)


def combine(frames: Sequence[pd.DataFrame], *, missing_content_type: str = "ALL") -> pd.DataFrame:
    """Concatenate frames from several reports; rows missing creatorContentType are filled in."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "creatorContentType" in df.columns:
        df["creatorContentType"] = df["creatorContentType"].fillna(missing_content_type)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> int:
    """Write a DataFrame to CSV (UTF-8, no index); returns the number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return len(df)
