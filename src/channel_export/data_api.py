"""YouTube Data API v3 pulls for the authorized user's own channel (OAuth, mine=True)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from googleapiclient.errors import HttpError

from .common import ApiCaller, chunked, describe_error, error_reason, http_status, is_quota_exhausted

log = logging.getLogger(__name__)

# https://developers.google.com/youtube/v3/docs/channels/list (auditDetails needs an MCN scope and
# contentOwnerDetails is for content partners, so both are left out).
CHANNEL_PARTS: tuple[str, ...] = (
    "id",
    "snippet",
    "statistics",
    "contentDetails",
    "brandingSettings",
    "status",
    "topicDetails",
    "localizations",
)

# https://developers.google.com/youtube/v3/docs/videos/list, checked 2026-09-25. fileDetails,
# processingDetails and suggestions are owner-only; brandPartner was added 2026-07-07.
VIDEO_PARTS: tuple[str, ...] = (
    "id",
    "snippet",
    "contentDetails",
    "statistics",
    "status",
    "topicDetails",
    "recordingDetails",
    "fileDetails",
    "processingDetails",
    "suggestions",
    "liveStreamingDetails",
    "localizations",
    "player",
    "paidProductPlacementDetails",
    "brandPartner",
)

VIDEOS_PER_CALL = 50  # videos.list accepts at most 50 ids per request
PLAYLIST_PAGE_SIZE = 50
COMMENTS_PAGE_SIZE = 100


def get_my_channel(youtube: Any, caller: ApiCaller) -> dict[str, Any]:
    """Fetch the authorized user's channel resource."""
    request = youtube.channels().list(part=",".join(CHANNEL_PARTS), mine=True)
    response = caller.execute("youtube", "channels.list", request, "data_api/channels.list.json")
    items = response.get("items") or []
    if not items:
        raise RuntimeError(
            "No YouTube channel is linked to the account you signed in with. If your channel is a "
            "Brand Account, delete secrets/token.json and sign in again, picking the channel on "
            "the account chooser."
        )
    if len(items) > 1:
        log.warning("Account has %d channels; using the first (%s).", len(items), items[0]["id"])
    return items[0]


def uploads_playlist_id(channel: dict[str, Any]) -> str:
    """ID of the channel's hidden 'uploads' playlist."""
    return channel["contentDetails"]["relatedPlaylists"]["uploads"]


def list_uploads(
    youtube: Any, caller: ApiCaller, playlist_id: str, max_videos: int | None = None
) -> list[dict[str, Any]]:
    """All playlistItems in the uploads playlist (newest first), paginated 50 at a time."""
    items: list[dict[str, Any]] = []
    token: str | None = None
    page = 0
    while True:
        page += 1
        kwargs: dict[str, Any] = {
            "part": "id,snippet,contentDetails,status",
            "playlistId": playlist_id,
            "maxResults": PLAYLIST_PAGE_SIZE,
        }
        if token:
            kwargs["pageToken"] = token
        response = caller.execute(
            "youtube",
            "playlistItems.list",
            youtube.playlistItems().list(**kwargs),
            f"data_api/playlistItems.list/page_{page:04d}.json",
        )
        items.extend(response.get("items") or [])
        log.info("Uploads playlist: %d items after page %d", len(items), page)
        if max_videos is not None and len(items) >= max_videos:
            return items[:max_videos]
        token = response.get("nextPageToken")
        if not token:
            return items


def upload_video_ids(playlist_items: Sequence[dict[str, Any]]) -> list[str]:
    """Video IDs from playlist items, de-duplicated, order preserved."""
    seen: dict[str, None] = {}
    for item in playlist_items:
        vid = (item.get("contentDetails") or {}).get("videoId") or (
            (item.get("snippet") or {}).get("resourceId") or {}
        ).get("videoId")
        if vid:
            seen.setdefault(vid, None)
    return list(seen)


def _videos_request(youtube: Any, parts: Sequence[str], ids: Sequence[str]) -> Any:
    return youtube.videos().list(part=",".join(parts), id=",".join(ids))


def _probe_parts(
    youtube: Any, caller: ApiCaller, sample_id: str, parts: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Try each part alone on one video; return (accepted, rejected)."""
    accepted: list[str] = []
    rejected: list[str] = []
    for part in parts:
        try:
            caller.execute(
                "youtube",
                "videos.list",
                _videos_request(youtube, [part], [sample_id]),
                f"data_api/videos.list/probe_{part}.json",
            )
            accepted.append(part)
        except HttpError as err:
            if is_quota_exhausted(err):
                raise
            log.warning("videos.list part '%s' rejected: %s", part, describe_error(err)["message"])
            rejected.append(part)
    if not accepted:
        raise RuntimeError("videos.list rejected every requested part")
    return accepted, rejected


def get_videos(
    youtube: Any,
    caller: ApiCaller,
    video_ids: Sequence[str],
    parts: Sequence[str] = VIDEO_PARTS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Full video resources in batches of 50. Returns (videos, parts dropped because they errored)."""
    parts = list(parts)
    dropped: list[str] = []
    videos: list[dict[str, Any]] = []
    batches = list(chunked(list(video_ids), VIDEOS_PER_CALL))
    for index, batch in enumerate(batches, 1):
        raw_name = f"data_api/videos.list/batch_{index:04d}.json"
        try:
            response = caller.execute(
                "youtube", "videos.list", _videos_request(youtube, parts, batch), raw_name
            )
        except HttpError as err:
            if index != 1 or http_status(err) not in (400, 403) or is_quota_exhausted(err):
                raise
            log.warning(
                "videos.list failed with all parts (%s); probing parts one by one.", error_reason(err)
            )
            parts, dropped = _probe_parts(youtube, caller, batch[0], parts)
            response = caller.execute(
                "youtube", "videos.list", _videos_request(youtube, parts, batch), raw_name
            )
        videos.extend(response.get("items") or [])
        log.info("videos.list: %d/%d videos", len(videos), len(video_ids))
    return videos, dropped


def list_comment_threads(youtube: Any, caller: ApiCaller, video_id: str) -> list[dict[str, Any]]:
    """All published top-level comment threads (with inline replies) for one video."""
    threads: list[dict[str, Any]] = []
    token: str | None = None
    page = 0
    while True:
        page += 1
        kwargs: dict[str, Any] = {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": COMMENTS_PAGE_SIZE,
            "textFormat": "plainText",
        }
        if token:
            kwargs["pageToken"] = token
        response = caller.execute(
            "youtube",
            "commentThreads.list",
            youtube.commentThreads().list(**kwargs),
            f"data_api/commentThreads.list/{video_id}/page_{page:04d}.json",
        )
        threads.extend(response.get("items") or [])
        token = response.get("nextPageToken")
        if not token:
            return threads


def list_replies(youtube: Any, caller: ApiCaller, parent_id: str) -> list[dict[str, Any]]:
    """All replies to one top-level comment via comments.list(parentId=...)."""
    replies: list[dict[str, Any]] = []
    token: str | None = None
    page = 0
    while True:
        page += 1
        kwargs: dict[str, Any] = {
            "part": "id,snippet",
            "parentId": parent_id,
            "maxResults": COMMENTS_PAGE_SIZE,
            "textFormat": "plainText",
        }
        if token:
            kwargs["pageToken"] = token
        response = caller.execute(
            "youtube",
            "comments.list",
            youtube.comments().list(**kwargs),
            f"data_api/comments.list/{parent_id}/page_{page:04d}.json",
        )
        replies.extend(response.get("items") or [])
        token = response.get("nextPageToken")
        if not token:
            return replies


def fetch_comments(
    youtube: Any,
    caller: ApiCaller,
    video_ids: Sequence[str],
    errors: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Comment threads for every video as ``{"videoId", "thread", "replies"}`` records.

    When a thread's ``totalReplyCount`` is larger than the replies embedded in the thread
    (commentThreads only inlines a few), the full reply list is fetched with comments.list.
    """
    errors = errors if errors is not None else []
    records: list[dict[str, Any]] = []
    for n, video_id in enumerate(video_ids, 1):
        try:
            threads = list_comment_threads(youtube, caller, video_id)
        except HttpError as err:
            reason = error_reason(err)
            if reason in ("commentsDisabled", "videoNotFound"):
                log.info("Comments unavailable for %s (%s)", video_id, reason)
                continue
            errors.append({"step": "comments", "video_id": video_id, **describe_error(err)})
            if is_quota_exhausted(err):
                log.error("Data API quota exhausted; stopping comment export.")
                break
            log.error("commentThreads.list failed for %s: %s", video_id, describe_error(err)["message"])
            continue
        for thread in threads:
            snippet = thread.get("snippet") or {}
            inline = list((thread.get("replies") or {}).get("comments") or [])
            total = int(snippet.get("totalReplyCount") or 0)
            replies = inline
            if total > len(inline):
                parent_id = (snippet.get("topLevelComment") or {}).get("id") or thread.get("id")
                try:
                    replies = list_replies(youtube, caller, parent_id)
                except HttpError as err:
                    errors.append({"step": "comment_replies", "parent_id": parent_id, **describe_error(err)})
                    log.error("comments.list failed for %s: %s", parent_id, describe_error(err)["message"])
            records.append({"videoId": video_id, "thread": thread, "replies": replies})
        if n % 10 == 0 or n == len(video_ids):
            log.info("Comments: %d/%d videos done (%d threads so far)", n, len(video_ids), len(records))
    return records
