"""Upload storage and file checks of the Claude MCP server (plan/10a section 5, rules in plan/10 section 12.1).

``save_stream`` / ``save_stream_async`` write an upload to a server-chosen path and cut it off at the size limit,
so an oversized file is never stored. ``validate`` runs ffprobe (no shell, 20 s limit, local files only, only the
mp4/mov and webm demuxers) and refuses everything outside the accepted range before any paid call is made.

``Rejected.user_message`` is a full sentence that can be shown to a chat user; it never contains a path, a file
name or ffprobe output.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass
from pathlib import Path

from src.server import config

FFPROBE = "ffprobe"
FFPROBE_TIMEOUT_S = 20
FFPROBE_DEMUXERS = "mov,matroska"
MAX_FFPROBE_OUTPUT = 1_000_000
MAX_DURATION_MISMATCH_S = 2.0   # container vs video track; audio tracks are not compared
CODES = ("unreadable", "no_video_stream", "several_video_streams", "container_not_allowed", "too_short", "too_long",
         "landscape", "too_large")


class Rejected(Exception):
    """The upload is refused. ``code`` is one of CODES; ``user_message`` is safe to show."""

    def __init__(self, code: str, user_message: str | None = None, cfg: config.Config = config.CONFIG):
        if code not in CODES:
            raise ValueError(f"unknown rejection code {code!r}")
        self.code, self.user_message = code, user_message or default_message(code, cfg)
        super().__init__(self.user_message)


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    width: int      # as displayed, after rotation
    height: int
    container: str
    size_bytes: int


def default_message(code: str, cfg: config.Config = config.CONFIG) -> str:
    low, high = f"{cfg.MIN_DURATION_S:g}", f"{cfg.MAX_DURATION_S:g}"
    containers = ", ".join(cfg.ALLOWED_CONTAINERS)
    return {
        "unreadable": f"The file could not be read as a video. Please export it again and upload one of: {containers}.",
        "no_video_stream": "The file has no video track. Please upload the Short itself, not only its audio.",
        "several_video_streams": "The file has more than one video track. Please export it as a single video and upload it again.",
        "container_not_allowed": f"This file type is not accepted. Please upload one of: {containers}.",
        "too_short": f"The video is shorter than {low} seconds. Drafts must be {low} to {high} seconds long.",
        "too_long": f"The video is longer than {high} seconds. Drafts must be {low} to {high} seconds long.",
        "landscape": "The video is wider than it is tall. Please upload a vertical or square Short.",
        "too_large": f"The file is larger than {cfg.MAX_UPLOAD_BYTES / config.MIB:g} MB. Please export a smaller version.",
    }[code]


def _open_new(dest: Path) -> int:
    return os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)


def _write(fd: int, chunk: bytes, written: int, max_bytes: int, cfg: config.Config) -> int:
    if not isinstance(chunk, (bytes, bytearray, memoryview)):
        raise TypeError("upload chunks must be bytes")
    written += len(chunk)
    if written > max_bytes:
        raise Rejected("too_large", cfg=cfg)
    view = memoryview(chunk)
    while view:
        view = view[os.write(fd, view):]
    return written


def _discard(fd: int, dest: Path) -> None:
    os.close(fd)
    try:
        os.unlink(dest)
    except FileNotFoundError:
        pass


def save_stream(chunks: Iterable[bytes], dest: Path, max_bytes: int = config.MAX_UPLOAD_BYTES,
                cfg: config.Config = config.CONFIG) -> int:
    """Write ``chunks`` to the new file ``dest`` and return its size. ``dest`` must not exist (FileExistsError).

    On crossing ``max_bytes`` the partial file is deleted and Rejected("too_large") raised; the partial file is
    also deleted when the source fails (for example a dropped connection), and that error propagates.
    """
    fd, written = _open_new(Path(dest)), 0
    try:
        for chunk in chunks:
            written = _write(fd, chunk, written, max_bytes, cfg)
    except BaseException:
        _discard(fd, Path(dest))
        raise
    os.close(fd)
    return written


async def save_stream_async(chunks: AsyncIterable[bytes], dest: Path, max_bytes: int = config.MAX_UPLOAD_BYTES,
                            cfg: config.Config = config.CONFIG) -> int:
    """``save_stream`` for an async source (a request body). File writes are small and blocking."""
    fd, written = _open_new(Path(dest)), 0
    try:
        async for chunk in chunks:
            written = _write(fd, chunk, written, max_bytes, cfg)
    except BaseException:
        _discard(fd, Path(dest))
        raise
    os.close(fd)
    return written


def probe(path: Path) -> dict:
    """ffprobe's JSON for a local file. Any failure (missing tool, timeout, error exit, bad output) is
    Rejected("unreadable"). Only the file protocol and the two demuxer families we accept are enabled, so a
    crafted playlist or reference file cannot make ffprobe open other files or the network."""
    cmd = [FFPROBE, "-v", "error", "-protocol_whitelist", "file", "-format_whitelist", FFPROBE_DEMUXERS,
           "-print_format", "json", "-show_format", "-show_streams", "-i", "file:" + str(Path(path).resolve())]
    try:
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              timeout=FFPROBE_TIMEOUT_S, shell=False, check=False,
                              env={"PATH": os.environ.get("PATH", os.defpath)})
        if done.returncode != 0 or len(done.stdout) > MAX_FFPROBE_OUTPUT:
            raise ValueError("ffprobe failed")
        data = json.loads(done.stdout)
        if not isinstance(data, dict):
            raise ValueError("unexpected ffprobe output")
        return data
    except (OSError, subprocess.SubprocessError, ValueError):
        raise Rejected("unreadable") from None


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rotation_degrees(stream: dict) -> float:
    """Display rotation of a video stream: the display-matrix side data, else the older ``rotate`` tag."""
    for side in stream.get("side_data_list") or []:
        if isinstance(side, dict) and _number(side.get("rotation")) is not None:
            return _number(side["rotation"])
    tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
    return _number(tags.get("rotate")) or 0.0


def display_size(stream: dict) -> tuple[int, int] | None:
    """(width, height) as a player shows it: swapped when the stream is rotated by 90 or 270 degrees."""
    width, height = _number(stream.get("width")), _number(stream.get("height"))
    if not width or not height or width < 1 or height < 1:
        return None
    if round(abs(rotation_degrees(stream))) % 180 == 90:
        width, height = height, width
    return int(width), int(height)


def container_of(data: dict, head: bytes) -> str | None:
    """mp4 / mov / webm / mkv / 3gp from ffprobe's format block and the first bytes of the file; None if unknown."""
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
    names = set(str(fmt.get("format_name", "")).split(","))
    if "matroska" in names or "webm" in names:
        return "webm" if b"\x42\x82\x84webm" in head[:64] else "mkv"
    if "mp4" in names or "mov" in names:
        tags = fmt.get("tags") if isinstance(fmt.get("tags"), dict) else {}
        brand = str(tags.get("major_brand", "")).strip().lower()
        return "mov" if brand == "qt" else "3gp" if brand.startswith("3g") else "mp4"
    return None


def inspect(data: dict, head: bytes, size_bytes: int, cfg: config.Config = config.CONFIG) -> MediaInfo:
    """The checks of ``validate`` on already parsed ffprobe output (pure, no file access). The length limits apply
    to the container's and the video track's duration alike (the shorter against the minimum, the longer against
    the maximum), and a file whose two durations disagree by more than MAX_DURATION_MISMATCH_S is refused."""
    streams = [s for s in data.get("streams") or [] if isinstance(s, dict)]
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
    if not streams or not fmt:
        raise Rejected("unreadable", cfg=cfg)
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not (s.get("disposition") or {}).get("attached_pic")]
    if not videos:
        raise Rejected("no_video_stream", cfg=cfg)
    if len(videos) > 1:
        raise Rejected("several_video_streams", cfg=cfg)
    container = container_of(data, head)
    if container is None:
        raise Rejected("unreadable", cfg=cfg)
    if container not in cfg.ALLOWED_CONTAINERS:
        raise Rejected("container_not_allowed", cfg=cfg)
    duration, video_duration = _number(fmt.get("duration")), _number(videos[0].get("duration"))
    duration = video_duration if duration is None else duration
    size = display_size(videos[0])
    if duration is None or duration <= 0 or size is None:
        raise Rejected("unreadable", cfg=cfg)
    video_duration = duration if video_duration is None else video_duration
    if min(duration, video_duration) < cfg.MIN_DURATION_S:
        raise Rejected("too_short", cfg=cfg)
    if max(duration, video_duration) > cfg.MAX_DURATION_S:
        raise Rejected("too_long", cfg=cfg)
    if abs(duration - video_duration) > MAX_DURATION_MISMATCH_S:
        raise Rejected("unreadable", cfg=cfg)
    if size[0] > size[1]:
        raise Rejected("landscape", cfg=cfg)
    return MediaInfo(duration, size[0], size[1], container, size_bytes)


def validate(path: Path, cfg: config.Config = config.CONFIG) -> MediaInfo:
    """Check a stored upload; returns its facts or raises Rejected(code, user_message). Costs nothing."""
    try:
        size_bytes = os.stat(path).st_size
        with open(path, "rb") as f:
            head = f.read(64)
    except OSError:
        raise Rejected("unreadable", cfg=cfg) from None
    if size_bytes > cfg.MAX_UPLOAD_BYTES:
        raise Rejected("too_large", cfg=cfg)
    if size_bytes == 0:
        raise Rejected("unreadable", cfg=cfg)
    return inspect(probe(path), head, size_bytes, cfg)
