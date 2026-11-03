"""Map Google Takeout video files to video_ids, hardlink them into data/videos/, and blind the selection.

Takeout names files by title (filesystem-illegal chars become ``_``, duplicate titles get ``(N)``
suffixes), so matching is: exact normalized title -> fuzzy title -> duration disambiguation for
ties -> ``--resolve`` manual overrides. Writes ``data/evals/hook_test/video_map.csv``, copies the
raw Takeout metadata into ``data/channel/takeout/``, and produces metadata-stripped blind copies
plus contact sheets for the leakage eyeball check (plan/02-hook-test.md section 3).

    ./venv/bin/python -m src.evals.hook_test.fetch_videos \\
        --src "~/Downloads/Takeout/YouTube and YouTube Music" [--export DIR] \\
        [--resolve VIDEO_ID=PATH ...] [--no-blind]
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.evals.hook_test.categorize import latest_export
from src.evals.hook_test.config import EXPORT_BASE, OUT_DIR

log = logging.getLogger(__name__)

VIDEOS_DIR = Path("data/videos")
TAKEOUT_DIR = Path("data/channel/takeout")
MAP_CSV = OUT_DIR / "video_map.csv"
SELECTION_CSV = OUT_DIR / "selection.csv"
BLIND_DIR = OUT_DIR / "blind"
FRAMES_DIR = OUT_DIR / "blind_frames"
PROBE_CACHE = VIDEOS_DIR / ".probe_cache.json"

FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# 5x7 bitmap font for blind_id labels (homebrew ffmpeg has no libfreetype/drawtext,
# and PIL is not a project dep). Rows top->bottom, bit 4 = leftmost pixel.
_FONT5X7 = {
    "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
    "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
    "2": [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
    "3": [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
    "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
    "5": [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
    "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
    "7": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
    "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
    "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
    "a": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "b": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
    "c": [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
    "d": [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
    "e": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
    "f": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
    "g": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
    "h": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "i": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    "j": [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
    "k": [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
    "l": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
    "m": [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
    "n": [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
    "o": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "p": [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
    "q": [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
    "r": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
    "s": [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
    "t": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    "u": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "v": [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
    "w": [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
    "x": [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
    "y": [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
    "z": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
    " ": [0] * 7,
    "?": [0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04],
}


def label_strip(text: str, width: int, scale: int = 4, height: int | None = None) -> bytes:
    """Render text (yellow on black) as PNG bytes of the given pixel width."""
    height = height or 7 * scale + 8
    px = bytearray(width * height * 3)

    def dot(x: int, y: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            px[i:i + 3] = b"\xff\xd7\x00"  # yellow

    x0 = 8
    y0 = (height - 7 * scale) // 2
    for ci, ch in enumerate(text):
        glyph = _FONT5X7.get(ch.lower(), _FONT5X7["?"])
        gx = x0 + ci * 6 * scale
        for ry, row in enumerate(glyph):
            for cx in range(5):
                if row >> (4 - cx) & 1:
                    for dy in range(scale):
                        for dx in range(scale):
                            dot(gx + cx * scale + dx, y0 + ry * scale + dy)
    return _png_rgb(width, height, px)


def _png_rgb(width: int, height: int, px: bytearray) -> bytes:
    """Minimal RGB PNG encoder (stdlib only)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))
    raw = b"".join(
        b"\x00" + bytes(px[y * width * 3:(y + 1) * width * 3]) for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))

VIDEO_EXTS = {".mp4", ".mov"}
# Chars Takeout turns into "_" in filenames: filesystem-illegal ones plus quotes/apostrophes.
TAKEN_OUT_CHARS = set('\\/:*?"<>|\'’‘“”')
DUP_SUFFIX_RE = re.compile(r"\(\d+\)\s*$")
FUZZY_MIN_RATIO = 0.9
DURATION_TOLERANCE_S = 1.0
SHEET_COLS, SHEET_ROWS = 2, 5          # 10 videos per sheet
FRAME_W, FRAME_H = 300, 534            # px per frame; sheet width = 2 cols * 3 frames * 300 = 1800


# ---------------------------------------------------------------- pure helpers

def normalize_exact(s: str) -> str:
    """Title <-> filename-stem normalization for exact matching.

    NFC, strip a trailing ``(N)`` dup suffix, strip whitespace, map Takeout's
    replaced chars to ``_``.
    """
    s = unicodedata.normalize("NFC", s)
    s = DUP_SUFFIX_RE.sub("", s)
    s = s.strip()
    s = "".join("_" if c in TAKEN_OUT_CHARS else c for c in s)
    # Takeout collapses runs of replaced chars: "game??!?" -> "game_!_"
    return re.sub(r"_+", "_", s)


def normalize_fuzzy(s: str) -> str:
    """Aggressive normalization for fuzzy matching: alnum only, ``_``-joined, casefolded."""
    s = normalize_exact(s).casefold()
    s = "".join(c if c.isalnum() else "_" for c in s)
    return re.sub(r"_+", "_", s).strip("_")


def parse_resolve(pairs: list[str]) -> dict[str, str]:
    """Parse ``VIDEO_ID=PATH`` overrides into {video_id: path}."""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--resolve expects VIDEO_ID=PATH, got {p!r}")
        vid, path = p.split("=", 1)
        out[vid.strip()] = path.strip()
    return out


def load_takeout_metadata(meta_dir: Path) -> pd.DataFrame:
    """Concat the ``videos*.csv`` chunks (differing column sets) into one df keyed by Video ID."""
    files = sorted(meta_dir.glob("videos*.csv"))
    if not files:
        raise FileNotFoundError(f"no videos*.csv under {meta_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.drop_duplicates(subset="Video ID").set_index("Video ID")
    log.info("takeout metadata: %d unique video ids from %s", len(df), meta_dir)
    return df


def merged_takeout_videos(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per video_id with the column set the repo keeps."""
    tag_cols = [c for c in meta.columns if re.fullmatch(r"Tag \d+", c)]
    tags = meta[tag_cols].apply(
        lambda row: "|".join(str(v) for v in row if pd.notna(v) and str(v).strip()), axis=1
    )
    out = pd.DataFrame({
        "video_id": meta.index,
        "title": meta["Video Title (Original)"],
        "description": meta.get("Video Description (Original)"),
        "privacy": meta["Privacy"],
        "state": meta["Video State"],
        "category": meta["Video Category"],
        "channel_id": meta.get("Channel ID"),
        "approx_duration_s": pd.to_numeric(meta["Approx Duration (ms)"], errors="coerce") / 1000.0,
        "created_at": meta["Video Create Timestamp"],
        "published_at": meta["Video Publish Timestamp"],
        "tags": tags,
    })
    return out.reset_index(drop=True)


@dataclass
class FileEntry:
    path: Path
    stem: str
    duration: float | None


@dataclass
class Match:
    video_id: str
    method: str


def match_files(
    files: list[FileEntry],
    meta: pd.DataFrame,
    overrides: dict[str, str],
) -> dict[Path, Match]:
    """Assign each file a video_id. Every id and every file used at most once."""
    matches: dict[Path, Match] = {}
    used_ids: set[str] = set()

    # --resolve overrides first (paths are as passed; resolve against cwd).
    by_path = {f.path.resolve(): f for f in files}
    for vid, raw in overrides.items():
        entry = by_path.get(Path(raw).expanduser().resolve())
        if entry is None:
            log.warning("--resolve %s=%s: path not found among scanned files", vid, raw)
            continue
        if vid in used_ids:
            log.warning("--resolve %s: video_id already used, skipping", vid)
            continue
        matches[entry.path] = Match(vid, "manual")
        used_ids.add(vid)

    remaining = [f for f in files if f.path not in matches]
    ids = [vid for vid in meta.index if vid not in used_ids]

    exact_ids: dict[str, list[str]] = {}
    for vid in ids:
        exact_ids.setdefault(normalize_exact(str(meta.at[vid, "Video Title (Original)"])), []).append(vid)
    exact_files: dict[str, list[FileEntry]] = {}
    for f in remaining:
        exact_files.setdefault(normalize_exact(f.stem), []).append(f)

    def take(entry: FileEntry, vid: str, method: str) -> None:
        matches[entry.path] = Match(vid, method)
        used_ids.add(vid)

    # title_exact: key with exactly one unmatched id and one file
    for key, fs in exact_files.items():
        cand_ids = [v for v in exact_ids.get(key, []) if v not in used_ids]
        free = [f for f in fs if f.path not in matches]
        if len(cand_ids) == 1 and len(free) == 1:
            take(free[0], cand_ids[0], "title_exact")

    # title_fuzzy: unique best difflib ratio >= threshold over remaining ids
    fuzzy_ids = {v: normalize_fuzzy(str(meta.at[v, "Video Title (Original)"]))
                 for v in ids if v not in used_ids}
    for f in remaining:
        if f.path in matches:
            continue
        target = normalize_fuzzy(f.stem)
        scored = sorted(
            ((difflib.SequenceMatcher(None, target, fkey).ratio(), v)
             for v, fkey in fuzzy_ids.items() if v not in used_ids),
            reverse=True,
        )
        if not scored or scored[0][0] < FUZZY_MIN_RATIO:
            continue
        if len(scored) > 1 and scored[1][0] == scored[0][0]:
            log.info("fuzzy tie for %s (%s vs %s), leaving for duration step",
                     f.path.name, scored[0][1], scored[1][1])
            continue
        take(f, scored[0][1], "title_fuzzy")

    # title+duration: disambiguate remaining ties within shared exact keys.
    # A file only claims an id when its best diff is strictly better than the
    # runner-up; contested ids go to the smaller diff and the loser re-evaluates.
    for key, fs in exact_files.items():
        for _ in range(len(fs) + 1):
            cand_ids = [v for v in exact_ids.get(key, []) if v not in used_ids]
            free = [f for f in fs if f.path not in matches]
            if not cand_ids or not free:
                break
            claims: dict[str, tuple[float, FileEntry]] = {}
            for f in free:
                if f.duration is None:
                    continue
                diffs = sorted(
                    (abs(f.duration - float(meta.at[v, "Approx Duration (ms)"]) / 1000.0), v)
                    for v in cand_ids
                    if pd.notna(pd.to_numeric(meta.at[v, "Approx Duration (ms)"], errors="coerce"))
                )
                diffs = [d for d in diffs if d[0] <= DURATION_TOLERANCE_S]
                if not diffs or (len(diffs) > 1 and diffs[1][0] == diffs[0][0]):
                    continue
                diff, vid = diffs[0]
                if vid not in claims or diff < claims[vid][0]:
                    claims[vid] = (diff, f)
            if not claims:
                break
            for vid, (_, f) in claims.items():
                if f.path not in matches and vid not in used_ids:
                    take(f, vid, "title+duration")

    return matches


# ---------------------------------------------------------------- ffprobe / ffmpeg

def probe_durations(files: list[Path], cache_path: Path = PROBE_CACHE) -> dict[Path, float]:
    """ffprobe each file; cache by path+size+mtime."""
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    out = {}
    for p in files:
        st = p.stat()
        key = str(p)
        hit = cache.get(key)
        if hit and hit["size"] == st.st_size and hit["mtime"] == st.st_mtime:
            out[p] = hit["duration"]
            continue
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
            capture_output=True, text=True)
        try:
            dur = float(r.stdout.strip())
        except ValueError:
            log.warning("ffprobe failed on %s: %s", p.name, r.stderr.strip()[:200])
            dur = None
        cache[key] = {"size": st.st_size, "mtime": st.st_mtime, "duration": dur}
        out[p] = dur
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return out


def hardlink(src: Path, dst: Path) -> str:
    """Hardlink src -> dst. Returns 'skip' | 'link' | 'relink'."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.samefile(src):
            return "skip"
        dst.unlink()
        os.link(src, dst)
        return "relink"
    os.link(src, dst)
    return "link"


def blind_video(src: Path, dst: Path) -> bool:
    """Metadata-stripped copy at dst. Skips if dst is newer than src. Returns True if rebuilt."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [FFMPEG, "-nostdin", "-y", "-i", str(src), "-map_metadata", "-1", "-map_chapters", "-1",
         "-c", "copy", str(dst)],
        capture_output=True, check=True)
    return True


def format_tags(path: Path) -> dict:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)],
        capture_output=True, text=True)
    return json.loads(r.stdout).get("format", {}).get("tags", {})


def extract_frame(video: Path, t: float, out: Path) -> None:
    subprocess.run(
        [FFMPEG, "-nostdin", "-y", "-ss", f"{t:.2f}", "-i", str(video),
         "-frames:v", "1", str(out)],
        capture_output=True, check=True)


def contact_sheets(blind_videos: list[tuple[str, str, Path, float]], frames_dir: Path,
                   scratch: Path) -> list[Path]:
    """3 frames per blind video, blind_id label, ~10 videos per sheet per split. Returns sheet paths."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    tiles: list[tuple[str, Path, str]] = []
    for blind_id, split, video, dur in blind_videos:
        times = [0.5, max(0.5, dur / 2), max(0.5, dur - 0.5)]
        frames = []
        for i, t in enumerate(times):
            fr = scratch / f"{blind_id}_{i}.jpg"
            extract_frame(video, t, fr)
            frames.append(fr)
        label = scratch / f"{blind_id}_label.png"
        label.write_bytes(label_strip(blind_id, FRAME_W * 3))
        tile = scratch / f"{blind_id}_tile.jpg"
        subprocess.run(
            [FFMPEG, "-nostdin", "-y", "-i", str(frames[0]), "-i", str(frames[1]),
             "-i", str(frames[2]), "-i", str(label),
             "-filter_complex",
             (";".join(
                 f"[{i}]scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease,"
                 f"pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2:black[{x}]"
                 for i, x in enumerate("abc"))
              + ";[a][b][c]hstack=3[frames];[3][frames]vstack=2"),
             str(tile)],
            capture_output=True, check=True)
        tiles.append((blind_id, tile, split))

    sheets = []
    per = SHEET_COLS * SHEET_ROWS
    groups: dict[str, list[tuple[str, Path]]] = {}
    for blind_id, tile, split in tiles:
        groups.setdefault(split, []).append((blind_id, tile))
    for split, group in sorted(groups.items()):
        for n, start in enumerate(range(0, len(group), per), 1):
            chunk = [t for _, t in group[start:start + per]]
            # pad chunk to a full grid so vstack always sees SHEET_ROWS inputs
            grid = list(chunk) + [chunk[-1]] * (per - len(chunk))
            cols = []
            for c in range(SHEET_COLS):
                col_tiles = [grid[c + SHEET_COLS * r] for r in range(SHEET_ROWS)]
                col = scratch / f"_{split}{n}_col{c}.jpg"
                inputs = [x for t in col_tiles for x in ("-i", str(t))]
                subprocess.run(
                    [FFMPEG, "-nostdin", "-y", *inputs,
                     "-filter_complex", f"vstack=inputs={len(col_tiles)}",
                     "-frames:v", "1", str(col)],
                    capture_output=True, check=True)
                cols.append(col)
            sheet = frames_dir / f"{split}_{n:02d}.jpg"
            subprocess.run(
                [FFMPEG, "-nostdin", "-y", "-i", str(cols[0]), "-i", str(cols[1]),
                 "-filter_complex", "hstack=2", "-frames:v", "1", str(sheet)],
                capture_output=True, check=True)
            sheets.append(sheet)
    return sheets


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="Takeout root (contains videos/ and video metadata/)")
    parser.add_argument("--export", type=Path, help="export dir (default: newest under data/channel)")
    parser.add_argument("--resolve", nargs="*", default=[], metavar="VIDEO_ID=PATH")
    parser.add_argument("--no-blind", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    src = args.src.expanduser()
    meta = load_takeout_metadata(src / "video metadata")
    export_dir = args.export or latest_export(EXPORT_BASE)
    export_videos = pd.read_csv(export_dir / "videos.csv").set_index("video_id")
    log.info("export: %s (%d videos)", export_dir, len(export_videos))

    scan = sorted(p for p in (src / "videos").iterdir() if p.suffix.lower() in VIDEO_EXTS)
    log.info("scanning %d files under %s", len(scan), src / "videos")
    durations = probe_durations(scan)
    files = [FileEntry(p, unicodedata.normalize("NFC", p.stem), durations.get(p)) for p in scan]

    overrides = parse_resolve(args.resolve)
    matches = match_files(files, meta, overrides)

    rows, unmatched = [], []
    for f in files:
        m = matches.get(f.path)
        if m is None:
            unmatched.append(f)
            continue
        vid = m.video_id
        ext = f.path.suffix.lower()
        dst = VIDEOS_DIR / f"{vid}{ext}"
        action = hardlink(f.path, dst)
        in_export = vid in export_videos.index
        exp_dur = float(export_videos.at[vid, "duration_seconds"]) if in_export else float("nan")
        diff = (f.duration - exp_dur) if (f.duration is not None and in_export) else float("nan")
        rows.append({
            "video_id": vid,
            "path": str(dst),
            "duration_ffprobe": f.duration,
            "match_method": m.method,
            "source": "takeout",
            "takeout_filename": f.path.name,
            "title": meta.at[vid, "Video Title (Original)"] if vid in meta.index else None,
            "privacy": meta.at[vid, "Privacy"] if vid in meta.index else None,
            "in_export": in_export,
            "export_duration_s": exp_dur if in_export else None,
            "dur_diff_s": round(diff, 2) if diff == diff else None,
            "duration_ok": (abs(diff) <= DURATION_TOLERANCE_S) if diff == diff else None,
        })
        log.debug("%s -> %s (%s, %s)", f.path.name, vid, m.method, action)

    vmap = pd.DataFrame(rows).sort_values("video_id")
    MAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    vmap.to_csv(MAP_CSV, index=False)

    # copy raw takeout metadata + merged videos.csv
    raw = TAKEOUT_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for f in sorted((src / "video metadata").glob("*.csv")):
        shutil.copy2(f, raw / f.name)
    merged_takeout_videos(meta).to_csv(TAKEOUT_DIR / "videos.csv", index=False)

    # summary
    log.info("files scanned: %d, matched: %d, unmatched: %d", len(files), len(vmap), len(unmatched))
    for method, n in vmap["match_method"].value_counts().items():
        log.info("  %-14s %d", method, n)
    for f in unmatched:
        log.warning("unmatched file: %s (dur=%.1fs)", f.path.name, f.duration or -1)
    bad = vmap[vmap["duration_ok"] == False]
    log.info("duration mismatches vs export (|diff|>1s): %d", len(bad))
    for _, r in bad.sort_values("dur_diff_s", key=abs, ascending=False).head(10).iterrows():
        log.warning("  %s %s ffprobe=%.1f export=%.0f diff=%+.1f",
                    r.video_id, r.takeout_filename, r.duration_ffprobe or -1,
                    r.export_duration_s or -1, r.dur_diff_s or 0)

    sel = pd.read_csv(SELECTION_CSV) if SELECTION_CSV.exists() else pd.DataFrame()
    if len(sel):
        sel_map = vmap.set_index("video_id")
        covered = sel[sel["video_id"].isin(sel_map.index)]
        ok = covered[covered["video_id"].isin(vmap[vmap["duration_ok"] == True]["video_id"])]
        log.info("selection coverage: %d/%d mapped, %d duration_ok", len(covered), len(sel), len(ok))

    if not args.no_blind and len(sel):
        blind_videos = []
        for _, r in sel.iterrows():
            if r["video_id"] not in sel_map.index:
                log.warning("selection %s not mapped, skipping blind", r["video_id"])
                continue
            dst = BLIND_DIR / f"{r['blind_id']}.mp4"
            blind_video(Path(sel_map.at[r["video_id"], "path"]), dst)
            tags = format_tags(dst)
            leaked = {k: v for k, v in tags.items()
                      if k.lower() in {"title", "comment", "creation_time"}}
            if leaked:
                log.warning("blind %s still has tags: %s", dst.name, leaked)
            blind_videos.append((r["blind_id"], r["split"], dst, probe_single(dst) or 0.0))
        log.info("blind videos: %d", len(blind_videos))
        scratch = FRAMES_DIR / "_scratch"
        sheets = contact_sheets(sorted(blind_videos), FRAMES_DIR, scratch)
        for s in sheets:
            log.info("contact sheet: %s", s)
    return 0


def probe_single(path: Path) -> float | None:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
