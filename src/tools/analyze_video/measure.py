"""Code-measured video features (plan/03a-feature-checklist.md, group G).

Deterministic, free, ffmpeg/ffprobe + numpy only. One ``measure(path)`` call
returns a flat dict of features for one file; the CLI batches over a video
directory and writes ``code_features.csv`` plus a manifest with every
threshold used, so a feature value can always be traced to its parameters.

    ./venv/bin/python -m src.tools.analyze_video.measure \
        --videos data/videos --ids data/evals/hook_test/categories.csv \
        --out data/features/code_features.csv [--workers 4] [--limit N]

Pure functions (``motion_stats``, ``dhash``, ``parse_*``) take arrays/strings so
they are unit-testable without ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeasureParams:
    """Every threshold in one place; written to the manifest."""

    scene_threshold: float = 0.3     # ffmpeg select=gt(scene,T), 0..1
    fps: float = 4.0                 # frame sampling for motion / hashes
    frame_w: int = 72                # 9:16 at a size where a 9x8 block grid is exact
    frame_h: int = 128
    static_motion_max: float = 0.01  # mean |diff| / 255 below this = static window
    static_window_s: float = 0.5
    black_luma_max: float = 16 / 255
    edge_window_s: float = 3.0       # "first 3 s" / "last 3 s"
    silence_noise_db: float = -35.0
    silence_min_s: float = 0.3
    env_frame_s: float = 0.05        # RMS envelope frame
    env_smooth_frames: int = 5       # 250 ms moving average
    onset_rise_db: float = 6.0       # envelope jump that counts as an onset
    onset_floor_dbfs: float = -40.0


DEFAULT_PARAMS = MeasureParams()

FEATURE_COLUMNS = [
    "duration_s", "width", "height", "has_audio",
    "cuts_per_s", "cuts_first3s", "cuts_last3s", "n_cuts",
    "motion_mean", "motion_first3s", "motion_last3s", "static_frac", "black_frac",
    "first_last_dhash_dist", "first_last_pixel_diff",
    "loudness_lufs", "loudness_range", "silence_frac", "audio_onset_s",
    "t_loudness_peak_s", "loudness_peak_frac", "onsets_per_s",
]


# ----------------------------------------------------------------- subprocess

def _run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)


def probe(path: Path) -> dict:
    """Container/stream facts from ffprobe: duration, size, audio presence."""
    cp = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if cp.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path.name}: {cp.stderr.decode(errors='replace')[-300:]}")
    info = json.loads(cp.stdout.decode())
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path.name}")
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0.0)
    return {
        "duration_s": duration,
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "has_audio": audio is not None,
    }


# ----------------------------------------------------------------- video

_PTS_RE = re.compile(r"pts_time:\s*([0-9.]+)")


def parse_showinfo_times(stderr: str) -> list[float]:
    return [float(m.group(1)) for m in _PTS_RE.finditer(stderr)]


def scene_cuts(path: Path, threshold: float) -> list[float]:
    """Timestamps where ffmpeg's scene score exceeds ``threshold`` (hard cuts, big transitions)."""
    cp = _run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-an",
               "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"])
    return parse_showinfo_times(cp.stderr.decode(errors="replace"))


def frames_gray(path: Path, p: MeasureParams) -> np.ndarray:
    """Decode to a [T, H, W] uint8 array at p.fps, downscaled grayscale."""
    cp = _run(["ffmpeg", "-nostats", "-hide_banner", "-loglevel", "error", "-i", str(path), "-an",
               "-vf", f"fps={p.fps},scale={p.frame_w}:{p.frame_h}:flags=area,format=gray",
               "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    if cp.returncode != 0:
        raise RuntimeError(f"frame decode failed for {path.name}: {cp.stderr.decode(errors='replace')[-300:]}")
    buf = np.frombuffer(cp.stdout, dtype=np.uint8)
    n = len(buf) // (p.frame_w * p.frame_h)
    return buf[: n * p.frame_w * p.frame_h].reshape(n, p.frame_h, p.frame_w)


def motion_stats(frames: np.ndarray, p: MeasureParams) -> dict:
    """Frame-difference motion, static share, black share (pure numpy)."""
    t = frames.shape[0]
    if t < 2:
        return {"motion_mean": None, "motion_first3s": None, "motion_last3s": None,
                "static_frac": None, "black_frac": float((frames.mean(axis=(1, 2)) / 255 < p.black_luma_max).mean()) if t else None}
    f = frames.astype(np.int16)
    d = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2)) / 255.0          # one value per interval
    k = max(1, int(round(p.edge_window_s * p.fps)))                # intervals in 3 s
    win = max(1, int(round(p.static_window_s * p.fps)))             # intervals in 0.5 s
    n_win = len(d) // win
    windows = d[: n_win * win].reshape(n_win, win).mean(axis=1) if n_win else d[None].mean(axis=1)
    return {
        "motion_mean": float(d.mean()),
        "motion_first3s": float(d[:k].mean()),
        "motion_last3s": float(d[-k:].mean()),
        "static_frac": float((windows < p.static_motion_max).mean()),
        "black_frac": float((frames.mean(axis=(1, 2)) / 255 < p.black_luma_max).mean()),
    }


def dhash(frame: np.ndarray, grid: tuple[int, int] = (8, 9)) -> int:
    """Difference hash: block-average to rows x (cols) then compare horizontal neighbours -> 64-bit int."""
    rows, cols = grid
    h, w = frame.shape
    bh, bw = h // rows, w // cols
    blocks = frame[: rows * bh, : cols * bw].reshape(rows, bh, cols, bw).mean(axis=(1, 3))
    bits = (blocks[:, 1:] > blocks[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def loop_proxies(frames: np.ndarray) -> dict:
    """How different the last frame is from the first (objective loop-seam proxy, D2 substitute)."""
    if frames.shape[0] < 2:
        return {"first_last_dhash_dist": None, "first_last_pixel_diff": None}
    first, last = frames[0], frames[-1]
    return {
        "first_last_dhash_dist": hamming(dhash(first), dhash(last)),
        "first_last_pixel_diff": float(np.abs(first.astype(np.int16) - last.astype(np.int16)).mean() / 255.0),
    }


# ----------------------------------------------------------------- audio

_LUFS_RE = re.compile(r"I:\s+(-?[0-9.]+)\s+LUFS")
_LRA_RE = re.compile(r"LRA:\s+(-?[0-9.]+)\s+LU\b")


def parse_ebur128(stderr: str) -> dict:
    """Integrated loudness and loudness range from the ebur128 summary block (last match wins)."""
    i = _LUFS_RE.findall(stderr)
    lra = _LRA_RE.findall(stderr)
    return {"loudness_lufs": float(i[-1]) if i else None, "loudness_range": float(lra[-1]) if lra else None}


def audio_loudness(path: Path) -> dict:
    cp = _run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-vn", "-af", "ebur128", "-f", "null", "-"])
    out = parse_ebur128(cp.stderr.decode(errors="replace"))
    if out["loudness_lufs"] is not None and out["loudness_lufs"] < -70:   # ebur128 reports -inf-ish for silence
        out["loudness_lufs"] = -70.0
    return out


_SIL_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SIL_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")


def parse_silencedetect(stderr: str, duration: float) -> dict:
    """Silence share and first non-silent moment from silencedetect log lines."""
    starts = [max(0.0, float(x)) for x in _SIL_START_RE.findall(stderr)]
    ends = [float(x) for x in _SIL_END_RE.findall(stderr)]
    if len(ends) < len(starts):          # silence runs to the end of file
        ends.append(duration)
    total = sum(max(0.0, min(e, duration) - s) for s, e in zip(starts, ends))
    onset = 0.0
    if starts and starts[0] <= 0.05:
        onset = ends[0] if ends[0] < duration else None
    return {"silence_frac": min(1.0, total / duration) if duration > 0 else None, "audio_onset_s": onset}


def audio_silence(path: Path, duration: float, p: MeasureParams) -> dict:
    cp = _run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-vn",
               "-af", f"silencedetect=noise={p.silence_noise_db}dB:d={p.silence_min_s}", "-f", "null", "-"])
    return parse_silencedetect(cp.stderr.decode(errors="replace"), duration)


def pcm_mono(path: Path, sr: int = 16000) -> np.ndarray:
    cp = _run(["ffmpeg", "-nostats", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn",
               "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"])
    if cp.returncode != 0:
        raise RuntimeError(f"audio decode failed for {path.name}: {cp.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(cp.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def envelope_stats(pcm: np.ndarray, sr: int, duration: float, p: MeasureParams) -> dict:
    """Loudness-peak time and onset rate from a short-time RMS envelope (pure numpy)."""
    n = int(sr * p.env_frame_s)
    if n <= 0 or len(pcm) < 2 * n:
        return {"t_loudness_peak_s": None, "loudness_peak_frac": None, "onsets_per_s": None}
    frames = pcm[: (len(pcm) // n) * n].reshape(-1, n)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)
    k = p.env_smooth_frames
    smooth = np.convolve(rms_db, np.ones(k) / k, mode="same")
    peak_i = int(smooth.argmax())
    t_peak = (peak_i + 0.5) * p.env_frame_s
    # an onset = the level is >= onset_rise_db above where it was one smoothing window earlier;
    # count rising edges of that condition so a single jump is not counted on every frame it spans
    rises = smooth[k:] - smooth[:-k]
    hot = (rises >= p.onset_rise_db) & (smooth[k:] > p.onset_floor_dbfs)
    onsets = int((hot[1:] & ~hot[:-1]).sum() + (1 if hot.size and hot[0] else 0))
    return {
        "t_loudness_peak_s": t_peak,
        "loudness_peak_frac": t_peak / duration if duration > 0 else None,
        "onsets_per_s": onsets / duration if duration > 0 else None,
    }


# ----------------------------------------------------------------- one video

def measure(path: Path, p: MeasureParams = DEFAULT_PARAMS) -> dict:
    """All group-G features for one file. Audio features are None when there is no audio stream."""
    path = Path(path)
    out: dict = {k: None for k in FEATURE_COLUMNS}
    out.update(probe(path))
    dur = out["duration_s"]

    cuts = scene_cuts(path, p.scene_threshold)
    out.update({
        "n_cuts": len(cuts),
        "cuts_per_s": len(cuts) / dur if dur else None,
        "cuts_first3s": sum(1 for t in cuts if t <= p.edge_window_s),
        "cuts_last3s": sum(1 for t in cuts if t >= dur - p.edge_window_s),
    })

    frames = frames_gray(path, p)
    out.update(motion_stats(frames, p))
    out.update(loop_proxies(frames))

    if out["has_audio"]:
        out.update(audio_loudness(path))
        out.update(audio_silence(path, dur, p))
        out.update(envelope_stats(pcm_mono(path), 16000, dur, p))
    return out


# ----------------------------------------------------------------- batch CLI

def _git_sha() -> str | None:
    cp = _run(["git", "rev-parse", "--short", "HEAD"])
    return cp.stdout.decode().strip() if cp.returncode == 0 else None


def _ffmpeg_version() -> str | None:
    cp = _run(["ffmpeg", "-version"])
    return cp.stdout.decode().splitlines()[0] if cp.returncode == 0 else None


def run_batch(video_dir: Path, ids: list[str], out_csv: Path, workers: int = 4,
              p: MeasureParams = DEFAULT_PARAMS) -> dict:
    """Measure every id that has a file; write CSV + manifest; return the manifest."""
    import pandas as pd

    files = {vid: video_dir / f"{vid}.mp4" for vid in ids}
    missing = sorted(v for v, f in files.items() if not f.exists())
    todo = [(v, f) for v, f in files.items() if f.exists()]
    rows, failures = [], {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(measure, f, p): v for v, f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            vid = futs[fut]
            try:
                rows.append({"video_id": vid, **fut.result()})
            except Exception as e:  # one bad file must not kill the batch
                failures[vid] = str(e)[:200]
            if i % 25 == 0 or i == len(todo):
                log.info("%d/%d measured (%.0fs)", i, len(todo), time.time() - t0)
    df = pd.DataFrame(rows, columns=["video_id", *FEATURE_COLUMNS]).sort_values("video_id")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    manifest = {
        "params": asdict(p), "n_requested": len(ids), "n_measured": len(df), "missing_files": missing,
        "failures": failures, "ffmpeg": _ffmpeg_version(), "git_sha": _git_sha(),
        "seconds": round(time.time() - t0, 1), "out_csv": str(out_csv),
    }
    out_csv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=Path("data/videos"))
    ap.add_argument("--ids", type=Path, required=True, help="CSV with a video_id column (optionally category)")
    ap.add_argument("--category", default="brawl_stars", help="keep rows with this category if the column exists")
    ap.add_argument("--out", type=Path, default=Path("data/features/code_features.csv"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ids_df = pd.read_csv(args.ids)
    if "category" in ids_df.columns and args.category:
        ids_df = ids_df[ids_df["category"] == args.category]
    ids = ids_df["video_id"].drop_duplicates().tolist()[: args.limit]
    m = run_batch(args.videos, ids, args.out, workers=args.workers)
    log.info("measured %d/%d, %d missing files, %d failures, %.0fs -> %s",
             m["n_measured"], m["n_requested"], len(m["missing_files"]), len(m["failures"]), m["seconds"], args.out)
    return 0 if not m["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
