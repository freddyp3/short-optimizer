"""Tests for the code-measured features (src/tools/analyze_video/measure.py)."""

import shutil
import subprocess

import numpy as np
import pytest

from src.tools.analyze_video import measure as M

P = M.MeasureParams()


# ------------------------------------------------------------- parsers

def test_parse_showinfo_times():
    err = ("[Parsed_showinfo_1 @ 0x1] n:   0 pts:  12800 pts_time:0.5     duration: ...\n"
           "[Parsed_showinfo_1 @ 0x1] n:   1 pts:  76800 pts_time:3.0 ...\n")
    assert M.parse_showinfo_times(err) == [0.5, 3.0]
    assert M.parse_showinfo_times("no cuts here") == []


def test_parse_ebur128_takes_summary_block():
    err = ("[Parsed_ebur128_0 @ 0x1] t: 1.0  TARGET:-23 LUFS  M: -20.1 S: -21.0  I: -22.0 LUFS   LRA: 1.0 LU\n"
           "[Parsed_ebur128_0 @ 0x1] Summary:\n  Integrated loudness:\n    I:   -18.3 LUFS\n"
           "  Loudness range:\n    LRA:     5.2 LU\n")
    assert M.parse_ebur128(err) == {"loudness_lufs": -18.3, "loudness_range": 5.2}
    assert M.parse_ebur128("") == {"loudness_lufs": None, "loudness_range": None}


def test_parse_silencedetect_leading_silence_sets_onset():
    err = ("[silencedetect @ 0x1] silence_start: 0\n"
           "[silencedetect @ 0x1] silence_end: 1.5 | silence_duration: 1.5\n"
           "[silencedetect @ 0x1] silence_start: 8\n")      # runs to the end
    out = M.parse_silencedetect(err, duration=10.0)
    assert out["audio_onset_s"] == 1.5
    assert out["silence_frac"] == pytest.approx((1.5 + 2.0) / 10.0)


def test_parse_silencedetect_no_silence():
    assert M.parse_silencedetect("", duration=10.0) == {"silence_frac": 0.0, "audio_onset_s": 0.0}


def test_parse_silencedetect_all_silent():
    out = M.parse_silencedetect("[silencedetect @ 0x1] silence_start: 0\n", duration=10.0)
    assert out == {"silence_frac": 1.0, "audio_onset_s": None}


# ------------------------------------------------------------- numpy features

def _frames(n, value):
    return np.full((n, P.frame_h, P.frame_w), value, dtype=np.uint8)


def test_motion_stats_static_video():
    out = M.motion_stats(_frames(40, 120), P)          # 10 s at 4 fps, identical frames
    assert out["motion_mean"] == 0.0
    assert out["static_frac"] == 1.0
    assert out["black_frac"] == 0.0


def test_motion_stats_flicker_and_black():
    f = _frames(40, 0)
    f[1::2] = 255                                       # alternate black / white every frame
    out = M.motion_stats(f, P)
    assert out["motion_mean"] == pytest.approx(1.0)
    assert out["static_frac"] == 0.0
    assert out["black_frac"] == pytest.approx(0.5)
    assert out["motion_first3s"] == pytest.approx(1.0)
    assert out["motion_last3s"] == pytest.approx(1.0)


def test_dhash_identical_and_opposite():
    flat = _frames(1, 100)[0]
    grad = np.tile(np.linspace(0, 255, P.frame_w, dtype=np.uint8), (P.frame_h, 1))
    assert M.hamming(M.dhash(flat), M.dhash(flat)) == 0
    assert M.hamming(M.dhash(flat), M.dhash(grad)) == 64   # flat: all bits 0; rising gradient: all bits 1
    assert M.loop_proxies(np.stack([grad, grad]))["first_last_pixel_diff"] == 0.0


def test_envelope_stats_peak_and_onset():
    sr, dur = 16000, 4.0
    t = np.arange(int(sr * dur)) / sr
    pcm = 0.001 * np.sin(2 * np.pi * 440 * t)
    pcm[int(sr * 2.5):int(sr * 3.0)] = 0.5 * np.sin(2 * np.pi * 440 * t[: int(sr * 0.5)])   # loud burst at 2.5-3.0 s
    out = M.envelope_stats(pcm.astype(np.float32), sr, dur, P)
    assert 2.4 <= out["t_loudness_peak_s"] <= 3.1
    assert out["loudness_peak_frac"] == pytest.approx(out["t_loudness_peak_s"] / dur)
    assert out["onsets_per_s"] > 0


# ------------------------------------------------------------- integration

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_measure_synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=144x256:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    out = M.measure(path)
    assert set(M.FEATURE_COLUMNS) <= set(out)
    assert out["duration_s"] == pytest.approx(3.0, abs=0.2)
    assert out["has_audio"] is True
    assert 0.0 <= out["static_frac"] <= 1.0
    assert out["loudness_lufs"] is not None
    assert out["silence_frac"] == pytest.approx(0.0, abs=0.05)
    assert 0 <= out["first_last_dhash_dist"] <= 64
