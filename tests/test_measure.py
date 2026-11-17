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


def _clip(path, seconds: int = 3, audio: bool = True, codec=("-c:v", "libx264", "-pix_fmt", "yuv420p")):
    sound = ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest"] if audio else []
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=144x256:rate=10",
                    *sound[:4], *codec, *sound[4:], str(path)], check=True, stdin=subprocess.DEVNULL, timeout=120)
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_every_ffmpeg_call_is_restricted_and_gets_only_the_path(tmp_path, monkeypatch):
    path = _clip(tmp_path / "my clip: one.mp4")              # a space and a colon in the name must not matter
    calls, real = [], subprocess.run

    def spy(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return real(cmd, **kwargs)

    monkeypatch.setenv("GEMINI_API_KEY_PAID", "do-not-pass-on")
    monkeypatch.setattr(M.subprocess, "run", spy)
    out = M.measure(path)
    assert out["duration_s"] == pytest.approx(3.0, abs=0.2) and out["loudness_lufs"] is not None and out["n_cuts"] is not None
    assert [c[0] for c, _ in calls] == ["ffprobe", "ffmpeg", "ffmpeg", "ffmpeg", "ffmpeg", "ffmpeg", "ffmpeg"]
    for cmd, kw in calls:
        i = cmd.index("-i")
        assert cmd[i - 4:i + 2] == ["-protocol_whitelist", "file", "-format_whitelist", "mov,matroska", "-i",
                                    "file:" + str(path.resolve())], cmd
        assert cmd.count("-i") == 1 and str(path) not in cmd
        assert kw["env"] == {"PATH": M.os.environ["PATH"]} and kw["timeout"] == 120 and kw["shell"] is False
        assert kw["stdin"] == subprocess.DEVNULL
    assert "do-not-pass-on" not in repr(calls)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_playlist_or_script_dressed_as_a_video_opens_nothing(tmp_path):
    real = _clip(tmp_path / "real.mp4", audio=False)
    playlist = tmp_path / "playlist.mp4"
    playlist.write_text(f"#EXTM3U\n#EXT-X-TARGETDURATION:3\n#EXTINF:3,\n{real}\n#EXT-X-ENDLIST\n")
    concat = tmp_path / "concat.mp4"
    concat.write_text("ffconcat version 1.0\nfile 'real.mp4'\n")
    open_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]
    loose = subprocess.run([*open_cmd, str(concat)], capture_output=True, timeout=60)
    assert loose.returncode == 0                             # unrestricted, ffprobe follows it to real.mp4
    for crafted in (playlist, concat):
        strict = M._run([*open_cmd, *M.ff_input(crafted)])
        assert strict.returncode != 0 and strict.stdout.strip() == b"", crafted.name
        with pytest.raises(RuntimeError):
            M.probe(crafted)
        with pytest.raises(RuntimeError):
            M.frames_gray(crafted, P)
        with pytest.raises(RuntimeError):
            M.opening_frames(crafted, P)
        with pytest.raises(RuntimeError):
            M.pcm_mono(crafted)
        assert M.scene_cuts(crafted, P.scene_threshold) == [] and M.audio_loudness(crafted)["loudness_lufs"] is None
        assert M.audio_silence(crafted, 3.0, P)["silence_frac"] == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_accepted_containers_still_measure(tmp_path):
    webm = _clip(tmp_path / "clip.webm", audio=False, codec=("-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8"))
    mov, mkv = _clip(tmp_path / "clip.mov"), _clip(tmp_path / "clip.mkv")
    for path, audio in ((webm, False), (mov, True), (mkv, True)):
        out = M.measure(path)
        assert out["duration_s"] == pytest.approx(3.0, abs=0.2) and out["has_audio"] is audio and out["motion_mean"] is not None


def test_child_environment_is_the_path_only(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("HOME", "/somewhere")
    assert M.ff_env() == {"PATH": M.os.environ["PATH"]}
    monkeypatch.delenv("PATH")
    assert M.ff_env() == {"PATH": M.os.defpath}


def _frames_values(values):
    """One uint8 frame per value (uniform luma), stacked."""
    return np.stack([np.full((P.frame_h, P.frame_w), v, dtype=np.uint8) for v in values])


class TestOpeningStats:
    # window = frames with i / open_fps < 0.5 -> indices 0..4 at 10 fps

    def test_black_first_frame(self):
        out = M.opening_stats(_frames_values([0] + [200] * 9), P)
        assert out["open_first_frame_black"] is True

    def test_bright_first_frame(self):
        out = M.opening_stats(_frames_values([200, 0] + [0] * 8), P)
        assert out["open_first_frame_black"] is False

    def test_open_luma(self):
        out = M.opening_stats(_frames_values([0, 0, 0, 0, 255] + [255] * 5), P)
        assert out["open_luma"] == pytest.approx(0.2, abs=1e-6)

    def test_still_opening_motion(self):
        out = M.opening_stats(_frames_values([120] * 10), P)
        assert out["open_motion"] == 0.0

    def test_moving_opening_motion(self):
        # window frames [0..4]: diffs 0->200->0->200->0 all = 200/255
        out = M.opening_stats(_frames_values([0, 200, 0, 200, 0] + [0] * 5), P)
        assert out["open_motion"] == pytest.approx(200 / 255, abs=1e-6)

    def test_motion_median_ignores_late_frames(self):
        # a big change at frame 8 (outside the window) must not affect open_motion
        out = M.opening_stats(_frames_values([0, 0, 0, 0, 0, 0, 0, 0, 255, 255]), P)
        assert out["open_motion"] == 0.0

    def test_one_frame(self):
        out = M.opening_stats(_frames_values([0]), P)
        assert out["open_first_frame_black"] is True
        assert out["open_motion"] is None
        assert out["open_luma"] == 0.0

    def test_empty_frames(self):
        out = M.opening_stats(np.zeros((0, P.frame_h, P.frame_w), dtype=np.uint8), P)
        assert out == {"open_first_frame_black": None, "open_motion": None,
                       "open_luma": None}
