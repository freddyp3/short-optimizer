"""Tests for the pure helpers of the predict_retention tool (no data/ access, no network)."""

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.models.judge import simulator as M
from src.tools.predict_retention import tool as P


def test_label_thirds():
    assert [P.label(x) for x in (0, 33, 34, 66, 67, 100)] == ["below", "below", "typical", "typical", "above", "above"]


def test_percentile():
    assert P.percentile(2.5, np.array([1.0, 2.0, 3.0, 4.0])) == 50.0


def test_mid_signal_is_zero_when_gemini_agrees_with_the_reference():
    ref = 100 * np.exp(-0.006 * np.arange(100))[None, :]
    ref = 100 * ref / ref[:, [M.I2]]
    pts = ref[:, M.CHECK_IDX]
    assert P.mid_signal(pts, pts, ref)[0] == pytest.approx(0.0)
    better = pts.copy()
    better[:, 3] *= 1.1
    assert P.mid_signal(better, better, ref)[0] > 0


def test_draft_row_has_no_outcomes_and_fills_structure():
    template = pd.Series({"video_id": "x", "blind_id": "b", "split": "train", "hold_resid": 3.0, "late_hold": 0.1,
                          "engagedViews": 5000, "duration_s": 20.0, "duration_seconds": 20.0, "published_at": "2024-01-01T00:00:00Z",
                          "t_premise_clear_s": 1.0, "payoff_count_all": 2, "main_payoff_frac": 0.5, "tail_all": 3.0,
                          "dead_time_s": 1.0, "arc_ord": 1, "arc": "builds", "cuts_per_s": 0.3, "first_last_dhash_dist": 10,
                          "is_short": 1.0, "t_loudness_peak_s": 4.0, "t_main_payoff_s": 9.0}, dtype=object)
    v2 = {"result": {"payoffs": [{"t_s": 8.0, "kind": "reveal", "weight": 3}], "t_premise_clear_s": None, "arc": "single_peak",
                     "dead_time_s": 2.0, "ending": "cut_on_payoff"}}
    row = P.draft_row(v2, {"cuts_per_s": 0.8, "first_last_dhash_dist": 20, "t_loudness_peak_s": 5.0}, 25.0, template)
    assert all(row[c] != row[c] for c in ("hold_resid", "late_hold", "engagedViews"))      # NaN: no outcomes
    assert row["t_premise_clear_s"] == 25.0 and row["tail_all"] == 17.0 and row["arc_ord"] == 1
    assert row["cuts_per_s"] == 0.8 and row["is_short"] == 0.0 and row["video_id"] == "draft"


def test_blind_copy_runs_ffmpeg_restricted_with_a_time_limit_and_no_secrets(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        return subprocess.CompletedProcess(cmd, 0)

    src = tmp_path / "upload"
    src.write_bytes(b"not really a video")
    monkeypatch.setenv("GEMINI_API_KEY_PAID", "do-not-pass-on")
    monkeypatch.setattr(P.subprocess, "run", fake_run)
    out = P.blind_copy(src, tmp_path)
    cmd, i = seen["cmd"], seen["cmd"].index("-i")
    assert cmd[0] == "ffmpeg" and cmd[-1] == str(out) and out.parent == tmp_path and out.name.startswith("draft_")
    assert cmd[i - 4:i + 2] == ["-protocol_whitelist", "file", "-format_whitelist", "mov,matroska", "-i", "file:" + str(src.resolve())]
    assert cmd[i + 2:-1] == ["-map_metadata", "-1", "-c", "copy"]
    assert seen["timeout"] == 60 == P.BLIND_COPY_TIMEOUT_S and seen["shell"] is False and seen["check"] is True
    assert list(seen["env"]) == ["PATH"] and seen["stdin"] == subprocess.DEVNULL


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_blind_copy_copies_a_real_clip_and_refuses_a_script_file(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=90x160:rate=5",
                    "-metadata", "title=secret title", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
                   check=True, stdin=subprocess.DEVNULL, timeout=120)
    (tmp_path / "out").mkdir()
    out = P.blind_copy(clip, tmp_path / "out")
    assert out.stat().st_size > 0 and b"secret title" not in out.read_bytes() and b"secret title" in clip.read_bytes()
    concat = tmp_path / "concat.mp4"                         # unrestricted, ffmpeg would follow this to clip.mp4
    concat.write_text("ffconcat version 1.0\nfile 'clip.mp4'\n")
    with pytest.raises(subprocess.CalledProcessError):
        P.blind_copy(concat, tmp_path / "out")


def test_evidence_is_worded_for_a_reader_and_keeps_the_numbers():
    assert P.EVIDENCE == {
        "early": "Moderate evidence: confirmed on one set of 50 unseen videos (rank correlation 0.56). Predicts direction, not exact numbers.",
        "mid": "Good evidence: confirmed on two separate sets of unseen videos (rank correlations 0.56 and 0.63). Predicts direction, not exact numbers.",
        "late": "Good evidence: confirmed on two separate sets of unseen videos (rank correlations 0.48 and 0.60). Predicts direction, not exact numbers.",
        "drop_moments": "Good evidence: on two separate sets of unseen videos the predicted moments were steeper drops than chance "
                        "(score 0.65 and 0.66, where 0.50 is chance and 1.00 is always the steepest second).",
        "rewatch_level": "Weak evidence: confirmed on two sets of unseen videos but with small effect (rank correlations 0.43 and 0.27). "
                         "Treat as a hint."}
    assert P.NO_EXAMPLES_NOTE == ("Guess only: no past Short was similar enough to compare with, and without examples this item "
                                  "was not reliable in testing.")
    for text in (*P.EVIDENCE.values(), P.NO_EXAMPLES_NOTE):
        assert "GATE" not in text and "HELDOUT" not in text


def test_curve_grid_has_no_repeated_percent():
    assert P.runtime_pct() == list(range(1, 101)) and [int(v * 100) for v in M.GRID] != P.runtime_pct()   # int() repeated 28
