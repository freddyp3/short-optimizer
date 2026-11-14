"""Tests for the pure helpers of the predict_retention tool (no data/ access, no network)."""

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
