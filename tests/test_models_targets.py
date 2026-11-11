"""Tests for the pure functions in src/models/targets.py (synthetic frames, no data/ access)."""

import numpy as np
import pandas as pd
import pytest

from src.models import targets as T


def curve(video_id, values):
    r = np.arange(1, 101) / 100
    return pd.DataFrame({"video_id": video_id, "elapsedVideoTimeRatio": r, "audienceWatchRatio": values(r),
                         "relativeRetentionPerformance": 0.6})


class TestCurveTargets:
    def test_flat_curve_holds_everyone(self):
        t = T.curve_targets(curve("a", lambda r: np.full_like(r, 0.9))).iloc[0]
        assert t["early_hold"] == t["mid_hold"] == t["late_hold"] == pytest.approx(1.0)
        assert t["replay_level"] == pytest.approx(0.9) and t["mean_rrp"] == pytest.approx(0.6)

    def test_linear_decline(self):
        t = T.curve_targets(curve("a", lambda r: 1.2 - r)).iloc[0]
        assert t["replay_level"] == pytest.approx(1.18)
        assert t["mid_hold"] == pytest.approx(0.45 / 0.95)
        assert t["late_hold"] == pytest.approx(0.25 / 0.45)

    def test_one_row_per_video_regardless_of_order(self):
        both = pd.concat([curve("b", lambda r: 1 - r / 2), curve("a", lambda r: 1 - r / 2)]).sample(frac=1, random_state=0)
        out = T.curve_targets(both)
        assert sorted(out["video_id"]) == ["a", "b"] and out["mid_hold"].nunique() == 1


def test_rate_targets_use_the_shorts_row():
    m = pd.DataFrame({"video_id": ["a", "a"], "creatorContentType": ["shorts", "videoOnDemand"],
                      "engagedViews": [1000, 5], "likes": [99, 0], "shares": [0, 0], "comments": [9, 0]})
    out = T.rate_targets(m)
    assert len(out) == 1
    assert out["like_rate"].item() == pytest.approx(np.log(0.1))
    assert out["share_rate"].item() == pytest.approx(np.log(0.001))


def test_residual_on_views_removes_the_views_trend():
    rng = np.random.default_rng(0)
    views = rng.integers(1_000, 1_000_000, 80).astype(float)
    r = T.residual_on_views(2 * np.log(views) + rng.normal(0, 1, 80), views)
    assert abs(np.corrcoef(r, np.log(views))[0, 1]) < 1e-9


def test_dropoff_predictions():
    df = pd.DataFrame({"duration_s": [20.0, 30.0], "tail_all": [4.0, 30.0], "t_main_payoff_s": [10.0, np.nan],
                       "t_premise_clear_s": [1.5, 30.0], "t_loudness_peak_s": [3.0, 7.0]})
    p = T.dropoff_predictions(df)
    assert set(p) == set(T.DROPOFF_RULES)
    assert p["last_payoff"].tolist() == [16.0, 0.0]
    assert p["premise_clear"].iloc[0] == 1.5 and np.isnan(p["premise_clear"].iloc[1])
