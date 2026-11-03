"""Tests for ground_truth.py (plan section 10): synthetic curves, known cliff, flat flag.

Note: ``t*`` is the *start* of the ~1 s window with the largest relative loss,
so a curve that drops at ~7.6 s yields ``t*_post`` ~7.0 s.
"""

import numpy as np
import pytest

from src.evals.hook_test.ground_truth import curve_gt, loss_windows, rank_score


def cliff_curve(D=21.0, drop_t=7.6):
    """audienceWatchRatio ~1 until drop_t, then drops to ~0.55 and stays."""
    r = np.arange(1, 101) / 100
    t = r * D
    a = np.where(t < drop_t, 1.0, 0.55)
    return r, a


class TestCurveGt:
    def test_cliff_window_starts_at_7s_of_21s(self):
        r, a = cliff_curve(21.0, 7.6)
        gt = curve_gt(r, a, 21.0)
        assert not gt["flat_curve"]
        assert abs(gt["t_star_post"] - 7.0) <= 0.25

    def test_flat_curve_flag(self):
        r = np.arange(1, 101) / 100
        a = 1.0 - 0.3 * r  # smooth monotone decay, no cliff
        gt = curve_gt(r, a, 21.0)
        assert gt["flat_curve"]

    def test_rise_keeps_negative_loss(self):
        r = np.arange(1, 101) / 100
        a = np.where(r > 0.5, 1.0, 0.8)  # curve rises midway (loop)
        t, loss, w = loss_windows(r, a, 20.0)
        assert loss.min() < 0  # negative loss retained, not clipped

    def test_rank_score_cliff_is_1(self):
        r, a = cliff_curve(21.0, 7.6)
        gt = curve_gt(r, a, 21.0)
        assert rank_score(gt["t_star_post"], gt["t"], gt["loss"]) > 0.95
        assert 0.3 < rank_score(1.0, gt["t"], gt["loss"]) < 0.7

    def test_end_windows_in_range(self):
        r, a = cliff_curve(21.0, 7.6)
        gt = curve_gt(r, a, 21.0)
        assert 0 <= rank_score(20.9, gt["t"], gt["loss"]) <= 1.0
        assert 0 <= rank_score(0.0, gt["t"], gt["loss"]) <= 1.0

    @pytest.mark.parametrize("D", [8.0, 60.0])
    def test_w_sane(self, D):
        r = np.arange(1, 101) / 100
        t, loss, w = loss_windows(r, np.linspace(1, 0.7, 100), D)
        # window duration ~1 s: t[w] - t[0] should be ~1 s
        assert abs((t[w] - t[0]) - 1.0) < 0.5
        assert len(loss) == len(t)

    def test_post_restriction(self):
        # biggest loss in first second -> t*_all ~0 but t*_post later
        r, a = cliff_curve(21.0, 7.6)
        a[1:10] = 0.2  # steep instant swipe-out in the first ~2 s
        gt = curve_gt(r, a, 21.0)
        assert gt["t_star_all"] < 1.0
        assert gt["t_star_post"] >= 1.5
