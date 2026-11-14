"""Tests for the retention simulator's arithmetic (synthetic curves, no data/ access)."""

import numpy as np
import pytest

from src.models.judge import simulator as M


def curve(rate=0.006, start=120.0):
    c = start * np.exp(-rate * np.arange(100))
    return 100 * c / c[M.I2]


class TestAssemble:
    def test_no_specialist_no_shape_is_exactly_the_reference(self):
        ref = curve()
        np.testing.assert_allclose(M.assemble(ref), ref, rtol=1e-12)

    def test_anchor_points_match_the_targets(self):
        ref = curve()
        out = M.assemble(ref, {"early": 0.9, "mid": 0.6, "late": 0.8})
        assert out[M.I2] == pytest.approx(100)
        assert out[M.I25] == pytest.approx(90) and out[M.I75] == pytest.approx(54) and out[M.I95] == pytest.approx(43.2)

    def test_untouched_segments_keep_the_reference_ratio(self):
        ref = curve()
        out = M.assemble(ref, {"late": 0.5})
        np.testing.assert_allclose(out[: M.I75 + 1], ref[: M.I75 + 1], rtol=1e-12)
        assert out[M.I95] / out[M.I75] == pytest.approx(0.5)

    def test_shape_moves_the_bends_but_not_the_anchors(self):
        ref = curve()
        loss = np.full(99, 0.001)
        loss[40] = 0.2                                   # one big drop inside the mid segment
        out = M.assemble(ref, {"mid": 0.6}, loss)
        assert out[M.I75] / out[M.I25] == pytest.approx(0.6)
        assert out[M.I25] == pytest.approx(ref[M.I25]) and out[M.I95] / out[M.I75] == pytest.approx(ref[M.I95] / ref[M.I75])
        steps = 1 - out[1:] / out[:-1]
        assert np.argmax(steps[M.I25:M.I75]) + M.I25 == 40

    def test_targets_are_clipped_and_curve_stays_positive(self):
        out = M.assemble(curve(), {"early": 9.0, "mid": -1.0, "late": 0.0})
        assert np.all(out > 0) and out[M.I25] == pytest.approx(160)

    def test_ratios(self):
        r = M.ratios(M.assemble(curve(), {"early": 0.9, "mid": 0.6, "late": 0.8}))
        np.testing.assert_allclose(r, [0.9, 0.6, 0.8])


class TestCalibration:
    def test_recovers_a_planted_slope(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        a, b = M.fit_calibration(x, 0.02 + 0.5 * x + rng.normal(0, 0.01, 200))
        assert a == pytest.approx(0.02, abs=0.005) and b == pytest.approx(0.5, abs=0.01)

    def test_negative_slope_is_not_trusted(self):
        x = np.arange(50.0)
        a, b = M.fit_calibration(x, 1.0 - 0.1 * x)
        assert b == 0.0 and a == pytest.approx((1.0 - 0.1 * x).mean())

    def test_useless_signal_gets_a_slope_near_zero(self):
        rng = np.random.default_rng(1)
        _, b = M.fit_calibration(rng.normal(size=2000), rng.normal(0, 0.05, 2000))
        assert abs(b) < 0.01


class TestReference:
    def test_excludes_and_takes_the_closest_in_length(self):
        curves = np.vstack([curve(0.004 + 0.0001 * i) for i in range(40)])
        dur = np.arange(40.0)
        ref = M.reference(curves, dur, np.arange(40), 10.0, exclude=[10])
        chosen = np.setdiff1d(np.arange(40), [10])
        chosen = chosen[np.argsort(np.abs(dur[chosen] - 10.0), kind="stable")][: M.ANCHOR_N]
        np.testing.assert_allclose(ref, curves[chosen].mean(axis=0))
        assert 10 not in chosen

    def test_normalise(self):
        c = M.normalise(np.vstack([np.linspace(2.0, 0.5, 100)]))
        assert c[0, M.I2] == pytest.approx(100)


class TestMeasures:
    def pieces(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        ref = np.vstack([curve(0.006)] * n)
        truth = np.vstack([M.assemble(ref[0], {"late": 0.8 + 0.1 * rng.normal()}) for _ in range(n)])
        return M.Pieces([f"v{i}" for i in range(n)], np.full(n, 20.0), truth, ref, {}, np.zeros((n, 99)), None)

    def test_perfect_prediction_has_skill_one_and_reference_has_skill_zero(self):
        p = self.pieces()
        assert M.measures(p.truth, p)["skill"] == pytest.approx(1.0)
        m = M.measures(p.ref, p)
        assert m["skill"] == pytest.approx(0.0) and m["skill_p"] > 0.5

    def test_band_covers_about_80_percent(self):
        rng = np.random.default_rng(3)
        n = 145
        ref = np.vstack([curve()] * n)
        truth = ref + rng.normal(0, 3, ref.shape)
        p = M.Pieces([f"v{i}" for i in range(n)], np.full(n, 20.0), truth, ref, {}, np.zeros((n, 99)), None)
        cov, band = M.band_coverage(ref, p)
        assert 0.72 < cov < 0.88 and band.shape == (2, 5)
