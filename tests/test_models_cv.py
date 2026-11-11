"""Tests for src/models/features.py and cv.py on synthetic data (no data/ access)."""

import numpy as np
import pandas as pd
import pytest

from src.models import cv
from src.models import features as F

PREV = {"nominal": {"format_macro": {"decision": "keep", "levels": ["edit", "gameplay_clip", "other"],
                                     "counts": {"edit": 50, "gameplay_clip": 40, "irl": 6, "slideshow": 6},
                                     "other_count": 12}}}


def fake_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: rng.normal(size=n) for c in set(F.M1_COLS + F.CODE_COLS) - {"dhash_x_short"}})
    d["video_id"] = [f"v{i:03d}" for i in range(n)]
    d["published_at"] = pd.date_range("2024-03-01", periods=n, freq="3D", tz="UTC").astype(str)
    d["duration_s"] = rng.uniform(8, 60, n)
    d["is_short"] = (d["duration_s"] <= 20).astype(float)
    d["t_first_event_s"] = rng.uniform(0, 3, n)
    d["t_loudness_peak_s"] = rng.uniform(0, 8, n)
    d["t_main_payoff_s"] = rng.uniform(0, 8, n)
    d["surprise_peak"] = rng.choice(["none", "mild", "strong"], n)
    d["opening_frame"] = rng.choice(["mid_action", "menu_or_lobby"], n)
    d["ending_from_tail_all"] = rng.choice(["cut_on_payoff", "long_tail"], n)
    for b in ("big_number_on_screen", "low_health_visible"):
        d[b] = rng.random(n) < 0.3
    for f in F.H_FIELDS:
        d[f] = rng.choice(["a", "b", "c"], n)
    d["format_macro"] = rng.choice(["edit", "gameplay_clip", "irl", "slideshow"], n)
    return d


@pytest.fixture(scope="module")
def data():
    return F.prepare(fake_df(), PREV)


class TestPrepare:
    def test_onehot_drops_reference_and_merges_rare(self, data):
        assert F.onehot_cols(data) == ("format_macro=gameplay_clip", "format_macro=other")
        df = fake_df()
        assert data.col("format_macro=other").sum() == df["format_macro"].isin(["irl", "slideshow"]).sum()

    def test_interaction_and_log(self, data):
        df = fake_df()
        np.testing.assert_allclose(data.col("dhash_x_short"), df["first_last_dhash_dist"] * df["is_short"])
        np.testing.assert_allclose(data.col("log1p:t_first_event_s"), np.log1p(df["t_first_event_s"]))

    def test_registered_rows(self, data):
        rs = F.registered_recipes(data)
        assert rs["M1_core_mechanics"].n_params == 9
        assert rs["M2_core"].n_params == 10
        assert rs["M4_code_only"].n_params == 14
        assert rs["mean_only"].n_params == 0


class TestHistory:
    def test_only_earlier_fitted_videos(self, data):
        n = len(data)
        fit = np.arange(0, n, 2)
        y = np.zeros(len(fit))
        y[-1] = 100.0                                   # the latest fitted video carries all the signal
        H = F.history_features(data, fit, y, np.arange(n))
        assert H[: fit[-1] + 1, 0].max() == 0.0         # nobody published at or before it can see it
        assert H[n - 1, 2] == pytest.approx(100.0 / F.RECENT_N)

    def test_first_video_has_no_neighbours(self, data):
        H = F.history_features(data, np.arange(len(data)), np.ones(len(data)), np.array([0]))
        assert H[0, 0] == 0.0 and np.isnan(H[0, 1]) and H[0, 2] == 0.0

    def test_shrunk_mean(self, data):
        n = len(data)
        H = F.history_features(data, np.arange(n), np.full(n, 8.0), np.array([n - 1]))
        assert H[0, 0] == pytest.approx(8.0 * F.H_K / (F.H_K + F.H_SHRINK))

    def test_own_label_never_used(self, data):
        n = len(data)
        y = np.zeros(n)
        y[50] = 1000.0
        assert F.history_features(data, np.arange(n), y, np.array([50]))[0, 0] == 0.0

    def test_other_split_uses_train_neighbours(self, data):
        other = F.prepare(fake_df(10, seed=1), PREV)
        r = F.Recipe("x", h1=True)
        y = np.full(len(data), 8.0)
        _, Xte, _, names = F.design(r, data, np.arange(len(data)), y, te_data=other)
        assert names == ["similar_past_mean"] and Xte.shape == (10, 1)


class TestDesign:
    def test_zscore_uses_training_rows_only(self, data):
        tr, te = np.arange(0, 100), np.arange(100, 120)
        Xtr, Xte, _, _ = F.design(F.Recipe("m1", cols=F.M1_COLS), data, tr, np.zeros(100), te=te)
        np.testing.assert_allclose(Xtr.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(Xtr.std(axis=0), 1, atol=1e-9)
        assert Xte.shape == (20, 9)

    def test_selection_picks_the_planted_column(self, data):
        tr = np.arange(100)
        r = F.Recipe("s", select_from=F.CODE_COLS, select_k=1)
        *_, names = F.design(r, data, tr, 3 * data.col("motion_mean")[tr], te=np.arange(100, 120))
        assert names == ["motion_mean"]

    def test_y_cap_clips_training_labels(self, data):
        y = np.arange(100.0)
        _, _, yu, _ = F.design(F.Recipe("c", cols=("cuts_per_s",), y_cap=0.1), data, np.arange(100), y)
        assert yu.min() == pytest.approx(np.quantile(y, 0.1)) and yu.max() == pytest.approx(np.quantile(y, 0.9))


class TestRidge:
    def test_matches_closed_form(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(50, 4)), rng.normal(size=50)
        _, coef = cv.ridge_path(X, y, X, np.array([0.0, 7.0]))
        for j, a in enumerate((0.0, 7.0)):
            np.testing.assert_allclose(coef[:, j], np.linalg.solve(X.T @ X + a * np.eye(4), X.T @ (y - y.mean())),
                                       atol=1e-8)

    def test_no_features_predicts_the_mean(self):
        p, _ = cv.ridge_path(np.empty((5, 0)), np.arange(5.0), np.empty((3, 0)))
        assert np.all(p == 2.0)

    def test_kfold_partitions(self):
        folds = cv.kfold(125, 10, np.random.default_rng(0))
        assert sorted(np.concatenate(folds)) == list(range(125)) and {len(f) for f in folds} == {12, 13}


class TestCV:
    def test_signal_found_and_noise_rejected(self, data):
        rng = np.random.default_rng(1)
        m1 = F.Recipe("m1", cols=F.M1_COLS)
        signal = 2 * data.col("tail_all") - data.col("cuts_per_s") + rng.normal(0, 1, len(data))
        noise = rng.normal(0, 1, len(data))
        good = cv.nested_cv(data, signal, m1, repeats=2)
        base = cv.nested_cv(data, signal, F.Recipe("mean_only"), repeats=2)
        assert good["spearman"].mean() > 0.7 and good["mse"].mean() < 0.5 * base["mse"].mean()
        assert cv.nested_cv(data, noise, m1, repeats=2)["mse"].mean() > 0.95 * cv.nested_cv(
            data, noise, F.Recipe("mean_only"), repeats=2)["mse"].mean()
        coef = cv.fit_full(data, signal, m1, 1.0)
        assert coef["tail_all"] > 1 and coef["cuts_per_s"] < 0

    def test_history_feature_does_not_leak_in_cv(self, data):
        y = np.random.default_rng(2).normal(0, 1, len(data))     # pure noise: H1 must not beat the mean
        r = cv.nested_cv(data, y, F.Recipe("m0", h1=True, recent_form=True), repeats=2)
        assert r["spearman"].mean() < 0.25

    def test_permutation_p(self, data):
        rng = np.random.default_rng(3)
        m1 = F.Recipe("m1", cols=F.M1_COLS)
        signal = 2 * data.col("tail_all") + rng.normal(0, 1, len(data))
        out = cv.permutation_test(data, signal, [m1, F.Recipe("date", cols=("years",))], n_perm=60)
        assert out["m1"]["p"] < 0.05 and out["m1"]["p_adjusted"] >= out["m1"]["p"]
        assert out["date"]["p_adjusted"] > 0.05
