"""Tests for stats.py additions (plan section 10 numbers)."""

import numpy as np
import pytest

from src.evals.hook_test.stats import (
    auc, auc_permutation_p, binom_sf, mann_whitney_u, spearman, wilson_ci)


def test_binom_sf():
    assert binom_sf(26, 40) == pytest.approx(0.0403, abs=1e-4)
    assert binom_sf(48, 80) == pytest.approx(0.0465, abs=1e-3)
    assert binom_sf(25, 40) == pytest.approx(0.0769, abs=1e-4)


def test_wilson():
    lo, hi = wilson_ci(26, 40)
    assert lo == pytest.approx(0.495, abs=0.005)
    assert hi == pytest.approx(0.779, abs=0.005)


def test_auc_perfect_random_inverted():
    labels = np.array([0] * 20 + [1] * 20)
    s = np.arange(40)
    assert auc(labels, s) == 1.0
    assert auc(labels, -s) == 0.0
    rng = np.random.default_rng(0)
    rs = rng.normal(size=40)
    assert 0.2 < auc(labels, rs) < 0.8


def test_auc_permutation_p():
    labels = np.array([0] * 20 + [1] * 20)
    perfect = np.arange(40, dtype=float)
    assert auc_permutation_p(labels, perfect, n_perms=2000, seed=0) < 0.01
    # random predictions -> p roughly uniform across seeds
    ps = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        scores = rng.normal(size=40)
        ps.append(auc_permutation_p(labels, scores, n_perms=200, seed=seed))
    assert 0.4 < np.mean(ps) < 0.6


def test_mann_whitney_identical():
    x = np.array([1.0, 2, 3, 4, 5])
    _, p = mann_whitney_u(x, x.copy())
    assert p > 0.3  # identical samples -> not significant


def test_spearman():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
