"""Hook test constants (plan/02-hook-test.md). Freeze before the GATE run; never tune after seeing results."""

from __future__ import annotations

from pathlib import Path

SEED = 340
EXPORT_BASE = Path("data/channel")
OUT_DIR = Path("data/evals/hook_test")

# 2.1 Eligible pool
REFERENCE_CATEGORY = "brawl_stars"
MIN_DURATION_S, MAX_DURATION_S = 8, 60
PUBLISHED_BEFORE = "2025-03-01"
VIEW_CHANGE_DATE = "2025-03-31"
MAX_POST_CHANGE_SHARE = 0.20
RELAXED_POST_CHANGE_SHARE = 0.35
MIN_POOL_BEFORE_RELAX = 120
MIN_ENGAGED_VIEWS = 1000
CURVE_POINTS = 100

# 2.2 Label metric
ERA_CUTOFF = "2024-09-01"
CANDIDATE_QUANTILE = 0.25
HOOK_SECONDS = 3.0

# 2.4 Splits (pairs = one strong + one weak)
DEV_PAIRS, GATE_PAIRS = 10, 20
HELDOUT_MIN_PAIRS = 15
DEV_PAIRS_SHRUNK, HELDOUT_MIN_PAIRS_SHRUNK = 6, 10

# Balance checks, strong vs weak
BALANCE_MIN_P = 0.2
BALANCE_MAX_MEDIAN_DURATION_DIFF_S = 2.0
MAX_SEED_TRIES = 20
