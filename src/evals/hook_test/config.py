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
# Rerandomization: try seeds SEED..SEED+N-1, keep draws where every split passes the rules above,
# pick the lowest score = max |standardized mean difference| (strong vs weak, pool SD) over
# {dev, gate, heldout} x {duration, publish date}. Ties -> lowest seed.
RERANDOMIZATION_SEEDS = 50
BALANCED_SPLITS = ("dev", "gate", "heldout")

# --- Phase 4-8: Gemini runner / scoring (sections 4, 7, 8) -------------------
import os as _os

DEFAULT_MODEL = "gemini-3.8-flash"
MODELS = ("gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-4-argon")
# (input $/M tokens, output $/M tokens). Actual spend is 0 on the free tier;
# these are nominal paid-equivalent prices for cost reporting.
PRICES_USD_PER_M = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-4-argon": (2.0, 10.0),
}
# Verified against google-genai 2.32.0 source (types.py): per-video fps goes on
# the Part as `video_metadata=types.VideoMetadata(fps=...)`; media resolution is
# also per-Part: `media_resolution=types.PartMediaResolution(level=...)` with
# PartMediaResolutionLevel.MEDIA_RESOLUTION_{LOW,MEDIUM,HIGH,ULTRA_HIGH}.
FPS = 5.0
MEDIA_RESOLUTION = "MEDIA_RESOLUTION_HIGH"
GEMINI_TIER = _os.environ.get("GEMINI_TIER", "free")
BUDGET_CAP_USD = 30.0          # enforced against nominal cost only when tier != "free"
MAX_RETRIES = 6                # ~429s before declaring daily-quota exhaustion
UPLOAD_TTL_H = 46              # Files API entries expire ~48 h
ASSUMED_TOKENS_PER_FRAME = 300 # high-res estimate (plan 4.1); refine via count_tokens
AUDIO_TOKENS_PER_S = 32
PROMPT_TOKENS_EST = 1500
OUTPUT_TOKENS_EST = 3600       # ~600 JSON + up to ~3000 thinking

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "hook_test_v1.txt"
BLIND_DIR = OUT_DIR / "blind"
RAW_DIR = OUT_DIR / "raw"
RESULTS_DIR = OUT_DIR / "results"
UPLOADS_JSON = OUT_DIR / "uploads.json"
DEV_LOG = OUT_DIR / "dev_log.jsonl"
SELECTION_CSV_PATH = OUT_DIR / "selection.csv"
SELECTION_JSON_PATH = OUT_DIR / "selection.json"
VIDEO_MAP_PATH = OUT_DIR / "video_map.csv"

# Gate thresholds (section 7 / 12)
GATE_N, GATE_PASS = 40, 26
AMBIGUOUS_LO = 23              # A in 23..25 -> AMBIGUOUS
T_OBS_MIN = 0.60
P_MAX = 0.05
POOLED_N, POOLED_PASS = 80, 48
PERMS = 10_000
DROPOFF_MIN_T_S = 1.5          # t*_post restricted to t >= 1.5 s
FLAT_CURVE_MIN_RANGE = 0.03    # max(loss) - median(loss) below this -> flat_curve

# prompt version -> schema version (unknown prompt files default to v1)
PROMPT_SCHEMAS = {"hook_test_v1": "v1", "hook_test_v2": "v2"}
