"""Classify exported videos into content categories from title keywords.

The channel's upload history mixes Brawl Stars / Clash Royale with dozens of
hyper-casual mobile games and some non-gaming posts. Game type is a selection
confound for the hook test (see plan/02-hook-test.md section 2), so every video
gets a ``category`` column that ``select.py`` can filter on.

Writes ``data/evals/hook_test/categories.csv``. Deterministic, no network.

    ./venv/bin/python -m src.evals.hook_test.categorize [--export DIR]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_EXPORT = Path("data/channel")
OUT_CSV = Path("data/evals/hook_test/categories.csv")

# Creator-confirmed Brawl Stars Shorts whose titles are hashtag-only (Sep 2025),
# so no keyword can catch them.
BRAWL_STARS_VIDEO_IDS = frozenset({
    "ra7rkIN7XqI", "nCUy_br_hVc", "SdbzY3th2fY", "4HGp6o8jb3U",
    "1L3rQUmf44Y", "wGVQzjMbCTM", "-RsBklJcj5g",
})

# First match wins. Keywords are lowercase regex alternations matched against
# the lowercased title; keep them specific enough to avoid cross-talk
# (e.g. "run"/"game" alone would match half the corpus, so they only count
# inside the game-pattern group after the Brawl Stars / Clash groups).
CATEGORY_PATTERNS: list[tuple[str, str]] = [
    ("brawl_stars", r"brawl"),
    ("clash_royale", r"clash"),
    (
        "misc_mobile_games",
        r"dop|delete one|gameplay|walkthrough|level ?\d|puzzle|simulator|race|"
        r"runner|running|stack|dash|angela|kitty|sonic|spider|skibidi|sakura|"
        r"zombie|barbie|minecraft|thief|finger|sandwich|juice|teeth|tall man|"
        r"rope man|donut|bike|bicycle|outfit|queen|squeeze|ice ?cream|"
        r"shape shifting|build a|talking",
    ),
]


def categorize_title(title: str) -> str:
    """Map one video title to a category; unmatched titles are 'non_gaming_other'."""
    t = (title or "").lower()
    for category, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, t):
            return category
    return "non_gaming_other"


def categorize_video(video_id: str, title: str) -> str:
    """Category for one video: manual Brawl Stars overrides first, then title keywords."""
    return "brawl_stars" if video_id in BRAWL_STARS_VIDEO_IDS else categorize_title(title)


def latest_export(base: Path) -> Path:
    """Newest timestamped export directory under base."""
    dirs = sorted(
        p for p in base.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{8}T\d{6}Z", p.name) and (p / "videos.csv").exists()
    )
    if not dirs:
        raise FileNotFoundError(f"no export with videos.csv under {base}")
    return dirs[-1]


ERA_CUTOFF = "2024-09-01"  # late-era dummy in the residual model (plan section 2.2)


def hold_resid(df: pd.DataFrame, fit_mask: pd.Series) -> pd.Series:
    """Residual of averageViewPercentage on [1, log(duration), is_late_era].

    The trend line is fit on ``fit_mask`` rows only (the reference class) and
    applied to every row, so e.g. a misc-games Short's residual means
    "vs the Brawl Stars trend for its length", not vs its own category.
    NaN where inputs are missing.
    """
    import numpy as np

    out = pd.Series(np.nan, index=df.index)
    ok = df["averageViewPercentage"].notna() & df["duration_seconds"].notna()
    fit = fit_mask & ok
    if fit.sum() < 10:
        log.warning("only %d rows to fit residual model; leaving hold_resid empty", fit.sum())
        return out
    X = np.column_stack(
        [
            np.ones(fit.sum()),
            np.log(df.loc[fit, "duration_seconds"]),
            (pd.to_datetime(df.loc[fit, "published_at"]) >= ERA_CUTOFF).astype(float),
        ]
    )
    coef, *_ = np.linalg.lstsq(X, df.loc[fit, "averageViewPercentage"], rcond=None)
    X_all = np.column_stack(
        [
            np.ones(ok.sum()),
            np.log(df.loc[ok, "duration_seconds"]),
            (pd.to_datetime(df.loc[ok, "published_at"]) >= ERA_CUTOFF).astype(float),
        ]
    )
    out.loc[ok] = df.loc[ok, "averageViewPercentage"] - X_all @ coef
    return out


def categorize_export(export_dir: Path) -> pd.DataFrame:
    """Category + key columns for every Short in one export directory."""
    videos = pd.read_csv(export_dir / "videos.csv")
    metrics = pd.read_csv(export_dir / "video_metrics.csv")
    shorts = videos[videos["content_type"] == "shorts"].copy()
    shorts["category"] = [categorize_video(v, t) for v, t in zip(shorts["video_id"], shorts["title"])]
    sm = metrics[metrics["creatorContentType"] == "shorts"][
        ["video_id", "engagedViews", "averageViewPercentage"]
    ]
    shorts = shorts.merge(sm, on="video_id", how="left")
    # Reference class ~ the plan's eligible pool minus the curve/post-change
    # checks (select.py re-fits on the true eligible set; this is for browsing).
    ref = (
        (shorts["category"] == "brawl_stars")
        & shorts["duration_seconds"].between(8, 60)
        & (pd.to_datetime(shorts["published_at"]) < "2025-03-01")
        & (shorts["engagedViews"] >= 1000)
    )
    shorts["hold_resid"] = hold_resid(shorts, ref)
    cols = [
        "video_id", "title", "category", "published_at", "duration_seconds",
        "engagedViews", "averageViewPercentage", "hold_resid",
    ]
    return shorts[cols].sort_values(["category", "published_at"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, help="export dir (default: newest under data/channel)")
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    export_dir = args.export or latest_export(DEFAULT_EXPORT)
    df = categorize_export(export_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("%s: %d shorts from %s", args.out, len(df), export_dir)
    for category, n in df["category"].value_counts().items():
        log.info("  %-18s %d", category, n)
    review = df[df["category"] == "non_gaming_other"]
    if len(review):
        log.info("non_gaming_other is a catch-all; eyeball it for miscategorized games.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
