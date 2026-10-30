import numpy as np
import pandas as pd
import pytest

from src.evals.hook_test.data import ExportTables
from src.evals.hook_test.select import build_frame, eligible_pool, label_candidates, select
from src.evals.hook_test.stats import mann_whitney_u


def synthetic_tables(n: int = 240, seed: int = 0) -> ExportTables:
    """n Brawl Stars Shorts with a planted length confound: shorter videos have higher averageViewPercentage."""
    rng = np.random.default_rng(seed)
    ids = [f"vid{i:03d}" for i in range(n)]
    dur = np.clip(np.round(rng.lognormal(np.log(22), 0.45, n)), 8, 60)  # channel-like: median ~22 s
    pub = pd.Timestamp("2023-08-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 570, n), unit="D")
    quality = rng.normal(0, 1, n)
    avp = 160 - 30 * np.log(dur) + 8 * quality
    rrp = 1 / (1 + np.exp(-1.5 * quality))
    videos = pd.DataFrame({
        "video_id": ids, "title": [f"brawl stars clip {i}" for i in range(n)],
        "published_at": pub.strftime("%Y-%m-%dT%H:%M:%SZ"), "content_type": "shorts", "duration_seconds": dur,
    })
    metrics = pd.DataFrame({"video_id": ids, "creatorContentType": "shorts", "engagedViews": 5000, "averageViewPercentage": avp})
    ratio = np.round(np.arange(1, 101) / 100, 2)
    retention = pd.DataFrame({
        "video_id": np.repeat(ids, 100), "elapsedVideoTimeRatio": np.tile(ratio, n),
        "audienceWatchRatio": np.tile(np.linspace(1.2, 0.4, 100), n), "relativeRetentionPerformance": np.repeat(rrp, 100),
    })
    daily = pd.DataFrame({"video_id": ids, "day": "2024-12-01", "engagedViews": 5000})
    return ExportTables(videos, metrics, retention, daily)


def test_planted_length_confound_is_removed():
    result = select(build_frame(synthetic_tables()), seed=340)
    v = result["videos"]
    for split in ("gate", "heldout"):
        rows = v[v["split"] == split]
        s, w = rows[rows["label"] == "strong"], rows[rows["label"] == "weak"]
        assert abs(s["duration_seconds"].median() - w["duration_seconds"].median()) <= 2
    # Raw averageViewPercentage would pick short videos as strong; check the confound really was planted.
    frame = build_frame(synthetic_tables())
    top = frame.nlargest(60, "averageViewPercentage")["duration_seconds"].median()
    bottom = frame.nsmallest(60, "averageViewPercentage")["duration_seconds"].median()
    assert bottom - top > 15


def test_splits_are_disjoint_balanced_and_sized():
    result = select(build_frame(synthetic_tables()), seed=340)
    v = result["videos"]
    ids = {s: set(v.loc[v["split"] == s, "video_id"]) for s in ("dev", "gate", "heldout")}
    assert not (ids["dev"] & ids["gate"]) and not (ids["dev"] & ids["heldout"]) and not (ids["gate"] & ids["heldout"])
    assert v["video_id"].is_unique
    assert result["splits"]["gate"] == {"strong": 20, "weak": 20}
    assert result["splits"]["dev"]["strong"] == result["splits"]["dev"]["weak"] in (6, 10)
    assert result["splits"]["heldout"]["strong"] == result["splits"]["heldout"]["weak"] >= 10
    assert v["blind_id"].is_unique and v["blind_id"].str.fullmatch(r"v[0-9a-f]{8}").all()


def test_rerandomization_picks_lowest_passing_score():
    result = select(build_frame(synthetic_tables()), seed=340)
    passing = [t for t in result["attempts"] if t["passed"]]
    assert len(result["attempts"]) == 50 and passing
    best = min(passing, key=lambda t: (t["score"], t["seed"]))
    assert result["seed_used"] == best["seed"]
    assert result["rerandomization"]["best_score"] == pytest.approx(best["score"])
    assert all(result["balance"][s]["passed"] for s in ("dev", "gate", "heldout"))


def test_same_seed_same_output():
    a = select(build_frame(synthetic_tables()), seed=340)["videos"]
    b = select(build_frame(synthetic_tables()), seed=340)["videos"]
    pd.testing.assert_frame_equal(a, b)


def test_eligibility_rules_exclude_planted_videos():
    t = synthetic_tables()
    t.videos.loc[t.videos["video_id"] == "vid000", "published_at"] = "2025-04-01T12:00:00Z"
    t.daily = pd.concat([t.daily, pd.DataFrame({"video_id": ["vid001"], "day": ["2025-06-01"], "engagedViews": [5000]})])
    t.metrics.loc[t.metrics["video_id"] == "vid002", "engagedViews"] = 500
    t.retention = t.retention.drop(t.retention.index[(t.retention["video_id"] == "vid003")][:1])
    t.videos.loc[t.videos["video_id"] == "vid004", "title"] = "dop 2 level 5"
    t.videos.loc[t.videos["video_id"] == "vid005", "duration_seconds"] = 75.0
    pool, funnel = eligible_pool(build_frame(t), None, 0.20)
    assert not {"vid000", "vid001", "vid002", "vid003", "vid004", "vid005"} & set(pool["video_id"])
    assert len(pool) == 240 - 6
    assert all(step["excluded"] == 1 for step in funnel[1:7])


def test_available_filter_applies_rule_5():
    frame = build_frame(synthetic_tables())
    available = set(frame["video_id"]) - {"vid010", "vid011"}
    pool, funnel = eligible_pool(frame, available, 0.20)
    assert len(pool) == 238 and funnel[-1] == {"rule": "video_file_available", "excluded": 2, "remaining": 238}


def test_labels_are_residual_quartiles_regardless_of_rrp():
    t = synthetic_tables()
    t.metrics.loc[t.metrics["video_id"] == "vid020", "averageViewPercentage"] += 200
    t.retention.loc[t.retention["video_id"] == "vid020", "relativeRetentionPerformance"] = 0.2
    t.metrics.loc[t.metrics["video_id"] == "vid021", "averageViewPercentage"] -= 200
    t.retention.loc[t.retention["video_id"] == "vid021", "relativeRetentionPerformance"] = 0.9
    pool, _ = eligible_pool(build_frame(t), None, 0.20)
    cands, info = label_candidates(pool)
    row = cands.set_index("video_id")
    assert row.loc["vid020", "label"] == "strong" and row.loc["vid021", "label"] == "weak"
    assert info["candidates"]["strong"] == info["candidates"]["weak"] == 60


def test_categorize_brawl_only_with_overrides():
    from src.evals.hook_test.categorize import categorize_video
    assert categorize_video("x", "Brawl Stars new brawler #brawlstars") == "brawl_stars"
    assert categorize_video("x", "test #clashroyale") == "clash_royale"
    assert categorize_video("x", "Dop 2 Level 121") == "misc_mobile_games"
    assert categorize_video("x", "#aliza") == "non_gaming_other"
    assert categorize_video("-RsBklJcj5g", "#aliza") == "brawl_stars"


def test_hook3s_reads_curve_at_three_seconds():
    t = synthetic_tables(n=40)
    frame = build_frame(t).set_index("video_id")
    d = frame.loc["vid000", "duration_seconds"]
    expected = np.interp(min(1, 3 / d), np.arange(1, 101) / 100, np.linspace(1.2, 0.4, 100))
    assert frame.loc["vid000", "hook3s"] == pytest.approx(expected)


def test_mann_whitney_identical_and_separated():
    assert mann_whitney_u([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])[1] == pytest.approx(1.0)
    assert mann_whitney_u(range(20), range(100, 120))[1] < 1e-6
