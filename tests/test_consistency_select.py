"""Tests for consistency_select.py (pure functions, synthetic frames, no ffmpeg)."""

import pandas as pd
import pytest

from src.tools.analyze_video import consistency_select as cs


def fake_train(n=126):
    return pd.DataFrame({
        "video_id": [f"vid{i:03d}" for i in range(n)],
        "split": "train",
        "duration_seconds": [8.0 + (i * 0.4) % 52 for i in range(n)],
        "published_at": "2024-01-01",
    })


class TestBanding:
    def test_10_per_band_from_126(self):
        sel = cs.select_videos(fake_train())
        assert len(sel) == 30
        assert sel["band"].value_counts().to_dict() == {
            "short": 10, "middle": 10, "long": 10}

    def test_banding_is_thirds(self):
        df = cs.assign_bands(fake_train())
        assert df["band"].value_counts().to_dict() == {
            "short": 42, "middle": 42, "long": 42}

    def test_reproducible(self):
        a = cs.select_videos(fake_train())
        b = cs.select_videos(fake_train())
        assert a["video_id"].tolist() == b["video_id"].tolist()

    def test_band_order_in_duration(self):
        df = cs.assign_bands(fake_train())
        edges = cs.band_edges(df)
        assert edges["short"][1] <= edges["middle"][0]
        assert edges["middle"][1] <= edges["long"][0]


class TestSafety:
    def test_safe_cols_drop_outcomes(self):
        df = fake_train().assign(hold_resid=0.1, hold_score=50.0, engagedViews=100,
                                 mean_rrp=0.5, hook3s=1.0)
        out = df[cs.SAFE_COLS]
        for c in ("hold_resid", "hold_score", "engagedViews", "mean_rrp", "hook3s"):
            assert c not in out.columns

    def test_collision_check_raises(self):
        with pytest.raises(SystemExit, match="collision"):
            cs.check_blind_collisions({"a": "vdeadbeef"}, {"vdeadbeef", "v00000000"})
        cs.check_blind_collisions({"a": "v11111111"}, {"vdeadbeef"})  # no raise

    def test_leaked_tags(self):
        assert cs.leaked_tags({"title": "x", "encoder": "Lavf"}) == {"title": "x"}
        assert cs.leaked_tags({"encoder": "Lavf"}) == {}
        assert cs.leaked_tags({}) == {}
