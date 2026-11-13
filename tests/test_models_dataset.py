"""Tests for src/models/dataset.py (pure functions, synthetic frames, no data/ access)."""

import numpy as np
import pandas as pd
import pytest

from src.models import dataset as ds


def frames(n=12):
    ids = [f"vid{i:02d}" for i in range(n)]
    rng = np.random.default_rng(0)
    pool = pd.DataFrame({
        "video_id": ids,
        "split": "train",
        "hold_resid": rng.normal(0, 5, n),
        "engagedViews": rng.integers(1_000, 1_000_000, n),
    })
    gemini = pd.DataFrame({
        "video_id": ids,
        "status": "ok",
        "invalid_reason": None,
        "source_run_id": "run",
        "t_premise_clear_s": [None, 2.0] * (n // 2),
        "dead_time_s": 1.0,
    })
    code = pd.DataFrame({
        "video_id": ids + ["extra"],
        "duration_s": [10.0 + 2 * i for i in range(n)] + [30.0],
        "cuts_per_s": 0.5,
    })
    return pool, gemini, code


class TestGuard:
    @pytest.mark.parametrize("split", ["gate", "heldout"])
    def test_locked_without_marker(self, split, tmp_path):
        with pytest.raises(ds.SplitLocked):
            ds.require_unlocked(split, tmp_path / "FROZEN")

    def test_gate_opens_with_marker_heldout_needs_its_own_line(self, tmp_path):
        marker = tmp_path / "FROZEN"
        marker.write_text("sha256=abc\n")
        ds.require_unlocked("gate", marker)
        with pytest.raises(ds.SplitLocked):
            ds.require_unlocked("heldout", marker)
        marker.write_text("sha256=abc\nheldout_open=true\n")
        ds.require_unlocked("heldout", marker)

    @pytest.mark.parametrize("split", ["train", "dev"])
    def test_train_and_dev_always_open(self, split, tmp_path):
        ds.require_unlocked(split, tmp_path / "FROZEN")

    def test_unknown_split(self, tmp_path):
        with pytest.raises(ValueError):
            ds.require_unlocked("test", tmp_path / "FROZEN")

    def test_load_split_checks_before_reading(self, tmp_path):
        with pytest.raises(ds.SplitLocked):
            ds.load_split("gate", marker=tmp_path / "FROZEN")


class TestHoldResidViews:
    def test_uncorrelated_with_log_views(self):
        pool, _, _ = frames(60)
        pool["hold_resid"] += 3 * np.log(pool["engagedViews"])
        out = ds.add_hold_resid_views(pool)
        r = np.corrcoef(out["hold_resid_views"], np.log(out["engagedViews"]))[0, 1]
        assert abs(r) < 1e-9
        assert abs(out["hold_resid_views"].mean()) < 1e-9

    def test_input_not_modified(self):
        pool, _, _ = frames()
        ds.add_hold_resid_views(pool)
        assert "hold_resid_views" not in pool


class TestAssemble:
    def test_one_row_per_pool_video(self):
        pool, gemini, code = frames()
        df = ds.assemble(pool, gemini, code)
        assert df["video_id"].tolist() == pool["video_id"].tolist()
        assert not set(ds.GEMINI_BOOKKEEPING) & set(df.columns) - {"split"}
        assert (df["split"] == "train").all()

    def test_null_premise_filled_with_duration(self):
        pool, gemini, code = frames()
        df = ds.assemble(pool, gemini, code)
        assert df["t_premise_clear_s"].iloc[0] == df["duration_s"].iloc[0]
        assert df["t_premise_clear_s"].iloc[1] == 2.0

    def test_is_short_boundary(self):
        pool, gemini, code = frames()
        df = ds.assemble(pool, gemini, code)
        assert df.loc[df["duration_s"] == 20.0, "is_short"].item() == 1.0
        assert df.loc[df["duration_s"] == 22.0, "is_short"].item() == 0.0

    def test_missing_gemini_row_raises(self):
        pool, gemini, code = frames()
        with pytest.raises(ValueError, match="no gemini features"):
            ds.assemble(pool, gemini.iloc[1:], code)

    def test_missing_code_row_raises(self):
        pool, gemini, code = frames()
        with pytest.raises(ValueError, match="no code features"):
            ds.assemble(pool, gemini, code.iloc[1:])

    def test_invalid_gemini_row_raises(self):
        pool, gemini, code = frames()
        gemini.loc[0, "status"] = "invalid"
        with pytest.raises(ValueError, match="status"):
            ds.assemble(pool, gemini, code)

    def test_duplicate_ids_raise(self):
        pool, gemini, code = frames()
        with pytest.raises(ValueError, match="duplicate"):
            ds.assemble(pool, pd.concat([gemini, gemini.iloc[:1]]), code)
