"""Tests for duplicates.py scoring/verdicts, exclusion_plan, and pool exclusion."""

import json

import pytest

from src.models import pool as pool_mod
from src.tools.analyze_video import duplicates as D


def seq(vals):
    """A 'hash sequence' as plain ints; hamming = popcount(xor) but for tests we
    use 0 for identical and craft distances via bit patterns."""
    return list(vals)


class TestScorePair:
    def test_identical(self):
        h = seq([0xAAA, 0xBBB, 0xCCC, 0xDDD, 0xEEE])
        s = D.score_pair(h, list(h))
        assert s["best_offset_s"] == 0 and s["median_hamming"] == 0.0
        assert s["match_frac"] == 1.0 and s["n_aligned"] == 5

    def test_offset_by_one(self):
        a = seq([0xA1, 0xB2, 0xC3, 0xD4])
        b = seq([0x999, 0xA1, 0xB2, 0xC3, 0xD4])  # same content shifted +1 frame
        s = D.score_pair(a, b)
        assert s["best_offset_s"] == 1
        assert s["median_hamming"] == 0.0
        assert s["match_frac"] == 1.0 and s["n_aligned"] == 4

    def test_unrelated(self):
        a = seq([0xAAAAAAAAAAAAAAAA] * 4)
        b = seq([0x5555555555555555] * 4)  # hamming 64 everywhere, every offset
        s = D.score_pair(a, b)
        assert s["median_hamming"] > 0
        assert D.verdict(s) is None


class TestVerdict:
    def mk(self, med, frac):
        return {"median_hamming": med, "match_frac": frac}

    def test_duplicate(self):
        assert D.verdict(self.mk(8, 0.8)) == "duplicate"
        assert D.verdict(self.mk(0, 1.0)) == "duplicate"

    def test_near(self):
        assert D.verdict(self.mk(9, 0.8)) == "near"    # fails dup frac? no: med 9 > 8
        assert D.verdict(self.mk(16, 0.4)) == "near"
        assert D.verdict(self.mk(20, 0.5)) == "near"

    def test_none(self):
        assert D.verdict(self.mk(20, 0.4)) is None
        assert D.verdict(self.mk(9, 0.7)) == "near"    # med <= 16 still near


class TestExclusion:
    def info(self):
        return {
            "t1": {"split": "train", "published_at": "2024-01-01"},
            "t2": {"split": "train", "published_at": "2024-06-01"},
            "t3": {"split": "train", "published_at": "2024-03-01"},
            "e1": {"split": "dev", "published_at": "2024-02-01"},
            "e2": {"split": "gate", "published_at": "2024-04-01"},
        }

    def test_train_only_keeps_earliest(self):
        plan = D.exclusion_plan([("t1", "t2"), ("t2", "t3")], self.info())
        assert plan["exclude"] == ["t2", "t3"]
        assert plan["groups"][0]["keep"] == "t1"
        assert plan["eval_duplicates"] == []

    def test_train_eval_drops_train(self):
        plan = D.exclusion_plan([("t1", "e1")], self.info())
        assert plan["exclude"] == ["t1"]
        assert "e1" not in plan["exclude"]

    def test_eval_eval_reported_only(self):
        plan = D.exclusion_plan([("e1", "e2")], self.info())
        assert plan["exclude"] == []
        assert plan["eval_duplicates"] == [sorted(("e1", "e2"))]

    def test_train_eval_eval_drops_train_and_reports(self):
        plan = D.exclusion_plan([("t1", "e1"), ("e1", "e2")], self.info())
        assert plan["exclude"] == ["t1"]
        assert plan["eval_duplicates"] == [sorted(("e1", "e2", "t1"))] or \
            plan["eval_duplicates"] == [["e1", "e2", "t1"]]


class TestPoolExclusion:
    def test_train_rows_applies_exclude_file(self, monkeypatch, tmp_path):
        df = __import__("pandas").DataFrame({
            "video_id": ["a", "b", "c"],
            "split": "train", "duration_seconds": 10.0,
            "published_at": "2024-01-01", "hold_resid": 1.0,
        })
        monkeypatch.setattr(pool_mod, "labelled_pool", lambda **kw: df)
        excl = tmp_path / "duplicates_exclude.json"
        excl.write_text(json.dumps({"exclude": ["b"], "groups": [], "eval_duplicates": []}))
        monkeypatch.setattr(pool_mod, "DUPLICATES_EXCLUDE", excl)
        out = pool_mod.train_rows()
        assert out["video_id"].tolist() == ["a", "c"]
        out2 = pool_mod.train_rows(exclude_duplicates=False)
        assert len(out2) == 3
