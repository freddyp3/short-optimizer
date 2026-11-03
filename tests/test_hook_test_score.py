"""Tests for score.py on fabricated raw folders, and run.py's resume loop."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evals.hook_test import config as C
from src.evals.hook_test.run import run_calls
from src.evals.hook_test.score import aggregate, decide, load_raw, score_run

RESULT = lambda p, d: {
    "video_description": "d", "analysis": "a",
    "p_strong": p, "group": "strong" if p >= 0.5 else "weak", "confidence": "medium",
    "dropoff_timestamp_s": d, "hook_verdict": "ok",
    "video_length_s_estimate": 21.0, "moments": [], "reasoning": "x",
}


def make_selection(n=40, split="gate"):
    rows = []
    for i in range(n):
        rows.append({
            "video_id": f"vid{i:03d}", "split": split,
            "label": "strong" if i < n // 2 else "weak",
            "blind_id": f"b{i:03d}", "hold_resid": float(n // 2 - i),
            "duration_seconds": 21.0,
            "published_at": "2024-06-01T00:00:00Z",
        })
    return pd.DataFrame(rows)


def drop_r_for(i):
    """Per-video cliff ratio so permutation nulls are meaningful."""
    return 0.30 + 0.01 * i


def retention_cliff(vids):
    """Retention df: each video's curve drops at drop_r_for(its index)."""
    rows = []
    for i, vid in enumerate(vids):
        dr = drop_r_for(i)
        for j in range(1, 101):
            r = j / 100
            rows.append({"video_id": vid, "elapsedVideoTimeRatio": r,
                         "audienceWatchRatio": 1.0 if r < dr else 0.55,
                         "relativeRetentionPerformance": 1.0})
    return pd.DataFrame(rows)


def write_raw(raw_dir: Path, selection: pd.DataFrame, run_id="test_run",
              n_correct=None, runs=3, invalid_ids=(), dropoff=7.0):
    # dropoff may be a float or a callable of the selection row index
    """Fabricate raw json files; strong videos get p~0.8, weak ~0.2. Flip the
    same number of strong->low and weak->high so the rank-based call gets
    exactly n_correct right."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "settings.json").write_text(json.dumps(
        {"split": selection["split"].iloc[0]}))
    n = len(selection)
    if n_correct is not None and n_correct <= 1:  # fraction shorthand
        n_correct = round(n * n_correct)
    n_flip = (n - (n if n_correct is None else n_correct)) // 2
    strong_idx = [i for i, (_, v) in enumerate(selection.iterrows()) if v["label"] == "strong"]
    weak_idx = [i for i, (_, v) in enumerate(selection.iterrows()) if v["label"] == "weak"]
    flip_idx = set(strong_idx[:n_flip] + weak_idx[:n_flip])
    for i, (_, v) in enumerate(selection.iterrows()):
        hi = (v["label"] == "strong") != (i in flip_idx)
        for k in range(1, runs + 1):
            status = "invalid" if v["blind_id"] in invalid_ids else "ok"
            rec = {
                "blind_id": v["blind_id"], "run_k": k, "status": status,
                "model": "gemini-3.8-flash", "prompt_version": "hook_test_v1",
                "fps": 5.0, "media_resolution": "MEDIA_RESOLUTION_HIGH",
                "usage": {"total_tokens": 34000}, "cost_usd_paid_equiv": 0.04,
                "cost_usd_actual": 0.0, "run_id": run_id,
                "result": None if status == "invalid" else RESULT(
                    0.8 if hi else 0.2,
                    dropoff(i) if callable(dropoff) else dropoff),
            }
            if status == "invalid":
                rec["invalid_reason"] = "schema"
            (raw_dir / f"{v['blind_id']}__r{k}.json").write_text(json.dumps(rec))


def score_gate(tmp_path, correct, invalid_ids=(), dropoff=7.0, retention=True):
    sel = make_selection()
    raw_dir = tmp_path / "raw" / "run1"
    write_raw(raw_dir, sel, n_correct=correct, invalid_ids=invalid_ids,
              dropoff=dropoff)
    ret = retention_cliff(sel["video_id"]) if retention else pd.DataFrame()
    return score_run(raw_dir, "run1", selection=sel,
                     titles={v: f"title {v}" for v in sel["video_id"]},
                     retention=ret, out_dir=tmp_path / "out", n_perms=500)


class TestAggregate:
    def test_mean_median(self, tmp_path):
        sel = make_selection(4)
        raw_dir = tmp_path / "r"
        raw_dir.mkdir()
        for k, p in [(1, 0.6), (2, 0.8), (3, 0.7)]:
            (raw_dir / f"b000__r{k}.json").write_text(json.dumps({
                "blind_id": "b000", "run_k": k, "status": "ok",
                "result": RESULT(p, float(k)),
                "usage": {"total_tokens": 1}, "cost_usd_paid_equiv": 0}))
        raw = load_raw(raw_dir)
        agg = aggregate(raw).set_index("blind_id").loc["b000"]
        assert agg["p_strong_mean"] == pytest.approx(0.7)
        assert agg["dropoff_pred_s"] == 2.0
        assert agg["n_valid"] == 3


class TestDecide:
    def test_tree(self):
        assert decide(26, False, False) == "GO"
        assert decide(40, True, False) == "GO"
        assert decide(25, True, False) == "CONDITIONAL_GO"
        assert decide(24, False, False) == "AMBIGUOUS_RUN_HELDOUT"
        assert decide(20, True, False) == "CONDITIONAL_GO_SCOPED"
        assert decide(20, False, False) == "NO_GO"
        assert decide(30, False, True) == "INVALID_CONFOUNDED"


class TestScoreRun:
    def test_go(self, tmp_path):
        s = score_gate(tmp_path, correct=40)
        assert s["criterion_A"]["correct"] == 40
        assert s["decision"] == "GO"
        assert s["criterion_A"]["auc"] == 1.0
        assert (tmp_path / "out" / "scores.json").exists()
        assert "Caveats" in (tmp_path / "out" / "report.md").read_text()

    def test_28_correct_go(self, tmp_path):
        # rank-based calls on a balanced set can only give even counts
        s = score_gate(tmp_path, correct=28)
        assert s["criterion_A"]["correct"] == 28
        assert s["decision"] == "GO"

    def test_24_correct_b_fails(self, tmp_path):
        s = score_gate(tmp_path, correct=24, dropoff=1.0)
        # dropoff=1.0s lands in the swipe-out region; rank_score ~ mid, T_obs < 0.60
        assert s["decision"] in {"AMBIGUOUS_RUN_HELDOUT", "NO_GO",
                                 "CONDITIONAL_GO"}

    def test_invalid_videos_flag(self, tmp_path):
        s = score_gate(tmp_path, correct=1.0, invalid_ids={"b000", "b001", "b002"})
        assert s["n_invalid_videos"] == 3
        assert s["decision"] == "INVALID_INFRA"

    def test_criterion_b_pass_on_true_cliff(self, tmp_path):
        # predict each video's own drop-off (t* ~ drop_t - ~0.7 s)
        s = score_gate(tmp_path, correct=1.0,
                       dropoff=lambda i: 21 * drop_r_for(i) - 0.7)
        B = s["criterion_B"]
        assert B["T_obs"] > 0.9
        assert B["pass"]
        assert B["null1_uniform_p"] < 0.05 and B["null2_permutation_p"] < 0.05

    def test_metadata_baselines_present(self, tmp_path):
        s = score_gate(tmp_path, correct=1.0)
        assert "duration_only" in s["metadata_baselines"]
        assert not s["confounded"]

    def test_stability_reported(self, tmp_path):
        s = score_gate(tmp_path, correct=1.0)
        assert s["stability"]["mean_pairwise_spearman"] == pytest.approx(1.0)
        assert s["stability"]["group_flip_share"] == 0.0


class FakeTester:
    def __init__(self):
        self.calls = []

    def analyze(self, blind_id, path, prompt_text, duration_s, **kw):
        self.calls.append(blind_id)
        return {"blind_id": blind_id, "status": "ok", "result": RESULT(0.5, 1.0),
                "usage": {"total_tokens": 1}, "cost_usd_paid_equiv": 0.0,
                "cost_usd_actual": 0.0}


def run_settings(tmp_path):
    return {"run_id": "r1", "split": "gate", "model": "gemini-3.8-flash",
            "tier": "free", "fps": 5.0, "media_resolution": "MEDIA_RESOLUTION_HIGH",
            "prompt_version": "v1", "prompt_sha256": "x", "seed_base": C.SEED}


class TestResume:
    def test_resume_skips_existing(self, tmp_path):
        sel = make_selection(4)
        run_dir = tmp_path / "raw" / "r1"
        run_dir.mkdir(parents=True)
        (run_dir / "b000__r1.json").write_text(json.dumps({"blind_id": "b000"}))
        t = FakeTester()
        rc = run_calls(t, sel, run_settings(tmp_path), "prompt", run_dir, runs=1,
                       blind_dir=tmp_path)
        assert rc == 0
        assert "b000" not in t.calls
        assert len(t.calls) == 3
        assert len(list(run_dir.glob("*__r1.json"))) == 4

    def test_seeded_order(self, tmp_path):
        sel = make_selection(6)
        run_dir = tmp_path / "raw" / "r1"
        t = FakeTester()
        run_calls(t, sel, run_settings(tmp_path), "p", run_dir, runs=1,
                  blind_dir=tmp_path)
        assert sorted(t.calls) == sorted(f"b{i:03d}" for i in range(6))
        assert t.calls != [f"b{i:03d}" for i in range(6)]  # shuffled


# --------------------------------------------------------------- v2 schema

V2_RESULT = lambda avp, d: {
    "video_description": "d", "analysis": "a",
    "hook_verdict": "ok", "dropoff_timestamp_s": d,
    "predicted_avg_view_pct": avp, "confidence": "medium",
    "video_length_s_estimate": 21.0, "moments": [], "reasoning": "x",
}


def write_raw_v2(raw_dir: Path, selection: pd.DataFrame, avp_fn, runs=3,
                 seed=0, dropoff=7.0):
    """avp_fn(i, k, label) -> predicted_avg_view_pct for video i, run k."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "settings.json").write_text(json.dumps(
        {"split": selection["split"].iloc[0]}))
    for i, (_, v) in enumerate(selection.iterrows()):
        for k in range(1, runs + 1):
            rec = {
                "blind_id": v["blind_id"], "run_k": k, "status": "ok",
                "schema_version": "v2",
                "model": "gemini-3.8-flash", "prompt_version": "hook_test_v2",
                "fps": 5.0, "media_resolution": "MEDIA_RESOLUTION_HIGH",
                "usage": {"total_tokens": 34000}, "cost_usd_paid_equiv": 0.04,
                "cost_usd_actual": 0.0, "run_id": "v2run",
                "result": V2_RESULT(avp_fn(i, k, v["label"]), dropoff),
            }
            (raw_dir / f"{v['blind_id']}__r{k}.json").write_text(json.dumps(rec))


def make_selection_v2(n=40):
    sel = make_selection(n)
    # spread durations/dates so the residual fit is meaningful
    rng = np.random.default_rng(1)
    sel["duration_seconds"] = rng.uniform(8, 60, n)
    sel["hold_score"] = 60 - 0.8 * np.log(sel["duration_seconds"]) * 20 \
        + sel["hold_resid"] * 2
    return sel


class TestV2Scoring:
    def test_residual_ranking_beats_length_effect(self, tmp_path):
        sel = make_selection_v2()
        rng = np.random.default_rng(0)
        # pred_avp = length trend + label signal + noise
        def avp(i, k, label):
            dur = sel.iloc[i]["duration_seconds"]
            return (70 - 0.9 * np.log(dur) * 20
                    + (12 if label == "strong" else -12)
                    + rng.normal(0, 3))
        raw_dir = tmp_path / "raw" / "v2"
        write_raw_v2(raw_dir, sel, avp)
        s = score_run(raw_dir, "v2", selection=sel,
                      titles={v: "t" for v in sel["video_id"]},
                      retention=retention_cliff(sel["video_id"]),
                      out_dir=tmp_path / "out", n_perms=500)
        assert s["schema_version"] == "v2"
        # residuals keep the label signal
        assert s["criterion_A"]["correct"] >= 30
        assert s["criterion_A"]["spearman_hold_resid"] > 0.7
        # length effect removed
        assert abs(s["v2_diagnostics"]["spearman_score_duration"]) < 0.2
        # raw pred_avp is still length-correlated
        assert s["v2_diagnostics"]["spearman_pred_avp_duration"] < -0.5
        assert s["criterion_A"]["hard_group_accuracy"] is None

    def test_pure_length_predictions_auc_half(self, tmp_path):
        sel = make_selection_v2()
        # pred_avp driven ONLY by duration -> residual ~ 0 -> rank ~ random
        def avp(i, k, label):
            return 90 - 30 * np.log(sel.iloc[i]["duration_seconds"])
        raw_dir = tmp_path / "raw" / "v2len"
        write_raw_v2(raw_dir, sel, avp)
        s = score_run(raw_dir, "v2len", selection=sel,
                      titles={v: "t" for v in sel["video_id"]},
                      retention=retention_cliff(sel["video_id"]),
                      out_dir=tmp_path / "out2", n_perms=200)
        assert abs(s["criterion_A"]["auc"] - 0.5) < 0.15

    def test_v2_diagnostics_present(self, tmp_path):
        sel = make_selection_v2()
        raw_dir = tmp_path / "raw" / "v2d"
        write_raw_v2(raw_dir, sel, lambda i, k, l: 60.0 + i)
        s = score_run(raw_dir, "v2d", selection=sel,
                      titles={v: "t" for v in sel["video_id"]},
                      retention=retention_cliff(sel["video_id"]),
                      out_dir=tmp_path / "out3", n_perms=100)
        d = s["v2_diagnostics"]
        for key in ("spearman_pred_avp_hold_score", "mae_pred_avp_hold_score",
                    "spearman_pred_avp_duration", "spearman_score_duration",
                    "pred_avp_summary"):
            assert key in d
        assert d["pred_avp_summary"]["max"] == 60.0 + 39
