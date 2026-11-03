"""Tests for schema.py validation (plan section 10)."""

import pytest

from src.evals.hook_test.schema import HookTestResult, validate_result

GOOD = {
    "video_description": "gameplay with a hook",
    "analysis": "hook is strong, mid sags",
    "p_strong": 0.8,
    "group": "strong",
    "confidence": "medium",
    "dropoff_timestamp_s": 7.0,
    "hook_verdict": "strong",
    "video_length_s_estimate": 21.0,
    "moments": [{"t_seconds": 1.0, "kind": "hook", "note": "big damage number"}],
    "reasoning": "strong hook and sustained action",
}


def test_valid():
    r, reason, flags = validate_result(GOOD, 21.0)
    assert reason is None and r.p_strong == 0.8 and flags == []


def test_dropoff_out_of_range_invalid_not_clamped():
    d = dict(GOOD, dropoff_timestamp_s=22.0)
    r, reason, _ = validate_result(d, 21.0)
    assert r is None and "dropoff" in reason


def test_p_strong_out_of_range_invalid():
    r, reason, _ = validate_result(dict(GOOD, p_strong=1.5), 21.0)
    assert r is None and reason


def test_missing_field_fails():
    d = dict(GOOD)
    del d["p_strong"]
    r, reason, _ = validate_result(d, 21.0)
    assert r is None


def test_extra_field_fails():
    r, reason, _ = validate_result(dict(GOOD, title="leak"), 21.0)
    assert r is None


def test_length_estimate_flag_not_invalid():
    r, reason, flags = validate_result(dict(GOOD, video_length_s_estimate=40.0), 21.0)
    assert r is not None and reason is None
    assert any("length_est_off" in f for f in flags)


def test_bad_enum_fails():
    r, _, _ = validate_result(dict(GOOD, hook_verdict="meh"), 21.0)
    assert r is None


GOOD_V2 = {
    "video_description": "desc",
    "analysis": "reasoning",
    "hook_verdict": "ok",
    "dropoff_timestamp_s": 7.0,
    "predicted_avg_view_pct": 75.0,
    "confidence": "medium",
    "video_length_s_estimate": 21.0,
    "moments": [{"t_seconds": 1.0, "kind": "hook", "note": "x"}],
    "reasoning": "ok",
}


def test_v2_valid():
    r, reason, flags = validate_result(GOOD_V2, 21.0, "v2")
    assert reason is None and r.predicted_avg_view_pct == 75.0


def test_v2_avp_range():
    r, reason, _ = validate_result(dict(GOOD_V2, predicted_avg_view_pct=401), 21.0, "v2")
    assert r is None and reason
    r, reason, _ = validate_result(dict(GOOD_V2, predicted_avg_view_pct=-1), 21.0, "v2")
    assert r is None


def test_v2_extra_field_rejected():
    r, reason, _ = validate_result(dict(GOOD_V2, p_strong=0.5), 21.0, "v2")
    assert r is None and "extra field" in reason


def test_v1_rejects_v2_fields():
    r, reason, _ = validate_result(GOOD_V2, 21.0, "v1")
    assert r is None


def test_schema_class_selection():
    from src.evals.hook_test.schema import HookTestResult, HookTestResultV2, schema_class
    assert schema_class("v1") is HookTestResult
    assert schema_class("v2") is HookTestResultV2
    assert schema_class("unknown") is HookTestResult
