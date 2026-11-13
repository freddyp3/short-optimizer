import pytest

from src.models.judge import ledger as L


def test_total_by_stage(tmp_path):
    p = tmp_path / "ledger.jsonl"
    L.record(1.5, "A", "x", p)
    L.record(0.25, "C", "y", p)
    assert L.total(p) == pytest.approx(1.75) and L.total(p, "C") == pytest.approx(0.25)
    assert L.total(tmp_path / "missing.jsonl") == 0.0


def test_call_crossing_stage_cap_is_refused(tmp_path):
    p = tmp_path / "ledger.jsonl"
    L.record(0.95, "C", "x", p)
    L.check(0.04, "C", 1.0, p)
    with pytest.raises(L.BudgetExceeded, match="stage C"):
        L.check(0.06, "C", 1.0, p)


def test_call_crossing_hard_cap_is_refused(tmp_path):
    p = tmp_path / "ledger.jsonl"
    L.record(49.99, "A", "x", p)
    with pytest.raises(L.BudgetExceeded, match="hard cap"):
        L.check(0.02, "C", 10.0, p)
