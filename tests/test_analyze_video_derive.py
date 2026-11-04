"""Tests for derive.py: derived payoff/ending features (plan/03b section 1)."""

import copy

import pytest

from src.tools.analyze_video.derive import derive_features, resolve_ending

DUR = 20.0


def po(t, kind="reveal", w=1):
    return {"t_s": t, "kind": kind, "weight": w}


def result(**kw):
    r = {
        "timeline": "scratch", "notes": "n", "uncertain_fields": ["x"],
        "payoffs": [], "opening_frame": "mid_action", "ending": "short_tail",
        "arc": "single_peak", "dead_time_s": 1.0,
    }
    r.update(kw)
    return r


class TestResolveEnding:
    @pytest.mark.parametrize("g", ["outro_or_black", "result_screen_hold", "cliffhanger"])
    @pytest.mark.parametrize("tail", [0.0, 2.0, 30.0])
    def test_semantic_endings_survive_any_tail(self, g, tail):
        assert resolve_ending(g, tail) == (g, "gemini")

    @pytest.mark.parametrize("tail,expected", [
        (0.8, "cut_on_payoff"), (1.0, "cut_on_payoff"),
        (2.9, "short_tail"), (3.0, "short_tail"), (3.1, "long_tail"),
    ])
    def test_tail_buckets(self, tail, expected):
        for g in ("cut_on_payoff", "short_tail", "long_tail"):
            assert resolve_ending(g, tail) == (expected, "code")


class TestCounts:
    def test_clear_and_all(self):
        r = result(payoffs=[po(2, w=1), po(8, w=2), po(15, w=3)])
        out = derive_features(r, DUR, [])
        assert out["payoff_count"] == 2
        assert out["payoff_count_all"] == 3
        assert out["no_payoff"] is False

    def test_no_payoffs(self):
        out = derive_features(result(), DUR, [])
        assert out["payoff_count"] == 0 and out["payoff_count_all"] == 0
        assert out["no_payoff"] is True


class TestMainPayoff:
    def test_weight3_wins(self):
        r = result(payoffs=[po(5, w=2), po(15, kind="punchline", w=3)])
        out = derive_features(r, DUR, [])
        assert out["t_main_payoff_s"] == 15.0
        assert out["main_payoff_frac"] == 0.75
        assert out["main_payoff_kind"] == "punchline"

    def test_no_weight3_falls_back_to_max_weight(self):
        r = result(payoffs=[po(5, w=1), po(9, kind="kill_or_win", w=2)])
        out = derive_features(r, DUR, [])
        assert out["t_main_payoff_s"] == 9.0
        assert out["main_payoff_kind"] == "kill_or_win"

    def test_tie_goes_to_earliest(self):
        r = result(payoffs=[po(9, w=2), po(14, w=2)])
        out = derive_features(r, DUR, [])
        assert out["t_main_payoff_s"] == 9.0

    def test_no_payoffs_frac_is_one(self):
        out = derive_features(result(), DUR, [])
        assert out["t_main_payoff_s"] is None
        assert out["main_payoff_frac"] == 1.0
        assert out["main_payoff_kind"] is None


class TestTails:
    def test_tail_all_and_clear(self):
        r = result(payoffs=[po(15, w=2), po(18, w=1)])
        out = derive_features(r, DUR, [])
        assert out["tail_all"] == 2.0      # last payoff of any weight
        assert out["tail_clear"] == 5.0    # last weight>=2 payoff

    def test_tail_clear_falls_back_to_any_weight(self):
        r = result(payoffs=[po(18, w=1)])
        out = derive_features(r, DUR, [])
        assert out["tail_clear"] == 2.0

    def test_no_payoffs_tails_are_duration(self):
        out = derive_features(result(), DUR, [])
        assert out["tail_all"] == DUR and out["tail_clear"] == DUR

    def test_endings_from_both_tails(self):
        r = result(ending="short_tail", payoffs=[po(15, w=2), po(18, w=1)])
        out = derive_features(r, DUR, [])
        assert out["ending_from_tail_all"] == "short_tail"   # tail 2.0
        assert out["ending_from_tail_clear"] == "long_tail"  # tail 5.0
        assert out["ending_gemini"] == "short_tail"

    def test_kept_ending_survives(self):
        r = result(ending="cliffhanger", payoffs=[po(18, w=1)])
        out = derive_features(r, DUR, [])
        assert out["ending_from_tail_all"] == "cliffhanger"


class TestSnap:
    def test_snaps_only_snap_kinds(self):
        r = result(payoffs=[po(9.7, kind="reveal", w=3), po(14.8, kind="kill_or_win", w=1)])
        out = derive_features(r, DUR, cuts=[10.0])
        assert out["t_main_payoff_s_snap"] == 10.0          # reveal snapped
        assert out["tail_all_snap"] == DUR - 14.8           # kill_or_win not snapped

    def test_no_snap_beyond_half_second(self):
        r = result(payoffs=[po(9.4, kind="reveal", w=3)])
        out = derive_features(r, DUR, cuts=[10.0])
        assert out["t_main_payoff_s_snap"] == 9.4

    def test_snap_tie_goes_earlier(self):
        r = result(payoffs=[po(10.0, kind="reveal", w=3)])
        out = derive_features(r, DUR, cuts=[9.5, 10.5])
        assert out["t_main_payoff_s_snap"] == 9.5

    def test_snap_recomputes_tails(self):
        r = result(payoffs=[po(14.8, kind="punchline", w=3), po(17.9, kind="reveal", w=1)])
        out = derive_features(r, DUR, cuts=[15.0, 18.0])
        assert out["tail_all_snap"] == pytest.approx(2.0)
        assert out["tail_clear_snap"] == pytest.approx(5.0)
        assert out["main_payoff_frac_snap"] == pytest.approx(0.75)


class TestPassthrough:
    def test_scalar_fields_pass_through(self):
        r = result(opening_frame="menu_or_lobby", arc="builds", dead_time_s=4.0)
        out = derive_features(r, DUR, [])
        assert out["opening_frame"] == "menu_or_lobby"
        assert out["arc"] == "builds" and out["dead_time_s"] == 4.0
        for k in ("timeline", "notes", "uncertain_fields", "payoffs"):
            assert k not in out

    def test_does_not_mutate_input(self):
        r = result(payoffs=[po(9.7, kind="reveal", w=3)])
        snapshot = copy.deepcopy(r)
        derive_features(r, DUR, cuts=[10.0])
        assert r == snapshot
