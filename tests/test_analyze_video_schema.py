"""Tests for analyze_video's VideoFeatures schema + validate_features (no network)."""

import pytest

from src.tools.analyze_video.schema import validate_features

DURATION = 20.0


def valid_data(duration=DURATION):
    return {
        "timeline": "0.0-1.0 intro; 19.0 payoff",
        "payoffs": [{"t_s": duration - 1.0, "kind": "kill_or_win", "weight": 3}],
        "format_macro": "gameplay_clip",
        "voice": "none",
        "text_role": "none",
        "t_first_event_s": 0.5,
        "opening_frame": "mid_action",
        "t_premise_clear_s": 1.0,
        "premise_type": "implicit_situation",
        "premise_legible_muted": True,
        "premise_legible_nonplayer": True,
        "arc": "single_peak",
        "resolution": "resolved",
        "twist": False,
        "surprise_peak": "mild",
        "dead_time_s": 1.0,
        "beat_before_main_payoff": True,
        "info_new_per_5s": "medium",
        "main_payoff_legible_muted": True,
        "ending": "cut_on_payoff",
        "loop_seam": "compatible",
        "rewatch_trigger": "none",
        "audio_bed": "game_only_or_none",
        "audio_visual_sync": False,
        "t_audio_change_s": None,
        "audio_carries_payoff": False,
        "primary_appeal": "skill_hype",
        "relatability": "shared_player_experience",
        "game_moment_kind": "elimination",
        "big_number_on_screen": True,
        "low_health_visible": False,
        "result_screen_shown": True,
        "uncertain_fields": [],
        "notes": "",
    }


class TestValid:
    def test_valid_passes_no_flags(self):
        f, reason, flags = validate_features(valid_data(), DURATION)
        assert reason is None and flags == []
        assert f.payoffs[0].weight == 3


class TestInvalid:
    def test_timestamp_beyond_duration(self):
        d = valid_data()
        d["t_first_event_s"] = DURATION + 1.0
        f, reason, flags = validate_features(d, DURATION)
        assert f is None and "t_first_event_s" in reason

    def test_nullable_timestamp_beyond_duration(self):
        d = valid_data()
        d["t_premise_clear_s"] = DURATION + 1.0
        d["resolution"] = "no_premise"
        f, reason, flags = validate_features(d, DURATION)
        assert f is None and "t_premise_clear_s" in reason

    def test_payoff_beyond_duration(self):
        d = valid_data()
        d["payoffs"] = [{"t_s": DURATION + 1.0, "kind": "reveal", "weight": 3}]
        f, reason, flags = validate_features(d, DURATION)
        assert f is None and "payoff" in reason

    def test_extra_top_level_field(self):
        d = valid_data()
        d["bogus"] = 1
        f, reason, flags = validate_features(d, DURATION)
        assert f is None and "extra field bogus" in reason

    def test_extra_payoff_field(self):
        d = valid_data()
        d["payoffs"][0]["bogus"] = 1
        f, reason, flags = validate_features(d, DURATION)
        assert f is None and "extra field payoffs.bogus" in reason

    def test_bad_enum_invalid(self):
        d = valid_data()
        d["arc"] = "wiggly"
        f, reason, flags = validate_features(d, DURATION)
        assert f is None and reason.startswith("schema:")


class TestFlags:
    def test_two_weight3_payoffs(self):
        d = valid_data()
        d["payoffs"] = [
            {"t_s": 5.0, "kind": "reveal", "weight": 3},
            {"t_s": 15.0, "kind": "kill_or_win", "weight": 3},
        ]
        d["ending"] = "long_tail"
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None and "main_payoff_count:2" in flags

    def test_payoffs_not_time_ordered(self):
        d = valid_data()
        d["payoffs"] = [
            {"t_s": 15.0, "kind": "reveal", "weight": 1},
            {"t_s": 19.0, "kind": "kill_or_win", "weight": 3},
        ]
        f, reason, flags = validate_features(d, DURATION)
        assert "payoffs_not_time_ordered" not in flags  # this one is ordered
        d["payoffs"] = [
            {"t_s": 19.0, "kind": "kill_or_win", "weight": 3},
            {"t_s": 5.0, "kind": "reveal", "weight": 1},
        ]
        d["ending"] = "long_tail"
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None and "payoffs_not_time_ordered" in flags

    def test_arc_without_payoffs(self):
        d = valid_data()
        d["payoffs"] = []
        d["arc"] = "builds"
        d["ending"] = "long_tail"
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None and "arc_without_payoffs:builds" in flags

    def test_premise_null_vs_resolution(self):
        d = valid_data()
        d["t_premise_clear_s"] = None
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None and "premise_null_vs_resolution_mismatch" in flags
        d["resolution"] = "no_premise"
        f, reason, flags = validate_features(d, DURATION)
        assert "premise_null_vs_resolution_mismatch" not in flags

    def test_first_event_after_premise(self):
        d = valid_data()
        d["t_first_event_s"] = 0.6
        d["t_premise_clear_s"] = 0.0
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None and "first_event_after_premise:0.6>0.0" in flags

    def test_first_event_after_premise_within_tolerance(self):
        d = valid_data()
        d["t_first_event_s"] = 0.4
        d["t_premise_clear_s"] = 0.0
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None
        assert not any(fl.startswith("first_event_after_premise") for fl in flags)

    def test_first_event_after_premise_null_premise(self):
        d = valid_data()
        d["t_premise_clear_s"] = None
        d["resolution"] = "no_premise"
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None
        assert not any(fl.startswith("first_event_after_premise") for fl in flags)

    def test_dead_time_past_last_payoff(self):
        d = valid_data()
        d["t_first_event_s"] = 0.0
        d["payoffs"] = [{"t_s": 7.8, "kind": "punchline", "weight": 3}]
        d["dead_time_s"] = 9.0
        d["ending"] = "long_tail"
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None and "dead_time_past_last_payoff:9.0s" in flags

    def test_dead_time_past_last_payoff_within_tolerance(self):
        d = valid_data()
        d["t_first_event_s"] = 0.0
        d["payoffs"] = [{"t_s": 7.8, "kind": "punchline", "weight": 3}]
        d["dead_time_s"] = 8.0  # limit is 7.8 + 0.5 = 8.3
        d["ending"] = "long_tail"
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None
        assert not any(fl.startswith("dead_time_past_last_payoff") for fl in flags)

    def test_dead_time_no_payoffs_no_flag(self):
        d = valid_data()
        d["payoffs"] = []
        d["arc"] = "none"
        d["dead_time_s"] = 18.0
        d["ending"] = "long_tail"
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None
        assert not any(fl.startswith("dead_time_past_last_payoff") for fl in flags)

    def test_ending_vs_tail(self):
        d = valid_data()
        d["payoffs"] = [{"t_s": 15.0, "kind": "kill_or_win", "weight": 3}]
        # ending cut_on_payoff but tail is 5 s
        f, reason, flags = validate_features(d, DURATION)
        assert reason is None
        assert any(fl.startswith("ending_vs_tail:cut_on_payoff") for fl in flags)


class TestSDKSchema:
    def test_t_schema_converts_and_orders(self):
        from google.genai import Client, _transformers as t
        from src.tools.analyze_video.schema import VideoFeatures
        schema = t.t_schema(Client(api_key="x"), VideoFeatures)
        order = schema.property_ordering
        assert order[0] == "timeline" and order[1] == "payoffs"
        assert order[-2:] == ["uncertain_fields", "notes"]
        assert schema.additional_properties is None
