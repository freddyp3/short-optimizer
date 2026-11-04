"""Response schema for ``analyze_video`` feature extraction (plan/03a-feature-checklist.md, prompts v0-v3).

``VideoFeatures`` doubles as the Gemini ``response_schema`` and the validator for raw responses, like the hook
test schema. Field order is generation order (the SDK emits ``propertyOrdering`` in declaration order): the
``timeline`` scratchpad first, then ``payoffs`` (the backbone most derived features come from), then groups A-F,
then ``notes``. Only Gemini-extracted fields live here; code features (group G), channel history (group H),
ordinal coding and derived features are computed elsewhere.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

PROMPT_VERSION = "v2"

# note: weight is an int with bounds, not Literal[1, 2, 3]: Gemini schema enums must be strings.
class Payoff(BaseModel):
    t_s: float = Field(ge=0.0)
    kind: Literal["punchline", "reveal", "kill_or_win", "goal_or_objective", "loot_result",
                  "fail_or_backfire", "spectacle", "other"]
    weight: int = Field(ge=1, le=3)


class VideoFeatures(BaseModel):
    timeline: str
    payoffs: list[Payoff] = Field(default_factory=list, max_length=6)

    # A. Format
    format_macro: Literal["gameplay_clip", "edit", "text_skit", "irl", "loot_opening", "menu_showcase",
                          "slideshow", "other"]
    voice: Literal["none", "creator_live", "group_live", "voiceover", "meme_voice_clip"]
    text_role: Literal["none", "subtitles", "narrative", "label_only", "mixed"]

    # B. Opening
    t_first_event_s: float = Field(ge=0.0)
    opening_frame: Literal["mid_action", "premise_card", "menu_or_lobby", "loading_or_black", "static_image",
                           "irl_setup"]
    t_premise_clear_s: Optional[float] = Field(ge=0.0)
    premise_type: Literal["explicit_question_or_tease", "implicit_situation", "payoff_first", "none"]
    premise_legible_muted: bool
    premise_legible_nonplayer: bool

    # C. Structure
    arc: Literal["single_peak", "builds", "front_loaded", "repeated_same_level", "none"]
    resolution: Literal["resolved", "partial", "unresolved", "no_premise"]
    twist: bool
    surprise_peak: Literal["none", "mild", "strong"]
    dead_time_s: float = Field(ge=0.0)
    beat_before_main_payoff: bool
    info_new_per_5s: Literal["low", "medium", "high"]
    main_payoff_legible_muted: bool

    # D. Ending and loop
    ending: Literal["cut_on_payoff", "short_tail", "long_tail", "outro_or_black", "result_screen_hold",
                    "cliffhanger"]
    loop_seam: Literal["broken", "compatible", "seamless"]
    rewatch_trigger: Literal["none", "recontextualised_opening", "blink_and_miss", "ambiguity", "audio_moment"]

    # E. Audio
    audio_bed: Literal["game_only_or_none", "music", "meme_sound", "live_voices", "voiceover", "mixed"]
    audio_visual_sync: bool
    t_audio_change_s: Optional[float] = Field(ge=0.0)
    audio_carries_payoff: bool

    # F. Register
    primary_appeal: Literal["humor", "skill_hype", "suspense_luck", "relatable_frustration", "curiosity_info",
                            "aesthetic_vibe", "nostalgia", "drama"]
    relatability: Literal["none", "shared_player_experience", "in_group_reference", "creator_personal"]
    game_moment_kind: Literal["elimination", "multi_kill_or_wipe", "clutch_low_hp", "goal_or_objective",
                              "rare_drop", "skin_reveal", "glitch_or_bug", "troll_or_teaming", "none"]
    big_number_on_screen: bool
    low_health_visible: bool
    result_screen_shown: bool

    uncertain_fields: list[str] = Field(default_factory=list)  # debugging only, never a model input
    notes: str = ""


class TimelineEntry(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    seen: str
    heard: str


class VideoFeaturesV3(VideoFeatures):
    """v3 (plan/03b section 2): the timeline is one entry per fixed 1-second window; all other fields as v2."""

    timeline: list[TimelineEntry] = Field(max_length=61)


SCHEMAS = {"v0": VideoFeatures, "v1": VideoFeatures, "v2": VideoFeatures, "v3": VideoFeaturesV3}

TIMESTAMP_FIELDS = ("t_first_event_s", "t_premise_clear_s", "t_audio_change_s")
TOLERANCE_S = 0.5
WINDOW_TOLERANCE_S = 0.05


def windows(duration_s: float) -> list[tuple[float, float]]:
    """Fixed 1-second windows covering [0, duration]; the last one ends at the duration (rounded to 0.1 s).
    A remainder under 0.1 s is folded into the previous window, so no window is empty."""
    n = max(1, math.ceil(duration_s - 0.1))
    return [(float(i), float(i + 1) if i < n - 1 else round(duration_s, 1)) for i in range(n)]


def render_prompt(template: str, duration_s: float) -> str:
    """Fill a prompt template's placeholders (v0-v2 only use ``duration_s``; v3 adds the window list)."""
    w = windows(duration_s)
    return template.format(duration_s=duration_s, n_windows=len(w),
                           windows=", ".join(f"{a:.1f}-{b:.1f}" for a, b in w))


def validate_features(data, duration_s: float, version: str = PROMPT_VERSION):
    """Validate a parsed JSON object against the schema and the video's duration.

    Returns (features, invalid_reason, quality_flags). ``features`` is None when invalid_reason is set.
    Out-of-range timestamps invalidate; internal inconsistencies are quality flags, read in the DEV pass
    (plan/03a section 6.1) to decide which definitions Gemini misreads. For v3, a timeline that does not match
    the given windows exactly (count, order, bounds within 0.05 s) is invalid.
    """
    cls = SCHEMAS[version]
    if isinstance(data, dict):
        extra = set(data) - set(cls.model_fields)
        if extra:
            return None, f"schema: extra field {sorted(extra)[0]}", []
        for p in data.get("payoffs") or []:
            if isinstance(p, dict) and (pextra := set(p) - set(Payoff.model_fields)):
                return None, f"schema: extra field payoffs.{sorted(pextra)[0]}", []
        if cls is VideoFeaturesV3:
            for e in data.get("timeline") or []:
                if isinstance(e, dict) and (textra := set(e) - set(TimelineEntry.model_fields)):
                    return None, f"schema: extra field timeline.{sorted(textra)[0]}", []
    try:
        f = cls.model_validate(data)
    except ValidationError as e:
        return None, f"schema: {e.errors()[0]['loc']} {e.errors()[0]['msg']}", []

    if cls is VideoFeaturesV3:
        w = windows(duration_s)
        if len(f.timeline) != len(w):
            return None, f"timeline: {len(f.timeline)} entries for {len(w)} windows", []
        for i, (e, (a, b)) in enumerate(zip(f.timeline, w)):
            if abs(e.start_s - a) > WINDOW_TOLERANCE_S or abs(e.end_s - b) > WINDOW_TOLERANCE_S:
                return None, f"timeline[{i}]: {e.start_s}-{e.end_s} != window {a:.1f}-{b:.1f}", []

    limit = duration_s + TOLERANCE_S
    for name in TIMESTAMP_FIELDS:
        t = getattr(f, name)
        if t is not None and t > limit:
            return None, f"{name} {t} > duration {duration_s} + {TOLERANCE_S}", []
    if any(p.t_s > limit for p in f.payoffs):
        return None, f"payoff t_s > duration {duration_s} + {TOLERANCE_S}", []
    if f.dead_time_s > limit:
        return None, f"dead_time_s {f.dead_time_s} > duration {duration_s}", []

    flags = []
    times = [p.t_s for p in f.payoffs]
    if times != sorted(times):
        flags.append("payoffs_not_time_ordered")
    n_main = sum(p.weight == 3 for p in f.payoffs)
    if f.payoffs and n_main != 1:
        flags.append(f"main_payoff_count:{n_main}")
    if not f.payoffs and f.arc != "none":
        flags.append(f"arc_without_payoffs:{f.arc}")
    if f.t_premise_clear_s is not None and f.t_premise_clear_s < f.t_first_event_s - TOLERANCE_S:
        flags.append(f"first_event_after_premise:{f.t_first_event_s:.1f}>{f.t_premise_clear_s:.1f}")
    if (f.t_premise_clear_s is None) != (f.resolution == "no_premise"):
        flags.append("premise_null_vs_resolution_mismatch")
    if f.payoffs and f.dead_time_s > max(times) - f.t_first_event_s + TOLERANCE_S:
        flags.append(f"dead_time_past_last_payoff:{f.dead_time_s:.1f}s")
    if f.payoffs:
        tail = duration_s - max(times)
        expected = {"cut_on_payoff": (None, 1.0), "short_tail": (1.0, 3.0), "long_tail": (3.0, None)}
        lo, hi = expected.get(f.ending, (None, None))
        if (lo is not None and tail < lo - TOLERANCE_S) or (hi is not None and tail > hi + TOLERANCE_S):
            flags.append(f"ending_vs_tail:{f.ending}:{tail:.1f}s")
    return f, None, flags
