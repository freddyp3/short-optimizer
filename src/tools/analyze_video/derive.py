"""Derived model features from a Gemini feature-extraction record (plan/03b section 1).

Gemini's raw answers are always kept next to the derived values (``ending_gemini``).
Timings come in raw and cut-snapped variants; the consistency A/B (03b 4.3) picks one
of each pair. All functions are pure; ``derive_features`` never mutates its input.
"""

from __future__ import annotations

KEEP_ENDINGS = {"outro_or_black", "result_screen_hold", "cliffhanger"}
SNAP_KINDS = {"punchline", "reveal", "spectacle"}
SNAP_MAX_S = 0.5
SKIP_FIELDS = {"timeline", "notes", "uncertain_fields", "payoffs"}


def resolve_ending(gemini_value: str, tail_s: float) -> tuple[str, str]:
    """D1 resolution: semantic endings stay Gemini's; tail-based endings come from code."""
    if gemini_value in KEEP_ENDINGS:
        return gemini_value, "gemini"
    if tail_s <= 1.0:
        return "cut_on_payoff", "code"
    if tail_s <= 3.0:
        return "short_tail", "code"
    return "long_tail", "code"


def _main_payoff(payoffs: list) -> dict | None:
    """The weight-3 payoff; if none, the max weight, earliest on tie (03a C1)."""
    if not payoffs:
        return None
    w3 = [p for p in payoffs if p["weight"] == 3]
    if w3:
        return min(w3, key=lambda p: p["t_s"])
    w = max(p["weight"] for p in payoffs)
    return min((p for p in payoffs if p["weight"] == w), key=lambda p: p["t_s"])


def _tails(payoffs: list, duration_s: float) -> tuple[float, float]:
    """(tail_all, tail_clear); both fall back to duration when there is no payoff."""
    if not payoffs:
        return duration_s, duration_s
    last = max(p["t_s"] for p in payoffs)
    clear = [p["t_s"] for p in payoffs if p["weight"] >= 2]
    return duration_s - last, duration_s - (max(clear) if clear else last)


def _snap(payoffs: list, cuts: list[float]) -> list:
    """t_s snapped to the nearest cut within SNAP_MAX_S, for SNAP_KINDS only (ties: earlier)."""
    out = []
    for p in payoffs:
        p = dict(p)
        if p["kind"] in SNAP_KINDS and cuts:
            c = min(cuts, key=lambda t: (abs(t - p["t_s"]), t))
            if abs(c - p["t_s"]) <= SNAP_MAX_S:
                p["t_s"] = c
        out.append(p)
    return out


def derive_features(result: dict, duration_s: float, cuts: list[float]) -> dict:
    """Flat dict: derived payoff/ending features + every other scalar Gemini field."""
    out = {k: v for k, v in result.items() if k not in SKIP_FIELDS}
    payoffs = result.get("payoffs") or []

    out["payoff_count"] = sum(1 for p in payoffs if p["weight"] >= 2)
    out["payoff_count_all"] = len(payoffs)
    out["no_payoff"] = not payoffs

    main = _main_payoff(payoffs)
    out["t_main_payoff_s"] = main["t_s"] if main else None
    out["main_payoff_frac"] = main["t_s"] / duration_s if main else 1.0
    out["main_payoff_kind"] = main["kind"] if main else None

    tail_all, tail_clear = _tails(payoffs, duration_s)
    out["tail_all"] = tail_all
    out["tail_clear"] = tail_clear
    out["ending_from_tail_all"] = resolve_ending(result.get("ending"), tail_all)[0]
    out["ending_from_tail_clear"] = resolve_ending(result.get("ending"), tail_clear)[0]
    out["ending_gemini"] = result.get("ending")

    snapped = _snap(payoffs, cuts or [])
    smain = _main_payoff(snapped)
    out["t_main_payoff_s_snap"] = smain["t_s"] if smain else None
    out["main_payoff_frac_snap"] = smain["t_s"] / duration_s if smain else 1.0
    st_all, st_clear = _tails(snapped, duration_s)
    out["tail_all_snap"] = st_all
    out["tail_clear_snap"] = st_clear
    return out
