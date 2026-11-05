"""Prevalence rule (plan/03a section 6.2b), outcome-blind, on the training pool's extracted features.

Drop any boolean or one-hot level present in fewer than MIN_ROWS or more than n - MIN_ROWS rows; rare levels of a
nominal field are merged into "other" (if "other" is then still rare, it is merged into nothing and the field keeps
only its common levels). A nominal field left with fewer than two levels is dropped. Reads only the feature table.

    ./venv/bin/python -m src.tools.analyze_video.prevalence
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FEATURES = Path("data/features/gemini_features_train_v2.csv")
CODE = Path("data/features/code_features.csv")
OUT = Path("data/features/prevalence.json")
MIN_ROWS = 10

# full-model nominal fields after the consistency gate (plan/03b section 6); `ending` is the tail-resolved one
NOMINAL = ["format_macro", "voice", "text_role", "opening_frame", "premise_type", "resolution", "ending_from_tail_all",
           "audio_bed", "primary_appeal", "relatability"]
BOOLS = ["premise_legible_muted", "main_payoff_legible_muted", "audio_carries_payoff", "big_number_on_screen",
         "low_health_visible", "result_screen_shown", "no_payoff"]
# derived booleans named in 03a (A2, B2, B3/section 4, C3, D1, F2) and the code one (G9)
DERIVED = {
    "group_live": lambda d: d["voice"] == "group_live",
    "opening_dead": lambda d: d["opening_frame"].isin(["menu_or_lobby", "loading_or_black"]),
    "no_premise": lambda d: d["t_premise_clear_s"].isna(),
    "unresolved": lambda d: d["resolution"] == "unresolved",
    "dead_ending": lambda d: d["ending_from_tail_all"].isin(["long_tail", "outro_or_black"]),
    "relatable_any": lambda d: d["relatability"] != "none",
}


def rare(count: int, n: int) -> bool:
    return count < MIN_ROWS or count > n - MIN_ROWS


def main() -> int:
    d = pd.read_csv(FEATURES).query("status == 'ok'")
    code = pd.read_csv(CODE, usecols=["video_id", "open_first_frame_black"])
    d = d.merge(code, on="video_id", how="left")
    n = len(d)
    out = {"n": n, "min_rows": MIN_ROWS, "nominal": {}, "bools": {}}
    for f in NOMINAL:
        counts = d[f].astype(str).value_counts().to_dict()
        keep = sorted(lv for lv, c in counts.items() if not rare(c, n))
        merged = sorted(lv for lv in counts if lv not in keep)
        other = sum(counts[lv] for lv in merged)
        if merged and not rare(other, n):
            keep.append("other")
        out["nominal"][f] = {"counts": counts, "levels": keep, "merged_into_other": merged, "other_count": other,
                             "decision": "keep" if len(keep) >= 2 else "drop (fewer than two common levels)"}
    bools = {f: d[f].astype(bool) for f in BOOLS}
    bools |= {f: fn(d) for f, fn in DERIVED.items()}
    bools["open_first_frame_black"] = d["open_first_frame_black"].astype(bool)
    for f, s in bools.items():
        c = int(s.sum())
        out["bools"][f] = {"true": c, "decision": "drop" if rare(c, n) else "keep"}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"n = {n}")
    for f, r in out["nominal"].items():
        print(f"{f:22} {r['decision']:6} levels={r['levels']} merged={r['merged_into_other']}")
    for f, r in out["bools"].items():
        print(f"{f:26} true={r['true']:>3}  {r['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
