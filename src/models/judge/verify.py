"""Exit checks for the judging loop (plan/03c section 20).

    ./venv/bin/python -m src.models.judge.verify extraction      # Stage A
    ./venv/bin/python -m src.models.judge.verify reasons TAG     # Stage D item 5
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.models import targets as T
from src.models.judge.library import RAW_DIR, V3_RUNS
from src.models.judge.prompt import LETTERS
from src.models.judge.run import RAW
from src.tools.analyze_video.ab_score import ARC_ORD, NO_CHANGE
from src.tools.analyze_video.derive import derive_features
from src.tools.analyze_video.schema import windows

NEW_RUNS = V3_RUNS[1:]
TOL_S = 1.0
MAX_PAYOFF_DISAGREE = 0.20


def v3_records(df: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for v, b in zip(df["video_id"], df["blind_id"]):
        for run in V3_RUNS:
            p = RAW_DIR / run / f"{b}__r1.json"
            if p.exists():
                out[v] = json.loads(p.read_text())
                break
    return out


def extraction() -> int:
    df, *_ = T.load()
    recs = v3_records(df)
    ok = {v: r for v, r in recs.items() if r["status"] == "ok"}
    fails = []
    print(f"records found {len(recs)} / {len(df)}, valid {len(ok)}")
    if len(ok) != len(df):
        fails.append("not every video has a valid one-second timeline")
    bad_windows = [v for v, r in ok.items() if len(r["result"]["timeline"]) != len(windows(r["duration_s_file"]))]
    print(f"window count wrong: {len(bad_windows)}")
    if bad_windows:
        fails.append("window count mismatch")
    sel = pd.read_csv(C.SELECTION_CSV_PATH)
    locked = set(sel.loc[sel["split"].isin(["gate", "heldout"]), "blind_id"])
    present = {p.name.split("__")[0] for run in NEW_RUNS for p in (RAW_DIR / run).glob("*__r*.json")}
    print(f"GATE/HELDOUT files in the new runs: {len(locked & present)}")
    if locked & present:
        fails.append("GATE/HELDOUT videos were extracted")

    rows = []
    for row in df.itertuples():
        r = ok.get(row.video_id)
        if r is None:
            continue
        d = derive_features(r["result"], r["duration_s_file"], [])
        prem = d["t_premise_clear_s"] if d["t_premise_clear_s"] is not None else r["duration_s_file"]
        main3, main2 = d["t_main_payoff_s"], row.t_main_payoff_s
        both_none = main3 is None and main2 != main2
        rows.append({
            "premise": abs(prem - row.t_premise_clear_s) <= TOL_S,
            "main_payoff": both_none or (main3 is not None and main2 == main2 and abs(main3 - main2) <= TOL_S),
            "tail": abs(d["tail_all"] - row.tail_all) <= TOL_S,
            "arc": ARC_ORD[d["arc"]] == row.arc_ord,
            "payoff_count": d["payoff_count_all"] == row.payoff_count_all})
    agree = pd.DataFrame(rows).mean()
    print("agreement of the new run with the frozen v2 features (share of videos):")
    print(agree.round(2).to_string())
    if 1 - agree["main_payoff"] > MAX_PAYOFF_DISAGREE:
        fails.append(f"main payoff time disagrees on {1 - agree['main_payoff']:.0%} of videos (limit {MAX_PAYOFF_DISAGREE:.0%})")

    first, last = [], []
    for r in ok.values():
        tl = r["result"]["timeline"]
        n3 = max(1, len(tl) // 3)
        share = lambda es: np.mean([bool(NO_CHANGE.match(e[f])) for e in es for f in ("seen", "heard")])
        first.append(share(tl[:n3]))
        last.append(share(tl[-n3:]))
    print(f"'no change' share: first third {np.mean(first):.1%}, last third {np.mean(last):.1%}")
    if np.mean(last) - np.mean(first) >= 0.20:
        fails.append("timeline gets lazy towards the end")
    cost = sum(json.loads(p.read_text()).get("cost_usd_paid_equiv", 0) for run in NEW_RUNS
               for p in (RAW_DIR / run).glob("*__r*.json"))
    print(f"cost of the new runs: ${cost:.2f}")
    print("EXIT CHECK:", "PASS" if not fails else "FAIL: " + "; ".join(fails))
    return 1 if fails else 0


NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*\d)")


def reasons(tag: str) -> int:
    """Every lesson must cite an example that was shown, and numbers quoted in a lesson must occur in that
    example's block of the prompt."""
    n = cited = cited_ok = nums = nums_ok = 0
    for p in sorted((RAW / tag).glob("*.json")):
        r = json.loads(p.read_text())
        if r["status"] != "ok":
            continue
        n += 1
        blocks = dict(zip(LETTERS, re.split(r"### Example [A-E] ", r["prompt"])[1:]))
        for les in r["result"]["lessons"]:
            cited += 1
            block = blocks.get(les["example"])
            cited_ok += block is not None
            if block is None:
                continue
            block = block.split("\n## ")[0]
            for x in NUM.findall(les["observation"]):
                nums += 1
                nums_ok += bool(re.search(rf"(?<![\d.]){re.escape(x)}(?:\.0)?(?![\d])", block))
    print(f"{tag}: {n} outputs, {cited} lessons; cited example exists {cited_ok}/{cited}; "
          f"numbers quoted that occur in the cited example {nums_ok}/{nums}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extraction")
    sub.add_parser("reasons").add_argument("tag")
    a = p.parse_args(argv)
    return extraction() if a.cmd == "extraction" else reasons(a.tag)


if __name__ == "__main__":
    raise SystemExit(main())
