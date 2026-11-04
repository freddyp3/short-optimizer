"""Score the v2-vs-v3 consistency A/B (plan/03b section 4). Pre-registered: committed before any A/B call.

Reads only the raw extraction records, the selection csv and the cut files; never loads an outcome column.

    ./venv/bin/python -m src.tools.analyze_video.ab_score --v2 data/features/gemini/raw/<run_id_v2> \
        --v3 data/features/gemini/raw/<run_id_v3> [--v3-read-ok yes|no]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.analyze_video.derive import derive_features

SELECTION = Path("data/features/gemini/consistency_selection.csv")
CUTS_DIR = Path("data/features/cuts")
OUT_DIR = Path("data/features/gemini/ab_report")
SEED, N_BOOT = 340, 2000
ICC_MIN, KAPPA_MIN, TS_MAX_MEDIAN_S, TS_WINDOW_S = 0.7, 0.6, 1.0, 1.0
TIE_MARGIN = 0.05

ORDINAL_LEVELS = {
    "arc_ord": [0, 1, 2],
    "loop_seam": ["broken", "compatible", "seamless"],
    "surprise_peak": ["none", "mild", "strong"],
    "info_new_per_5s": ["low", "medium", "high"],
}
ARC_ORD = {"none": 0, "repeated_same_level": 0, "front_loaded": 0, "single_peak": 1, "builds": 2}
NOMINAL = ["format_macro", "voice", "text_role", "opening_frame", "premise_type", "resolution", "ending_gemini",
           "ending_from_tail_all", "ending_from_tail_clear", "rewatch_trigger", "audio_bed", "primary_appeal",
           "relatability", "game_moment_kind", "main_payoff_kind", "arc"]
BOOLS = ["premise_legible_muted", "premise_legible_nonplayer", "twist", "beat_before_main_payoff",
         "main_payoff_legible_muted", "audio_visual_sync", "audio_carries_payoff", "big_number_on_screen",
         "low_health_visible", "result_screen_shown", "no_payoff"]
TIMESTAMPS = ["t_first_event_s", "t_premise_clear_s", "t_main_payoff_s", "t_main_payoff_s_snap"]
CONTINUOUS = ["payoff_count", "payoff_count_all", "dead_time_s", "tail_all", "tail_clear", "tail_all_snap",
              "tail_clear_snap", "main_payoff_frac", "main_payoff_frac_snap"]
KIND = {**{f: "kappa" for f in NOMINAL + BOOLS}, **{f: "wkappa" for f in ORDINAL_LEVELS},
        **{f: "ts" for f in TIMESTAMPS}, **{f: "icc" for f in CONTINUOUS}}
# core features (03b 4.2); "tail" and the main-payoff timestamp resolve to a variant (raw + tail_all for step 2)
CORE = ["t_premise_clear_s", "payoff_count", "t_main_payoff_s", "tail", "dead_time_s", "arc_ord", "loop_seam"]
SUBSTITUTES = {"dead_time_s": "static_frac (G4)", "loop_seam": "first_last_dhash_dist (G6)"}
NO_CHANGE = re.compile(r"^\s*no change\.?\s*$", re.I)


# ----------------------------------------------------------------- statistics

def icc21(*cols) -> float:
    """ICC(2,1), two-way random effects, absolute agreement, single rater (Shrout & Fleiss 1979); one column per run."""
    d = np.column_stack(cols).astype(float)
    n, k = d.shape
    if n < 2:
        return float("nan")
    g = d.mean()
    ssr = k * ((d.mean(axis=1) - g) ** 2).sum()
    ssc = n * ((d.mean(axis=0) - g) ** 2).sum()
    sst = ((d - g) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    if den <= 0:
        return 1.0 if sst == 0 else float("nan")
    return float((msr - mse) / den)


def kappa(a, b, levels=None, weighted: bool = False) -> float:
    """Cohen's kappa; linear weights when ``weighted`` (levels give the order). Both runs constant and equal -> 1."""
    a, b = list(a), list(b)
    levels = list(levels) if levels is not None else sorted({str(v) for v in a + b})
    idx = {str(lv): i for i, lv in enumerate(levels)}
    L = len(levels)
    m = np.zeros((L, L))
    for u, v in zip(a, b):
        m[idx[str(u)], idx[str(v)]] += 1
    m /= m.sum()
    if weighted and L > 1:
        i, j = np.indices((L, L))
        w = 1 - np.abs(i - j) / (L - 1)
    else:
        w = np.eye(L)
    po = (w * m).sum()
    pe = (w * np.outer(m.sum(axis=1), m.sum(axis=0))).sum()
    if np.isclose(1 - pe, 0):
        return 1.0 if np.isclose(po, 1) else float("nan")
    return float((po - pe) / (1 - pe))


def ts_stats(x, y) -> tuple[float, float]:
    """(median |dt|, share of |dt| <= TS_WINDOW_S)."""
    d = np.abs(np.asarray(x, float) - np.asarray(y, float))
    return float(np.median(d)), float((d <= TS_WINDOW_S).mean())


def stat(kind: str, x, y, levels=None) -> dict:
    if kind == "icc":
        return {"icc": icc21(x, y)}
    if kind == "kappa":
        return {"kappa": kappa(x, y)}
    if kind == "wkappa":
        return {"kappa": kappa(x, y, levels, weighted=True)}
    med, share = ts_stats(x, y)
    return {"median_abs_dt": med, "share_within_1s": share}


def headline(kind: str, s: dict) -> float:
    """The value compared with the threshold and used in the tie-break mean (timestamps: share within 1 s)."""
    return s["icc"] if kind == "icc" else s["kappa"] if kind in ("kappa", "wkappa") else s["share_within_1s"]


def passes(kind: str, s: dict) -> bool:
    if kind == "icc":
        return s["icc"] >= ICC_MIN
    if kind in ("kappa", "wkappa"):
        return s["kappa"] >= KAPPA_MIN
    return s["median_abs_dt"] <= TS_MAX_MEDIAN_S


def bootstrap_ci(kind: str, x, y, levels=None) -> list[float]:
    x, y = np.asarray(x, dtype=object), np.asarray(y, dtype=object)
    rng = np.random.default_rng(SEED)
    vals = []
    for idx in rng.integers(0, len(x), (N_BOOT, len(x))):
        vals.append(headline(kind, stat(kind, x[idx], y[idx], levels)))
    vals = np.asarray(vals, float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


# ----------------------------------------------------------------- data

def load_run(run_dir: Path, sel: pd.DataFrame) -> dict:
    """{(blind_id, k): record or None}; None when missing or invalid."""
    out = {}
    for b in sel["blind_id"]:
        for k in (1, 2):
            p = run_dir / f"{b}__r{k}.json"
            rec = json.loads(p.read_text()) if p.exists() else None
            out[(b, k)] = rec if rec and rec.get("status") == "ok" else None
    return out


def features(rec: dict, dur: float, cuts: list[float]) -> dict:
    f = derive_features(rec["result"], dur, cuts)
    f["arc_ord"] = ARC_ORD[f["arc"]]
    for t in ("t_premise_clear_s", "t_main_payoff_s", "t_main_payoff_s_snap"):
        if f.get(t) is None:
            f[t] = dur                       # 03a section 4: null -> duration
    return f


def feature_table(runs: dict, sel: pd.DataFrame, ids: list[str]) -> dict:
    """{feature: (run1 values, run2 values)} over ``ids``."""
    meta = sel.set_index("blind_id")
    rows = {1: [], 2: []}
    for b in ids:
        dur = float(meta.at[b, "duration_s_file"])
        cuts = json.loads((CUTS_DIR / f"{b}.json").read_text())["cuts"]
        for k in (1, 2):
            rows[k].append(features(runs[(b, k)], dur, cuts))
    return {f: ([r[f] for r in rows[1]], [r[f] for r in rows[2]]) for f in KIND}


def score_features(table: dict, names: list[str]) -> dict:
    out = {}
    for f in names:
        kind, lv = KIND[f], ORDINAL_LEVELS.get(f)
        x, y = table[f]
        s = stat(kind, x, y, lv)
        out[f] = {"kind": kind, **s, "pass": passes(kind, s), "ci95": bootstrap_ci(kind, x, y, lv)}
    return out


def core_names(snap: bool, tail: str) -> dict:
    suffix = "_snap" if snap else ""
    return {"t_premise_clear_s": "t_premise_clear_s", "payoff_count": "payoff_count",
            "t_main_payoff_s": "t_main_payoff_s" + suffix, "tail": tail + suffix, "dead_time_s": "dead_time_s",
            "arc_ord": "arc_ord", "loop_seam": "loop_seam"}


# ----------------------------------------------------------------- v3 checks

def read_list(sel: pd.DataFrame) -> list[str]:
    """5 videos for the manual timeline read: the median-duration video per band, then the 2 longest others."""
    picks = []
    for band in ("short", "middle", "long"):
        g = sel[sel["band"] == band].sort_values(["duration_s_file", "blind_id"]).reset_index(drop=True)
        picks.append(g.at[len(g) // 2, "blind_id"])
    for b in sel.sort_values(["duration_s_file", "blind_id"], ascending=[False, True])["blind_id"]:
        if len(picks) == 5:
            break
        if b not in picks:
            picks.append(b)
    return picks


def laziness(v2: dict, v3: dict, sel: pd.DataFrame) -> dict:
    long_ = sel[sel["band"] == "long"]
    first, last, n_payoffs, missed = [], [], 0, 0
    for _, r in long_.iterrows():
        b, dur = r["blind_id"], float(r["duration_s_file"])
        for k in (1, 2):
            rec = v3[(b, k)]
            if rec is None:
                continue
            tl = rec["result"]["timeline"]
            n3 = len(tl) // 3
            share = lambda es: np.mean([bool(NO_CHANGE.match(e[f])) for e in es for f in ("seen", "heard")])
            first.append(share(tl[:n3]))
            last.append(share(tl[len(tl) - n3:]))
        v3_times = [p["t_s"] for k in (1, 2) if v3[(b, k)] for p in v3[(b, k)]["result"]["payoffs"]]
        for k in (1, 2):
            if v2[(b, k)] is None:
                continue
            for p in v2[(b, k)]["result"]["payoffs"]:
                if p["t_s"] >= 2 / 3 * dur:
                    n_payoffs += 1
                    missed += not any(abs(p["t_s"] - t) <= 1.0 for t in v3_times)
    gap = float(np.mean(last) - np.mean(first)) if first else float("nan")
    miss = missed / n_payoffs if n_payoffs else 0.0
    return {"no_change_first_third": float(np.mean(first)) if first else None,
            "no_change_last_third": float(np.mean(last)) if last else None, "no_change_gap": gap,
            "v2_late_payoffs": n_payoffs, "missed_by_v3": missed, "miss_share": miss,
            "lazy": bool((gap >= 0.20) or (miss > 0.20))}


def validity_guard(v2: dict, v3: dict, read_ok: str | None) -> dict:
    inv = sum(r is None for r in v3.values()) / len(v3)
    clear = lambda runs: np.mean([sum(p["weight"] >= 2 for p in r["result"]["payoffs"])
                                  for r in runs.values() if r is not None])
    m2, m3 = clear(v2), clear(v3)
    ratio_ok = (m3 == 0) if m2 == 0 else abs(m3 / m2 - 1) <= 0.25
    checks = {"v3_invalid_rate": inv, "invalid_ok": inv <= 0.10, "v2_mean_clear_payoffs": m2,
              "v3_mean_clear_payoffs": m3, "payoff_ratio_ok": bool(ratio_ok),
              "manual_read": read_ok or "pending"}
    checks["pass"] = None if read_ok is None and checks["invalid_ok"] and ratio_ok else \
        bool(checks["invalid_ok"] and ratio_ok and read_ok == "yes")
    return checks


# ----------------------------------------------------------------- decision

def decide(v2: dict, v3: dict, sel: pd.DataFrame, read_ok: str | None) -> dict:
    report = {"guard": validity_guard(v2, v3, read_ok), "laziness": laziness(v2, v3, sel),
              "read_list": read_list(sel)}
    ids = [b for b in sel["blind_id"] if all(r[(b, k)] is not None for r in (v2, v3) for k in (1, 2))]
    report["n_videos_compared"] = len(ids)
    tables = {"v2": feature_table(v2, sel, ids), "v3": feature_table(v3, sel, ids)}
    base = core_names(snap=False, tail="tail_all")
    cores = {v: score_features(tables[v], list(base.values())) for v in tables}
    report["core_step2"] = cores
    passed = {v: sum(s["pass"] for s in cores[v].values()) for v in cores}
    mean = {v: float(np.nanmean([headline(s["kind"], s) for s in cores[v].values()])) for v in cores}
    report["core_pass_count"], report["core_mean"] = passed, mean

    if report["guard"]["pass"] is False:
        winner, why = "v2", "v3 failed the validity guard"
    elif report["laziness"]["lazy"]:
        winner, why = None, "v3 is lazy: run the 4.4 fallback on the long band before deciding"
    elif report["guard"]["pass"] is None:
        winner, why = None, "pending the manual timeline read (--v3-read-ok yes|no)"
    elif passed["v3"] != passed["v2"]:
        winner = "v3" if passed["v3"] > passed["v2"] else "v2"
        why = f"more core features pass ({passed['v3']} vs {passed['v2']})"
    elif mean["v3"] - mean["v2"] >= TIE_MARGIN:
        winner, why = "v3", f"tie on passes; mean leads by {mean['v3'] - mean['v2']:.3f}"
    else:
        winner, why = "v2", "tie on passes and no 0.05 lead on the mean (incumbent)"
    report["winner"], report["reason"] = winner, why
    if winner is None:
        return report

    t = tables[winner]
    ts_raw, ts_snap = (ts_stats(*t[f])[1] for f in ("t_main_payoff_s", "t_main_payoff_s_snap"))
    snap = ts_snap > ts_raw
    sfx = "_snap" if snap else ""
    icc_all, icc_clear = icc21(*t["tail_all" + sfx]), icc21(*t["tail_clear" + sfx])
    tail = "tail_clear" if icc_clear > icc_all else "tail_all"
    report["variants"] = {"snap": snap, "main_share_raw": ts_raw, "main_share_snap": ts_snap, "tail": tail,
                          "tail_icc_all": icc_all, "tail_icc_clear": icc_clear}

    names = core_names(snap, tail)
    core = score_features(t, list(names.values()))
    gate = {}
    for c, f in names.items():
        s = core[f]
        gate[c] = {"feature": f, **s, "decision": "keep" if s["pass"]
                   else f"substitute: {SUBSTITUTES[c]}" if c in SUBSTITUTES else "drop"}
    full = [f for f in KIND if f not in set(names.values())]
    report["gate_core"] = gate
    report["gate_full"] = {f: {**s, "decision": "keep" if s["pass"] else "drop"}
                           for f, s in score_features(t, full).items()}
    return report


def to_markdown(r: dict) -> str:
    lines = [f"# A/B report: winner {r['winner']} ({r['reason']})", "",
             f"Videos compared (valid in all four runs): {r['n_videos_compared']}", "",
             f"Guard: {json.dumps(r['guard'])}", f"Laziness: {json.dumps(r['laziness'])}",
             f"Manual read list: {', '.join(r['read_list'])}", "",
             "| core feature | v2 | v2 pass | v3 | v3 pass |", "|---|---|---|---|---|"]
    for f in r["core_step2"]["v2"]:
        a, b = r["core_step2"]["v2"][f], r["core_step2"]["v3"][f]
        lines.append(f"| {f} | {headline(a['kind'], a):.3f} {a['ci95']} | {a['pass']} | "
                     f"{headline(b['kind'], b):.3f} {b['ci95']} | {b['pass']} |")
    lines.append(f"\nPasses: {r['core_pass_count']}; means: {r['core_mean']}")
    if "gate_core" in r:
        lines += ["", f"Variants: {json.dumps(r['variants'])}", "", "| gate (core) | value | decision |",
                  "|---|---|---|"]
        lines += [f"| {c} ({g['feature']}) | {headline(g['kind'], g):.3f} | {g['decision']} |"
                  for c, g in r["gate_core"].items()]
        lines += ["", "| gate (full) | value | decision |", "|---|---|---|"]
        lines += [f"| {f} | {headline(g['kind'], g):.3f} | {g['decision']} |" for f, g in r["gate_full"].items()]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--v2", type=Path, required=True)
    p.add_argument("--v3", type=Path, required=True)
    p.add_argument("--v3-read-ok", choices=["yes", "no"], default=None)
    args = p.parse_args(argv)
    sel = pd.read_csv(SELECTION)
    r = decide(load_run(args.v2, sel), load_run(args.v3, sel), sel, args.v3_read_ok)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.v2.name}__{args.v3.name}"
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(r, indent=2, default=str))
    md = to_markdown(r)
    (OUT_DIR / f"{stem}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
