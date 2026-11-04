"""Tests for ab_score.py (the pre-registered v2-vs-v3 A/B scorer, plan/03b section 4)."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.analyze_video import ab_score as A
from src.tools.analyze_video.schema import windows

REAL_RUN = Path("data/features/gemini/raw/20261104T023841Z_4d7bfb6")


# ----------------------------------------------------------------- reference values

def test_icc21_shrout_fleiss():
    m = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                  [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])
    assert A.icc21(m[:, 0], m[:, 1], m[:, 2], m[:, 3]) == pytest.approx(0.29, abs=0.005)


def test_icc21_identical():
    assert A.icc21([1, 2, 3], [1, 2, 3]) == 1.0


def test_kappa_2x2():
    a = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert A.kappa(a, b) == pytest.approx(0.40, abs=0.005)


def test_kappa_both_constant_equal():
    assert A.kappa(["x"] * 5, ["x"] * 5) == 1.0


def test_weighted_kappa():
    a = ["broken", "compatible", "seamless", "broken"]
    b = ["broken", "seamless", "seamless", "compatible"]
    assert A.kappa(a, b, levels=["broken", "compatible", "seamless"],
                   weighted=True) == pytest.approx(0.5)


def test_ts_stats():
    med, share = A.ts_stats([0, 1, 5], [0.5, 3, 5])
    assert med == 0.5 and share == pytest.approx(2 / 3)


# ----------------------------------------------------------------- end-to-end fixtures

@pytest.fixture
def template_result():
    rec = json.loads(sorted(REAL_RUN.glob("*__r1.json"))[0].read_text())
    return rec["result"]


def fake_sel(tmp_path):
    """30 fake videos: 10 per band at durations 12 / 20 / 40 s."""
    rows = []
    for band, dur in (("short", 12.0), ("middle", 20.0), ("long", 40.0)):
        for i in range(10):
            rows.append({"video_id": f"{band}{i}", "blind_id": f"v{band[:1]}{i:07d}",
                         "band": band, "duration_s_export": dur,
                         "duration_s_file": dur, "blind_path": f"x/{band}{i}.mp4"})
    sel = pd.DataFrame(rows)
    sel.to_csv(tmp_path / "sel.csv", index=False)
    return sel


def fake_cuts(tmp_path, sel):
    d = tmp_path / "cuts"
    d.mkdir()
    for b in sel["blind_id"]:
        (d / f"{b}.json").write_text(json.dumps(
            {"blind_id": b, "threshold": 0.3, "duration_s": 10.0, "cuts": []}))
    return d


def v2_result(template, dur):
    r = copy.deepcopy(template)
    for p in r["payoffs"]:
        p["t_s"] = min(p["t_s"], dur - 0.5)
    return r


def v3_result(template, dur, lazy_last_third=False):
    r = v2_result(template, dur)
    w = windows(dur)
    n3 = len(w) // 3
    r["timeline"] = [
        {"start_s": a, "end_s": b,
         "seen": "no change" if lazy_last_third and i >= len(w) - n3 else "x",
         "heard": "no change" if lazy_last_third and i >= len(w) - n3 else "y"}
        for i, (a, b) in enumerate(w)]
    return r


def write_run(run_dir, sel, results):
    """results: {blind_id: (r1_result, r2_result)}; a result of None writes an invalid record."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for b, (r1, r2) in results.items():
        for k, res in ((1, r1), (2, r2)):
            rec = ({"status": "ok", "result": res} if res is not None
                   else {"status": "invalid", "invalid_reason": "fake", "result": None})
            (run_dir / f"{b}__r{k}.json").write_text(json.dumps(rec))


def noisy_result(res):
    """A perturbed copy: main payoff downgraded, a new w3 payoff elsewhere, more dead time."""
    r = copy.deepcopy(res)
    if r["payoffs"]:
        r["payoffs"][0]["weight"] = 1
        r["payoffs"].append({"t_s": max(1.0, r["payoffs"][0]["t_s"] - 3.0),
                             "kind": "reveal", "weight": 3})
    r["dead_time_s"] = r.get("dead_time_s", 0.0) + 6.0
    return r


def build_runs(tmp_path, sel, template, v3_lazy=False, v3_dead_jitter=0.0):
    """v3 identical r1/r2 (optionally lazy/dead-jittered); v2 noisy between runs."""
    durs = dict(zip(sel["blind_id"], sel["duration_s_file"]))
    v2, v3 = {}, {}
    for b in sel["blind_id"]:
        base = v2_result(template, durs[b])
        v2[b] = (base, noisy_result(base))
        r3 = v3_result(template, durs[b], lazy_last_third=v3_lazy)
        r3b = copy.deepcopy(r3)
        if v3_dead_jitter:
            r3b["dead_time_s"] += v3_dead_jitter
        v3[b] = (r3, r3b)
    d2, d3 = tmp_path / "runv2", tmp_path / "runv3"
    write_run(d2, sel, v2)
    write_run(d3, sel, v3)
    return d2, d3


@pytest.fixture
def env(tmp_path, monkeypatch, template_result):
    sel = fake_sel(tmp_path)
    cuts_dir = fake_cuts(tmp_path, sel)
    monkeypatch.setattr(A, "SELECTION", tmp_path / "sel.csv")
    monkeypatch.setattr(A, "CUTS_DIR", cuts_dir)
    monkeypatch.setattr(A, "OUT_DIR", tmp_path / "ab_report")
    return {"sel": sel, "template": template_result, "tmp": tmp_path}


def decide_env(env, read_ok, **kw):
    d2, d3 = build_runs(env["tmp"], env["sel"], env["template"], **kw)
    return A.decide(A.load_run(d2, env["sel"]), A.load_run(d3, env["sel"]),
                    env["sel"], read_ok)


# ----------------------------------------------------------------- end-to-end cases

def test_v3_wins_when_identical_and_v2_noisy(env):
    r = decide_env(env, "yes")
    assert r["winner"] == "v3"
    assert "pass" in r["reason"] or "mean" in r["reason"]


def test_pending_read_blocks_decision(env):
    r = decide_env(env, None)
    assert r["winner"] is None and "pending" in r["reason"]


def test_20pct_invalid_fails_guard(env):
    d2, d3 = build_runs(env["tmp"], env["sel"], env["template"])
    # poison 20 % of the 60 v3 records (6 videos, both runs)
    rec = {"status": "invalid", "invalid_reason": "fake", "result": None}
    for i, b in enumerate(env["sel"]["blind_id"]):
        if i % 5 == 0:
            for k in (1, 2):
                (d3 / f"{b}__r{k}.json").write_text(json.dumps(rec))
    r = A.decide(A.load_run(d2, env["sel"]), A.load_run(d3, env["sel"]),
                 env["sel"], "yes")
    assert r["winner"] == "v2" and "guard" in r["reason"]
    assert r["guard"]["v3_invalid_rate"] == pytest.approx(0.20)


def test_lazy_long_band_triggers_fallback(env):
    r = decide_env(env, "yes", v3_lazy=True)
    assert r["laziness"]["lazy"] is True
    assert r["winner"] is None and "4.4" in r["reason"]


def test_identical_versions_tie_to_v2(env, tmp_path):
    sel, tpl = env["sel"], env["template"]
    durs = dict(zip(sel["blind_id"], sel["duration_s_file"]))
    # identical results in both dirs; v3-shaped timelines so laziness can read them
    same = {b: (v3_result(tpl, durs[b]), v3_result(tpl, durs[b])) for b in sel["blind_id"]}
    d2, d3 = tmp_path / "rv2", tmp_path / "rv3"
    write_run(d2, sel, same)
    write_run(d3, sel, same)
    r = A.decide(A.load_run(d2, sel), A.load_run(d3, sel), sel, "yes")
    assert r["winner"] == "v2" and "incumbent" in r["reason"]


def test_gate_written_and_dead_time_substitute(env):
    d2, d3 = build_runs(env["tmp"], env["sel"], env["template"], v3_dead_jitter=7.0)
    rc = A.main(["--v2", str(d2), "--v3", str(d3), "--v3-read-ok", "yes"])
    assert rc == 0
    out = env["tmp"] / "ab_report"
    stem = f"{d2.name}__{d3.name}"
    r = json.loads((out / f"{stem}.json").read_text())
    assert (out / f"{stem}.md").exists()
    assert r["winner"] == "v3"
    assert "gate_core" in r and "gate_full" in r
    assert r["gate_core"]["dead_time_s"]["pass"] is False
    assert "static_frac" in r["gate_core"]["dead_time_s"]["decision"]


def test_read_list(env):
    picks = A.read_list(env["sel"])
    assert len(picks) == 5 and len(set(picks)) == 5
    # median-duration member of each band is present
    for band in ("short", "middle", "long"):
        g = env["sel"][env["sel"]["band"] == band].sort_values(
            ["duration_s_file", "blind_id"]).reset_index(drop=True)
        assert g.at[len(g) // 2, "blind_id"] in picks
