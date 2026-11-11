"""The iteration loop on the train pool (plan/03c section 10). Train (and, with --pool train_dev, DEV) rows only; GATE/HELDOUT are never loaded.

    ./venv/bin/python -m src.models.experiments run                  # iteration 0: registered rows + baselines
    ./venv/bin/python -m src.models.experiments run NAME [NAME ...]  # candidates from CANDIDATES
    ./venv/bin/python -m src.models.experiments perm                 # permutation test over everything tried
    ./venv/bin/python -m src.models.experiments coef NAME

Every run is appended to data/models/experiments/log.jsonl. A candidate's hypothesis is written in its ``note``
before it is run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import cv
from src.models.dataset import load_split
from src.models.features import BASELINES, CODE_COLS, M1_COLS, Data, Recipe, onehot_cols, prepare, registered_recipes

OUT_DIR = Path("data/models/experiments")
LOG = OUT_DIR / "log.jsonl"
LABELS = ("hold_resid", "hold_resid_views")
MIN_REPEATS_BEATING = 9
P_MAX = 0.05


def candidates(data: Data) -> dict[str, Recipe]:
    """Loop candidates (iteration 1+), in the order they were written down. Never edit or delete an entry."""
    static = tuple(data.cols)
    log_m1 = tuple(f"log1p:{c}" if c in ("t_premise_clear_s", "tail_all", "dead_time_s") else c for c in M1_COLS)
    rs: list[Recipe] = [
        Recipe("c01_m1_capped_labels", cols=M1_COLS, y_cap=0.05,
               note="Iteration 0: every row shrinks to the mean. Two labels (-24.6, +28.1) sit 4-5 SD out and "
                    "dominate squared error. Hypothesis: capping training labels at the 5/95 % quantiles lets M1 "
                    "fit the bulk."),
        Recipe("c02_m1_log_seconds", cols=log_m1,
               note="Seconds features (premise time, tail, dead time) are right-skewed; a few long values dominate "
                    "a linear fit. Hypothesis: log1p makes the relation linear enough for ridge."),
        Recipe("c03_m1_static_frac", cols=tuple("static_frac" if c == "dead_time_s" else c for c in M1_COLS),
               note="dead_time_s passed the consistency gate with a wide CI. Hypothesis: its pre-named code "
                    "substitute static_frac (03a C6 -> G4) is less noisy."),
        Recipe("c04_select3", select_from=static, select_k=3,
               note="The univariate screen's strongest columns are outside M1 (opening, format, onsets). "
                    "Hypothesis: 3 columns picked inside each fold from all 52 static ones carry signal. The pick "
                    "is inside CV, so the score pays for it."),
        Recipe("c05_m1_plus_select3", cols=M1_COLS, select_from=static, select_k=3,
               note="Same as c04 but on top of the registered core."),
        Recipe("c06_format_only", cols=tuple(c for c in onehot_cols(data) if c.startswith(
            ("format_macro=", "opening_frame=", "premise_type="))),
               note="03a decision 7.2 kept format out of the core. Hypothesis: on this pool, what the video is "
                    "(format, opening, premise type) predicts hold better than how it is built."),
        Recipe("c07_select3_capped", select_from=static, select_k=3, y_cap=0.05,
               note="c01 and c04 together: fold-selected columns with capped labels."),
        Recipe("c08_format_capped", cols=tuple(c for c in onehot_cols(data) if c.startswith(
            ("format_macro=", "opening_frame=", "premise_type="))), y_cap=0.05,
               note="c06 is the only row with positive Spearman (0.09, R2 0.007, 8/10) and the only one that does "
                    "not shrink fully. c01 showed capping helps the constant. Hypothesis: together they clear 9/10."),
        Recipe("c09_format_plus_m1", cols=M1_COLS + tuple(c for c in onehot_cols(data) if c.startswith(
            ("format_macro=", "opening_frame=", "premise_type="))),
               note="Hypothesis: mechanics matter only once format is held fixed (a long tail means something "
                    "different in a loot opening than in a meme edit)."),
        Recipe("c10_all_descriptive_capped", cols=onehot_cols(data), y_cap=0.05,
               note="Last candidate under the cap. Hypothesis: the wider description (voice, text role, audio bed, "
                    "appeal, relatability, ending) adds to format; ridge decides how much."),
        # --- second loop (plan/03c section 13, pool train_dev); written before any of them was run ---
        Recipe("n01_m1_by_length", cols=tuple(f"{g}:{c}" for g in ("short", "long") for c in M1_COLS
                                              if c != "dhash_x_short") + ("short:log_duration",),
               note="Iteration 0 on train+DEV: still nothing, and no single feature separates DEV strong from "
                    "weak. 03a 1.1 says the label is replays under ~20 s and mid-video hold above, so one "
                    "coefficient per feature may average two opposite effects. Hypothesis: separate coefficients "
                    "for short and long videos (plus a length-group offset) find it."),
        Recipe("n02_m1_fractions", cols=tuple(f"frac:{c}" if c in ("t_premise_clear_s", "tail_all", "dead_time_s")
                                              else c for c in M1_COLS),
               note="The label is residualised on duration but the seconds features are not, so they mostly "
                    "measure length. Hypothesis: as shares of the video's length they carry the structure signal."),
        Recipe("n03_m1_plus_duration_era", cols=M1_COLS + ("log_duration", "era"),
               note="Same cause as n02, other fix: give ridge log(duration) and era so the mechanics coefficients "
                    "are read at fixed length (a suppressor setup)."),
        Recipe("n04_m1_squares", cols=M1_COLS, squares=True,
               note="n01-n03 changed nothing. Every model so far is a straight line per feature, and 03a marks "
                    "main_payoff_frac as direction unknown. Hypothesis: the effects are U-shaped (e.g. a payoff "
                    "neither too early nor too late), which a squared term per feature can fit."),
    ]
    return {r.name: r for r in rs}


def all_recipes(data: Data) -> dict[str, Recipe]:
    return registered_recipes(data) | candidates(data)


POOLS = {"train": ("train",), "train_dev": ("train", "dev")}
POOL = "train"


def log_path() -> Path:
    return LOG if POOL == "train" else OUT_DIR / f"log_{POOL}.jsonl"


def load():
    df = pd.concat([load_split(s) for s in POOLS[POOL]], ignore_index=True)
    ys = {lab: df[lab].to_numpy(dtype=float) for lab in LABELS}
    return prepare(df), ys | {"is_train": (df["split"] == "train").to_numpy()}


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def read_log() -> list[dict]:
    return [json.loads(line) for line in log_path().read_text().splitlines()] if log_path().exists() else []


def append_log(rec: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with log_path().open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run(names: list[str], everything: bool = False) -> int:
    data, ys = load()
    recipes = all_recipes(data)
    names = list(recipes) if everything else names or list(registered_recipes(data))
    core = ys["is_train"]
    res = {lab: {n: cv.nested_cv(data, ys[lab], recipes[n]) for n in dict.fromkeys(list(BASELINES) + names)}
           for lab in LABELS}
    print(f"{'recipe':28}{'params':>7}{'alpha':>9}{'R2':>8}{'Spearman':>10}{'+-':>6}{'MAE':>7}{'beats':>7}"
          f"{'Sp(views)':>11}{'R2(125)':>9}")
    for n in dict.fromkeys(list(BASELINES) + names):
        r, rv = res["hold_resid"][n], res["hold_resid_views"][n]
        base = res["hold_resid"]
        r2 = 1 - r["mse"] / base["mean_only"]["mse"]
        beats = min(int((r["mse"] < base[b]["mse"]).sum()) for b in BASELINES if b != n)
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": git_sha(), "recipe": n,
               "n_params": recipes[n].n_params, "note": recipes[n].note, "alpha_median": r["alpha_median"],
               "r2": float(r2.mean()), "r2_sd": float(r2.std()), "spearman": float(r["spearman"].mean()),
               "spearman_sd": float(r["spearman"].std()), "mae": float(r["mae"].mean()),
               "mse_per_repeat": r["mse"].tolist(), "repeats_beating_every_baseline": beats,
               "spearman_views": float(rv["spearman"].mean()),
               "r2_non_dev": float((1 - ((r["preds"][core] - ys["hold_resid"][core, None]) ** 2).mean(axis=0) / (
                   (base["mean_only"]["preds"][core] - ys["hold_resid"][core, None]) ** 2).mean(axis=0)).mean()),
               "r2_views": float((1 - rv["mse"] / res["hold_resid_views"]["mean_only"]["mse"]).mean())}
        if n in names:
            append_log(rec)
        print(f"{n:28}{rec['n_params']:>7}{rec['alpha_median']:>9.3g}{rec['r2']:>8.3f}{rec['spearman']:>10.3f}"
              f"{rec['spearman_sd']:>6.2f}{rec['mae']:>7.2f}{beats:>5}/10{rec['spearman_views']:>11.3f}{rec['r2_non_dev']:>9.3f}")
    return 0


def perm(n_perm: int) -> int:
    data, ys = load()
    recipes = all_recipes(data)
    tried = list(dict.fromkeys(r["recipe"] for r in read_log() if r.get("recipe") and r["recipe"] not in BASELINES))
    out = cv.permutation_test(data, ys["hold_resid"], [recipes[n] for n in tried], n_perm=n_perm)
    append_log({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": git_sha(),
                "permutation": out, "n_perm": n_perm, "tried": tried})
    latest = {r["recipe"]: r for r in read_log() if r.get("recipe")}
    print(f"{len(tried)} candidates tried, {n_perm} shuffles")
    print(f"{'recipe':28}{'Sp(1x10)':>9}{'p':>8}{'p_adj':>8}{'beats':>7}{'Sp(views)':>11}  acceptable")
    for n, o in out.items():
        r = latest[n]
        ok = (r["repeats_beating_every_baseline"] >= MIN_REPEATS_BEATING and o["p_adjusted"] <= P_MAX
              and r["spearman_views"] > 0)
        print(f"{n:28}{o['spearman_1x10']:>9.3f}{o['p']:>8.3f}{o['p_adjusted']:>8.3f}"
              f"{r['repeats_beating_every_baseline']:>5}/10{r['spearman_views']:>11.3f}  {'YES' if ok else 'no'}")
    return 0


def coef(name: str) -> int:
    data, ys = load()
    recipe = all_recipes(data)[name]
    for lab in LABELS:
        alpha, _ = cv.choose_alpha(data, ys[lab], recipe)
        c = cv.fit_full(data, ys[lab], recipe, alpha)
        print(f"\n{name} on {lab}: alpha = {alpha:.3g} (coefficient per 1 SD of the feature)")
        for k, v in sorted(c.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {k:36}{v:+.3f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    p.add_argument("--pool", choices=list(POOLS), default="train")
    r = sub.add_parser("run")
    r.add_argument("names", nargs="*")
    r.add_argument("--all", action="store_true", help="every registered row and every candidate")
    sub.add_parser("perm").add_argument("--n-perm", type=int, default=cv.N_PERM)
    sub.add_parser("coef").add_argument("name")
    a = p.parse_args(argv)
    global POOL
    POOL = a.pool
    return run(a.names, a.all) if a.cmd == "run" else perm(a.n_perm) if a.cmd == "perm" else coef(a.name)


if __name__ == "__main__":
    raise SystemExit(main())
