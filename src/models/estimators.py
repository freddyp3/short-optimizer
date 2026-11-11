"""Other estimators against ridge on the targets that work (plan/03c section 15). Train + DEV only.

    ./venv/bin/python -m src.models.estimators                    # elastic net, random forest, boosted trees
    ./venv/bin/python -m src.models.estimators --models tabpfn    # needs `pip install tabpfn` and TABPFN_TOKEN in .env

Models, tuning grids and the rule were fixed before running: a model replaces ridge for a pair only if its error
is lower in all 3 repeats and its R2 is higher by >= 0.02, and it then passes a permutation test.

Results (2026-11-11, train + DEV, 145 videos; nested-CV R2 / rank correlation, same folds for every model)
---------------------------------------------------------------------------------------------------------
    pair                     ridge        elastic net   random forest   boosted trees   TabPFN
    late_hold  x M1 core     0.09 / 0.40  0.07 / 0.39   0.24 / 0.52     0.19 / 0.48     0.28 / 0.55
    late_hold  x M3 full     0.16 / 0.44  0.13 / 0.43   0.23 / 0.50     0.21 / 0.49     0.24 / 0.51
    share_rate x M1 core     0.12 / 0.38  0.11 / 0.38   0.12 / 0.35     0.17 / 0.39     0.16 / 0.40
    share_rate x M3 full     0.21 / 0.48  0.20 / 0.46   0.19 / 0.44     0.14 / 0.40     0.24 / 0.50
    share_rate x format      0.18 / 0.43  0.17 / 0.39   0.15 / 0.34     0.17 / 0.40     0.18 / 0.41
    early_hold x M1 core     0.11 / 0.31  0.13 / 0.31   0.16 / 0.34     0.17 / 0.33     0.14 / 0.31

Decision: the random forest replaces ridge for late_hold with the M1 features. Its permutation test (fixed
settings min_samples_leaf 5, max_features 0.33, 719 shuffles): observed rank correlation 0.517, largest shuffle
0.308, p = 0.0014. Ridge stays for share_rate (M3 or format only). Elastic net never won.

TabPFN (tabular foundation model, tabpfn 9.2.0, default settings) was tested as a benchmark and is not used:
it was the strongest model on late_hold (0.55 vs the forest's 0.52), but its weights are licensed for
non-commercial, non-production use and need an account and token to download, so anyone running this repo would
need their own. Its permutation test was stopped before it finished; its numbers above are CV scores only.
tabpfn is deliberately not in requirements.txt.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

from src.evals.hook_test.stats import spearman
from src.models import cv, targets as T
from src.models.features import design

PAIRS = (("late_hold", "M1_core_mechanics"), ("late_hold", "M3_full_static"), ("share_rate", "M1_core_mechanics"),
         ("share_rate", "M3_full_static"), ("share_rate", "c06_format_only"), ("early_hold", "M1_core_mechanics"))
REPEATS, INNER_K = 3, 5
MIN_R2_GAIN = 0.02
SEED = cv.SEED

MODELS = {
    "elastic_net": (lambda: ElasticNet(max_iter=20_000),
                    {"l1_ratio": [0.2, 0.5, 0.8, 1.0], "alpha": list(np.logspace(-3, 1, 13))}),
    "random_forest": (lambda: RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
                      {"min_samples_leaf": [3, 5, 10], "max_features": [0.33, 1.0]}),
    "boosted_trees": (lambda: GradientBoostingRegressor(learning_rate=0.05, subsample=0.8, min_samples_leaf=5,
                                                        random_state=SEED),
                      {"n_estimators": [50, 100, 200], "max_depth": [1, 2]}),
}


def make_tabpfn():
    from dotenv import load_dotenv
    from tabpfn import TabPFNRegressor
    load_dotenv()
    return TabPFNRegressor(random_state=SEED)


MODELS["tabpfn"] = (make_tabpfn, {})   # default settings, nothing to tune


def nested(data, y: np.ndarray, recipe, make, grid, repeats: int = REPEATS) -> np.ndarray:
    """Out-of-fold predictions (n, repeats) on the same outer folds as cv.nested_cv."""
    n = len(data)
    preds = np.empty((n, repeats))
    for r in range(repeats):
        for j, f in enumerate(cv.kfold(n, cv.K, np.random.default_rng(SEED + r))):
            tr = np.setdiff1d(np.arange(n), f)
            Xtr, Xte, ytr, _ = design(recipe, data, tr, y[tr], te=f)
            if not grid:
                preds[f, r] = make().fit(Xtr, ytr).predict(Xte)
                continue
            inner = KFold(INNER_K, shuffle=True, random_state=SEED + 1000 * (r + 1) + j)
            search = GridSearchCV(make(), grid, cv=inner, scoring="neg_mean_squared_error").fit(Xtr, ytr)
            preds[f, r] = search.predict(Xte)
    return preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=["elastic_net", "random_forest", "boosted_trees"])
    chosen = ap.parse_args(argv).models
    df, data, recipes, _ = T.load()
    print(f"{'target':12}{'recipe':20}{'model':16}{'R2':>8}{'Spearman':>10}{'vs ridge':>10}{'wins':>6}  replaces ridge")
    for target, rname in PAIRS:
        y, recipe = df[target].to_numpy(dtype=float), recipes[rname]
        mse = lambda p: ((p - y[:, None]) ** 2).mean(axis=0)
        base = mse(cv.nested_cv(data, y, recipes["mean_only"], repeats=REPEATS)["preds"])
        ridge = cv.nested_cv(data, y, recipe, repeats=REPEATS)
        r2_ridge = float((1 - ridge["mse"] / base).mean())
        print(f"{target:12}{rname:20}{'ridge':16}{r2_ridge:>8.3f}{ridge['spearman'].mean():>10.3f}")
        for mname, (make, grid) in ((m, MODELS[m]) for m in chosen):
            p = nested(data, y, recipe, make, grid)
            r2 = float((1 - mse(p) / base).mean())
            wins = int((mse(p) < ridge["mse"]).sum())
            rho = float(np.mean([spearman(p[:, r], y) for r in range(REPEATS)]))
            ok = wins == REPEATS and r2 - r2_ridge >= MIN_R2_GAIN
            T.append_log({"estimator": mname, "target": target, "recipe": rname, "r2": r2, "r2_ridge": r2_ridge,
                          "spearman": rho, "repeats_beating_ridge": wins, "candidate_to_replace_ridge": ok})
            print(f"{'':32}{mname:16}{r2:>8.3f}{rho:>10.3f}{r2 - r2_ridge:>+10.3f}{wins:>4}/{REPEATS}  "
                  f"{'CANDIDATE (needs permutation test)' if ok else 'no'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
