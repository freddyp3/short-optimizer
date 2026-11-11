"""Ridge and cross-validation for the retention model (plan/03c section 4), numpy only.

``nested_cv`` gives the reported score: the penalty is re-chosen inside every outer training fold.
``choose_alpha`` picks the penalty for the final fit. ``permutation_test`` asks whether a score could be chance,
and charges for every candidate that was tried (plan/03c section 10.1).
"""

from __future__ import annotations

import numpy as np

from src.evals.hook_test.stats import spearman
from src.models.features import Data, Recipe, design

ALPHAS = np.logspace(-2, 4, 25)
SEED = 340
K, REPEATS, N_PERM = 10, 10, 1000


def ridge_path(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, alphas=ALPHAS):
    """Predictions (n_test, n_alphas) and coefficients (p, n_alphas). X is already z-scored; the intercept
    (the training mean of y) is not penalised."""
    mu = ytr.mean()
    if Xtr.shape[1] == 0:
        return np.full((len(Xte), len(alphas)), mu), np.empty((0, len(alphas)))
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    shrink = s[:, None] / (s[:, None] ** 2 + np.asarray(alphas)[None, :])
    coef = Vt.T @ (shrink * (U.T @ (ytr - mu))[:, None])
    return mu + Xte @ coef, coef


def kfold(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [np.sort(f) for f in np.array_split(rng.permutation(n), k)]


def oof_path(data: Data, y: np.ndarray, recipe: Recipe, rows: np.ndarray, folds, alphas=ALPHAS) -> np.ndarray:
    """Out-of-fold predictions for ``rows`` (positions into data), one column per alpha."""
    out = np.empty((len(rows), len(alphas)))
    for f in folds:
        mask = np.ones(len(rows), bool)
        mask[f] = False
        Xtr, Xte, ytr, _ = design(recipe, data, rows[mask], y[rows[mask]], te=rows[f])
        out[f] = ridge_path(Xtr, ytr, Xte, alphas)[0]
    return out


def _best(mse: np.ndarray) -> int:
    return int(np.flatnonzero(mse <= mse.min() * (1 + 1e-12))[-1])  # ties -> the larger penalty


def choose_alpha(data: Data, y: np.ndarray, recipe: Recipe, alphas=ALPHAS, k=K, repeats=REPEATS, seed=SEED):
    rows = np.arange(len(data))
    mse = np.mean([((oof_path(data, y, recipe, rows, kfold(len(rows), k, np.random.default_rng(seed + r)), alphas)
                     - y[:, None]) ** 2).mean(axis=0) for r in range(repeats)], axis=0)
    return float(alphas[_best(mse)]), mse


def nested_cv(data: Data, y: np.ndarray, recipe: Recipe, alphas=ALPHAS, k=K, repeats=REPEATS, seed=SEED) -> dict:
    """Outer k-fold x repeats; inside each outer training fold the penalty is chosen by one inner k-fold."""
    n = len(data)
    preds = np.empty((n, repeats))
    chosen = []
    for r in range(repeats):
        for j, f in enumerate(kfold(n, k, np.random.default_rng(seed + r))):
            tr = np.setdiff1d(np.arange(n), f)
            inner = kfold(len(tr), k, np.random.default_rng(seed + 1000 * (r + 1) + j))
            a = _best(((oof_path(data, y, recipe, tr, inner, alphas) - y[tr][:, None]) ** 2).mean(axis=0))
            Xtr, Xte, ytr, _ = design(recipe, data, tr, y[tr], te=f)
            preds[f, r] = ridge_path(Xtr, ytr, Xte, alphas[[a]])[0][:, 0]
            chosen.append(float(alphas[a]))
    err = preds - y[:, None]
    with np.errstate(invalid="ignore"):
        rho = np.array([spearman(preds[:, r], y) for r in range(repeats)])
    return {"mse": (err ** 2).mean(axis=0), "mae": np.abs(err).mean(axis=0), "spearman": np.nan_to_num(rho),
            "alpha_median": float(np.median(chosen)), "preds": preds}


def fit_full(data: Data, y: np.ndarray, recipe: Recipe, alpha: float) -> dict:
    """Fit on every row; coefficients are per standard deviation of each feature."""
    rows = np.arange(len(data))
    X, _, yu, names = design(recipe, data, rows, y, te=rows[:0])
    return dict(zip(names, ridge_path(X, yu, X[:0], np.array([alpha]))[1][:, 0]))


def perm_stat(data: Data, y: np.ndarray, recipe: Recipe, folds, alphas=ALPHAS) -> float:
    """Spearman of out-of-fold predictions from one k-fold CV, at the penalty with the lowest CV error."""
    p = oof_path(data, y, recipe, np.arange(len(data)), folds, alphas)
    rho = spearman(p[:, _best(((p - y[:, None]) ** 2).mean(axis=0))], y)
    return 0.0 if rho != rho else rho


def permutation_test(data: Data, y: np.ndarray, recipes: list[Recipe], n_perm=N_PERM, k=K, seed=SEED) -> dict:
    """Per-recipe p, and p adjusted for the search: each shuffle is scored on every recipe and keeps the best."""
    folds = kfold(len(data), k, np.random.default_rng(seed))
    obs = np.array([perm_stat(data, y, r, folds) for r in recipes])
    rng = np.random.default_rng(seed + 1)
    null = np.empty((n_perm, len(recipes)))
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i] = [perm_stat(data, yp, r, folds) for r in recipes]
    p = (1 + (null >= obs[None, :]).sum(axis=0)) / (1 + n_perm)
    p_adj = (1 + (null.max(axis=1)[:, None] >= obs[None, :]).sum(axis=0)) / (1 + n_perm)
    return {r.name: {"spearman_1x10": float(o), "p": float(a), "p_adjusted": float(b)}
            for r, o, a, b in zip(recipes, obs, p, p_adj)}
