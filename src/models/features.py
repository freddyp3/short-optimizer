"""Feature recipes for the retention model (plan/03c section 3, model rows from plan/03a section 5).

``prepare`` turns the table from ``dataset.py`` into a numeric ``Data`` object (outcome-blind columns only).
A ``Recipe`` names the columns a model uses. Anything that looks at labels (the channel-history features H1-H3,
within-fold feature selection, z-scoring) is fitted on the training fold inside ``design``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C

PREVALENCE = Path("data/features/prevalence.json")
MIN_ROWS = 10
IS_SHORT_MAX_S = 20.0
PEAK_ALIGN_TOLERANCE_S = 1.0
SECONDS_PER_YEAR = 365.25 * 24 * 3600

H_FIELDS = ["format_macro", "voice", "premise_type", "audio_bed", "primary_appeal", "relatability",
            "game_moment_kind"]
H_K, H_SHRINK, H_YEAR_WEIGHT, RECENT_N = 5, 3, 0.1, 5

M1_COLS = ("t_premise_clear_s", "payoff_count_all", "main_payoff_frac", "tail_all", "dead_time_s", "arc_ord",
           "first_last_dhash_dist", "dhash_x_short", "cuts_per_s")
CODE_COLS = ("cuts_per_s", "cuts_first3s", "cuts_last3s", "motion_mean", "motion_first3s", "motion_last3s",
             "static_frac", "black_frac", "first_last_dhash_dist", "loudness_lufs", "loudness_range",
             "silence_frac", "audio_onset_s", "onsets_per_s")
FULL_NUMERIC = ("t_first_event_s", "surprise_ord", "log_duration", "era")
FULL_BOOLS = ("big_number_on_screen", "low_health_visible", "opening_dead", "dead_ending",
              "loudness_peak_aligned_with_payoff")
SURPRISE_ORD = {"none": 0, "mild": 1, "strong": 2}


@dataclass
class Data:
    """Numeric view of one split. ``M`` holds every static column; history features are built per fold."""
    video_id: np.ndarray
    cols: list[str]
    M: np.ndarray
    codes: np.ndarray          # (n, len(H_FIELDS)) descriptive levels as strings
    years: np.ndarray          # publish time in years
    _cache: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.video_id)

    def col(self, name: str) -> np.ndarray:
        if name not in self._cache:
            if name == "years":
                self._cache[name] = self.years
            elif name.startswith("log1p:"):
                self._cache[name] = np.log1p(np.clip(self.col(name[6:]), 0, None))
            elif name.startswith("frac:"):        # seconds as a share of the video's length
                self._cache[name] = self.col(name[5:]) / np.exp(self.col("log_duration"))
            elif name.startswith(("short:", "long:")):   # the column inside one length group, 0 outside it
                kind, base = name.split(":", 1)
                short = np.exp(self.col("log_duration")) <= IS_SHORT_MAX_S
                self._cache[name] = self.col(base) * (short if kind == "short" else ~short)
            else:
                self._cache[name] = self.M[:, self.cols.index(name)]
        return self._cache[name]

    def matrix(self, names) -> np.ndarray:
        return np.column_stack([self.col(n) for n in names]) if names else np.empty((len(self), 0))


def onehot_levels(prevalence: dict) -> dict[str, list[str]]:
    """Kept levels per nominal field, minus the most common one (the reference level)."""
    out = {}
    for f, r in prevalence["nominal"].items():
        if r["decision"] != "keep":
            continue
        counts = {lv: (r["other_count"] if lv == "other" else r["counts"].get(lv, 0)) for lv in r["levels"]}
        ref = max(counts, key=counts.get)
        out[f] = [lv for lv in r["levels"] if lv != ref]
    return out


def prepare(df: pd.DataFrame, prevalence: dict | None = None) -> Data:
    prevalence = prevalence or json.loads(PREVALENCE.read_text())
    d = df.copy()
    published = pd.to_datetime(d["published_at"], utc=True)
    d["years"] = (published - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() / SECONDS_PER_YEAR
    d["era"] = (published >= pd.Timestamp(C.ERA_CUTOFF, tz="UTC")).astype(float)
    d["log_duration"] = np.log(d["duration_s"])
    d["dhash_x_short"] = d["first_last_dhash_dist"] * d["is_short"]
    d["surprise_ord"] = d["surprise_peak"].astype(str).map(SURPRISE_ORD)
    d["opening_dead"] = d["opening_frame"].isin(["menu_or_lobby", "loading_or_black"])
    d["dead_ending"] = d["ending_from_tail_all"].isin(["long_tail", "outro_or_black"])
    d["loudness_peak_aligned_with_payoff"] = (
        (d["t_loudness_peak_s"] - d["t_main_payoff_s"]).abs() <= PEAK_ALIGN_TOLERANCE_S)
    names = list(dict.fromkeys(M1_COLS + CODE_COLS + FULL_NUMERIC + FULL_BOOLS))
    for f, levels in onehot_levels(prevalence).items():
        kept = set(prevalence["nominal"][f]["levels"]) - {"other"}
        level = d[f].astype(str).where(d[f].astype(str).isin(kept), "other")
        for lv in levels:
            d[f"{f}={lv}"] = level == lv
            names.append(f"{f}={lv}")
    return Data(video_id=d["video_id"].to_numpy(), cols=names, M=d[names].to_numpy(dtype=float),
                codes=d[H_FIELDS].astype(str).to_numpy(), years=d["years"].to_numpy(dtype=float))


def onehot_cols(data: Data) -> tuple[str, ...]:
    return tuple(c for c in data.cols if "=" in c)


@dataclass(frozen=True)
class Recipe:
    name: str
    cols: tuple[str, ...] = ()
    h1: bool = False                       # similar_past_mean
    h2: bool = False                       # nearest_past_dist
    recent_form: bool = False              # H3 recent_form_5
    select_from: tuple[str, ...] = ()      # extra columns picked inside each fold by |correlation| with the label
    select_k: int = 0
    y_cap: float | None = None             # clip training labels to the [q, 1-q] quantiles of the fold
    squares: bool = False                  # add the square of every z-scored feature (allows U-shapes)
    note: str = ""

    @property
    def n_params(self) -> int:
        return (len(self.cols) + self.h1 + self.h2 + self.recent_form + self.select_k) * (1 + self.squares)


def history_features(data: Data, fit_idx: np.ndarray, y_fit: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """H1 similar_past_mean, H2 nearest_past_dist, H3 recent_form_5 for ``rows``; neighbours are fitted-fold videos
    published strictly earlier. No earlier video: H1 = H3 = 0, H2 = NaN (filled with the fold mean later)."""
    fy = data.years[fit_idx]
    dt = data.years[rows][:, None] - fy[None, :]
    mismatch = (data.codes[rows][:, None, :] != data.codes[fit_idx][None, :, :]).mean(axis=2)
    dist = mismatch + H_YEAR_WEIGHT * np.abs(dt)
    out = np.zeros((len(rows), 3))
    out[:, 1] = np.nan
    for i in range(len(rows)):
        earlier = np.flatnonzero(dt[i] > 0)
        if not len(earlier):
            continue
        near = earlier[np.lexsort((-fy[earlier], dist[i, earlier]))][:H_K]
        out[i, 0] = y_fit[near].sum() / (H_K + H_SHRINK)
        out[i, 1] = dist[i, near[0]]
        out[i, 2] = y_fit[earlier[np.argsort(-fy[earlier])][:RECENT_N]].mean()
    return out


def design(recipe: Recipe, data: Data, tr: np.ndarray, y_tr: np.ndarray,
           te_data: Data | None = None, te: np.ndarray | None = None):
    """(X_train, X_test, y_train_used, names), z-scored on the training rows. Test rows default to ``data[te]``;
    pass ``te_data`` to score another split against neighbours from ``data[tr]``."""
    te_data = te_data or data
    te = np.arange(len(te_data)) if te is None else te
    if recipe.y_cap is not None:
        y_tr = np.clip(y_tr, *np.quantile(y_tr, [recipe.y_cap, 1 - recipe.y_cap]))
    names = list(recipe.cols)
    if recipe.select_k:
        cand = [c for c in recipe.select_from if c not in names]
        Xc = data.matrix(cand)[tr]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num([abs(np.corrcoef(np.nan_to_num(x, nan=np.nanmean(x)), y_tr)[0, 1]) for x in Xc.T])
        names += [cand[j] for j in np.argsort(-r, kind="stable")[:recipe.select_k]]
    Xtr, Xte = data.matrix(names)[tr], te_data.matrix(names)[te]
    hist = [j for j, on in enumerate((recipe.h1, recipe.h2, recipe.recent_form)) if on]
    if hist:
        Htr = history_features(data, tr, y_tr, tr)
        if te_data is data:
            Hte = history_features(data, tr, y_tr, te)
        else:
            both = _stack(data, tr, te_data, te)
            n = len(tr)
            Hte = history_features(both, np.arange(n), y_tr, np.arange(n, n + len(te)))
        Xtr, Xte = np.column_stack([Xtr, Htr[:, hist]]), np.column_stack([Xte, Hte[:, hist]])
        names += [("similar_past_mean", "nearest_past_dist", "recent_form_5")[j] for j in hist]
    if not names:
        return Xtr, Xte, y_tr, names
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd[~(sd > 0)] = 1.0
    z = lambda X: np.nan_to_num((X - mu) / sd, nan=0.0)
    Xtr, Xte = z(Xtr), z(Xte)
    if recipe.squares:
        Xtr, Xte = np.column_stack([Xtr, Xtr ** 2 - 1]), np.column_stack([Xte, Xte ** 2 - 1])
        names += [f"{n}^2" for n in names]
    return Xtr, Xte, y_tr, names


def _stack(a: Data, ia: np.ndarray, b: Data, ib: np.ndarray) -> Data:
    return Data(video_id=np.concatenate([a.video_id[ia], b.video_id[ib]]), cols=[], M=np.empty((0, 0)),
                codes=np.concatenate([a.codes[ia], b.codes[ib]]), years=np.concatenate([a.years[ia], b.years[ib]]))


def registered_recipes(data: Data) -> dict[str, Recipe]:
    """The model rows and baselines exactly as registered in plan/03a section 5 (iteration 0)."""
    n = len(data)
    bools = tuple(b for b in FULL_BOOLS if MIN_ROWS <= np.nansum(data.col(b)) <= n - MIN_ROWS)  # prevalence rule
    full_static = tuple(dict.fromkeys(M1_COLS + FULL_NUMERIC + bools + CODE_COLS + onehot_cols(data)))
    rs = [
        Recipe("mean_only"),
        Recipe("date_only", cols=("years",)),
        Recipe("duration_only", cols=("log_duration",)),
        Recipe("recent_form_5", recent_form=True),
        Recipe("M0_theme_only", h1=True),
        Recipe("M1_core_mechanics", cols=M1_COLS),
        Recipe("M2_core", cols=M1_COLS, h1=True),
        Recipe("M3_full", cols=full_static, h1=True, h2=True),
        Recipe("M4_code_only", cols=CODE_COLS),
    ]
    return {r.name: r for r in rs}


BASELINES = ("mean_only", "date_only", "duration_only", "recent_form_5")
