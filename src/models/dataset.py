"""One table per split for the retention model (plan/03c section 2).

Joins the labelled pool (``pool.py``) to the Gemini features and the code features,
applies the 03a section 4 null rule and adds the second label ``hold_resid_views``.

GATE and HELDOUT rows cannot be loaded until ``src/models/FROZEN`` exists.

    ./venv/bin/python -m src.models.dataset                     # train summary
    ./venv/bin/python -m src.models.dataset --verify-durations  # ffprobe every blinded file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.models.pool import DUPLICATES_EXCLUDE, labelled_pool, train_rows

FEATURES_DIR = Path("data/features")
GEMINI_TRAIN = FEATURES_DIR / "gemini_features_train_v2.csv"
GEMINI_EVAL = FEATURES_DIR / "gemini_features_eval_v2.csv"
CODE_FEATURES = FEATURES_DIR / "code_features.csv"
TRAIN_SELECTION = FEATURES_DIR / "gemini" / "train_selection.csv"
FROZEN_MARKER = Path(__file__).parent / "FROZEN"

SPLITS = ("train", "dev", "gate", "heldout")
LOCKED_SPLITS = ("gate", "heldout")
HELDOUT_OPEN_LINE = "heldout_open=true"
IS_SHORT_MAX_S = 20.0
DURATION_TOLERANCE_S = 0.1
GEMINI_BOOKKEEPING = ["split", "status", "invalid_reason", "source_run_id"]


class SplitLocked(RuntimeError):
    pass


def require_unlocked(split: str, marker: Path = FROZEN_MARKER) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split in LOCKED_SPLITS and not Path(marker).exists():
        raise SplitLocked(f"{split} is locked until the model is frozen ({marker} missing, plan/03c section 7)")
    if split == "heldout" and HELDOUT_OPEN_LINE not in Path(marker).read_text().splitlines():
        raise SplitLocked(f"heldout stays locked until GATE has been reviewed ({HELDOUT_OPEN_LINE!r} not in {marker})")


def add_hold_resid_views(pool: pd.DataFrame) -> pd.DataFrame:
    """``hold_resid`` residualised on log(engagedViews), fitted on every row given (03a decision 7.1)."""
    pool = pool.copy()
    X = np.column_stack([np.ones(len(pool)), np.log(pool["engagedViews"].astype(float))])
    y = pool["hold_resid"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pool["hold_resid_views"] = y - X @ beta
    return pool


def assemble(pool: pd.DataFrame, gemini: pd.DataFrame, code: pd.DataFrame) -> pd.DataFrame:
    """Join label rows to both feature tables; every pool row must have valid features."""
    for name, df in (("pool", pool), ("gemini", gemini), ("code", code)):
        if df["video_id"].duplicated().any():
            raise ValueError(f"duplicate video_id in {name}")
    ids = set(pool["video_id"])
    for name, df in (("gemini", gemini), ("code", code)):
        missing = sorted(ids - set(df["video_id"]))
        if missing:
            raise ValueError(f"{len(missing)} pool videos have no {name} features, e.g. {missing[:3]}")
    gemini = gemini[gemini["video_id"].isin(ids)]
    if "status" in gemini and (gemini["status"] != "ok").any():
        raise ValueError("gemini features contain rows with status != ok")
    gemini = gemini.drop(columns=[c for c in GEMINI_BOOKKEEPING if c in gemini])
    df = pool.merge(gemini, on="video_id", validate="one_to_one").merge(code, on="video_id", validate="one_to_one")
    df["t_premise_clear_s"] = df["t_premise_clear_s"].fillna(df["duration_s"])
    df["is_short"] = (df["duration_s"] <= IS_SHORT_MAX_S).astype(float)
    return df.reset_index(drop=True)


def load_split(split: str = "train", marker: Path = FROZEN_MARKER) -> pd.DataFrame:
    require_unlocked(split, marker)
    pool = add_hold_resid_views(labelled_pool())
    if split == "train":
        pool = pool[pool["video_id"].isin(set(train_rows()["video_id"]))]
    else:
        pool = pool[pool["split"] == split]
    gemini = pd.read_csv(GEMINI_TRAIN if split == "train" else GEMINI_EVAL)
    return assemble(pool.reset_index(drop=True), gemini, pd.read_csv(CODE_FEATURES))


def blind_paths() -> pd.DataFrame:
    """video_id -> blinded file, for all splits. Reads ids and paths only, no features or labels."""
    train = pd.read_csv(TRAIN_SELECTION)[["video_id", "blind_path"]]
    ev = pd.read_csv(GEMINI_EVAL, usecols=["video_id", "blind_id"])
    ev["blind_path"] = [str(C.BLIND_DIR / f"{b}.mp4") for b in ev["blind_id"]]
    return pd.concat([train, ev[["video_id", "blind_path"]]], ignore_index=True)


def verify_durations(tolerance_s: float = DURATION_TOLERANCE_S) -> pd.DataFrame:
    """Rows where code_features ``duration_s`` and the blinded file's ffprobe duration differ by more than the tolerance."""
    from src.tools.analyze_video.extract import blind_duration

    df = blind_paths().merge(pd.read_csv(CODE_FEATURES, usecols=["video_id", "duration_s"]), on="video_id")
    df["blind_duration_s"] = [blind_duration(Path(p)) for p in df["blind_path"]]
    df["diff_s"] = (df["duration_s"] - df["blind_duration_s"]).abs()
    print(f"checked {len(df)} files; max |diff| {df['diff_s'].max():.3f} s")
    return df[df["diff_s"] > tolerance_s]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--verify-durations", action="store_true")
    args = p.parse_args(argv)
    if args.verify_durations:
        bad = verify_durations()
        if len(bad):
            print(bad[["video_id", "duration_s", "blind_duration_s", "diff_s"]].to_string(index=False))
            return 1
        return 0
    df = load_split("train")
    print(f"train: {len(df)} rows x {df.shape[1]} columns (duplicates file: {DUPLICATES_EXCLUDE})")
    print(f"is_short: {int(df['is_short'].sum())}; nulls left: {int(df.isna().any(axis=1).sum())} rows")
    print(df[["hold_resid", "hold_resid_views"]].describe().round(2).to_string())
    print(f"corr(hold_resid_views, log engagedViews) = {np.corrcoef(df['hold_resid_views'], np.log(df['engagedViews']))[0, 1]:.3f}")
    print(f"corr(hold_resid, hold_resid_views) = {df['hold_resid'].corr(df['hold_resid_views']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
