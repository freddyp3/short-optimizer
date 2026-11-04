"""The labelled pool for the retention model (plan/03 section 0b step 4).

Reuses the hook test's eligibility funnel and ``hold_resid`` so the label is
identical to the one the hook test is scored on. Adds a ``split`` column:

    train    eligible pool minus every hook-test video  (~126 rows; the only rows
             anyone may look at per-video before the model is frozen). Rows listed
             in data/features/duplicates_exclude.json are dropped (plan/03b
             "Duplicates") unless exclude_duplicates=False.
    dev / gate / heldout   from data/evals/hook_test/selection.json

    ./venv/bin/python -m src.models.pool            # prints counts per split
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.data import load_export
from src.evals.hook_test.select import build_frame, eligible_pool, label_candidates

POOL_COLS = ["video_id", "split", "hold_resid", "hold_score", "duration_seconds", "published_at",
             "engagedViews", "mean_rrp", "hook3s"]


def labelled_pool(export_dir: Path | None = None,
                  selection_json: Path = C.SELECTION_JSON_PATH,
                  available: set[str] | None = None) -> pd.DataFrame:
    """Eligible Shorts with ``hold_resid`` and a split column. Deterministic, no network."""
    sel = json.loads(Path(selection_json).read_text())
    export_dir = export_dir or Path(sel["export_dir"])
    frame = build_frame(load_export(export_dir))
    pool, _ = eligible_pool(frame, available, C.MAX_POST_CHANGE_SHARE)
    pool, _ = label_candidates(pool)
    split = {v["video_id"]: v["split"] for v in sel["videos"]}
    pool["split"] = pool["video_id"].map(split).fillna("train")
    return pool[POOL_COLS].reset_index(drop=True)


DUPLICATES_EXCLUDE = Path("data/features/duplicates_exclude.json")


def train_rows(exclude_duplicates: bool = True, **kw) -> pd.DataFrame:
    df = labelled_pool(**kw).query("split == 'train'").reset_index(drop=True)
    if exclude_duplicates and DUPLICATES_EXCLUDE.exists():
        excl = set(json.loads(DUPLICATES_EXCLUDE.read_text())["exclude"])
        df = df[~df["video_id"].isin(excl)].reset_index(drop=True)
    return df


if __name__ == "__main__":
    p = labelled_pool()
    print(p["split"].value_counts().to_string())
    print(f"\nhold_resid on train: mean {p.query('split == \"train\"').hold_resid.mean():.2f}, "
          f"sd {p.query('split == \"train\"').hold_resid.std():.2f}")
