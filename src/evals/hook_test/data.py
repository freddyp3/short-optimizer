"""Load the channel-export tables the hook test needs from one data/channel/<ts>/ directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.evals.hook_test.categorize import latest_export
from src.evals.hook_test.config import EXPORT_BASE


@dataclass
class ExportTables:
    videos: pd.DataFrame
    metrics: pd.DataFrame
    retention: pd.DataFrame
    daily: pd.DataFrame
    export_dir: Path | None = None


def load_export(export_dir: Path | None = None) -> ExportTables:
    """Read videos/video_metrics/retention/video_daily from export_dir (default: newest full export)."""
    export_dir = export_dir or latest_export(EXPORT_BASE)
    read = lambda name: pd.read_csv(export_dir / f"{name}.csv")
    return ExportTables(
        videos=read("videos"),
        metrics=read("video_metrics"),
        retention=read("retention"),
        daily=read("video_daily"),
        export_dir=export_dir,
    )
