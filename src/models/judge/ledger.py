"""Spending ledger and caps for the Gemini judging work (plan/03c section 20.0, 20.11).

Every paid call appends one line. Runners call ``check`` before each call; a call that would cross a cap is refused.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path("data/models/judge/ledger.jsonl")
HARD_CAP_USD = 50.0
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def entries(path: Path = LEDGER) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def total(path: Path = LEDGER, stage: str | None = None) -> float:
    return float(sum(e["usd"] for e in entries(path) if stage is None or e["stage"] == stage))


def check(next_call_usd: float, stage: str, stage_cap_usd: float, path: Path = LEDGER,
          hard_cap_usd: float = HARD_CAP_USD) -> None:
    """Raise if this call would take the stage or the whole section over its cap."""
    if total(path, stage) + next_call_usd > stage_cap_usd:
        raise BudgetExceeded(f"stage {stage}: ${total(path, stage):.2f} spent, cap ${stage_cap_usd:.2f}")
    if total(path) + next_call_usd > hard_cap_usd:
        raise BudgetExceeded(f"section total ${total(path):.2f} spent, hard cap ${hard_cap_usd:.2f}")


def record(usd: float, stage: str, what: str, path: Path = LEDGER) -> None:
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stage": stage,
                                "what": what, "usd": round(float(usd), 6)}) + "\n")
