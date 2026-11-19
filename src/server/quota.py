"""Draft quota and spending caps of the Claude MCP server (plan/10a section 4, plan/10 section 12.2-12.3).

Same SQLite file as the store. One table, ``spend``, one row per event:

    reservation   the estimate, written by ``check_and_reserve`` before any paid call (server-paid drafts)
    cost          running cost of an unsettled draft (``add_cost``)
    actual        final cost; replaces reservation + cost at ``settle``
    own_key       an unsettled draft run on the user's own Gemini key (usd 0; never counted against the caps)
    own_cost      running cost of an unsettled own-key draft
    own_actual    a settled own-key draft (usd is informational)
    refused_KIND  a refused attempt, for the daily summary; never counted as a draft

Spend counted against the caps is, per draft, the larger of reservation and running cost until it is settled,
then the actual cost, also where that is above the estimate. A new server-paid draft is refused when that sum plus
one estimate would pass the day or month cap, and a running job is stopped once its recorded cost passes
``BUDGET_FACTOR`` estimates, so a day's recorded spend is bounded (see ``check_and_reserve``). A draft belongs to
the UTC day and month in which it was reserved.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.server import config
from src.server.store import PAID_BY, connect, init_schema, transaction

EPS = 1e-9
REFUSAL_KINDS = ("user_daily", "day_cap", "month_cap", "own_key_daily")
_COUNTED = "('reservation', 'cost', 'actual')"

SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    month TEXT NOT NULL,
    token_id TEXT NOT NULL,
    draft_id TEXT NOT NULL,
    usd REAL NOT NULL,
    kind TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS spend_day ON spend (day, kind);
CREATE INDEX IF NOT EXISTS spend_month ON spend (month, kind);
CREATE INDEX IF NOT EXISTS spend_draft ON spend (draft_id);
"""


class LimitReached(Exception):
    """A quota or cap refuses the draft. ``kind`` is one of REFUSAL_KINDS; ``retry_after_utc`` is ISO 8601."""

    def __init__(self, kind: str, retry_after_utc: str):
        super().__init__(f"limit reached: {kind}, retry after {retry_after_utc}")
        self.kind, self.retry_after_utc = kind, retry_after_utc


class BudgetExceeded(RuntimeError):
    """The draft has cost more than ``cfg.BUDGET_FACTOR`` times the estimate, or has no reservation; stop the job."""


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class Quota:
    def __init__(self, db_path: Path | str, clock: Callable[[], float] = time.time, cfg: config.Config = config.CONFIG):
        self.db_path, self.clock, self.cfg = Path(db_path), clock, cfg
        init_schema(self.db_path, SCHEMA)

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), timezone.utc)

    def _periods(self) -> tuple[str, str, str, str]:
        """(day, month, next UTC midnight, first instant of next month)."""
        now = self._now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        next_month = (midnight.replace(day=1) + timedelta(days=32)).replace(day=1)
        return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m"), _iso(midnight + timedelta(days=1)), _iso(next_month)

    @staticmethod
    def _spent(con: sqlite3.Connection, column: str, value: str) -> float:
        return con.execute(f"SELECT COALESCE(SUM(usd), 0) FROM (SELECT MAX(usd) AS usd FROM spend WHERE {column} = ? "
                           f"AND kind IN {_COUNTED} GROUP BY draft_id)", (value,)).fetchone()[0]

    @staticmethod
    def _drafts(con: sqlite3.Connection, day: str, own_key: bool, token_id: str | None = None) -> int:
        kinds = "('own_key', 'own_actual')" if own_key else "('reservation', 'actual')"
        who, args = ("AND token_id = ?", (day, token_id)) if token_id is not None else ("", (day,))
        return con.execute(f"SELECT COUNT(DISTINCT draft_id) FROM spend WHERE day = ? AND kind IN {kinds} {who}",
                           args).fetchone()[0]

    def check_and_reserve(self, token_id: str, draft_id: str, paid_by: str) -> None:
        """Atomically apply the limits and, if the draft may run, record it (with the estimate when the server
        pays). Raises LimitReached; the refusal is recorded for the summary and does not count as a draft.

        A server-paid draft is refused when settled actual spend + open drafts at max(reservation, running cost)
        + one estimate would pass a cap. A draft admitted at the edge can still cost BUDGET_FACTOR estimates, as
        can each job already running, so the most a day can record is
        DAY_CAP - EST + BUDGET_FACTOR * EST + (MAX_JOBS_TOTAL - 1) * (BUDGET_FACTOR - 1) * EST
        plus, per stopped job, the one call that crossed its limit (defaults: 1.00 - 0.12 + 0.24 + 0.12 = $1.24)."""
        if paid_by not in PAID_BY:
            raise ValueError(f"paid_by must be one of {PAID_BY}")
        day, month, next_day, next_month = self._periods()
        est, refusal = self.cfg.EST_COST_PER_DRAFT_USD, None
        with transaction(self.db_path) as con:
            if con.execute("SELECT 1 FROM spend WHERE draft_id = ? AND kind NOT LIKE 'refused_%'", (draft_id,)).fetchone():
                raise ValueError("this draft already has a reservation")
            if paid_by == "own_key":
                if self._drafts(con, day, True, token_id) >= self.cfg.OWN_KEY_DRAFTS_PER_DAY:
                    refusal = LimitReached("own_key_daily", next_day)
            elif self._drafts(con, day, False, token_id) >= self.cfg.DRAFTS_PER_DAY:
                refusal = LimitReached("user_daily", next_day)
            elif self._spent(con, "day", day) + est > self.cfg.DAY_CAP_USD + EPS:
                refusal = LimitReached("day_cap", next_day)
            elif self._spent(con, "month", month) + est > self.cfg.MONTH_CAP_USD + EPS:
                refusal = LimitReached("month_cap", next_month)
            kind, usd = (f"refused_{refusal.kind}", 0.0) if refusal else \
                ("own_key", 0.0) if paid_by == "own_key" else ("reservation", est)
            con.execute("INSERT INTO spend (day, month, token_id, draft_id, usd, kind) VALUES (?, ?, ?, ?, ?, ?)",
                        (day, month, token_id, draft_id, usd, kind))
        if refusal:
            raise refusal

    def add_cost(self, draft_id: str, usd: float) -> None:
        """Record money already spent by a running job. Raises BudgetExceeded (after recording) once the draft
        has cost more than ``cfg.BUDGET_FACTOR`` times the estimate, or if the draft was never reserved."""
        usd = float(usd)
        if not 0 <= usd < float("inf"):
            raise ValueError("usd must be a finite amount >= 0")
        with transaction(self.db_path) as con:
            open_row = con.execute("SELECT day, month, token_id, kind FROM spend WHERE draft_id = ? "
                                   "AND kind IN ('reservation', 'own_key')", (draft_id,)).fetchone()
            if open_row is None:
                raise BudgetExceeded("no open reservation for this draft")
            day, month, token_id, kind = open_row
            running = "cost" if kind == "reservation" else "own_cost"
            if con.execute("UPDATE spend SET usd = usd + ? WHERE draft_id = ? AND kind = ?",
                           (usd, draft_id, running)).rowcount == 0:
                con.execute("INSERT INTO spend (day, month, token_id, draft_id, usd, kind) VALUES (?, ?, ?, ?, ?, ?)",
                            (day, month, token_id, draft_id, usd, running))
            total = con.execute("SELECT usd FROM spend WHERE draft_id = ? AND kind = ?", (draft_id, running)).fetchone()[0]
        if total > self.cfg.BUDGET_FACTOR * self.cfg.EST_COST_PER_DRAFT_USD + EPS:
            raise BudgetExceeded(f"draft cost ${total:.2f} passed {self.cfg.BUDGET_FACTOR:g}x the estimate")

    def recorded_cost(self, draft_id: str) -> float:
        """What the running job of this draft has recorded as spent so far (all its deliveries together)."""
        con = connect(self.db_path)
        try:
            return float(con.execute("SELECT COALESCE(SUM(usd), 0) FROM spend WHERE draft_id = ? "
                                     "AND kind IN ('cost', 'own_cost')", (draft_id,)).fetchone()[0])
        finally:
            con.close()

    def settle(self, draft_id: str, actual_usd: float) -> None:
        """Replace the reservation (and running cost) with the actual cost. The draft still counts as one of the
        day's drafts, whether the job worked or failed. No-op for a draft without a reservation."""
        actual_usd = float(actual_usd)
        if not 0 <= actual_usd < float("inf"):
            raise ValueError("actual_usd must be a finite amount >= 0")
        with transaction(self.db_path) as con:
            row = con.execute("SELECT day, month, token_id, kind FROM spend WHERE draft_id = ? "
                              "AND kind IN ('reservation', 'actual', 'own_key', 'own_actual')", (draft_id,)).fetchone()
            if row is None:
                return
            day, month, token_id, kind = row
            if kind in ("own_key", "own_actual"):
                con.execute("DELETE FROM spend WHERE draft_id = ? AND kind = 'own_cost'", (draft_id,))
                con.execute("UPDATE spend SET usd = ?, kind = 'own_actual' WHERE draft_id = ? AND kind = ?",
                            (actual_usd, draft_id, kind))
                return
            con.execute("DELETE FROM spend WHERE draft_id = ? AND kind IN ('reservation', 'cost', 'actual')", (draft_id,))
            con.execute("INSERT INTO spend (day, month, token_id, draft_id, usd, kind) VALUES (?, ?, ?, ?, ?, 'actual')",
                        (day, month, token_id, draft_id, actual_usd))

    def cancel(self, draft_id: str) -> bool:
        """Drop an unsettled reservation for a job that never started (no paid call made), so it does not count
        as a draft. False if there is nothing to cancel or money was already recorded."""
        with transaction(self.db_path) as con:
            if con.execute("SELECT 1 FROM spend WHERE draft_id = ? AND kind IN ('cost', 'own_cost', 'actual')",
                           (draft_id,)).fetchone():
                return False
            return con.execute("DELETE FROM spend WHERE draft_id = ? AND kind IN ('reservation', 'own_key')",
                               (draft_id,)).rowcount > 0

    def usage(self, token_id: str) -> dict:
        """What the tools show a user: server-paid and own-key drafts used and left today, and the reset time."""
        day, _, next_day, _ = self._periods()
        con = connect(self.db_path)
        try:
            used, own = self._drafts(con, day, False, token_id), self._drafts(con, day, True, token_id)
        finally:
            con.close()
        return {"drafts_today": used, "drafts_left_today": max(0, self.cfg.DRAFTS_PER_DAY - used),
                "drafts_per_day": self.cfg.DRAFTS_PER_DAY, "own_key_drafts_today": own,
                "own_key_drafts_left_today": max(0, self.cfg.OWN_KEY_DRAFTS_PER_DAY - own), "resets_at_utc": next_day}

    def summary(self, day: str | None = None) -> dict:
        """Drafts, spend and refusals of one UTC day (``YYYY-MM-DD``, default today), plus its month's spend."""
        day = self._periods()[0] if day is None else datetime.strptime(day, "%Y-%m-%d").strftime("%Y-%m-%d")
        month = day[:7]
        con = connect(self.db_path)
        try:
            by_kind = dict(con.execute("SELECT kind, COALESCE(SUM(usd), 0) FROM spend WHERE day = ? GROUP BY kind", (day,)))
            refusals = dict(con.execute("SELECT SUBSTR(kind, 9), COUNT(*) FROM spend WHERE day = ? "
                                        "AND kind LIKE 'refused_%' GROUP BY kind", (day,)))
            return {"day": day, "drafts": self._drafts(con, day, False), "own_key_drafts": self._drafts(con, day, True),
                    "spend_usd": round(self._spent(con, "day", day), 6),
                    "settled_usd": round(by_kind.get("actual", 0.0), 6),
                    "own_key_usd": round(by_kind.get("own_actual", 0.0), 6),
                    "refusals": {k: refusals.get(k, 0) for k in REFUSAL_KINDS},
                    "month": month, "month_spend_usd": round(self._spent(con, "month", month), 6),
                    "day_cap_usd": self.cfg.DAY_CAP_USD, "month_cap_usd": self.cfg.MONTH_CAP_USD}
        finally:
            con.close()
