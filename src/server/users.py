"""Who signed up and what they used: the owner's commands for the users of the Claude MCP server.

    ./venv/bin/python -m src.server.users list              # everyone who signed in with Google, newest last
    ./venv/bin/python -m src.server.users list --pending    # only those waiting for approval
    ./venv/bin/python -m src.server.users approve EMAIL     # let somebody in (also lifts a block)
    ./venv/bin/python -m src.server.users block EMAIL       # shut somebody out and delete their tokens
    ./venv/bin/python -m src.server.users stats             # signups, active users, drafts and spend

No network code: run it in the server's shell. ``EMAIL`` can also be a user id (``u_...``). Invite tokens
(``src.server.auth``) appear in the usage table under their label.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from src.server import config
from src.server.auth import Tokens
from src.server.oauth import Users
from src.server.quota import SCHEMA as SPEND_SCHEMA
from src.server.store import connect, init_schema

DAY_S = 86400.0


def usage(db_path: Path | str) -> dict[str, dict]:
    """Per user id or token id: drafts analysed (server-paid and own-key), what the server paid, first and last day."""
    init_schema(db_path, SPEND_SCHEMA)
    con = connect(db_path)
    try:
        rows = con.execute(
            "SELECT token_id, SUM(kind IN ('reservation', 'actual')), SUM(kind IN ('own_key', 'own_actual')), "
            "COALESCE(SUM(CASE WHEN kind IN ('reservation', 'cost', 'actual') THEN usd END), 0), MIN(day), MAX(day) "
            "FROM spend WHERE kind NOT LIKE 'refused_%' GROUP BY token_id").fetchall()
    finally:
        con.close()
    return {r[0]: {"drafts": int(r[1] or 0), "own_key_drafts": int(r[2] or 0), "usd": float(r[3]), "first_day": r[4],
                   "last_day": r[5]} for r in rows}


def stats(db_path: Path | str, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    users, used = Users(db_path).all(), usage(db_path)
    return {"signups": len(users), "signups_last_7_days": sum(u.created_at > now - 7 * DAY_S for u in users),
            "pending": sum(u.status == "pending" for u in users), "approved": sum(u.status == "approved" for u in users),
            "blocked": sum(u.status == "blocked" for u in users),
            "active_last_7_days": sum((u.last_seen_at or 0) > now - 7 * DAY_S for u in users),
            "users_with_a_draft": sum(1 for u in users if used.get(u.id, {}).get("drafts", 0) + used.get(u.id, {}).get("own_key_drafts", 0)),
            "drafts": sum(v["drafts"] for v in used.values()), "own_key_drafts": sum(v["own_key_drafts"] for v in used.values()),
            "server_paid_usd": round(sum(v["usd"] for v in used.values()), 4)}


def _day(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M") if ts is not None else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.server.users", description="Signups, approval and usage.")
    parser.add_argument("--db", type=Path, default=None, help="database file (default: SO_DB_PATH / config)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="users who signed in with Google").add_argument("--pending", action="store_true")
    sub.add_parser("approve", help="let a user in").add_argument("who", help="email address or user id")
    sub.add_parser("block", help="shut a user out").add_argument("who", help="email address or user id")
    sub.add_parser("stats", help="signups, active users, drafts and spend")
    args = parser.parse_args(argv)
    db = Path(config.DB_PATH if args.db is None else args.db)
    users = Users(db)
    if args.command == "list":
        used = usage(db)
        shown = [u for u in users.all() if not args.pending or u.status == "pending"]
        print(f"{'id':<12}{'status':<10}{'signed up (UTC)':<18}{'last seen (UTC)':<18}{'drafts':>7}{'own key':>9}{'paid $':>8}  email")
        if not shown:
            print("(nobody)")
        for u in shown:
            v = used.get(u.id, {})
            print(f"{u.id:<12}{u.status:<10}{_day(u.created_at):<18}{_day(u.last_seen_at):<18}{v.get('drafts', 0):>7}"
                  f"{v.get('own_key_drafts', 0):>9}{v.get('usd', 0.0):>8.2f}  {u.email}")
        labels = {t.id: t.label for t in Tokens(db).list_tokens()}
        invited = [(labels[i], v) for i, v in used.items() if i in labels]
        if invited and not args.pending:
            print("\ninvite tokens:")
            for label, v in invited:
                print(f"  {label:<20}{v['drafts']:>7}{v['own_key_drafts']:>9}{v['usd']:>8.2f}  {v['first_day']} to {v['last_day']}")
        return 0
    if args.command == "stats":
        for name, value in stats(db).items():
            print(f"{name.replace('_', ' '):<22}{value}")
        return 0
    found = users.find(args.who)
    if not found:
        print("no user with that email address or id")
        return 1
    for u in found:
        (users.approve if args.command == "approve" else users.block)(u.id)
        print(f"{'approved' if args.command == 'approve' else 'blocked'} {u.id} {u.email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
