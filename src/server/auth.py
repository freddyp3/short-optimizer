"""Access tokens of the Claude MCP server (plan/10a section 2, plan/10 section 12.4 "Access").

Invite-only: each user gets a random 32-byte token, shown once. Only its SHA-256 is stored, and verification
compares hashes with ``hmac.compare_digest`` against every stored token, so timing says nothing about how close
a guess was. The short ``token_id`` is public and is what logs carry. There is no network code here; tokens are
managed from the server's shell:

    ./venv/bin/python -m src.server.auth create --label NAME   # prints the token once
    ./venv/bin/python -m src.server.auth list
    ./venv/bin/python -m src.server.auth revoke ID
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.server import config
from src.server.store import connect, init_schema, transaction

TOKEN_BYTES = 32
MAX_TOKEN_CHARS = 128
MAX_LABEL_CHARS = 80
MAX_TRACKED_ADDRESSES = 10_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS tokens (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    sha256_hex TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    revoked_at REAL
);
"""


@dataclass(frozen=True)
class TokenInfo:
    id: str
    label: str
    created_at: float
    revoked_at: float | None = None

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None


def hash_token(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8", "surrogatepass")).hexdigest()


class Tokens:
    def __init__(self, db_path: Path | str | None = None, clock: Callable[[], float] = time.time):
        self.db_path, self.clock = Path(config.DB_PATH if db_path is None else db_path), clock
        init_schema(self.db_path, SCHEMA)

    def create_token(self, label: str) -> tuple[str, str]:
        """New token as ``(token_id, plain)``. The plain token exists only in the return value."""
        label = " ".join(str(label).split())
        if not label or len(label) > MAX_LABEL_CHARS:
            raise ValueError(f"label must be 1-{MAX_LABEL_CHARS} characters")
        plain = secrets.token_urlsafe(TOKEN_BYTES)
        for _ in range(5):
            token_id = secrets.token_hex(4)
            try:
                with transaction(self.db_path) as con:
                    con.execute("INSERT INTO tokens (id, label, sha256_hex, created_at) VALUES (?, ?, ?, ?)",
                                (token_id, label, hash_token(plain), self.clock()))
                return token_id, plain
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("could not allocate a token id")

    def verify(self, plain: str) -> TokenInfo | None:
        """The token's record, or None for an unknown, revoked or malformed token."""
        if not isinstance(plain, str) or not plain or len(plain) > MAX_TOKEN_CHARS:
            return None
        digest = hash_token(plain).encode()
        found = None
        for row in self._rows():
            if hmac.compare_digest(digest, row[2].encode()):
                found = row
        if found is None or found[4] is not None:
            return None
        return TokenInfo(found[0], found[1], found[3], found[4])

    def revoke(self, token_id: str) -> bool:
        """True if an active token was revoked; False if the id is unknown or already revoked."""
        with transaction(self.db_path) as con:
            return con.execute("UPDATE tokens SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                               (self.clock(), token_id)).rowcount == 1

    def list_tokens(self) -> list[TokenInfo]:
        """Every token, oldest first, without hashes."""
        return [TokenInfo(r[0], r[1], r[3], r[4]) for r in self._rows()]

    def _rows(self) -> list[tuple]:
        con = connect(self.db_path)
        try:
            return con.execute("SELECT id, label, sha256_hex, created_at, revoked_at FROM tokens "
                               "ORDER BY created_at, id").fetchall()
        finally:
            con.close()


def create_token(label: str, db_path: Path | str | None = None) -> tuple[str, str]:
    return Tokens(db_path).create_token(label)


def verify(plain: str, db_path: Path | str | None = None) -> TokenInfo | None:
    return Tokens(db_path).verify(plain)


def revoke(token_id: str, db_path: Path | str | None = None) -> bool:
    return Tokens(db_path).revoke(token_id)


def list_tokens(db_path: Path | str | None = None) -> list[TokenInfo]:
    return Tokens(db_path).list_tokens()


class FailureLimiter:
    """In-memory count of failed verifications per client address over a sliding window.

    ``record_failure(addr)`` after every rejected token; ``blocked(addr)`` before verifying. An address is blocked
    while it has ``max_failures`` or more failures inside the last ``window_s`` seconds. Memory is bounded: at
    most ``max_addresses`` are tracked, the least recently failing ones are dropped first.
    """

    def __init__(self, max_failures: int = config.BAD_TOKEN_MAX, window_s: float = config.BAD_TOKEN_WINDOW_S,
                 clock: Callable[[], float] = time.time, max_addresses: int = MAX_TRACKED_ADDRESSES):
        self.max_failures, self.window_s, self.clock, self.max_addresses = max_failures, window_s, clock, max_addresses
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, addr: str, now: float) -> deque[float] | None:
        times = self._failures.get(addr)
        if times is None:
            return None
        while times and times[0] <= now - self.window_s:
            times.popleft()
        if not times:
            del self._failures[addr]
            return None
        return times

    def record_failure(self, addr: str) -> None:
        now = self.clock()
        with self._lock:
            times = self._recent(addr, now)
            if times is None:
                if len(self._failures) >= self.max_addresses:
                    for other in list(self._failures):
                        self._recent(other, now)
                    while len(self._failures) >= self.max_addresses:
                        del self._failures[min(self._failures, key=lambda a: self._failures[a][-1])]
                times = self._failures[addr] = deque(maxlen=max(1, self.max_failures))
            times.append(now)

    def blocked(self, addr: str) -> bool:
        with self._lock:
            times = self._recent(addr, self.clock())
            return times is not None and len(times) >= self.max_failures


def _when(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M") if ts is not None else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.server.auth", description="Manage access tokens.")
    parser.add_argument("--db", type=Path, default=None, help="database file (default: SO_DB_PATH / config)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create", help="create a token and print it once").add_argument("--label", required=True)
    sub.add_parser("list", help="list tokens (ids and labels only)")
    sub.add_parser("revoke", help="revoke a token by id").add_argument("id")
    args = parser.parse_args(argv)
    tokens = Tokens(args.db)
    if args.command == "create":
        token_id, plain = tokens.create_token(args.label)
        print(f"token id: {token_id}")
        print(f"token:    {plain}")
        print("This token is shown once and cannot be recovered. Send it to the user over a private channel.")
    elif args.command == "list":
        print(f"{'id':<10}{'created (UTC)':<18}{'revoked (UTC)':<18}label")
        for t in tokens.list_tokens():
            print(f"{t.id:<10}{_when(t.created_at):<18}{_when(t.revoked_at):<18}{t.label}")
    elif not tokens.revoke(args.id):
        print("no active token with that id")
        return 1
    else:
        print(f"revoked {args.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
