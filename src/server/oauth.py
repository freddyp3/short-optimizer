"""Sign-in for real users: the server is its own OAuth 2.1 authorization server and Google says who the user is.

Claude (or any MCP client) registers itself (dynamic client registration), sends the user to ``/authorize`` and
gets an access token from ``/token``; those endpoints come from the MCP SDK and call ``Provider`` below. In
between, the user signs in with Google and lands on ``/oauth/google/callback``:

    /authorize -> Google -> /oauth/google/callback -> (approved?) -> consent page -> /oauth/consent -> client

A first sign-in creates the user (that is the signup; ``users.created_at``). Nobody can use the tools before the
owner approves them (``python -m src.server.users approve EMAIL``), except the addresses in
``AUTO_APPROVE_EMAILS``. The consent page names the connecting client and needs a click, so a link built by
somebody else cannot quietly connect their client to a signed-in user's account.

Stored: the user's Google subject id, email and name; clients; and only SHA-256 hashes of authorization codes,
access tokens and refresh tokens. A user's id (``u_`` + 8 hex characters) is what drafts, quotas and log lines
carry, in the place of the token id of an invite token.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from mcp.server.auth.provider import (AccessToken, AuthorizationCode, AuthorizationParams, RefreshToken,
                                      RegistrationError, TokenError)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from src.server.store import connect, init_schema, transaction

ACCESS_TTL_S = 3600
REFRESH_TTL_S = 30 * 86400
CODE_TTL_S = 300
PENDING_TTL_S = 900
MAX_CLIENTS = 2000
MAX_PENDING = 2000
SEEN_EVERY_S = 60.0
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
CALLBACK_PATH, CONSENT_PATH = "/oauth/google/callback", "/oauth/consent"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    google_sub TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    approved_at REAL,
    blocked_at REAL,
    last_seen_at REAL
);
CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id TEXT PRIMARY KEY,
    info_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS oauth_pending (
    state_sha TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    params_json TEXT NOT NULL,
    expires_at REAL NOT NULL,
    user_id TEXT,
    ticket_sha TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS oauth_codes (
    code_sha TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    params_json TEXT NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS oauth_tokens (
    token_sha TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    grant_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    scopes TEXT NOT NULL,
    resource TEXT,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS oauth_tokens_grant ON oauth_tokens (grant_id);
CREATE INDEX IF NOT EXISTS oauth_tokens_user ON oauth_tokens (user_id);
"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


class IdentityError(Exception):
    """Google did not confirm who the user is."""


@dataclass(frozen=True)
class Identity:
    sub: str
    email: str
    name: str = ""


@dataclass(frozen=True)
class User:
    id: str
    google_sub: str
    email: str
    name: str
    created_at: float
    approved_at: float | None
    blocked_at: float | None
    last_seen_at: float | None

    @property
    def status(self) -> str:
        return "blocked" if self.blocked_at is not None else "approved" if self.approved_at is not None else "pending"


class Google:
    """The Google leg: where to send the user, and who came back. Uses the authorization-code flow; the identity
    is read from Google's userinfo endpoint with the token Google just issued over TLS, so no signature of ours
    is involved. Only a verified email address counts."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, post: Callable | None = None,
                 get: Callable | None = None):
        self.client_id, self._secret, self.redirect_uri, self._post, self._get = client_id, client_secret, redirect_uri, post, get

    def url(self, state: str) -> str:
        return GOOGLE_AUTH_URL + "?" + urlencode({
            "client_id": self.client_id, "redirect_uri": self.redirect_uri, "response_type": "code",
            "scope": "openid email profile", "state": state, "prompt": "select_account"})

    def identify(self, code: str) -> Identity:
        import httpx

        post, get = self._post or httpx.post, self._get or httpx.get
        try:
            r = post(GOOGLE_TOKEN_URL, data={"code": code, "client_id": self.client_id, "client_secret": self._secret,
                                             "redirect_uri": self.redirect_uri, "grant_type": "authorization_code"}, timeout=15)
            token = r.json().get("access_token") if r.status_code == 200 else None
            if not token:
                raise IdentityError("no token")
            r = get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {token}"}, timeout=15)
            info = r.json() if r.status_code == 200 else {}
        except IdentityError:
            raise
        except Exception as exc:
            raise IdentityError(type(exc).__name__) from None
        sub, email = str(info.get("sub") or ""), str(info.get("email") or "").strip().lower()
        if not sub or not email or info.get("email_verified") not in (True, "true"):
            raise IdentityError("no verified email")
        return Identity(sub[:255], email[:254], " ".join(str(info.get("name") or "").split())[:120])


_USER_COLUMNS = "id, google_sub, email, name, created_at, approved_at, blocked_at, last_seen_at"


class Users:
    def __init__(self, db_path: Path | str, clock: Callable[[], float] = time.time, auto_approve: tuple[str, ...] = ()):
        self.db_path, self.clock = Path(db_path), clock
        self.auto_approve = {e.strip().lower() for e in auto_approve if e.strip()}
        self._seen: dict[str, float] = {}
        init_schema(self.db_path, SCHEMA)

    def sign_in(self, who: Identity) -> tuple[User, bool]:
        """The user of this Google account and whether this sign-in created them (a signup). The email and name
        follow the Google account; an address in ``auto_approve`` is approved on the spot."""
        now = self.clock()
        with transaction(self.db_path) as con:
            row = con.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE google_sub = ?", (who.sub,)).fetchone()
            new = row is None
            if new:
                for _ in range(5):
                    user_id = "u_" + secrets.token_hex(4)
                    if not con.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
                        break
                con.execute("INSERT INTO users (id, google_sub, email, name, created_at) VALUES (?, ?, ?, ?, ?)",
                            (user_id, who.sub, who.email, who.name, now))
            else:
                con.execute("UPDATE users SET email = ?, name = ? WHERE google_sub = ?", (who.email, who.name, who.sub))
            if who.email in self.auto_approve:
                con.execute("UPDATE users SET approved_at = ? WHERE google_sub = ? AND approved_at IS NULL", (now, who.sub))
            return User(*con.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE google_sub = ?", (who.sub,)).fetchone()), new

    def _query(self, where: str = "", args: tuple = ()) -> list[User]:
        con = connect(self.db_path)
        try:
            return [User(*r) for r in con.execute(f"SELECT {_USER_COLUMNS} FROM users {where} ORDER BY created_at, id", args)]
        finally:
            con.close()

    def get(self, user_id: str) -> User | None:
        found = self._query("WHERE id = ?", (user_id,))
        return found[0] if found else None

    def find(self, who: str) -> list[User]:
        """Users with this id or this email address (an address can belong to more than one Google account)."""
        return self._query("WHERE id = ? OR email = ?", (who.strip(), who.strip().lower()))

    def all(self) -> list[User]:
        return self._query()

    def _set(self, user_id: str, sql: str) -> bool:
        with transaction(self.db_path) as con:
            return con.execute(f"UPDATE users SET {sql} WHERE id = ?", (self.clock(), user_id)).rowcount == 1

    def approve(self, user_id: str) -> bool:
        """Let the user in (also lifts a block). False for an unknown id."""
        return self._set(user_id, "blocked_at = NULL, approved_at = COALESCE(approved_at, ?)")

    def block(self, user_id: str) -> bool:
        """Shut the user out and delete their tokens. False for an unknown id."""
        done = self._set(user_id, "blocked_at = ?")
        with transaction(self.db_path) as con:
            con.execute("DELETE FROM oauth_tokens WHERE user_id = ?", (user_id,))
            con.execute("DELETE FROM oauth_codes WHERE user_id = ?", (user_id,))
        return done

    def seen(self, user_id: str) -> None:
        """Note that the user made a request (written at most once a minute per user)."""
        now = self.clock()
        if now - self._seen.get(user_id, 0.0) < SEEN_EVERY_S:
            return
        if len(self._seen) > 10_000:
            self._seen.clear()
        self._seen[user_id] = now
        with transaction(self.db_path) as con:
            con.execute("UPDATE users SET last_seen_at = ? WHERE id = ?", (now, user_id))


class Provider:
    """Storage behind the SDK's OAuth endpoints (``OAuthAuthorizationServerProvider``). Synchronous SQLite inside
    async methods: every call is one short transaction on a local file."""

    def __init__(self, db_path: Path | str, users: Users, google: Google, clock: Callable[[], float] = time.time):
        self.db_path, self.users, self.google, self.clock = Path(db_path), users, google, clock
        init_schema(self.db_path, SCHEMA)

    def _wall(self, expires_at: float) -> float:
        """An expiry of ours on the wall clock. Expiry is decided here, on ``self.clock``; the SDK's handlers check
        it once more against ``time.time()``, so they are given the same remaining time on that clock."""
        return time.time() + (expires_at - self.clock())

    # ---- clients

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        con = connect(self.db_path)
        try:
            row = con.execute("SELECT info_json FROM oauth_clients WHERE client_id = ?", (client_id,)).fetchone()
        finally:
            con.close()
        return OAuthClientInformationFull.model_validate_json(row[0]) if row else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        with transaction(self.db_path) as con:
            if con.execute("SELECT COUNT(*) FROM oauth_clients").fetchone()[0] >= MAX_CLIENTS:
                con.execute("DELETE FROM oauth_clients WHERE client_id NOT IN (SELECT client_id FROM oauth_tokens) "
                            "AND created_at < ?", (self.clock() - 86400,))
                if con.execute("SELECT COUNT(*) FROM oauth_clients").fetchone()[0] >= MAX_CLIENTS:
                    raise RegistrationError(error="invalid_client_metadata", error_description="too many clients; try later")
            con.execute("INSERT INTO oauth_clients (client_id, info_json, created_at) VALUES (?, ?, ?)",
                        (client_info.client_id, client_info.model_dump_json(), self.clock()))

    # ---- authorize: park the request, send the user to Google

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        state, now = secrets.token_urlsafe(32), self.clock()
        with transaction(self.db_path) as con:
            con.execute("DELETE FROM oauth_pending WHERE expires_at <= ?", (now,))
            if con.execute("SELECT COUNT(*) FROM oauth_pending").fetchone()[0] >= MAX_PENDING:
                con.execute("DELETE FROM oauth_pending WHERE state_sha IN (SELECT state_sha FROM oauth_pending "
                            "ORDER BY expires_at LIMIT ?)", (MAX_PENDING // 10,))
            con.execute("INSERT INTO oauth_pending (state_sha, client_id, params_json, expires_at) VALUES (?, ?, ?, ?)",
                        (sha(state), client.client_id, params.model_dump_json(), now + PENDING_TTL_S))
        return self.google.url(state)

    def take_pending(self, state: str) -> tuple[str, AuthorizationParams] | None:
        """The parked request the user came back from Google for: ``(client_id, params)``. One use."""
        with transaction(self.db_path) as con:
            row = con.execute("SELECT client_id, params_json FROM oauth_pending WHERE state_sha = ? AND expires_at > ? "
                              "AND ticket_sha IS NULL", (sha(state), self.clock())).fetchone()
            con.execute("DELETE FROM oauth_pending WHERE state_sha = ? AND ticket_sha IS NULL", (sha(state),))
        return (row[0], AuthorizationParams.model_validate_json(row[1])) if row else None

    def offer_consent(self, client_id: str, params: AuthorizationParams, user_id: str) -> str:
        """Park the request again, now bound to the signed-in user, behind a one-time ticket for the consent page."""
        ticket = secrets.token_urlsafe(32)
        with transaction(self.db_path) as con:
            con.execute("INSERT INTO oauth_pending (state_sha, client_id, params_json, expires_at, user_id, ticket_sha) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (sha("ticket:" + ticket), client_id, params.model_dump_json(),
                                                      self.clock() + PENDING_TTL_S, user_id, sha(ticket)))
        return ticket

    def take_consent(self, ticket: str) -> tuple[str, AuthorizationParams, str] | None:
        """``(client_id, params, user_id)`` of a consent ticket. One use."""
        with transaction(self.db_path) as con:
            row = con.execute("SELECT client_id, params_json, user_id FROM oauth_pending WHERE ticket_sha = ? "
                              "AND expires_at > ?", (sha(ticket), self.clock())).fetchone()
            con.execute("DELETE FROM oauth_pending WHERE ticket_sha = ?", (sha(ticket),))
        return (row[0], AuthorizationParams.model_validate_json(row[1]), row[2]) if row else None

    def issue_code(self, client_id: str, params: AuthorizationParams, user_id: str) -> str:
        code, now = secrets.token_urlsafe(32), self.clock()
        with transaction(self.db_path) as con:
            con.execute("DELETE FROM oauth_codes WHERE expires_at <= ?", (now,))
            con.execute("INSERT INTO oauth_codes (code_sha, client_id, user_id, params_json, expires_at) VALUES (?, ?, ?, ?, ?)",
                        (sha(code), client_id, user_id, params.model_dump_json(), now + CODE_TTL_S))
        return code

    # ---- codes and tokens

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str) -> AuthorizationCode | None:
        con = connect(self.db_path)
        try:
            row = con.execute("SELECT user_id, params_json, expires_at FROM oauth_codes WHERE code_sha = ? AND client_id = ? "
                              "AND expires_at > ?", (sha(authorization_code), client.client_id, self.clock())).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        p = AuthorizationParams.model_validate_json(row[1])
        return AuthorizationCode(code=authorization_code, scopes=p.scopes or [], expires_at=self._wall(row[2]), client_id=client.client_id,
                                 code_challenge=p.code_challenge, redirect_uri=p.redirect_uri,
                                 redirect_uri_provided_explicitly=p.redirect_uri_provided_explicitly, resource=p.resource,
                                 subject=row[0])

    def _issue(self, con, client_id: str, user_id: str, scopes: list[str], resource: str | None) -> OAuthToken:
        access, refresh, grant, now = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_hex(8), self.clock()
        con.execute("DELETE FROM oauth_tokens WHERE expires_at <= ?", (now,))
        con.executemany("INSERT INTO oauth_tokens (token_sha, kind, grant_id, client_id, user_id, scopes, resource, expires_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [(sha(access), "access", grant, client_id, user_id, " ".join(scopes), resource, now + ACCESS_TTL_S),
                         (sha(refresh), "refresh", grant, client_id, user_id, " ".join(scopes), resource, now + REFRESH_TTL_S)])
        return OAuthToken(access_token=access, expires_in=ACCESS_TTL_S, refresh_token=refresh, scope=" ".join(scopes) or None)

    def _allowed(self, con, user_id: str) -> bool:
        row = con.execute("SELECT approved_at, blocked_at FROM users WHERE id = ?", (user_id,)).fetchone()
        return row is not None and row[0] is not None and row[1] is None

    async def exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode) -> OAuthToken:
        with transaction(self.db_path) as con:
            used = con.execute("DELETE FROM oauth_codes WHERE code_sha = ? AND client_id = ?",
                               (sha(authorization_code.code), client.client_id)).rowcount
            if used != 1 or not authorization_code.subject or not self._allowed(con, authorization_code.subject):
                raise TokenError(error="invalid_grant", error_description="authorization code is not valid")
            return self._issue(con, client.client_id, authorization_code.subject, authorization_code.scopes,
                               authorization_code.resource)

    def _token(self, token: str, kind: str, client_id: str | None = None) -> tuple | None:
        con = connect(self.db_path)
        try:
            row = con.execute("SELECT client_id, user_id, scopes, resource, expires_at FROM oauth_tokens WHERE token_sha = ? "
                              "AND kind = ? AND expires_at > ?", (sha(token), kind, self.clock())).fetchone()
            if row is None or (client_id is not None and row[0] != client_id) or not self._allowed(con, row[1]):
                return None
            return row
        finally:
            con.close()

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        row = self._token(refresh_token, "refresh", client.client_id)
        return row and RefreshToken(token=refresh_token, client_id=row[0], scopes=row[2].split(), expires_at=int(self._wall(row[4])),
                                    resource=row[3], subject=row[1])

    async def exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken,
                                     scopes: list[str]) -> OAuthToken:
        with transaction(self.db_path) as con:
            row = con.execute("SELECT grant_id, user_id FROM oauth_tokens WHERE token_sha = ? AND kind = 'refresh' "
                              "AND client_id = ?", (sha(refresh_token.token), client.client_id)).fetchone()
            if row is None or not self._allowed(con, row[1]):
                raise TokenError(error="invalid_grant", error_description="refresh token is not valid")
            con.execute("DELETE FROM oauth_tokens WHERE grant_id = ?", (row[0],))
            return self._issue(con, client.client_id, row[1], scopes or refresh_token.scopes, refresh_token.resource)

    async def load_access_token(self, token: str) -> AccessToken | None:
        row = self._token(token, "access")
        return row and AccessToken(token=token, client_id=row[0], scopes=row[2].split(), expires_at=int(self._wall(row[4])),
                                   resource=row[3], subject=row[1])

    def user_of(self, access_token: str) -> str | None:
        """The id of the approved user this access token belongs to, or None."""
        row = self._token(access_token, "access") if isinstance(access_token, str) and access_token else None
        return row[1] if row else None

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        with transaction(self.db_path) as con:
            row = con.execute("SELECT grant_id FROM oauth_tokens WHERE token_sha = ?", (sha(token.token),)).fetchone()
            if row:
                con.execute("DELETE FROM oauth_tokens WHERE grant_id = ?", (row[0],))

    async def exchange_identity_assertion(self, client, params) -> OAuthToken:
        raise TokenError(error="unsupported_grant_type", error_description="not supported")
