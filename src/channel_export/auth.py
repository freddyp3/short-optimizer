"""OAuth 2.0 installed-app ("Desktop app") authorization with a cached, auto-refreshed token."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

log = logging.getLogger(__name__)

YOUTUBE_READONLY = "https://www.googleapis.com/auth/youtube.readonly"
ANALYTICS_READONLY = "https://www.googleapis.com/auth/yt-analytics.readonly"
ANALYTICS_MONETARY_READONLY = "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"

# youtube.readonly: Data API reads, and required by Analytics reports.query since 2018.
# yt-analytics.readonly: Analytics reports.query and the Reporting API.
BASE_SCOPES: tuple[str, ...] = (YOUTUBE_READONLY, ANALYTICS_READONLY)

DEFAULT_CLIENT_SECRET = Path("secrets/client_secret.json")
DEFAULT_TOKEN = Path("secrets/token.json")
CLIENT_SECRET_ENV = "YT_CLIENT_SECRET"


class AuthSetupError(RuntimeError):
    """Raised when the OAuth client secret is missing or unusable."""


def scopes_for(monetary: bool) -> list[str]:
    """Scopes to request; the monetary scope is only added when asked for."""
    scopes = list(BASE_SCOPES)
    if monetary:
        scopes.append(ANALYTICS_MONETARY_READONLY)
    return scopes


def client_type(path: Path) -> str | None:
    """OAuth client type in a client-secret JSON: 'installed' (Desktop app), 'web', or None."""
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):
        for key in ("installed", "web"):
            if key in data:
                return key
    return None


def _discover_desktop_client(folder: Path) -> Path | None:
    """Find the single Desktop-app client JSON saved under Google's default download name."""
    candidates = sorted(folder.glob("client_secret*.json"))
    desktop = [p for p in candidates if client_type(p) == "installed"]
    if len(desktop) == 1:
        return desktop[0]
    if len(desktop) > 1:
        names = ", ".join(p.name for p in desktop)
        raise AuthSetupError(
            f"Several Desktop-app OAuth clients in {folder}: {names}. Rename the one to use to "
            f"client_secret.json, set ${CLIENT_SECRET_ENV}, or pass --client-secret."
        )
    if candidates:
        raise AuthSetupError(
            f"{folder} has client secret files but none is a 'Desktop app' client (the file must "
            "contain an 'installed' section). Create a Desktop app client; see src/channel_export/README.md."
        )
    return None


def client_secret_path(explicit: Path | None = None, default: Path = DEFAULT_CLIENT_SECRET) -> Path:
    """Client secret location: explicit arg > $YT_CLIENT_SECRET > secrets/client_secret.json >
    the only Desktop-app ``client_secret*.json`` in the same folder."""
    if explicit is not None:
        return explicit
    env = os.environ.get(CLIENT_SECRET_ENV)
    if env:
        return Path(env)
    if default.exists():
        return default
    return _discover_desktop_client(default.parent) or default


def _token_scopes(token_path: Path) -> set[str]:
    try:
        data: dict[str, Any] = json.loads(token_path.read_text("utf-8"))
    except (OSError, ValueError):
        return set()
    scopes = data.get("scopes") or []
    if isinstance(scopes, str):
        scopes = scopes.split()
    return set(scopes)


def _save_token(creds: Credentials, token_path: Path) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), "utf-8")
    with contextlib.suppress(OSError):
        token_path.chmod(0o600)
    log.info("Saved OAuth token to %s", token_path)


def _run_flow(client_secret: Path | None, scopes: list[str], open_browser: bool) -> Credentials:
    secret = client_secret_path(client_secret)
    if not secret.exists():
        raise AuthSetupError(
            f"OAuth client secret not found at {secret}. Create a 'Desktop app' OAuth client in "
            "Google Cloud Console, download its JSON to secrets/client_secret.json (or set "
            f"${CLIENT_SECRET_ENV}). See src/channel_export/README.md."
        )
    if client_type(secret) == "web":
        log.warning(
            "%s is a 'Web application' client; sign-in will likely fail with redirect_uri_mismatch. "
            "Use a 'Desktop app' client instead.",
            secret,
        )
    log.info("Using OAuth client file %s", secret)
    log.info(
        "Opening the Google sign-in page. If your channel is a Brand Account, choose the CHANNEL "
        "(not your personal profile) on the account picker."
    )
    flow = InstalledAppFlow.from_client_secrets_file(str(secret), scopes=scopes)
    # prompt=consent guarantees a refresh token is issued on every sign-in.
    return flow.run_local_server(port=0, open_browser=open_browser, prompt="consent")


def get_credentials(
    monetary: bool = False,
    client_secret: Path | None = None,
    token_path: Path = DEFAULT_TOKEN,
    open_browser: bool = True,
) -> Credentials:
    """Return valid user credentials, refreshing the cached token or re-running the browser flow."""
    scopes = scopes_for(monetary)
    creds: Credentials | None = None

    if token_path.exists():
        missing = set(scopes) - _token_scopes(token_path)
        if missing:
            log.info("Cached token lacks scopes %s; signing in again.", sorted(missing))
        else:
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), scopes)
            except (ValueError, KeyError) as err:
                log.warning("Ignoring unreadable token file %s (%s).", token_path, err)

    if creds is not None and creds.valid:
        return creds

    if creds is not None and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds, token_path)
            return creds
        except RefreshError as err:
            log.warning(
                "Could not refresh the saved token (%s). Apps whose OAuth consent screen is in "
                "'Testing' get refresh tokens that expire after 7 days; running the browser "
                "sign-in again.",
                err,
            )

    creds = _run_flow(client_secret, scopes, open_browser)
    _save_token(creds, token_path)
    return creds


def build_services(creds: Credentials) -> tuple[Any, Any, Any]:
    """Build (youtube v3, youtubeAnalytics v2, youtubereporting v1) clients from bundled discovery docs."""
    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    analytics = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)
    reporting = build("youtubereporting", "v1", credentials=creds, cache_discovery=False)
    return youtube, analytics, reporting
