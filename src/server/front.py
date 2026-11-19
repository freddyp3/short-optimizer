"""The always-on front of the server, for when the tools run on another machine.

Two copies of the same server can be chained. The **front** (a small always-on machine) is the public address: it
does sign-in, keeps the users, and speaks MCP to Claude. It does not run the tools; it passes every tool call on
to the **home** server, which has the library, the uploads and the reviews, and may be switched off.

    Claude -> front (/mcp, sign-in, users)  --tools/call-->  home (/mcp)  [may be offline]

The front proves itself to the home server with a shared secret and names the signed-in user in ``X-SO-User``;
the home server then treats the call as that user's (``trusted_user``). When the home server does not answer,
the tool call gets a normal result saying the service is offline, so signing up and asking for access keep
working around the clock while the tools work whenever the home server is on.
"""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import Callable

USER_HEADER = "x-so-user"
USER_ID = re.compile(r"(u_)?[0-9a-f]{8}")
TIMEOUT_S = 150.0
MSG_OFFLINE = ("Shorts Pre-Publish Copilot is offline at the moment, so a draft cannot be checked right now. Nothing is "
               "wrong with the user's access. Tell the user this plainly and suggest trying again later; do not retry "
               "in a loop.")
ACCEPT = "application/json, text/event-stream"


class Upstream:
    """The home server as seen from the front. ``post(url, json=, headers=, timeout=)`` defaults to ``requests.post``."""

    def __init__(self, base_url: str, secret: str, post: Callable | None = None, timeout_s: float = TIMEOUT_S):
        if not secret or len(secret) < 32:
            raise ValueError("the proxy secret must have at least 32 characters")
        self.url, self._secret, self._post, self.timeout_s = base_url.rstrip("/") + "/mcp", secret, post, timeout_s

    def call(self, user_id: str, name: str, arguments: dict) -> tuple[dict, bool] | None:
        """The home server's answer to one tool call as ``(result, is_error)``, or None when it cannot be reached
        or answers with anything but a tool result."""
        post = self._post
        if post is None:
            import requests

            post = requests.post
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        headers = {"Authorization": f"Bearer {self._secret}", USER_HEADER: user_id, "Accept": ACCEPT}
        try:
            r = post(self.url, json=body, headers=headers, timeout=self.timeout_s)
            if r.status_code != 200:
                return None
            result = r.json()["result"]
            return json.loads(result["content"][0]["text"]), bool(result.get("isError"))
        except Exception:
            return None


def trusted_user(secret: str | None, presented: str | None, user_header: str | None) -> str | None:
    """On the home server: the user a request from the front is made for, or None when the request does not
    carry the shared secret and a well-formed user id. The comparison takes the same time for any guess."""
    if not secret or not isinstance(presented, str) or not presented:
        return None
    if not hmac.compare_digest(presented.encode("utf-8", "surrogatepass"), secret.encode()):
        return None
    user = (user_header or "").strip()
    return user if USER_ID.fullmatch(user) else None
