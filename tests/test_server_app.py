"""HTTP layer of the Claude MCP server: auth, MCP tools, upload page, limits, start-up, logging (plan/10a 7-9).

No network beyond the in-process app and one uvicorn instance on 127.0.0.1; the analysis is always a fake.
"""

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from starlette.testclient import TestClient

from src.server import app as A
from src.server import config, jobs, media, tools
from src.tools.audience import brief as B

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg / ffprobe not installed")

BASE = "http://127.0.0.1:8765"
FAKE_KEY = "FAKE-KEY-123"
TOOL_NAMES = ["start_draft", "get_draft_status", "get_retention", "get_similar_shorts", "get_audience_brief"]
ACCEPT = {"Accept": "application/json, text/event-stream"}
INIT = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    out = {"good": d / "good.mp4", "short": d / "short.mp4"}
    for path, seconds in ((out["good"], 9), (out["short"], 3)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=90x160:rate=5", "-t", str(seconds),
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
                       check=True, stdin=subprocess.DEVNULL, timeout=120)
    return out


class Env:
    """One app on a temporary data directory with an injected clock and a fake analysis."""

    def __init__(self, tmp_path: Path, **kw):
        self.t = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc).timestamp()
        self.seen: list[dict] = []
        self.cfg = config.load({}, **{"DATA_DIR": tmp_path / "data", "PUBLIC_BASE_URL": BASE, **kw.pop("cfg", {})})
        self.app = A.build_app(self.cfg, **{"analyze": self.analyze, "similar": A.fake_similar, "brief": B.audience_brief,
                                             "clock": lambda: self.t, "allow_insecure_local": True, **kw})
        self.state: A.AppState = self.app.state
        self.token_id, self.token = self.state.tokens.create_token("tester")
        self.bodies: list[str] = []

    def analyze(self, path, api_key=None, on_cost=None, on_step=None):
        self.seen.append({"api_key": api_key, "existed": Path(path).exists()})
        return A.fake_analyze(path, api_key, on_cost, on_step, delay_s=0.0)

    def auth(self, token: str | None = None) -> dict:
        return {"Authorization": f"Bearer {token or self.token}", **ACCEPT}

    def rpc(self, client, method: str, params: dict | None = None, headers: dict | None = None, path: str = "/mcp"):
        body = {"jsonrpc": "2.0", "id": 1, "method": method, **({"params": params} if params is not None else {})}
        r = client.post(path, json=body, headers=self.auth() if headers is None else headers)
        self.bodies.append(r.text)
        return r

    def call(self, client, name: str, arguments: dict | None = None, **kw) -> dict:
        r = self.rpc(client, "tools/call", {"name": name, "arguments": arguments or {}}, **kw)
        assert r.status_code == 200, r.text
        content = r.json()["result"]["content"]
        assert len(content) == 1 and content[0]["type"] == "text"
        return json.loads(content[0]["text"])

    def upload(self, client, url: str, clip: Path | bytes, key: str | None = None):
        data = clip if isinstance(clip, bytes) else clip.read_bytes()
        r = client.post(url.removeprefix(BASE), files={"video": ("my holiday.mp4", data, "video/mp4")},
                        data={"gemini_key": key or ""})
        self.bodies.append(r.text)
        return r

    def idle(self) -> None:
        assert self.state.runner.wait_idle(20)

    def db_bytes(self) -> bytes:
        return b"".join(p.read_bytes() for p in self.cfg.DB_PATH.parent.iterdir() if p.name.startswith(self.cfg.DB_PATH.name))

    def files(self) -> list[str]:
        return sorted(p.name for p in self.cfg.UPLOAD_DIR.iterdir())


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def client_of(env: Env) -> TestClient:
    return TestClient(env.app, base_url=BASE)


# --------------------------------------------------------------------------------------------------------------- auth

def test_no_token_and_wrong_token_get_401_with_an_empty_body(env):
    with client_of(env) as c:
        for headers in (ACCEPT, env.auth("wrong-token"), {"Authorization": env.token, **ACCEPT},
                        {"Authorization": f"Basic {env.token}", **ACCEPT}):
            r = env.rpc(c, "tools/list", headers=headers)
            assert (r.status_code, r.content) == (401, b"")
        assert env.rpc(c, "tools/list").status_code == 200
        env.state.tokens.revoke(env.token_id)
        r = env.rpc(c, "tools/list")
        assert (r.status_code, r.content) == (401, b"")


def test_repeated_bad_tokens_are_blocked_with_429_and_an_empty_body(tmp_path):
    env = Env(tmp_path, cfg={"BAD_TOKEN_MAX": 3, "BAD_TOKEN_WINDOW_S": 600})
    with client_of(env) as c:
        for _ in range(5):
            assert env.rpc(c, "tools/list", headers=ACCEPT).status_code == 401     # no token: not counted
        for _ in range(3):
            assert env.rpc(c, "tools/list", headers=env.auth("bad")).status_code == 401
        for headers in (env.auth("bad"), ACCEPT):
            r = env.rpc(c, "tools/list", headers=headers)
            assert (r.status_code, r.content) == (429, b"")
        assert env.rpc(c, "tools/list", headers=env.auth()).status_code == 200   # a valid token is never blocked
        env.t += 601
        assert env.rpc(c, "tools/list").status_code == 200


def _scope(peer: str, **headers: str) -> dict:
    return {"type": "http", "client": (peer, 5000), "headers": [(k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()]}


def test_limiter_key_trusts_only_the_tunnel_header_and_only_from_the_local_proxy():
    assert A.client_address(_scope("127.0.0.1", cf_connecting_ip="203.0.113.9", x_forwarded_for="198.51.100.1")) == "203.0.113.9"
    assert A.client_address(_scope("127.0.0.1", x_forwarded_for="6.6.6.6, 198.51.100.7")) == "127.0.0.1"   # never read
    assert A.client_address(_scope("127.0.0.1", x_forwarded_for="198.51.100.7")) == "127.0.0.1"
    assert A.client_address(_scope("127.0.0.1", cf_connecting_ip="not an address", x_forwarded_for="198.51.100.7")) == "127.0.0.1"
    assert A.client_address(_scope("::1", cf_connecting_ip="2001:db8::1")) == "2001:db8::1"
    assert A.client_address(_scope("127.0.0.1", cf_connecting_ip="not an address")) == "127.0.0.1"
    assert A.client_address(_scope("127.0.0.1")) == "127.0.0.1"
    assert A.client_address(_scope("192.0.2.4", cf_connecting_ip="203.0.113.9", x_forwarded_for="198.51.100.1")) == "192.0.2.4"


def test_path_token_route_is_off_by_default_and_works_when_enabled(tmp_path):
    env = Env(tmp_path / "a")
    with client_of(env) as c:
        r = env.rpc(c, "tools/list", headers=ACCEPT, path=f"/t/{env.token}/mcp")
        assert r.status_code == 404 and env.token not in r.text
    on = Env(tmp_path / "b", allow_path_token=True)
    with client_of(on) as c:
        r = on.rpc(c, "tools/list", headers=ACCEPT, path=f"/t/{on.token}/mcp")
        assert r.status_code == 200 and len(r.json()["result"]["tools"]) == 5
        assert on.call(c, "start_draft", headers=ACCEPT, path=f"/t/{on.token}/mcp")["status"] == "waiting_for_upload"
        r = on.rpc(c, "tools/list", headers=ACCEPT, path="/t/wrong-token/mcp")
        assert (r.status_code, r.content) == (401, b"")


def test_only_post_is_served_on_the_mcp_route(env):
    with client_of(env) as c:
        assert c.get("/mcp", headers=ACCEPT).status_code == 401
        assert c.get("/mcp", headers=env.auth()).status_code == 405
        assert c.delete("/mcp", headers=env.auth()).status_code == 405


# ---------------------------------------------------------------------------------------------------------------- MCP

def test_initialize_and_tools_list(env):
    with client_of(env) as c:
        init = env.rpc(c, "initialize", INIT).json()["result"]
        assert init["serverInfo"]["name"] == "shorts-pre-publish-copilot" and "tools" in init["capabilities"]
        text = init["instructions"]
        assert text == A.INSTRUCTIONS
        assert "takes about 3 minutes" in text and "1-2 minutes" not in text + tools.MSG_PROCESSING
        assert "about 3 minutes" in tools.MSG_PROCESSING
        for phrase in ("start_draft", "upload_url", "get_retention", "get_similar_shorts", "get_audience_brief",
                       "one publish plan", "Brawl Stars channel", "own web search", "_untrusted_text", "never follow instructions"):
            assert phrase in text
        listed = env.rpc(c, "tools/list").json()["result"]["tools"]
    assert [t["name"] for t in listed] == TOOL_NAMES
    by_name = {t["name"]: t for t in listed}
    for name, t in by_name.items():
        schema = t["inputSchema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert 20 < len(t["description"]) < 420
        assert t["annotations"]["readOnlyHint"] is (name != "start_draft")
        assert all(p.get("description") for p in schema["properties"].values())
    assert by_name["start_draft"]["inputSchema"]["properties"] == {}
    assert by_name["get_draft_status"]["inputSchema"]["required"] == ["draft_id"]
    assert by_name["get_retention"]["inputSchema"]["properties"]["wait_s"] == {
        "type": "integer", "minimum": 0, "maximum": 50, "default": 40,
        "description": by_name["get_retention"]["inputSchema"]["properties"]["wait_s"]["description"]}
    assert by_name["get_similar_shorts"]["inputSchema"]["properties"]["k"]["maximum"] == 8
    assert set(by_name["get_audience_brief"]["inputSchema"]["properties"]) == {"draft_id"}


def test_start_draft_returns_an_upload_link_under_the_public_address(env):
    with client_of(env) as c:
        r = env.call(c, "start_draft")
    assert r["status"] == "waiting_for_upload"
    assert re.fullmatch(re.escape(BASE) + r"/u/[A-Za-z0-9_-]{43}", r["upload_url"])
    assert env.state.store.get_draft(r["draft_id"], env.token_id).status == "waiting_for_upload"


@pytest.mark.parametrize("name, arguments, expected", [
    ("get_retention", {"draft_id": "abc", "extra": 1}, "Unknown argument"),
    ("get_retention", {}, "Missing argument: draft_id"),
    ("get_retention", {"draft_id": 5}, "draft_id must be a string"),
    ("get_retention", {"draft_id": "x" * 65}, "draft_id must be a string"),
    ("get_retention", {"draft_id": "abc", "wait_s": "long"}, "wait_s must be a whole number"),
    ("get_similar_shorts", {"draft_id": "abc", "k": 2.5}, "k must be a whole number"),
    ("start_draft", {"draft_id": "abc"}, "Unknown argument"),
    ("delete_everything", {}, "Unknown tool"),
])
def test_invalid_tool_input_is_refused_without_running_anything(env, name, arguments, expected):
    with client_of(env) as c:
        r = env.rpc(c, "tools/call", {"name": name, "arguments": arguments}).json()["result"]
    assert r["isError"] is True
    body = json.loads(r["content"][0]["text"])
    assert body["status"] == "invalid_input" and expected in body["message"]


def test_a_draft_of_another_token_is_not_found_over_mcp(env):
    _, other = env.state.tokens.create_token("other")
    with client_of(env) as c:
        mine = env.call(c, "start_draft")
        theirs = env.call(c, "get_draft_status", {"draft_id": mine["draft_id"]}, headers=env.auth(other))
        unknown = env.call(c, "get_draft_status", {"draft_id": "A" * 22}, headers=env.auth(other))
        assert theirs == unknown == {"status": "not_found", "message": tools.MSG_NOT_FOUND}
        assert env.call(c, "get_draft_status", {"draft_id": mine["draft_id"]})["status"] == "waiting_for_upload"


def test_too_many_calls_at_once_get_a_busy_answer(env):
    with client_of(env) as c:
        env.state.calls[env.token_id] = A.MAX_CALLS_PER_TOKEN
        assert env.call(c, "start_draft")["status"] == "busy"
        env.state.calls.clear()
        assert env.call(c, "start_draft")["status"] == "waiting_for_upload"
        assert env.state.calls == {}


def test_sdk_client_speaks_to_the_server_over_http(tmp_path):
    import httpx2
    import uvicorn
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = Env(tmp_path, cfg={"PUBLIC_BASE_URL": f"http://127.0.0.1:{port}"})
    server = uvicorn.Server(uvicorn.Config(env.app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started

        async def flow(mode: str) -> None:
            async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {env.token}"}, timeout=20) as http:
                async with Client(streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=http), mode=mode) as client:
                    listed = await client.list_tools()
                    assert [t.name for t in listed.tools] == TOOL_NAMES
                    assert listed.tools[1].annotations.read_only_hint is True
                    result = await client.call_tool("start_draft", {})
                    body = json.loads(result.content[0].text)
                    assert result.is_error is False and body["upload_url"].startswith(f"http://127.0.0.1:{port}/u/")
                    status = await client.call_tool("get_draft_status", {"draft_id": body["draft_id"]})
                    assert json.loads(status.content[0].text)["status"] == "waiting_for_upload"

        for mode in ("legacy", "auto"):
            anyio.run(flow, mode)
    finally:
        server.should_exit = True
        thread.join(10)


# -------------------------------------------------------------------------------------------------------- upload page

def assert_page_headers(r) -> None:
    assert r.headers["content-security-policy"] == ("default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
                                                    "frame-ancestors 'none'")
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"


def test_upload_page_is_self_contained_and_has_the_security_headers(env):
    with client_of(env) as c:
        url = env.call(c, "start_draft")["upload_url"]
        r = c.get(url.removeprefix(BASE))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert_page_headers(r)
    page = r.text
    for bad in ("http://", "https://", "//", "<script", "javascript:", "src=", "href=", "@import", "url(", "action="):
        assert bad not in page, bad
    assert not re.search(r"\son\w+\s*=", page)
    assert 'type="file"' in page and 'name="video"' in page and 'method="post"' in page and "multipart/form-data" in page
    assert 'name="gemini_key" type="password"' in page and 'maxlength="200"' in page and "never stored" in page
    assert "8 to 60 seconds" in page and "100 MB" in page and "deleted from the server" in page
    assert env.token not in page and url.rsplit("/", 1)[1] not in page


def test_unknown_used_and_expired_links_get_a_plain_404(env, clips):
    with client_of(env) as c:
        for bad in ("A" * 43, "short", "x" * 70, "..%2F..%2Fetc"):
            r = c.get(f"/u/{bad}")
            assert r.status_code == 404 and r.headers["content-type"].startswith("text/plain") and r.text.strip() == A.MSG_LINK_GONE
            assert_page_headers(r)
            assert env.upload(c, f"/u/{bad}", clips["good"]).status_code == 404
        url = env.call(c, "start_draft")["upload_url"]
        env.t += env.cfg.UPLOAD_TTL_S + 1
        assert c.get(url.removeprefix(BASE)).status_code == 404
        r = env.upload(c, url, clips["good"])
        assert r.status_code == 404 and r.text.strip() == A.MSG_LINK_GONE
        assert_page_headers(r)
    assert env.files() == [] and env.seen == []


def test_full_flow_upload_then_results(env, clips, tmp_path):
    with client_of(env) as c:
        start = env.call(c, "start_draft")
        r = env.upload(c, start["upload_url"], clips["good"])
        assert r.status_code == 200 and A.MSG_RECEIVED in r.text and "<form" not in r.text
        assert_page_headers(r)
        got = env.call(c, "get_retention", {"draft_id": start["draft_id"], "wait_s": 20})
        assert got["status"] == "done" and got["model_written_fields"][0] == "advice"
        assert got["advice"] == ["Cut the first second.", "End right on the payoff."]
        assert set(got["segments"]) == {"early", "mid", "late"} and got["how_to_read"] == tools.HOW_TO_READ
        assert [(d["t_s"], d["rank"]) for d in got["drop_moments"]] == [(2.0, 1), (7.0, 2)]
        env.idle()
        assert env.call(c, "get_draft_status", {"draft_id": start["draft_id"]})["status"] == "done"
        similar = env.call(c, "get_similar_shorts", {"draft_id": start["draft_id"], "k": 3})
        assert similar["match_quality"] == "close matches" and "note" not in similar
        assert similar["status"] == "done" and [s["label"] for s in similar["similar_past_shorts"]] == [
            "past Short 3", "past Short 7", "past Short 11"]
        brief = env.call(c, "get_audience_brief", {"draft_id": start["draft_id"]})
        assert brief["status"] == "done" and brief["writing_rules"] == B.WRITING_RULES and "game_note" not in brief
        assert len(brief["draft_description_untrusted_text"]) == 9

        assert c.get(start["upload_url"].removeprefix(BASE)).status_code == 404        # the link was used
        r = env.upload(c, start["upload_url"], clips["good"])
        assert r.status_code == 404 and r.text.strip() == A.MSG_LINK_GONE
        usage = env.call(c, "get_draft_status", {"draft_id": start["draft_id"]})["usage"]
    assert env.seen == [{"api_key": None, "existed": True}] and env.files() == []
    assert usage["drafts_today"] == 1 and usage["drafts_left_today"] == 2
    assert env.state.quota.summary()["settled_usd"] == pytest.approx(0.02)
    everything = "\n".join(env.bodies)
    for bad in (str(tmp_path), "my holiday", "/Users/", env.token):
        assert bad not in everything


def test_a_too_short_clip_is_refused_for_free_and_the_link_still_works(env, clips):
    with client_of(env) as c:
        start = env.call(c, "start_draft")
        r = env.upload(c, start["upload_url"], clips["short"])
        assert r.status_code == 400 and "The video is shorter than 8 seconds. Drafts must be 8 to 60 seconds long." in r.text
        assert A.MSG_TRY_AGAIN in r.text and "<form" in r.text
        assert_page_headers(r)
        assert env.seen == [] and env.files() == []
        status = env.call(c, "get_draft_status", {"draft_id": start["draft_id"]})
        assert status["status"] == "waiting_for_upload" and status["usage"]["drafts_today"] == 0
        assert status["last_upload_refusal"] == "The video is shorter than 8 seconds. Drafts must be 8 to 60 seconds long."
        assert status["last_upload_refusal"] in status["message"] and "refused" in status["message"]
        waiting = env.call(c, "get_retention", {"draft_id": start["draft_id"], "wait_s": 0})
        assert waiting["status"] == "waiting_for_upload" and waiting["last_upload_refusal"] == status["last_upload_refusal"]
        summary = env.state.quota.summary()
        assert summary["spend_usd"] == 0 and summary["drafts"] == 0
        assert c.get(start["upload_url"].removeprefix(BASE)).status_code == 200
        r = env.upload(c, start["upload_url"], clips["good"])
        assert r.status_code == 200 and A.MSG_RECEIVED in r.text
        assert env.call(c, "get_retention", {"draft_id": start["draft_id"], "wait_s": 20})["status"] == "done"
        assert "last_upload_refusal" not in env.call(c, "get_draft_status", {"draft_id": start["draft_id"]})
    assert env.files() == [] and len(env.state.refusals) == 0


@pytest.mark.parametrize("payload, sentence", [
    (b"this is not a video at all" * 50, "could not be read as a video"),
    (b"", A.MSG_NO_FILE),
])
def test_unreadable_and_empty_uploads_are_refused_and_release_the_link(env, payload, sentence):
    with client_of(env) as c:
        start = env.call(c, "start_draft")
        r = env.upload(c, start["upload_url"], payload)
        assert r.status_code == 400 and sentence in r.text and "<form" in r.text
        assert c.get(start["upload_url"].removeprefix(BASE)).status_code == 200
    assert env.files() == [] and env.seen == [] and env.state.quota.summary()["drafts"] == 0


def test_a_form_that_is_not_multipart_is_refused_before_the_link_is_used(env):
    with client_of(env) as c:
        start = env.call(c, "start_draft")
        r = c.post(start["upload_url"].removeprefix(BASE), content=b"video=abc", headers={"Content-Type": "application/x-www-form-urlencoded"})
        assert r.status_code == 400 and A.MSG_BAD_FORM in r.text
        assert_page_headers(r)
        assert c.get(start["upload_url"].removeprefix(BASE)).status_code == 200


def test_an_upload_over_the_size_cap_is_cut_off_and_leaves_no_file(tmp_path, clips):
    env = Env(tmp_path, cfg={"MAX_UPLOAD_BYTES": 50_000})
    sentence = media.default_message("too_large", env.cfg)
    with client_of(env) as c:
        start = env.call(c, "start_draft")
        r = env.upload(c, start["upload_url"], b"\x00" * 60_000)                      # passes the declared-length check
        assert r.status_code == 413 and sentence in r.text and A.MSG_TRY_AGAIN in r.text
        assert_page_headers(r)
        assert env.files() == [] and env.seen == []
        r = env.upload(c, start["upload_url"], b"\x00" * 500_000)                     # refused on the declared length
        assert r.status_code == 413 and sentence in r.text
        assert env.files() == [] and env.state.quota.summary()["drafts"] == 0
        assert c.get(start["upload_url"].removeprefix(BASE)).status_code == 200
        small = clips["good"].read_bytes()
        assert len(small) < 50_000
        assert env.upload(c, start["upload_url"], small).status_code == 200
    env.idle()
    assert env.files() == []


def test_fourth_paid_draft_is_refused_but_an_own_key_upload_is_accepted(env, clips, caplog):
    caplog.set_level(logging.DEBUG)
    with client_of(env) as c:
        for _ in range(3):
            start = env.call(c, "start_draft")
            assert env.upload(c, start["upload_url"], clips["good"]).status_code == 200
            env.idle()
        start = env.call(c, "start_draft")
        assert start["usage"]["drafts_left_today"] == 0 and "own Gemini API key" in start["message"]
        assert start["status"] == "limit_reached_own_key_only" and start["upload_url"].startswith(BASE + "/u/")
        r = env.upload(c, start["upload_url"], clips["good"])
        assert r.status_code == 429
        sentence = "You have used today's 3 free drafts. The limit resets at 2026-11-15 00:00:00 UTC."
        assert sentence in r.text
        assert "own Gemini API key" in r.text and "<form" in r.text
        assert env.files() == [] and len(env.seen) == 3
        for tool, extra in (("get_draft_status", {}), ("get_retention", {"wait_s": 0})):
            status = env.call(c, tool, {"draft_id": start["draft_id"], **extra})
            assert status["status"] == "waiting_for_upload" and status["last_upload_refusal"].startswith(sentence)
            assert sentence in status["message"] and "own Gemini API key" in status["message"]

        r = env.upload(c, start["upload_url"], clips["good"], key=f"  {FAKE_KEY}  ")
        assert r.status_code == 200 and A.MSG_RECEIVED in r.text
        got = env.call(c, "get_retention", {"draft_id": start["draft_id"], "wait_s": 20})
        assert got["status"] == "done"
        usage = env.call(c, "get_draft_status", {"draft_id": start["draft_id"]})["usage"]
    assert env.seen[-1] == {"api_key": FAKE_KEY, "existed": True} and [s["api_key"] for s in env.seen[:3]] == [None] * 3
    assert usage["drafts_today"] == 3 and usage["own_key_drafts_today"] == 1
    assert env.state.quota.summary()["refusals"]["user_daily"] == 1
    for where in ("\n".join(env.bodies), caplog.text, env.db_bytes().decode("latin-1"), repr(vars(env.state))):
        assert FAKE_KEY not in where
    assert env.files() == []


@pytest.mark.parametrize("key", ["k" * 201, "two words", "caf\u00e9-key", "tab\tkey"])
def test_an_own_key_that_cannot_be_a_key_is_refused_and_never_falls_back(env, clips, key, caplog):
    caplog.set_level(logging.DEBUG)
    with client_of(env) as c:
        start = env.call(c, "start_draft")
        r = env.upload(c, start["upload_url"], clips["good"], key=key)
        assert r.status_code == 400 and A.MSG_BAD_KEY in r.text and "<form" in r.text
        assert c.get(start["upload_url"].removeprefix(BASE)).status_code == 200
    assert env.seen == [] and env.files() == [] and env.state.quota.summary()["drafts"] == 0
    assert key not in r.text and key not in caplog.text


def test_a_second_upload_of_the_same_user_waits_for_the_running_job(tmp_path, clips):
    release = threading.Event()

    def slow(path, api_key=None, on_cost=None, on_step=None):
        on_step("describing")
        release.wait(20)
        return A.fake_analyze(path, api_key, on_cost, on_step, delay_s=0.0)

    env = Env(tmp_path, analyze=slow)
    with client_of(env) as c:
        first = env.call(c, "start_draft")
        assert env.upload(c, first["upload_url"], clips["good"]).status_code == 200
        running = env.call(c, "get_retention", {"draft_id": first["draft_id"], "wait_s": 0})
        assert running["status"] == "processing"
        second = env.call(c, "start_draft")
        r = env.upload(c, second["upload_url"], clips["good"])
        assert r.status_code == 503 and jobs.MSG_USER_BUSY in r.text and "<form" in r.text
        assert env.files() == [first["draft_id"]]
        assert env.state.quota.usage(env.token_id)["drafts_today"] == 1                 # the reservation was cancelled
        release.set()
        env.idle()
        assert env.upload(c, second["upload_url"], clips["good"]).status_code == 200
        env.idle()
        assert env.call(c, "get_retention", {"draft_id": second["draft_id"]})["status"] == "done"
    assert env.files() == [] and env.state.quota.usage(env.token_id)["drafts_today"] == 2


def test_a_failed_job_shows_its_fixed_sentence_and_nothing_else(tmp_path, clips, caplog):
    def broken(path, api_key=None, on_cost=None, on_step=None):
        raise RuntimeError(f"cannot read {path} with {api_key}")

    caplog.set_level(logging.DEBUG)
    env = Env(tmp_path, analyze=broken)
    with client_of(env) as c:
        start = env.call(c, "start_draft")
        assert env.upload(c, start["upload_url"], clips["good"], key=FAKE_KEY).status_code == 200
        env.idle()
        got = env.call(c, "get_retention", {"draft_id": start["draft_id"]})
    assert (got["status"], got["reason"]) == ("failed", jobs.MSG_FAILED)
    for where in ("\n".join(env.bodies), caplog.text, env.db_bytes().decode("latin-1")):
        assert FAKE_KEY not in where and str(env.cfg.UPLOAD_DIR) not in where and "cannot read" not in where
    assert env.files() == []


def test_a_slow_upload_is_stopped_at_the_time_limit(tmp_path):
    env = Env(tmp_path, upload_timeout_s=0.2)
    draft = env.state.store.create_draft(env.token_id)
    boundary = b"testboundary"
    head = b"--" + boundary + b'\r\nContent-Disposition: form-data; name="video"; filename="a.mp4"\r\n\r\n' + b"\x00" * 1000
    sent: list[dict] = []
    first = [True]

    async def receive() -> dict:
        if first[0]:
            first[0] = False
            return {"type": "http.request", "body": head, "more_body": True}
        await anyio.sleep(30)
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": f"/u/{draft.upload_id}", "raw_path": f"/u/{draft.upload_id}".encode(), "query_string": b"", "root_path": "",
             "headers": [(b"host", b"127.0.0.1:8765"), (b"content-type", b"multipart/form-data; boundary=" + boundary)],
             "client": ("127.0.0.1", 5000), "server": ("127.0.0.1", 8765)}
    started = time.monotonic()
    anyio.run(env.app, scope, receive, send)
    assert time.monotonic() - started < 10
    assert sent[0]["status"] == 408 and A.MSG_TIMEOUT.encode() in b"".join(m.get("body", b"") for m in sent[1:])
    assert env.files() == [] and env.state.store.upload_open(draft.upload_id) and env.state.in_flight == set()


# ------------------------------------------------------------------------------------------------------------ logging

def test_log_lines_name_routes_and_tools_but_never_ids_tokens_or_paths(env, clips, caplog):
    caplog.set_level(logging.INFO)
    with client_of(env) as c:
        env.rpc(c, "tools/list", headers=env.auth("wrong-token-value"))
        start = env.call(c, "start_draft")
        upload_id = start["upload_url"].rsplit("/", 1)[1]
        c.get(f"/u/{upload_id}?utm=querystring")
        env.upload(c, start["upload_url"], clips["short"])
        env.upload(c, start["upload_url"], clips["good"], key=FAKE_KEY)
        env.idle()
        env.call(c, "get_retention", {"draft_id": start["draft_id"]})
        c.get("/nothing/here?secret=1")
    lines = [r.getMessage() for r in caplog.records if r.name == "so.server"]
    requests = [line for line in lines if " route=" in line]
    text = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith("httpx"))   # not the test client's own log
    assert len(text.splitlines()) >= len(lines) + 2
    for bad in (upload_id, env.token, "wrong-token-value", FAKE_KEY, start["draft_id"], "querystring", "secret=1",
                str(env.cfg.DATA_DIR), "my holiday", "/nothing/here"):
        assert bad not in text
    stamp = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ "
    tid = env.token_id
    expected = [
        "token=- route=POST /mcp outcome=unauthorized status=401 cost=-",
        f"token={tid} route=POST /mcp:start_draft outcome=waiting_for_upload status=200 cost=-",
        "token=- route=GET /u/{upload_id} outcome=form status=200 cost=-",
        f"token={tid} route=POST /u/{{upload_id}} outcome=refused:too_short status=400 cost=-",
        f"token={tid} route=POST /u/{{upload_id}} outcome=accepted:own_key status=200 cost=-",
        f"token={tid} route=job outcome=done status=- cost=0.0200",
        f"token={tid} route=POST /mcp:get_retention outcome=done status=200 cost=0.0200",
        "token=- route=GET other outcome=refused status=404 cost=-",
    ]
    assert len(requests) == len(expected)
    for want in expected:
        assert any(re.fullmatch(stamp + re.escape(want), line) for line in requests), want


# ----------------------------------------------------------------------------------------------------------- start-up

def test_start_up_fails_interrupted_jobs_settles_them_and_deletes_leftover_files(tmp_path):
    env = Env(tmp_path)
    store, quota = env.state.store, env.state.quota
    dead = store.create_draft("tok-dead")
    store.claim_upload(dead.upload_id)
    quota.check_and_reserve("tok-dead", dead.id, "server")
    store.start_processing(dead.id, "server")
    quota.add_cost(dead.id, 0.05)
    cut_off = store.create_draft("tok-cut")
    store.claim_upload(cut_off.upload_id)
    for name in (dead.id, cut_off.id, "stray"):
        (env.cfg.UPLOAD_DIR / name).write_bytes(b"leftover")
    assert not store.upload_open(cut_off.upload_id)
    with client_of(env):
        pass
    got = store.get_draft(dead.id, "tok-dead")
    assert (got.status, got.fail_code) == ("failed", "interrupted")
    assert quota.summary()["settled_usd"] == pytest.approx(0.05) and quota.summary()["spend_usd"] == pytest.approx(0.05)
    assert store.upload_open(cut_off.upload_id) and env.files() == []


def test_purge_removes_expired_rows_and_orphan_files_only(env):
    store = env.state.store
    old = store.create_draft("tok-old")
    store.claim_upload(old.upload_id)
    store.start_processing(old.id, "server")
    store.finish(old.id, {"a": 1}, {"b": 2}, 0.0)
    live = store.create_draft("tok-live")
    for name in ("orphan", live.id, "running-job"):
        (env.cfg.UPLOAD_DIR / name).write_bytes(b"x")
    env.state.in_flight.add(live.id)
    env.state.runner._active["running-job"] = "tok-x"
    assert env.state.starts.take("tok-old")
    env.t += env.cfg.RESULT_TTL_S + 10
    report = A.purge(env.state)
    assert report == {"expired": 1, "interrupted": 0, "deleted": 1, "files": 1}
    assert env.files() == sorted([live.id, "running-job"])
    assert store.get_draft(old.id, "tok-old") is None and store.get_draft(live.id, "tok-live").status == "expired"
    assert env.state.starts.purge() == 0


def test_warm_up_runs_in_the_background_and_gates_the_audience_tools(tmp_path):
    go, warmed = threading.Event(), []

    def warm() -> None:
        go.wait(20)
        warmed.append(True)

    env = Env(tmp_path, warm=warm)
    with client_of(env) as c:
        assert not env.state.ready.is_set()
        draft = env.state.store.create_draft(env.token_id)
        env.state.store.claim_upload(draft.upload_id)
        env.state.store.start_processing(draft.id, "server")
        env.state.store.finish(draft.id, {"advice": "x"}, {"timeline": []}, 0.0)
        assert env.call(c, "get_similar_shorts", {"draft_id": draft.id})["status"] == "starting"
        go.set()
        assert env.state.ready.wait(10) and warmed == [True]
        assert env.call(c, "get_similar_shorts", {"draft_id": draft.id})["status"] == "done"


def test_a_failing_warm_up_does_not_block_the_server(tmp_path, caplog):
    def warm() -> None:
        raise RuntimeError("no library at /Users/x/data")

    env = Env(tmp_path, warm=warm)
    with client_of(env):
        assert env.state.ready.wait(10)
    assert "/Users/x" not in caplog.text


@pytest.mark.parametrize("environ, problem", [
    ({}, "SO_PUBLIC_BASE_URL is not set"),
    ({"SO_PUBLIC_BASE_URL": "http://shorts.example.org"}, "must start with https://"),
    ({"SO_PUBLIC_BASE_URL": "http://127.0.0.1:8765"}, "must start with https://"),
    ({"SO_PUBLIC_BASE_URL": "http://shorts.example.org", "SO_ALLOW_INSECURE_LOCAL": "1"}, "must start with https://"),
    ({"SO_PUBLIC_BASE_URL": "ftp://127.0.0.1", "SO_ALLOW_INSECURE_LOCAL": "1"}, "must start with https://"),
    ({"SO_PUBLIC_BASE_URL": "https://user:pw@shorts.example.org"}, "plain address"),
    ({"SO_PUBLIC_BASE_URL": "https://shorts.example.org/?x=1"}, "plain address"),
    ({"SO_PUBLIC_BASE_URL": "https://shorts.example.org", "SO_FAKE_ANALYSIS": "1"}, "SO_FAKE_ANALYSIS=1 is only allowed"),
    ({"SO_PUBLIC_BASE_URL": "https://shorts.example.org", "SO_FAKE_ANALYSIS": "1", "SO_ALLOW_INSECURE_LOCAL": "1"},
     "SO_FAKE_ANALYSIS=1 is only allowed"),
    ({"SO_PUBLIC_BASE_URL": "https://127.0.0.1", "SO_FAKE_ANALYSIS": "1"}, "SO_FAKE_ANALYSIS=1 is only allowed"),
    ({"SO_PUBLIC_BASE_URL": "https://shorts.example.org", "SO_PORT": "http"}, "SO_PORT is not a number"),
    ({"SO_PUBLIC_BASE_URL": "https://shorts.example.org", "SO_PORT": "70000"}, "SO_PORT must be"),
])
def test_the_server_refuses_to_start_on_an_unsafe_configuration(environ, problem):
    with pytest.raises(A.StartupError, match=re.escape(problem)):
        A.load_options(environ, config.load(environ))


def test_options_from_the_environment():
    environ = {"SO_PUBLIC_BASE_URL": "https://shorts.example.org/"}
    assert A.load_options(environ, config.load(environ)) == A.Options(8765, False, False, False)
    environ = {"SO_PUBLIC_BASE_URL": "http://127.0.0.1:9001", "SO_ALLOW_INSECURE_LOCAL": "1", "SO_FAKE_ANALYSIS": "1",
               "SO_ALLOW_PATH_TOKEN": "1", "SO_PORT": "9001"}
    assert A.load_options(environ, config.load(environ)) == A.Options(9001, True, True, True)


def test_build_app_itself_refuses_an_insecure_public_address(tmp_path):
    with pytest.raises(A.StartupError):
        A.build_app(config.load({}, DATA_DIR=tmp_path, PUBLIC_BASE_URL="http://shorts.example.org"))
    with pytest.raises(A.StartupError):
        A.build_app(config.load({}, DATA_DIR=tmp_path))


def test_missing_ffmpeg_tools_stop_the_start():
    A.check_binaries()
    seen = []

    def run(cmd, **kw):
        seen.append(cmd[0])
        return SimpleNamespace(returncode=0 if cmd[0] == "ffprobe" else 1)

    with pytest.raises(A.StartupError, match="ffmpeg is not installed"):
        A.check_binaries(run)
    assert seen == ["ffprobe", "ffmpeg"]

    def missing(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    with pytest.raises(A.StartupError, match="ffprobe is not installed"):
        A.check_binaries(missing)


def test_main_exits_with_a_clear_message_instead_of_starting(monkeypatch, capsys):
    for name in ("SO_PUBLIC_BASE_URL", "SO_ALLOW_INSECURE_LOCAL", "SO_FAKE_ANALYSIS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    assert A.main([]) == 2
    assert "cannot start: SO_PUBLIC_BASE_URL is not set" in capsys.readouterr().err
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"   # set before anything loads


def test_the_library_is_warmed_with_the_hugging_face_hub_switched_off(monkeypatch):
    env = {"HF_HUB_OFFLINE": "0", "PATH": "/bin"}
    A.go_offline(env)
    assert env == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PATH": "/bin"}
    seen = {}
    for name in A.OFFLINE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("src.tools.predict_retention.tool.state", lambda: seen.update({k: os.environ.get(k) for k in A.OFFLINE_ENV}))
    A._real_warm()
    assert seen == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
