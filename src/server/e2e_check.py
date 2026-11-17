"""Stand-alone end-to-end check of the MCP server, written as an outside client (plan/10 section 8, T2 / T3).

    ./venv/bin/python src/server/e2e_check.py --base-url https://example.org --videos v1.mp4 v2.mp4 v3.mp4 \
        --scratch /tmp/e2e [--skip-paid] [--server-log FILE] [--data-dir DIR]

Tokens come from ``--token-a`` / ``--token-b`` or, better (not visible in ``ps``), from the environment variables
``SO_E2E_TOKEN_A`` / ``SO_E2E_TOKEN_B``. They are never printed.

The script imports nothing from this repository. It speaks MCP over streamable HTTP with the official ``mcp`` SDK
client, and uses httpx for the upload page, the way a browser would. It knows the tools only from what the server
advertises. One line per check (PASS / FAIL / SKIP); every tool result is written as JSON into ``--scratch``; the
exit code is 1 when any check failed.

MONEY: without ``--skip-paid`` it uploads each of the three ``--videos`` exactly once on the server's key (three
paid analyses). It never uploads a real video twice and stops the paid part at the first paid draft that does not
end in ``done``. All other uploads are refused files, a daily-limit refusal, or a draft on a fake own key.

``--server-log`` and ``--data-dir`` are optional and only make sense when the server runs on this machine: they add
the checks that need the server's files (log lines, upload directory, SQLite bytes, spend table). Without them
those parts are skipped, so the script works unchanged against a public address.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

EXPECTED_TOOLS = {"start_draft", "get_draft_status", "get_retention", "get_similar_shorts", "get_audience_brief"}
FAKE_KEY = "AIzaFAKEKEY0000000000000000000000000000"
VERDICTS = {"above", "typical", "below"}
LEVELS = {"low", "medium", "high"}
SECURITY_HEADERS = ("content-security-policy", "x-frame-options", "x-content-type-options", "referrer-policy")
PAID_WAIT_S = 360.0
DEFAULT_IDS_CSV = Path(__file__).resolve().parents[2] / "data" / "evals" / "hook_test" / "categories.csv"
YT_ID = re.compile(r"[A-Za-z0-9_-]{11}")


class Run:
    """State of one run: secrets to mask, results received, report lines."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base = args.base_url.rstrip("/")
        self.tokens = {"A": args.token_a, "B": args.token_b}
        self.scratch = Path(args.scratch)
        (self.scratch / "results").mkdir(parents=True, exist_ok=True)
        (self.scratch / "final").mkdir(parents=True, exist_ok=True)
        self.clients: dict[str, Client] = {}
        self.results: list[dict] = []          # every tool result: {file, check, token, tool, arguments, is_error, result}
        self.report: list[dict] = []
        self.upload_ids: list[str] = []
        self.paid_uploads = 0
        self.paid_ok = True                    # False after a paid draft that did not end in done: no further paid draft
        self.videos: list[dict] = []           # summary per real video
        self.form_fields: tuple[str, str] | None = None
        self.seq = 0

    def mask(self, text: object) -> str:
        text = str(text)
        for secret in (*self.tokens.values(), FAKE_KEY):
            if secret:
                text = text.replace(secret, "***")
        for upload_id in self.upload_ids:
            text = text.replace(upload_id, upload_id[:4] + "…")
        return text

    def line(self, check: str, status: str, notes: list[str], problems: list[str] | None = None) -> None:
        problems = problems or []
        detail = self.mask(" | ".join([*(f"PROBLEM: {p}" for p in problems), *notes])).replace("\n", " ")
        self.report.append({"check": check, "status": status, "problems": [self.mask(p) for p in problems],
                            "notes": [self.mask(n) for n in notes]})
        print(f"[{status}] {check:<3} {detail}", flush=True)

    def done(self, check: str, notes: list[str], problems: list[str]) -> None:
        self.line(check, "FAIL" if problems else "PASS", notes, problems)

    async def call(self, check: str, token: str, tool: str, arguments: dict | None = None) -> dict:
        """One tools/call. Returns the parsed JSON result; a protocol error or a non-JSON answer comes back as
        ``{"_protocol_error": ...}`` / ``{"_not_json": ...}`` so that checks can report it."""
        self.seq += 1
        entry = {"check": check, "token": token, "tool": tool, "arguments": arguments or {}, "is_error": None, "result": None}
        try:
            res = await self.clients[token].call_tool(tool, arguments or {}, read_timeout_seconds=120)
            entry["is_error"] = bool(res.is_error)
            text = "".join(getattr(c, "text", "") for c in res.content)
            try:
                entry["result"] = json.loads(text)
            except ValueError:
                entry["result"] = {"_not_json": text[:2000]}
        except Exception as exc:
            entry["result"] = {"_protocol_error": f"{type(exc).__name__}: {self.mask(exc)}"[:500]}
        if not isinstance(entry["result"], dict):
            entry["result"] = {"_not_an_object": entry["result"]}
        entry["file"] = f"{self.seq:03d}_{check}_{token}_{tool}.json"
        (self.scratch / "results" / entry["file"]).write_text(json.dumps(entry, indent=1, ensure_ascii=False))
        self.results.append(entry)
        return entry["result"]

    def is_error(self) -> bool | None:
        return self.results[-1]["is_error"]


# ------------------------------------------------------------------------------------------------------ small helpers

class Page(HTMLParser):
    """Visible text, form fields and every URL-carrying attribute of an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.inputs: list[dict] = []
        self.urls: list[tuple[str, str, str]] = []
        self.scripts = self.script_src = 0
        self.form: dict | None = None
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = dict(attrs)
        if tag in ("style", "script", "title"):
            self._skip += 1
        if tag == "script":
            self.scripts += 1
            self.script_src += "src" in a
        if tag == "input":
            self.inputs.append(a)
        if tag == "form":
            self.form = a
        for name in ("src", "href", "action", "formaction", "poster", "data", "srcset"):
            if a.get(name):
                self.urls.append((tag, name, a[name]))

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script", "title"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.text.append(data.strip())


def parse(html_text: str) -> Page:
    page = Page()
    page.feed(html_text)
    return page


def said(resp: httpx.Response) -> str:
    """What the page says to the user (without the form's labels when a message is there)."""
    if "html" not in resp.headers.get("content-type", ""):
        return resp.text.strip()[:400]
    m = re.search(r'<p class="msg">(.*?)</p>', resp.text, re.S)
    return (re.sub(r"<[^>]+>", "", m.group(1)) if m else " ".join(parse(resp.text).text))[:400]


def walk(value, path: str = "$"):
    """(path, string) for every key and string leaf of a JSON value."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield f"{path}.{k} (key)", str(k)
            yield from walk(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(value, str):
        yield path, value


def snippet(text: str, at: int, width: int = 30) -> str:
    return text[max(0, at - width): at + width].replace("\n", " ")


def num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def ffprobe_duration(path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=30, check=False).stdout.strip()
        return float(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def make_clips(folder: Path) -> dict[str, Path]:
    """The refusal clips: text renamed .mp4, 3 s portrait, 9 s landscape, 9 s valid portrait. Needs ffmpeg."""
    folder.mkdir(parents=True, exist_ok=True)
    clips = {"text": folder / "not_a_video.mp4", "short": folder / "short3.mp4", "landscape": folder / "land9.mp4",
             "valid": folder / "valid9.mp4"}
    clips["text"].write_text("this is a text file, not a video\n" * 40)
    recipes = {"short": ("360x640", "3"), "landscape": ("640x360", "9"), "valid": ("360x640", "9")}
    for name, (size, seconds) in recipes.items():
        if not clips[name].exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30", "-f", "lavfi",
                            "-i", "sine=frequency=440", "-t", seconds, "-pix_fmt", "yuv420p", "-shortest", str(clips[name])],
                           check=True, timeout=120, stdin=subprocess.DEVNULL)
    return clips


def log_text(args: argparse.Namespace) -> str | None:
    if not args.server_log:
        return None
    return Path(args.server_log).read_text(errors="replace")


# ------------------------------------------------------------------------------------------------------------ upload

def upload_id_of(url: str) -> str:
    return urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]


async def fetch_page(url: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as http:
        return await http.get(url)


async def upload(run: Run, url: str, path: Path, own_key: str = "", real_video: bool = False) -> httpx.Response:
    """Multipart POST like a browser: the file field first, then the own-key field (empty when not used)."""
    if real_video:
        if run.args.skip_paid or run.paid_uploads >= 3 or not run.paid_ok:
            raise RuntimeError("refusing to start another paid analysis")
        run.paid_uploads += 1
    file_field, key_field = run.form_fields or ("video", "gemini_key")

    def post() -> httpx.Response:
        with path.open("rb") as fh, httpx.Client(timeout=httpx.Timeout(30, read=180, write=180), follow_redirects=False) as http:
            return http.post(url, files=[(file_field, (path.name, fh, "video/mp4")), (key_field, (None, own_key))])

    return await asyncio.to_thread(post)


# ------------------------------------------------------------------------------------------------------- free checks

async def f1(run: Run) -> None:
    notes, problems = [], []
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "e2e", "version": "0"}}}
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as http:
        for name, auth in (("no token", {}), ("wrong token", {"Authorization": f"Bearer {secrets.token_urlsafe(32)}"})):
            r = await http.post(f"{run.base}/mcp", json=body, headers={**headers, **auth})
            notes.append(f"{name}: {r.status_code}, body {len(r.content)} bytes")
            if r.status_code != 401:
                problems.append(f"{name}: status {r.status_code}, expected 401")
            if r.content:
                problems.append(f"{name}: body is not empty ({len(r.content)} bytes)")
    run.done("F1", notes, problems)


async def connect(run: Run, stack: contextlib.AsyncExitStack, token: str, mode: str = "legacy") -> Client:
    http = create_mcp_http_client(headers={"Authorization": f"Bearer {run.tokens[token]}"})
    await stack.enter_async_context(http)
    client = Client(streamable_http_client(f"{run.base}/mcp", http_client=http), mode=mode, read_timeout_seconds=120,
                    cache=None)
    return await stack.enter_async_context(client)


async def f2(run: Run, stack: contextlib.AsyncExitStack) -> bool:
    notes, problems = [], []
    try:
        for token in ("A", "B"):
            run.clients[token] = await connect(run, stack, token)
    except Exception as exc:
        run.done("F2", notes, [f"initialize failed: {type(exc).__name__}: {exc}"[:400]])
        return False
    client = run.clients["A"]
    info = client.server_info
    notes.append(f"initialize ok (protocol {client.protocol_version}, server {getattr(info, 'name', None)} "
                 f"{getattr(info, 'version', None)})")
    instructions = client.instructions or ""
    notes.append(f"instructions {len(instructions)} chars")
    if not instructions.strip():
        problems.append("server instructions are empty")
    (run.scratch / "results" / "000_instructions.txt").write_text(instructions)
    listed = (await client.list_tools()).tools
    names = [t.name for t in listed]
    notes.append(f"tools: {', '.join(names)}")
    if set(names) != EXPECTED_TOOLS or len(names) != len(EXPECTED_TOOLS):
        problems.append(f"tool list differs from the expected five: {sorted(names)}")
    dumped = []
    for t in listed:
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
        dumped.append({"name": t.name, "description": t.description, "input_schema": schema})
        if not (t.description or "").strip():
            problems.append(f"{t.name}: no description")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            problems.append(f"{t.name}: no object input schema")
    (run.scratch / "results" / "000_tools.json").write_text(json.dumps(dumped, indent=1))
    try:    # informational: what a client that negotiates automatically gets
        async with contextlib.AsyncExitStack() as probe:
            auto = await connect(run, probe, "A", mode="auto")
            notes.append(f"mode=auto also connects (protocol {auto.protocol_version})")
    except Exception as exc:
        notes.append(f"note: mode=auto did not connect ({type(exc).__name__})")
    run.done("F2", notes, problems)
    return True


async def f3(run: Run) -> None:
    notes, problems = [], []
    res = await run.call("F3", "A", "get_retention", {"draft_id": secrets.token_urlsafe(16), "wait_s": 0})
    notes.append(f"result: {json.dumps(res)[:200]}")
    if "_protocol_error" in res or run.is_error():
        problems.append("unknown draft id gave a protocol/tool error instead of a normal result")
    if res.get("status") != "not_found":
        problems.append(f"status is {res.get('status')!r}, expected 'not_found'")
    if not res.get("message"):
        problems.append("no message")
    run.done("F3", notes, problems)


def check_start(run: Run, res: dict, problems: list[str]) -> tuple[str, str]:
    draft_id, url = res.get("draft_id"), res.get("upload_url")
    if not isinstance(draft_id, str) or not draft_id:
        problems.append(f"no draft_id (result: {json.dumps(res)[:200]})")
    if not isinstance(url, str) or not url.startswith(run.base + "/"):
        problems.append(f"upload_url is not under the base URL: {str(url)[:60]!r}")
    else:
        run.upload_ids.append(upload_id_of(url))
    return draft_id or "", url or ""


async def f4(run: Run) -> tuple[str, str]:
    notes, problems = [], []
    res = await run.call("F4", "A", "start_draft")
    draft_id, url = check_start(run, res, problems)
    limits = res.get("limits") or {}
    notes.append(f"limits: {json.dumps(limits)}; expires_in_s {res.get('expires_in_s')}; usage {json.dumps(res.get('usage'))}")
    if limits.get("min_duration_s") != 8 or limits.get("max_duration_s") != 60:
        problems.append("limits do not state 8-60 s")
    if url:
        r = await fetch_page(url)
        notes.append(f"upload page {r.status_code}, {len(r.content)} bytes")
        if r.status_code != 200:
            problems.append(f"upload page status {r.status_code}")
        for h in SECURITY_HEADERS:
            if not r.headers.get(h):
                problems.append(f"missing header {h}")
        if "no-store" not in r.headers.get("cache-control", ""):
            problems.append(f"Cache-Control is {r.headers.get('cache-control')!r}, expected no-store")
        notes.append("headers: " + "; ".join(f"{h}={r.headers.get(h)}" for h in (*SECURITY_HEADERS, "cache-control")))
        page, host = parse(r.text), urlsplit(run.base).netloc
        foreign = [u for u in re.findall(r"https?://[^\s\"'<>)]+", r.text) if urlsplit(u).netloc != host]
        foreign += [f"{tag} {name}={value}" for tag, name, value in page.urls
                    if urlsplit(urljoin(url, value)).netloc != host]
        if foreign:
            problems.append(f"page refers to another host: {foreign[:5]}")
        if page.script_src:
            problems.append(f"{page.script_src} <script src> on the page")
        notes.append(f"{page.scripts} script tags, {len(page.urls)} URL attributes")
        files = [i.get("name") for i in page.inputs if i.get("type") == "file"]
        keys = [i.get("name") for i in page.inputs if i.get("type") == "password"]
        if len(files) == 1 and len(keys) == 1 and files[0] and keys[0]:
            run.form_fields = (files[0], keys[0])
            notes.append(f"form fields: file={files[0]!r}, own key={keys[0]!r}, form method={(page.form or {}).get('method')}")
        else:
            problems.append(f"could not find one file field and one password field (file: {files}, password: {keys})")
    run.done("F4", notes, problems)
    return draft_id, url


async def f5(run: Run, draft_id: str, url: str, clips: dict[str, Path]) -> None:
    notes, problems = [], []
    expect = {"text": None, "short": r"short|8", "landscape": r"vertical|landscape|square|portrait"}
    for name, pattern in expect.items():
        r = await upload(run, url, clips[name])
        sentence = said(r)
        notes.append(f"{name}: {r.status_code} \"{sentence}\"")
        if not 400 <= r.status_code < 500:
            problems.append(f"{name}: status {r.status_code}, expected a 4xx refusal")
        if len(sentence.split()) < 4:
            problems.append(f"{name}: no sentence in the refusal")
        if pattern and not re.search(pattern, sentence, re.I):
            problems.append(f"{name}: the refusal does not say why (expected /{pattern}/)")
        again = await fetch_page(url)
        if again.status_code != 200:
            problems.append(f"{name}: the link is not valid any more after the refusal (GET {again.status_code})")
        status = await run.call("F5", "A", "get_draft_status", {"draft_id": draft_id})
        usage = status.get("usage") or {}
        if status.get("status") != "waiting_for_upload":
            problems.append(f"{name}: status after refusal is {status.get('status')!r}")
        if usage.get("drafts_today") != 0:
            problems.append(f"{name}: usage after refusal: {json.dumps(usage)}")
    notes.append(f"after all three: status {status.get('status')}, usage {json.dumps(usage)}")
    run.done("F5", notes, problems)


async def f6(run: Run, draft_id: str) -> None:
    notes, problems = [], []
    random_id = secrets.token_urlsafe(16)
    for tool, extra in (("get_draft_status", {}), ("get_retention", {"wait_s": 0})):
        foreign = await run.call("F6", "B", tool, {"draft_id": draft_id, **extra})
        unknown = await run.call("F6", "B", tool, {"draft_id": random_id, **extra})
        notes.append(f"{tool}: {json.dumps(foreign)[:120]}")
        if foreign != unknown:
            problems.append(f"{tool}: foreign draft answer differs from the answer for a random id")
        if foreign.get("status") != "not_found":
            problems.append(f"{tool}: status {foreign.get('status')!r} for another token's draft")
    own = await run.call("F6", "A", "get_draft_status", {"draft_id": draft_id})
    if own.get("status") != "waiting_for_upload":
        problems.append(f"the owner no longer sees the draft as waiting: {own.get('status')!r}")
    run.done("F6", notes, problems)


async def f7(run: Run) -> None:
    if run.args.warmup_wait > 0:
        await asyncio.sleep(run.args.warmup_wait)
        return run.line("F7", "PASS", [f"slept {run.args.warmup_wait:g} s (--warmup-wait); the warm-up itself is not visible "
                                       "to a client before a job runs (jobs show step 'starting' until it is ready)"])
    run.line("F7", "SKIP", ["the warm-up is not observable from outside before a job runs (a job reports step 'starting' "
                            "until the library is ready); proceeding"])


# ------------------------------------------------------------------------------------------------------- paid checks

async def wait_for_end(run: Run, check: str, draft_id: str, max_s: float = PAID_WAIT_S) -> tuple[dict, list[str], float]:
    """Poll get_draft_status and get_retention until the draft is done or failed, or ``max_s`` passed."""
    start, steps, res = time.monotonic(), [], {}
    while True:
        status = await run.call(check, "A", "get_draft_status", {"draft_id": draft_id})
        label = status.get("status", "?") + (f":{status['step']}" if status.get("step") else "")
        if not steps or steps[-1][1] != label:
            steps.append((time.monotonic() - start, label))
        left = max_s - (time.monotonic() - start)
        res = await run.call(check, "A", "get_retention", {"draft_id": draft_id, "wait_s": int(max(0, min(10, left)))})
        if res.get("status") in ("done", "failed", "not_found", "expired") or "_protocol_error" in res:
            break
        if res.get("status") == "waiting_for_upload" and time.monotonic() - start > 30:
            break
        if time.monotonic() - start >= max_s:
            break
    elapsed = time.monotonic() - start
    steps.append((elapsed, f"get_retention:{res.get('status', '?')}"))
    return res, [f"{t:.0f}s {s}" for t, s in steps], elapsed


def validate_retention(res: dict, file_duration: float | None, problems: list[str]) -> dict:
    """Shape check of a finished get_retention result. Returns the summary used in the report."""
    out: dict = {}
    duration = res.get("duration_s")
    if not num(duration):
        problems.append("no duration_s in the result")
        duration = file_duration
    elif file_duration is not None and abs(duration - file_duration) > 1.0:
        problems.append(f"duration_s {duration} differs from the file's {file_duration:.2f}")
    out["duration_s"] = duration
    segments = res.get("segments")
    if not isinstance(segments, dict) or set(segments) != {"early", "mid", "late"}:
        problems.append(f"segments are not exactly early/mid/late: {list(segments) if isinstance(segments, dict) else segments}")
        segments = segments if isinstance(segments, dict) else {}
    out["verdicts"] = {}
    for name, seg in segments.items():
        seg = seg if isinstance(seg, dict) else {}
        out["verdicts"][name] = seg.get("verdict")
        if seg.get("verdict") not in VERDICTS:
            problems.append(f"segment {name}: verdict {seg.get('verdict')!r}")
        why = seg.get("why")
        if not why or (isinstance(why, list) and not all(isinstance(w, str) and w.strip() for w in why)):
            problems.append(f"segment {name}: empty why")
        if not isinstance(seg.get("evidence"), str) or not seg["evidence"].strip():
            problems.append(f"segment {name}: no evidence text")
    drops = res.get("drop_moments")
    if not isinstance(drops, list) or len(drops) != 2:
        problems.append(f"drop_moments: expected exactly two, got {len(drops) if isinstance(drops, list) else drops!r}")
        drops = drops if isinstance(drops, list) else []
    out["drop_moments"] = drops
    for d in drops:
        d = d if isinstance(d, dict) else {}
        if not num(d.get("t_s")) or (duration is not None and not 0 <= d["t_s"] <= duration):
            problems.append(f"drop moment t_s {d.get('t_s')!r} is not inside 0-{duration}")
        if not isinstance(d.get("reason"), str) or not d["reason"].strip():
            problems.append("drop moment without a reason")
    out["rewatch_level"] = res.get("rewatch_level")
    if res.get("rewatch_level") not in LEVELS:
        problems.append(f"rewatch_level {res.get('rewatch_level')!r}")
    out["advice"] = res.get("advice")
    advice = res.get("advice")
    if not isinstance(advice, list) or not advice or not all(isinstance(a, str) and a.strip() for a in advice):
        problems.append("advice is not a list of sentences")
    curve = res.get("curve")
    if not isinstance(curve, dict):
        problems.append("no curve")
    else:
        lists = {k: curve.get(k) for k in ("viewers_per_100", "lower", "upper")}
        if not all(isinstance(v, list) and v and all(num(x) for x in v) for v in lists.values()):
            problems.append(f"curve lacks numeric viewers_per_100 / lower / upper (keys: {list(curve)})")
        elif not len(lists["lower"]) == len(lists["upper"]) == len(lists["viewers_per_100"]):
            problems.append("curve lists have different lengths")
        else:
            bad = [i for i, (lo, mid, hi) in enumerate(zip(lists["lower"], lists["viewers_per_100"], lists["upper"]))
                   if not lo <= mid <= hi]
            if bad:
                problems.append(f"curve: value outside its own band at indexes {bad[:6]}")
            out["curve_first_last"] = [lists["viewers_per_100"][0], lists["viewers_per_100"][-1]]
    if not res.get("caveats"):
        problems.append("no caveats")
    out["cost_usd"] = res.get("cost_usd")
    if not num(res.get("cost_usd")):
        problems.append(f"no numeric cost_usd ({res.get('cost_usd')!r})")
    return out


def validate_similar(res: dict, k: int, problems: list[str]) -> list[dict]:
    items = res.get("similar_past_shorts")
    if res.get("status") != "done" or not isinstance(items, list):
        problems.append(f"get_similar_shorts: {json.dumps(res)[:200]}")
        return []
    if len(items) != k:
        problems.append(f"get_similar_shorts: {len(items)} items, expected {k}")
    sims = []
    for i, item in enumerate(items):
        if not re.fullmatch(r"past Short \d+", str(item.get("label"))):
            problems.append(f"similar[{i}]: label {item.get('label')!r}")
        sim = item.get("similarity")
        if not num(sim) or not 0 <= sim <= 1:
            problems.append(f"similar[{i}]: similarity {sim!r}")
        sims.append(sim if num(sim) else -1)
        points = item.get("viewers_per_100")
        if not isinstance(points, dict) or len(points) != 5 or not all(num(v) for v in points.values()):
            problems.append(f"similar[{i}]: viewers_per_100 is not five numeric checkpoints: {points!r}")
        comments = next((v for key, v in item.items() if "comment" in key), None)
        if not isinstance(comments, list):
            problems.append(f"similar[{i}]: no comment list")
        else:
            for c in comments:
                if not isinstance(c, dict) or set(c) != {"text", "likes"}:
                    problems.append(f"similar[{i}]: a comment has keys other than text and likes: {sorted(c) if isinstance(c, dict) else c!r}")
                    break
    if sims != sorted(sims, reverse=True):
        problems.append(f"similarities are not in descending order: {sims}")
    return [{"label": it.get("label"), "similarity": it.get("similarity"), "duration_s": it.get("duration_s"),
             "rewatch_level": it.get("rewatch_level"), "like_rate_tier": it.get("like_rate_tier"),
             "n_comments": len(next((v for key, v in it.items() if "comment" in key and isinstance(v, list)), []))}
            for it in items]


def validate_brief(res: dict, problems: list[str]) -> dict:
    if res.get("status") != "done":
        problems.append(f"get_audience_brief: {json.dumps(res)[:200]}")
        return {}
    field = lambda part: next((k for k in res if part in k), None)
    description, hints = field("draft_description"), field("search_hints")
    if not description or not res.get(description):
        problems.append("audience brief: no draft description")
    if not hints:
        problems.append("audience brief: no search hints field")
    viewers = res.get("your_viewers")
    groups = viewers.get("groups") if isinstance(viewers, dict) else viewers
    counts = {}
    if not isinstance(groups, list) or not groups:
        problems.append(f"audience brief: your_viewers has no groups ({json.dumps(viewers)[:120]})")
    else:
        for g in groups:
            if not isinstance(g, dict) or not num(g.get("count")):
                problems.append(f"audience brief: a group without a count: {json.dumps(g)[:100]}")
            else:
                counts[str(g.get("type"))] = g["count"]
    for key in ("writing_rules", "untrusted_note"):
        if not isinstance(res.get(key), str) or not res[key].strip():
            problems.append(f"audience brief: no {key}")
    return {"search_hints": res.get(hints) if hints else None, "group_counts": counts,
            "n_description_lines": len(res.get(description) or []) if description else 0,
            "n_comments": viewers.get("n_comments") if isinstance(viewers, dict) else None,
            "n_past_shorts_with_comments": viewers.get("n_past_shorts_with_comments") if isinstance(viewers, dict) else None}


def forbidden_ids(run: Run) -> dict[str, str]:
    ids = {Path(v).stem: "test video id" for v in run.args.videos or [] if YT_ID.fullmatch(Path(v).stem)}
    path = Path(run.args.library_ids_csv)
    if path.is_file():
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                vid = (row.get("video_id") or "").strip()
                if YT_ID.fullmatch(vid):
                    ids.setdefault(vid, "library video id")
    return ids


def scan(run: Run, check: str, start: int = 0) -> None:
    """Search every tool result received so far (from index ``start``) for things that must not leave the server."""
    ids = forbidden_ids(run)
    plain = {"/Users/": "local path", "/tmp/": "local path", "data/": "data path", "GEMINI": "the word GEMINI",
             FAKE_KEY: "own key"}
    plain.update({t: f"token {name}" for name, t in run.tokens.items() if t})
    hits = []
    for entry in run.results[start:]:
        for path, text in walk(entry["result"]):
            for needle, what in plain.items():
                if (at := text.find(needle)) >= 0:
                    hits.append(f"{entry['file']} {path}: {what} near \"{snippet(text, at)}\"")
            for vid, what in ids.items():
                if (at := text.find(vid)) >= 0:
                    hits.append(f"{entry['file']} {path}: {what} near \"{snippet(text, at)}\"")
            if ("comment" in path or "your_viewers" in path) and not path.endswith("(key)"):
                if (m := re.search(r"@\w", text)):
                    hits.append(f"{entry['file']} {path}: @handle near \"{snippet(text, m.start())}\"")
    n = len(run.results) - start
    notes = [f"scanned {n} tool results against {len(ids)} video ids (library list "
             f"{'read' if Path(run.args.library_ids_csv).is_file() else 'NOT available'}), paths, tokens, GEMINI, @handles"]
    run.done(check, notes, hits[:40] + ([f"... and {len(hits) - 40} more"] if len(hits) > 40 else []))


async def paid_draft(run: Run, n: int, check: str, video: Path, draft_id: str, url: str, used_before: int) -> None:
    """Upload one real video, wait for the result, validate it, fetch similar Shorts and the brief."""
    notes, problems = [], []
    summary = {"video": video.name, "check": check}
    run.videos.append(summary)
    file_duration = ffprobe_duration(video)
    t0 = time.monotonic()
    r = await upload(run, url, video, real_video=True)
    summary["upload_s"] = round(time.monotonic() - t0, 1)
    notes.append(f"upload {r.status_code} in {summary['upload_s']} s: \"{said(r)}\"")
    if r.status_code != 200:
        run.paid_ok = False
        problems.append("the upload was not accepted; the paid part stops here")
        return run.done(check, notes, problems)
    res, steps, waited = await wait_for_end(run, check, draft_id)
    summary["wall_s"] = round(time.monotonic() - t0, 1)
    summary["steps"] = steps
    notes.append(f"steps: {' > '.join(steps)}; upload to done {summary['wall_s']} s")
    if res.get("status") != "done":
        run.paid_ok = False
        problems.append(f"draft did not end in done: {json.dumps(res)[:300]}; the paid part stops here")
        return run.done(check, notes, problems)
    (run.scratch / "final" / f"video{n}_get_retention.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    summary.update(validate_retention(res, file_duration, problems))
    notes.append(f"cost ${summary.get('cost_usd')}; verdicts {json.dumps(summary.get('verdicts'))}; drops at "
                 f"{[d.get('t_s') for d in summary.get('drop_moments', []) if isinstance(d, dict)]} of {summary.get('duration_s')} s; "
                 f"rewatch {summary.get('rewatch_level')}")
    run.done(check, notes, problems)

    notes, problems = [], []
    sub = f"{check}b" if check != "P1" else "P2"
    similar = await run.call(sub, "A", "get_similar_shorts", {"draft_id": draft_id, "k": 5})
    (run.scratch / "final" / f"video{n}_get_similar_shorts.json").write_text(json.dumps(similar, indent=1, ensure_ascii=False))
    summary["similar"] = validate_similar(similar, 5, problems)
    brief = await run.call(sub, "A", "get_audience_brief", {"draft_id": draft_id})
    (run.scratch / "final" / f"video{n}_get_audience_brief.json").write_text(json.dumps(brief, indent=1, ensure_ascii=False))
    summary["brief"] = validate_brief(brief, problems)
    notes.append("similar: " + ", ".join(f"{s['label']} {s['similarity']}" for s in summary["similar"]))
    notes.append(f"search hints {json.dumps(summary['brief'].get('search_hints'))}; groups {json.dumps(summary['brief'].get('group_counts'))}")
    notes.append(f"sizes: retention {len(json.dumps(res))} B, similar {len(json.dumps(similar))} B, brief {len(json.dumps(brief))} B")
    run.done(sub, notes, problems)

    scan(run, "P3" if check == "P1" else f"{check}c")

    notes, problems = [], []
    gone = await fetch_page(url)
    status = await run.call(check, "A", "get_draft_status", {"draft_id": draft_id})
    usage = status.get("usage") or {}
    notes.append(f"used link GET {gone.status_code}; status {status.get('status')}; usage {json.dumps(usage)}")
    if gone.status_code != 404:
        problems.append(f"the used upload link answers {gone.status_code}, expected 404")
    if status.get("status") != "done":
        problems.append(f"get_draft_status says {status.get('status')!r}")
    if usage.get("drafts_today") != used_before + 1 or usage.get("drafts_left_today") != 3 - (used_before + 1):
        problems.append(f"usage is not {used_before + 1} used / {2 - used_before} left")
    run.done("P4" if check == "P1" else f"{check}d", notes, problems)


async def start_for(run: Run, check: str, problems: list[str], token: str = "A") -> tuple[str, str, dict]:
    res = await run.call(check, token, "start_draft")
    draft_id, url = check_start(run, res, problems)
    return draft_id, url, res


# ---------------------------------------------------------------------------------------------- limit and end checks

async def l1(run: Run, clips: dict[str, Path]) -> None:
    notes, problems = [], []
    before = len(log_text(run.args) or "")
    draft_id, url, res = await start_for(run, "L1", problems)
    notes.append(f"start_draft says: \"{str(res.get('message'))[-230:]}\"; usage {json.dumps(res.get('usage'))}")
    if not url:
        return run.done("L1", notes, problems)
    r = await upload(run, url, clips["valid"])
    sentence = said(r)
    notes.append(f"upload {r.status_code} \"{sentence}\"")
    if r.status_code == 200:
        problems.append("the 4th upload of the day was ACCEPTED")
    if not re.search(r"limit|used", sentence, re.I) or not re.search(r"reset", sentence, re.I):
        problems.append("the refusal does not name the daily limit and when it resets")
    for pause in (0, 3):
        await asyncio.sleep(pause)
        status = await run.call("L1", "A", "get_draft_status", {"draft_id": draft_id})
        if status.get("status") != "waiting_for_upload":
            problems.append(f"status after the refused upload is {status.get('status')!r}")
    notes.append(f"status {status.get('status')}; usage {json.dumps(status.get('usage'))}")
    log = log_text(run.args)
    if log is None:
        notes.append("server log not given: 'no job ran' checked through get_draft_status only")
    else:
        new = [ln for ln in log[before:].splitlines() if "route=job" in ln or "accepted" in ln]
        refusal = [ln for ln in log[before:].splitlines() if "limit" in ln]
        notes.append(f"log since L1 began: {len(new)} job/accepted lines; refusal line: {refusal[:1]}")
        if new:
            problems.append(f"the server log shows a job after the refused upload: {new[:2]}")
    run.done("L1", notes, problems)


async def l2(run: Run, clips: dict[str, Path]) -> None:
    notes, problems = [], []
    draft_id, url, res = await start_for(run, "L2", problems)
    if not url:
        return run.done("L2", notes, problems)
    r = await upload(run, url, clips["valid"], own_key=FAKE_KEY)
    notes.append(f"upload with own key: {r.status_code} \"{said(r)}\"")
    if r.status_code != 200:
        problems.append("the own-key upload was not accepted")
        return run.done("L2", notes, problems)
    final, steps, waited = await wait_for_end(run, "L2", draft_id)
    notes.append(f"steps: {' > '.join(steps)}")
    notes.append(f"get_retention says: status {final.get('status')!r}, reason \"{final.get('reason')}\", message \"{final.get('message')}\"")
    if final.get("status") != "failed":
        problems.append(f"the draft on a fake key ended as {final.get('status')!r}, expected failed")
    elif not re.search(r"key", str(final.get("reason")), re.I):
        notes.append("NOTE: the failure sentence is generic; it does not say the user's own key was rejected")
    usage = final.get("usage") or {}
    notes.append(f"usage {json.dumps(usage)}")
    where = [e["file"] for e in run.results if FAKE_KEY in json.dumps(e["result"])]
    if where:
        problems.append(f"the fake key appears in tool results: {where}")
    log = log_text(run.args)
    if log is None:
        notes.append("server log not given: key-in-log check skipped")
    elif FAKE_KEY in log or FAKE_KEY[:12] in log:
        problems.append("the fake key appears in the server log")
    else:
        notes.append("key not in the server log")
    if run.args.data_dir:
        found = []
        for p in sorted(Path(run.args.data_dir).rglob("*")):
            if p.is_file() and FAKE_KEY.encode() in p.read_bytes():
                found.append(p.name)
        if found:
            problems.append(f"the fake key is in the server's data files: {found}")
        else:
            notes.append("key not in any file of the data dir (SQLite, WAL, uploads), read as bytes")
    else:
        notes.append("data dir not given: SQLite byte search skipped")
    run.done("L2", notes, problems)


async def l3(run: Run) -> None:
    notes, problems = [], []
    draft_id, url, res = await start_for(run, "L3", problems, token="B")
    usage = res.get("usage") or {}
    notes.append(f"token B start_draft: status {res.get('status')}, usage {json.dumps(usage)}")
    if res.get("status") != "waiting_for_upload":
        problems.append(f"token B cannot start a draft: {json.dumps(res)[:200]}")
    if usage.get("drafts_today") != 0 or usage.get("drafts_left_today") != 3:
        problems.append("token B's allowance is not untouched")
    run.done("L3", notes, problems)


async def c1(run: Run) -> None:
    notes, problems = [], []
    if run.args.data_dir:
        deadline = time.monotonic() + 15
        uploads = Path(run.args.data_dir) / "uploads"
        while (left := [p.name for p in uploads.iterdir()] if uploads.is_dir() else []) and time.monotonic() < deadline:
            await asyncio.sleep(1)
        notes.append(f"upload dir holds {len(left)} files")
        if left:
            problems.append(f"{len(left)} files are left in the upload directory")
    else:
        notes.append("data dir not given: upload directory not checked")
    log = log_text(run.args)
    if log is None:
        notes.append("server log not given: log search skipped")
    else:
        needles = {f"token {k}": v for k, v in run.tokens.items() if v}
        needles.update({f"upload id #{i + 1}": u for i, u in enumerate(run.upload_ids)})
        needles["AIza"] = "AIza"
        needles.update({f"file name of video {i + 1}": Path(v).stem for i, v in enumerate(run.args.videos or [])})
        needles.update({"/Users/": "/Users/", "Traceback": "Traceback"})
        hit = [name for name, needle in needles.items() if needle and needle in log]
        notes.append(f"log: {len(log.splitlines())} lines searched for {len(needles)} strings (tokens, {len(run.upload_ids)} "
                     "upload ids, AIza, video file names, /Users/, Traceback)")
        if hit:
            problems.append(f"found in the server log: {hit}")
        odd = [ln for ln in log.splitlines() if " error " in ln or "warning" in ln or "status=5" in ln]
        if odd:
            notes.append(f"{len(odd)} warning/error lines in the log, first: {odd[0][:200]}")
    run.done("C1", notes, problems)


def c2(run: Run) -> None:
    notes = []
    costs = [v.get("cost_usd") for v in run.videos if num(v.get("cost_usd"))]
    notes.append(f"cost reported by get_retention across {len(costs)} real drafts: ${sum(costs):.4f} ({costs})")
    db = Path(run.args.data_dir) / "server.sqlite3" if run.args.data_dir else None
    if db and db.is_file():
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
            try:
                spend = con.execute("SELECT day, kind, COUNT(*), ROUND(SUM(usd), 4) FROM spend GROUP BY day, kind ORDER BY 1, 2").fetchall()
                drafts = con.execute("SELECT status, paid_by, COUNT(*), ROUND(COALESCE(SUM(cost_usd), 0), 4) FROM drafts "
                                     "GROUP BY status, paid_by ORDER BY 1, 2").fetchall()
            finally:
                con.close()
            notes.append("spend table (day, kind, rows, usd): " + "; ".join(str(tuple(r)) for r in spend))
            notes.append("drafts table (status, paid_by, n, cost): " + "; ".join(str(tuple(r)) for r in drafts))
        except sqlite3.Error as exc:
            notes.append(f"could not read the server database: {type(exc).__name__}")
    else:
        notes.append("quota summary skipped (needs --data-dir; the server has no tool for it)")
    run.line("C2", "PASS" if (costs or run.args.skip_paid) else "SKIP", notes)


# -------------------------------------------------------------------------------------------------------------- main

async def main_async(run: Run) -> None:
    args = run.args
    clips = make_clips(run.scratch / "clips")
    await f1(run)
    async with contextlib.AsyncExitStack() as stack:
        if not await f2(run, stack):
            return
        await f3(run)
        draft_id, url = await f4(run)
        if not (draft_id and url):
            return run.line("F5", "FAIL", [], ["no draft from F4; stopping"])
        await f5(run, draft_id, url, clips)
        await f6(run, draft_id)
        await f7(run)
        if args.skip_paid:
            for check in ("P1", "P2", "P3", "P4", "P5", "P6"):
                run.line(check, "SKIP", ["--skip-paid"])
            run.line("L1", "SKIP", ["--skip-paid: the '4th draft of the day' cannot be tested without three real drafts"])
        else:
            videos = [Path(v) for v in args.videos]
            await paid_draft(run, 1, "P1", videos[0], draft_id, url, used_before=0)
            for n, check in ((2, "P5"), (3, "P6")):
                if not run.paid_ok:
                    run.line(check, "SKIP", ["an earlier paid draft did not finish; no further paid draft is started"])
                    continue
                problems: list[str] = []
                d, u, _ = await start_for(run, check, problems)
                if not (d and u):
                    run.paid_ok = False
                    run.done(check, [], problems)
                    continue
                await paid_draft(run, n, check, videos[n - 1], d, u, used_before=n - 1)
            if run.paid_uploads == 3 and run.paid_ok:
                await l1(run, clips)
            else:
                run.line("L1", "SKIP", ["fewer than three real drafts finished, so there is no '4th draft of the day'"])
        await l2(run, clips)
        await l3(run)
        scan(run, "S1")
        await c1(run)
        c2(run)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8791 or the public https address (no path)")
    ap.add_argument("--token-a", default=os.environ.get("SO_E2E_TOKEN_A"), help="default: $SO_E2E_TOKEN_A")
    ap.add_argument("--token-b", default=os.environ.get("SO_E2E_TOKEN_B"), help="default: $SO_E2E_TOKEN_B")
    ap.add_argument("--videos", nargs=3, metavar="VIDEO", help="three real videos (8-60 s, vertical); needed without --skip-paid")
    ap.add_argument("--scratch", required=True, help="directory for results and generated clips")
    ap.add_argument("--skip-paid", action="store_true", help="only the free checks (F, L2, L3, C)")
    ap.add_argument("--server-log", help="the server's log file, when it runs on this machine")
    ap.add_argument("--data-dir", help="the server's SO_DATA_DIR, when it runs on this machine")
    ap.add_argument("--library-ids-csv", default=str(DEFAULT_IDS_CSV), help="CSV with a video_id column (leak scan)")
    ap.add_argument("--warmup-wait", type=float, default=0.0, help="seconds to sleep at F7")
    args = ap.parse_args()
    if not args.token_a or not args.token_b:
        ap.error("two tokens are needed (--token-a/--token-b or SO_E2E_TOKEN_A/SO_E2E_TOKEN_B)")
    if not args.skip_paid:
        if not args.videos:
            ap.error("--videos is required without --skip-paid")
        missing = [v for v in args.videos if not Path(v).is_file()]
        if missing or len(set(args.videos)) != 3:
            ap.error(f"--videos must be three different existing files (missing: {len(missing)})")
    if not shutil.which("ffmpeg"):
        ap.error("ffmpeg is needed to generate the refusal clips")
    run = Run(args)
    print(f"e2e check against {run.base} ({'free checks only' if args.skip_paid else 'FULL RUN: three paid analyses'})", flush=True)
    try:
        asyncio.run(main_async(run))
    except KeyboardInterrupt:
        run.line("RUN", "FAIL", [], ["interrupted"])
    except BaseException as exc:
        run.line("RUN", "FAIL", [], [f"the script stopped: {type(exc).__name__}: {exc}"[:500]])
    failed = [r["check"] for r in run.report if r["status"] == "FAIL"]
    (run.scratch / "report.json").write_text(json.dumps(
        {"base_url": run.base, "skip_paid": args.skip_paid, "checks": run.report, "videos": run.videos,
         "paid_uploads": run.paid_uploads}, indent=1, ensure_ascii=False))
    print(f"{len(run.report)} checks: {sum(r['status'] == 'PASS' for r in run.report)} PASS, {len(failed)} FAIL, "
          f"{sum(r['status'] == 'SKIP' for r in run.report)} SKIP{'; failed: ' + ', '.join(failed) if failed else ''}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
