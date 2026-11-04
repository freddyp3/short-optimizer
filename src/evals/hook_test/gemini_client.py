"""Gemini client for the hook test: upload cache, generate wrapper, retry, usage+cost.

Free tier target ($0): ``tier`` comes from config/``.env`` (``GEMINI_TIER``);
``cost_usd_actual`` is 0 on free while ``cost_usd_paid_equiv`` records what the
call would cost at paid prices. Rate limits: 429s honour any retry delay found
in the error, else exponential backoff (cap ~120 s); a per-day-quota error or
429s persisting past MAX_RETRIES raises ``DailyQuotaExceeded`` so the run stops
cleanly and can be resumed with the same ``--run-id``.

SDK notes (google-genai 2.32.0, verified in types.py):
- per-video fps: ``Part(file_data=..., video_metadata=types.VideoMetadata(fps=5.0))``
- per-part media resolution: ``PartMediaResolution(level=PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH)``
- Files API: ``client.files.upload(file=path, config={"display_name": ...})``,
  poll ``client.files.get(name=...)`` until ``file.state == "ACTIVE"``; ``file.uri``
  feeds ``FileData.file_uri``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.evals.hook_test import config as C
from src.evals.hook_test.schema import validate_result

log = logging.getLogger(__name__)

DAILY_QUOTA_RE = re.compile(r"per.?day|daily|PerDay|RPD|RequestsPerDay", re.I)
RETRY_DELAY_RE = re.compile(r"retryDelay\"?\s*[:=]\s*\"?(\d+(?:\.\d+)?)s", re.I)


class DailyQuotaExceeded(Exception):
    """Daily quota hit or 429s past MAX_RETRIES; rerun with --run-id to resume."""


@dataclass
class CallResult:
    record: dict = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def nominal_cost(model: str, prompt_tokens: int, output_tokens: int) -> float:
    """Paid-equivalent USD for one call."""
    pin, pout = C.PRICES_USD_PER_M.get(model, (0.0, 0.0))
    return prompt_tokens * pin / 1e6 + output_tokens * pout / 1e6


def estimate_call_cost(model: str, duration_s: float, fps: float = C.FPS) -> float:
    """Nominal cost estimate when count_tokens is unavailable (plan 4.1 token model)."""
    tin = duration_s * fps * C.ASSUMED_TOKENS_PER_FRAME + duration_s * C.AUDIO_TOKENS_PER_S \
        + C.PROMPT_TOKENS_EST
    return nominal_cost(model, int(tin), C.OUTPUT_TOKENS_EST)


def is_rate_limit(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)


def is_daily_quota(e: Exception) -> bool:
    blob = str(e) + json.dumps(getattr(e, "details", {}) or {}, default=str)
    return DAILY_QUOTA_RE.search(blob) is not None


def retry_delay_s(e: Exception, attempt: int) -> float:
    blob = str(e) + json.dumps(getattr(e, "details", {}) or {}, default=str)
    m = RETRY_DELAY_RE.search(blob)
    if m:
        return min(float(m.group(1)), 120.0)
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", blob, re.I)
    if m:
        return min(float(m.group(1)), 120.0)
    return min(5.0 * 2**attempt, 120.0)


def extract_text(resp) -> str:
    """JSON response text, excluding thought-summary parts.

    In 2.32.0 ``resp.text`` already skips parts with ``part.thought == True``
    (GenerateContentResponse._get_text); we filter again in the fallback path
    so a mocked/edge response can't leak the scratch-pad summary into the JSON.
    """
    t = getattr(resp, "text", None)
    if t:
        return t
    try:
        return "".join(
            getattr(p, "text", "") or ""
            for c in resp.candidates for p in c.content.parts
            if getattr(p, "thought", None) is not True)
    except Exception:
        return ""


def extract_thought_summary(resp) -> str:
    """Joined thought-summary parts (include_thoughts=True)."""
    try:
        return "\n".join(
            getattr(p, "text", "") or ""
            for c in resp.candidates for p in c.content.parts
            if getattr(p, "thought", None) is True)
    except Exception:
        return ""


def extract_usage(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return {}
    g = lambda k: getattr(u, k, None) or 0
    return {
        "prompt_tokens": g("prompt_token_count"),
        "response_tokens": g("candidates_token_count") or g("response_token_count"),
        "thinking_tokens": g("thoughts_token_count"),
        "total_tokens": g("total_token_count"),
    }


def new_genai_client(api_key: str):
    """Import-time-free construction so tests never need the SDK or a key."""
    from google import genai
    from google.genai import types
    # SDK-level retries off: _call_with_retry owns retry timing (each retry spends free-tier quota)
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1)))


class GeminiHookTester:
    """Uploads blind files once, runs generate_content, retries on 429/5xx."""

    def __init__(self, client=None, api_key: str | None = None,
                 model: str = C.DEFAULT_MODEL, tier: str = C.GEMINI_TIER,
                 uploads_path: Path = C.UPLOADS_JSON, sleep=time.sleep,
                 poll_sleep: float = 5.0):
        self.client = client or new_genai_client(api_key)
        self.model = model
        self.tier = tier
        self.uploads_path = Path(uploads_path)
        self.sleep = sleep
        self.poll_sleep = poll_sleep
        self._uploads = json.loads(self.uploads_path.read_text()) \
            if self.uploads_path.exists() else {}

    # ------------------------------------------------------------- uploads

    def _save_uploads(self):
        self.uploads_path.parent.mkdir(parents=True, exist_ok=True)
        self.uploads_path.write_text(json.dumps(self._uploads, indent=2))

    def upload_video(self, blind_id: str, path: Path):
        """Files API upload with display_name=blind_id; cached in uploads.json.

        Re-uploads when the cache entry is >UPLOAD_TTL_H old or the file is
        gone on the server. Polls until state ACTIVE.
        """
        from google.genai import types  # only needed on the real path
        path = Path(path)
        sha = sha256_file(path)
        ent = self._uploads.get(blind_id)
        fresh = (ent and ent.get("sha256") == sha and ent.get("file_name")
                 and (datetime.now(timezone.utc).timestamp()
                      - ent.get("uploaded_at", 0)) < C.UPLOAD_TTL_H * 3600)
        f = None
        if fresh:
            try:
                f = self.client.files.get(name=ent["file_name"])
            except Exception:
                f = None
        if f is None:
            log.info("uploading %s (%s)", path.name, blind_id)
            f = self.client.files.upload(
                file=path, config=types.UploadFileConfig(display_name=blind_id))
            self._uploads[blind_id] = {
                "file_name": f.name, "uploaded_at": datetime.now(timezone.utc).timestamp(),
                "sha256": sha}
            self._save_uploads()
        for _ in range(120):  # ~10 min worst case at poll_sleep=5
            state = getattr(f, "state", None)
            if str(state).endswith("ACTIVE") or state == "ACTIVE":
                return f
            self.sleep(self.poll_sleep)
            f = self.client.files.get(name=f.name)
        raise TimeoutError(f"file {f.name} not ACTIVE after polling")

    # ------------------------------------------------------------- request

    def _build_contents(self, file_obj, prompt_text: str, fps: float,
                        media_resolution: str):
        """[video part, prompt text] only — no title/metadata (blinding rule 5.1)."""
        from google.genai import types
        video = types.Part(
            file_data=types.FileData(
                file_uri=getattr(file_obj, "uri", None) or file_obj.name,
                mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=fps),
            media_resolution=types.PartMediaResolution(level=media_resolution),
        )
        return [video, prompt_text]

    def _generate(self, contents, response_schema):
        from google.genai import types
        return self.client.models.generate_content(
            model=self.model, contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ))

    def analyze(self, blind_id: str, path: Path, prompt_text: str,
                duration_s: float, fps: float = C.FPS,
                media_resolution: str = C.MEDIA_RESOLUTION,
                prompt_version: str | None = None,
                prompt_sha256: str | None = None) -> dict:
        """One analyze call (with retries). Returns the raw-record dict."""
        schema_version = C.PROMPT_SCHEMAS.get(prompt_version, "v1")
        file_obj = self.upload_video(blind_id, path)
        contents = self._build_contents(file_obj, prompt_text, fps, media_resolution)

        prompt_meta = {"prompt_version": prompt_version, "prompt_sha256": prompt_sha256}
        t0 = time.time()
        resp = self._call_with_retry(contents, schema_version)
        rec = self._record(resp, blind_id, fps, media_resolution, duration_s,
                         latency=time.time() - t0, schema_version=schema_version)
        rec.update(prompt_meta)
        if rec["status"] == "invalid":
            # one retry on schema-invalid output (plan section 4)
            t0 = time.time()
            resp = self._call_with_retry(contents, schema_version)
            rec2 = self._record(resp, blind_id, fps, media_resolution, duration_s,
                                latency=time.time() - t0, schema_version=schema_version)
            rec2.update(prompt_meta)
            rec2["retried_after_invalid"] = True
            if rec2["status"] == "invalid":
                rec2["first_attempt"] = rec
                return rec2
            rec2["first_attempt_invalid"] = rec
            return rec2
        return rec

    def _schema(self, schema_version: str):
        """Response schema for the call. Overridable (analyze_video uses its own)."""
        from src.evals.hook_test.schema import schema_class
        return schema_class(schema_version)

    def _validate(self, data, duration_s: float, schema_version: str = "v1"):
        """Validate parsed JSON. Overridable; returns (result, reason, flags)."""
        return validate_result(data, duration_s, schema_version)

    def _call_with_retry(self, contents, schema_version="v1"):
        for attempt in range(C.MAX_RETRIES + 1):
            try:
                return self._generate(contents, self._schema(schema_version))
            except Exception as e:
                if is_daily_quota(e):
                    raise DailyQuotaExceeded(
                        f"daily quota hit ({e}); rerun with the same --run-id to resume"
                    ) from e
                if is_rate_limit(e):
                    if attempt >= C.MAX_RETRIES:
                        raise DailyQuotaExceeded(
                            f"429 persisted after {C.MAX_RETRIES} retries; daily quota hit, "
                            "rerun the same command with --run-id to resume") from e
                    d = retry_delay_s(e, attempt)
                    log.warning("429, retrying in %.0fs (attempt %d)", d, attempt + 1)
                    self.sleep(d)
                    continue
                if self._is_5xx(e):
                    if attempt >= 4:
                        raise
                    # free-tier 503s seem to count against the 20/day quota: wait long, retry rarely
                    d = min(60.0 * 2**attempt, 300.0)
                    log.warning("%s, retrying in %.0fs (attempt %d)", e, d, attempt + 1)
                    self.sleep(d)
                    continue
                raise
        raise AssertionError("unreachable")

    @staticmethod
    def _is_5xx(e: Exception) -> bool:
        code = getattr(e, "code", None) or getattr(e, "status_code", None) or 0
        return 500 <= int(code) < 600 if str(code).isdigit() else False

    def _record(self, resp, blind_id: str, fps: float, media_resolution: str,
                duration_s: float, latency: float = 0.0,
                schema_version: str = "v1") -> dict:
        text = extract_text(resp)
        usage = extract_usage(resp)
        out_tokens = usage.get("response_tokens", 0) + usage.get("thinking_tokens", 0)
        cost = nominal_cost(self.model, usage.get("prompt_tokens", 0), out_tokens)
        rec = {
            "blind_id": blind_id,
            "model": self.model,
            "schema_version": schema_version,
            "thought_summary": extract_thought_summary(resp),
            "tier": self.tier,
            "fps": fps,
            "media_resolution": media_resolution,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_s": round(latency, 2),
            "usage": usage,
            "cost_usd_paid_equiv": round(cost, 6),
            "cost_usd_actual": 0.0 if self.tier == "free" else round(cost, 6),
            "raw_text": text,
        }
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            rec.update(status="invalid", invalid_reason="unparseable JSON",
                       result=None)
            return rec
        r, reason, flags = self._validate(data, duration_s, schema_version)
        if reason:
            rec.update(status="invalid", invalid_reason=reason, result=data)
        else:
            rec.update(status="ok", result=r.model_dump(), quality_flags=flags)
        return rec
