"""Tests for gemini_client.py with a fully mocked genai client (no network)."""

import json
import time
from types import SimpleNamespace

import pytest

from src.evals.hook_test import config as C
from src.evals.hook_test.gemini_client import (
    DailyQuotaExceeded, GeminiHookTester, estimate_call_cost, is_daily_quota,
    is_network_error, is_rate_limit, nominal_cost, retry_delay_s)

GOOD_JSON = json.dumps({
    "video_description": "desc", "analysis": "scratch pad",
    "p_strong": 0.8, "group": "strong", "confidence": "medium",
    "dropoff_timestamp_s": 7.0, "hook_verdict": "strong",
    "video_length_s_estimate": 21.0,
    "moments": [{"t_seconds": 1.0, "kind": "hook", "note": "x"}],
    "reasoning": "ok",
})


def fake_resp(text=GOOD_JSON, prompt=34000, resp=600, think=3000):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt, response_token_count=resp,
            thoughts_token_count=think, total_token_count=prompt + resp + think,
            cached_content_token_count=0),
    )


class FakeFiles:
    def __init__(self):
        self.uploads = []
        self._files = {}

    def upload(self, file, config=None):
        f = SimpleNamespace(name=f"files/{len(self.uploads)}", state="ACTIVE",
                            uri=f"gs://fake/{len(self.uploads)}",
                            display_name=getattr(config, "display_name", None))
        self.uploads.append((file, config))
        self._files[f.name] = f
        return f

    def get(self, name):
        if name not in self._files:
            raise KeyError(name)
        return self._files[name]


class FakeModels:
    def __init__(self, script):
        self.script = list(script)  # each item: resp or exception
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.script.pop(0) if self.script else fake_resp()
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script=()):
        self.files = FakeFiles()
        self.models = FakeModels(script)


class Err429(Exception):
    code = 429


@pytest.fixture
def blind_file(tmp_path):
    p = tmp_path / "vAAA111.mp4"
    p.write_bytes(b"fakevideo")
    return p


class TestAnalyze:
    def test_ok_call(self, tmp_path, blind_file):
        t = GeminiHookTester(client=FakeClient([fake_resp()]),
                             uploads_path=tmp_path / "u.json", sleep=lambda s: 0)
        rec = t.analyze("vAAA111", blind_file, "prompt", 21.0)
        assert rec["status"] == "ok"
        assert rec["result"]["p_strong"] == 0.8
        # free tier: actual 0, nominal > 0
        assert rec["cost_usd_actual"] == 0.0
        assert rec["cost_usd_paid_equiv"] > 0

    def test_paid_tier_cost(self, tmp_path, blind_file):
        t = GeminiHookTester(client=FakeClient([fake_resp()]), tier="paid",
                             uploads_path=tmp_path / "u.json", sleep=lambda s: 0)
        rec = t.analyze("vAAA111", blind_file, "prompt", 21.0)
        exp = 34000 * 0.75 / 1e6 + 3600 * 3.75 / 1e6
        assert rec["cost_usd_actual"] == pytest.approx(exp, abs=1e-6)
        assert rec["cost_usd_actual"] == rec["cost_usd_paid_equiv"]

    def test_blinding(self, tmp_path, blind_file):
        fake = FakeClient([fake_resp()])
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json",
                             sleep=lambda s: 0)
        t.analyze("vAAA111", blind_file, "the prompt text", 21.0)
        up_file, up_cfg = fake.files.uploads[0]
        assert up_cfg.display_name == "vAAA111"  # upload name is the blind id
        contents = fake.models.calls[0]["contents"]
        blob = json.dumps(contents, default=str)
        assert "vAAA111" not in blob  # no blind id leaks into the request
        title = "MY SECRET VIDEO TITLE"
        assert title not in blob and len(contents) == 2
        video_part, text_part = contents
        assert text_part == "the prompt text"
        assert video_part.video_metadata.fps == C.FPS
        assert str(video_part.media_resolution.level).endswith("HIGH")

    def test_resume_upload_cache(self, tmp_path, blind_file):
        fake = FakeClient([fake_resp(), fake_resp()])
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json",
                             sleep=lambda s: 0)
        t.upload_video("vAAA111", blind_file)
        t.upload_video("vAAA111", blind_file)  # cached, no second upload
        assert len(fake.files.uploads) == 1

    def test_retry_on_429(self, tmp_path, blind_file):
        fake = FakeClient([Err429("RESOURCE_EXHAUSTED 429"), fake_resp()])
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json",
                             sleep=lambda s: 0)
        rec = t.analyze("vAAA111", blind_file, "p", 21.0)
        assert rec["status"] == "ok" and len(fake.models.calls) == 2

    def test_retry_on_dropped_connection(self, tmp_path, blind_file):
        class ReadError(ConnectionError):
            pass
        fake = FakeClient([ReadError("[Errno 54] Connection reset by peer")] * 2 + [fake_resp()])
        waits = []
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json", sleep=waits.append)
        rec = t.analyze("vabc", blind_file, "PROMPT", 21.0)
        assert rec["status"] == "ok" and len(fake.models.calls) == 3 and waits == [10.0, 30.0]

    def test_dropped_connection_gives_up(self, tmp_path, blind_file):
        fake = FakeClient([ConnectionError("reset")] * 10)
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json", sleep=lambda s: 0)
        with pytest.raises(ConnectionError):
            t.analyze("vabc", blind_file, "PROMPT", 21.0)
        assert len(fake.models.calls) == 5
        assert not is_network_error(ValueError("x"))

    def test_daily_quota_stops(self, tmp_path, blind_file):
        fake = FakeClient([Err429("quota GenerateRequestsPerDayPerProject exceeded")] * 10)
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json",
                             sleep=lambda s: 0)
        with pytest.raises(DailyQuotaExceeded):
            t.analyze("vAAA111", blind_file, "p", 21.0)

    def test_schema_invalid_retried_once(self, tmp_path, blind_file):
        fake = FakeClient([fake_resp("not json"), fake_resp()])
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json",
                             sleep=lambda s: 0)
        rec = t.analyze("vAAA111", blind_file, "p", 21.0)
        assert rec["status"] == "ok"
        assert rec.get("first_attempt_invalid")
        assert len(fake.models.calls) == 2

    def test_schema_invalid_twice_stored_invalid(self, tmp_path, blind_file):
        fake = FakeClient([fake_resp("bad"), fake_resp("still bad")])
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json",
                             sleep=lambda s: 0)
        rec = t.analyze("vAAA111", blind_file, "p", 21.0)
        assert rec["status"] == "invalid" and len(fake.models.calls) == 2


class TestHelpers:
    def test_cost_math(self):
        assert nominal_cost("gemini-3.8-flash", 1_000_000, 1_000_000) == pytest.approx(4.5)
        assert estimate_call_cost("gemini-3.8-flash", 21.0) > 0

    def test_429_detection(self):
        assert is_rate_limit(Err429("429"))
        assert is_daily_quota(Exception("per day quota exceeded"))
        assert not is_daily_quota(Err429("slow down"))

    def test_retry_delay(self):
        e = Exception('retryDelay": "30s"')
        assert retry_delay_s(e, 0) == 30.0
        assert retry_delay_s(Exception("x"), 0) == 5.0
        assert retry_delay_s(Exception("x"), 10) == 120.0


class TestSchemaOrdering:
    def test_property_ordering_sent_to_api(self):
        """The schema the SDK sends must list fields in declaration order, so
        Gemini writes the scratch-pad fields before the ratings."""
        from google.genai._transformers import t_schema
        from src.evals.hook_test.schema import HookTestResult
        schema = t_schema(None, HookTestResult)
        top = schema.property_ordering
        assert top == [
            "video_description", "analysis", "hook_verdict",
            "dropoff_timestamp_s", "p_strong", "group", "confidence",
            "video_length_s_estimate", "moments", "reasoning",
        ]

    def test_no_additional_properties_in_sent_schema(self):
        """The API 400s on additionalProperties; schema sent must not contain it
        (extra-field strictness lives in validate_result, not the wire schema)."""
        from google.genai._transformers import t_schema
        from src.evals.hook_test.schema import HookTestResult
        schema = t_schema(None, HookTestResult)
        assert schema.additional_properties is None
        items = schema.properties["moments"].items
        assert items.additional_properties is None


class TestThoughts:
    def test_thought_parts_excluded_from_json_kept_as_summary(self, tmp_path, blind_file):
        resp = SimpleNamespace(
            text=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=1, response_token_count=1,
                thoughts_token_count=500, total_token_count=502,
                cached_content_token_count=0),
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
                SimpleNamespace(text="thinking about the hook...", thought=True),
                SimpleNamespace(text=GOOD_JSON, thought=None),
            ]))],
        )
        t = GeminiHookTester(client=FakeClient([resp]),
                             uploads_path=tmp_path / "u.json", sleep=lambda s: 0)
        rec = t.analyze("vAAA111", blind_file, "p", 21.0)
        assert rec["status"] == "ok"
        assert rec["raw_text"] == GOOD_JSON
        assert "thinking about the hook" in rec["thought_summary"]
        assert GOOD_JSON not in rec["thought_summary"]

    def test_thinking_config_set(self, tmp_path, blind_file):
        fake = FakeClient([fake_resp()])
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json",
                             sleep=lambda s: 0)
        t.analyze("vAAA111", blind_file, "p", 21.0)
        cfg = fake.models.calls[0]["config"]
        assert cfg.thinking_config is not None
        assert cfg.thinking_config.include_thoughts is True


class TestV2Schema:
    def test_v2_ordering_and_no_additional_props(self):
        from google.genai._transformers import t_schema
        from src.evals.hook_test.schema import HookTestResultV2
        schema = t_schema(None, HookTestResultV2)
        assert schema.property_ordering == [
            "video_description", "analysis", "hook_verdict",
            "dropoff_timestamp_s", "predicted_avg_view_pct", "confidence",
            "video_length_s_estimate", "moments", "reasoning",
        ]
        assert schema.additional_properties is None
        assert schema.properties["moments"].items.additional_properties is None

    def test_schema_selected_by_prompt_version(self, tmp_path, blind_file):
        from src.evals.hook_test.schema import HookTestResult, HookTestResultV2
        fake = FakeClient([fake_resp()])
        t = GeminiHookTester(client=fake, uploads_path=tmp_path / "u.json",
                             sleep=lambda s: 0)
        t.analyze("vAAA111", blind_file, "p", 21.0, prompt_version="hook_test_v1")
        assert fake.models.calls[0]["config"].response_schema is HookTestResult
        fake2 = FakeClient([fake_resp()])
        t2 = GeminiHookTester(client=fake2, uploads_path=tmp_path / "u2.json",
                              sleep=lambda s: 0)
        t2.analyze("vAAA111", blind_file, "p", 21.0, prompt_version="hook_test_v2")
        assert fake2.models.calls[0]["config"].response_schema is HookTestResultV2
