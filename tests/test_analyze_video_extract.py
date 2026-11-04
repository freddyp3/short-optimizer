"""Tests for extract.py with a fully mocked genai client (no network)."""

import json
from types import SimpleNamespace

import pytest

from src.tools.analyze_video import extract
from src.tools.analyze_video.extract import GeminiFeatureExtractor
from src.tools.analyze_video.schema import PROMPT_VERSION, VideoFeatures
from test_analyze_video_schema import valid_data
from test_hook_test_gemini_client import FakeClient, fake_resp

GOOD_JSON = json.dumps(valid_data(21.0))


@pytest.fixture
def blind_file(tmp_path):
    p = tmp_path / "vAAA111.mp4"
    p.write_bytes(b"fakevideo")
    return p


class TestExtractor:
    def test_ok_call_uses_video_features(self, tmp_path, blind_file):
        fake = FakeClient([fake_resp(GOOD_JSON)])
        t = GeminiFeatureExtractor(client=fake, uploads_path=tmp_path / "u.json",
                                   sleep=lambda s: 0)
        rec = t.analyze("vAAA111", blind_file, "prompt", 21.0,
                        prompt_version=PROMPT_VERSION)
        assert rec["status"] == "ok"
        assert rec["result"]["format_macro"] == "gameplay_clip"
        assert rec["result"]["payoffs"][0]["weight"] == 3
        assert rec["schema_version"] == PROMPT_VERSION
        assert rec["tier"] == "paid"  # real cost recorded, not free-tier 0
        assert rec["cost_usd_actual"] > 0
        assert fake.models.calls[0]["config"].response_schema is VideoFeatures

    def test_invalid_retried_once(self, tmp_path, blind_file):
        fake = FakeClient([fake_resp("bad"), fake_resp(GOOD_JSON)])
        t = GeminiFeatureExtractor(client=fake, uploads_path=tmp_path / "u.json",
                                   sleep=lambda s: 0)
        rec = t.analyze("vAAA111", blind_file, "p", 21.0)
        assert rec["status"] == "ok" and len(fake.models.calls) == 2

    def test_requires_paid_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GEMINI_API_KEY_PAID", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY_PAID"):
            GeminiFeatureExtractor(uploads_path=tmp_path / "u.json")

    def test_uses_env_paid_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_API_KEY_PAID", "paid-key-x")
        import src.evals.hook_test.gemini_client as gc
        seen = {}
        monkeypatch.setattr(gc, "new_genai_client",
                            lambda k: seen.setdefault("key", k) or FakeClient())
        t = GeminiFeatureExtractor(uploads_path=tmp_path / "u.json")
        assert t.tier == "paid" and seen["key"] == "paid-key-x"


class TestCLI:
    def test_split_gate_rejected(self):
        with pytest.raises(SystemExit):
            extract.main(["--split", "gate"])

    def test_split_heldout_rejected(self):
        with pytest.raises(SystemExit):
            extract.main(["--split", "heldout"])

    def test_pick_dev_videos_stratified(self):
        vids = extract.pick_dev_videos(10)
        assert len(vids) == 10
        assert (vids["label"] == "strong").sum() == 5
        assert (vids["label"] == "weak").sum() == 5
        assert (vids["split"] == "dev").all()
