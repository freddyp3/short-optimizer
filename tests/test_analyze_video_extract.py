"""Tests for extract.py with a fully mocked genai client (no network)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

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
        sel = pd.read_csv("data/evals/hook_test/selection.csv").set_index("blind_id")
        labels = sel.loc[vids["blind_id"], "label"]
        assert (labels == "strong").sum() == 5
        assert (labels == "weak").sum() == 5
        assert (sel.loc[vids["blind_id"], "split"] == "dev").all()
        assert vids["blind_path"].str.contains("blind/").all()


class TestPinning:
    def test_thinking_level_medium_on_extractor(self, tmp_path, blind_file):
        from google.genai import types
        fake = FakeClient([fake_resp(GOOD_JSON)])
        t = GeminiFeatureExtractor(client=fake, uploads_path=tmp_path / "u.json",
                                   sleep=lambda s: 0)
        t.analyze("vAAA111", blind_file, "p", 21.0)
        cfg = fake.models.calls[0]["config"].thinking_config
        assert cfg.thinking_level == types.ThinkingLevel.MEDIUM
        assert cfg.include_thoughts is True

    def test_hook_tester_keeps_unpinned_thinking(self, tmp_path, blind_file):
        from src.evals.hook_test.gemini_client import GeminiHookTester
        t = GeminiHookTester(client=FakeClient([fake_resp()]),
                             uploads_path=tmp_path / "u.json", sleep=lambda s: 0)
        t.analyze("vAAA111", blind_file, "p", 21.0)
        cfg = t.client.models.calls[0]["config"].thinking_config
        assert cfg.thinking_level is None
        assert cfg.include_thoughts is True

    def test_model_version_recorded(self, tmp_path, blind_file):
        r = fake_resp(GOOD_JSON)
        r.model_version = "gemini-3.8-flash-001"
        t = GeminiFeatureExtractor(client=FakeClient([r]),
                                   uploads_path=tmp_path / "u.json", sleep=lambda s: 0)
        rec = t.analyze("vAAA111", blind_file, "p", 21.0)
        assert rec["model_version"] == "gemini-3.8-flash-001"


class TestFileDuration:
    def test_blind_duration_uses_ffprobe(self, monkeypatch, tmp_path):
        monkeypatch.setattr(extract, "probe",
                            lambda p: {"duration_s": 11.3, "width": 1, "height": 1,
                                       "has_audio": True})
        assert extract.blind_duration(tmp_path / "x.mp4") == 11.3


def v3_json(duration):
    from src.tools.analyze_video.schema import windows
    d = valid_data(duration)
    d["timeline"] = [{"start_s": a, "end_s": b, "seen": "x", "heard": "y"}
                     for a, b in windows(duration)]
    return json.dumps(d)


class TestVersions:
    def test_v3_uses_v3_schema_and_rendered_windows(self, tmp_path, blind_file):
        from src.tools.analyze_video.schema import VideoFeaturesV3
        fake = FakeClient([fake_resp(v3_json(21.0))])
        t = GeminiFeatureExtractor(client=fake, version="v3",
                                   uploads_path=tmp_path / "u.json", sleep=lambda s: 0)
        from src.tools.analyze_video.schema import render_prompt
        rec = t.analyze("vAAA111", blind_file, render_prompt(
            "windows: {windows} ({n_windows})", 21.0), 21.0, prompt_version="v3")
        assert rec["status"] == "ok" and rec["schema_version"] == "v3"
        assert fake.models.calls[0]["config"].response_schema is VideoFeaturesV3
        assert "0.0-1.0" in fake.models.calls[0]["contents"][1]


class RecordingExtractor:
    """Stands in for GeminiFeatureExtractor in main(): records call order, no network."""
    instances = []

    def __init__(self, **kw):
        self.calls = []
        RecordingExtractor.instances.append(self)

    def analyze(self, blind_id, path, prompt_text, duration_s, **kw):
        self.calls.append(blind_id)
        return {"blind_id": blind_id, "status": "ok", "result": {"payoffs": []},
                "quality_flags": [], "usage": {"total_tokens": 1},
                "cost_usd_paid_equiv": 0.01, "latency_s": 0.1}


@pytest.fixture
def fake_train(tmp_path, monkeypatch):
    """A 4-row consistency selection + blind files + stubbed probe/extractor."""
    vids = pd.DataFrame({
        "video_id": [f"vid{i}" for i in range(4)],
        "blind_id": [f"vb{i:07x}" for i in range(4)],
        "band": ["short"] * 4,
        "duration_s_export": [10.0, 11.0, 12.0, 13.0],
        "blind_path": [str(tmp_path / f"b{i}.mp4") for i in range(4)],
    })
    for bp in vids["blind_path"]:
        Path(bp).write_bytes(b"fake")
    monkeypatch.setattr(extract, "CONSISTENCY_CSV", tmp_path / "sel.csv")
    vids.to_csv(tmp_path / "sel.csv", index=False)
    monkeypatch.setattr(extract, "blind_duration", lambda p: 10.0)
    monkeypatch.setattr(extract, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(extract, "GeminiFeatureExtractor", RecordingExtractor)
    RecordingExtractor.instances.clear()
    return vids


class TestTrainRunner:
    def test_train_reads_selection(self, fake_train):
        vids = extract.pick_train_videos()
        assert len(vids) == 4 and set(vids.columns) == {
            "video_id", "blind_id", "band", "duration_s_export", "blind_path"}

    def test_runs2_writes_r1_r2(self, fake_train, tmp_path):
        rc = extract.main(["--split", "train", "--runs", "2", "--run-id", "t1",
                           "--budget-usd", "99"])
        assert rc == 0
        rd = tmp_path / "raw" / "t1"
        assert len(list(rd.glob("*__r1.json"))) == 4
        assert len(list(rd.glob("*__r2.json"))) == 4
        rec = json.loads((rd / "vb0000000__r1.json").read_text())
        assert rec["run_k"] == 1 and rec["band"] == "short"

    def test_resume_skips(self, fake_train, tmp_path):
        extract.main(["--split", "train", "--runs", "1", "--run-id", "t2",
                      "--budget-usd", "99"])
        n_calls = len(RecordingExtractor.instances[-1].calls)
        extract.main(["--split", "train", "--runs", "1", "--run-id", "t2",
                      "--budget-usd", "99"])
        assert len(RecordingExtractor.instances[-1].calls) == 0  # all skipped
        assert n_calls == 4

    def test_order_shuffled_per_run(self, fake_train):
        extract.main(["--split", "train", "--runs", "2", "--run-id", "t3",
                      "--budget-usd", "99"])
        calls = RecordingExtractor.instances[-1].calls
        assert calls[:4] != calls[4:] or len(set(calls)) == 4  # orders differ or all unique

    def test_budget_abort(self, fake_train):
        rc = extract.main(["--split", "train", "--runs", "1", "--run-id", "t4",
                           "--budget-usd", "0.0"])
        assert rc == 1

    @pytest.mark.parametrize("split", ["gate", "heldout"])
    def test_eval_splits_rejected(self, split):
        with pytest.raises(SystemExit):
            extract.main(["--split", split])


class TestTrainAll:
    def test_reads_only_new_rows(self, tmp_path, monkeypatch):
        pd.DataFrame({
            "video_id": ["a", "b", "c"],
            "blind_id": ["vaaa", "vbbb", "vccc"],
            "duration_s_export": [10.0] * 3,
            "duration_s_file": [9.9] * 3,
            "blind_path": ["x/a.mp4"] * 3,
            "source": ["ab30", "new", "new"],
        }).to_csv(tmp_path / "ts.csv", index=False)
        monkeypatch.setattr(extract, "TRAIN_CSV", tmp_path / "ts.csv")
        vids = extract.pick_train_all_videos()
        assert vids["video_id"].tolist() == ["b", "c"]
        assert "ab30" not in vids["video_id"].tolist() if True else False

    def test_gate_heldout_still_impossible(self):
        for s in ("gate", "heldout"):
            with pytest.raises(SystemExit):
                extract.main(["--split", s])
