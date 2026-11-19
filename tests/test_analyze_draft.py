"""Tests for analyze_draft: key handling, cost and step callbacks, cleanup, output hygiene (all Gemini calls mocked;
no data/ access, no network, no key)."""

import json
import logging
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evals.hook_test import config as C
from src.evals.hook_test.gemini_client import extract_usage, nominal_cost
from src.tools.analyze_video import extract as E
from src.tools.audience import brief as B
from src.tools.predict_retention import tool as P
from test_hook_test_gemini_client import FakeClient, fake_resp
from test_judge import fake_lib

KEY = "AIzaSy-test-own-key-0123456789abcdef"
ENV_KEY = "AIzaSy-test-env-key-zyxwvutsrq987654"
N = 10
TIMELINE = [{"start_s": float(s), "end_s": s + 1.0, "seen": f"The player moves Shelly {s}", "heard": "music"} for s in range(12)]
PAYOFFS = [{"t_s": 6.0, "kind": "reveal", "weight": 3}]
COSTS = {"v2": 0.03, "v3": 0.04, "L0": 0.01, "L2": 0.02}
REAL_EXTRACT = P.extract


class Client(FakeClient):
    def __init__(self, key=None, script=()):
        super().__init__(script)
        self.key, self.deleted = key, []
        self.files.delete = lambda name: self.deleted.append(name)


class Env:
    """Everything analyze_draft touches, faked; records what was called with what."""

    def __init__(self, monkeypatch, tmp_path):
        lib = fake_lib(N, 2)
        emb = np.random.default_rng(2).normal(size=(N, 8))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        self.st = SimpleNamespace(lib=lib, emb=emb)
        self.clients, self.extract_clients, self.judge_clients, self.tmp_dirs, self.upload_caches = [], [], [], [], []
        self.fail_at, self.log_key = None, None
        self.video = tmp_path / "upload_dir" / "original name.mp4"
        self.video.parent.mkdir()
        self.video.write_bytes(b"fakevideo")
        monkeypatch.setattr(P, "state", lambda: self.st)
        monkeypatch.setattr(B, "_LABELS", None)
        monkeypatch.setattr(P, "new_genai_client", self.new_client)
        monkeypatch.setattr(P, "blind_copy", self.blind_copy)
        monkeypatch.setattr(P, "measure", lambda path: {"duration_s": 12.0, "cuts_per_s": 0.5})
        monkeypatch.setattr(P, "extract", self.extract)
        monkeypatch.setattr(P, "judge_one", self.judge_one)
        monkeypatch.setattr(P, "draft_row", lambda v2, code, dur, template: pd.Series(
            {"t_premise_clear_s": 1.5, "tail_all": 6.0, "dead_time_s": float("nan"), "cuts_per_s": np.float64(0.5), "arc": "single_peak",
             "payoff_count_all": 1, "main_payoff_frac": 0.5, "video_id": "draft"}, dtype=object))
        monkeypatch.setattr(P, "draft_embedding", lambda timeline: emb[:1] * 1.0)
        monkeypatch.setattr(P, "with_draft", lambda st, row, timeline, payoffs, e=None: lib)
        monkeypatch.setattr(P, "build_result", self.build_result)
        monkeypatch.delenv("GEMINI_API_KEY_PAID", raising=False)

    def new_client(self, key):
        self.clients.append(Client(key))
        return self.clients[-1]

    def blind_copy(self, path, out_dir):
        self.tmp_dirs.append(Path(out_dir))
        out = Path(out_dir) / "draft_0123456789.mp4"
        out.write_bytes(b"blind")
        return out

    def fail(self, where, client):
        if self.log_key:
            logging.getLogger("judge").warning("retrying after %r", RuntimeError(f"key={self.log_key}"))
        if self.fail_at == where:
            try:
                raise ValueError(f"inner: API key {client.key} rejected")
            except ValueError as inner:
                raise RuntimeError(f"400 INVALID_ARGUMENT for key {client.key} at {where}") from inner

    def extract(self, path, blind_id, dur, version, client=None, uploads_path=None):
        assert Path(path).exists() and Path(path).parent == self.tmp_dirs[-1]
        self.extract_clients.append(client)
        self.upload_caches.append(Path(uploads_path))
        cache = json.loads(Path(uploads_path).read_text()) if Path(uploads_path).exists() else {}
        cache[blind_id] = {"file_name": f"files/{version}", "uploaded_at": 0, "sha256": "x"}
        Path(uploads_path).write_text(json.dumps(cache))
        self.fail(version, client)
        result = {"payoffs": PAYOFFS} if version == "v2" else {"timeline": TIMELINE}
        return {"status": "ok", "result": result, "cost_usd_paid_equiv": COSTS[version]}

    def judge_one(self, lib, i, variant, arm, model, client):
        self.judge_clients.append(client)
        self.fail(variant, client)
        shown = lib.df.video_id[:3].tolist() if variant == "L2" else []
        return {"status": "ok", "cost_usd": COSTS[variant], "neighbours": shown, "result": {}}

    def build_result(self, st, lib, row, v2, a, b, dur, cost):
        return {"duration_s": dur, "similar_past_shorts_used": b["neighbours"], "cost_usd": round(cost, 4), "advice": "cut the tail"}


@pytest.fixture
def env(monkeypatch, tmp_path):
    return Env(monkeypatch, tmp_path)


def cleaned_up(env) -> bool:
    return all(not d.exists() for d in env.tmp_dirs) and all(not p.exists() for p in env.upload_caches)


def test_explicit_key_reaches_all_four_calls_and_nothing_else(env, monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY_PAID", ENV_KEY)
    env.log_key = KEY
    with caplog.at_level(logging.DEBUG):
        out = P.analyze_draft(env.video, api_key=KEY)
    assert [c.key for c in env.clients] == [KEY]
    assert env.extract_clients == [env.clients[0]] * 2 and env.judge_clients == [env.clients[0]] * 2
    text = json.dumps([out.result, out.artifacts])
    assert KEY not in text and ENV_KEY not in text
    assert len(caplog.records) == 4 and P.REDACTED in caplog.text
    assert KEY not in caplog.text and all(KEY not in r.getMessage() and KEY not in str(r.args) for r in caplog.records)
    assert out.cost_usd == pytest.approx(0.10) and out.result["cost_usd"] == 0.1
    assert P._SECRETS == set()


def test_no_key_argument_uses_the_paid_key_from_the_environment(env, monkeypatch):
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY_PAID not set"):
        P.analyze_draft(env.video)
    monkeypatch.setenv("GEMINI_API_KEY_PAID", ENV_KEY)
    P.analyze_draft(env.video)
    assert [c.key for c in env.clients] == [ENV_KEY]


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_key_is_refused_and_never_falls_back(env, monkeypatch, empty):
    monkeypatch.setenv("GEMINI_API_KEY_PAID", ENV_KEY)
    with pytest.raises(RuntimeError, match="empty"):
        P.analyze_draft(env.video, api_key=empty)
    assert env.clients == [] and env.tmp_dirs == []


@pytest.mark.parametrize("where", ["v2", "v3", "L0", "L2"])
def test_key_never_appears_in_a_raised_exception(env, caplog, where):
    env.fail_at = where
    with caplog.at_level(logging.DEBUG), pytest.raises(Exception) as ei:
        try:
            P.analyze_draft(env.video, api_key=KEY)
        except Exception:
            logging.getLogger("server").exception("job failed")
            raise
    e = ei.value
    assert isinstance(e, RuntimeError) and "400 INVALID_ARGUMENT" in str(e) and P.REDACTED in str(e)
    assert e.__cause__ is None and e.__context__ is None
    shown = str(e) + repr(e) + repr(e.args) + "".join(traceback.format_exception(e)) + caplog.text
    assert KEY not in shown
    assert cleaned_up(env) and env.clients[0].deleted == (["files/v2"] if where == "v2" else ["files/v2", "files/v3"])
    assert P._SECRETS == set()


def test_errors_without_the_key_keep_their_type(env, monkeypatch):
    class Odd(Exception):
        pass

    def boom(path):
        raise Odd("ffprobe failed")

    monkeypatch.setattr(P, "measure", boom)
    with pytest.raises(Odd, match="ffprobe failed"):
        P.analyze_draft(env.video, api_key=KEY)
    assert cleaned_up(env) and env.clients[0].deleted == []


def test_log_records_are_redacted_only_while_the_job_runs(env, caplog):
    P.analyze_draft(env.video, api_key=KEY)
    with caplog.at_level(logging.INFO):
        logging.getLogger("other").info("unrelated %s", KEY)
    assert KEY in caplog.text


def test_on_cost_is_called_once_per_paid_call(env):
    costs = []
    out = P.analyze_draft(env.video, api_key=KEY, on_cost=costs.append)
    assert costs == [0.03, 0.04, 0.01, 0.02] and out.cost_usd == pytest.approx(sum(costs))


@pytest.mark.parametrize("stop_after,deleted,judged", [(1, ["files/v2"], 0), (2, ["files/v2", "files/v3"], 0),
                                                       (3, ["files/v2", "files/v3"], 1)])
def test_exception_from_on_cost_aborts_and_cleans_up(env, stop_after, deleted, judged):
    class BudgetExceeded(Exception):
        pass

    costs, steps = [], []

    def on_cost(usd):
        costs.append(usd)
        if len(costs) == stop_after:
            raise BudgetExceeded("over the cap")

    with pytest.raises(BudgetExceeded, match="over the cap"):
        P.analyze_draft(env.video, api_key=KEY, on_cost=on_cost, on_step=steps.append)
    assert len(costs) == stop_after and len(env.judge_clients) == judged and "modelling" not in steps
    assert cleaned_up(env) and env.clients[0].deleted == deleted
    assert P._SECRETS == set()


def test_on_step_order(env):
    steps = []
    P.analyze_draft(env.video, api_key=KEY, on_step=steps.append)
    assert steps == ["measuring", "describing", "judging", "modelling"] == list(P.STEPS)


def test_success_cleans_up_and_uses_a_private_upload_cache(env):
    P.analyze_draft(env.video, api_key=KEY)
    assert len(env.tmp_dirs) == 1 and cleaned_up(env) and env.video.exists()
    assert set(env.upload_caches) == {env.tmp_dirs[0] / "uploads.json"}
    assert env.upload_caches[0].resolve() != E.UPLOADS_JSON.resolve()
    assert env.clients[0].deleted == ["files/v2", "files/v3"]


def test_a_failing_delete_does_not_hide_the_result(env, monkeypatch, caplog):
    def new_client(key):
        c = Client(key)
        c.files.delete = lambda name: (_ for _ in ()).throw(RuntimeError(f"delete failed for {key}"))
        env.clients.append(c)
        return c

    monkeypatch.setattr(P, "new_genai_client", new_client)
    out = P.analyze_draft(env.video, api_key=KEY)
    assert out.result["advice"] == "cut the tail" and KEY not in caplog.text and "could not delete" in caplog.text


def test_output_is_json_and_holds_no_id_or_path(env, tmp_path):
    out = P.analyze_draft(env.video, api_key=KEY)
    assert set(out.artifacts) == {"timeline", "payoffs", "structure", "embedding", "duration_s"}
    text = json.dumps([out.result, out.artifacts], allow_nan=False)
    lib = env.st.lib
    assert not any(v in text for v in lib.df.video_id) and not any(b in text for b in lib.df.blind_id)
    for leak in (str(tmp_path), str(env.tmp_dirs[0]), env.tmp_dirs[0].name, env.video.name, "upload_dir", "draft_0123456789", "/"):
        assert leak not in text
    assert out.result["similar_past_shorts_used"] == [B.library_label(i) for i in range(3)]
    assert all(x.startswith("past Short ") for x in out.result["similar_past_shorts_used"])


def test_artifacts_content(env):
    a = P.analyze_draft(env.video, api_key=KEY).artifacts
    assert a["timeline"] == TIMELINE and a["payoffs"] == PAYOFFS and a["duration_s"] == 12.0
    assert a["embedding"] == pytest.approx(env.st.emb[0].tolist()) and all(type(x) is float for x in a["embedding"])
    assert a["structure"] == {"t_premise_clear_s": 1.5, "tail_all": 6.0, "dead_time_s": None, "cuts_per_s": 0.5, "arc": "single_peak",
                              "payoff_count_all": 1.0, "main_payoff_frac": 0.5}
    assert B.similar_for_draft.__defaults__ == (5,)


def test_artifacts_feed_the_audience_tools(env, monkeypatch):
    monkeypatch.setattr(B, "_COMMENTS", pd.DataFrame({"video_id": env.st.lib.df.video_id, "text": "nice one", "like_count": 1}))
    a = P.analyze_draft(env.video, api_key=KEY).artifacts
    sim = B.similar_for_draft(a, 3)
    brief = B.audience_brief(a, sim)
    assert sim[0]["similarity"] == 1.0 and len(brief["draft_description_untrusted_text"]) == 12
    assert brief["search_hints_untrusted_text"] == ["Shelly"] and brief["your_viewers"]["n_comments"] == 3


def test_invalid_judging_is_charged_then_fails(env, monkeypatch):
    inner = env.judge_one
    monkeypatch.setattr(P, "judge_one", lambda *a: {**inner(*a), "status": "invalid", "invalid_reason": "bad curve"})
    costs = []
    with pytest.raises(RuntimeError, match="judging failed: bad curve"):
        P.analyze_draft(env.video, api_key=KEY, on_cost=costs.append)
    assert len(costs) == 4


def test_predict_retention_wraps_analyze_draft_with_a_given_client(env):
    mine = Client()
    res = P.predict_retention(env.video, client=mine)
    assert res["advice"] == "cut the tail" and env.clients == [] and env.judge_clients == [mine, mine]
    with pytest.raises(ValueError):
        P.analyze_draft(env.video, api_key=KEY, client=mine)


def test_real_extractor_uses_the_job_client_and_cache_and_files_are_deleted(env, monkeypatch, tmp_path):
    """The unpatched extract path against a fake genai client that answers with unparseable text."""
    research_cache = tmp_path / "research_uploads.json"
    monkeypatch.setattr(E, "UPLOADS_JSON", research_cache)
    monkeypatch.setattr(P, "extract", REAL_EXTRACT)
    made = []

    def new_client(key):
        made.append(Client(key, [fake_resp("not json"), fake_resp("still not json")]))
        return made[-1]

    monkeypatch.setattr(P, "new_genai_client", new_client)
    costs = []
    with pytest.raises(RuntimeError, match="extraction v2 failed: unparseable JSON"):
        P.analyze_draft(env.video, api_key=KEY, on_cost=costs.append)
    client = made[0]
    assert len(made) == 1 and client.key == KEY and len(client.models.calls) == 2
    assert len(client.files.uploads) == 1 and client.deleted == ["files/0"]
    assert client.files.uploads[0][1].display_name == "draft_0123456789_v2"
    assert not research_cache.exists() and not env.tmp_dirs[0].exists()
    u = extract_usage(fake_resp("x"))
    one = nominal_cost(C.DEFAULT_MODEL, u["prompt_tokens"], u["response_tokens"] + u["thinking_tokens"])
    assert one > 0 and costs == [pytest.approx(2 * one, rel=1e-3)]


def test_extraction_cost_counts_the_invalid_first_attempt():
    assert P.extraction_cost({"cost_usd_paid_equiv": 0.03}) == 0.03
    assert P.extraction_cost({"cost_usd_paid_equiv": 0.03, "first_attempt_invalid": {"cost_usd_paid_equiv": 0.02}}) == pytest.approx(0.05)
    assert P.extraction_cost({"cost_usd_paid_equiv": 0.03, "first_attempt": {"cost_usd_paid_equiv": 0.01}}) == pytest.approx(0.04)


def test_safe_error():
    assert P.safe_error(RuntimeError("plain"), KEY) is None and P.safe_error(RuntimeError(KEY), None) is None
    e = P.safe_error(KeyError(KEY), KEY)
    assert isinstance(e, RuntimeError) and KEY not in str(e) and str(e).startswith("KeyError: ")
    try:
        try:
            raise ValueError(f"deep {KEY}")
        except ValueError:
            raise RuntimeError("outer says nothing")
    except RuntimeError as outer:
        assert str(P.safe_error(outer, KEY)) == "RuntimeError: outer says nothing"


class Kept(dict):
    """A checkpoint that keeps stages in memory, as JSON, like the server's."""

    def put(self, stage, value):
        self.setdefault(stage, json.loads(json.dumps(value)))


def test_a_retry_with_a_checkpoint_makes_no_finished_call_again(env):
    kept, first, second = Kept(), [], []
    env.fail_at = "L2"                                   # the first attempt dies in its fourth call
    with pytest.raises(Exception):
        P.analyze_draft(env.video, api_key=KEY, on_cost=first.append, checkpoint=kept)
    assert first == [0.03, 0.04, 0.01] and sorted(kept) == ["extract_v2", "extract_v3", "judge_L0"]
    assert (len(env.extract_clients), len(env.judge_clients)) == (2, 2)
    env.fail_at = None
    out = P.analyze_draft(env.video, api_key=KEY, on_cost=second.append, checkpoint=kept)
    assert (len(env.extract_clients), len(env.judge_clients)) == (2, 3)          # only the fourth call ran
    assert second == [0.02] and out.cost_usd == pytest.approx(0.02)
    assert sorted(kept) == ["extract_v2", "extract_v3", "judge_L0", "judge_L2"]
    plain = P.analyze_draft(env.video, api_key=KEY)                               # same answer as an unbroken run
    assert {**out.result, "cost_usd": 0} == {**plain.result, "cost_usd": 0} and out.artifacts == plain.artifacts
    third = []
    P.analyze_draft(env.video, api_key=KEY, on_cost=third.append, checkpoint=kept)
    assert third == [] and (len(env.extract_clients), len(env.judge_clients)) == (4, 5)   # nothing beyond the plain run


def test_an_invalid_call_is_not_kept_for_the_retry(env, monkeypatch):
    kept = Kept()
    monkeypatch.setattr(P, "judge_one", lambda lib, i, variant, arm, model, client: {
        "status": "invalid", "cost_usd": 0.01, "invalid_reason": "x", "neighbours": []})
    with pytest.raises(RuntimeError):
        P.analyze_draft(env.video, api_key=KEY, checkpoint=kept)
    assert sorted(kept) == ["extract_v2", "extract_v3"]
