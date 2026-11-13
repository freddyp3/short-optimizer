"""Tests for the judging pipeline (synthetic library, no data/ access, no network)."""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.judge import library as L
from src.models.judge import prompt as P
from src.models.judge import score as S


def fake_lib(n=30, seed=0) -> L.Library:
    rng = np.random.default_rng(seed)
    dur = rng.uniform(9, 58, n).round(1)
    df = pd.DataFrame({
        "video_id": [f"VIDEOID{i:03d}" for i in range(n)], "blind_id": [f"vblind{i:03d}" for i in range(n)],
        "duration_s": dur, "duration_seconds": np.ceil(dur), "published_at": "2024-06-01T00:00:00Z",
        "title": [f"SECRET TITLE {i}" for i in range(n)], "engagedViews": rng.integers(1000, 10**6, n),
        "t_premise_clear_s": 1.5, "tail_all": rng.uniform(0, 5, n), "dead_time_s": 1.0, "cuts_per_s": 0.4,
        "arc": "single_peak", "early_hold": rng.normal(size=n), "mid_hold": rng.normal(size=n),
        "late_hold": rng.normal(size=n), "replay_level": rng.normal(size=n), "like_rate": rng.normal(size=n)})
    tl = {v: [{"start_s": float(s), "end_s": float(s + 1), "seen": f"thing {s}", "heard": "music"}
              for s in range(int(d))] for v, d in zip(df.video_id, dur)}
    emb = rng.normal(size=(n, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    Z = rng.normal(size=(n, 9))
    return L.Library(
        df=df, timelines=tl, payoffs={v: [{"t_s": 4.0, "kind": "reveal", "weight": 3}] for v in df.video_id},
        points=rng.uniform(701, 799, (n, 5)).round(0), rewatch=L.tiers(df.replay_level).to_numpy(),
        likes=L.tiers(df.like_rate).to_numpy(), late=df.late_hold.to_numpy(), sim_e=emb @ emb.T,
        dist_s=np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)), hint_points=rng.uniform(40, 99, (n, 5)),
        hint_late_pct=rng.uniform(0, 100, n), split=L.make_split(df, n_pilot=9))


@pytest.fixture(scope="module")
def lib():
    return fake_lib()


class TestSplit:
    def test_partition_and_size(self, lib):
        s = lib.split
        assert len(s["pilot"]) == 9 and not set(s["pilot"]) & set(s["validation"])
        assert set(s["pilot"]) | set(s["validation"]) == set(lib.df.video_id)

    def test_deterministic_and_stratified(self, lib):
        assert L.make_split(lib.df, 9) == L.make_split(lib.df, 9)
        third = pd.qcut(lib.df.duration_s, 3, labels=False)
        assert third[lib.df.video_id.isin(lib.split["pilot"])].value_counts().tolist() == [3, 3, 3]

    def test_written_once(self, lib, tmp_path):
        import json
        p = tmp_path / "split.json"
        p.write_text(json.dumps(lib.split))
        assert L.load_split(lib.df, p) == lib.split
        with pytest.raises(ValueError):
            L.load_split(lib.df.iloc[:-1], p)


class TestNeighbours:
    @pytest.mark.parametrize("arm", ["E", "S", "random"])
    def test_five_distinct_never_self(self, lib, arm):
        for i in range(len(lib)):
            nb = [j for j, _ in lib.neighbours(i, arm, np.random.default_rng(i))]
            assert len(set(nb)) == 5 and i not in nb

    def test_threshold_arm_may_show_fewer_or_none(self, lib):
        for i in range(len(lib)):
            nb = lib.neighbours(i, "T30")
            assert len(nb) <= 5 and i not in [j for j, _ in nb] and all(s >= 0.30 for _, s in nb)
            assert [s for _, s in nb] == sorted((s for _, s in nb), reverse=True)
        assert lib.neighbours(0, "T100") == []

    def test_no_example_means_the_plain_prompt(self, lib):
        assert P.build_prompt(lib, 0, "L2", "T100")[0] == P.build_prompt(lib, 0, "L0", "E")[0]

    def test_contrast_pick(self):
        order = np.arange(20)
        late = np.arange(20, dtype=float)
        late[[3, 7]] = [-50, 99]
        got = L.pick_contrast(order, late)
        assert {3, 0, 7, 14} <= set(got) and len(got) == 5 and max(got) < 15 and got == sorted(got)

    def test_arm_e_is_drawn_from_the_15_most_similar(self, lib):
        o = lib.others(0)
        top = set(o[np.argsort(-lib.sim_e[0, o])][:15])
        assert {j for j, _ in lib.neighbours(0, "E")} <= top

    def test_anchor_excludes_self(self, lib):
        other = copy.deepcopy(lib)
        other.points[4] = 9999
        np.testing.assert_array_equal(lib.anchor(4), other.anchor(4))


class TestPromptLeaks:
    @pytest.mark.parametrize("variant", sorted(P.VARIANTS))
    @pytest.mark.parametrize("arm", ["E", "S"])
    def test_prompt_does_not_depend_on_the_drafts_own_outcomes(self, lib, variant, arm):
        for i in (0, 7, 19):
            other = copy.deepcopy(lib)
            other.points[i] = 9999
            other.rewatch[i] = "LEAK"
            other.likes[i] = "LEAK"
            other.late[i] = 1e6
            for col in ("early_hold", "mid_hold", "late_hold", "replay_level", "like_rate"):
                other.df.loc[i, col] = 1e6
            assert P.build_prompt(lib, i, variant, arm)[0] == P.build_prompt(other, i, variant, arm)[0]

    def test_no_ids_titles_views_or_dates(self, lib):
        for i in range(len(lib)):
            text = P.build_prompt(lib, i, "L4", "E")[0]
            assert "VIDEOID" not in text and "vblind" not in text and "SECRET TITLE" not in text
            assert "2024-06" not in text and str(lib.df.engagedViews[i]) not in text

    def test_ladder_adds_one_thing_at_a_time(self, lib):
        t = {v: P.build_prompt(lib, 3, v, "E")[0] for v in P.VARIANTS}
        assert "### Example A" not in t["L0"] and "### Example E" in t["L1"]
        assert "Real audience" not in t["L1"] and t["L2"].count("Real audience") == 5
        assert "How to use the examples" not in t["L2"] and "How to use the examples" in t["L3"]
        assert "Statistical hint" not in t["L3"] and "Statistical hint" in t["L4"]

    def test_outcomes_shown_belong_to_the_neighbours(self, lib):
        text, shown = P.build_prompt(lib, 3, "L2", "E")
        for j in shown:
            assert P._points(lib.points[j]) in text
        assert P._points(lib.points[3]) not in text


GOOD = {"reasoning": "x", "lessons": [{"example": "B", "observation": "late payoff"}],
        "retention": {"at_10": 95, "at_25": 85, "at_50": 70, "at_75": 60, "at_95": 50},
        "drop_moments": [{"t_s": 3.0, "reason": "a"}, {"t_s": 9.0, "reason": "b"}],
        "rewatch_level": "high", "like_rate_tier": "low", "advice": ["cut the tail at 12 s"]}


class TestValidate:
    def test_good(self):
        j, reason = P.validate(GOOD, 20.0, 5)
        assert reason is None and j.retention.at_95 == 50

    @pytest.mark.parametrize("patch,msg", [
        ({"drop_moments": GOOD["drop_moments"][:1]}, "drop moments"),
        ({"drop_moments": [{"t_s": 3.0, "reason": "a"}, {"t_s": 25.0, "reason": "b"}]}, "outside the video"),
        ({"retention": GOOD["retention"] | {"at_10": 400}}, "0-150"),
        ({"rewatch_level": "huge"}, "schema"),
    ])
    def test_invalid(self, patch, msg):
        j, reason = P.validate(GOOD | patch, 20.0, 5)
        assert j is None and msg in reason

    def test_lesson_must_cite_a_shown_example(self):
        assert "not shown" in P.validate(GOOD, 20.0, 0)[1]
        assert P.validate(GOOD | {"lessons": []}, 20.0, 0)[1] is None


class TestNumbers:
    def test_chunked_embeddings_are_unit_vectors(self):
        enc = lambda pieces: np.array([[len(p), 1.0, 2.0] for p in pieces])
        e = L.chunked_embeddings([["a"] * 20, ["b"] * 3], enc, chunk=8)
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0)

    def test_curve_points(self):
        r = np.arange(1, 101) / 100
        ret = pd.DataFrame({"video_id": "a", "elapsedVideoTimeRatio": r, "audienceWatchRatio": 1.2 - r})
        np.testing.assert_allclose(L.curve_points(ret, ["a"]).to_numpy()[0],
                                   100 * (1.2 - np.array(L.CHECKPOINTS)) / 1.18)

    def test_implied_points_from_constant_loss(self):
        seg = SimpleNamespace(n_videos=2, per_video=99, r=np.tile(np.arange(1, 100) / 100, 2))
        pts = L.implied_points(seg, np.full(198, 0.01))
        np.testing.assert_allclose(pts[0], [100 * 0.99 ** k for k in (8, 23, 48, 73, 93)])

    def test_holds_from_points(self):
        np.testing.assert_allclose(S.holds_from_points(np.array([[95.0, 80, 60, 40, 30]])), [[80, 0.5, 0.75]])

    def test_perfect_prediction_scores_one(self, lib):
        Y = lib.df[list(S.HOLDS)].to_numpy()
        assert S.hold_scores(Y, Y).tolist() == pytest.approx([1, 1, 1, 1])
