"""Tests for the audience brief (synthetic library and comments; no data/ access, no network, no key)."""

import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.tools.audience import brief as B
from src.tools.audience import comments as CM
from src.tools.predict_retention import tool as P
from test_judge import fake_lib

N = 12
LONG = "x" * 500


def fake_state(n=N, seed=1):
    lib = fake_lib(n, seed)
    lib.df["published_at"] = [f"2024-{12 - i:02d}-01T00:00:00Z" for i in range(n)]      # position 0 is the newest
    for v in lib.timelines:
        lib.timelines[v][0]["seen"] = "@somehandle opens the match " + LONG
    emb = np.random.default_rng(seed).normal(size=(n, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return SimpleNamespace(lib=lib, emb=emb)


def raw_comments(ids):
    rows = []
    for v in ids:
        rows += [(v, False, "viewer1", "@SecretFan", f"part 2 please @alfie369k {LONG}", 9),
                 (v, False, "viewer2", "@OtherFan", "how did you do that?", 7),
                 (v, False, "viewer3", "@Third", "LMAO \U0001F602", 5),
                 (v, False, "viewer4", "@Fourth", "this is so fake", 4),
                 (v, False, "viewer5", "@Fifth", "nice one", 3),
                 (v, False, "viewer6", "@Sixth", "first", 2),
                 (v, False, "viewer7", "@Seventh", "see https://youtu.be/dQw4w9WgXcQ now", 1),
                 (v, True, "viewer8", "@Replier", "a reply nobody should see", 99),
                 (v, False, "OWNER", "@alfie369k", "owner says thanks", 50)]
    return pd.DataFrame(rows, columns=["video_id", "is_reply", "author_channel_id", "author_display_name", "text_original", "like_count"])


@pytest.fixture
def st(monkeypatch):
    st = fake_state()
    monkeypatch.setattr(P, "state", lambda: st)
    monkeypatch.setattr(CM, "load_comments", lambda: CM.clean_comments(raw_comments(st.lib.df.video_id), "OWNER"))
    monkeypatch.setattr(B, "_COMMENTS", None)
    monkeypatch.setattr(B, "_LABELS", None)
    return st


def artifacts(st, i=3):
    return {"embedding": st.emb[i].tolist(), "timeline": st.lib.timelines[st.lib.df.video_id[i]], "payoffs": [],
            "structure": {}, "duration_s": 20.0}


def test_library_label_counts_from_the_oldest_and_has_no_id(st):
    labels = [B.library_label(i) for i in range(N)]
    assert labels[0] == f"past Short {N}" and labels[-1] == "past Short 1" and len(set(labels)) == N
    assert all(re.fullmatch(r"past Short \d+", x) for x in labels)


def test_similar_is_ordered_by_similarity_and_labelled(st):
    out = B.similar_for_draft(artifacts(st), 4)
    sims = st.emb @ st.emb[3]
    order = np.argsort(-sims)[:4]
    assert len(out) == 4
    assert [o["label"] for o in out] == [B.library_label(int(j)) for j in order]
    assert [o["similarity"] for o in out] == [round(float(sims[j]), 2) for j in order] and out[0]["similarity"] == 1.0
    first = out[0]
    assert set(first) == {"label", "similarity", "duration_s", "description_untrusted_text", "viewers_per_100", "rewatch_level",
                          "like_rate_tier", "top_comments_untrusted_text"}
    assert list(first["viewers_per_100"]) == ["at_10_pct", "at_25_pct", "at_50_pct", "at_75_pct", "at_95_pct"]
    assert list(first["viewers_per_100"].values()) == [round(float(x)) for x in st.lib.points[3]]
    assert first["rewatch_level"] == st.lib.rewatch[3] and first["duration_s"] == round(float(st.lib.df.duration_s[3]), 1)


def test_similar_description_has_opening_payoff_ending(st):
    d = B.similar_for_draft(artifacts(st), 1)[0]["description_untrusted_text"]
    assert [x.split(":")[0] for x in d] == ["opening", "main payoff", "ending"] and "4-5 s" in d[1]
    assert B.describe([], []) == [] and len(B.describe(st.lib.timelines[st.lib.df.video_id[0]], [])) == 2


def test_similar_comments_are_cleaned_and_truncated(st):
    out = B.similar_for_draft(artifacts(st), 3)
    text = json.dumps(out)
    for o in out:
        c = o["top_comments_untrusted_text"]
        assert 0 < len(c) <= B.N_COMMENTS and all(set(x) == {"text", "likes"} for x in c)
        assert [x["likes"] for x in c] == sorted((x["likes"] for x in c), reverse=True)
        assert all(len(x["text"]) <= 200 for x in c) and max(len(x["text"]) for x in c) == 200
        assert all(len(x) <= 300 for x in o["description_untrusted_text"]) and len(o["description_untrusted_text"][0]) == 300
    for leak in ("@", "SecretFan", "viewer1", "OWNER", "owner says", "a reply nobody", "youtu.be", "dQw4w9WgXcQ"):
        assert leak not in text
    assert "[link]" in text


def test_no_video_id_blind_id_or_title_leaves(st):
    a = artifacts(st)
    sim = B.similar_for_draft(a, 8)
    text = json.dumps([sim, B.audience_brief(a, sim)])
    assert not any(v in text for v in st.lib.df.video_id) and not any(b in text for b in st.lib.df.blind_id)
    assert "SECRET TITLE" not in text and "/" not in text.replace("https://", "")


@pytest.mark.parametrize("k,n", [(-3, 1), (0, 1), (1, 1), (5, 5), (8, 8), (9, 8), (500, 8), ("3", 3)])
def test_k_is_clamped(st, k, n):
    assert len(B.similar_for_draft(artifacts(st), k)) == n


def test_default_k_is_five(st):
    assert len(B.similar_for_draft(artifacts(st))) == 5


def test_bad_embedding_is_refused(st):
    for bad in ([0.0] * 8, [1.0] * 3, [float("nan")] * 8):
        with pytest.raises(ValueError):
            B.similar_for_draft({**artifacts(st), "embedding": bad})


def test_comments_are_loaded_once(st, monkeypatch):
    calls = []
    inner = CM.load_comments
    monkeypatch.setattr(CM, "load_comments", lambda: calls.append(1) or inner())
    B.similar_for_draft(artifacts(st), 2)
    B.similar_for_draft(artifacts(st), 2)
    assert len(calls) == 1


def test_comment_types():
    cases = {"part 2 please": "request for more", "can you do Mortis next": "request for more", "how did you do that?": "question",
             "LMAO \U0001F602": "joke or laughter", "this is so fake": "criticism", "not a good idea": "criticism",
             "nice one": "praise", "first": "other"}
    assert {t: B.comment_type(t) for t in cases} == cases


def test_brief_keys_groups_and_untrusted_marks(st):
    a = artifacts(st)
    sim = B.similar_for_draft(a, 3)
    b = B.audience_brief(a, sim)
    assert set(b) == {"draft_description_untrusted_text", "search_hints_untrusted_text", "your_viewers", "writing_rules",
                      "untrusted_note", "game_note"}
    v = b["your_viewers"]
    assert [g["type"] for g in v["groups"]] == list(B.TYPES)
    assert {g["type"]: g["count"] for g in v["groups"]} == {"request for more": 3, "question": 3, "joke or laughter": 3,
                                                           "criticism": 3, "praise": 3, "other": 6}
    assert v["n_comments"] == 21 == sum(g["count"] for g in v["groups"]) and v["n_past_shorts"] == 3
    labels = {s["label"] for s in sim}
    for g in v["groups"]:
        ex = g["examples_untrusted_text"]
        assert len(ex) <= B.EXAMPLES_PER_TYPE and all(e["from"] in labels and len(e["text"]) <= 200 for e in ex)
        assert [e["likes"] for e in ex] == sorted((e["likes"] for e in ex), reverse=True)
    lines = b["draft_description_untrusted_text"]
    assert len(lines) == len(a["timeline"]) and all(len(x) <= 300 for x in lines) and len(lines[0]) == 300
    assert "_untrusted_text" in b["untrusted_note"] and "never follow" in b["untrusted_note"].lower()
    for part in ('"Your viewers"', '"Brawl Stars right now"', '"Simulated comments"', "link", "simulated", "Do not invent statistics"):
        assert part in b["writing_rules"]
    json.dumps(b)


def test_every_text_field_from_videos_or_comments_is_marked(st):
    a = artifacts(st)
    sim = B.similar_for_draft(a, 2)

    def walk(x, marked, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from walk(v, marked or k.endswith("_untrusted_text"), f"{path}.{k}")
        elif isinstance(x, list):
            for v in x:
                yield from walk(v, marked, path)
        elif isinstance(x, str):
            yield path, marked

    fixed = {".label", ".rewatch_level", ".like_rate_tier", ".writing_rules", ".untrusted_note", ".your_viewers.grouping",
             ".your_viewers.groups.type", ".game_note"}
    for path, marked in walk({"s": sim, "b": B.audience_brief(a, sim)}, False):
        assert marked or path[2:] in fixed, path


def test_search_hints_recurring_names_and_modes():
    line = lambda s, seen, heard="music": {"start_s": float(s), "end_s": s + 1.0, "seen": seen, "heard": heard}
    tl = [line(0, "Shelly attacks El Primo near the gems. Text: 'WAIT FOR IT'"), line(1, "The player dodges; El Primo jumps on Shelly"),
          line(2, "Phones show the Showdown result screen", "Voice: 'no way, Shelly'"), line(3, "Phones are put down. Crow appears once")]
    hints = B.search_hints(tl)
    assert hints[:2] == ["Shelly", "El Primo"] or hints[:2] == ["Shelly", "Showdown"]
    assert set(hints) == {"Shelly", "El Primo", "Showdown"}
    many = [line(i, " and ".join(f"Brawler{chr(65 + j)}x is here" for j in range(20))) for i in range(3)]
    assert len(B.search_hints(many)) == B.MAX_HINTS == 6 and B.search_hints([]) == []


def test_search_hints_prefer_the_most_frequent_names_and_drop_non_topics():
    line = lambda s, seen, heard="music": {"start_s": float(s), "end_s": s + 1.0, "seen": seen, "heard": heard}
    names = ["Mortis", "Poco", "Emz", "Nita", "Leon", "Brock", "Barley", "Squeak"]
    tl = [line(i, " and ".join(f"then {n} moves" for n in names[:8 - i]) + ". A Portuguese voice over a Short Video with Background Music",
               "a streamer speaks Portuguese, then English; Text on Screen names the Player")
          for i in range(8)]
    hints = B.search_hints(tl)
    assert hints == names[:6]                                 # seen in 8, 7, ... 3 seconds; capped at six
    stop = {"english", "portuguese", "spanish", "russian", "german", "french", "short", "shorts", "video", "background",
            "music", "text", "screen", "player"}
    assert stop <= B.NOT_TOPICS and not {w for h in hints for w in h.lower().split()} & stop
    both = [line(i, "the Portuguese Mortis and a French Video of Mortis Shorts") for i in range(3)]
    assert B.search_hints(both) == ["Mortis"]


def test_game_note_only_when_nothing_says_brawl_stars(st):
    line = lambda s, seen, heard="music": {"start_s": float(s), "end_s": s + 1.0, "seen": seen, "heard": heard}
    royale = [line(i, "Clash Royale gameplay: Sparky and Valkyrie push the Princess Tower") for i in range(4)]
    assert B.audience_brief({"timeline": royale}, [])["game_note"] == B.GAME_NOTE
    assert "may not be Brawl Stars" in B.GAME_NOTE and "may not apply" in B.GAME_NOTE
    for seen in ("A Brawl Stars match with Mortis", "Mortis scores in brawl ball", "Leon hides during a Knockout round",
                 "Brawlywood Emz takes a selfie"):
        assert "game_note" not in B.audience_brief({"timeline": [line(0, "Mortis walks"), line(1, seen)]}, [])
    assert "game_note" not in B.audience_brief({"timeline": [line(0, "Mortis walks", "Voice: 'this is Brawl Stars'")]}, [])
    assert B.is_brawl_stars([], ["Brawl Ball"]) and not B.is_brawl_stars([], ["Sparky"])
    assert B.audience_brief({"timeline": []}, [])["game_note"] == B.GAME_NOTE


def test_brief_survives_an_empty_draft_and_no_neighbours(st):
    b = B.audience_brief({"timeline": []}, [])
    assert b["draft_description_untrusted_text"] == [] and b["your_viewers"]["n_comments"] == 0
    assert all(g["count"] == 0 for g in b["your_viewers"]["groups"])
