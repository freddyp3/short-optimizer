import numpy as np
import pandas as pd

from src.tools.audience import comments as A


def raw():
    return pd.DataFrame({
        "video_id": ["a", "a", "a", "a", "b"], "is_reply": [False, True, False, False, False],
        "author_channel_id": ["u1", "u2", "OWNER", "u3", "u4"], "author_display_name": ["@x"] * 5,
        "text_original": ["@alfie369k so lucky", "a reply", "thanks!", "first", "@someone"],
        "like_count": [2, 9, 50, 7, 1]})


def test_clean_drops_replies_owner_mentions_and_empty():
    c = A.clean_comments(raw(), "OWNER")
    assert c["text"].tolist() == ["first", "so lucky"] and c["like_count"].tolist() == [7, 2]
    assert list(c.columns) == ["video_id", "text", "like_count"]


def test_top_comments_truncates():
    c = pd.DataFrame({"video_id": ["a"] * 3, "text": ["x" * 500, "y", "z"], "like_count": [3, 2, 1]})
    out = A.top_comments(c, "a", n=2, max_chars=10)
    assert out == [{"text": "x" * 10, "likes": 3}, {"text": "y", "likes": 2}]


def test_neighbour_test_finds_planted_structure_and_ignores_noise():
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(6), 10)
    base = rng.normal(size=(6, 16))
    cent = base[group] + 0.3 * rng.normal(size=(60, 16))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    sim = (group[:, None] == group[None, :]).astype(float) + 0.01 * rng.random((60, 60))
    good = A.neighbour_test(sim, cent, np.ones(60, bool))
    assert good["difference"] > 0.2 and good["p"] < 0.01
    noise = A.neighbour_test(rng.random((60, 60)), cent, np.ones(60, bool))
    assert noise["p"] > 0.05
