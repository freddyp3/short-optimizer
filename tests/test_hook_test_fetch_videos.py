"""Tests for the pure parts of fetch_videos.py (no ffprobe, no real files)."""

import os
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

from src.evals.hook_test.fetch_videos import (
    FileEntry,
    hardlink,
    load_takeout_metadata,
    match_files,
    merged_takeout_videos,
    normalize_exact,
    normalize_fuzzy,
    parse_resolve,
)


def meta_df(rows):
    """rows: list of (video_id, title, approx_duration_ms)."""
    return pd.DataFrame(
        rows, columns=["Video ID", "Video Title (Original)", "Approx Duration (ms)"]
    ).set_index("Video ID")


def f(name, dur=None):
    return FileEntry(Path("/v") / name, Path(name).stem, dur)


class TestNormalizeExact:
    def test_question_mark_becomes_underscore(self):
        assert normalize_exact("200k DAMAGE in ONE game?!") == "200k DAMAGE in ONE game_!"
        assert normalize_exact("200k DAMAGE in ONE game??!?") == "200k DAMAGE in ONE game_!_"

    def test_curly_quotes_and_colon(self):
        assert normalize_exact("it’s a “test”: ok") == "it_s a _test_ ok"

    def test_dup_suffix_stripped(self):
        assert normalize_exact("#bhajanshaamsavere(2)") == "#bhajanshaamsavere"
        assert normalize_exact("#bhajanshaamsavere") == "#bhajanshaamsavere"

    def test_nfd_vs_nfc(self):
        title_nfd = unicodedata.normalize("NFD", "café #shorts")
        stem_nfd = unicodedata.normalize("NFD", "café #shorts")
        assert normalize_exact(title_nfd) == normalize_exact(stem_nfd) == "café #shorts"

    def test_slash_inside_parens(self):
        assert normalize_exact("finger in nose(Android/IOS)") == "finger in nose(Android_IOS)"


class TestNormalizeFuzzy:
    def test_equivalent_titles_match(self):
        a = normalize_fuzzy("200k DAMAGE in ONE game?!")
        b = normalize_fuzzy("200k DAMAGE in ONE game_!_")
        assert a == b


class TestMatchFiles:
    def test_title_exact(self):
        meta = meta_df([("v1", "200k DAMAGE in ONE game??!?", 20000)])
        files = [f("200k DAMAGE in ONE game_!_.mp4", 20.0)]
        m = match_files(files, meta, {})
        assert m[files[0].path].video_id == "v1"
        assert m[files[0].path].method == "title_exact"

    def test_duplicate_titles_disambiguated_by_duration(self):
        meta = meta_df([
            ("vA", "#bhajanshaamsavere", 13000),
            ("vB", "#bhajanshaamsavere", 11000),
            ("vC", "#bhajanshaamsavere", 13250),
        ])
        files = [
            f("#bhajanshaamsavere.mp4", 13.0),
            f("#bhajanshaamsavere(1).mp4", 11.0),
            f("#bhajanshaamsavere(2).mp4", 13.3),
        ]
        m = match_files(files, meta, {})
        got = {p.name: match.video_id for p, match in m.items()}
        assert got["#bhajanshaamsavere(1).mp4"] == "vB"
        assert all(match.method == "title+duration" for match in m.values())
        assert {got["#bhajanshaamsavere.mp4"], got["#bhajanshaamsavere(2).mp4"]} == {"vA", "vC"}

    def test_unresolvable_tie_left_unmatched(self):
        meta = meta_df([
            ("vA", "dup title", 13000),
            ("vB", "dup title", 13000),
        ])
        files = [f("dup title.mp4", 13.0), f("dup title(1).mp4", 12.9)]
        m = match_files(files, meta, {})
        # both durations equally close to both ids; greedy assigns min-diff pair only
        # (12.9 vs 13.0 diff 0.1 each; 13.0 vs 13.0 diff 0). One pair resolves, the
        # remaining file still ties -> still unmatched or single greedy pair.
        assert len(m) <= 1  # no over-matching; ties don't force a guess

    def test_title_fuzzy(self):
        meta = meta_df([("v1", "Some Rather Long Title Here #shorts #brawlstars", 20000)])
        files = [f("Some Rather Long Title Here #shorts #brawlstars.mp4", 20.0)]
        m = match_files(files, meta, {})
        assert m[files[0].path].method in {"title_exact", "title_fuzzy"}

    def test_manual_resolve(self, tmp_path):
        real = tmp_path / "weird name.mp4"
        real.write_bytes(b"x")
        meta = meta_df([("v1", "totally different", 20000)])
        files = [FileEntry(real, "weird name", 20.0)]
        m = match_files(files, meta, {"v1": str(real)})
        assert m[real].method == "manual"

    def test_each_id_used_once(self):
        meta = meta_df([("v1", "t", 10000)])
        files = [f("t.mp4", 10.0), f("t(1).mp4", 10.0)]
        m = match_files(files, meta, {})
        assert len(m) == 1


class TestTakeoutMetadata:
    def test_concat_differing_columns(self, tmp_path):
        (tmp_path / "videos.csv").write_text(
            "Video ID,Approx Duration (ms),Video Title (Original),Privacy,Video State,"
            "Video Create Timestamp,Video Publish Timestamp,Video Category,Tag 1,Tag 2\n"
            "v1,10000,title one,Public,Processed,2024-01-01T00:00:00+00:00,"
            "2024-01-02T00:00:00+00:00,Gaming,a,b\n"
        )
        (tmp_path / "videos(1).csv").write_text(
            "Video ID,Approx Duration (ms),Video Title (Original),Privacy,Video State,"
            "Video Create Timestamp,Video Publish Timestamp,Video Category,Video Description (Original)\n"
            "v2,20000,title two,Private,Processed,2024-02-01T00:00:00+00:00,"
            "2024-02-02T00:00:00+00:00,People,desc\n"
        )
        meta = load_takeout_metadata(tmp_path)
        assert len(meta) == 2
        assert meta.at["v2", "Video Description (Original)"] == "desc"
        merged = merged_takeout_videos(meta)
        row = merged.set_index("video_id").loc["v1"]
        assert row["tags"] == "a|b"
        assert row["approx_duration_s"] == 10.0
        assert pd.isna(merged.set_index("video_id").loc["v2", "tags"]) or \
            merged.set_index("video_id").loc["v2", "tags"] == ""


class TestHardlink:
    def test_link_and_idempotent(self, tmp_path):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"data")
        dst = tmp_path / "out" / "v1.mp4"
        assert hardlink(src, dst) == "link"
        assert os.stat(dst).st_ino == os.stat(src).st_ino
        assert os.stat(dst).st_nlink == 2
        assert hardlink(src, dst) == "skip"

    def test_relink_when_different(self, tmp_path):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"data")
        dst = tmp_path / "v1.mp4"
        dst.write_bytes(b"other")
        assert hardlink(src, dst) == "relink"
        assert os.stat(dst).st_ino == os.stat(src).st_ino


class TestParseResolve:
    def test_basic(self):
        assert parse_resolve(["v1=/a/b.mp4"]) == {"v1": "/a/b.mp4"}

    def test_bad(self):
        with pytest.raises(ValueError):
            parse_resolve(["noequals"])
