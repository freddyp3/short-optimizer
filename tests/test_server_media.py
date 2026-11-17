"""Upload streaming and ffprobe checks on tiny generated videos (plan/10a section 5, plan/10 section 12.1)."""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from src.server import media
from src.server.config import Config
from src.server.media import CODES, MediaInfo, Rejected, save_stream, save_stream_async, validate

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg / ffprobe not installed")

X264 = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, stdin=subprocess.DEVNULL, timeout=120)


def _src(size: str, seconds: float, rate: int = 5) -> list[str]:
    return ["-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}", "-t", str(seconds)]


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    p = {name: d / name for name in (
        "good.mp4", "square.mp4", "short.mp4", "long.mp4", "landscape.mp4", "rotated_to_portrait.mp4",
        "rotated_to_landscape.mp4", "audio.mp4", "with_audio.mp4", "two_video.mp4", "good.mkv", "good.mov",
        "good.webm", "sound.mp3", "notes.mp4", "playlist.mp4", "empty.mp4", "truncated.mp4")}
    _ffmpeg(*_src("90x160", 9), *X264, str(p["good.mp4"]))
    _ffmpeg(*_src("96x96", 9), *X264, str(p["square.mp4"]))
    _ffmpeg(*_src("90x160", 3), *X264, str(p["short.mp4"]))
    _ffmpeg(*_src("32x64", 61, rate=1), *X264, str(p["long.mp4"]))
    _ffmpeg(*_src("160x90", 9), *X264, str(p["landscape.mp4"]))
    _ffmpeg("-display_rotation", "90", "-i", str(p["landscape.mp4"]), "-c", "copy", str(p["rotated_to_portrait.mp4"]))
    _ffmpeg("-display_rotation", "270", "-i", str(p["good.mp4"]), "-c", "copy", str(p["rotated_to_landscape.mp4"]))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=9", "-c:a", "aac", str(p["audio.mp4"]))
    _ffmpeg("-i", str(p["good.mp4"]), "-i", str(p["audio.mp4"]), "-c", "copy", str(p["with_audio.mp4"]))
    _ffmpeg("-i", str(p["good.mp4"]), "-map", "0:v", "-map", "0:v", "-c", "copy", str(p["two_video.mp4"]))
    _ffmpeg("-i", str(p["good.mp4"]), "-c", "copy", str(p["good.mkv"]))
    _ffmpeg("-i", str(p["good.mp4"]), "-c", "copy", str(p["good.mov"]))
    _ffmpeg(*_src("90x160", 9), "-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8", str(p["good.webm"]))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=9", str(p["sound.mp3"]))
    p["notes.mp4"].write_text("this is not a video\n" * 50)
    p["playlist.mp4"].write_text(f"#EXTM3U\n#EXT-X-TARGETDURATION:9\n#EXTINF:9,\n{p['good.mp4']}\n#EXT-X-ENDLIST\n")
    p["empty.mp4"].write_bytes(b"")
    p["truncated.mp4"].write_bytes(p["good.mp4"].read_bytes()[:200])
    return p


def _code(path: Path, cfg: Config | None = None) -> str:
    with pytest.raises(Rejected) as err:
        validate(path, cfg) if cfg else validate(path)
    assert_safe(err.value, path)
    return err.value.code


def assert_safe(exc: Rejected, path: Path) -> None:
    for text in (exc.user_message, str(exc), repr(exc.args)):
        assert str(path) not in text and path.name not in text and str(path.parent) not in text
        assert "/" not in text and "\\" not in text
    assert exc.user_message.endswith(".") and exc.user_message[0].isupper() and len(exc.user_message) < 200


def test_valid_portrait_mp4(clips):
    info = validate(clips["good.mp4"])
    assert isinstance(info, MediaInfo)
    assert (info.width, info.height, info.container) == (90, 160, "mp4")
    assert info.duration_s == pytest.approx(9.0, abs=0.3)
    assert info.size_bytes == clips["good.mp4"].stat().st_size


def test_valid_variants(clips):
    assert validate(clips["good.mov"]).container == "mov"
    assert validate(clips["good.webm"]).container == "webm"
    assert validate(clips["with_audio.mp4"]).container == "mp4"
    square = validate(clips["square.mp4"])
    assert square.width == square.height == 96


@pytest.mark.parametrize("name, code", [
    ("short.mp4", "too_short"), ("long.mp4", "too_long"), ("landscape.mp4", "landscape"),
    ("rotated_to_landscape.mp4", "landscape"), ("audio.mp4", "no_video_stream"),
    ("two_video.mp4", "several_video_streams"), ("good.mkv", "container_not_allowed"),
    ("notes.mp4", "unreadable"), ("empty.mp4", "unreadable"), ("truncated.mp4", "unreadable"),
    ("sound.mp3", "unreadable"), ("playlist.mp4", "unreadable"), ("missing.mp4", "unreadable")])
def test_rejections(clips, name, code):
    assert _code(clips.get(name, clips["good.mp4"].parent / name)) == code


def test_too_large(clips):
    size = clips["good.mp4"].stat().st_size
    assert _code(clips["good.mp4"], Config(MAX_UPLOAD_BYTES=size - 1)) == "too_large"
    assert validate(clips["good.mp4"], Config(MAX_UPLOAD_BYTES=size)).size_bytes == size


def test_every_code_is_covered_and_has_a_safe_sentence():
    assert set(CODES) == {"unreadable", "no_video_stream", "several_video_streams", "container_not_allowed",
                          "too_short", "too_long", "landscape", "too_large"}
    for code in CODES:
        assert_safe(Rejected(code), Path("/srv/uploads/abc.mp4"))
    assert "8 to 60 seconds" in Rejected("too_short").user_message
    assert "100 MB" in Rejected("too_large").user_message
    assert Rejected("too_long", "Custom sentence.").user_message == "Custom sentence."
    with pytest.raises(ValueError):
        Rejected("weird")


def test_rotation_flag_makes_a_landscape_stream_portrait(clips):
    info = validate(clips["rotated_to_portrait.mp4"])
    assert (info.width, info.height) == (90, 160)


@pytest.mark.parametrize("stream, expected", [
    ({"width": 1920, "height": 1080}, (1920, 1080)),
    ({"width": 1920, "height": 1080, "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]}, (1080, 1920)),
    ({"width": 1920, "height": 1080, "side_data_list": [{"rotation": 270}]}, (1080, 1920)),
    ({"width": 1080, "height": 1920, "side_data_list": [{"rotation": 180}]}, (1080, 1920)),
    ({"width": 1920, "height": 1080, "tags": {"rotate": "90"}}, (1080, 1920)),
    ({"width": 1080, "height": 1920, "side_data_list": [{"side_data_type": "Other"}], "tags": {"rotate": "junk"}}, (1080, 1920)),
    ({"width": 0, "height": 1080}, None), ({"height": 1080}, None), ({"width": "x", "height": 1}, None)])
def test_display_size(stream, expected):
    assert media.display_size(stream) == expected


def _probe_json(**video) -> dict:
    return {"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "20.0", "tags": {"major_brand": "isom"}},
            "streams": [{"codec_type": "video", "width": 1080, "height": 1920, "duration": "20.0", **video},
                        {"codec_type": "audio"}]}


def test_inspect_on_parsed_output():
    assert media.inspect(_probe_json(), b"", 10) == MediaInfo(20.0, 1080, 1920, "mp4", 10)
    with pytest.raises(Rejected) as err:
        media.inspect(_probe_json(side_data_list=[{"rotation": 90}]), b"", 10)
    assert err.value.code == "landscape"

    cover = _probe_json()
    cover["streams"].append({"codec_type": "video", "width": 500, "height": 500, "disposition": {"attached_pic": 1}})
    assert media.inspect(cover, b"", 10).width == 1080
    cover["streams"].pop(0)
    with pytest.raises(Rejected) as err:
        media.inspect(cover, b"", 10)
    assert err.value.code == "no_video_stream"


@pytest.mark.parametrize("change, code", [
    ({"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "N/A"}}, "unreadable"),
    ({"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "nan"}}, "unreadable"),
    ({"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "-5"}}, "unreadable"),
    ({"format": {"format_name": "avi", "duration": "20"}}, "unreadable"),
    ({"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "20", "tags": {"major_brand": "3gp4"}}}, "container_not_allowed"),
    ({"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "60.01"}}, "too_long"),
    ({"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "7.99"}}, "too_short"),
    ({"format": "nonsense"}, "unreadable"), ({"streams": []}, "unreadable"), ({"streams": "x"}, "unreadable")])
def test_inspect_rejects(change, code):
    data = {**_probe_json(), **change}
    if isinstance(data["streams"], list) and data["streams"]:
        del data["streams"][0]["duration"]
    with pytest.raises(Rejected) as err:
        media.inspect(data, b"", 10)
    assert err.value.code == code


def test_duration_falls_back_to_the_video_track():
    data = _probe_json()
    del data["format"]["duration"]
    assert media.inspect(data, b"", 10).duration_s == 20.0


def test_length_bounds_are_inclusive():
    for seconds in ("8.0", "60.0"):
        data = _probe_json(duration=seconds)
        data["format"]["duration"] = seconds
        assert media.inspect(data, b"", 1).duration_s == float(seconds)


def test_short_video_track_under_long_audio_is_too_short():
    data = _probe_json(duration="2.0")
    data["format"]["duration"] = "30.0"
    with pytest.raises(Rejected) as err:
        media.inspect(data, b"", 1)
    assert err.value.code == "too_short"


def test_the_longer_of_container_and_video_track_counts_for_the_maximum():
    data = _probe_json(duration="61.0")                      # the container says 20 s, the video track runs longer
    data["format"]["duration"] = "59.5"
    with pytest.raises(Rejected) as err:
        media.inspect(data, b"", 1)
    assert err.value.code == "too_long"
    data = _probe_json(duration="600.0")
    with pytest.raises(Rejected) as err:
        media.inspect(data, b"", 1)
    assert err.value.code == "too_long"


@pytest.mark.parametrize("container, video, code", [
    ("20.0", "22.0", None), ("22.0", "20.0", None), ("19.63", "19.63", None),
    ("20.0", "22.01", "unreadable"), ("30.0", "20.0", "unreadable"), ("9.0", "59.0", "unreadable")])
def test_container_and_video_track_must_agree_within_two_seconds(container, video, code):
    data = _probe_json(duration=video)
    data["format"]["duration"] = container
    if code is None:
        assert media.inspect(data, b"", 1).duration_s == float(container)
    else:
        with pytest.raises(Rejected) as err:
            media.inspect(data, b"", 1)
        assert err.value.code == code
    assert media.MAX_DURATION_MISMATCH_S == 2.0


def test_an_audio_track_of_another_length_does_not_count_as_a_mismatch(clips):
    data = _probe_json()
    data["streams"][1]["duration"] = "45.0"                  # only the video track is compared with the container
    data["streams"].insert(0, {"codec_type": "audio", "duration": "3.0"})
    assert media.inspect(data, b"", 1).duration_s == 20.0
    assert validate(clips["with_audio.mp4"]).duration_s == pytest.approx(9.0, abs=0.3)
    assert validate(clips["good.webm"]).duration_s == pytest.approx(9.0, abs=0.3)      # no track duration in webm


def test_ffprobe_is_called_without_a_shell_and_with_a_timeout(clips, monkeypatch):
    seen = {}
    real = subprocess.run

    def spy(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        return real(cmd, **kwargs)

    monkeypatch.setenv("SECRET_FOR_TEST", "do-not-pass-on")
    monkeypatch.setattr(media.subprocess, "run", spy)
    validate(clips["good.mp4"])
    assert isinstance(seen["cmd"], list) and seen["cmd"][0] == "ffprobe"
    assert seen["shell"] is False and seen["timeout"] == 20 and seen["stdin"] == subprocess.DEVNULL
    assert seen["cmd"][-1] == "file:" + str(clips["good.mp4"].resolve())
    for flag in ("-print_format", "json", "-show_format", "-show_streams", "-protocol_whitelist", "-format_whitelist"):
        assert flag in seen["cmd"]
    assert list(seen["env"]) == ["PATH"]


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired("ffprobe", 20), FileNotFoundError("ffprobe"), PermissionError("ffprobe"),
    subprocess.CompletedProcess([], 1, b"", b""), subprocess.CompletedProcess([], 0, b"not json", b""),
    subprocess.CompletedProcess([], 0, b"[1, 2]", b""), subprocess.CompletedProcess([], 0, b"{}", b"")],
    ids=["timeout", "missing-tool", "not-executable", "error-exit", "bad-json", "wrong-json", "empty-json"])
def test_ffprobe_failure_is_unreadable(clips, monkeypatch, failure):
    def fake(cmd, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(media.subprocess, "run", fake)
    assert _code(clips["good.mp4"]) == "unreadable"


def test_odd_file_names_are_not_read_as_options(clips, tmp_path, monkeypatch):
    odd = tmp_path / "-version"
    shutil.copy(clips["good.mp4"], odd)
    monkeypatch.chdir(tmp_path)
    assert validate(Path("-version")).container == "mp4"


def test_save_stream_writes_the_file(tmp_path):
    dest = tmp_path / "up.bin"
    assert save_stream([b"abc", b"", bytearray(b"de"), memoryview(b"f")], dest, 100) == 6
    assert dest.read_bytes() == b"abcdef" and dest.stat().st_mode & 0o077 == 0
    assert save_stream(iter([b"x" * 10]), tmp_path / "exact.bin", 10) == 10
    assert save_stream([], tmp_path / "none.bin", 10) == 0


def test_save_stream_cuts_off_and_deletes_at_the_limit(tmp_path):
    dest, pulled = tmp_path / "big.bin", []

    def chunks():
        for i in range(100):
            pulled.append(i)
            yield b"x" * 1000

    with pytest.raises(Rejected) as err:
        save_stream(chunks(), dest, 2500)
    assert err.value.code == "too_large"
    assert_safe(err.value, dest)
    assert not dest.exists() and list(tmp_path.iterdir()) == []
    assert len(pulled) == 3, "nothing is read after the limit is crossed"


def test_save_stream_deletes_the_partial_file_when_the_source_fails(tmp_path):
    dest = tmp_path / "partial.bin"

    def chunks():
        yield b"abc"
        raise ConnectionError("client went away")

    with pytest.raises(ConnectionError):
        save_stream(chunks(), dest, 100)
    assert not dest.exists()
    with pytest.raises(TypeError):
        save_stream(["text"], dest, 100)
    assert not dest.exists()


def test_save_stream_never_overwrites(tmp_path):
    dest = tmp_path / "taken.bin"
    dest.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        save_stream([b"new"], dest, 100)
    assert dest.read_bytes() == b"original"
    target, link = tmp_path / "target.bin", tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(OSError):
        save_stream([b"new"], link, 100)
    assert not target.exists()


def test_save_stream_async(tmp_path):
    async def chunks(n: int, pulled: list):
        for i in range(n):
            pulled.append(i)
            yield b"y" * 1000

    async def failing():
        yield b"abc"
        raise ConnectionError("client went away")

    ok, big, broken, pulled = tmp_path / "ok.bin", tmp_path / "big.bin", tmp_path / "broken.bin", []
    assert asyncio.run(save_stream_async(chunks(3, []), ok, 3000)) == 3000
    assert ok.read_bytes() == b"y" * 3000
    with pytest.raises(Rejected) as err:
        asyncio.run(save_stream_async(chunks(100, pulled), big, 2500))
    assert err.value.code == "too_large" and not big.exists() and len(pulled) == 3
    with pytest.raises(ConnectionError):
        asyncio.run(save_stream_async(failing(), broken, 100))
    assert not broken.exists()


def test_saved_upload_validates(clips, tmp_path):
    data = clips["good.mp4"].read_bytes()
    dest = tmp_path / "0123abcd"
    save_stream((data[i:i + 4096] for i in range(0, len(data), 4096)), dest, len(data))
    assert validate(dest).container == "mp4", "the stored name has no extension; the content decides"
