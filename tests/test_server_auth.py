"""Access tokens, the failure limiter, the admin CLI and config overrides (plan/10a sections 1-2)."""

import hashlib
import hmac
import logging
import sqlite3
from pathlib import Path

import pytest

from src.server import auth, config
from src.server.auth import FailureLimiter, Tokens


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def tokens(tmp_path, clock):
    return Tokens(tmp_path / "db" / "server.sqlite3", clock)


def _dump(db_path: Path) -> str:
    con = sqlite3.connect(db_path)
    try:
        return repr(con.execute("SELECT * FROM tokens").fetchall())
    finally:
        con.close()


def test_create_then_verify(tokens, clock):
    token_id, plain = tokens.create_token("  Alfie   test ")
    info = tokens.verify(plain)
    assert info is not None and info.id == token_id and info.label == "Alfie test"
    assert info.created_at == clock.t and not info.revoked
    assert len(plain) == 43 and plain != token_id and len(token_id) == 8


def test_tokens_are_unique_and_independent(tokens):
    (id_a, a), (id_b, b) = tokens.create_token("a"), tokens.create_token("b")
    assert a != b and id_a != id_b
    assert tokens.verify(a).id == id_a and tokens.verify(b).id == id_b


def test_only_the_hash_is_stored(tokens):
    _, plain = tokens.create_token("a")
    stored = _dump(tokens.db_path)
    assert plain not in stored
    assert hashlib.sha256(plain.encode()).hexdigest() in stored
    for suffix in ("", "-wal"):
        path = Path(str(tokens.db_path) + suffix)
        assert not path.exists() or plain.encode() not in path.read_bytes()


def test_database_file_is_private(tokens):
    assert tokens.db_path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("bad", ["", "wrong", "x" * 43, "x" * 5000, None, 12, b"bytes", "caf\u00e9", "\udcff"],
                         ids=["empty", "wrong", "right-length", "huge", "none", "int", "bytes", "non-ascii", "surrogate"])
def test_verify_rejects_unknown_and_malformed(tokens, bad):
    tokens.create_token("a")
    assert tokens.verify(bad) is None


def test_verify_rejects_near_misses(tokens):
    _, plain = tokens.create_token("a")
    assert tokens.verify(plain[:-1]) is None
    assert tokens.verify(plain + "A") is None
    assert tokens.verify(plain.swapcase()) is None
    assert tokens.verify(hashlib.sha256(plain.encode()).hexdigest()) is None


def test_verify_compares_in_constant_time(tokens, monkeypatch):
    _, first = tokens.create_token("a")
    tokens.create_token("b")
    tokens.create_token("c")
    calls, real = [], hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    assert tokens.verify(first) is not None
    assert len(calls) == 3, "every stored hash is compared, with no early exit on a match"
    calls.clear()
    assert tokens.verify("not-a-token") is None
    assert len(calls) == 3
    assert all(len(a) == len(b) == 64 for a, b in calls)


def test_revoke(tokens, clock):
    token_id, plain = tokens.create_token("a")
    _, other = tokens.create_token("b")
    clock.t += 5
    assert tokens.revoke(token_id) is True
    assert tokens.verify(plain) is None
    assert tokens.verify(other) is not None
    assert tokens.revoke(token_id) is False
    assert tokens.revoke("nope") is False
    revoked = [t for t in tokens.list_tokens() if t.id == token_id][0]
    assert revoked.revoked and revoked.revoked_at == clock.t


def test_list_tokens_has_no_secrets(tokens, clock):
    _, a = tokens.create_token("first")
    clock.t += 1
    _, b = tokens.create_token("second")
    listed = tokens.list_tokens()
    assert [t.label for t in listed] == ["first", "second"]
    text = repr(listed)
    assert a not in text and b not in text and hashlib.sha256(a.encode()).hexdigest() not in text


@pytest.mark.parametrize("label", ["", "   ", "x" * 81], ids=["empty", "blank", "long"])
def test_label_is_validated(tokens, label):
    with pytest.raises(ValueError):
        tokens.create_token(label)


def test_token_never_logged(tokens, caplog):
    with caplog.at_level(logging.DEBUG):
        _, plain = tokens.create_token("a")
        tokens.verify(plain)
        tokens.verify("wrong")
    assert plain not in caplog.text


def test_module_level_functions(tmp_path):
    db = tmp_path / "s.sqlite3"
    token_id, plain = auth.create_token("a", db)
    assert auth.verify(plain, db).id == token_id
    assert [t.id for t in auth.list_tokens(db)] == [token_id]
    assert auth.revoke(token_id, db) and auth.verify(plain, db) is None


def test_limiter_blocks_after_max_failures(clock):
    limiter = FailureLimiter(max_failures=3, window_s=600, clock=clock)
    for _ in range(2):
        limiter.record_failure("1.2.3.4")
    assert not limiter.blocked("1.2.3.4")
    limiter.record_failure("1.2.3.4")
    assert limiter.blocked("1.2.3.4")
    assert not limiter.blocked("5.6.7.8")


def test_limiter_window_slides(clock):
    limiter = FailureLimiter(max_failures=2, window_s=600, clock=clock)
    limiter.record_failure("a")
    clock.t += 400
    limiter.record_failure("a")
    assert limiter.blocked("a")
    clock.t += 199
    assert limiter.blocked("a")
    clock.t += 1
    assert not limiter.blocked("a"), "the first failure is 600 s old"
    limiter.record_failure("a")
    assert limiter.blocked("a")
    clock.t += 600
    assert not limiter.blocked("a")
    assert limiter._failures == {}


def test_limiter_stays_blocked_while_failures_continue(clock):
    limiter = FailureLimiter(max_failures=2, window_s=10, clock=clock)
    for _ in range(50):
        limiter.record_failure("a")
        clock.t += 1
    assert limiter.blocked("a")
    assert len(limiter._failures["a"]) <= 2


def test_limiter_memory_is_bounded(clock):
    limiter = FailureLimiter(max_failures=2, window_s=600, clock=clock, max_addresses=5)
    for i in range(50):
        limiter.record_failure(f"addr{i}")
        clock.t += 1
    assert len(limiter._failures) <= 5
    assert "addr49" in limiter._failures and "addr0" not in limiter._failures


def test_limiter_defaults_come_from_config():
    limiter = FailureLimiter()
    assert (limiter.max_failures, limiter.window_s) == (config.BAD_TOKEN_MAX, config.BAD_TOKEN_WINDOW_S)


def test_cli_create_list_revoke(tmp_path, capsys):
    db = tmp_path / "cli.sqlite3"
    assert auth.main(["--db", str(db), "create", "--label", "tester one"]) == 0
    out = capsys.readouterr().out
    token_id = out.split("token id:")[1].split()[0]
    plain = out.split("token:")[1].split()[0]
    assert out.count(plain) == 1
    assert auth.verify(plain, db).id == token_id

    assert auth.main(["--db", str(db), "list"]) == 0
    listed = capsys.readouterr().out
    assert token_id in listed and "tester one" in listed and plain not in listed

    assert auth.main(["--db", str(db), "revoke", token_id]) == 0
    assert auth.verify(plain, db) is None
    capsys.readouterr()
    assert auth.main(["--db", str(db), "revoke", token_id]) == 1
    assert auth.main(["--db", str(db), "revoke", "missing"]) == 1
    assert plain not in capsys.readouterr().out


def test_config_defaults_match_the_spec():
    cfg = config.Config()
    assert cfg.UPLOAD_DIR == Path("data/server/uploads") and cfg.DB_PATH == Path("data/server/server.sqlite3")
    assert (cfg.MIN_DURATION_S, cfg.MAX_DURATION_S, cfg.MAX_UPLOAD_BYTES) == (8.0, 60.0, 100 * 1024 * 1024)
    assert cfg.ALLOWED_CONTAINERS == ("mp4", "mov", "webm")
    assert (cfg.UPLOAD_TTL_S, cfg.RESULT_TTL_S) == (1800, 7 * 86400)
    assert (cfg.DRAFTS_PER_DAY, cfg.OWN_KEY_DRAFTS_PER_DAY) == (3, 10)
    assert (cfg.DAY_CAP_USD, cfg.MONTH_CAP_USD, cfg.EST_COST_PER_DRAFT_USD) == (1.0, 10.0, 0.12)
    assert (cfg.MAX_JOBS_TOTAL, cfg.MAX_JOBS_PER_USER) == (2, 1)
    assert (cfg.BAD_TOKEN_MAX, cfg.BAD_TOKEN_WINDOW_S) == (10, 600)
    assert cfg.PUBLIC_BASE_URL is None


def test_config_load_reads_prefixed_variables(tmp_path):
    cfg = config.load({"SO_DATA_DIR": str(tmp_path), "SO_DRAFTS_PER_DAY": "5", "SO_DAY_CAP_USD": "2.5",
                       "SO_ALLOWED_CONTAINERS": "mp4, MOV", "SO_PUBLIC_BASE_URL": "https://x.example/",
                       "DRAFTS_PER_DAY": "99", "SO_MONTH_CAP_USD": "  "})
    assert cfg.DATA_DIR == tmp_path and cfg.UPLOAD_DIR == tmp_path / "uploads"
    assert cfg.DB_PATH == tmp_path / "server.sqlite3"
    assert cfg.DRAFTS_PER_DAY == 5 and cfg.DAY_CAP_USD == 2.5 and cfg.MONTH_CAP_USD == 10.0
    assert cfg.ALLOWED_CONTAINERS == ("mp4", "mov") and cfg.PUBLIC_BASE_URL == "https://x.example"
    assert config.load({}, DRAFTS_PER_DAY=1).DRAFTS_PER_DAY == 1
    assert config.load({"SO_DB_PATH": str(tmp_path / "x.db")}).DB_PATH == tmp_path / "x.db"


@pytest.mark.parametrize("env", [{"SO_DRAFTS_PER_DAY": "many"}, {"SO_DAY_CAP_USD": "-1"}, {"SO_DAY_CAP_USD": "nan"},
                                 {"SO_MIN_DURATION_S": "90"}, {"SO_ALLOWED_CONTAINERS": ","}])
def test_config_rejects_bad_values(env):
    with pytest.raises(config.ConfigError):
        config.load(env)
