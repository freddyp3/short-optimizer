"""The always-on front and the home server behind it: sign-in at the front, tools at home, offline answers."""

import json

import pytest

from src.server import app as A
from src.server import front, oauth
from test_server_app import ACCEPT, BASE, Env, client_of
from test_server_oauth import FakeGoogle, bearer, sign_in

SECRET = "s" * 40


class Wire:
    """What stands between the two servers in a test: the front's ``post`` calls the home app, or nothing answers."""

    def __init__(self):
        self.home, self.online, self.sent = None, True, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.sent.append({"url": url, "json": json, "headers": dict(headers)})
        if not self.online:
            raise ConnectionError("no route to host")
        return self.home.post("/mcp", json=json, headers=headers)


@pytest.fixture
def pair(tmp_path):
    wire = Wire()
    home = Env(tmp_path / "home", proxy_secret=SECRET)
    fr = Env(tmp_path / "front", google=FakeGoogle(), auto_approve=("owner@example.com",),
             upstream=front.Upstream("https://home.example", SECRET, post=wire.post))
    with client_of(home) as hc, client_of(fr) as fc:
        wire.home = hc
        yield fr, fc, home, hc, wire


def test_a_tool_call_at_the_front_runs_at_home_as_the_signed_in_user(pair):
    fr, fc, home, hc, wire = pair
    got = sign_in(fc, "g-owner")
    start = fr.call(fc, "start_draft", headers=bearer(got["access_token"]))
    (user,) = fr.state.users.all()
    assert start["status"] == "waiting_for_upload" and start["upload_url"].startswith(BASE + "/u/")
    assert home.state.store.get_draft(start["draft_id"], user.id) is not None          # the draft lives at home, under the user
    assert fr.state.store.get_draft(start["draft_id"], user.id) is None
    sent = wire.sent[-1]
    assert sent["url"] == "https://home.example/mcp" and sent["headers"][front.USER_HEADER] == user.id
    assert sent["headers"]["Authorization"] == f"Bearer {SECRET}" and got["access_token"] not in json.dumps(sent)
    status = fr.call(fc, "get_draft_status", {"draft_id": start["draft_id"]}, headers=bearer(got["access_token"]))
    assert status["status"] == "waiting_for_upload"


def test_with_the_home_server_off_sign_in_still_works_and_tools_say_offline(pair):
    fr, fc, home, hc, wire = pair
    wire.online = False
    got = sign_in(fc, "g-owner")                                                       # signing up needs only the front
    assert [u.email for u in fr.state.users.all()] == ["owner@example.com"]
    listed = fr.rpc(fc, "tools/list", headers=bearer(got["access_token"])).json()["result"]["tools"]
    assert [t["name"] for t in listed] == list(A.TOOLS)
    out = fr.call(fc, "start_draft", headers=bearer(got["access_token"]))
    assert out == {"status": "offline", "message": front.MSG_OFFLINE}
    wire.online = True
    assert fr.call(fc, "start_draft", headers=bearer(got["access_token"]))["status"] == "waiting_for_upload"


def test_the_front_checks_arguments_and_sign_in_before_anything_is_passed_on(pair):
    fr, fc, home, hc, wire = pair
    got = sign_in(fc, "g-owner")
    bad = fr.call(fc, "get_draft_status", {"draft_id": "x", "extra": 1}, headers=bearer(got["access_token"]))
    assert bad["status"] == "invalid_input" and wire.sent == []
    assert fr.rpc(fc, "tools/call", {"name": "start_draft", "arguments": {}}, headers=bearer("wrong")).status_code == 401
    assert wire.sent == []


def test_a_home_answer_that_is_not_a_tool_result_counts_as_offline(tmp_path):
    class R:
        status_code = 502

        def json(self):
            return {}

    up = front.Upstream("https://home.example/", SECRET, post=lambda *a, **k: R())
    assert up.url == "https://home.example/mcp" and up.call("u_0123abcd", "start_draft", {}) is None
    with pytest.raises(ValueError):
        front.Upstream("https://home.example", "short")


def test_the_home_server_trusts_the_secret_with_a_user_id_and_nothing_else(tmp_path):
    home = Env(tmp_path, proxy_secret=SECRET)
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "start_draft", "arguments": {}}}

    def post(c, secret, user):
        headers = {"Authorization": f"Bearer {secret}", **ACCEPT, **({front.USER_HEADER: user} if user is not None else {})}
        return c.post("/mcp", json=call, headers=headers)

    with client_of(home) as c:
        ok = post(c, SECRET, "u_0123abcd")
        assert ok.status_code == 200
        draft = json.loads(ok.json()["result"]["content"][0]["text"])["draft_id"]
        assert home.state.store.get_draft(draft, "u_0123abcd") is not None
        for secret, user in ((SECRET, None), (SECRET, ""), (SECRET, "u_0123abcd; drop"), (SECRET, "../etc"),
                             ("t" * 40, "u_0123abcd"), (SECRET[:-1], "u_0123abcd")):
            assert post(c, secret, user).status_code == 401, (secret[:3], user)
        assert home.call(c, "start_draft")["status"] == "waiting_for_upload"            # invite tokens still work at home
    plain = Env(tmp_path / "plain")                                                     # no secret configured: no trust at all
    with client_of(plain) as c:
        assert post(c, SECRET, "u_0123abcd").status_code == 401
        assert post(c, "", "u_0123abcd").status_code == 401


def test_trusted_user_rules():
    assert front.trusted_user(SECRET, SECRET, " u_0123abcd ") == "u_0123abcd"
    assert front.trusted_user(SECRET, SECRET, "42580e76") == "42580e76"
    for secret, presented, user in ((None, SECRET, "u_0123abcd"), ("", "", "u_0123abcd"), (SECRET, None, "u_0123abcd"),
                                    (SECRET, SECRET, "U_0123ABCD"), (SECRET, SECRET, "u_0123abcde"), (SECRET, SECRET, None)):
        assert front.trusted_user(secret, presented, user) is None
