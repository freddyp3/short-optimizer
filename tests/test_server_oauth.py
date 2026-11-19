"""Google sign-in of the Claude MCP server: the OAuth flow a client goes through, signups, approval and blocking."""

import base64
import hashlib
import re
from urllib.parse import parse_qs, urlsplit

import pytest

from src.server import app as A
from src.server import oauth
from test_server_app import ACCEPT, BASE, Env, client_of, clips  # noqa: F401  (clips is a fixture)

REDIRECT = "https://claude.ai/api/mcp/auth_callback"
VERIFIER = "v" * 64
CHALLENGE = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b"=").decode()
PEOPLE = {"g-owner": oauth.Identity("sub-owner", "owner@example.com", "Owner"),
          "g-ann": oauth.Identity("sub-ann", "ann@example.com", "Ann <b>"),
          "g-bob": oauth.Identity("sub-bob", "bob@example.com", "Bob")}


class FakeGoogle:
    def url(self, state: str) -> str:
        return f"https://accounts.example/auth?state={state}"

    def identify(self, code: str) -> oauth.Identity:
        if code not in PEOPLE:
            raise oauth.IdentityError("no verified email")
        return PEOPLE[code]


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path, google=FakeGoogle(), auto_approve=("Owner@Example.com ",))


def register(c, name: str = "Claude", redirect: str = REDIRECT) -> str:
    r = c.post("/register", json={"redirect_uris": [redirect], "client_name": name, "token_endpoint_auth_method": "none",
                                  "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"]})
    assert r.status_code == 201, r.text
    return r.json()["client_id"]


def to_google(c, client_id: str, state: str = "st-1", redirect: str = REDIRECT) -> str:
    """/authorize; returns the state the server handed to Google."""
    r = c.get("/authorize", params={"response_type": "code", "client_id": client_id, "redirect_uri": redirect,
                                    "code_challenge": CHALLENGE, "code_challenge_method": "S256", "state": state},
              follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("https://accounts.example/auth?state="), r.text
    return r.headers["location"].split("state=")[1]


def back_from_google(c, state: str, code: str):
    return c.get(oauth.CALLBACK_PATH, params={"state": state, "code": code}, follow_redirects=False)


def ticket_of(page: str) -> str:
    return re.search(r'name="ticket" value="([^"]+)"', page).group(1)


def consent(c, ticket: str, choice: str = "allow"):
    return c.post(oauth.CONSENT_PATH, data={"ticket": ticket, "choice": choice}, follow_redirects=False)


def tokens_for(c, client_id: str, code: str, verifier: str = VERIFIER):
    return c.post("/token", data={"grant_type": "authorization_code", "code": code, "code_verifier": verifier,
                                  "client_id": client_id, "redirect_uri": REDIRECT})


def sign_in(c, google_code: str, client_id: str | None = None) -> dict:
    """The whole flow for an approved user; returns the token response."""
    client_id = client_id or register(c)
    page = back_from_google(c, to_google(c, client_id), google_code)
    assert page.status_code == 200 and "Allow" in page.text, page.text
    done = consent(c, ticket_of(page.text))
    q = parse_qs(urlsplit(done.headers["location"]).query)
    assert done.status_code == 302 and done.headers["location"].startswith(REDIRECT) and q["state"] == ["st-1"]
    r = tokens_for(c, client_id, q["code"][0])
    assert r.status_code == 200, r.text
    return {**r.json(), "client_id": client_id}


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", **ACCEPT}


def test_discovery_points_a_client_at_the_sign_in(env):
    with client_of(env) as c:
        r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=ACCEPT)
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == f'Bearer resource_metadata="{BASE}/.well-known/oauth-protected-resource/mcp"'
        resource = c.get("/.well-known/oauth-protected-resource/mcp").json()
        assert resource["resource"] == f"{BASE}/mcp" and resource["authorization_servers"] == [BASE]
        meta = c.get("/.well-known/oauth-authorization-server").json()
        assert meta["issuer"] == BASE and meta["authorization_endpoint"] == f"{BASE}/authorize"
        assert meta["token_endpoint"] == f"{BASE}/token" and meta["registration_endpoint"] == f"{BASE}/register"
        assert "S256" in meta["code_challenge_methods_supported"]


def test_without_google_the_server_offers_no_sign_in(tmp_path):
    env = Env(tmp_path)
    with client_of(env) as c:
        r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=ACCEPT)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        for path in ("/.well-known/oauth-authorization-server", "/.well-known/oauth-protected-resource/mcp", "/authorize",
                     oauth.CALLBACK_PATH):
            assert c.get(path).status_code == 404
        assert c.post("/register", json={}).status_code == 404


def test_an_auto_approved_user_signs_in_and_uses_the_tools_as_themselves(env):
    with client_of(env) as c:
        got = sign_in(c, "g-owner")
        assert got["token_type"] == "Bearer" and got["expires_in"] == oauth.ACCESS_TTL_S and got["refresh_token"]
        start = env.call(c, "start_draft", headers=bearer(got["access_token"]))
        assert start["status"] == "waiting_for_upload"
    (user,) = env.state.users.all()
    assert (user.email, user.name, user.status) == ("owner@example.com", "Owner", "approved") and user.id.startswith("u_")
    assert env.state.store.get_draft(start["draft_id"], user.id) is not None            # the draft belongs to the user
    assert env.state.users.get(user.id).last_seen_at == env.t
    assert got["access_token"].encode() not in env.db_bytes() and got["refresh_token"].encode() not in env.db_bytes()


def test_invite_tokens_keep_working_next_to_sign_in(env):
    with client_of(env) as c:
        assert env.call(c, "start_draft")["status"] == "waiting_for_upload"


def test_a_new_user_waits_for_approval_then_gets_in(env):
    with client_of(env) as c:
        client_id = register(c)
        page = back_from_google(c, to_google(c, client_id), "g-ann")
        assert page.status_code == 200 and "Waiting for approval" in page.text and "ann@example.com" in page.text
        assert "ticket" not in page.text and "<form" not in page.text
        (ann,) = env.state.users.all()
        assert (ann.status, ann.created_at, ann.approved_at) == ("pending", env.t, None)
        again = back_from_google(c, to_google(c, client_id), "g-ann")                   # signing in again changes nothing
        assert "Waiting for approval" in again.text and len(env.state.users.all()) == 1
        assert env.state.users.approve(ann.id) and not env.state.users.approve("u_missing")
        got = sign_in(c, "g-ann", client_id)
        assert env.call(c, "start_draft", headers=bearer(got["access_token"]))["status"] == "waiting_for_upload"


def test_the_consent_page_names_the_client_and_escapes_what_it_shows(env):
    with client_of(env) as c:
        env.state.users.approve(env.state.users.sign_in(PEOPLE["g-ann"])[0].id)
        client_id = register(c, name='Evil <script>alert(1)</script>')
        page = back_from_google(c, to_google(c, client_id), "g-ann")
    assert "<script" not in page.text and "Evil &lt;script&gt;" in page.text and "claude.ai" in page.text
    assert "form-action 'self' https://claude.ai" in page.headers["content-security-policy"]
    assert page.headers["cache-control"] == "no-store" and page.headers["x-frame-options"] == "DENY"


def test_states_tickets_and_codes_work_once(env):
    with client_of(env) as c:
        client_id = register(c)
        state = to_google(c, client_id)
        page = back_from_google(c, state, "g-owner")
        assert back_from_google(c, state, "g-owner").status_code == 400                 # the state is spent
        assert back_from_google(c, "made-up", "g-owner").status_code == 400
        ticket = ticket_of(page.text)
        assert back_from_google(c, "ticket:" + ticket, "g-owner").status_code == 400    # a ticket is not a state
        done = consent(c, ticket)
        assert consent(c, ticket).status_code == 400 and consent(c, "made-up").status_code == 400
        code = parse_qs(urlsplit(done.headers["location"]).query)["code"][0]
        assert tokens_for(c, client_id, code, verifier="w" * 64).status_code == 400     # wrong PKCE verifier
        other = register(c, name="Other")
        assert tokens_for(c, other, code).status_code == 400                            # another client's code
        assert tokens_for(c, client_id, code).status_code == 200
        assert tokens_for(c, client_id, code).status_code == 400                        # the code is spent


def test_cancel_and_a_refusal_at_google_send_the_client_an_error(env):
    with client_of(env) as c:
        client_id = register(c)
        page = back_from_google(c, to_google(c, client_id), "g-owner")
        r = consent(c, ticket_of(page.text), "deny")
        q = parse_qs(urlsplit(r.headers["location"]).query)
        assert r.status_code == 302 and q["error"] == ["access_denied"] and q["state"] == ["st-1"] and "code" not in q
        r = c.get(oauth.CALLBACK_PATH, params={"state": to_google(c, client_id), "error": "access_denied"}, follow_redirects=False)
        assert r.status_code == 302 and parse_qs(urlsplit(r.headers["location"]).query)["error"] == ["access_denied"]
        r = back_from_google(c, to_google(c, client_id), "unknown-google-code")
        assert r.status_code == 502 and A.MSG_GOOGLE_FAILED in r.text
    assert [u.email for u in env.state.users.all()] == ["owner@example.com"]


def test_refresh_rotates_the_tokens(env):
    with client_of(env) as c:
        got = sign_in(c, "g-owner")
        r = c.post("/token", data={"grant_type": "refresh_token", "refresh_token": got["refresh_token"],
                                   "client_id": got["client_id"]})
        assert r.status_code == 200, r.text
        new = r.json()
        assert new["access_token"] != got["access_token"] and new["refresh_token"] != got["refresh_token"]
        assert env.rpc(c, "tools/list", headers=bearer(got["access_token"])).status_code == 401
        assert env.rpc(c, "tools/list", headers=bearer(new["access_token"])).status_code == 200
        again = c.post("/token", data={"grant_type": "refresh_token", "refresh_token": got["refresh_token"],
                                       "client_id": got["client_id"]})
        assert again.status_code == 400


def test_an_access_token_expires(env):
    with client_of(env) as c:
        got = sign_in(c, "g-owner")
        env.t += oauth.ACCESS_TTL_S + 1
        assert env.rpc(c, "tools/list", headers=bearer(got["access_token"])).status_code == 401


def test_blocking_a_user_shuts_them_out_at_once(env):
    with client_of(env) as c:
        got = sign_in(c, "g-owner")
        (user,) = env.state.users.all()
        assert env.state.users.block(user.id)
        assert env.rpc(c, "tools/list", headers=bearer(got["access_token"])).status_code == 401
        r = c.post("/token", data={"grant_type": "refresh_token", "refresh_token": got["refresh_token"],
                                   "client_id": got["client_id"]})
        assert r.status_code == 400
        page = back_from_google(c, to_google(c, got["client_id"]), "g-owner")
        assert page.status_code == 403 and A.MSG_BLOCKED in page.text
        assert env.state.users.approve(user.id)                                         # approving lifts the block
        assert "Allow" in back_from_google(c, to_google(c, got["client_id"]), "g-owner").text


def test_users_do_not_see_each_others_drafts(env):
    with client_of(env) as c:
        for code in ("g-ann", "g-bob"):
            env.state.users.approve(env.state.users.sign_in(PEOPLE[code])[0].id)
        ann, bob = sign_in(c, "g-ann"), sign_in(c, "g-bob")
        draft = env.call(c, "start_draft", headers=bearer(ann["access_token"]))["draft_id"]
        assert env.call(c, "get_draft_status", {"draft_id": draft}, headers=bearer(ann["access_token"]))["status"] == "waiting_for_upload"
        assert env.call(c, "get_draft_status", {"draft_id": draft}, headers=bearer(bob["access_token"]))["status"] == "not_found"


def test_the_log_names_users_by_id_only(env, caplog):
    with caplog.at_level("INFO"), client_of(env) as c:
        got = sign_in(c, "g-owner")
        env.call(c, "start_draft", headers=bearer(got["access_token"]))
    text = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("so."))
    (user,) = env.state.users.all()
    assert f"token={user.id} route=GET /oauth/google/callback outcome=signup:approved" in text
    assert f"token={user.id} route=POST /oauth/consent outcome=allowed" in text
    assert f"token={user.id} route=POST /mcp:start_draft" in text and "route=POST /oauth " in text
    for secret in ("owner@example.com", "sub-owner", got["access_token"], got["refresh_token"]):
        assert secret not in text


def test_google_identity_needs_a_verified_email():
    class R:
        def __init__(self, body, status=200):
            self.body, self.status_code = body, status

        def json(self):
            return self.body

    def google(info, token_status=200):
        return oauth.Google("cid", "secret", f"{BASE}{oauth.CALLBACK_PATH}", post=lambda *a, **k: R({"access_token": "t"}, token_status),
                            get=lambda *a, **k: R(info))

    ok = google({"sub": "1", "email": " Ann@Example.com", "email_verified": True, "name": "Ann  A"}).identify("code")
    assert ok == oauth.Identity("1", "ann@example.com", "Ann A")
    for info in ({"sub": "1", "email": "a@b.c", "email_verified": False}, {"sub": "1", "email": "a@b.c"},
                 {"email": "a@b.c", "email_verified": True}, {"sub": "1", "email_verified": True}):
        with pytest.raises(oauth.IdentityError):
            google(info).identify("code")
    with pytest.raises(oauth.IdentityError):
        google({"sub": "1", "email": "a@b.c", "email_verified": True}, token_status=400).identify("code")
    url = google({}).url("the-state")
    q = parse_qs(urlsplit(url).query)
    assert url.startswith(oauth.GOOGLE_AUTH_URL) and q["state"] == ["the-state"] and q["client_id"] == ["cid"]
    assert q["redirect_uri"] == [f"{BASE}{oauth.CALLBACK_PATH}"] and "secret" not in url


# -------------------------------------------------------------------------------------------------- the owner's commands

def test_the_owner_can_list_approve_block_and_count(env, clips, capsys):
    from src.server import users as U

    db = str(env.cfg.DB_PATH)
    with client_of(env) as c:
        client_id = register(c)
        back_from_google(c, to_google(c, client_id), "g-ann")
        assert U.main(["--db", db, "list", "--pending"]) == 0
        out = capsys.readouterr().out
        assert "ann@example.com" in out and "pending" in out
        assert U.main(["--db", db, "approve", "ANN@example.com"]) == 0 and "approved" in capsys.readouterr().out
        assert U.main(["--db", db, "approve", "nobody@example.com"]) == 1
        got = sign_in(c, "g-ann", client_id)
        start = env.call(c, "start_draft", headers=bearer(got["access_token"]))
        assert env.upload(c, start["upload_url"], clips["good"]).status_code == 200
        env.idle()
        invited = env.call(c, "start_draft")
        assert env.upload(c, invited["upload_url"], clips["good"]).status_code == 200
        env.idle()
        capsys.readouterr()
        assert U.main(["--db", db, "list"]) == 0
        out = capsys.readouterr().out
        row = next(line for line in out.splitlines() if "ann@example.com" in line)
        assert "approved" in row and " 1 " in row and "0.02" in row and "tester" in out
        s = U.stats(db, now=env.t)
        assert (s["signups"], s["signups_last_7_days"], s["approved"], s["pending"], s["users_with_a_draft"]) == (1, 1, 1, 0, 1)
        assert (s["drafts"], s["active_last_7_days"], s["server_paid_usd"]) == (2, 1, pytest.approx(0.04))
        (ann,) = env.state.users.all()
        assert U.main(["--db", db, "block", ann.id]) == 0
        assert env.rpc(c, "tools/list", headers=bearer(got["access_token"])).status_code == 401
        assert U.stats(db, now=env.t)["blocked"] == 1
        assert U.main(["--db", db, "stats"]) == 0 and "signups" in capsys.readouterr().out
