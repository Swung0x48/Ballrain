"""Pool gateway (ballrain.pool.server_*, tools/pool/): forum OAuth against the mock forum (eligible / ineligible with
their reason codes, state / cookie / single-use rules), token issue / rotation / revocation, hello checks, size,
garbage and rate limits, privacy (no email, no IP, no token in storage or logs). The task / chunk / verifier path on
the fake game is in tests/test_pool_verifier.py."""
import json
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from ballrain.instances import REPO_ROOT
from ballrain.pool import protocol as P
from ballrain.pool.reasons import Reason
from ballrain.pool.server_app import PoolApp, RateLimiter, make_server
from ballrain.pool.server_config import EngineRef, PoolConfig, load_config, version_tuple
from ballrain.pool.server_oauth import ForumClient, ForumConfig, ForumError, eligibility, parse_user
from ballrain.pool.server_store import Store
from ballrain.pool.server_tasks import PolicySource

sys.path.insert(0, str(REPO_ROOT / "tools" / "pool"))
from mock_forum import MockForum  # noqa: E402

PUBLIC = "http://127.0.0.1:9"  # the app is called directly; only the forum runs over HTTP
ENGINE_SHA = "e" * 64
GAME = {n: f"{i + 1:064x}" for i, n in enumerate(P.GAME_FILE_LIST_2002)}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def browser_get(url: str) -> tuple:
    """(status, location) of a GET without following redirects (the player's browser at the forum)."""
    try:
        r = urllib.request.build_opener(NoRedirect).open(url, timeout=10)
        return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Location")


def make_cfg(tmp_path, forum_url, secret="dev-secret", **over) -> PoolConfig:
    cfg = PoolConfig(pool_dir=tmp_path / "pool", public_base_url=PUBLIC,
                     forum=ForumConfig(base_url=forum_url, client_id="cid", client_secret=secret,
                                       redirect_uri=f"{PUBLIC}/pool/oauth/callback"),
                     engines={"x87soft": EngineRef("x87soft", [ENGINE_SHA], game_files={n: GAME[n] for n in P.GAME_FILE_LIST}),
                              "2002": EngineRef("2002", [ENGINE_SHA], game_files=dict(GAME))})
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class Gw:
    """The app plus a mock forum, with the controller's and the browser's side of the login."""

    def __init__(self, tmp_path, users=None, secret="dev-secret", **cfg_over):
        self.forum = MockForum("cid", "dev-secret", f"{PUBLIC}/pool/oauth/callback", users).serve("127.0.0.1", 0)
        self.cfg = make_cfg(tmp_path, self.forum.base_url, secret, **cfg_over)
        self.store = Store(self.cfg.pool_dir / "state.sqlite3")
        self.logs = []
        self.policies = PolicySource(self.cfg.pool_dir, "test", tmp_path, [])
        self.app = PoolApp(self.cfg, self.store, self.policies, ForumClient(self.cfg.forum), log=self.logs.append,
                           rate_rules={k: (1000.0, 1000) for k in ("login", "poll_addr", "poll", "public", "badauth",
                                                                   "api", "chunk")})

    def call(self, method, target, body=b"", token=None, session=None, headers=None):
        h = dict(headers or {})
        if token:
            h["Authorization"] = f"Bearer {token}"
        if session:
            h["X-Pool-Session"] = session
        if isinstance(body, dict):
            body = P.dumps(body)
        r = self.app.handle(method, target, h, body, "127.0.0.1")
        return r

    def js(self, r):
        return json.loads(r.body)

    def start_login(self):
        r = self.call("POST", "/pool/login", P.encode_message("login_start", client_version="0.3.0"))
        assert r.status == 200
        return P.parse_message("login_started", r.body)

    def confirm(self, target: str):
        """A controller login link (/pool/login?login=...): the confirmation page, then its "continue" link with the
        page's cookie (the player's browser). Returns (page, continue target, cookie header value)."""
        page = self.call("GET", target)
        assert page.status == 200, page.body
        text = page.body.decode()
        href = text.split('<a href="')[1].split('"')[0].replace("&amp;", "&")
        ck = dict(page.headers)["Set-Cookie"]
        assert "HttpOnly" in ck and "SameSite=Strict" in ck and "Path=/pool/login" in ck
        return page, href, ck.split(";")[0]

    def browser(self, target: str, user: str, cookie=True):
        """/pool/login?... -> (the confirmation page of a controller login) -> forum (mock user) -> the callback page;
        returns the callback Response."""
        if "login=" in target:
            _, href, ck = self.confirm(target)
            r = self.call("GET", href, headers={"Cookie": ck})
        else:
            r = self.call("GET", target)
        assert r.status == 302, r.body
        h = dict(r.headers)
        loc = h["Location"]
        assert loc.startswith(self.forum.base_url + "/oauth/authorize?")
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(loc).query)
        assert q["redirect_uri"] == [f"{PUBLIC}/pool/oauth/callback"] and q["scope"] == ["user.read"] and q["state"][0]
        st, back = browser_get(loc + "&mock_user=" + user)
        assert st == 302 and back.startswith(f"{PUBLIC}/pool/oauth/callback?")
        nonce = h["Set-Cookie"].split(";")[0]
        assert "HttpOnly" in h["Set-Cookie"] and "SameSite=Lax" in h["Set-Cookie"]
        target = back[len(PUBLIC):]
        return self.call("GET", target, headers={"Cookie": nonce} if cookie else None), target, nonce

    def login(self, user: str):
        ls = self.start_login()
        page, _, _ = self.browser(urllib.parse.urlsplit(ls["browser_url"]).path + "?" +
                                  urllib.parse.urlsplit(ls["browser_url"]).query, user)
        poll = P.encode_message("login_poll", login_id=ls["login_id"], poll_secret=ls["poll_secret"])
        st = P.parse_message("login_status", self.call("POST", "/pool/login", poll).body)
        return st, page, ls

    def hello(self, token, **over):
        m = dict(client_version="0.3.0", platform="windows", engine={"kind": "x87soft", "sha256": ENGINE_SHA, "version": "1"},
                 game_files={n: GAME[n] for n in P.GAME_FILE_LIST}, instances=2)
        m.update(over)
        return self.call("POST", "/pool/hello", P.encode_message("hello", **m) if "raw" not in over else over["raw"],
                         token=token)

    def close(self):
        self.forum.shutdown()
        self.store.close()


@pytest.fixture
def gw(tmp_path):
    g = Gw(tmp_path)
    yield g
    g.close()


# ============================================================================================== OAuth

def test_controller_login_eligible_gets_a_token(gw):
    st, page, ls = gw.login("eligible")
    assert st["status"] == "ok" and st["token"].startswith("bp1_") and len(st["token"]) >= 40
    assert st["user"] == {"forum_id": 1001, "username": "eligible_player", "nickname": "合格玩家",
                          "avatar_url": "https://forum.invalid/avatars/1001.png"}
    assert page.status == 200 and "登录成功" in page.body.decode() and st["token"] not in page.body.decode()
    # delivered once: a second poll does not return the token again
    poll = P.encode_message("login_poll", login_id=ls["login_id"], poll_secret=ls["poll_secret"])
    again = P.parse_message("login_status", gw.call("POST", "/pool/login", poll).body)
    assert again["status"] == "expired" and "token" not in again
    # a wrong poll secret learns nothing
    bad = P.encode_message("login_poll", login_id=ls["login_id"], poll_secret="x" * 43)
    assert P.parse_message("login_status", gw.call("POST", "/pool/login", bad).body)["status"] == "expired"
    me = gw.call("GET", "/pool/me", token=st["token"])
    m = P.parse_message("me", me.body)
    assert m["token_state"] == "active" and m["leaderboard"] is True and m["stats"]["steps_verified"] == 0


@pytest.mark.parametrize("user,code,params", [
    ("too_new", Reason.ACCOUNT_TOO_NEW, {"days": 10, "min_days": 30}),
    ("unverified", Reason.EMAIL_NOT_VERIFIED, {}),
    ("few_replies", Reason.TOO_FEW_REPLIES, {"replies": 2, "min_replies": 5}),
])
def test_ineligible_accounts_are_refused_with_the_reason(gw, user, code, params):
    st, page, _ = gw.login(user)
    assert st["status"] == "refused" and "token" not in st
    assert st["reason"]["code"] == code.value and st["reason"]["params"] == params
    assert page.status == 403 and st["reason"]["message"] in page.body.decode()
    assert gw.store.q("SELECT COUNT(*) AS n FROM tokens")[0]["n"] == 0
    assert gw.store.account({"too_new": 1002, "unverified": 1003, "few_replies": 1004}[user]) is None


def test_eligibility_order_and_malformed_forum_records():
    now = 1_800_000_000
    rec = {"joined": now - 5 * 86400, "email_confirmed": False, "comments": 0}
    assert eligibility(rec, now)["code"] == "account_too_new"  # several fail: age first, then email, then replies
    assert eligibility(dict(rec, joined=now - 31 * 86400), now)["code"] == "email_not_verified"
    assert eligibility(dict(rec, joined=now - 30 * 86400, email_confirmed=True, comments=5), now) is None
    good = {"id": 7, "username": "u", "nickname": "n\x00ick" + "x" * 100, "avatar_url": "javascript:alert(1)",
            "is_email_confirmed": True, "joined_at": "2024-01-01T00:00:00Z", "comment_count": 9}
    p = parse_user(good)
    assert p["user"]["nickname"] == ("nick" + "x" * 100)[:64] and p["user"]["avatar_url"] is None
    assert parse_user(dict(good, nickname=None))["user"]["nickname"] == "u"
    for bad in (dict(good, id="x"), dict(good, id=0), dict(good, username=""), dict(good, joined_at="yesterday"),
                dict(good, is_email_confirmed="yes"), dict(good, comment_count=True), dict(good, comment_count=-1)):
        with pytest.raises(ForumError):
            parse_user(bad)


def test_state_cookie_and_single_use_rules(gw):
    ls = gw.start_login()
    tgt = "/pool/login?" + urllib.parse.urlsplit(ls["browser_url"]).query
    page, cb, nonce = gw.browser(tgt, "eligible", cookie=False)  # no cookie: not the browser that started it
    assert page.status == 401 and "oauth_state_mismatch" in page.body.decode()
    again = gw.call("GET", cb, headers={"Cookie": nonce})  # the state was used up by the first attempt
    assert again.status == 401 and "oauth_state_mismatch" in again.body.decode()
    r = gw.call("GET", "/pool/oauth/callback?code=abc&state=" + "s" * 43, headers={"Cookie": nonce})
    assert r.status == 401 and "oauth_state_mismatch" in r.body.decode()
    # the player denies the authorisation: login_failed, the controller's poll says refused
    ls = gw.start_login()
    _, href, lck = gw.confirm("/pool/login?" + urllib.parse.urlsplit(ls["browser_url"]).query)
    r = gw.call("GET", href, headers={"Cookie": lck})
    loc, ck = dict(r.headers)["Location"], dict(r.headers)["Set-Cookie"].split(";")[0]
    st, back = browser_get(loc + "&mock_user=deny")
    page = gw.call("GET", back[len(PUBLIC):], headers={"Cookie": ck})
    assert page.status == 401 and "login_failed" in page.body.decode()
    poll = P.encode_message("login_poll", login_id=ls["login_id"], poll_secret=ls["poll_secret"])
    assert P.parse_message("login_status", gw.call("POST", "/pool/login", poll).body)["reason"]["code"] == "login_failed"
    # an unknown / used login id
    assert gw.call("GET", "/pool/login?login=" + "z" * 32).status == 400


def test_mock_forum_enforces_the_application_settings(gw):
    f = gw.forum
    q = {"response_type": "code", "client_id": "cid", "redirect_uri": f"{PUBLIC}/pool/oauth/callback", "state": "s1"}
    url = f"{f.base_url}/oauth/authorize?"
    assert browser_get(url + urllib.parse.urlencode(dict(q, redirect_uri=f"{PUBLIC}/evil")))[0] == 400
    assert browser_get(url + urllib.parse.urlencode(dict(q, response_type="token")))[0] == 400  # no implicit grant
    assert browser_get(url + urllib.parse.urlencode({k: v for k, v in q.items() if k != "state"}))[0] == 400
    st, back = browser_get(url + urllib.parse.urlencode(dict(q, mock_user="eligible")))
    code = urllib.parse.parse_qs(urllib.parse.urlsplit(back).query)["code"][0]
    good = ForumClient(ForumConfig(f.base_url, "cid", "dev-secret", q["redirect_uri"]))
    with pytest.raises(ForumError):  # wrong secret
        ForumClient(ForumConfig(f.base_url, "cid", "wrong", q["redirect_uri"])).exchange(code)
    with pytest.raises(ForumError):  # another redirect_uri than the authorisation's
        ForumClient(ForumConfig(f.base_url, "cid", "dev-secret", f"{PUBLIC}/x")).exchange(code)
    access = good.exchange(code)
    with pytest.raises(ForumError):  # single use
        good.exchange(code)
    u = good.user(access)
    assert "email" not in u and u["id"] == 1001  # the client drops email while parsing


def test_wrong_forum_secret_is_login_failed(tmp_path):
    g = Gw(tmp_path, secret="not-the-secret")
    try:
        st, page, _ = g.login("eligible")
        assert st["status"] == "refused" and st["reason"]["code"] == "login_failed" and page.status == 401
    finally:
        g.close()


def test_browser_only_login_shows_the_token_once(gw):
    r = gw.call("GET", "/pool/login")
    assert r.status == 200 and b'href="/pool/login?go=1"' in r.body
    assert "default-src 'none'" in dict(r.headers)["Content-Security-Policy"]
    page, _, _ = gw.browser("/pool/login?go=1", "eligible")
    text = page.body.decode()
    assert page.status == 200 and "只显示这一次" in text and dict(page.headers)["Cache-Control"] == "no-store"
    tok = text.split("user-select:all\">")[1].split("<")[0]
    assert P.parse_message("me", gw.call("GET", "/pool/me", token=tok).body)["user"]["forum_id"] == 1001


# ============================================================================================== tokens

def test_token_rotation_and_revocation_with_reason(gw):
    t1 = gw.login("eligible")[0]["token"]
    t2 = gw.login("eligible")[0]["token"]
    assert t1 != t2
    r = gw.hello(t1)
    assert r.status == 401 and gw.js(r)["reason"]["code"] == "token_invalid"
    assert gw.hello(t2).status == 200
    sid = P.parse_message("hello_ok", gw.hello(t2).body)["session_id"]
    # an admin revocation, shown verbatim
    from ballrain.pool.reasons import payload
    gw.store.revoke(1001, payload(Reason.TOKEN_REVOKED, note="测试吊销"))
    r = gw.hello(t2)
    assert r.status == 403 and gw.js(r)["reason"]["code"] == "token_revoked" and "测试吊销" in gw.js(r)["reason"]["message"]
    assert gw.call("GET", "/pool/task", token=t2, session=sid).status == 403
    me = P.parse_message("me", gw.call("GET", "/pool/me", token=t2).body)
    assert me["token_state"] == "revoked" and me["revoke_reason"]["code"] == "token_revoked"
    hb = P.encode_message("heartbeat", session_id=sid, running=True, instances=[], steps_per_s=0.0)
    h = P.parse_message("heartbeat_ok", gw.call("POST", "/pool/heartbeat", hb, token=t2, session=sid).body)
    assert h["max_instances"] == 0 and h["revoked"]["code"] == "token_revoked"
    # a new login does not get around it
    st, page, _ = gw.login("eligible")
    assert st["status"] == "refused" and st["reason"]["code"] == "token_revoked" and page.status == 403
    # missing / malformed tokens
    assert gw.js(gw.call("GET", "/pool/me"))["reason"]["code"] == "token_missing"
    assert gw.js(gw.call("GET", "/pool/me", token="short"))["reason"]["code"] == "token_invalid"
    assert gw.js(gw.call("GET", "/pool/me", headers={"Authorization": "Basic abc"}))["reason"]["code"] == "token_invalid"


def test_leaderboard_flag_defaults_on_and_can_be_turned_off(gw):
    tok = gw.login("eligible")[0]["token"]
    m = P.parse_message("me", gw.call("POST", "/pool/me", P.encode_message("me_update", leaderboard=False), token=tok).body)
    assert m["leaderboard"] is False
    assert P.parse_message("leaderboard", gw.call("GET", "/pool/leaderboard").body)["entries"] == []


# ============================================================================================== hello / limits

def test_hello_checks(gw):
    tok = gw.login("eligible")[0]["token"]
    ok = P.parse_message("hello_ok", gw.hello(tok).body)
    assert ok["max_instances"] == 8 and ok["leaderboard"] is True and len(ok["session_id"]) >= 16

    def code(r):
        return r.status, gw.js(r)["reason"]["code"], gw.js(r)["reason"]["params"]
    assert code(gw.hello(tok, engine={"kind": "x87soft", "sha256": "f" * 64, "version": "1"}))[:2] == (400, "engine_unknown")
    gf = dict({n: GAME[n] for n in P.GAME_FILE_LIST}, **{"Text/": "0" * 64})
    st, c, p = code(gw.hello(tok, game_files=gf))
    assert (st, c, p["file"]) == (400, "game_files_mismatch", "Text/")
    gf = {n: GAME[n] for n in P.GAME_FILE_LIST if n != "base.cmo"}
    assert code(gw.hello(tok, game_files=gf))[2]["file"] == "base.cmo"
    # tasks are only issued for the engine kinds the verifier replays with ([verify] engines, default x87soft)
    st, c, p = code(gw.hello(tok, engine={"kind": "2002", "sha256": ENGINE_SHA, "version": "1"}))
    assert (st, c, p) == (400, "engine_mismatch", {"got": "2002", "want": "x87soft"})
    gw.cfg.verify_engines = ("2002", "x87soft")
    # the 2002 engine also reads the retail Bin/VxMath.dll and BuildingBlocks/physics_RT.dll (not in x87soft's list)
    st, c, p = code(gw.hello(tok, engine={"kind": "2002", "sha256": ENGINE_SHA, "version": "1"}))
    assert (c, p["file"]) == ("game_files_mismatch", "Bin/VxMath.dll")
    gw.cfg.verify_engines = ("x87soft",)
    gf = dict({n: GAME[n] for n in P.GAME_FILE_LIST}, **{"Bin/VxMath.dll": GAME["Bin/VxMath.dll"]})
    assert code(gw.hello(tok, game_files=gf))[1:] == ("game_files_mismatch", {"file": "Bin/VxMath.dll", "got": "-"})
    assert code(gw.hello(tok, instances=9))[1:] == ("too_many_instances", {"max_instances": 8})
    assert code(gw.hello(tok, client_version="0.0.9"))[1] == "client_too_old"
    assert code(gw.hello(tok, client_version="0.1.0"))[1] == "client_too_old"  # chunk format 1 (the float32 chain)
    assert code(gw.hello(tok, client_version="0.2.0"))[1] == "client_too_old"  # chunk format 2 (chain v2)
    raw = json.loads(P.encode_message("hello", client_version="0.3.0", platform="linux",
                                      engine={"kind": "x87soft", "sha256": ENGINE_SHA, "version": "1"},
                                      game_files={n: GAME[n] for n in P.GAME_FILE_LIST}, instances=1))
    assert code(gw.hello(tok, raw=P.dumps(dict(raw, extra=1))))[1] == "bad_message"
    assert code(gw.hello(tok, raw=P.dumps(dict(raw, protocol=2))))[1] == "protocol_unsupported"
    assert code(gw.hello(tok, raw=b"\xff\xfe garbage"))[1] == "bad_message"
    assert code(gw.hello(tok, raw=b'{"a":1,"a":2}'))[1] == "bad_message"
    assert code(gw.hello(tok, raw=b"[" * 100 + b"]" * 100))[1] == "bad_message"
    # task without / with a stale session
    assert code(gw.call("GET", "/pool/task", token=tok))[1] == "session_expired"
    assert code(gw.call("GET", "/pool/task", token=tok, session="a" * 32))[1] == "session_expired"
    # no policy published: no_task with a retry hint
    nt = P.parse_message("no_task", gw.call("GET", "/pool/task", token=tok, session=ok["session_id"]).body)
    assert nt["reason"]["code"] == "no_task" and nt["retry_s"] >= 1


def test_garbage_and_wrong_routes(gw):
    tok = gw.login("eligible")[0]["token"]
    sid = P.parse_message("hello_ok", gw.hello(tok).body)["session_id"]
    r = gw.call("POST", "/pool/chunk", b"\x00" * 1000, token=tok, session=sid)
    assert r.status == 400 and gw.js(r)["reason"]["code"] == "bad_chunk"
    r = gw.call("POST", "/pool/chunk", b"x" * (P.MAX_CHUNK_BYTES + 1), token=tok, session=sid)
    assert r.status == 413 and gw.js(r)["reason"]["code"] == "chunk_too_large"
    assert gw.call("POST", "/pool/hello", b"{" * 20000, token=tok).status == 413
    assert gw.call("GET", "/pool/nothing").status == 404
    assert gw.call("PUT", "/pool/hello", token=tok).status == 405
    assert gw.call("GET", "/pool/hello", token=tok).status == 405
    assert gw.call("GET", "/pool/weights/" + "a" * 64 + ".npz").status == 404
    assert gw.call("GET", "/pool/weights/../state.sqlite3").status == 404
    assert gw.call("GET", "/pool/assets/" + "a" * 64 + ".json").status == 404
    assert gw.call("GET", "/pool/" + "a" * 3000).status == 414
    lb = gw.call("POST", "/pool/login", b'{"type":"login_poll","protocol":1}')
    assert lb.status == 400 and gw.js(lb)["reason"]["code"] == "bad_message"


def test_rate_limits(tmp_path):
    t = [0.0]
    g = Gw(tmp_path)
    try:
        g.app.rl = RateLimiter({"login": (1.0, 3), "public": (1.0, 2), "badauth": (1.0, 2), "api": (1.0, 2),
                                "chunk": (1.0, 2)}, clock=lambda: t[0])
        body = P.encode_message("login_start", client_version="0.3.0")
        assert [g.call("POST", "/pool/login", body).status for _ in range(4)] == [200, 200, 200, 429]
        r = g.call("POST", "/pool/login", body)
        assert g.js(r)["reason"]["code"] == "rate_limited" and dict(r.headers)["Retry-After"] == "1"
        t[0] += 2.0
        assert g.call("POST", "/pool/login", body).status == 200
        st = [g.call("GET", "/pool/me", token="bp1_" + "x" * 40).status for _ in range(4)]  # token guessing
        assert st[:2] == [401, 401] and st[-1] == 429
    finally:
        g.close()


def test_http_server_size_limits_and_logging(gw):
    srv = make_server(gw.app, "127.0.0.1", 0)
    import threading
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    port = srv.server_address[1]

    def raw(req: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            s.sendall(req)
            out = b""
            while True:
                b = s.recv(65536)
                if not b:
                    break
                out += b
                if b"\r\n\r\n" in out and len(out) > out.index(b"\r\n\r\n") + 4 + 10:
                    break
        return out

    try:
        # a 5 MiB chunk is refused from its Content-Length alone (nothing is read)
        r = raw(b"POST /pool/chunk HTTP/1.1\r\nHost: x\r\nContent-Length: 5242880\r\n\r\n")
        assert r.startswith(b"HTTP/1.1 413") and b"chunk_too_large" in r
        r = raw(b"POST /pool/hello HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
        assert r.startswith(b"HTTP/1.1 411")
        r = raw(b"POST /pool/login HTTP/1.1\r\nHost: x\r\n\r\n")
        assert r.startswith(b"HTTP/1.1 411")
        r = raw(b"POST /pool/login HTTP/1.1\r\nHost: x\r\nContent-Length: -5\r\n\r\n")
        assert r.startswith(b"HTTP/1.1 400")
        r = raw(b"GET /pool/leaderboard HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        assert r.startswith(b"HTTP/1.1 200") and b"X-Content-Type-Options: nosniff" in r
        # the OAuth callback's query (code, state) and the client address are never logged
        raw(b"GET /pool/oauth/callback?code=SECRETCODE&state=SECRETSTATE HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    finally:
        srv.shutdown()
        srv.server_close()
    assert not any("SECRETCODE" in m or "127.0.0.1" in m for m in gw.logs)


def test_privacy_nothing_personal_is_stored(gw, capsys):
    tokens = [gw.login(u)[0].get("token") for u in ("eligible", "too_new", "revoked_later")]
    gw.store.db.execute("PRAGMA wal_checkpoint(FULL)")
    blob = b"".join(p.read_bytes() for p in gw.cfg.pool_dir.glob("state.sqlite3*"))
    for e in gw.forum.emails():
        assert e.encode() not in blob
    for t in tokens:
        if t:
            assert t.encode() not in blob  # only the hash
    out = capsys.readouterr()
    assert not any(e in out.out + out.err for e in gw.forum.emails())
    assert not any(e in m for e in gw.forum.emails() for m in gw.logs)
    cols = {r["name"] for r in gw.store.q("PRAGMA table_info(accounts)")}
    assert not {"email", "ip"} & cols


def test_config_from_toml_and_env(tmp_path):
    text = ('[forum]\nclient_id = "from-file"\nclient_secret = "file-secret"\n'
            '[pool]\npublic_base_url = "https://example.invalid/pool"\nmax_instances = 99\n'
            '[engines.x87soft]\nbinary = "runtime/hl-win/bin-x86soft/BallanceMMOSimTool.exe"\n')
    (tmp_path / "config.toml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="trust_forwarded"):  # https = behind the tunnel: the forwarded address is needed
        load_config(tmp_path / "config.toml", tmp_path, environ={})
    (tmp_path / "config.toml").write_text(text.replace("max_instances = 99\n", "max_instances = 99\ntrust_forwarded = true\n"),
                                          encoding="utf-8")
    cfg = load_config(tmp_path / "config.toml", tmp_path, environ={"BALLRAIN_POOL_FORUM_CLIENT_SECRET": "env-secret"})
    assert cfg.trust_forwarded is True
    assert cfg.forum.client_id == "from-file" and cfg.forum.client_secret == "env-secret"
    assert cfg.forum.redirect_uri == "https://example.invalid/pool/oauth/callback" and cfg.secure_cookies
    assert cfg.max_instances == 64 and cfg.forum.base_url == "https://forum.ballance.top"
    assert Path(cfg.engines["x87soft"].binary).is_absolute() and set(cfg.engines) == {"x87soft"}
    assert cfg.ckpt_dir == REPO_ROOT / "runtime" / "m2-train-mac" / "r6"
    assert version_tuple("0.10.1") > version_tuple("0.9.9") and version_tuple("1.0") == version_tuple("1.0.0")


def test_pool_runtime_config_is_not_tracked():
    import subprocess
    out = subprocess.run(["git", "ls-files", "runtime"], capture_output=True, text=True, cwd=REPO_ROOT, timeout=60)
    assert "runtime/pool" not in out.stdout
    ign = subprocess.run(["git", "check-ignore", "runtime/pool/config.toml"], capture_output=True, text=True,
                         cwd=REPO_ROOT, timeout=60)
    assert ign.returncode == 0


R6 = REPO_ROOT / "runtime" / "m2-train-mac" / "r6"
DATASETS = REPO_ROOT / "runtime" / "datasets" / "tas-v1"


@pytest.mark.skipif(not (R6.is_dir() and any(R6.glob("ckpt_*.pt")) and DATASETS.is_dir()),
                    reason="the r6 checkpoints / the tas-v1 dataset are not on this machine")
def test_policy_source_publishes_r6_and_draws_curriculum_starts(tmp_path):
    pytest.importorskip("torch")
    ps = PolicySource(tmp_path / "pool", "r6", DATASETS, [REPO_ROOT / "benchmarks" / "refpaths",
                                                          REPO_ROOT / "runtime" / "refpaths-alt"], ckpt_dir=R6)
    pub = ps.refresh()
    if pub is None:
        pytest.skip(f"no readable r6 checkpoint: {ps.last_error}")
    assert (tmp_path / "pool" / "weights" / f"{pub.policy_sha}.npz").is_file()
    assert (tmp_path / "pool" / "assets" / f"{pub.starts_sha}.npz").is_file()
    assert (tmp_path / "pool" / "assets" / f"{pub.path_sha}.npz").is_file()
    assert pub.record == pub.manifest["env"]["start_record"] and pub.curriculum is not None
    assert ps.refresh() is pub  # the same checkpoint is not exported again
    cu = pub.curriculum
    draws, pub.draws = pub.draws, None  # the curriculum draw alone (the rehearsal / at-rest draws: below)
    ks = [k for k, rest in (ps.pick_start(pub) for _ in range(400))]
    pub.draws = draws
    excl = pub.manifest["env"]["exclude_starts"] or []
    valid = set(pub.valid.tolist())
    assert all(k in valid for k in ks)
    assert not any(lo <= k < hi for k in ks for lo, hi in excl)
    assert all(k == 0 or k >= cu.frontier for k in ks)  # the trainer's draw: [frontier, end] (k = 0 at the start)
    near = sum(cu.frontier <= k <= cu.frontier + cu.cfg.band for k in ks)
    assert near > 0
    t = ps.new_task("2002", 1000, 8192, 60000, 3600)
    d = P.check(P.TASK, t.to_dict())
    assert d["engine"] == "2002" and d["chunk_steps"] == pub.manifest["rollout"]["chunk"]
    assert d["assets"] == {"starts": pub.starts_sha, "path": pub.path_sha}
    for _ in range(200):  # with the run's own rehearsal / at-rest draws: every task passes the schema
        P.check(P.TASK, ps.new_task("2002", 1000, 8192, 60000, 3600).to_dict())


# ============================================================================================== review fixes

def test_a_controller_login_link_sent_to_someone_else_does_not_hand_over_their_token(gw):
    """The attacker starts a controller login and sends the victim browser_url. The link no longer goes straight to the
    forum: it shows the login code and continues only from the browser that rendered the page (one-time link nonce +
    SameSite=Strict cookie). The attacker's own confirm link, opened by the victim, shows the page again."""
    ls = gw.start_login()
    assert P.USER_CODE_RE.fullmatch(ls["user_code"])
    tgt = "/pool/login?" + urllib.parse.urlsplit(ls["browser_url"]).query
    victim_page = gw.call("GET", tgt)
    assert victim_page.status == 200 and ls["user_code"] in victim_page.body.decode()  # no redirect
    assert "Location" not in dict(victim_page.headers)
    _, att_href, att_ck = gw.confirm(tgt)  # the attacker renders the page himself and sends its link
    for hdr in (None, {"Cookie": dict(victim_page.headers)["Set-Cookie"].split(";")[0]}):
        r = gw.call("GET", att_href, headers=hdr)  # the victim's browser: no cookie, or the cookie of another render
        assert r.status == 200 and "Location" not in dict(r.headers), r.status
    poll = P.encode_message("login_poll", login_id=ls["login_id"], poll_secret=ls["poll_secret"])
    assert P.parse_message("login_status", gw.call("POST", "/pool/login", poll).body)["status"] == "pending"
    # the right browser, its own fresh page: continues once (the link nonce is single use)
    _, href, ck = gw.confirm(tgt)
    assert gw.call("GET", href, headers={"Cookie": ck}).status == 302
    again = gw.call("GET", href, headers={"Cookie": ck})
    assert again.status == 200 and "Location" not in dict(again.headers)
    # the success page names the forum account and the login code
    st, page, ls2 = gw.login("eligible")
    text = page.body.decode()
    assert st["status"] == "ok" and "eligible_player" in text and ls2["user_code"] in text


def test_login_polls_do_not_use_up_the_callback_budget(tmp_path):
    """DEFAULT_RULES with a fake clock: the controller polls every 2 s (and on rate_limited waits retry_s); the
    player's browser on the same address must still get its confirmation page and callback 30-100 s later."""
    from ballrain.pool.server_app import DEFAULT_RULES
    t = [0.0]
    g = Gw(tmp_path)
    try:
        g.app.rl = RateLimiter(DEFAULT_RULES, clock=lambda: t[0])
        ls = g.start_login()
        poll = P.encode_message("login_poll", login_id=ls["login_id"], poll_secret=ls["poll_secret"])
        while t[0] < 100:
            r = g.call("POST", "/pool/login", poll)
            assert r.status == 200, r.body  # 0.5 polls per second fit the per-login bucket
            t[0] += 2.0
        tgt = "/pool/login?" + urllib.parse.urlsplit(ls["browser_url"]).query
        page, _, _ = g.browser(tgt, "eligible")
        assert page.status == 200 and "登录成功" in page.body.decode()
        # many unknown login ids from one address hit the per-address cap, not fresh per-id buckets
        codes = [g.call("POST", "/pool/login", P.encode_message("login_poll", login_id=f"{i:016d}x",
                                                                poll_secret="s" * 32)).status for i in range(80)]
        assert 429 in codes
    finally:
        g.close()


def test_unencodable_refusal_parameters_still_get_an_answer(gw):
    body = b'{"protocol":100000000000000000000,"type":"login_start","client_version":"0.3.0"}'
    r = gw.call("POST", "/pool/login", body)
    assert r.status == 400 and gw.js(r)["reason"]["code"] == "protocol_unsupported"
    assert gw.js(r)["reason"]["params"]["got"] == "100000000000000000000"[:20]
    assert not gw.logs  # not an internal error


def _honest_like_chunk(steps=4, n_obs=3, codec_obs=None):
    import numpy as np
    arrays = {"actions": np.zeros(steps, np.uint8), "rewards": np.zeros(steps, np.float32),
              "logp": np.full(steps, -0.5, np.float32), "values": np.zeros(steps + 1, np.float32),
              "flags": np.zeros(steps, np.uint8), "obs": np.zeros((steps + 1, n_obs), np.float32)}
    arrays["obs_hash"] = P.chain_from_digests([bytes(32)] * len(arrays["obs"]), P.chain_seed("t1"))
    h = {"task_id": "t1", "seq": 0, "step0": 0, "steps": steps, "n_obs": n_obs, "n_act": 9, "policy_sha": "a" * 64,
         "chain_prev": P.chain_seed("t1").hex(), "final": False, "end": None}
    return P.encode_chunk(h, arrays)


def _resection(data: bytes, name: str, stored: bytes, codec: str):
    """Replace one section's stored bytes (and codec) and re-seal the chunk (header nbytes, trailing digest)."""
    import hashlib
    import struct
    magic, fmt, flags, hlen = struct.unpack_from("<4sHHI", data, 0)
    h = json.loads(data[12:12 + hlen])
    body = data[12 + hlen:-32]
    parts, off = {}, 0
    for s in h["arrays"]:
        parts[s["name"]] = body[off:off + s["nbytes"]]
        off += s["nbytes"]
    parts[name] = stored
    for s in h["arrays"]:
        if s["name"] == name:
            s["nbytes"], s["codec"] = len(stored), codec
    hj = P.dumps(h)
    out = struct.pack("<4sHHI", magic, fmt, flags, len(hj)) + hj + b"".join(parts[s["name"]] for s in h["arrays"])
    return out + hashlib.sha256(out).digest()


def test_padded_and_oversized_sections_are_refused():
    import numpy as np
    from compression import zstd
    ok = _honest_like_chunk()
    P.decode_chunk(ok)
    raw = np.zeros((5, 3), "<f2").tobytes()
    # a frame padded with empty raw blocks before the data decodes to the same bytes
    padded = _pad_frame(raw, 300)
    assert zstd.decompress(padded) == raw and len(padded) > P.zstd_bound(len(raw))
    with pytest.raises(P.ProtocolError, match="padded"):
        P.decode_chunk(_resection(ok, "obs", padded, "zstd"))
    # a raw section must be exactly its decoded size
    with pytest.raises(P.ProtocolError):
        P.decode_chunk(_resection(ok, "rewards", np.zeros(4, "<f4").tobytes() + b"\0" * 16, "raw"))
    # a frame asking for a 128 MiB window is refused (no 128 MiB commit per decode)
    big_window = _frame_with_window(raw, 27)
    assert zstd.decompress(big_window) == raw
    with pytest.raises(P.ProtocolError, match="zstd"):
        P.zstd_decompress_exact(big_window, len(raw))


def _pad_frame(raw: bytes, n_empty: int) -> bytes:
    """A zstd frame (no content size, single segment off) with n_empty empty raw blocks before one raw block."""
    import struct
    magic = struct.pack("<I", 0xFD2FB528)
    fhd = 0x00  # no checksum, no dict, single-segment 0, FCS flag 0
    wd = (17 - 10) << 3  # window 128 KiB
    blocks = b""
    for _ in range(n_empty):
        blocks += (0 << 1 | 0).to_bytes(3, "little")  # last=0, raw, size 0
    hdr = (len(raw) << 3) | (0 << 1) | 1  # last=1, raw
    blocks += hdr.to_bytes(3, "little") + raw
    return magic + bytes([fhd, wd]) + blocks


def _frame_with_window(raw: bytes, window_log: int) -> bytes:
    import struct
    magic = struct.pack("<I", 0xFD2FB528)
    wd = (window_log - 10) << 3
    hdr = (len(raw) << 3) | 1
    return magic + bytes([0x00, wd]) + hdr.to_bytes(3, "little") + raw


def _recv_all(s) -> bytes:
    out = b""
    while True:
        b = s.recv(65536)
        if not b:
            return out
        out += b


def test_http_layer_refuses_before_reading_and_bounds_connections(gw):
    import threading
    import time as _t
    srv = make_server(gw.app, "127.0.0.1", 0, max_connections=2, deadline_s=2)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        # no token: refused from the headers alone (the 4 MiB body is never sent)
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            s.sendall(b"POST /pool/chunk HTTP/1.1\r\nHost: x\r\nContent-Length: 4194304\r\n\r\n")
            out = _recv_all(s)
        assert out.startswith(b"HTTP/1.1 401") and b"token_missing" in out and b"Connection: close" in out
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            s.sendall(b"POST /pool/chunk HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer bp1_" + b"x" * 40 +
                      b"\r\nContent-Length: 4194304\r\n\r\n")
            out = _recv_all(s)
        assert out.startswith(b"HTTP/1.1 401") and b"token_invalid" in out
        # a malformed request target: an answer, one log line without the address, no traceback
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            s.sendall(b"GET http://[::1/x HTTP/1.1\r\nHost: x\r\n\r\n")
            out = _recv_all(s)
        assert out.startswith(b"HTTP/1.1 400")
        # two idle connections hold both slots: the third gets the fixed 503; the watchdog closes the idle ones
        a = socket.create_connection(("127.0.0.1", port), timeout=10)
        b = socket.create_connection(("127.0.0.1", port), timeout=10)
        a.sendall(b"GET /pool/leaderboard HTTP/1.1\r\n")  # a request that never finishes
        b.sendall(b"GET /pool/leaderboard HTTP/1.1\r\n")
        _t.sleep(0.3)
        with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
            c.sendall(b"GET /pool/leaderboard HTTP/1.1\r\nHost: x\r\n\r\n")
            assert c.recv(65536).startswith(b"HTTP/1.1 503")
        t0 = _t.monotonic()
        for s in (a, b):
            s.settimeout(10)
            while s.recv(65536):
                pass
            s.close()
        assert _t.monotonic() - t0 < 6  # shut by the 2 s deadline, not the 30 s socket timeout
        _t.sleep(0.3)
        with socket.create_connection(("127.0.0.1", port), timeout=10) as c:  # the slots are free again
            c.sendall(b"GET /pool/leaderboard HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            assert c.recv(65536).startswith(b"HTTP/1.1 200")
    finally:
        srv.shutdown()
        srv.server_close()
    assert not any("127.0.0.1" in m for m in gw.logs)


def test_probation_counts_steps_not_chunks(tmp_path):
    from ballrain.pool.server_verify import select_task
    st = Store(tmp_path / "s.sqlite3")
    try:
        st.upsert_account({"forum_id": 5, "username": "u", "nickname": "n"})
        for i in range(300):  # 300 one-step "stopped" tasks
            st.q("INSERT INTO chunks(task_id, seq, forum_id, steps, nbytes, file, received, selected, status, verify) "
                 "VALUES(?,0,5,1,100,'f',0,1,'held','pending')", (f"t{i}",))
        assert select_task(st, 5, 200 * 256, 0.0)  # still in probation: 300 steps of 51200
        st.q("UPDATE chunks SET steps=256, verify='passed', status='released'")
        assert not select_task(st, 5, 200 * 256, 0.0)
        assert select_task(st, 5, 200 * 256, 1.0)
        # an escalation: the next n tasks are replayed in full, then the draw again
        st.escalate(5, 2)
        assert [select_task(st, 5, 200 * 256, 0.0) for _ in range(3)] == [True, True, False]
        n = sum(select_task(st, 5, 200 * 256, 0.15) for _ in range(4000))
        assert 0.12 < n / 4000 < 0.18
    finally:
        st.close()


def test_store_refuses_chunks_of_closed_tasks_and_revoked_accounts(tmp_path):
    from ballrain.pool.reasons import payload
    from ballrain.pool.server_store import ChunkRefused
    st = Store(tmp_path / "s.sqlite3")
    try:
        st.upsert_account({"forum_id": 5, "username": "u", "nickname": "n"})
        task = {"task_id": "tA", "policy_sha": "a" * 64, "issued_at": st.now(), "deadline_s": 3600}
        st.add_task(task, 5, "sess")
        st.add_chunk("tA", 0, 5, 64, 10, "tA/0", True, False, None)
        with pytest.raises(ChunkRefused) as e:
            st.add_chunk("tA", 0, 5, 64, 10, "tA/0", True, False, None)
        assert e.value.why == "duplicate"
        with pytest.raises(ChunkRefused) as e:
            st.add_chunk("tA", 2, 5, 64, 10, "tA/2", True, False, None)
        assert e.value.why == "order"
        st.q("UPDATE tasks SET deadline=0")
        st.close_expired()  # the verifier closed it between the gateway's checks and the insert
        with pytest.raises(ChunkRefused) as e:
            st.add_chunk("tA", 1, 5, 64, 10, "tA/1", True, False, None)
        assert e.value.why == "expired" and st.task("tA")["state"] == "closed"
        st.add_task(dict(task, task_id="tB"), 5, "sess")
        st.revoke(5, payload(Reason.TOKEN_REVOKED, note="x"))
        with pytest.raises(ChunkRefused) as e:
            st.add_chunk("tB", 0, 5, 64, 10, "tB/0", True, False, None)
        assert e.value.why == "revoked" and not st.chunks("tB")
    finally:
        st.close()


def test_gateway_unlinks_the_file_of_a_chunk_refused_in_the_transaction(tmp_path, monkeypatch):
    """The account is revoked (by the verifier) while a chunk is being accepted: the chunk is refused with the stored
    reason, no row, no file. Uses the verifier fixture world for a real task."""
    pytest.importorskip("ballrain.fakegame")
    if sys.platform != "win32":
        pytest.skip("ballrain.fakegame's shared memory needs Windows")
    from ballrain.pool import dev, server_app
    from ballrain.pool.reasons import payload
    world = dev.fake_world(tmp_path / "w")
    g = Gw(tmp_path / "g")
    try:
        g.policies.datasets_dir = world["datasets"]
        g.policies.path_dirs = world["path_dirs"]
        g.policies.publish_state(world["state"])
        tok = g.login("eligible")[0]["token"]
        sid = P.parse_message("hello_ok", g.hello(tok).body)["session_id"]
        t = P.parse_message("task", g.call("GET", "/pool/task", token=tok, session=sid).body)["task"]
        import numpy as np
        n_obs = g.policies.current.n_obs
        steps = t["chunk_steps"]
        arrays = {"actions": np.zeros(steps, np.uint8), "rewards": np.zeros(steps, np.float32),
                  "logp": np.full(steps, -0.5, np.float32), "values": np.zeros(steps + 1, np.float32),
                  "flags": np.zeros(steps, np.uint8), "obs": np.zeros((steps + 1, n_obs), np.float32)}
        arrays["obs_hash"] = P.chain_from_digests([bytes(32)] * len(arrays["obs"]), P.chain_seed(t["task_id"]))
        h = {"task_id": t["task_id"], "seq": 0, "step0": 0, "steps": steps, "n_obs": n_obs, "n_act": 9,
             "policy_sha": t["policy_sha"], "chain_prev": P.chain_seed(t["task_id"]).hex(), "final": False, "end": None}
        data = P.encode_chunk(h, arrays)
        real = g.app._consistency

        def revoke_meanwhile(c, task):  # the verifier revokes the account while this chunk is being accepted
            g.store.revoke(1001, payload(Reason.TOKEN_REVOKED_REPLAY, task_id="x", chunk=0, step=1))
            return None
        monkeypatch.setattr(g.app, "_consistency", revoke_meanwhile)
        r = g.call("POST", "/pool/chunk", data, token=tok, session=sid)
        assert r.status == 403 and g.js(r)["reason"]["code"] == "token_revoked_replay"
        assert not g.store.chunks(t["task_id"])
        assert not list((g.cfg.pool_dir / "chunks").rglob("*.brpc"))
        # per-account cap on held bytes, checked when a chunk arrives
        g.store.unrevoke(1001)
        monkeypatch.setattr(g.app, "_consistency", real)
        g.cfg.max_pending_bytes_per_account = len(data) - 1
        r = g.call("POST", "/pool/chunk", data, token=tok, session=sid)
        assert r.status == 503 and g.js(r)["reason"]["code"] == "server_busy"
        g.cfg.max_pending_bytes_per_account = 1 << 30
        assert g.call("POST", "/pool/chunk", data, token=tok, session=sid).status == 200
        # the global cap counts only held chunks: released ones do not block new work
        g.cfg.max_total_bytes = len(data) + 10
        g.store.q("UPDATE chunks SET status='released', verify='passed'")
        t2 = P.parse_message("task", g.call("GET", "/pool/task", token=tok, session=sid).body)
        assert t2["type"] == "task"
    finally:
        g.close()


def test_forum_requests_name_the_gateway_agent():
    """The forum's Cloudflare bans urllib's default Python-urllib agent (error 1010, HTTP 403): both server-side forum
    calls send USER_AGENT."""
    import io
    from ballrain.pool import server_oauth as SO
    seen = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            seen.append((req.full_url.rsplit("/", 2)[-2:], req.get_header("User-agent")))
            return _Resp(b'{"access_token": "t", "id": 1}')
    fc = ForumClient(ForumConfig("https://forum.example", "cid", "sec", "https://x/pool/oauth/callback"), opener=_Opener())
    assert fc.exchange("code") == "t"
    fc.user("t")
    assert [ua for _, ua in seen] == [SO.USER_AGENT, SO.USER_AGENT] and "python" not in SO.USER_AGENT.lower()


def test_forum_user_fields_in_other_serialisations():
    """joined_at as ISO with fraction / without zone / a space, or Unix seconds / ms; 0 / 1 flags; digit-string counts.
    A failure names the type, never the value."""
    good = {"id": 7, "username": "u", "nickname": "n", "avatar_url": None, "is_email_confirmed": True,
            "joined_at": "2024-01-01T00:00:00Z", "comment_count": 9}
    want = parse_user(good)["joined"]
    for j in ("2024-01-01T00:00:00.000000Z", "2024-01-01T00:00:00+00:00", "2024-01-01 00:00:00", 1704067200,
              1704067200000, "1704067200"):
        assert parse_user(dict(good, joined_at=j))["joined"] == want
    assert parse_user(dict(good, is_email_confirmed=1))["email_confirmed"] is True
    assert parse_user(dict(good, is_email_confirmed="0"))["email_confirmed"] is False
    assert parse_user(dict(good, comment_count="12"))["comments"] == 12
    with pytest.raises(ForumError, match=r"joined_at \(NoneType\)"):
        parse_user(dict(good, joined_at=None))
    with pytest.raises(ForumError, match=r"joined_at \(str, length 9\)"):
        parse_user(dict(good, joined_at="yesterday"))
