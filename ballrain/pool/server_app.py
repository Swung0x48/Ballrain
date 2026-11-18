"""The gateway's HTTP application (docs/pool-prototype.md 2): routing, authentication, rate and size limits, the
forum login pages, hello / task / chunk / heartbeat / me / leaderboard, sha-addressed weights and assets.

PoolApp.handle(method, target, headers, body, client) -> Response is a pure function of its inputs and the store (tests
call it directly); serve() wraps it in a ThreadingHTTPServer bound to 127.0.0.1 (HTTPS comes from the tunnel later).
The HTTP layer checks Content-Length against PoolApp.body_limit before reading a body, refuses chunked bodies, sets
socket timeouts and logs neither client addresses nor query strings (an OAuth callback carries code and state).

Login: the controller's flow (POST login_start, browser to GET /pool/login?login=<id>, POST login_poll returns the
token once) or a browser-only flow (GET /pool/login -> link -> the callback page shows the token once). The plaintext
token exists only in memory until delivered; the store keeps its sha256. A new login gives the account a new token
(the old one stops working); a revoked account's login is refused with the stored reason.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import re
import secrets
import threading
import time
import traceback
import urllib.parse
from http.cookies import SimpleCookie
from pathlib import Path

from . import protocol as P
from .reasons import Reason, http_status, payload
from .server_config import PoolConfig, version_tuple
from .server_oauth import ForumClient, ForumError, eligibility, parse_user
from .server_store import ChunkRefused, Store, sha_hex
from .server_tasks import PolicySource
from .server_verify import select_chunk

TOKEN_PREFIX = "bp1_"
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{32,128}")
SESSION_RE = re.compile(r"[A-Za-z0-9_-]{16,64}")
SHA_FILE = re.compile(r"/pool/(weights|assets)/([0-9a-f]{64})\.(json|npz)")
COOKIE = "pool_oauth"
LOGIN_COOKIE = "pool_login"
USER_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I
MAX_TARGET = 2048
CHUNK_DEADLINE_S = 120  # a request (body read included) must finish within this (the HTTP layer's watchdog)
MAX_CONNECTIONS = 128
JSON_LIMIT = 16 * 1024
LOGIN_LIMIT = 4 * 1024

SECURITY_HEADERS = [("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer"),
                    ("X-Frame-Options", "DENY")]


class Response:
    """status, body, content type, headers. A sha-addressed file is sent from `file` in blocks by the HTTP layer
    (never held in memory for a slow reader); .body still reads it for callers of PoolApp.handle (tests)."""

    def __init__(self, status: int, body: bytes = b"", ctype: str = "application/json; charset=utf-8",
                 headers: list | None = None, file: Path | None = None):
        self.status, self._body, self.ctype, self.headers, self.file = status, body, ctype, list(headers or []), file

    @property
    def body(self) -> bytes:
        return self.file.read_bytes() if self.file is not None else self._body

    @property
    def length(self) -> int:
        return self.file.stat().st_size if self.file is not None else len(self._body)


def _json(kind: str, http: int = 200, headers=None, **fields) -> Response:
    return Response(http, P.encode_message(kind, **fields), headers=list(headers or []))


def _error(reason: dict, status: int | None = None) -> Response:
    st = status or http_status(reason["code"])
    h = [("Retry-After", str(reason["params"].get("retry_s", 1)))] if reason["code"] == Reason.RATE_LIMITED.value else []
    return _json("error", st, h, reason=reason)


# a refusal that cannot fail to encode (used when building the proper one raised)
_FALLBACK_ERROR = P.dumps({"protocol": P.PROTOCOL_VERSION, "type": "error",
                           "reason": payload(Reason.BAD_MESSAGE, detail="malformed request")})


def _safe_error(reason_fn, status: int | None = None) -> Response:
    try:
        return _error(reason_fn(), status)
    except Exception:  # noqa: BLE001  (a parameter the schema refuses: answer with the fixed message)
        return Response(status or 400, _FALLBACK_ERROR)


def _err(r: Reason, **params) -> P.ProtocolError:
    return P.ProtocolError(r, **params)


def new_user_code() -> str:
    """The login code the controller and the confirmation page both show: XXXX-XXXX (40 bits)."""
    c = "".join(secrets.choice(USER_CODE_ALPHABET) for _ in range(8))
    return f"{c[:4]}-{c[4:]}"


def _stored_reason(js: str | None, fallback: Reason) -> dict:
    try:
        d = json.loads(js) if js else None
        if isinstance(d, dict) and "code" in d:
            return payload(Reason(d["code"]), **(d.get("params") or {}))
    except (ValueError, KeyError):
        pass
    return payload(fallback, note="?")


# ============================================================================================== rate limits

class RateLimiter:
    """Token buckets keyed by (class, key); keys are hashed and kept in memory only (pruned)."""

    def __init__(self, rules: dict, clock=time.monotonic, max_keys: int = 100_000):
        self.rules = rules  # class -> (rate per second, burst)
        self.clock = clock
        self.max_keys = max_keys
        self.b: dict = {}
        self.lock = threading.Lock()

    def take(self, cls: str, key: str, cost: float = 1.0) -> float:
        """0 when allowed, else the seconds to wait."""
        rate, burst = self.rules[cls] if cls in self.rules else DEFAULT_RULES[cls]
        k = (cls, hashlib.blake2b(key.encode(), digest_size=12).digest())
        now = self.clock()
        with self.lock:
            if len(self.b) > self.max_keys:
                self.b = {kk: v for kk, v in self.b.items() if now - v[1] < 600}
            tokens, t = self.b.get(k, (burst, now))
            tokens = min(burst, tokens + (now - t) * rate)
            if tokens >= cost:
                self.b[k] = (tokens - cost, now)
                return 0.0
            self.b[k] = (tokens, now)
            return (cost - tokens) / rate


DEFAULT_RULES = {
    "login": (10 / 60, 20),  # login_start, the login / confirmation pages and the OAuth callback, per client address
    #                          (one login takes 4: start, confirmation page, continue, callback)
    "poll_addr": (5.0, 50),  # controller login polls, per client address (checked first: unknown ids are free buckets)
    "poll": (1.0, 10),  # controller login polls, per login_id (the controller polls every poll_s = 2 s)
    "public": (5.0, 60),  # weights / assets / leaderboard / pages, per address
    "badauth": (10 / 60, 20),  # failed token checks, per address
    "api": (10.0, 60),  # authenticated requests, per account
    "chunk": (10.0, 30),  # chunks, per account
}


# ============================================================================================== the app

class PoolApp:
    def __init__(self, cfg: PoolConfig, store: Store, policies: PolicySource, forum: ForumClient,
                 clock=time.time, rate_rules: dict | None = None, monotonic=time.monotonic, log=None):
        self.cfg, self.store, self.policies, self.forum = cfg, store, policies, forum
        self.clock = clock
        self.rl = RateLimiter(rate_rules or DEFAULT_RULES, clock=monotonic)
        self.log = log or (lambda msg: None)
        self._tokens: dict = {}  # login_id -> (token, user, expires): the plaintext token until it is delivered once
        self._tlock = threading.Lock()
        self._sem = threading.BoundedSemaphore(cfg.max_concurrent)
        self._chunk_lock = threading.Lock()
        self._board = (0.0, None)
        self._ref = {}  # engine kind -> reference game-file hashes
        self.chunks_dir = Path(cfg.pool_dir) / "chunks"

    def now(self) -> int:
        return int(self.clock())

    # ------------------------------------------------------------------------------------------ limits
    @staticmethod
    def body_limit(method: str, path: str) -> int:
        """The largest request body a route accepts (checked against Content-Length before reading)."""
        if method != "POST":
            return 0
        if path == "/pool/chunk":
            return P.MAX_CHUNK_BYTES
        if path == "/pool/login":
            return LOGIN_LIMIT
        if path in ("/pool/hello", "/pool/heartbeat", "/pool/me", "/pool/task"):  # POST /pool/task: task_end
            return JSON_LIMIT
        return 0

    def _limit(self, cls: str, key: str):
        w = self.rl.take(cls, key)
        if w > 0:
            raise _err(Reason.RATE_LIMITED, retry_s=max(1, int(w + 0.999)))

    # ------------------------------------------------------------------------------------------ dispatch
    def handle(self, method: str, target: str, headers: dict, body: bytes = b"", client: str = "") -> Response:
        headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        if self.cfg.trust_forwarded and headers.get("cf-connecting-ip"):
            client = headers["cf-connecting-ip"][:64]
        if not self._sem.acquire(blocking=False):
            return _error(payload(Reason.SERVER_BUSY), 503)
        try:
            if len(target) > MAX_TARGET:
                return _error(payload(Reason.BAD_MESSAGE, detail="request target too long"), 414)
            try:
                u = urllib.parse.urlsplit(target)
            except ValueError:
                return _error(payload(Reason.BAD_MESSAGE, detail="request target"), 400)
            path = u.path
            if len(body) > self.body_limit(method, path):
                r = (payload(Reason.CHUNK_TOO_LARGE, size=len(body), limit=P.MAX_CHUNK_BYTES) if path == "/pool/chunk"
                     else payload(Reason.BAD_MESSAGE, detail="request body too large"))
                return _error(r, 413)
            return self._route(method, path, u.query, headers, body, client)
        except P.ProtocolError as e:
            return _safe_error(e.payload)
        except Exception as e:  # noqa: BLE001
            self.log(f"internal error on {method} {target.split('?')[0][:80]}: {type(e).__name__}: {str(e)[:200]}\n"
                     f"{traceback.format_exc(limit=5)}")
            return _safe_error(lambda: payload(Reason.INTERNAL_ERROR), 500)
        finally:
            self._sem.release()

    AUTH_ROUTES = ("/pool/hello", "/pool/task", "/pool/chunk", "/pool/heartbeat", "/pool/me")

    def preauth(self, method: str, path: str, headers: dict, client: str = "") -> Response | None:
        """The HTTP layer's check before it reads a request body: an authenticated route without a valid, known token
        is refused here (badauth limit included), so an anonymous client cannot make the gateway buffer bodies.
        None = go on (handle() checks everything again)."""
        if path not in self.AUTH_ROUTES or self.body_limit(method, path) == 0:
            return None
        headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        if self.cfg.trust_forwarded and headers.get("cf-connecting-ip"):
            client = headers["cf-connecting-ip"][:64]
        try:
            self._auth(headers, client, allow_revoked=path in ("/pool/heartbeat", "/pool/me"))
        except P.ProtocolError as e:
            return _safe_error(e.payload)
        except Exception:  # noqa: BLE001
            return _safe_error(lambda: payload(Reason.INTERNAL_ERROR), 500)
        return None

    def _route(self, method, path, query, headers, body, client) -> Response:
        if path == "/pool/login":
            if method == "POST":
                return self.login_api(body, client)
            if method == "GET":
                self._limit("login", client)
                return self.login_page(urllib.parse.parse_qs(query, max_num_fields=4), headers)
        elif path == "/pool/oauth/callback" and method == "GET":
            self._limit("login", client)
            return self.callback(urllib.parse.parse_qs(query, max_num_fields=8), headers)
        elif path == "/pool/leaderboard" and method == "GET":
            self._limit("public", client)
            return self.leaderboard()
        elif method == "GET" and SHA_FILE.fullmatch(path):
            self._limit("public", client)
            return self.sha_file(path)
        elif path in ("/pool/hello", "/pool/task", "/pool/chunk", "/pool/heartbeat", "/pool/me"):
            want = {"/pool/hello": ("POST",), "/pool/task": ("GET", "POST"), "/pool/chunk": ("POST",),
                    "/pool/heartbeat": ("POST",), "/pool/me": ("GET", "POST")}[path]
            if method not in want:
                return _error(payload(Reason.BAD_MESSAGE, detail=f"{method} not allowed"), 405)
            tok, acct = self._auth(headers, client, allow_revoked=path in ("/pool/heartbeat", "/pool/me"))
            self._limit("chunk" if path == "/pool/chunk" else "api", f"acct:{acct['forum_id']}")
            if path == "/pool/hello":
                return self.hello(tok, acct, body)
            if path == "/pool/me":
                return self.me(acct, body if method == "POST" else None)
            sess = self._session(headers, tok, acct)
            if path == "/pool/task":
                return self.task(acct, sess) if method == "GET" else self.task_end(acct, body)
            if path == "/pool/chunk":
                return self.chunk(acct, sess, body)
            return self.heartbeat(acct, sess, body)
        return _error(payload(Reason.BAD_MESSAGE, detail="not found"), 404)

    # ------------------------------------------------------------------------------------------ auth
    def _auth(self, headers, client, allow_revoked=False):
        h = headers.get("authorization", "")
        if not h:
            raise _err(Reason.TOKEN_MISSING)
        scheme, _, tok = h.partition(" ")
        tok = tok.strip()
        if scheme.lower() != "bearer" or not TOKEN_RE.fullmatch(tok):
            self._limit("badauth", client)
            raise _err(Reason.TOKEN_INVALID)
        r = self.store.token(tok)
        if r is None or not r[0]["active"] or r[1] is None:
            self._limit("badauth", client)
            raise _err(Reason.TOKEN_INVALID)
        t, acct = r
        if acct["revoked"] and not allow_revoked:
            rs = _stored_reason(acct["revoke_reason"], Reason.TOKEN_REVOKED)
            raise P.ProtocolError(Reason(rs["code"]), **rs["params"])
        return t, acct

    def _session(self, headers, tok, acct):
        sid = headers.get("x-pool-session", "")
        if not SESSION_RE.fullmatch(sid):
            raise _err(Reason.SESSION_EXPIRED)
        s = self.store.session(sid)
        if (s is None or s["token_hash"] != tok["token_hash"] or s["forum_id"] != acct["forum_id"]
                or self.now() - s["last_seen"] > self.cfg.session_ttl_s):
            raise _err(Reason.SESSION_EXPIRED)
        return s

    @staticmethod
    def _user(acct) -> dict:
        return P.check(P.USER, {"forum_id": int(acct["forum_id"]), "username": acct["username"][:64] or "?",
                                "nickname": acct["nickname"][:64], "avatar_url": acct["avatar_url"]}, "user")

    # ------------------------------------------------------------------------------------------ login
    def login_api(self, body: bytes, client: str = "") -> Response:
        o = P.loads_strict(body, LOGIN_LIMIT)
        kind = o.get("type") if isinstance(o, dict) else None
        if kind not in ("login_start", "login_poll"):
            self._limit("login", client)
            raise _err(Reason.BAD_MESSAGE, detail="type must be login_start or login_poll")
        if kind == "login_poll":  # own buckets: a polling controller must not use up the browser's callback budget
            self._limit("poll_addr", client)
            lid = o.get("login_id")
            self._limit("poll", f"login:{lid if isinstance(lid, str) else '?'}"[:80])
        else:
            self._limit("login", client)
        msg = P.validate(kind, o)
        self.store.prune(max(self.cfg.login_ttl_s * 6, 3600))
        if kind == "login_start":
            self._check_version(msg["client_version"])
            lid, sec = secrets.token_urlsafe(24), secrets.token_urlsafe(32)
            code = new_user_code()
            self.store.add_login(lid, sec, code)
            return _json("login_started", login_id=lid, poll_secret=sec,
                         browser_url=f"{self.cfg.public_base_url}/pool/login?login={lid}", user_code=code,
                         expires_in_s=self.cfg.login_ttl_s, poll_s=2)
        row = self.store.login(msg["login_id"])
        if row is None or not hmac.compare_digest(row["poll_hash"], sha_hex(msg["poll_secret"])):
            return _json("login_status", status="expired")
        if self.now() - row["created"] > self.cfg.login_ttl_s and row["status"] == "pending":
            return _json("login_status", status="expired")
        if row["status"] == "pending":
            return _json("login_status", status="pending")
        if row["status"] == "refused":
            return _json("login_status", status="refused",
                         reason=_stored_reason(row["reason"], Reason.LOGIN_FAILED))
        with self._tlock:
            item = self._tokens.pop(msg["login_id"], None)
        if row["delivered"] or item is None or item[2] < self.now():
            return _json("login_status", status="expired")
        self.store.mark_delivered(msg["login_id"])
        return _json("login_status", status="ok", token=item[0], user=item[1])

    def _redirect_to_forum(self, login_id: str | None) -> Response:
        if not self.cfg.forum.client_id:
            return self._page("论坛登录暂不可用", "<p>网关还没有配置论坛应用。</p>", 503)
        state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        self.store.add_state(state, nonce, login_id)
        ck = (f"{COOKIE}={nonce}; Path=/pool/oauth/callback; Max-Age={self.cfg.login_ttl_s}; HttpOnly; SameSite=Lax"
              + ("; Secure" if self.cfg.secure_cookies else ""))
        return Response(302, b"", "text/plain; charset=utf-8",
                        [("Location", self.forum.authorize_url(state)), ("Set-Cookie", ck), ("Cache-Control", "no-store")])

    def login_page(self, q: dict, headers: dict | None = None) -> Response:
        if "login" in q:
            lid = q["login"][0]
            row = self.store.login(lid) if SESSION_RE.fullmatch(lid or "") else None
            if row is None or row["status"] != "pending" or self.now() - row["created"] > self.cfg.login_ttl_s:
                return self._reason_page(payload(Reason.OAUTH_STATE_MISMATCH), 400)
            confirm = (q.get("confirm") or [""])[0]
            if confirm and 16 <= len(confirm) <= 128 and self.store.take_login_confirm(
                    lid, confirm, self._cookie(headers or {}, LOGIN_COOKIE)):
                return self._redirect_to_forum(lid)
            return self._confirm_page(lid, row["user_code"] or "")
        if q.get("go") == ["1"]:
            return self._redirect_to_forum(None)
        return self._page("Ballrain 算力池：登录",
                          "<p>用 Ballance 社区论坛账号登录，领取算力池令牌。</p>"
                          "<p>领取条件：论坛注册满 {d} 天、邮箱已验证、至少 {r} 条回复。网关只保存论坛用户 ID、用户名和昵称，"
                          "不保存邮箱。</p><p><a href=\"/pool/login?go=1\">用论坛账号登录</a></p>"
                          .format(d=int(self.cfg.min_account_days), r=int(self.cfg.min_replies)))

    @staticmethod
    def _cookie(headers: dict, name: str) -> str | None:
        if not headers.get("cookie"):
            return None
        try:
            c = SimpleCookie()
            c.load(headers["cookie"][:4096])
            return c[name].value if name in c else None
        except Exception:  # noqa: BLE001  (a malformed cookie header is no cookie)
            return None

    def _confirm_page(self, lid: str, code: str) -> Response:
        """The controller's login link does not go straight to the forum: this page shows the login code the controller
        shows and continues only from this browser (a one-time link nonce plus a SameSite=Strict cookie, both bound to
        the login). Someone who sends a player their own login link gets this page, not the player's token."""
        cn, bc = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        self.store.set_login_confirm(lid, cn, bc)
        ck = (f"{LOGIN_COOKIE}={bc}; Path=/pool/login; Max-Age={self.cfg.login_ttl_s}; HttpOnly; SameSite=Strict"
              + ("; Secure" if self.cfg.secure_cookies else ""))
        href = html.escape(f"/pool/login?login={lid}&confirm={cn}")
        r = self._page("Ballrain 算力池：确认登录",
                       "<p>主控程序正在请求用你的论坛账号登录算力池。请核对主控程序上显示的登录代码：</p>"
                       f"<p style=\"font-size:2em;letter-spacing:.15em;font-family:monospace\">{html.escape(code)}</p>"
                       "<p><b>只有当你自己刚在主控程序里点了“登录”，并且上面的代码和主控程序上显示的完全一样时，才继续。</b>"
                       "如果这个链接是别人发给你的，请直接关闭此页：继续的话，对方会拿到用你的账号登录的令牌。</p>"
                       f"<p><a href=\"{href}\">代码一致，继续登录论坛</a></p>")
        r.headers.append(("Set-Cookie", ck))
        return r

    def callback(self, q: dict, headers: dict) -> Response:
        state = (q.get("state") or [""])[0]
        nonce = self._cookie(headers, COOKIE)
        lid = self.store.take_state(state, nonce, self.cfg.login_ttl_s) if 16 <= len(state) <= 128 else None
        if lid is None:
            return self._reason_page(payload(Reason.OAUTH_STATE_MISMATCH), 401)
        login_id = lid or None

        def refuse(reason: dict, status: int) -> Response:
            if login_id:
                self.store.finish_login(login_id, "refused", reason)
            return self._reason_page(reason, status)

        if "error" in q or not q.get("code"):
            return refuse(payload(Reason.LOGIN_FAILED, detail="授权被取消" if "error" in q else "缺少授权码"), 401)
        code = q["code"][0]
        if not 1 <= len(code) <= 1024:
            return refuse(payload(Reason.LOGIN_FAILED, detail="授权码无效"), 401)
        try:
            access = self.forum.exchange(code)
            rec = parse_user(self.forum.user(access))
            del access
        except ForumError as e:
            return refuse(payload(Reason.LOGIN_FAILED, detail=e.detail), 401)
        user = rec["user"]
        acct = self.store.account(user["forum_id"])
        if acct is not None and acct["revoked"]:
            return refuse(_stored_reason(acct["revoke_reason"], Reason.TOKEN_REVOKED), 403)
        bad = eligibility(rec, self.clock(), self.cfg.min_account_days, self.cfg.min_replies)
        if bad is not None:
            return refuse(bad, 403)
        self.store.upsert_account(user)
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        self.store.issue_token(user["forum_id"], token)
        nick = html.escape(user["nickname"])
        if login_id:
            with self._tlock:
                self._tokens = {k: v for k, v in self._tokens.items() if v[2] >= self.now()}
                self._tokens[login_id] = (token, self._user(self.store.account(user["forum_id"])),
                                          self.now() + self.cfg.login_ttl_s)
            self.store.finish_login(login_id, "ok", None, user["forum_id"])
            row = self.store.login(login_id)
            code = html.escape((row["user_code"] if row is not None else "") or "")
            return self._page("登录成功", f"<p>{nick}（论坛账号 {html.escape(user['username'])}），登录成功，"
                                       "可以回到主控程序了。</p>"
                                       f"<p>登录代码：<code>{code}</code>。主控程序会自动收到令牌；这个页面可以关闭。</p>"
                                       "<p>如果这不是你自己的主控程序发起的登录，请立即在主控程序里重新登录"
                                       "（旧令牌随即失效）。</p>")
        return self._page("登录成功", f"<p>{nick}，登录成功。下面是你的算力池令牌，<b>只显示这一次</b>：</p>"
                                   f"<p><code style=\"font-size:1.1em;user-select:all\">{html.escape(token)}</code></p>"
                                   "<p>把它粘贴到主控程序的“令牌”一栏。不要把令牌发给别人；再次登录会换一个新令牌，"
                                   "旧令牌随即失效。</p>")

    def _page(self, title: str, inner: str, status: int = 200) -> Response:
        doc = ("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
               "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
               f"<title>{html.escape(title)}</title><style>body{{font-family:sans-serif;max-width:40em;margin:2em auto;"
               "padding:0 16px;line-height:1.6}code{word-break:break-all}</style></head>"
               f"<body><h1>{html.escape(title)}</h1>{inner}</body></html>")
        return Response(status, doc.encode("utf-8"), "text/html; charset=utf-8",
                        [("Cache-Control", "no-store"),
                         ("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'none'")])

    def _reason_page(self, reason: dict, status: int) -> Response:
        return self._page("没有领到令牌", f"<p>{html.escape(reason['message'])}</p>"
                                       f"<p style=\"color:#666\">原因码：{html.escape(reason['code'])}</p>"
                                       "<p><a href=\"/pool/login\">回到登录页</a></p>", status)

    # ------------------------------------------------------------------------------------------ hello
    def _check_version(self, v: str):
        if version_tuple(v) < version_tuple(self.cfg.min_client_version):
            raise _err(Reason.CLIENT_TOO_OLD, got=v, min_version=self.cfg.min_client_version)

    def reference_files(self, kind: str) -> dict:
        if kind not in self._ref:
            e = self.cfg.engines[kind]
            if e.game_files:
                ref = dict(e.game_files)
            else:
                from .task import hash_game_files
                ref = hash_game_files(e.root, kind)
            self._ref[kind] = ref
        return self._ref[kind]

    def hello(self, tok, acct, body: bytes) -> Response:
        m = P.parse_message("hello", body)
        self._check_version(m["client_version"])
        e = m["engine"]
        ref = self.cfg.engines.get(e["kind"])
        if ref is None or e["sha256"] not in ref.sha256:
            raise _err(Reason.ENGINE_UNKNOWN, engine=e["kind"], sha=e["sha256"][:16])
        want = self.reference_files(e["kind"])
        names = P.GAME_FILE_LIST_2002 if e["kind"] == "2002" else P.GAME_FILE_LIST
        for name in names:
            got = m["game_files"].get(name)
            if got is None or got != want.get(name):
                raise _err(Reason.GAME_FILES_MISMATCH, file=name, got=(got or "missing")[:16])
        extra = set(m["game_files"]) - set(names)
        if extra:
            raise _err(Reason.GAME_FILES_MISMATCH, file=sorted(extra)[0], got="-")
        if m["instances"] > self.cfg.max_instances:
            raise _err(Reason.TOO_MANY_INSTANCES, max_instances=self.cfg.max_instances)
        sid = secrets.token_urlsafe(24)
        self.store.add_session(sid, acct["forum_id"], tok["token_hash"], m)
        return _json("hello_ok", session_id=sid, max_instances=self.cfg.max_instances,
                     heartbeat_s=max(5, self.cfg.heartbeat_s), task_poll_s=max(1, self.cfg.task_poll_s),
                     user=self._user(acct), leaderboard=bool(acct["leaderboard"]), server_time=self.now())

    # ------------------------------------------------------------------------------------------ task
    def task(self, acct, sess) -> Response:
        self.store.touch_session(sess["session_id"])
        live = max(3 * self.cfg.heartbeat_s, 90)  # a session without heartbeats for this long holds no instances
        if self.store.open_tasks(acct["forum_id"], live) >= self.cfg.max_instances:
            raise _err(Reason.TOO_MANY_INSTANCES, max_instances=self.cfg.max_instances)
        if (self.store.held_bytes(acct["forum_id"]) > self.cfg.max_pending_bytes_per_account
                or self.store.held_bytes() > self.cfg.max_total_bytes):
            return _json("no_task", reason=payload(Reason.SERVER_BUSY), retry_s=300)
        if self.policies.current is None:
            return _json("no_task", reason=payload(Reason.NO_TASK), retry_s=max(30, self.cfg.task_poll_s))
        t = self.policies.new_task(sess["engine_kind"], self.now(), self.cfg.max_steps, self.cfg.max_frames,
                                   self.cfg.deadline_s)
        d = t.to_dict()
        self.store.add_task(d, acct["forum_id"], sess["session_id"])
        return _json("task", task=d)

    def task_end(self, acct, body: bytes) -> Response:
        m = P.parse_message("task_end", body)
        tid = m["task_id"]
        st = self.store.close_task(tid, acct["forum_id"], m["chunks"], m["reason"])
        if st is None:
            row = self.store.task(tid)
            raise _err(Reason.TASK_UNKNOWN if row is None else Reason.TASK_NOT_YOURS, task_id=tid)
        if st == "order":
            row = self.store.task(tid)
            raise _err(Reason.CHUNK_OUT_OF_ORDER, task_id=tid, chunk=m["chunks"], expected=int(row["next_seq"]))
        return _json("task_end_ok", task_id=tid, state=st)

    # ------------------------------------------------------------------------------------------ chunk
    def chunk(self, acct, sess, body: bytes) -> Response:
        if self.store.held_bytes(acct["forum_id"]) + len(body) > self.cfg.max_pending_bytes_per_account:
            raise _err(Reason.SERVER_BUSY)  # this account's unverified data is at its cap until the verifier catches up
        c = P.decode_chunk(body)  # the expensive part, outside the lock
        with self._chunk_lock:  # checks, file and row of one chunk at a time (no duplicate-seq race)
            return self._accept_chunk(acct, sess, body, c)

    def _accept_chunk(self, acct, sess, body: bytes, c) -> Response:
        h = c.header
        tid, seq = h["task_id"], h["seq"]
        row = self.store.task(tid)
        if row is None:
            raise _err(Reason.TASK_UNKNOWN, task_id=tid)
        if row["forum_id"] != acct["forum_id"]:
            raise _err(Reason.TASK_NOT_YOURS, task_id=tid)
        task = json.loads(row["body"])
        if seq < row["next_seq"]:
            raise _err(Reason.CHUNK_DUPLICATE, task_id=tid, chunk=seq)
        if row["final"] or row["state"] != "open":
            if self.now() >= row["deadline"] and not row["final"]:
                raise _err(Reason.TASK_EXPIRED, task_id=tid, deadline_s=task["deadline_s"])
            raise _err(Reason.CHUNK_OUT_OF_ORDER, task_id=tid, chunk=seq, expected=row["next_seq"])
        if self.now() >= row["deadline"]:
            raise _err(Reason.TASK_EXPIRED, task_id=tid, deadline_s=task["deadline_s"])
        if seq > row["next_seq"]:
            raise _err(Reason.CHUNK_OUT_OF_ORDER, task_id=tid, chunk=seq, expected=row["next_seq"])
        if h["policy_sha"] != task["policy_sha"]:
            raise _err(Reason.POLICY_MISMATCH, got=h["policy_sha"][:16], want=task["policy_sha"][:16])
        pub = self.policies.history.get(task["policy_sha"])
        n_obs, n_act = (pub.n_obs, pub.n_act) if pub else self._manifest_dims(task["policy_sha"])
        if h["n_obs"] != n_obs or h["n_act"] != n_act:
            raise _err(Reason.BAD_CHUNK, detail=f"n_obs / n_act differ from the policy ({n_obs} / {n_act})")
        cap = P.max_chunk_bytes(int(task["chunk_steps"]), n_obs)
        if len(body) > cap:  # an honest chunk of this task's dimensions is never this big (padding)
            raise _err(Reason.CHUNK_TOO_LARGE, size=len(body), limit=cap)
        steps = h["steps"]
        if h["step0"] != row["steps"]:
            raise _err(Reason.CHAIN_BROKEN, task_id=tid, chunk=seq, step=h["step0"])
        if (not h["final"] and steps != task["chunk_steps"]) or steps > task["chunk_steps"]:
            raise _err(Reason.BAD_CHUNK, detail=f"a chunk has {task['chunk_steps']} steps (the final one at most)")
        if h["step0"] + steps > task["max_steps"]:
            raise _err(Reason.BAD_CHUNK, detail="more steps than the task's max_steps")
        if seq == 0:
            if h["chain_prev"] != P.chain_seed(tid).hex():
                raise _err(Reason.CHAIN_BROKEN, task_id=tid, chunk=seq, step=0)
        else:
            prev_row = self.store.chunks(tid)[-1]
            prev = P.decode_chunk((self.chunks_dir / prev_row["file"]).read_bytes())
            why = P.check_chain_link(prev, c)
            if why is not None:
                raise _err(Reason.CHAIN_BROKEN, task_id=tid, chunk=seq, step=h["step0"])
        if self.store.held_bytes() + len(body) > self.cfg.max_total_bytes:
            raise _err(Reason.SERVER_BUSY)
        rel = f"{tid}/{seq:06d}.brpc"
        dst = self.chunks_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():  # a file without a row (an earlier refused insert): the row decides, not the file
            dst.unlink()
        tmp = dst.with_name(f".{dst.name}.{secrets.token_hex(4)}.tmp")
        tmp.write_bytes(body)
        tmp.replace(dst)
        selected = select_chunk(self.store, acct["forum_id"], self.cfg.probation_chunks * int(task["chunk_steps"]),
                                self.cfg.sample_rate)
        try:
            self.store.add_chunk(tid, seq, acct["forum_id"], steps, len(body), rel, selected, h["final"], h["end"])
        except ChunkRefused as e:  # re-checked inside the transaction (the verifier may have acted meanwhile)
            dst.unlink(missing_ok=True)
            if e.why == "revoked":
                rs = _stored_reason(e.detail, Reason.TOKEN_REVOKED)
                raise P.ProtocolError(Reason(rs["code"]), **rs["params"]) from None
            if e.why == "expired":
                raise _err(Reason.TASK_EXPIRED, task_id=tid, deadline_s=task["deadline_s"]) from None
            if e.why == "order":
                raise _err(Reason.CHUNK_OUT_OF_ORDER, task_id=tid, chunk=seq, expected=e.detail) from None
            raise _err(Reason.CHUNK_DUPLICATE, task_id=tid, chunk=seq) from None
        except BaseException:
            dst.unlink(missing_ok=True)
            raise
        self.store.touch_session(sess["session_id"])
        return _json("chunk_result", task_id=tid, seq=seq, status="accepted", verify="pending", steps=steps)

    def _manifest_dims(self, sha: str):
        from .weights import read_manifest
        m = read_manifest(Path(self.cfg.pool_dir) / "weights" / f"{sha}.json")
        return int(m["arch"]["n_obs"]), int(m["arch"]["n_act"])

    # ------------------------------------------------------------------------------------------ heartbeat / me
    def heartbeat(self, acct, sess, body: bytes) -> Response:
        m = P.parse_message("heartbeat", body)
        if m["session_id"] != sess["session_id"]:
            raise _err(Reason.SESSION_EXPIRED)
        if acct["revoked"]:
            return _json("heartbeat_ok", server_time=self.now(), max_instances=0,
                         revoked=_stored_reason(acct["revoke_reason"], Reason.TOKEN_REVOKED))
        self.store.touch_session(sess["session_id"], m["running"], m["steps_per_s"])
        return _json("heartbeat_ok", server_time=self.now(), max_instances=self.cfg.max_instances)

    def me(self, acct, body: bytes | None) -> Response:
        if body is not None:
            m = P.parse_message("me_update", body)
            self.store.set_leaderboard(acct["forum_id"], m["leaderboard"])
            acct = self.store.account(acct["forum_id"])
        extra = {}
        if acct["revoked"]:
            extra["revoke_reason"] = _stored_reason(acct["revoke_reason"], Reason.TOKEN_REVOKED)
        return _json("me", user=self._user(acct), leaderboard=bool(acct["leaderboard"]),
                     token_state="revoked" if acct["revoked"] else "active", stats=self.store.stats(acct["forum_id"]),
                     **extra)

    def leaderboard(self) -> Response:
        t, cached = self._board
        if cached is None or time.monotonic() - t > 30:
            rows = self.store.leaderboard(200)
            entries = [{"rank": i + 1, "nickname": (r["nickname"] or r["username"])[:64], "steps_verified": int(r["steps"])}
                       for i, r in enumerate(rows)]
            cached = P.encode_message("leaderboard", updated=self.now(), entries=entries)
            self._board = (time.monotonic(), cached)
        return Response(200, cached, headers=[("Cache-Control", "public, max-age=30")])

    # ------------------------------------------------------------------------------------------ sha files
    def sha_file(self, path: str) -> Response:
        m = SHA_FILE.fullmatch(path)
        kind, sha, ext = m.groups()
        if kind == "assets" and ext != "npz":
            return _error(payload(Reason.BAD_MESSAGE, detail="not found"), 404)
        f = Path(self.cfg.pool_dir) / kind / f"{sha}.{ext}"
        if not f.is_file():
            return _error(payload(Reason.BAD_MESSAGE, detail="not found"), 404)
        ctype = "application/json; charset=utf-8" if ext == "json" else "application/octet-stream"
        return Response(200, b"", ctype, [("Cache-Control", "public, max-age=31536000, immutable")], file=f)


# ============================================================================================== HTTP server

def make_server(app: PoolApp, host: str = "127.0.0.1", port: int = 8099, max_connections: int = MAX_CONNECTIONS,
                deadline_s: float = CHUNK_DEADLINE_S):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        server_version = "ballrain-pool"
        sys_version = ""
        protocol_version = "HTTP/1.1"
        timeout = 30

        def log_message(self, fmt, *args):  # no client address, no query string (OAuth code / state)
            pass

        def log_request(self, code="-", size="-"):
            pass

        def handle_one_request(self):
            self.server.watch(self.connection)  # the whole request, body and response included, has a deadline
            try:
                super().handle_one_request()
            finally:
                self.server.unwatch(self.connection)

        def _send(self, r: Response, close: bool = False):
            n = r.length
            self.send_response(r.status)
            self.send_header("Content-Type", r.ctype)
            self.send_header("Content-Length", str(n))
            for k, v in SECURITY_HEADERS + list(r.headers):
                self.send_header(k, v)
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            if self.command == "HEAD":
                return
            if r.file is None:
                self.wfile.write(r.body)
                return
            with open(r.file, "rb") as f:  # in blocks: a slow reader never pins a whole file in memory
                for b in iter(lambda: f.read(64 * 1024), b""):
                    self.wfile.write(b)

        def _read_body(self, n: int) -> bytes | None:
            out = bytearray()
            while len(out) < n:
                b = self.rfile.read(min(64 * 1024, n - len(out)))
                if not b:
                    return None
                out += b
            return bytes(out)

        def _do(self):
            try:
                path = urllib.parse.urlsplit(self.path).path if len(self.path) <= MAX_TARGET else ""
            except ValueError:
                return self._send(_error(payload(Reason.BAD_MESSAGE, detail="request target"), 400), close=True)
            if self.headers.get("Transfer-Encoding"):
                return self._send(_error(payload(Reason.BAD_MESSAGE, detail="chunked bodies are not accepted"), 411),
                                  close=True)
            limit = app.body_limit(self.command, path)
            cl = self.headers.get("Content-Length")
            n = 0
            if cl is not None:
                if not cl.isdigit() or len(cl) > 12:
                    return self._send(_error(payload(Reason.BAD_MESSAGE, detail="Content-Length")), close=True)
                n = int(cl)
            elif self.command == "POST":
                return self._send(_error(payload(Reason.BAD_MESSAGE, detail="Content-Length required"), 411), close=True)
            if n > limit:
                r = (payload(Reason.CHUNK_TOO_LARGE, size=n, limit=P.MAX_CHUNK_BYTES) if path == "/pool/chunk"
                     else payload(Reason.BAD_MESSAGE, detail="request body too large"))
                return self._send(_error(r, 413), close=True)
            if n:  # no body is read for a request without a valid token (the connection is closed instead)
                refused = app.preauth(self.command, path, dict(self.headers.items()), self.client_address[0])
                if refused is not None:
                    return self._send(refused, close=True)
            body = self._read_body(n) if n else b""
            if body is None:
                return self._send(_error(payload(Reason.BAD_MESSAGE, detail="short body")), close=True)
            r = app.handle(self.command, self.path, dict(self.headers.items()), body, self.client_address[0])
            self._send(r)

        do_GET = do_POST = _do

        def do_PUT(self):
            self._send(_error(payload(Reason.BAD_MESSAGE, detail="method not allowed"), 405), close=True)

        do_DELETE = do_PATCH = do_PUT

    busy = (b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json; charset=utf-8\r\n"
            b"Connection: close\r\nContent-Length: " + str(len(_BUSY)).encode() + b"\r\n\r\n" + _BUSY)

    class Server(ThreadingHTTPServer):
        """At most max_connections connection threads (the rest get a fixed 503 and are closed), and a watchdog that
        shuts any connection whose current request runs longer than deadline_s (a slow drip of bytes, a reader that
        never reads)."""
        daemon_threads = True
        request_queue_size = 64

        def __init__(self, addr, handler, max_connections=MAX_CONNECTIONS, deadline_s=CHUNK_DEADLINE_S):
            super().__init__(addr, handler)
            self._conns = threading.BoundedSemaphore(max_connections)
            self.deadline_s = deadline_s
            self._watched: dict = {}
            self._wlock = threading.Lock()
            self._closed = threading.Event()
            threading.Thread(target=self._watchdog, daemon=True, name="pool-http-watchdog").start()

        def watch(self, sock):
            with self._wlock:
                self._watched[sock] = time.monotonic()

        def unwatch(self, sock):
            with self._wlock:
                self._watched.pop(sock, None)

        def _watchdog(self):
            import socket as _s
            while not self._closed.wait(1.0):
                now = time.monotonic()
                with self._wlock:
                    late = [s for s, t in self._watched.items() if now - t > self.deadline_s]
                for s in late:
                    try:
                        s.shutdown(_s.SHUT_RDWR)
                    except OSError:
                        pass

        def process_request(self, request, client_address):
            if not self._conns.acquire(blocking=False):
                try:
                    request.settimeout(2.0)
                    request.sendall(busy)
                except OSError:
                    pass
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self._conns.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self._conns.release()

        def handle_error(self, request, client_address):  # one line, no client address, no traceback
            import sys as _sys
            e = _sys.exc_info()[1]
            app.log(f"http: {type(e).__name__}: {str(e)[:120]}")

        def server_close(self):
            self._closed.set()
            super().server_close()

    return Server((host, port), Handler, max_connections, deadline_s)


_BUSY = P.dumps({"protocol": P.PROTOCOL_VERSION, "type": "error", "reason": payload(Reason.SERVER_BUSY)})
