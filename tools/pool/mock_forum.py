"""A mock of the forum's OAuth (flarum-oauth-center, authorization code flow) for gateway development and tests.

    python tools/pool/mock_forum.py --port 8098 --client-id dev --client-secret dev-secret \
        --redirect-uri http://127.0.0.1:8099/pool/oauth/callback [--users users.json]

Endpoints, enforced like the real forum application is configured (docs/volunteer-compute.md 10.2):
- GET  /oauth/authorize: response_type=code, the registered client_id, the EXACT redirect_uri, a state (required);
  shows a page with one button per mock user; ?mock_user=<key> answers directly (tests): 302 to the redirect_uri with
  code + the same state;
- POST /oauth/token: grant_type=authorization_code, the code (single use, 60 s), the same redirect_uri, client_id
  and client_secret (form fields or HTTP Basic) -> {"access_token", "token_type", "expires_in": 600};
- GET  /api/user with "Authorization: Bearer <access token>" -> the user record (id, username, nickname,
  avatar_url, email, is_email_confirmed, joined_at, last_seen_at, discussion_count, comment_count).

Default users (keys): eligible, too_new (joined 10 days ago), unverified (email not confirmed), few_replies (2
replies), revoked_later (eligible; tests revoke it). --users: a JSON object {key: {field: value}} merged over them;
"joined_days_ago" sets joined_at. Every email is a .invalid address. Nothing is logged.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import html
import json
import secrets
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_USERS = {
    "eligible": {"id": 1001, "username": "eligible_player", "nickname": "合格玩家", "joined_days_ago": 400,
                 "is_email_confirmed": True, "comment_count": 20},
    "too_new": {"id": 1002, "username": "new_player", "nickname": "新玩家", "joined_days_ago": 10,
                "is_email_confirmed": True, "comment_count": 20},
    "unverified": {"id": 1003, "username": "unverified_player", "nickname": "未验证", "joined_days_ago": 400,
                   "is_email_confirmed": False, "comment_count": 20},
    "few_replies": {"id": 1004, "username": "quiet_player", "nickname": "潜水", "joined_days_ago": 400,
                    "is_email_confirmed": True, "comment_count": 2},
    "revoked_later": {"id": 1005, "username": "cheater", "nickname": "作弊者", "joined_days_ago": 400,
                      "is_email_confirmed": True, "comment_count": 50},
}


def user_record(key: str, u: dict, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    joined = datetime.fromtimestamp(now, timezone.utc) - timedelta(days=float(u.get("joined_days_ago", 400)))
    rec = {"id": int(u["id"]), "username": u["username"], "nickname": u.get("nickname", ""),
           "avatar_url": u.get("avatar_url", f"https://forum.invalid/avatars/{int(u['id'])}.png"),
           "email": u.get("email", f"{key}@mock-forum.invalid"),
           "is_email_confirmed": bool(u.get("is_email_confirmed", True)),
           "joined_at": u.get("joined_at", joined.isoformat()),
           "last_seen_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
           "discussion_count": int(u.get("discussion_count", 1)), "comment_count": int(u.get("comment_count", 0))}
    return rec


class MockForum:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, users: dict | None = None,
                 clock=time.time):
        self.client_id, self.client_secret, self.redirect_uri = client_id, client_secret, redirect_uri
        self.users = {k: dict(v) for k, v in DEFAULT_USERS.items()}
        for k, v in (users or {}).items():
            self.users[k] = dict(self.users.get(k, {}), **v)
        self.clock = clock
        self.codes: dict = {}  # code -> (user key, redirect_uri, expires)
        self.tokens: dict = {}  # access token -> (user key, expires)
        self.lock = threading.Lock()
        self.token_requests = 0
        self.user_requests = 0
        self.httpd = None
        self.thread = None

    # ------------------------------------------------------------------------------------------ logic
    def authorize(self, q: dict) -> tuple:
        """(status, location or page, error)"""
        g = {k: v[0] for k, v in q.items() if v}
        if g.get("response_type") != "code":
            return 400, None, "unsupported_response_type"
        if g.get("client_id") != self.client_id:
            return 400, None, "invalid_client"
        if g.get("redirect_uri") != self.redirect_uri:
            return 400, None, "redirect_uri mismatch"
        if not g.get("state"):
            return 400, None, "state required"
        key = g.get("mock_user")
        if key is None:
            return 200, None, None
        if key == "deny":
            sep = "&" if "?" in self.redirect_uri else "?"
            return 302, f"{self.redirect_uri}{sep}" + urllib.parse.urlencode({"error": "access_denied",
                                                                            "state": g["state"]}), None
        if key not in self.users:
            return 400, None, "unknown mock user"
        code = secrets.token_urlsafe(24)
        with self.lock:
            self.codes[code] = (key, g["redirect_uri"], self.clock() + 60)
        sep = "&" if "?" in self.redirect_uri else "?"
        return 302, f"{self.redirect_uri}{sep}" + urllib.parse.urlencode({"code": code, "state": g["state"]}), None

    def token(self, form: dict, auth_header: str | None) -> tuple:
        self.token_requests += 1
        f = {k: v[0] for k, v in form.items() if v}
        cid, sec = f.get("client_id"), f.get("client_secret")
        if auth_header and auth_header.lower().startswith("basic "):
            try:
                cid, _, sec = base64.b64decode(auth_header[6:]).decode().partition(":")
            except ValueError:
                return 401, {"error": "invalid_client"}
        if cid != self.client_id or not hmac.compare_digest(str(sec or ""), self.client_secret):
            return 401, {"error": "invalid_client"}
        if f.get("grant_type") != "authorization_code":
            return 400, {"error": "unsupported_grant_type"}
        with self.lock:
            item = self.codes.get(f.get("code", ""))
            if item is None or item[2] < self.clock() or f.get("redirect_uri") != item[1]:
                return 400, {"error": "invalid_grant"}
            del self.codes[f["code"]]  # single use
        tok = secrets.token_urlsafe(32)
        with self.lock:
            self.tokens[tok] = (item[0], self.clock() + 600)
        return 200, {"access_token": tok, "token_type": "Bearer", "expires_in": 600}

    def api_user(self, auth_header: str | None) -> tuple:
        self.user_requests += 1
        if not auth_header or not auth_header.startswith("Bearer "):
            return 401, {"error": "unauthorized"}
        with self.lock:
            item = self.tokens.get(auth_header[7:].strip())
        if item is None or item[1] < self.clock():
            return 401, {"error": "invalid_token"}
        return 200, user_record(item[0], self.users[item[0]], self.clock())

    def emails(self) -> list:
        return [user_record(k, u)["email"] for k, u in self.users.items()]

    # ------------------------------------------------------------------------------------------ HTTP
    def serve(self, host: str = "127.0.0.1", port: int = 0):
        forum = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 15

            def log_message(self, *a):
                pass

            def _send(self, status, body: bytes, ctype="application/json", headers=()):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urllib.parse.urlsplit(self.path)
                if u.path == "/oauth/authorize":
                    q = urllib.parse.parse_qs(u.query)
                    st, loc, err = forum.authorize(q)
                    if st == 302:
                        return self._send(302, b"", headers=[("Location", loc)])
                    if err:
                        return self._send(st, json.dumps({"error": err}).encode())
                    base = {k: v[0] for k, v in q.items()}
                    links = "".join(
                        f'<li><a href="/oauth/authorize?{html.escape(urllib.parse.urlencode(dict(base, mock_user=k)))}">'
                        f"{html.escape(k)}（{html.escape(str(v.get('nickname', '')))}）</a></li>"
                        for k, v in forum.users.items())
                    page = (f"<!doctype html><meta charset=utf-8><title>模拟论坛授权</title><h1>模拟论坛：选择登录用户</h1>"
                            f"<ul>{links}<li><a href=\"/oauth/authorize?"
                            f"{html.escape(urllib.parse.urlencode(dict(base, mock_user='deny')))}\">拒绝授权</a></li></ul>")
                    return self._send(200, page.encode(), "text/html; charset=utf-8")
                if u.path == "/api/user":
                    st, o = forum.api_user(self.headers.get("Authorization"))
                    return self._send(st, json.dumps(o, ensure_ascii=False).encode())
                self._send(404, b'{"error":"not found"}')

            def do_POST(self):
                u = urllib.parse.urlsplit(self.path)
                n = int(self.headers.get("Content-Length") or 0)
                if n > 8192:
                    return self._send(413, b'{"error":"too large"}')
                body = self.rfile.read(n).decode("utf-8", "replace")
                if u.path == "/oauth/token":
                    st, o = forum.token(urllib.parse.parse_qs(body), self.headers.get("Authorization"))
                    return self._send(st, json.dumps(o).encode())
                self._send(404, b'{"error":"not found"}')

        class S(ThreadingHTTPServer):
            daemon_threads = True

        self.httpd = S((host, port), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="mock-forum", daemon=True)
        self.thread.start()
        return self

    @property
    def base_url(self) -> str:
        h, p = self.httpd.server_address[:2]
        return f"http://{h}:{p}"

    def shutdown(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--client-id", default="ballrain-pool-dev")
    ap.add_argument("--client-secret", default="dev-secret")
    ap.add_argument("--redirect-uri", default="http://127.0.0.1:8099/pool/oauth/callback")
    ap.add_argument("--users", help="JSON file {key: {field: value}} merged over the default users")
    a = ap.parse_args(argv)
    users = json.loads(open(a.users, encoding="utf-8").read()) if a.users else None
    f = MockForum(a.client_id, a.client_secret, a.redirect_uri, users).serve(a.host, a.port)
    print(f"mock forum on {f.base_url} (users: {', '.join(f.users)})", flush=True)
    try:
        f.thread.join()
    except KeyboardInterrupt:
        f.shutdown()


if __name__ == "__main__":
    main()
