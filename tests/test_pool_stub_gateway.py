"""A stub pool gateway for the controller tests (stdlib http.server on 127.0.0.1, speaking ballrain.pool.protocol).
Not the real gateway: no OAuth, one user, everything in memory; chunks are decoded and chain-checked like the real one
(docs/pool-prototype.md 3.3) and kept for the test to inspect."""
from __future__ import annotations

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ballrain.pool import protocol as P
from ballrain.pool import reasons as R

TOKEN = "stub-token-" + "x" * 32


class StubGateway:
    def __init__(self, user=None, leaderboard: bool = True, max_instances: int = 8, heartbeat_s: int = 5):
        self.lock = threading.Lock()
        self.user = user or {"forum_id": 42, "username": "tester", "nickname": "测试玩家"}
        self.leaderboard = leaderboard
        self.max_instances = max_instances
        self.heartbeat_s = heartbeat_s
        self.token = TOKEN
        self.revoked = None  # reason payload once revoked
        self.hello_error = None  # reason payload to refuse hello with
        self.task_queue: list = []  # task dicts handed out once each
        self.files: dict = {}  # url path -> bytes (weights, assets)
        self.chunks: dict = {}  # task_id -> [Chunk]
        self.raw_chunks: dict = {}
        self.reject_chunks = None  # reason payload: reject every chunk
        self.revoke_on_chunk = None  # reason payload: chunk_result.token_revoked
        self.heartbeats: list = []
        self.me_updates: list = []
        self.hellos: list = []
        self.logins: dict = {}
        self.sessions: set = set()
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, status: int, body: bytes, ctype="application/json; charset=utf-8"):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _msg(self, kind: str, http: int = 200, **fields):
                self._send(http, P.encode_message(kind, **fields))

            def _err(self, code, **params):
                pl = R.payload(code, **params)
                self._send(R.http_status(code), P.encode_message("error", reason=pl))

            def _auth(self, session=False) -> bool:
                h = self.headers.get("Authorization", "")
                if h != f"Bearer {stub.token}":
                    self._err(R.Reason.TOKEN_INVALID)
                    return False
                if stub.revoked is not None:
                    self._send(403, P.encode_message("error", reason=stub.revoked))
                    return False
                if session and self.headers.get("X-Pool-Session") not in stub.sessions:
                    self._err(R.Reason.SESSION_EXPIRED)  # as the gateway
                    return False
                return True

            def _body(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n)

            def do_GET(self):  # noqa: N802
                p = self.path
                if p in stub.files:
                    return self._send(200, stub.files[p], "application/octet-stream")
                if p.startswith(("/pool/weights/", "/pool/assets/")):
                    return self._send(404, b"{}")
                if p == "/pool/task":
                    if not self._auth(True):
                        return
                    with stub.lock:
                        t = stub.task_queue.pop(0) if stub.task_queue else None
                    if t is None:
                        return self._msg("no_task", reason=R.payload(R.Reason.NO_TASK), retry_s=1)
                    return self._msg("task", task=t)
                if p == "/pool/me":
                    if not self._auth():
                        return
                    return self._me()
                if p == "/pool/leaderboard":
                    return self._msg("leaderboard", updated=int(time.time()), entries=[])
                self._send(404, b"{}")

            def _me(self):
                steps = sum(c.steps for cs in stub.chunks.values() for c in cs)
                self._msg("me", user=stub.user, leaderboard=stub.leaderboard, token_state="active",
                          stats={"steps_accepted": steps, "steps_pending": steps, "steps_verified": 0,
                                 "chunks_accepted": sum(len(v) for v in stub.chunks.values()), "chunks_rejected": 0})

            def do_POST(self):  # noqa: N802
                p = self.path
                body = self._body()
                try:
                    if p == "/pool/login":
                        m = P.loads_strict(body)
                        if m.get("type") == "login_start":
                            P.validate("login_start", m)
                            lid, sec = secrets.token_urlsafe(16), secrets.token_urlsafe(32)
                            stub.logins[lid] = sec
                            base = f"http://127.0.0.1:{stub.port}"
                            return self._msg("login_started", login_id=lid, poll_secret=sec,
                                             browser_url=f"{base}/pool/login?login={lid}", user_code="ABCD-EF23",
                                             expires_in_s=60, poll_s=1)
                        m = P.validate("login_poll", m)
                        if stub.logins.get(m["login_id"]) != m["poll_secret"]:
                            return self._msg("login_status", status="expired")
                        return self._msg("login_status", status="ok", token=stub.token, user=stub.user)
                    if p == "/pool/hello":
                        if not self._auth():
                            return
                        m = P.parse_message("hello", body)
                        stub.hellos.append(m)
                        if stub.hello_error is not None:
                            return self._send(400, P.encode_message("error", reason=stub.hello_error))
                        sid = secrets.token_urlsafe(16)
                        stub.sessions.add(sid)
                        return self._msg("hello_ok", session_id=sid, max_instances=stub.max_instances,
                                         heartbeat_s=stub.heartbeat_s, task_poll_s=1, user=stub.user,
                                         leaderboard=stub.leaderboard, server_time=int(time.time()))
                    if p == "/pool/heartbeat":
                        if not self._auth(True):
                            return
                        m = P.parse_message("heartbeat", body)
                        stub.heartbeats.append(m)
                        extra = {"revoked": stub.revoked} if stub.revoked else {}
                        return self._msg("heartbeat_ok", server_time=int(time.time()),
                                         max_instances=stub.max_instances, **extra)
                    if p == "/pool/me":
                        if not self._auth():
                            return
                        m = P.parse_message("me_update", body)
                        stub.me_updates.append(m["leaderboard"])
                        stub.leaderboard = m["leaderboard"]
                        return self._me()
                    if p == "/pool/chunk":
                        if not self._auth(True):
                            return
                        return self._chunk(body)
                except P.ProtocolError as e:
                    return self._send(400, P.encode_message("error", reason=e.payload()))
                self._send(404, b"{}")

            def _chunk(self, body: bytes):
                ch = P.decode_chunk(body)
                h = ch.header
                tid = h["task_id"]
                with stub.lock:
                    prev = stub.chunks.setdefault(tid, [])
                    stub.raw_chunks.setdefault(tid, []).append(body)
                    if stub.reject_chunks is not None:
                        return self._msg("chunk_result", task_id=tid, seq=h["seq"], status="rejected", verify="skipped",
                                         steps=0, reason=stub.reject_chunks)
                    if stub.revoke_on_chunk is not None:
                        stub.revoked = stub.revoke_on_chunk
                        return self._msg("chunk_result", task_id=tid, seq=h["seq"], status="rejected", verify="failed",
                                         steps=0, reason=stub.revoke_on_chunk, token_revoked=True)
                    if h["seq"] != len(prev):
                        return self._err(R.Reason.CHUNK_OUT_OF_ORDER, task_id=tid, chunk=h["seq"], expected=len(prev))
                    if prev:
                        bad = P.check_chain_link(prev[-1], ch)
                        if bad:
                            return self._err(R.Reason.CHAIN_BROKEN, task_id=tid, chunk=h["seq"], step=h["step0"])
                    elif h["chain_prev"] != P.chain_seed(tid).hex():
                        return self._err(R.Reason.CHAIN_BROKEN, task_id=tid, chunk=0, step=0)
                    prev.append(ch)
                return self._msg("chunk_result", task_id=tid, seq=h["seq"], status="accepted", verify="pending",
                                 steps=h["steps"])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False)


def test_stub_speaks_the_protocol():
    """The stub itself: the controller's client gets validated messages back (and refusals as GatewayRefusal)."""
    import pytest

    from ballrain.pool.client_api import GatewayClient, GatewayRefusal
    g = StubGateway()
    try:
        c = GatewayClient(g.url, TOKEN)
        st = c.login_start()
        assert c.login_poll(st["login_id"], st["poll_secret"])["token"] == TOKEN
        h = c.hello("windows", {"kind": "x87soft", "sha256": "0" * 64, "version": "t"},
                    {n: "0" * 64 for n in P.GAME_FILE_LIST}, 1)
        assert h["leaderboard"] is True and c.task()["type"] == "no_task"
        assert c.heartbeat(True, [], 0.0)["type"] == "heartbeat_ok"
        with pytest.raises(GatewayRefusal) as e:
            GatewayClient(g.url, "y" * 40).me()
        assert e.value.code == "token_invalid" and e.value.fatal
    finally:
        g.close()
