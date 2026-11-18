"""The controller's and the workers' gateway client (docs/pool-prototype.md section 2): stdlib HTTP only, every response
read with a size cap and validated against protocol.MESSAGES; downloads land in a sha-addressed cache and are checked
before use. Nothing received is ever unpickled or executed.

Errors:
- GatewayRefusal: the gateway answered with a reason ({"code", "message", "params"}); .message is shown verbatim;
  .fatal is True for protocol/reasons TOKEN_FATAL (stop everything, log in again).
- GatewayUnavailable: no usable answer (network down, timeout, a response that is not the protocol).
"""
from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import protocol as P
from .reasons import TOKEN_FATAL, Reason

CLIENT_VERSION = "0.3.0"  # 0.2: chunk format 2 (the engine-state chain); 0.3: format 3 (chain v3)
MAX_RESPONSE = P.MAX_JSON_BYTES
MAX_MANIFEST = 256 * 1024
MAX_DOWNLOAD = 64 * 1024 * 1024
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class GatewayError(Exception):
    pass


class GatewayUnavailable(GatewayError):
    pass


class GatewayRefusal(GatewayError):
    """A refusal with a reason object; .message is the gateway's Chinese text, to be shown as received."""

    def __init__(self, reason: dict, http_status: int | None = None, where: str = ""):
        self.reason = dict(reason)
        self.code = str(reason.get("code"))
        self.message = str(reason.get("message"))
        self.http_status = http_status
        self.where = where
        try:
            self.fatal = Reason(self.code) in TOKEN_FATAL
        except ValueError:
            self.fatal = False
        super().__init__(f"{self.code}: {self.message}")


def check_gateway_url(url: str) -> str:
    """The gateway base URL (no trailing slash): https, or http only for this machine (a local test gateway)."""
    u = urllib.parse.urlsplit(str(url).strip())
    host = (u.hostname or "").lower()
    if u.scheme not in ("https", "http") or not host:
        raise ValueError("网关地址必须以 https:// 开头")
    if u.scheme == "http" and host not in LOCAL_HOSTS:
        raise ValueError("网关地址必须使用 https（只有本机测试网关可以用 http）")
    if u.query or u.fragment or u.username or u.password:
        raise ValueError("网关地址不能带查询参数或账号信息")
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path.rstrip("/"), "", ""))


class GatewayClient:
    def __init__(self, base_url: str, token: str | None = None, session_id: str | None = None, timeout_s: float = 20.0):
        self.base = check_gateway_url(base_url)
        self.token = token
        self.session_id = session_id
        self.timeout_s = timeout_s
        # a fresh opener: no proxies from the environment for a local gateway, default handlers otherwise
        self._opener = urllib.request.build_opener()

    # ------------------------------------------------------------------------------------------ transport
    def _url(self, path: str) -> str:
        return self.base + path

    def _headers(self, auth: bool, session: bool, ctype: str | None) -> dict:
        h = {"User-Agent": f"ballrain-pool-controller/{CLIENT_VERSION}", "Accept": "application/json"}
        if ctype:
            h["Content-Type"] = ctype
        if auth and self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if session and self.session_id:
            h["X-Pool-Session"] = self.session_id
        return h

    def _raw(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None, auth=True,
             session=False, max_bytes: int = MAX_RESPONSE) -> tuple[int, bytes]:
        req = urllib.request.Request(self._url(path), data=body, method=method,
                                     headers=self._headers(auth, session, ctype))
        try:
            with self._opener.open(req, timeout=self.timeout_s) as r:
                return r.status, _read_capped(r, max_bytes)
        except urllib.error.HTTPError as e:
            try:
                data = _read_capped(e, MAX_RESPONSE)
            except GatewayUnavailable:
                data = b""
            return e.code, data
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise GatewayUnavailable(f"网关连接失败：{getattr(e, 'reason', e)}") from None

    def _call(self, method: str, path: str, expect: tuple[str, ...], msg: dict | None = None, body: bytes | None = None,
              ctype: str | None = None, auth=True, session=False) -> dict:
        if msg is not None:
            body, ctype = P.dumps(msg), "application/json; charset=utf-8"
        status, data = self._raw(method, path, body, ctype, auth, session)
        try:
            obj = P.loads_strict(data)
        except P.ProtocolError:
            raise GatewayUnavailable(f"网关的响应不是协议消息（HTTP {status}）") from None
        kind = obj.get("type") if isinstance(obj, dict) else None
        try:
            if kind == "error":
                raise GatewayRefusal(P.validate("error", obj)["reason"], status, path)
            if kind not in expect:
                raise GatewayUnavailable(f"网关返回了意外的消息 {str(kind)[:40]!r}（HTTP {status}）")
            return P.validate(kind, obj)
        except P.ProtocolError as e:
            raise GatewayUnavailable(f"网关的响应格式不对：{e}") from None

    # ------------------------------------------------------------------------------------------ login
    def login_start(self) -> dict:
        return self._call("POST", "/pool/login", ("login_started",),
                          P.make_message("login_start", client_version=CLIENT_VERSION), auth=False)

    def login_poll(self, login_id: str, poll_secret: str) -> dict:
        return self._call("POST", "/pool/login", ("login_status",),
                          P.make_message("login_poll", login_id=login_id, poll_secret=poll_secret), auth=False)

    # ------------------------------------------------------------------------------------------ session
    def hello(self, platform: str, engine: dict, game_files: dict, instances: int) -> dict:
        r = self._call("POST", "/pool/hello", ("hello_ok",),
                       P.make_message("hello", client_version=CLIENT_VERSION, platform=platform, engine=engine,
                                      game_files=game_files, instances=int(instances)))
        self.session_id = r["session_id"]
        return r

    def task(self) -> dict:
        """'task' or 'no_task' message."""
        return self._call("GET", "/pool/task", ("task", "no_task"), session=True)

    def chunk(self, data: bytes) -> dict:
        """chunk_result (a 'rejected' result is returned, not raised; an error message raises)."""
        return self._call("POST", "/pool/chunk", ("chunk_result",), body=bytes(data),
                          ctype="application/octet-stream", session=True)

    def heartbeat(self, running: bool, instances: list, steps_per_s: float) -> dict:
        return self._call("POST", "/pool/heartbeat", ("heartbeat_ok",),
                          P.make_message("heartbeat", session_id=self.session_id or "", running=bool(running),
                                         instances=instances, steps_per_s=float(steps_per_s)), session=True)

    def me(self) -> dict:
        return self._call("GET", "/pool/me", ("me",))

    def set_leaderboard(self, on: bool) -> dict:
        return self._call("POST", "/pool/me", ("me",), P.make_message("me_update", leaderboard=bool(on)))

    def leaderboard(self) -> dict:
        return self._call("GET", "/pool/leaderboard", ("leaderboard",), auth=False)

    # ------------------------------------------------------------------------------------------ downloads
    def download(self, path: str, dst: Path, max_bytes: int, expect_sha: str | None = None) -> Path:
        """GET a public, sha-addressed file into dst (written to a temporary name, checked, then renamed)."""
        status, data = self._raw("GET", path, auth=False, max_bytes=max_bytes)
        if status != 200:
            try:
                obj = P.validate("error", P.loads_strict(data))
                raise GatewayRefusal(obj["reason"], status, path)
            except P.ProtocolError:
                raise GatewayUnavailable(f"下载 {path} 失败（HTTP {status}）") from None
        if expect_sha is not None and hashlib.sha256(data).hexdigest() != expect_sha:
            raise GatewayUnavailable(f"下载的 {path} 的 sha256 不对")
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.part")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        return dst


def _read_capped(r, max_bytes: int) -> bytes:
    n = r.headers.get("Content-Length") if getattr(r, "headers", None) is not None else None
    if n is not None:
        try:
            if int(n) > max_bytes:
                raise GatewayUnavailable(f"响应太大（{n} 字节）")
        except ValueError:
            raise GatewayUnavailable("响应长度无效") from None
    data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise GatewayUnavailable("响应太大")
    return data


# ============================================================================================== sha-addressed cache

class Cache:
    """<data_dir>/cache: weights/<policy_sha>.{json,npz}, assets/<sha>.npz, and the installed forms (task.install_*)."""

    def __init__(self, root, api: GatewayClient):
        self.root = Path(root)
        self.api = api

    def policy(self, policy_sha: str):
        """(NumpyPolicy, manifest) for a task's policy version; downloads and checks (weights.load_policy)."""
        from . import weights as W
        if not P.HEX64.fullmatch(policy_sha):
            raise ValueError("policy sha")
        d = self.root / "weights"
        mp, npz = d / f"{policy_sha}.json", d / f"{policy_sha}.npz"
        try:
            if not mp.is_file():
                self.api.download(f"/pool/weights/{policy_sha}.json", mp, MAX_MANIFEST)
            m = W.read_manifest(mp)
            if m.get("policy_sha") != policy_sha or not P.HEX64.fullmatch(str(m.get("npz_sha256", ""))):
                raise ValueError("the cached manifest is not this policy version's")
            if not npz.is_file() or npz.stat().st_size != int(m.get("npz_bytes", -1)):
                self.api.download(f"/pool/weights/{policy_sha}.npz", npz, MAX_DOWNLOAD, expect_sha=m["npz_sha256"])
            return W.load_policy(npz, m, expect_sha=policy_sha), m
        except (ValueError, KeyError, TypeError, GatewayUnavailable):
            for p in (mp, npz):  # a bad or stale cached copy (or a sha mismatch): fetch both again next time
                p.unlink(missing_ok=True)
            raise

    def asset(self, sha: str) -> Path:
        if not P.HEX64.fullmatch(sha):
            raise ValueError("asset sha")
        p = self.root / "assets" / f"{sha}.npz"
        if not p.is_file():
            self.api.download(f"/pool/assets/{sha}.npz", p, MAX_DOWNLOAD, expect_sha=sha)
        return p

    def starts(self, sha: str) -> Path:
        from . import task as T
        return T.install_starts_pack(self.asset(sha), self.root, sha)

    def path(self, sha: str, level: int) -> Path:
        from . import task as T
        return T.install_path_asset(self.asset(sha), self.root, level, sha)
