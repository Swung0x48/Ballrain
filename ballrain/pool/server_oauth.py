"""Forum OAuth (Flarum, flarum-oauth-center: authorization code flow) and the eligibility rules of the pool token
(docs/volunteer-compute.md 10.2 / 11.2).

The gateway sends the browser to <forum>/oauth/authorize (response_type=code, client_id, the exact redirect_uri,
scope user.read, state), exchanges the code server-side at <forum>/oauth/token (client secret never leaves the
gateway) and reads <forum>/api/user once with the forum access token, which is then dropped. Of the user record
only id, username, nickname, avatar_url, is_email_confirmed, joined_at and comment_count are read; email is
discarded unread-into-anything (it is never put into a variable that outlives the parse, a log or an exception).
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from .reasons import Reason, payload

MAX_FORUM_RESPONSE = 64 * 1024
USER_FIELDS = ("id", "username", "nickname", "avatar_url", "is_email_confirmed", "joined_at", "comment_count")


class ForumError(RuntimeError):
    """The forum refused or returned something unusable. .detail is safe to show (no user data in it)."""

    def __init__(self, detail: str):
        self.detail = detail[:120]
        super().__init__(self.detail)


@dataclass
class ForumConfig:
    base_url: str  # https://forum.ballance.top (no trailing slash)
    client_id: str
    client_secret: str
    redirect_uri: str  # <public_base_url>/pool/oauth/callback, registered exactly at the forum
    scope: str = "user.read"
    timeout_s: float = 10.0


def _clean(s, n: int) -> str:
    s = "" if s is None else str(s)
    return "".join(ch for ch in s if ch.isprintable())[:n]


class ForumClient:
    def __init__(self, cfg: ForumConfig, opener=None):
        self.cfg = cfg
        self.opener = opener or urllib.request.build_opener()

    def authorize_url(self, state: str) -> str:
        q = urllib.parse.urlencode({"response_type": "code", "client_id": self.cfg.client_id,
                                    "redirect_uri": self.cfg.redirect_uri, "scope": self.cfg.scope, "state": state})
        return f"{self.cfg.base_url}/oauth/authorize?{q}"

    def _json(self, req) -> dict:
        try:
            with self.opener.open(req, timeout=self.cfg.timeout_s) as r:
                data = r.read(MAX_FORUM_RESPONSE + 1)
        except urllib.error.HTTPError as e:
            raise ForumError(f"forum HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise ForumError(f"forum unreachable ({type(e).__name__})") from None
        if len(data) > MAX_FORUM_RESPONSE:
            raise ForumError("forum response too large")
        try:
            o = json.loads(data.decode("utf-8"))
        except ValueError:
            raise ForumError("forum response is not JSON") from None
        if not isinstance(o, dict):
            raise ForumError("forum response is not an object")
        return o

    def exchange(self, code: str) -> str:
        body = urllib.parse.urlencode({"grant_type": "authorization_code", "code": code,
                                       "redirect_uri": self.cfg.redirect_uri, "client_id": self.cfg.client_id,
                                       "client_secret": self.cfg.client_secret}).encode("ascii")
        req = urllib.request.Request(f"{self.cfg.base_url}/oauth/token", data=body, method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded",
                                              "Accept": "application/json"})
        o = self._json(req)
        tok = o.get("access_token")
        if not isinstance(tok, str) or not tok or len(tok) > 4096:
            raise ForumError("no access token")
        return tok

    def user(self, access_token: str) -> dict:
        """The forum user's fields the pool reads (USER_FIELDS); email is dropped here."""
        req = urllib.request.Request(f"{self.cfg.base_url}/api/user", method="GET",
                                     headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"})
        o = self._json(req)
        if isinstance(o.get("data"), dict):  # JSON:API wrapping ({"data": {"id", "attributes": {...}}})
            d = o["data"]
            o = dict(d.get("attributes") or {}, id=d.get("id"))
        return {k: o.get(k) for k in USER_FIELDS}  # nothing else (email) survives this line


def parse_user(u: dict) -> dict:
    """The forum fields -> the pool's user record (protocol USER) + eligibility inputs; ForumError when malformed."""
    try:
        fid = int(u["id"])
    except (TypeError, ValueError, KeyError):
        raise ForumError("forum user id") from None
    if not 1 <= fid <= 2 ** 53:
        raise ForumError("forum user id")
    username = _clean(u.get("username"), 64)
    if not username:
        raise ForumError("forum username")
    nickname = _clean(u.get("nickname"), 64) or username
    av = u.get("avatar_url")
    av = _clean(av, 500) if isinstance(av, str) and av.startswith(("https://", "http://")) else None
    j = u.get("joined_at")
    try:
        joined = datetime.fromisoformat(str(j).replace("Z", "+00:00"))
        if joined.tzinfo is None:
            joined = joined.replace(tzinfo=timezone.utc)
    except ValueError:
        raise ForumError("forum joined_at") from None
    confirmed = u.get("is_email_confirmed")
    if not isinstance(confirmed, bool):
        raise ForumError("forum is_email_confirmed")
    cc = u.get("comment_count")
    if isinstance(cc, bool) or not isinstance(cc, int) or cc < 0:
        raise ForumError("forum comment_count")
    return {"user": {"forum_id": fid, "username": username, "nickname": nickname, "avatar_url": av},
            "joined": joined.timestamp(), "email_confirmed": confirmed, "comments": cc}


def eligibility(rec: dict, now: float, min_days: int = 30, min_replies: int = 5) -> dict | None:
    """None when the account may hold a pool token, else the refusal payload. Checked in this order: account age
    (account_too_new), email (email_not_verified), replies (too_few_replies)."""
    days = int(math.floor((now - rec["joined"]) / 86400))
    if days < min_days:
        return payload(Reason.ACCOUNT_TOO_NEW, days=max(days, 0), min_days=min_days)
    if not rec["email_confirmed"]:
        return payload(Reason.EMAIL_NOT_VERIFIED)
    if rec["comments"] < min_replies:
        return payload(Reason.TOO_FEW_REPLIES, replies=rec["comments"], min_replies=min_replies)
    return None
