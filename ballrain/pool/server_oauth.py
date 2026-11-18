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
# the forum sits behind Cloudflare, which bans urllib's default "Python-urllib/3.x" (error 1010, HTTP 403 on both
# /oauth/token and /api/user): every server-side forum request names the gateway with a generic agent instead
USER_AGENT = "ballrain-pool-gateway/1.0"
USER_FIELDS = ("id", "username", "nickname", "avatar_url", "is_email_confirmed", "joined_at", "comment_count")


@dataclass(frozen=True)
class EligibilityRules:
    """Which forum-account checks the gateway applies before it hands out a pool token ([eligibility] in
    runtime/pool/config.toml, server_config.load_config), in this order: account age, email, replies. Each can be
    switched off; unknown_join_date says what happens when the forum gives no join date (joined_at null: some forum
    accounts, e.g. imported ones, have none): "refuse" (a clear message) or "allow" (only the age check is skipped)."""
    account_age: bool = True
    min_account_days: int = 30
    unknown_join_date: str = "refuse"
    email_confirmed: bool = True
    replies: bool = True
    min_replies: int = 5

    def describe(self) -> str:
        """The conditions in force, for the login page (Chinese)."""
        parts = []
        if self.account_age and self.min_account_days > 0:
            parts.append(f"论坛注册满 {int(self.min_account_days)} 天"
                         + ("（论坛没有提供注册时间的账号不受这条限制）" if self.unknown_join_date == "allow" else ""))
        if self.email_confirmed:
            parts.append("邮箱已验证")
        if self.replies and self.min_replies > 0:
            parts.append(f"至少 {int(self.min_replies)} 条回复")
        return "、".join(parts) if parts else "任何论坛账号都可以"


UNKNOWN_JOIN_DATE = ("refuse", "allow")


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
                                              "Accept": "application/json", "User-Agent": USER_AGENT})
        o = self._json(req)
        tok = o.get("access_token")
        if not isinstance(tok, str) or not tok or len(tok) > 4096:
            raise ForumError("no access token")
        return tok

    def user(self, access_token: str) -> dict:
        """The forum user's fields the pool reads (USER_FIELDS); email is dropped here."""
        req = urllib.request.Request(f"{self.cfg.base_url}/api/user", method="GET",
                                     headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json",
                                              "User-Agent": USER_AGENT})
        o = self._json(req)
        if isinstance(o.get("data"), dict):  # JSON:API wrapping ({"data": {"id", "attributes": {...}}})
            d = o["data"]
            o = dict(d.get("attributes") or {}, id=d.get("id"))
        return {k: o.get(k) for k in USER_FIELDS}  # nothing else (email) survives this line


def parse_joined(j) -> datetime | None:
    """The forum's joined_at in any of the forms a forum API serialises a date in: ISO 8601 (with or without zone,
    fraction, "Z"), "YYYY-MM-DD HH:MM:SS", or a Unix time in seconds / milliseconds (number or digit string). A
    failure names only the value's type and length (never the value: it is user data)."""
    if j is None:  # the forum gives no join date for this account: eligibility decides (unknown_join_date)
        return None
    try:
        if isinstance(j, bool):
            raise ValueError
        if isinstance(j, str) and j.strip().lstrip("-").replace(".", "", 1).isdigit():
            j = float(j.strip())
        if isinstance(j, (int, float)):
            t = float(j)
            if not math.isfinite(t) or t <= 0:
                raise ValueError
            if t > 1e11:  # milliseconds
                t /= 1000.0
            return datetime.fromtimestamp(t, timezone.utc)
        if not isinstance(j, str) or not j.strip():
            raise ValueError
        d = datetime.fromisoformat(j.strip().replace("Z", "+00:00").replace("z", "+00:00"))
        return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        raise ForumError(f"forum joined_at ({type(j).__name__}"
                         + (f", length {len(j)}" if isinstance(j, str) else "") + ")") from None


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
    joined = parse_joined(u.get("joined_at"))
    confirmed = u.get("is_email_confirmed")  # a bool, or 0 / 1 (as a number or a string) from some serialisers
    if not isinstance(confirmed, bool):
        if confirmed in (0, 1) or confirmed in ("0", "1"):
            confirmed = str(confirmed) == "1"
        else:
            raise ForumError(f"forum is_email_confirmed ({type(confirmed).__name__})")
    cc = u.get("comment_count")  # an int, or a digit string
    if isinstance(cc, str) and cc.strip().isdigit():
        cc = int(cc.strip())
    if isinstance(cc, bool) or not isinstance(cc, int) or cc < 0:
        raise ForumError(f"forum comment_count ({type(cc).__name__})")
    return {"user": {"forum_id": fid, "username": username, "nickname": nickname, "avatar_url": av},
            "joined": joined.timestamp() if joined is not None else None, "email_confirmed": confirmed,
            "comments": cc}


def eligibility(rec: dict, now: float, min_days: int = 30, min_replies: int = 5,
                rules: EligibilityRules | None = None) -> dict | None:
    """None when the account may hold a pool token, else the refusal payload. Checked in this order (each only when
    rules switch it on): account age (account_too_new), email (email_not_verified), replies (too_few_replies).
    Without rules: all three with min_days / min_replies, an unknown join date refused."""
    r = rules or EligibilityRules(min_account_days=int(min_days), min_replies=int(min_replies))
    if r.account_age and r.min_account_days > 0:
        if rec.get("joined") is None:
            if r.unknown_join_date != "allow":
                # the existing code (old clients check codes against their own list), a message of its own
                return {"code": Reason.ACCOUNT_TOO_NEW.value,
                        "message": f"论坛没有提供这个账号的注册时间，无法核对“注册满 {int(r.min_account_days)} 天”的条件，"
                                   "暂时不能领取算力池令牌。请联系算力池管理员。",
                        "params": {"min_days": int(r.min_account_days), "join_date": "unknown"}}
        else:
            days = int(math.floor((now - rec["joined"]) / 86400))
            if days < r.min_account_days:
                return payload(Reason.ACCOUNT_TOO_NEW, days=max(days, 0), min_days=int(r.min_account_days))
    if r.email_confirmed and not rec["email_confirmed"]:
        return payload(Reason.EMAIL_NOT_VERIFIED)
    if r.replies and rec["comments"] < r.min_replies:
        return payload(Reason.TOO_FEW_REPLIES, replies=rec["comments"], min_replies=int(r.min_replies))
    return None
