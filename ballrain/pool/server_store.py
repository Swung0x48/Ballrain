"""Gateway state in sqlite (runtime/pool/state.sqlite3; docs/pool-prototype.md 6).

What is stored: forum user id, username, nickname, avatar URL, the "上榜" flag and the revocation state per forum
account; sha256 of each pool token (never the token); OAuth states and controller logins (sha256 of their secrets);
sessions (engine kind / sha, platform, client version); tasks; chunks (file name, size, verification state). No
email, no IP, no forum access token. Revocation is per forum account: a new login of a revoked account is refused
with the stored reason.

Thread-safe (one connection, one lock, WAL); the verifier may open the same file from another process.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS accounts (
    forum_id INTEGER PRIMARY KEY, username TEXT NOT NULL, nickname TEXT NOT NULL, avatar_url TEXT,
    created INTEGER NOT NULL, last_login INTEGER NOT NULL, leaderboard INTEGER NOT NULL DEFAULT 1,
    revoked INTEGER NOT NULL DEFAULT 0, revoke_reason TEXT, revoked_at INTEGER);
CREATE TABLE IF NOT EXISTS tokens (
    token_hash TEXT PRIMARY KEY, forum_id INTEGER NOT NULL, created INTEGER NOT NULL, active INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS tokens_by_account ON tokens(forum_id);
CREATE TABLE IF NOT EXISTS oauth_states (
    state_hash TEXT PRIMARY KEY, nonce_hash TEXT NOT NULL, login_id TEXT, created INTEGER NOT NULL,
    used INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS logins (
    login_id TEXT PRIMARY KEY, poll_hash TEXT NOT NULL, created INTEGER NOT NULL, status TEXT NOT NULL,
    reason TEXT, forum_id INTEGER, delivered INTEGER NOT NULL DEFAULT 0, user_code TEXT,
    confirm_hash TEXT, browser_hash TEXT);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, forum_id INTEGER NOT NULL, token_hash TEXT NOT NULL, engine_kind TEXT NOT NULL,
    engine_sha TEXT NOT NULL, platform TEXT NOT NULL, client_version TEXT NOT NULL, instances INTEGER NOT NULL,
    created INTEGER NOT NULL, last_seen INTEGER NOT NULL, running INTEGER NOT NULL DEFAULT 0,
    steps_per_s REAL NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY, forum_id INTEGER NOT NULL, session_id TEXT NOT NULL, body TEXT NOT NULL,
    policy_sha TEXT NOT NULL, issued_at INTEGER NOT NULL, deadline INTEGER NOT NULL, next_seq INTEGER NOT NULL DEFAULT 0,
    steps INTEGER NOT NULL DEFAULT 0, final INTEGER NOT NULL DEFAULT 0, end_reason TEXT,
    state TEXT NOT NULL DEFAULT 'open', verify_attempts INTEGER NOT NULL DEFAULT 0, verify_after INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS tasks_by_account ON tasks(forum_id, state);
CREATE TABLE IF NOT EXISTS chunks (
    task_id TEXT NOT NULL, seq INTEGER NOT NULL, forum_id INTEGER NOT NULL, steps INTEGER NOT NULL,
    nbytes INTEGER NOT NULL, file TEXT NOT NULL, received INTEGER NOT NULL, selected INTEGER NOT NULL,
    status TEXT NOT NULL, verify TEXT NOT NULL, reason TEXT, PRIMARY KEY (task_id, seq));
CREATE INDEX IF NOT EXISTS chunks_by_account ON chunks(forum_id, verify);
"""
# chunks.status: held (stored, not released to anything) | released (verified or sampled out) | rejected (failed the
#   replay) | quarantined (an unverified chunk of an account revoked for a replay mismatch)
# chunks.verify: pending | passed | failed | skipped (not selected for the replay) | replay_failed (verifier fault)
# tasks.state: open | closed (final chunk received or deadline passed; waiting for the verifier) | done | failed
# logins: user_code is shown by the controller and on the confirmation page; confirm_hash / browser_hash bind the
#   "continue to the forum" link to the browser that rendered that page (sha256 of the link nonce / of its cookie)
MIGRATIONS = {"logins": {"user_code": "TEXT", "confirm_hash": "TEXT", "browser_hash": "TEXT"}}


class ChunkRefused(Exception):
    """add_chunk refused the chunk inside its transaction: .why is "duplicate", "order", "expired" or "revoked"."""

    def __init__(self, why: str, detail=None):
        super().__init__(why)
        self.why, self.detail = why, detail


def sha_hex(s: str | bytes) -> str:
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


class Store:
    def __init__(self, path, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.executescript(SCHEMA)
            for table, cols in MIGRATIONS.items():  # a state file from an older prototype
                have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})").fetchall()}
                for c, typ in cols.items():
                    if c not in have:
                        self.db.execute(f"ALTER TABLE {table} ADD COLUMN {c} {typ}")

    def now(self) -> int:
        return int(self.clock())

    def q(self, sql: str, args=()) -> list:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def one(self, sql: str, args=()):
        r = self.q(sql, args)
        return r[0] if r else None

    def tx(self, fn):
        """Run fn(db) in one transaction (BEGIN IMMEDIATE ... COMMIT / ROLLBACK)."""
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                out = fn(self.db)
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")
            return out

    def close(self):
        with self.lock:
            self.db.close()

    # ------------------------------------------------------------------------------------------ accounts / tokens
    def account(self, forum_id: int):
        return self.one("SELECT * FROM accounts WHERE forum_id=?", (int(forum_id),))

    def upsert_account(self, user: dict) -> None:
        now = self.now()
        self.q("INSERT INTO accounts(forum_id, username, nickname, avatar_url, created, last_login) VALUES(?,?,?,?,?,?) "
               "ON CONFLICT(forum_id) DO UPDATE SET username=excluded.username, nickname=excluded.nickname, "
               "avatar_url=excluded.avatar_url, last_login=excluded.last_login",
               (int(user["forum_id"]), user["username"], user["nickname"], user.get("avatar_url"), now, now))

    def issue_token(self, forum_id: int, token: str) -> None:
        """The account's only active token from now on (older ones become invalid)."""
        h = sha_hex(token)

        def fn(db):
            db.execute("UPDATE tokens SET active=0 WHERE forum_id=?", (int(forum_id),))
            db.execute("INSERT INTO tokens(token_hash, forum_id, created, active) VALUES(?,?,?,1)",
                       (h, int(forum_id), self.now()))
        self.tx(fn)

    def token(self, token: str):
        """(token row, account row) for a presented token, or None."""
        t = self.one("SELECT * FROM tokens WHERE token_hash=?", (sha_hex(token),))
        if t is None:
            return None
        return t, self.account(t["forum_id"])

    def set_leaderboard(self, forum_id: int, on: bool) -> None:
        self.q("UPDATE accounts SET leaderboard=? WHERE forum_id=?", (1 if on else 0, int(forum_id)))

    def revoke(self, forum_id: int, reason: dict, quarantine: bool = True) -> None:
        """Revoke the account (all its tokens) with a stored reason payload; quarantine its unverified chunks."""
        def fn(db):
            db.execute("UPDATE accounts SET revoked=1, revoke_reason=?, revoked_at=? WHERE forum_id=?",
                       (json.dumps(reason, ensure_ascii=False), self.now(), int(forum_id)))
            if quarantine:
                db.execute("UPDATE chunks SET status='quarantined' WHERE forum_id=? AND status='held'", (int(forum_id),))
                db.execute("UPDATE chunks SET status='quarantined' WHERE forum_id=? AND status='released' "
                           "AND verify='skipped'", (int(forum_id),))
        self.tx(fn)

    def unrevoke(self, forum_id: int) -> None:
        self.q("UPDATE accounts SET revoked=0, revoke_reason=NULL, revoked_at=NULL WHERE forum_id=?", (int(forum_id),))

    # ------------------------------------------------------------------------------------------ oauth / logins
    def add_state(self, state: str, nonce: str, login_id: str | None) -> None:
        self.q("INSERT INTO oauth_states(state_hash, nonce_hash, login_id, created) VALUES(?,?,?,?)",
               (sha_hex(state), sha_hex(nonce), login_id, self.now()))

    def take_state(self, state: str, nonce: str | None, max_age: int):
        """The state's login_id ("" for a browser-only login) if the state is known, unused, not expired and was
        issued to this browser (nonce cookie); marks it used. None otherwise."""
        h = sha_hex(state)

        def fn(db):
            r = db.execute("SELECT * FROM oauth_states WHERE state_hash=?", (h,)).fetchone()
            if r is None or r["used"] or self.now() - r["created"] > max_age:
                return None
            db.execute("UPDATE oauth_states SET used=1 WHERE state_hash=?", (h,))
            if nonce is None or sha_hex(nonce) != r["nonce_hash"]:
                return None
            return r["login_id"] or ""
        return self.tx(fn)

    def add_login(self, login_id: str, poll_secret: str, user_code: str = "") -> None:
        self.q("INSERT INTO logins(login_id, poll_hash, created, status, user_code) VALUES(?,?,?, 'pending', ?)",
               (login_id, sha_hex(poll_secret), self.now(), user_code))

    def set_login_confirm(self, login_id: str, confirm_nonce: str, browser_secret: str) -> None:
        """The confirmation page was rendered: only its link, from the browser holding its cookie, continues."""
        self.q("UPDATE logins SET confirm_hash=?, browser_hash=? WHERE login_id=? AND status='pending'",
               (sha_hex(confirm_nonce), sha_hex(browser_secret), login_id))

    def take_login_confirm(self, login_id: str, confirm_nonce: str, browser_secret: str | None) -> bool:
        """True (once) when the link nonce and the browser cookie are the ones of the last rendered page."""
        def fn(db):
            r = db.execute("SELECT confirm_hash, browser_hash, status FROM logins WHERE login_id=?", (login_id,)).fetchone()
            if r is None or r["status"] != "pending" or not r["confirm_hash"] or not browser_secret:
                return False
            ok = (hmac.compare_digest(r["confirm_hash"], sha_hex(confirm_nonce))
                  and hmac.compare_digest(r["browser_hash"] or "", sha_hex(browser_secret)))
            if ok:
                db.execute("UPDATE logins SET confirm_hash=NULL WHERE login_id=?", (login_id,))
            return ok
        return self.tx(fn)

    def login(self, login_id: str):
        return self.one("SELECT * FROM logins WHERE login_id=?", (login_id,))

    def finish_login(self, login_id: str, status: str, reason: dict | None = None, forum_id: int | None = None):
        self.q("UPDATE logins SET status=?, reason=?, forum_id=? WHERE login_id=?",
               (status, json.dumps(reason, ensure_ascii=False) if reason else None, forum_id, login_id))

    def mark_delivered(self, login_id: str) -> None:
        self.q("UPDATE logins SET delivered=1 WHERE login_id=?", (login_id,))

    def prune(self, max_age: int) -> None:
        t = self.now() - max_age
        self.q("DELETE FROM oauth_states WHERE created < ?", (t,))
        self.q("DELETE FROM logins WHERE created < ?", (t,))

    # ------------------------------------------------------------------------------------------ sessions
    def add_session(self, sid: str, forum_id: int, token_hash: str, hello: dict) -> None:
        now = self.now()
        e = hello["engine"]
        self.q("INSERT INTO sessions(session_id, forum_id, token_hash, engine_kind, engine_sha, platform, client_version, "
               "instances, created, last_seen) VALUES(?,?,?,?,?,?,?,?,?,?)",
               (sid, int(forum_id), token_hash, e["kind"], e["sha256"], hello["platform"], hello["client_version"],
                int(hello["instances"]), now, now))

    def session(self, sid: str):
        return self.one("SELECT * FROM sessions WHERE session_id=?", (sid,))

    def touch_session(self, sid: str, running: bool | None = None, steps_per_s: float | None = None) -> None:
        if running is None:
            self.q("UPDATE sessions SET last_seen=? WHERE session_id=?", (self.now(), sid))
        else:
            self.q("UPDATE sessions SET last_seen=?, running=?, steps_per_s=? WHERE session_id=?",
                   (self.now(), 1 if running else 0, float(steps_per_s or 0.0), sid))

    # ------------------------------------------------------------------------------------------ tasks / chunks
    def add_task(self, task: dict, forum_id: int, session_id: str) -> None:
        self.q("INSERT INTO tasks(task_id, forum_id, session_id, body, policy_sha, issued_at, deadline) "
               "VALUES(?,?,?,?,?,?,?)",
               (task["task_id"], int(forum_id), session_id, json.dumps(task, sort_keys=True), task["policy_sha"],
                int(task["issued_at"]), int(task["issued_at"]) + int(task["deadline_s"])))

    def task(self, task_id: str):
        return self.one("SELECT * FROM tasks WHERE task_id=?", (task_id,))

    def open_tasks(self, forum_id: int, live_s: int) -> int:
        """Open, unexpired tasks of the account whose session was seen in the last live_s seconds (a hard-killed
        controller's tasks stop counting against the instance limit once its session goes quiet; they still expire
        at their deadline and the verifier takes what arrived)."""
        now = self.now()
        r = self.one("SELECT COUNT(*) AS n FROM tasks t JOIN sessions s ON s.session_id=t.session_id "
                     "WHERE t.forum_id=? AND t.state='open' AND t.deadline>? AND s.last_seen>=?",
                     (int(forum_id), now, now - int(live_s)))
        return int(r["n"])

    def add_chunk(self, task_id: str, seq: int, forum_id: int, steps: int, nbytes: int, file: str, selected: bool,
                  final: bool, end: str | None) -> None:
        """Insert a chunk and advance its task, in one IMMEDIATE transaction that re-checks what the gateway checked
        before (the verifier, possibly another process, may have closed the task or revoked the account meanwhile):
        the task is still open, before its deadline, expects this seq, and the account is not revoked. Raises
        ChunkRefused otherwise (nothing written)."""
        def fn(db):
            t = db.execute("SELECT state, next_seq, deadline, final FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            a = db.execute("SELECT revoked, revoke_reason FROM accounts WHERE forum_id=?", (int(forum_id),)).fetchone()
            if a is not None and a["revoked"]:
                raise ChunkRefused("revoked", a["revoke_reason"])
            if t is None or int(t["next_seq"]) > int(seq):
                raise ChunkRefused("duplicate")
            if t["state"] != "open" or t["final"]:
                expired = not t["final"] and self.now() >= int(t["deadline"])
                raise ChunkRefused("expired" if expired else "order", int(t["next_seq"]))
            if self.now() >= int(t["deadline"]):
                raise ChunkRefused("expired")
            if int(t["next_seq"]) != int(seq):
                raise ChunkRefused("order", int(t["next_seq"]))
            try:
                db.execute("INSERT INTO chunks(task_id, seq, forum_id, steps, nbytes, file, received, selected, status, "
                           "verify) VALUES(?,?,?,?,?,?,?,?, 'held', 'pending')",
                           (task_id, int(seq), int(forum_id), int(steps), int(nbytes), file, self.now(),
                            1 if selected else 0))
            except sqlite3.IntegrityError:
                raise ChunkRefused("duplicate") from None
            db.execute("UPDATE tasks SET next_seq=?, steps=steps+?, final=?, end_reason=?, state=? "
                       "WHERE task_id=? AND state='open'",
                       (int(seq) + 1, int(steps), 1 if final else 0, end, "closed" if final else "open", task_id))
        self.tx(fn)

    def chunks(self, task_id: str) -> list:
        return self.q("SELECT * FROM chunks WHERE task_id=? ORDER BY seq", (task_id,))

    def account_chunks(self, forum_id: int) -> int:
        r = self.one("SELECT COUNT(*) AS n FROM chunks WHERE forum_id=?", (int(forum_id),))
        return int(r["n"])

    def probation_steps(self, forum_id: int) -> int:
        """Steps of the account that count toward its probation: chunks waiting for the replay or passed by it (all of
        them are replayed during the probation), not rejected ones. Counted in steps, not chunks, so short "stopped"
        tasks cannot run the probation down."""
        r = self.one("SELECT COALESCE(SUM(steps), 0) AS n FROM chunks WHERE forum_id=? "
                     "AND verify IN ('pending', 'passed') AND status!='rejected'", (int(forum_id),))
        return int(r["n"])

    def held_bytes(self, forum_id: int | None = None) -> int:
        """Bytes of chunks still held (stored, waiting for the verifier), of one account or of everyone. Released,
        rejected, quarantined and replay_failed chunks do not count (they never block new work; disk retention is
        separate)."""
        if forum_id is None:
            r = self.one("SELECT COALESCE(SUM(nbytes), 0) AS n FROM chunks WHERE status='held' AND verify='pending'")
        else:
            r = self.one("SELECT COALESCE(SUM(nbytes), 0) AS n FROM chunks WHERE forum_id=? AND status='held' "
                         "AND verify='pending'",
                         (int(forum_id),))
        return int(r["n"])

    def close_task(self, task_id: str, forum_id: int, chunks: int, reason: str) -> str | None:
        """The client's task_end: an open task of the account with exactly `chunks` chunks received becomes closed (as
        at its deadline). Returns the task's state afterwards, None when the task is unknown / not the account's,
        "order" when the chunk count differs (chunks still on their way)."""
        out: list = []

        def fn(db):
            t = db.execute("SELECT state, final, next_seq, forum_id FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if t is None or int(t["forum_id"]) != int(forum_id):
                out.append(None)
                return
            if t["state"] != "open":
                out.append(str(t["state"]))
                return
            if int(t["next_seq"]) != int(chunks):
                out.append("order")
                return
            db.execute("UPDATE tasks SET state='closed', end_reason=? WHERE task_id=? AND state='open'",
                       (f"client: {reason}"[:300], task_id))
            out.append("closed")
        self.tx(fn)
        return out[0]

    def close_expired(self) -> int:
        """Open tasks past their deadline become closed (the verifier takes what arrived)."""
        with self.lock:
            cur = self.db.execute("UPDATE tasks SET state='closed' WHERE state='open' AND deadline<=?", (self.now(),))
            return cur.rowcount

    def tasks_to_verify(self, limit: int = 10) -> list:
        return self.q("SELECT * FROM tasks WHERE state='closed' AND verify_after<=? ORDER BY issued_at LIMIT ?",
                      (self.now(), int(limit)))

    def finish_verify(self, task_id: str, upto: int, passed: bool, reason: dict | None, failed_seq: int | None,
                      max_seq: int | None = None) -> None:
        """Record a task's verification: chunks <= upto passed (all of them on success), later ones up to max_seq (the
        last chunk of the verifier's snapshot) skipped; on a mismatch the failing chunk is rejected (its reason
        stored). A chunk that arrived after the snapshot is not decided here."""
        top = (1 << 62) if max_seq is None else int(max_seq)

        def fn(db):
            if passed:
                db.execute("UPDATE chunks SET verify='passed', status='released' WHERE task_id=? AND seq<=? "
                           "AND status='held'", (task_id, int(upto)))
                db.execute("UPDATE chunks SET verify='skipped', status='released' WHERE task_id=? AND seq>? AND seq<=? "
                           "AND status='held'", (task_id, int(upto), top))
                db.execute("UPDATE tasks SET state='done' WHERE task_id=?", (task_id,))
            else:
                db.execute("UPDATE chunks SET verify='passed', status='released' WHERE task_id=? AND seq<? "
                           "AND status='held'", (task_id, int(failed_seq)))
                db.execute("UPDATE chunks SET verify='failed', status='rejected', reason=? WHERE task_id=? AND seq=?",
                           (json.dumps(reason, ensure_ascii=False), task_id, int(failed_seq)))
                db.execute("UPDATE tasks SET state='failed' WHERE task_id=?", (task_id,))
        self.tx(fn)

    def verify_later(self, task_id: str, delay_s: int, give_up_after: int = 5) -> None:
        """A verifier fault: retry the task later; after give_up_after attempts its chunks stay held as
        replay_failed (not the player's fault, not counted as verified)."""
        def fn(db):
            r = db.execute("SELECT verify_attempts FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            n = int(r["verify_attempts"]) + 1 if r else 1
            if n >= give_up_after:
                db.execute("UPDATE tasks SET state='done', verify_attempts=? WHERE task_id=?", (n, task_id))
                db.execute("UPDATE chunks SET verify='replay_failed' WHERE task_id=? AND status='held'", (task_id,))
            else:
                db.execute("UPDATE tasks SET verify_attempts=?, verify_after=? WHERE task_id=?",
                           (n, self.now() + int(delay_s) * n, task_id))
        self.tx(fn)

    def set_task_state(self, task_id: str, state: str) -> None:
        self.q("UPDATE tasks SET state=? WHERE task_id=?", (state, task_id))

    def skip_task(self, task_id: str, max_seq: int | None = None) -> None:
        """No chunk of the task was selected: all (up to the snapshot's max_seq) are released unreplayed."""
        top = (1 << 62) if max_seq is None else int(max_seq)

        def fn(db):
            db.execute("UPDATE chunks SET verify='skipped', status='released' WHERE task_id=? AND status='held' "
                       "AND seq<=?", (task_id, top))
            db.execute("UPDATE tasks SET state='done' WHERE task_id=?", (task_id,))
        self.tx(fn)

    # ------------------------------------------------------------------------------------------ statistics
    def stats(self, forum_id: int) -> dict:
        f = int(forum_id)
        r = self.one("SELECT "
                     "COALESCE(SUM(CASE WHEN status!='rejected' THEN steps END), 0) AS acc, "
                     "COALESCE(SUM(CASE WHEN verify='pending' AND status='held' THEN steps END), 0) AS pend, "
                     "COALESCE(SUM(CASE WHEN status='released' THEN steps END), 0) AS ver, "
                     "COUNT(CASE WHEN status!='rejected' THEN 1 END) AS cacc, "
                     "COUNT(CASE WHEN status='rejected' THEN 1 END) AS crej FROM chunks WHERE forum_id=?", (f,))
        last = self.one("SELECT reason FROM chunks WHERE forum_id=? AND reason IS NOT NULL ORDER BY received DESC LIMIT 1",
                        (f,))
        out = {"steps_accepted": int(r["acc"]), "steps_pending": int(r["pend"]), "steps_verified": int(r["ver"]),
               "chunks_accepted": int(r["cacc"]), "chunks_rejected": int(r["crej"])}
        if last is not None:
            out["last_rejection"] = json.loads(last["reason"])
        return out

    def leaderboard(self, limit: int = 200) -> list:
        """(nickname, verified steps) of the accounts with the flag on and not revoked, most first."""
        return self.q("SELECT a.nickname AS nickname, a.username AS username, SUM(c.steps) AS steps "
                      "FROM accounts a JOIN chunks c ON c.forum_id=a.forum_id "
                      "WHERE a.leaderboard=1 AND a.revoked=0 AND c.status='released' "
                      "GROUP BY a.forum_id HAVING steps>0 ORDER BY steps DESC, a.forum_id LIMIT ?", (int(limit),))
