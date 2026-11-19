"""Verification at scale (docs/volunteer-compute.md 4.1): reputation tiers and volunteer cross-checks.

Tiers (assign()): trial (the probation: every task replayed by our verifier), normal (task_sample), trusted (after
trusted_after passed tasks: trusted_sample), suspect (after any mismatch: every task, until unflagged by hand). The
client never learns its tier or which task was drawn.

Cross-check: a fraction (cross_fraction) of the drawn tasks of normal / trusted accounts is not replayed by us but
played a second time by another random volunteer: the replica has the same start, ball order, sample and life seeds
and policy, so two honest clients of the same build (engine kind and sha, platform, client version) produce the same
actions, observations and rewards (compare(): the engine-state chain itself is seeded with the task id). Two plays that agree verify each other (no verifier time); two that differ
go to our verifier (arbitrate: both replayed in full), and only that replay may flag an account (a peer's word never
does). The same two accounts are not paired again within cross_cooldown_s; a cross-check that finds no peer within
cross_wait_s falls back to our replay. The replica's chunks never reach the learner (the same trajectory twice).
"""
from __future__ import annotations

import json
import secrets

from . import protocol as P
from . import task as T


def build_key(sess) -> str:
    """Two clients that must agree bit for bit: the same engine (kind, sha256), platform and client version."""
    return json.dumps([sess["engine_kind"], sess["engine_sha"], sess["platform"], sess["client_version"]])


def _draw(p: float) -> bool:
    return secrets.randbelow(1_000_000) < int(round(min(max(p, 0.0), 1.0) * 1_000_000))


def assign(store, cfg, forum_id: int, chunk_steps: int) -> tuple:
    """(tier, replay by us, cross-check) for a task about to be issued (fixed before the client plays it)."""
    if store.probation_steps(forum_id) < cfg.probation_chunks * int(chunk_steps):
        return "trial", True, False
    if store.is_suspect(forum_id):
        return "suspect", True, False
    if store.take_escalation(forum_id):
        return "escalated", True, False
    trusted = cfg.trusted_after > 0 and store.passed_tasks(forum_id) >= cfg.trusted_after
    tier = "trusted" if trusted else "normal"
    if not _draw(cfg.trusted_sample if trusted else cfg.task_sample):
        return tier, False, False
    if _draw(cfg.cross_fraction):
        return tier, False, True
    return tier, True, False


def replica_for(store, sess, forum_id: int, cfg, now: int):
    """A replica task for this session, if a cross-check waits for a peer of its build: (task dict, cross_id)."""
    if cfg.cross_fraction <= 0:
        return None
    c = store.take_cross(build_key(sess), forum_id, cfg.cross_cooldown_s)
    if c is None:
        return None
    a = store.task(c["task_a"])
    body = json.loads(a["body"])
    d = dict(body, task_id=T.new_task_id(), issued_at=int(now))
    return P.check(P.TASK, d, "task"), c["cross_id"]


# what two plays of one task must agree on, bit for bit: everything the learner (and the reward) takes. The engine-state
# chain (obs_hash) is seeded with the task id (protocol.chain_seed), so a replica's differs by construction; the rows
# it hashes are compared instead (the observations, the actions, and the reward inputs where present)
COMPARED = ("obs", "actions", "flags", "logp", "values", "rewards", "env_rewards", "sector", "frames", "sr_ms", "arc",
            "rin")


def compare(a_chunks: list, b_chunks: list):
    """None when two plays of one task agree (every COMPARED array, steps and end of every chunk), else a short
    description of the first difference."""
    if len(a_chunks) != len(b_chunks):
        return f"chunk count {len(a_chunks)} vs {len(b_chunks)}"
    for x, y in zip(a_chunks, b_chunks):
        hx, hy = x.header, y.header
        for k in ("steps", "step0", "final", "end"):
            if hx[k] != hy[k]:
                return f"chunk {hx['seq']}: {k} {hx[k]!r} vs {hy[k]!r}"
        for k in COMPARED:
            if (k in x.arrays) != (k in y.arrays):
                return f"chunk {hx['seq']}: {k} in one play only"
            if k in x.arrays and bytes(memoryview(x.arrays[k])) != bytes(memoryview(y.arrays[k])):
                return f"chunk {hx['seq']}: {k} differ"
    return None


def decide(store, pool_dir, log, wait_s: int, decode=None) -> int:
    """One pass: decide the cross-checks whose two tasks are closed, fall back the ones that waited too long.
    Returns the number decided."""
    from pathlib import Path
    decode = decode or (lambda f: P.decode_chunk(f.read_bytes()))
    n = 0
    now = store.now()
    for c in store.crosses("waiting"):
        if now - int(c["created"]) >= wait_s:  # no peer of this build in time: our verifier replays the original
            store.set_task_replay(c["task_a"])
            store.q("UPDATE chunks SET selected=1 WHERE task_id=?", (c["task_a"],))
            store.decide_cross(c["cross_id"], "fallback", {"why": "no peer"})
            n += 1
    for c in store.crosses("issued"):
        if not c["task_b"]:
            continue
        ta, tb = store.task(c["task_a"]), store.task(c["task_b"])
        if ta is None or tb is None or ta["state"] != "closed" or tb["state"] != "closed":
            continue
        root = Path(pool_dir) / "chunks"
        try:
            ca = [decode(root / r["file"]) for r in store.chunks(c["task_a"])]
            cb = [decode(root / r["file"]) for r in store.chunks(c["task_b"])]
            why = compare(ca, cb) if ca and cb else "a play without chunks"
        except Exception as e:  # noqa: BLE001
            why = f"unreadable: {type(e).__name__}"
        if why is None:
            store.pass_cross_task(c["task_a"], replica=False)
            store.pass_cross_task(c["task_b"], replica=True)
            store.decide_cross(c["cross_id"], "passed")
            log(f"cross-check {c['cross_id'][:8]}: the two plays agree ({len(ca)} chunks): both verified")
        else:  # arbitration: our verifier replays both in full; only its replay may flag an account
            for t in (c["task_a"], c["task_b"]):
                store.set_task_replay(t)
                store.q("UPDATE chunks SET selected=1 WHERE task_id=?", (t,))
            store.decide_cross(c["cross_id"], "arbitrate", {"why": why})
            log(f"cross-check {c['cross_id'][:8]}: the plays differ ({why}): both go to the verifier")
        n += 1
    return n
