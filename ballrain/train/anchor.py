"""Mastered-state anchor (scripts/train_m2.py --anchor-mastered): a KL term to a frozen teacher on the batch states of
lives that started in a start family the policy has measurably mastered. Level-agnostic: the families come from the
run's own start configuration, mastery from the learner's own success measure, nothing is a route position.

Start families (family_key): every --rest-starts entry ("rest:<entry>", identical entries are one family), every
rehearsal range ("rehearse:lo-hi"), curriculum TAS starts by bins of record frames ("band:lo-hi", "band:k0" for
level-load draws), respawn lives by sector ("respawn:s<n>") and plain level-load lives ("level", runs without TAS
starts). The worker tags each life with its start (WorkerSpec.start_tags; ballrain.env BallanceEnv._draw_start records
which list the frame came from), each chunk row with its life (row_families). All rows of a life in a mastered family
are anchored (--anchor-rows life, default), or only those before it reached its next checkpoint (first-cp).

Mastery (MasteredAnchor): rolling success over the last `window` lives of a family (success as the learner defines it:
the next checkpoint, or LevelFinish with --success finish). A family is mastered once it has >= min_lives lives in the
window at a rate >= threshold. Hysteresis: a mastered family whose rate drops below the threshold stays anchored for
`grace` more lives below it (a recovery resets the count); only then is it released. A released family needs a full
mastering window again.

Teachers: an explicit checkpoint for every family, or (automatic) a snapshot of the policy per family taken when it
became mastered: a policy whose own lives were in the mastering window. A family keeps that snapshot while anchored
(also during the grace lives: never a snapshot of the drifting policy). Snapshots are shared: a newly mastered family
reuses the newest stored snapshot that is no older than the oldest life of its mastering window (that policy played in
the window too), else the current policy is stored. At most max_teachers are kept: at the cap, the oldest snapshot
whose families are all at or above the threshold now is retired and its families move to the new snapshot (they are
mastered by the current policies as well); when there is none the oldest snapshot is dropped and its families are
released (logged). Optional ratchet: a family whose whole window was played by policies newer than its teacher, at a
rate at least the one it had when it got that teacher, moves to a newer snapshot (the anchor follows improvements,
e.g. a faster line, but never adopts a policy whose own lives fell short).

Balanced weighting (row_weights): each anchored family present in the batch has the same total weight, so the anchor
term is the mean over families of the mean KL within each family (not a mean over rows: a rare start is not diluted by
frequent ones). A family with fewer than min_rows rows gets n / min_rows of a full share (a handful of rows does not
carry a whole family's pull). Optional drop scaling: a family's share is multiplied by 1 + drop_scale * (peak rate -
current rate)+, so a family that is slipping is pulled harder.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np

KINDS = ("rest", "rehearse", "band", "respawn", "level")
INVALID_ENDS = ("error", "camera_off_quarter")  # lives that are no outcome (not counted)


# ------------------------------------------------------------------------------------------------ start families

def _num(v) -> str:
    v = float(v)
    return str(int(v)) if v == int(v) else repr(v)


def rest_entry_label(e) -> str:
    """A --rest-starts entry (parsed) as its spec text: lo-hi, lo-hi@x:z:jitter[:max_frames], lo-hi@branch:<id>:s,
    lo-hi@branch-move:<id>:s."""
    e = list(e)
    head = f"{_num(e[0])}-{_num(e[1])}"
    if len(e) == 2:
        return head
    if len(e) == 3 and isinstance(e[2], dict):
        d = e[2]
        kind = "branch-move" if d.get("branch_move") else "branch"
        rest = {k: v for k, v in d.items() if k not in ("branch_s", "id", "branch_move")}
        lab = f"{head}@{kind}:" + (f"{d['id']}:" if d.get("id") else "") + _num(d.get("branch_s", 0.0))
        return lab + (json.dumps(rest, sort_keys=True) if rest else "")
    return head + "@" + ":".join(_num(x) for x in e[2:])


def family_key(tag, *, band_frames: int, base_frame: int, kinds=KINDS) -> str | None:
    """The start family of a life from its start tag (WorkerSpec.start_tags); None = not a family (unknown source,
    e.g. an older worker, or a kind left out of `kinds`)."""
    if not isinstance(tag, dict):
        return None
    kind = tag.get("kind")
    src = tag.get("source") or {}
    s = src.get("src") if isinstance(src, dict) else None
    try:
        if kind == "respawn":
            return f"respawn:s{int(tag.get('sector') or 0)}" if "respawn" in kinds else None
        if s == "rest":
            return "rest:" + rest_entry_label(src["entry"]) if "rest" in kinds else None
        if s == "rehearse":
            lo, hi = src["entry"][:2]
            return f"rehearse:{int(lo)}-{int(hi)}" if "rehearse" in kinds else None
        if s == "curriculum":
            if "band" not in kinds:
                return None
            k = int(tag.get("frame") or 0)
            if k == 0:
                return "band:k0"
            bf = max(int(band_frames), 1)
            lo = base_frame + ((k - base_frame) // bf) * bf
            return f"band:{lo}-{lo + bf - 1}"
        if kind == "level" and s is None:
            return "level" if "level" in kinds else None
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return None


def row_families(chunks, key_of, past_cp_counts: bool = False) -> list:
    """Per batch row (the chunks concatenated in order) the family key of its life or None: chunks without start tags
    (pool chunks, older workers) give None; with past_cp_counts False (--success cp) rows observed after the life reached
    a later sector than it started in give None too (the family's mastered stretch ends at its next checkpoint)."""
    out = []
    for c in chunks:
        T = len(c["rewards"])
        rows = [None] * T
        ref, tags = c.get("start_ref"), c.get("start_tags")
        if not c.get("pool") and ref is not None and tags is not None:
            try:
                keys = [key_of(t) for t in tags]
                ref = np.asarray(ref)
                past = np.asarray(c.get("past_cp", np.zeros(T, bool)), bool)
                if len(ref) == T and len(past) == T:
                    rows = [keys[r] if 0 <= r < len(keys) and (past_cp_counts or not p) else None
                            for r, p in zip(ref.tolist(), past.tolist())]
            except Exception:  # noqa: BLE001  malformed tags: no anchor on this chunk, never a crash
                rows = [None] * T
        out.extend(rows)
    return out


# ------------------------------------------------------------------------------------------------ mastery state

class _Family:
    __slots__ = ("outcomes", "lives", "anchored", "teacher", "peak", "below", "since", "pending", "teacher_rate")

    def __init__(self, window: int):
        self.outcomes: deque = deque(maxlen=max(int(window), 1))  # (success, policy version at the life's start)
        self.lives = 0
        self.anchored = False
        self.teacher = None  # snapshot id (automatic teachers)
        self.peak = 0.0
        self.below = 0  # lives in a row with the rate below the threshold while anchored
        self.since = None  # update at which it became mastered
        self.pending = False  # mastered, waiting for a teacher snapshot
        self.teacher_rate = 0.0  # the rolling rate when the current teacher was assigned (ratchet)

    @property
    def rate(self) -> float:
        return float(np.mean([s for s, _ in self.outcomes])) if self.outcomes else float("nan")

    def to_dict(self) -> dict:
        return {"outcomes": [[bool(s), None if v is None else int(v)] for s, v in self.outcomes], "lives": self.lives,
                "anchored": self.anchored, "teacher": self.teacher, "peak": self.peak, "below": self.below,
                "since": self.since, "pending": self.pending, "teacher_rate": self.teacher_rate}


class MasteredAnchor:
    """Per-family rolling success, mastery with hysteresis, teacher snapshot references (no torch here: snapshots are
    opaque payloads made by the caller's snapshot function)."""

    def __init__(self, window: int = 40, min_lives: int = 40, threshold: float = 0.9, grace: int = 80,
                 drop_scale: float = 0.0, max_teachers: int = 8, explicit: bool = False, ratchet: bool = False):
        self.window, self.min_lives = int(window), int(min_lives)
        self.threshold, self.grace = float(threshold), int(grace)
        self.drop_scale, self.max_teachers = float(drop_scale), max(int(max_teachers), 1)
        self.explicit = bool(explicit)  # one explicit teacher for every family: no snapshots
        # ratchet: a family whose last `window` lives were all played by policies newer than its teacher, at a rate at
        # least the one it had when that teacher was assigned, moves to a newer snapshot (a later policy measured as
        # good on it): the anchor follows improvements without ever adopting a policy whose own lives fell short
        self.ratchet = bool(ratchet)
        self.families: dict[str, _Family] = {}
        self.snapshots: dict[int, dict] = {}  # snapshot id (= the update it was taken at) -> payload
        self.events: list = []  # (what, family / snapshot, detail) since the caller last drained them
        self.counts = {"mastered": 0, "released": 0, "retired": 0, "dropped": 0}

    # -------------------------------------------------------------------------- outcomes
    def add_life(self, key: str, success: bool, version: int | None = None, update: int | None = None) -> None:
        f = self.families.get(key)
        if f is None:
            f = self.families[key] = _Family(self.window)
        f.outcomes.append((bool(success), None if version is None else int(version)))
        f.lives += 1
        rate, n = f.rate, len(f.outcomes)
        if not f.anchored:
            if n >= self.min_lives and rate >= self.threshold:
                f.anchored, f.pending, f.peak, f.below, f.since = True, not self.explicit, rate, 0, update
                self.counts["mastered"] += 1
                self.events.append(("mastered", key, f"rate {rate:.2f} over {n} lives"))
            return
        if rate >= self.threshold:
            f.below = 0
            f.peak = max(f.peak, rate)
            return
        f.below += 1
        if f.below > self.grace:
            self._release(key, f"rate {rate:.2f} below {self.threshold:g} for {f.below} lives")

    def _release(self, key: str, why: str) -> None:
        f = self.families[key]
        f.anchored, f.pending, f.teacher, f.below, f.since = False, False, None, 0, None
        self.counts["released"] += 1
        self.events.append(("released", key, why))
        self._gc()

    def _gc(self) -> None:
        used = {f.teacher for f in self.families.values() if f.anchored and f.teacher is not None}
        for sid in [s for s in self.snapshots if s not in used]:
            del self.snapshots[sid]

    def healthy(self, f: _Family) -> bool:
        return len(f.outcomes) >= self.min_lives and f.rate >= self.threshold

    # -------------------------------------------------------------------------- teachers
    def refresh(self, update: int, snapshot) -> None:
        """Give every anchored family without a teacher one (automatic mode): snapshot() returns the payload of the
        current policy, called at most once. A family whose teacher was lost (dropped snapshot, explicit -> automatic
        on resume) gets the current policy only while it is healthy, otherwise it is released. With ratchet, a family
        whose full window was played by policies newer than its teacher at >= its teacher-time rate moves forward."""
        if self.explicit:
            for f in self.families.values():
                f.pending = False
            return
        for key, f in self.families.items():
            if f.anchored and not f.pending and f.teacher not in self.snapshots:
                if self.healthy(f):
                    f.pending = True
                else:
                    self._release(key, "teacher snapshot lost while below the threshold")
            elif (self.ratchet and f.anchored and not f.pending and f.teacher in self.snapshots and f.below == 0
                  and len(f.outcomes) == self.window and self.healthy(f) and f.rate >= f.teacher_rate
                  and all(v is not None and v > f.teacher for _, v in f.outcomes)):
                # only into a stored snapshot inside the window or a free slot: a ratchet never forces the cap
                # (which could drop the snapshot of a family in its grace)
                lo = min(v for _, v in f.outcomes)
                if len(self.snapshots) < self.max_teachers or any(lo <= s <= update for s in self.snapshots):
                    f.pending = True
        pending = [k for k, f in self.families.items() if f.anchored and f.pending]
        if not pending:
            return
        new_sid = None
        for key in pending:
            f = self.families[key]
            vs = [v for _, v in f.outcomes if v is not None]
            oldest = min(vs) if vs else update
            reuse = max((s for s in self.snapshots if oldest <= s <= update), default=None)
            f.teacher_rate = f.rate
            if reuse is not None:
                f.teacher, f.pending = reuse, False
                self.events.append(("teacher", key, f"snapshot u{reuse} (stored, inside its mastering window)"))
                continue
            if new_sid is None:
                new_sid = self._new_snapshot(update, snapshot)
            f.teacher, f.pending = new_sid, False
            self.events.append(("teacher", key, f"snapshot u{new_sid}"))
        self._gc()  # (a ratchet leaves its old snapshot behind)

    def _new_snapshot(self, update: int, snapshot) -> int:
        while len(self.snapshots) >= self.max_teachers:
            members = {s: [k for k, f in self.families.items() if f.anchored and f.teacher == s] for s in self.snapshots}
            retire = next((s for s in sorted(self.snapshots)
                           if all(self.healthy(self.families[k]) for k in members[s])), None)
            if retire is not None:
                for k in members[retire]:
                    self.families[k].teacher = int(update)
                del self.snapshots[retire]
                self.counts["retired"] += 1
                self.events.append(("retired", f"u{retire}", f"{len(members[retire])} families -> u{update}"))
                continue
            old = min(self.snapshots)
            del self.snapshots[old]
            self.counts["dropped"] += 1
            self.events.append(("dropped", f"u{old}", f"cap {self.max_teachers}: releases {members[old]}"))
            for k in members[old]:
                self._release(k, f"teacher u{old} dropped at the snapshot cap")
        self.snapshots[int(update)] = snapshot()
        return int(update)

    # -------------------------------------------------------------------------- batch
    def anchored(self) -> dict:
        """family key -> teacher id ("explicit" or a snapshot id) of the families anchored now."""
        out = {}
        for k, f in self.families.items():
            if not f.anchored:
                continue
            if self.explicit:
                out[k] = "explicit"
            elif f.teacher in self.snapshots:
                out[k] = f.teacher
        return out

    def weight(self, key: str) -> float:
        f = self.families[key]
        return 1.0 + self.drop_scale * max(0.0, f.peak - f.rate) if self.drop_scale else 1.0

    def summary(self) -> dict:
        an = [f for f in self.families.values() if f.anchored]
        return {"families": len(self.families), "mastered": len(an), "in_grace": sum(f.below > 0 for f in an),
                "teachers": len(self.snapshots), **{f"n_{k}": v for k, v in self.counts.items()}}

    def drain_events(self) -> list:
        ev, self.events = self.events, []
        return ev

    # -------------------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {"config": {"window": self.window, "min_lives": self.min_lives, "threshold": self.threshold,
                           "grace": self.grace, "drop_scale": self.drop_scale, "max_teachers": self.max_teachers,
                           "explicit": self.explicit, "ratchet": self.ratchet},
                "families": {k: f.to_dict() for k, f in self.families.items()},
                "snapshots": dict(self.snapshots), "counts": dict(self.counts)}

    def load_state_dict(self, st: dict) -> None:
        """Family statistics and snapshots of a checkpoint; the configuration stays this one's (the command line)."""
        self.families = {}
        for k, d in (st.get("families") or {}).items():
            f = _Family(self.window)
            for s, v in d.get("outcomes", []):
                f.outcomes.append((bool(s), v))
            f.lives = int(d.get("lives", len(f.outcomes)))
            f.anchored, f.teacher = bool(d.get("anchored")), d.get("teacher")
            f.peak, f.below, f.since = float(d.get("peak", 0.0)), int(d.get("below", 0)), d.get("since")
            f.pending = bool(d.get("pending")) and not self.explicit
            f.teacher_rate = float(d.get("teacher_rate", 0.0))
            self.families[k] = f
        self.snapshots = {int(s): p for s, p in (st.get("snapshots") or {}).items()}
        self.counts.update(st.get("counts") or {})
        if not self.explicit:
            self._gc()

    def drop_snapshot(self, sid: int, why: str) -> None:
        """Remove a stored snapshot (e.g. its observation layout no longer fits); its families get a new teacher at the
        next refresh while healthy, otherwise they are released."""
        self.snapshots.pop(int(sid), None)
        self.counts["dropped"] += 1
        self.events.append(("dropped", f"u{sid}", why))


# ------------------------------------------------------------------------------------------------ batch weights

def row_weights(row_keys, anchored: dict, weight_of=lambda k: 1.0, min_rows: int = 16):
    """Per-row anchor weights (float32, summing to 1 over the anchored rows): row r of family f gets
    w_f / (F * max(n_f, min_rows)) with F the number of anchored families present and n_f their rows, so the weighted
    sum of the per-row KL is the mean over families of each family's mean KL (times w_f). Rows of families that are not
    anchored (or None) get 0. Also returns {family: (row indices)} of the anchored families present."""
    n = len(row_keys)
    groups: dict = {}
    for i, k in enumerate(row_keys):
        if k is not None and k in anchored:
            groups.setdefault(k, []).append(i)
    w = np.zeros(n, np.float32)
    if not groups:
        return w, {}
    F = len(groups)
    for k, idx in groups.items():
        w[idx] = weight_of(k) / (F * max(len(idx), int(min_rows)))
    return w, {k: np.asarray(v, np.int64) for k, v in groups.items()}


# ------------------------------------------------------------------------------------------------ teachers

def teacher_layout_error(mine: dict, theirs: dict, t_obs: int, n_obs: int, t_act: int, n_act: int,
                         ignore: frozenset = frozenset()) -> str | None:
    """None when a teacher with features `theirs` (t_obs entries, t_act actions) can read this run's observations
    (features `mine`, n_obs entries), otherwise the reason. The teacher reads the first t_obs entries: a teacher without
    the f7 reach block or the f8 contact blocks (appended at the end) is allowed when its layout is a prefix of this
    one (ballrain.train.grow.check_prefix), and a plain-reach teacher under a mechanism-aware run (reach_version
    <tv>-mech1-...: the same 59-value layout)."""
    appended = ({"reach", "reach_version", "version"} if (mine.get("reach") and not theirs.get("reach")) else set())
    if mine.get("version") == "f8" and theirs.get("version") != "f8":
        appended.add("version")
    tv, mv = theirs.get("reach_version"), mine.get("reach_version")
    if tv and isinstance(mv, str) and mv.startswith(f"{tv}-mech1-"):
        appended.add("reach_version")
    # ignore: keys the caller has validated itself (an append-only branch repin: --teacher-allow-context-append)
    diff = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k) and k not in appended | set(ignore))
    if (t_obs != n_obs and not appended) or int(t_act) != n_act or diff:
        return f"observations differ (n_obs {t_obs} vs {n_obs}, features {diff})"
    if appended:
        from ballrain.train.grow import GrowError, check_prefix
        try:
            check_prefix(theirs, mine, t_obs, n_obs)
        except GrowError as e:
            return str(e)
    return None


def actor_payload(model, rms_state: dict, n_obs: int, features: dict, arch: dict, update: int) -> dict:
    """A teacher snapshot of the current policy: the actor's parameters only (the critic is not needed), the
    observation statistics it acts with, its layout and architecture."""
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("v.")}
    return {"actor": sd, "rms": {k: (v.copy() if hasattr(v, "copy") else v) for k, v in rms_state.items()},
            "n_obs": int(n_obs), "features": dict(features), "arch": dict(arch), "update": int(update)}


def build_teacher(payload: dict, n_act: int, device="cpu"):
    """(module, rms, n_obs) of a snapshot payload."""
    from ballrain.train.ppo import ActorCritic
    ar = payload["arch"]
    hidden = tuple(int(x) for x in str(ar["hidden"]).split(",")) if isinstance(ar["hidden"], str) else tuple(ar["hidden"])
    m = ActorCritic(int(payload["n_obs"]), n_act, hidden, ar.get("activation", "tanh"),
                    ar.get("actor_heads", "single")).to(device)
    missing, unexpected = m.load_state_dict(payload["actor"], strict=False)
    if unexpected or any(not k.startswith("v.") for k in missing):
        raise ValueError(f"snapshot does not fit its architecture (missing {missing}, unexpected {unexpected})")
    m.eval()
    return m, payload["rms"], int(payload["n_obs"])


def teacher_logp(teachers: dict, groups: dict, anchored: dict, raw: np.ndarray, n_act: int, device="cpu"):
    """Teacher log-probabilities (n, n_act) float32 on the anchored rows (0 elsewhere): teachers maps a teacher id to
    (module, rms, n_obs); every teacher reads raw[:, :n_obs] normalised with its own statistics."""
    import torch

    from ballrain.train.ppo import normalize
    out = np.zeros((len(raw), n_act), np.float32)
    by_teacher: dict = {}
    for k, idx in groups.items():
        by_teacher.setdefault(anchored[k], []).append(idx)
    with torch.no_grad():
        for tid, parts in by_teacher.items():
            idx = np.concatenate(parts)
            m, rs, t_obs = teachers[tid]
            x = torch.as_tensor(normalize(raw[idx, :t_obs], rs), device=device)
            out[idx] = torch.log_softmax(m(x)[0], -1).cpu().numpy()
    return out


def family_kl(model, obs_n: np.ndarray, t_logp: np.ndarray, groups: dict, device="cpu") -> dict:
    """KL(teacher || policy) of the current policy, mean per anchored family present in the batch."""
    import torch
    if not groups:
        return {}
    idx = np.concatenate(list(groups.values()))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.as_tensor(obs_n[idx], device=device))[0], -1).cpu().numpy()
    lt = t_logp[idx]
    kl = (np.exp(lt) * (lt - lp)).sum(-1)
    out, o = {}, 0
    for k, g in groups.items():
        out[k] = float(kl[o:o + len(g)].mean())
        o += len(g)
    return out
