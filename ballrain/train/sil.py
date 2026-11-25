"""Self-imitation (train_m2 --sil-coef): the policy re-learns its own actions that turned out better than expected.

After the GAE of each batch, the rows whose advantage is above `adv_min` batch standard deviations (an action followed
by more return than the value head predicted: e.g. a life that flies over the stone transformer where nearly all fail)
become candidates. A candidate enters the buffer of (raw observation, action, return) only once its own life has gone
on for `horizon` more steps without ending, or ended with LevelFinish within them (offer(): the worker's later rows and
chunks are followed, per worker); a life that dies, stalls or is cut off within the horizon takes its pending
candidates with it. The lambda-return only sees a short way ahead, so without this a life that flies past the
transformer and falls two seconds later would look good at the jump (user, 2026-11-25: learn only real successes). Before every update the buffer is re-scored with the current
value head: a row whose return no longer beats V(s) by `keep_min` advantage standard deviations is dropped (the policy
and value have caught up), and the buffer keeps the `capacity` best rows. Rows stay for as long as they are still
better than expected, so a rare success is replayed for many updates, not only in the batch it came in.

The loss term (ballrain.train.ppo.ppo_update, Oh et al. 2018, "Self-Imitation Learning", with lambda-returns):
    w = clamp((R - V(s)) / adv_std, 0, w_max)            (V(s) detached; adv_std of the current batch)
    L = coef * ( -mean(w * log pi(a|s)) + vcoef * 0.5 * mean(clamp(R - V(s), 0)^2) / adv_std^2 )
on a random `batch` of buffer rows per minibatch. Only better-than-expected outcomes push: a row that fell below V(s)
contributes nothing. Generic: no level, sector or route knowledge; the reward decides what is good (a life that flies
past the transformer and falls has a death penalty and a low return, so it is not imitated)."""
from __future__ import annotations

import numpy as np
import torch


class SILBuffer:
    def __init__(self, capacity: int = 65536, adv_min: float = 2.0, keep_min: float = 0.1, horizon: int = 0,
                 max_pending: int = 20000):
        self.capacity, self.adv_min, self.keep_min = int(capacity), float(adv_min), float(keep_min)
        self.horizon, self.max_pending = int(horizon), int(max_pending)
        self.pending: dict = {}  # worker key -> {"n": rows seen, "rows": [[due, raw, act, ret, score], ...]}
        self.confirmed = self.discarded = 0  # candidates that survived their horizon / died with their life
        self.raw = None  # (n, n_obs) float32
        self.act = np.zeros(0, np.int64)
        self.ret = np.zeros(0, np.float32)
        self.score = np.zeros(0, np.float32)  # (R - V) / adv_std at the last refresh
        self.added = 0  # rows taken in since the start (statistics)

    def __len__(self) -> int:
        return len(self.act)

    def add(self, raw: np.ndarray, act: np.ndarray, ret: np.ndarray, adv: np.ndarray, adv_std: float) -> int:
        """Take in the rows of a batch whose advantage is above adv_min * adv_std; returns how many."""
        if not adv_std > 0:
            return 0
        sel = np.flatnonzero(adv > self.adv_min * adv_std)
        if len(sel) == 0:
            return 0
        r = np.asarray(raw[sel], np.float32)
        self.raw = r if self.raw is None or len(self) == 0 else np.concatenate([self.raw, r])
        self.act = np.concatenate([self.act, np.asarray(act[sel], np.int64)])
        self.ret = np.concatenate([self.ret, np.asarray(ret[sel], np.float32)])
        self.score = np.concatenate([self.score, (np.asarray(adv[sel], np.float32) / adv_std)])
        self.added += len(sel)
        self._trim()
        return len(sel)

    def _append(self, raw, act, ret, score):
        raw = np.asarray(raw, np.float32).reshape(len(act), -1)
        self.raw = raw if self.raw is None or len(self) == 0 else np.concatenate([self.raw, raw])
        self.act = np.concatenate([self.act, np.asarray(act, np.int64)])
        self.ret = np.concatenate([self.ret, np.asarray(ret, np.float32)])
        self.score = np.concatenate([self.score, np.asarray(score, np.float32)])
        self.added += len(act)
        self._trim()

    def offer(self, chunks: list, raw: np.ndarray, act: np.ndarray, ret: np.ndarray, adv: np.ndarray,
              adv_std: float) -> int:
        """The batch's rows in chunk order (the chunks' rows concatenated, as in the PPO batch). Rows above the
        threshold become pending candidates of their worker; a candidate is confirmed (enters the buffer) after
        `horizon` further rows of its life, or when that life ends with LevelFinish; it is dropped when the life ends
        any other way first (death, stall, cap, error). Chunks of one worker must come in order. Returns the rows
        confirmed by this call."""
        thr = self.adv_min * adv_std if adv_std > 0 else np.inf
        before = self.added
        off = 0
        for c in chunks:
            T = len(c["actions"])
            key = (c.get("wid"), c.get("copy"))
            st = self.pending.setdefault(key, {"n": 0, "rows": []})
            ends = np.flatnonzero(np.asarray(c.get("terminated", np.zeros(T, bool)))
                                  | np.asarray(c.get("truncated", np.zeros(T, bool))))
            eps = c.get("episodes") or []
            reasons = [e.get("end") for e in eps] if len(eps) == len(ends) else [None] * len(ends)
            end_at = dict(zip(ends.tolist(), reasons))
            cand = set(np.flatnonzero(adv[off:off + T] > thr).tolist())
            rows = st["rows"]
            for t in range(T):
                st["n"] += 1
                if t in cand:
                    i = off + t
                    rows.append([st["n"] + self.horizon, raw[i], act[i], ret[i], adv[i] / adv_std])
                if t in end_at:  # the life ends at this row: its candidates are confirmed or dropped together
                    if rows:
                        if end_at[t] == "LevelFinish":
                            self._commit(rows)
                        else:
                            self.discarded += len(rows)
                        rows.clear()
                    continue
                k = 0
                while k < len(rows) and rows[k][0] <= st["n"]:
                    k += 1
                if k:
                    self._commit(rows[:k])
                    del rows[:k]
            if len(rows) > self.max_pending:  # a worker that stopped sending: forget its oldest candidates
                self.discarded += len(rows) - self.max_pending
                del rows[:len(rows) - self.max_pending]
            off += T
        return self.added - before

    def _commit(self, rows):
        self.confirmed += len(rows)
        self._append(np.stack([r[1] for r in rows]), [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows])

    def _trim(self):
        if len(self) > self.capacity:
            keep = np.sort(np.argpartition(-self.score, self.capacity - 1)[:self.capacity])
            self._take(keep)

    def _take(self, idx):
        self.raw, self.act, self.ret, self.score = self.raw[idx], self.act[idx], self.ret[idx], self.score[idx]

    def refresh(self, value_fn, obs_n: np.ndarray, adv_std: float, chunk: int = 16384) -> int:
        """Re-score with the current value head (value_fn: normalised obs -> values) and drop the rows that no longer
        beat it by keep_min; returns how many were dropped."""
        if len(self) == 0 or not adv_std > 0:
            return 0
        v = np.concatenate([value_fn(obs_n[i:i + chunk]) for i in range(0, len(obs_n), chunk)])
        self.score = ((self.ret - v) / adv_std).astype(np.float32)
        keep = np.flatnonzero(self.score > self.keep_min)
        dropped = len(self) - len(keep)
        if dropped:
            self._take(keep)
        return dropped


def value_fn(model, device="cpu"):
    def f(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            _, v = model(torch.as_tensor(x, device=device))
        return v.float().cpu().numpy()
    return f
