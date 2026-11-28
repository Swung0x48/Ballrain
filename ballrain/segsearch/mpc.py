"""Receding-horizon search with rollback (docs/segsearch.md "MPC search"): every worker keeps one headless engine at the
committed state (a ProcessSnapshot image); for the next window of W frames a cross-entropy search over the 9 actions
per 2 frames, started from the TAS keys at the same route position, evaluates each candidate by load + window + a
lookahead of the TAS keys (aligned by route position); the score is the route progress (arc length along the reference
line) reached at the end, a fall in the window disqualifies, a fall in the lookahead costs `fall_cost`, a pickup the
TAS had taken by that progress and the candidate has not disqualifies. The first `commit` frames of the best candidate
are committed on every worker, and the window moves on until gate B is crossed. The committed keys are then replayed
cold and timed at B like the other tools."""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import tas
from .core import COMBOS, RING, crossing, make_gates, rotate
from .engine import EngineSpec, PICKUPS, default_engine, keys_to_frames, rollout

BALL_OFF = (12, 13)  # Ev.BallOff, Ev.Dead


# ------------------------------------------------------------------------------------------------- pure parts
class Route:
    """Arc length along a reference polyline; projection searched near a hint (the route crosses over itself)."""

    def __init__(self, points):
        self.p = np.asarray(points, float)
        self.cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))]

    def project(self, pos, hint: float, back: float = 10.0, fwd: float = 200.0):
        lo = int(np.searchsorted(self.cum, hint - back))
        hi = int(np.searchsorted(self.cum, hint + fwd)) + 1
        d = np.linalg.norm(self.p[lo:hi] - np.asarray(pos, float), axis=1)
        i = int(np.argmin(d))
        return float(self.cum[lo + i]), float(d[i])

    def progress_track(self, positions, start_hint: float = 0.0, step: float = 5.0, with_dist: bool = False):
        """Monotone progress of a trajectory (each sample searched near the previous); with_dist: also each sample's
        distance to the reference line."""
        out, dist, s = [], [], start_hint
        for q in positions:
            s2, d = self.project(q, s, back=step, fwd=step * 4)
            s = max(s, s2)
            out.append(s)
            dist.append(d)
        return (np.asarray(out), np.asarray(dist)) if with_dist else np.asarray(out)


def initial_probs(base_keys, grain: int = 2, p_base: float = 0.55) -> np.ndarray:
    """(n_decisions, 9) action probabilities: the base (TAS) key's direction gets p_base, its 45-degree neighbours 0.12
    each, 90-degree ones 0.04 each, the rest share what is left."""
    n = len(base_keys) // grain
    P = np.full((n, len(COMBOS)), 0.0)
    for i in range(n):
        k = int(base_keys[i * grain]) & 15
        w = np.full(len(COMBOS), 1.0)
        if k in RING:
            w[COMBOS.index(k)] = 0
            for st, m in ((1, 0.12), (-1, 0.12), (2, 0.04), (-2, 0.04)):
                w[COMBOS.index(rotate(k, st))] = 0
            rest = (1 - p_base - 0.32) / w.sum()
            P[i] = w * rest
            P[i, COMBOS.index(k)] = p_base
            for st, m in ((1, 0.12), (-1, 0.12), (2, 0.04), (-2, 0.04)):
                P[i, COMBOS.index(rotate(k, st))] = m
        else:
            P[i] = 1.0 / len(COMBOS)
            P[i, 0] = max(P[i, 0], p_base)
            P[i] /= P[i].sum()
    return P


def sample(P, rng, n: int, grain: int = 2) -> np.ndarray:
    """n key arrays (n, n_decisions * grain) drawn from per-decision probabilities P."""
    idx = np.array([rng.choice(len(COMBOS), size=n, p=row) for row in P]).T  # (n, n_dec)
    keys = np.asarray(COMBOS, np.uint32)[idx]
    return np.repeat(keys, grain, axis=1)


def refit(P, elites, grain: int = 2, alpha: float = 0.7, floor: float = 0.01) -> np.ndarray:
    """CEM update: P <- (1 - alpha) P + alpha * elite frequencies, with a floor."""
    E = np.asarray(elites)[:, ::grain]
    F = np.zeros_like(P)
    for j, k in enumerate(COMBOS):
        F[:, j] = (E == k).mean(axis=0)
    Q = (1 - alpha) * P + alpha * F
    Q = np.maximum(Q, floor)
    return Q / Q.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------- workers
def _worker(conn, eng: EngineSpec, label: str, work: str, prefix_frames, delta: float, y_floor: float,
            ref_pos=None, ref_keys=None):
    """One engine at the committed state. A load the snapshot refuses (e.g. Windows started another thread-pool thread
    in the engine since the save) rebuilds the engine cold: prefix replay of TAS[:A] + the committed keys, a new image."""
    from ..client import SessionConfig
    from ..headless import HeadlessLauncher
    from .procsnap import SnapshotError
    from .snapsession import SnapSession
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    st = {"s": None, "hist": [], "ck": np.zeros(0, np.uint32), "lens": [], "rebuilds": 0}

    def boot():
        if st["s"] is not None:
            st["s"].close(kill=True)
        tas.save_raw(work / "prefix.bin", np.concatenate([prefix_frames, keys_to_frames(st["ck"], delta)]))
        s = SnapSession(SessionConfig(eng.level, work / "run", copy=label, prefix=work / "prefix.bin", events_txt=False,
                                      hello_timeout_s=120, obs_timeout_s=120),
                        HeadlessLauncher(binary=eng.binary, root=eng.root, socket="tcp", log=True, ball_order=eng.ball_order))
        s.start()
        with open(work.parent / f"pids-{label}.txt", "a") as f:
            f.write(f"{s.handle.pid}\n")
        st["s"] = s
        st["hist"] = [s.save()]

    def load():
        try:
            st["s"].load(st["hist"][-1])
        except SnapshotError:
            st["rebuilds"] += 1
            boot()

    def run(keys, block):
        """Step keys from the loaded state; (positions at block ends, end info)."""
        s = st["s"]
        fr = keys_to_frames(keys, delta)
        P, counts, off, off_at = [], {v: 0 for v in PICKUPS.values()}, False, None
        k = 0
        o = s.obs
        while k < len(fr):
            o = s.act(fr[k:k + block])
            k += block
            p = np.asarray(o.ball["pos"], float)
            P.append(p)
            for c in o.events["code"]:
                c = int(c)
                if c in PICKUPS:
                    counts[PICKUPS[c]] += 1
                if c in BALL_OFF:
                    off = True
            if off or p[1] < y_floor or s.ended:
                off = True
                off_at = k
                break
        v = (np.asarray(o.ball["speed"], float) + np.asarray(o.ball["speed_change"], float)).tolist()
        return P, {"off": off, "off_at": off_at, "counts": counts, "end_pos": P[-1].tolist() if P else None,
                   "end_vel": v, "frames": int(s.frames_done)}

    boot()
    try:
        conn.send(("ready", {"frames": int(st["s"].frames_done), "pos": np.asarray(st["s"].obs.ball["pos"], float).tolist()}))
        while True:
            cmd, arg = conn.recv()
            if cmd == "eval":
                out = []
                for item in arg:
                    keys, block = item[0], item[1]
                    load()
                    P, info = run(keys, block)
                    info["sync_j"] = None
                    if len(item) > 2 and not info["off"]:
                        # re-synchronised lookahead: the TAS keys from the TAS frame nearest to this state
                        f_exp, L = item[2], item[3]
                        lo, hi = max(f_exp - 24, 0), min(f_exp + 25, len(ref_pos))
                        j = lo + int(np.argmin(np.linalg.norm(ref_pos[lo:hi] - P[-1], axis=1)))
                        j = max(0, j + (item[4] if len(item) > 4 else 0))
                        P2, info2 = run(ref_keys[j:j + L], block)
                        info2["counts"] = {k: info["counts"][k] + info2["counts"][k] for k in info["counts"]}
                        info2["off_at"] = None if info2["off_at"] is None else info2["off_at"] + len(keys)
                        info2["sync_j"] = int(j)
                        P, info = P + P2, info2
                    info["path"] = np.asarray(P, np.float32)
                    out.append(info)
                conn.send(("ok", out))
            elif cmd == "commit":
                keys, block = arg
                load()
                P, info = run(keys, block)
                if info["off"]:
                    conn.send(("err", "fell in the committed keys"))
                    continue
                st["ck"] = np.r_[st["ck"], keys].astype(np.uint32)
                st["lens"].append(len(keys))
                try:
                    st["hist"].append(st["s"].save())
                except SnapshotError:  # e.g. a thread that would not go idle: rebuild at the new committed state
                    st["rebuilds"] += 1
                    boot()
                if len(st["hist"]) > 12:
                    st["hist"].pop(0)  # keep the last few for backtracking
                info["path"] = [q.tolist() for q in P]
                info["rebuilds"] = st["rebuilds"]
                conn.send(("ok", info))
            elif cmd == "back":
                n = int(arg)
                if n > len(st["lens"]):
                    conn.send(("err", "nothing to go back to"))
                    continue
                cut = sum(st["lens"][-n:])
                st["ck"] = st["ck"][:len(st["ck"]) - cut]
                del st["lens"][-n:]
                if n < len(st["hist"]):
                    del st["hist"][-n:]
                else:
                    boot()
                conn.send(("ok", len(st["ck"])))
            elif cmd == "close":
                break
    finally:
        if st["s"] is not None:
            st["s"].close(kill=True)


@dataclass
class MPCConfig:
    a: int
    b: int
    window: int = 32
    commit: int = 16
    lookahead: int = 200
    pop: int = 48
    iters: int = 3
    elites: int = 8
    p_base: float = 0.55
    alpha: float = 0.7
    fall_cost: float = 30.0
    off_route: float = 3.0  # corridor half-width around the reference line (units)
    corridor_cost: float = 10.0  # score lost per unit beyond it (at any sample of the candidate's path)
    block: int = 4
    radius: float = 8.0
    workers: int = 7
    seed: int = 0
    max_backtrack: int = 3
    margin: float = 0.5  # a candidate replaces the TAS keys only when it scores this much more (route units)
    null: bool = False
    oracle: str = ""  # a tas-v1 dataset: commit ITS keys (frame-aligned from A) and log how the score ranks its windows  # commit the TAS keys every window (pipeline check: must give the TAS)


class MPC:
    def __init__(self, cfg: MPCConfig, dataset: Path, out: Path, eng: EngineSpec | None = None, log=print):
        self.cfg, self.out, self.log = cfg, Path(out), log
        self.eng = eng or default_engine()
        self.frames = tas.load_raw(Path(dataset) / "run" / "framelog.bin")
        self.tkeys = self.frames["keys"].astype(np.uint32)
        self.delta = float(self.frames["delta"][cfg.a])
        st = np.load(Path(dataset) / "starts.npz")
        self.st_fd, self.st_pos = st["frames_done"], st["pos"].astype(float)
        self.rng = np.random.default_rng(cfg.seed)
        self.evals = 0
        self.engine_s = 0.0
        self.orows = []
        if cfg.oracle:
            self.okeys = tas.load_raw(Path(cfg.oracle) / "run" / "framelog.bin")["keys"].astype(np.uint32)

    # ---- reference: TAS from A on this engine (gate B, TAS progress per frame, TAS pickups by progress)
    def reference(self):
        c = self.cfg
        n = c.b - c.a + c.window + c.lookahead + 600
        r = rollout(self.eng, self.out / "work" / "ref", "hl-47", self.frames[:c.a], [], self.tkeys[c.a:c.a + n],
                    delta=self.delta, block=1, record_sr=True)
        self.ref = r
        g = make_gates(c.a + r["t"], r["pos"], r["vel"], c.a, c.b, c.b - c.a + 10 ** 6, c.radius)
        self.gate = g[-1]
        cr = crossing(self.gate, r["t"], r["pos"])
        if cr is None:
            raise RuntimeError("the TAS does not cross gate B on this engine")
        self.tas_cross, i = cr
        self.tas_t, self.tas_sr = int(r["t"][i]), float(r["sr"][i] - r["sr"][0])
        self.route = Route(r["pos"])  # through B + window + lookahead (the score never runs off its end)
        self.tas_prog = self.route.progress_track(r["pos"])  # per reference sample (t = 0, 1, 2, ...)
        self.y_floor = float(r["pos"][: i + 1, 1].min() - 15)
        ev = [(t, c_) for t, c_ in r["events"] if t > 0 and c_ in PICKUPS]
        self.tas_pick = [(float(self.tas_prog[min(int(t), len(self.tas_prog) - 1)]), PICKUPS[c_]) for t, c_ in ev]
        self.log(f"reference: TAS A->B {self.tas_sr:.1f} ms ({self.tas_t} frames, crossing {self.tas_cross:.2f}), "
                 f"route {self.route.cum[-1]:.0f}, pickups {self.tas_pick}")

    def tas_frame_at(self, s: float) -> int:
        """Absolute TAS frame where the TAS first reaches progress s."""
        i = int(np.searchsorted(self.tas_prog, s))
        return self.cfg.a + min(i, len(self.tas_prog) - 1)

    def need(self, s: float) -> dict:
        out = {v: 0 for v in PICKUPS.values()}
        for sp, k in self.tas_pick:
            if sp + 3.0 <= s:  # a pickup the TAS took 3 units of route back
                out[k] += 1
        return out

    # ---- workers
    def start_workers(self):
        c = self.cfg
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(c.workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, self.eng, f"hl-{40 + i}", str(self.out / "work" / f"hl-{40 + i}"),
                                                    self.frames[:c.a], self.delta, self.y_floor,
                                                    np.asarray(self.ref["pos"], float), self.tkeys[c.a:]), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        infos = [cn.recv()[1] for cn in self.conns]
        self.pos = np.asarray(infos[0]["pos"])

    def broadcast(self, cmd, arg):
        for cn in self.conns:
            cn.send((cmd, arg))
        res = [cn.recv() for cn in self.conns]
        bad = [r for r in res if r[0] != "ok"]
        if bad:
            raise RuntimeError(f"{cmd}: {bad[0][1]}")
        return [r[1] for r in res]

    def evaluate(self, cands, block):
        t0 = time.perf_counter()
        parts = [[] for _ in self.conns]
        for i, k in enumerate(cands):
            parts[i % len(self.conns)].append(k if isinstance(k, tuple) else (k, block))
        for cn, part in zip(self.conns, parts):
            cn.send(("eval", part))
        res = [cn.recv() for cn in self.conns]
        out = [None] * len(cands)
        for w, (st, r) in enumerate(res):
            if st != "ok":
                raise RuntimeError(r)
            for j, info in enumerate(r):
                out[w + j * len(self.conns)] = info
        self.evals += len(cands)
        self.engine_s += (time.perf_counter() - t0) * len(self.conns)
        return out

    def score(self, info, W):
        c = self.cfg
        if info["off"] and info["off_at"] is not None and info["off_at"] <= W:
            return -1e9, None
        # progress tracked along the candidate's own path (never jumps to a nearby later part of the route)
        path = np.asarray(info["path"], float)
        if not len(path):
            return -1e9, None
        prog, dist = self.route.progress_track(path, self.s, step=3.0, with_dist=True)
        s, d = float(prog[-1]), float(dist.max())  # d: the farthest the candidate gets from the reference line
        have = {k: self.counts.get(k, 0) + info["counts"].get(k, 0) for k in self.counts}
        if any(have[k] < v for k, v in self.need(s).items()):
            return -1e9, s
        sc = s - (c.fall_cost if info["off"] else 0.0) - max(0.0, d - c.off_route) * c.corridor_cost
        return sc, s

    # ---- main loop
    def run(self):
        c = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        self.reference()
        self.start_workers()
        self.s, _ = self.route.project(self.pos, 0.0)
        self.cur_pos = self.pos.tolist()
        self.counts = {v: 0 for v in PICKUPS.values()}
        committed = np.zeros(0, np.uint32)
        path = [self.pos.tolist()]
        t0 = time.perf_counter()
        crossed = None
        nwin = 0
        backtracks = 0
        hist = []  # (committed len, s, counts, path len) per commit for backtracking
        try:
            f_prev = None
            plan = None  # keys from the committed point on: the chosen window + the lookahead keys it was scored with
            rel = self.tkeys[c.a:]
            self.n_dev = 0
            while crossed is None and len(committed) < (c.b - c.a) * 2:
                W, L = c.window, c.lookahead
                if f_prev is None:
                    f0 = self.tas_frame_at(self.s)
                else:  # the TAS frame nearest (3D) to the committed state within 8 frames of the expected one
                    e = f_prev + c.commit - c.a
                    lo, hi = max(e - 8, 0), min(e + 9, len(self.ref["pos"]))
                    f0 = c.a + lo + int(np.argmin(np.linalg.norm(self.ref["pos"][lo:hi] - np.asarray(self.cur_pos), axis=1)))
                f_prev = f0
                fexp = f0 - c.a + W
                tas_win = self.tkeys[f0:f0 + W]
                if len(tas_win) < W:
                    tas_win = np.r_[tas_win, np.full(W - len(tas_win), tas_win[-1] if len(tas_win) else 0, np.uint32)]
                if c.oracle:
                    ow = self.okeys[c.a + len(committed):c.a + len(committed) + W]
                    it4 = self.evaluate([(ow, c.block, fexp, L), (tas_win, c.block, fexp, L), (ow, c.block, fexp, L, -2),
                                         (ow, c.block, fexp, L, 2)], c.block)
                    sc4 = [self.score(i, W)[0] for i in it4]
                    (self.out / "oracle_progress.txt").write_text(f"{len(committed)} {f0} {fexp}")
                    self.orows.append([len(committed), int(not np.array_equal(ow, tas_win))] + sc4)
                    keys = ow[:c.commit]
                    infos = self.broadcast("commit", (keys, 2))
                    info = infos[0]
                    committed = np.r_[committed, keys].astype(np.uint32)
                    seg = [path[-1]] + info["path"]
                    path.extend(info["path"])
                    for k, v in info["counts"].items():
                        self.counts[k] += v
                    self.s = float(self.route.progress_track(np.asarray(info["path"], float), self.s, step=3.0)[-1])
                    self.cur_pos = info["end_pos"]
                    tt = np.arange(len(seg)) * 2.0 + (len(committed) - c.commit)
                    cr = crossing(self.gate, tt, np.asarray(seg))
                    if cr is not None:
                        crossed = cr[0]
                    nwin += 1
                    continue
                base = plan[:W] if (plan is not None and len(plan) >= W and not c.null) else tas_win
                binfo = self.evaluate([(base, c.block, fexp, L)], c.block)[0]
                base_score = self.score(binfo, W)[0]
                best = (base_score, base.copy(), binfo)
                P = initial_probs(base, 2, c.p_base)
                for it in range(0 if c.null else c.iters):
                    cands = sample(P, self.rng, c.pop)
                    if it == 0 and not np.array_equal(tas_win, base):
                        cands = np.r_[tas_win[None, :], cands]
                    infos = self.evaluate([(k, c.block, fexp, L) for k in cands], c.block)
                    scores = np.asarray([self.score(i, W)[0] for i in infos])
                    order = np.argsort(-scores, kind="stable")
                    if scores[order[0]] > best[0]:
                        best = (scores[order[0]], cands[order[0]].copy(), infos[order[0]])
                    ok = [cands[j] for j in order[:c.elites] if scores[j] > -1e8]
                    if ok:
                        P = refit(P, ok, 2, c.alpha)
                if best[0] < base_score + c.margin:  # stay on the plan unless clearly better
                    best = (base_score, base.copy(), binfo)
                else:
                    self.n_dev += 1
                if best[0] <= -1e8:
                    f_prev, plan = None, None
                    if backtracks >= c.max_backtrack or not hist:
                        raise RuntimeError(f"no candidate survives at route {self.s:.1f} (frame {len(committed)})")
                    backtracks += 1
                    self.broadcast("back", 1)
                    n_keep, self.s, self.counts, npath = hist.pop()
                    committed = committed[:n_keep]
                    del path[npath:]
                    self.log(f"backtrack to frame {n_keep}")
                    continue
                backtracks = 0
                j = best[2].get("sync_j") if best[2] is not None else None
                full = np.r_[best[1], rel[j:j + L] if j is not None else np.zeros(0, np.uint32)].astype(np.uint32)
                keys = full[:c.commit]
                plan = full[c.commit:]
                infos = self.broadcast("commit", (keys, 2))
                info = infos[0]
                self.rebuilds = max(i.get("rebuilds", 0) for i in infos)
                hist.append((len(committed), self.s, dict(self.counts), len(path)))
                committed = np.r_[committed, keys].astype(np.uint32)
                seg = [path[-1]] + info["path"]
                path.extend(info["path"])
                for k, v in info["counts"].items():
                    self.counts[k] += v
                self.s = float(self.route.progress_track(np.asarray(info["path"], float), self.s, step=3.0)[-1])
                self.cur_pos = info["end_pos"]
                tt = np.arange(len(seg)) * 2.0 + (len(committed) - c.commit)
                cr = crossing(self.gate, tt, np.asarray(seg))
                if cr is not None:
                    crossed = cr[0]
                nwin += 1
                if nwin % 10 == 0 or crossed is not None:
                    tas_here = self.tas_frame_at(self.s) - c.a
                    self.log(f"window {nwin}: frame {len(committed)}, route {self.s:.1f} (TAS reaches it at {tas_here}: "
                             f"{len(committed) - tas_here:+d} frames), plan score {base_score:.2f}, chosen {best[0]:.2f}, "
                             f"{self.evals} evals, {self.n_dev} windows changed the plan, {self.rebuilds} rebuilds, "
                             f"{time.perf_counter() - t0:.0f} s")
        finally:
            for cn in self.conns:
                try:
                    cn.send(("close", None))
                except OSError:
                    pass
            for p in self.procs:
                p.join(30)
        self.wall = time.perf_counter() - t0
        self.path = path
        return committed, crossed

    def verify(self, keys):
        """Cold replay (fresh process, 1-frame blocks) of the committed keys: crossing of B, SR, end state."""
        c = self.cfg
        n = len(keys)
        r = rollout(self.eng, self.out / "work" / "verify", "hl-47", self.frames[:c.a], [], keys, delta=self.delta,
                    block=1, record_sr=True)
        cr = crossing(self.gate, r["t"], r["pos"])
        # determinism: the committed path (every 2nd frame, as the workers saw it) against the cold replay
        P = np.asarray(getattr(self, "path", []), float)
        ti = {int(t): j for j, t in enumerate(r["t"])}
        same = all(int(2 * j) in ti and np.array_equal(r["pos"][ti[2 * j]], q) for j, q in enumerate(P[: n // 2 + 1]))
        if cr is None:
            return {"crossed_B": False, "end": r["end"], "frames": n, "deterministic": bool(same)}
        i = cr[1]
        return {"crossed_B": True, "deterministic": bool(same), "n_compared": int(len(P)), "frames": int(r["t"][i]), "cross": cr[0], "sr_ms": float(r["sr"][i] - r["sr"][0]),
                "delta_frames": int(r["t"][i]) - self.tas_t, "delta_ms": float(r["sr"][i] - r["sr"][0]) - self.tas_sr,
                "pos": r["pos"][: i + 1].tolist(), "end_pos": r["pos"][i].tolist(),
                "end_vel": r["vel"][i].tolist(), "tas_end_vel": self.ref["vel"][self.tas_t].tolist()}


def run_mpc(cfg: MPCConfig, dataset: Path, out: Path, eng: EngineSpec | None = None) -> dict:
    from .search import logger, sha256
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log, f = logger(out / "search.log")
    m = MPC(cfg, dataset, out, eng, log)
    keys, crossed = m.run()
    v = m.verify(keys)
    res = {"a": cfg.a, "b": cfg.b, "config": asdict(cfg), "dataset": Path(dataset).name,
           "framelog_sha256": sha256(Path(dataset) / "run" / "framelog.bin"),
           "engine": {"binary": Path(m.eng.binary).parent.name, "ball_order": m.eng.ball_order},
           "tas": {"frames": m.tas_t, "sr_ms": m.tas_sr, "cross": m.tas_cross,
                   "pos_path": m.ref["pos"][: m.tas_t + 1].tolist()},
           "search_cross": crossed, "evals": m.evals, "engine_s": m.engine_s, "wall_s": m.wall,
           "keys_runs": [[int(k), int(n)] for k, n in __import__("ballrain.segsearch.core", fromlist=["to_runs"]).to_runs(keys)],
           "verify": v}
    if m.orows:
        res["oracle_rows"] = m.orows  # [committed frames, keys differ, score(oracle), score(TAS), score(oracle, sync -2), score(oracle, sync +2)]
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log(f"verify: {json.dumps({k: v[k] for k in v if k not in ('pos',)}, default=float)}")
    f.close()
    return res
