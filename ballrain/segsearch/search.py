"""Staged beam search between two TAS gates (docs/segsearch.md).

The stretch A -> B is cut into sub-gates every `gate_every` TAS frames along the TAS line. Stage g expands the k
earliest (plus k_speed fastest-moving) branches that crossed gate g: each gets m rollouts whose keys are the TAS keys
from the TAS's own gate-g frame on, with random local edits in the first `window` frames (or random holds there, or no
edit). A rollout runs on until the last gate, a ball-off, or until it is `slack` frames behind the best branch at a
gate; every gate it crosses adds a branch there. Branches that miss a pickup the TAS took before the previous gate are
dropped. The branches at B that pass the end-state check are replayed cold (fresh process, 1-frame blocks) and must
reproduce the search's positions exactly before they are reported."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import tas
from .core import EndCheck, Node, crossings, make_gates, mutate, random_holds, select, to_runs, from_runs
from .engine import EngineSpec, ball_off, default_engine, pickup_counts, rollout

COPIES = tuple(f"hl-{i}" for i in range(40, 48))  # this tool's headless copy labels (other sessions own the rest)


@dataclass
class SearchConfig:
    a: int
    b: int
    gate_every: int = 264
    radius: float = 6.0
    k: int = 6
    k_speed: int = 2
    k_surv: int = 3
    ahead: int = 2
    m: int = 10
    rounds: int = 1
    block: int = 4
    window: int = 528
    slack: float = 264.0
    p_plain: float = 0.1
    p_random: float = 0.15
    mean_edits: float = 2.0
    workers: int = 6
    seed: int = 0
    verify_top: int = 3
    verify_failed: int = 2  # also replay cold the earliest branches at B that fail the end check (shown, never claimed)
    y_margin: float = 15.0
    end: EndCheck = field(default_factory=EndCheck)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------- worker processes
_W = {}


def _init(labels, eng, out, tas_prefix):
    label = labels.get()
    _W.update(label=label, eng=eng, work=Path(out) / "work" / label, prefix=tas_prefix)


def _run(task):
    kw = dict(task)
    rid = kw.pop("rid")
    r = rollout(_W["eng"], _W["work"], _W["label"], _W["prefix"], kw.pop("base"), kw.pop("cont"), **kw)
    r["rid"] = rid
    r["copy"] = _W["label"]
    with open(Path(_W["work"]).parent / f"pids-{_W['label']}.txt", "a") as f:
        f.write(f"{r['pid']}\n")
    return r


class Search:
    def __init__(self, cfg: SearchConfig, tas_frames: np.ndarray, out: Path, eng: EngineSpec | None = None, log=print):
        self.cfg = cfg
        self.frames = tas_frames
        self.out = Path(out)
        self.eng = eng or default_engine()
        self.log = log
        self.delta = float(tas_frames["delta"][cfg.a])
        self.tkeys = tas_frames["keys"].astype(np.uint32)
        self.prefix = tas_frames[:cfg.a].copy()
        self.rng = np.random.default_rng(cfg.seed)
        self.rollouts: dict[int, dict] = {}
        self.archive: dict[int, list[Node]] = {}
        self.next_rid = 0
        self.engine_s = 0.0
        self.start_s = 0.0
        self.n_rollouts = 0
        self.ends: dict[str, int] = {}
        self.dropped: dict[int, list] = {}  # gate -> [branches dropped for a missed pickup, earliest such crossing]

    # -------------------------------------------------------------------------------------------- reference (TAS)
    def reference(self, pool=None):
        """The TAS itself from A, 1-frame blocks, on this engine: gates, TAS gate times, TAS pickups and end state."""
        c = self.cfg
        n = c.b - c.a + 600
        cont = self.tkeys[c.a:c.a + n]
        task = dict(rid=-1, base=np.zeros(0, np.uint32), cont=cont, delta=self.delta, block=1, record_sr=True)
        r = pool.apply(_run, (task,)) if pool is not None else rollout(self.eng, self.out / "work" / COPIES[0], COPIES[0],
                                                                       self.prefix, task["base"], cont, delta=self.delta,
                                                                       block=1, record_sr=True)
        self.ref = r
        if c.a + int(r["t"][-1]) < c.b:
            raise RuntimeError(f"the TAS reference ends ({r['end']}) at frame {c.a + int(r['t'][-1])}, before B = {c.b}")
        self.gates = make_gates(c.a + r["t"], r["pos"], r["vel"], c.a, c.b, c.gate_every, c.radius)
        cr = crossings(self.gates, 0, r["t"], r["pos"])
        if len(cr) != len(self.gates):
            raise RuntimeError(f"the TAS crosses only {len(cr)} of {len(self.gates)} gates on this engine")
        self.tas_time = {g: cr[g][0] for g in cr}
        ev = [e for e in r["events"] if e[0] > 0]  # the first observation carries the prefix's events
        self.tas_counts = {g: pickup_counts(ev, r["t"][cr[g][1]]) for g in cr}
        last = len(self.gates) - 1
        i = cr[last][1]
        self.ref_end = {"pos": r["pos"][i].tolist(), "vel": r["vel"][i].tolist(), "type": int(r["type"][i]),
                        "counts": self.tas_counts[last], "t": int(r["t"][i]), "sr_ms": float(r["sr"][i] - r["sr"][0]),
                        "cross": float(cr[last][0])}
        self.y_floor = float(r["pos"][: i + 1, 1].min() - c.y_margin)
        self.best = dict(self.tas_time)
        self.log(f"reference: {len(self.gates)} gates, TAS A->B {self.ref_end['sr_ms']:.1f} ms "
                 f"({self.ref_end['t']} frames), pickups {self.ref_end['counts']}")
        return r

    # -------------------------------------------------------------------------------------------- tasks
    def continuation(self, node: Node | None, plain: bool = False) -> np.ndarray:
        c = self.cfg
        if node is None:
            g, tau, L = -1, 0.0, 0
        else:
            g, tau, L = node.gate, node.time, node.end
        tg = self.tas_time.get(g, 0.0)
        idx = c.a + int(round(tg + (L - tau)))
        n = int((c.b - c.a - tg) * 1.3) + 300
        cont = self.tkeys[idx:idx + n]
        if len(cont) < n:
            cont = np.r_[cont, np.full(n - len(cont), cont[-1] if len(cont) else 0, np.uint32)]
        u = self.rng.random()
        if plain or u < c.p_plain:
            return cont.copy()
        if u < c.p_plain + c.p_random:
            w = min(c.window, len(cont))
            out = cont.copy()
            out[:w] = random_holds(w, self.rng, bias=int(cont[0]) & 15 or None)
            return out
        return mutate(cont, self.rng, c.window, 1 + self.rng.poisson(c.mean_edits))

    def node_keys(self, node: Node | None) -> np.ndarray:
        if node is None:
            return np.zeros(0, np.uint32)
        return self.rollouts[node.rollout]["keys"][:node.end]

    def task(self, node: Node | None, plain: bool = False) -> dict:
        c = self.cfg
        rid = self.next_rid
        self.next_rid += 1
        start_gate = 0 if node is None else node.gate + 1
        self.rollouts[rid] = {"parent": node, "keys": None}
        return dict(rid=rid, base=self.node_keys(node), cont=self.continuation(node, plain), delta=self.delta, block=c.block,
                    gates=self.gates, start_gate=start_gate, best=dict(self.best), slack=c.slack,
                    max_gap=2.5 * c.gate_every + 200, y_floor=self.y_floor)

    def absorb(self, r: dict) -> int:
        """Branches of a finished rollout into the archive; returns how many gates it crossed."""
        c = self.cfg
        rid = r["rid"]
        meta = self.rollouts[rid]
        parent: Node | None = meta["parent"]
        meta.update(keys=r["keys"], t=r["t"], pos=r["pos"], end=r["end"])
        self.engine_s += r["wall"]
        self.start_s += r["wall_start"]
        self.n_rollouts += 1
        self.ends[r["end"]] = self.ends.get(r["end"], 0) + 1
        ev = [e for e in r["events"] if e[0] > r["base_len"]]
        base_counts = parent.counts if parent is not None else {k: 0 for k in self.ref_end["counts"]}
        n = 0
        made = []
        for g, tc in sorted(r["cross"].items()):
            j = int(np.searchsorted(r["t"], tc))  # first sample at or after the crossing
            j = min(j, len(r["t"]) - 1)
            end = int(r["t"][j])
            if ball_off(ev, end):
                break
            pc = pickup_counts(ev, end)
            counts = {k: base_counts.get(k, 0) + pc.get(k, 0) for k in base_counts}
            need = self.tas_counts.get(g - 1, {})
            if any(counts.get(k, 0) < v for k, v in need.items()):
                d = self.dropped.setdefault(g, [0, None])
                d[0] += 1
                d[1] = min(d[1] if d[1] is not None else 1e18, float(tc))
                break
            nd = Node(rid, end, g, float(tc), r["pos"][j].copy(), r["vel"][j].copy(), counts, type=int(r["type"][j]))
            self.archive.setdefault(g, []).append(nd)
            if tc < self.best.get(g, 1e18):
                self.best[g] = float(tc)
            made.append(nd)
            n += 1
        for nd in made:
            nd.reach = max(made[-1].gate, nd.gate)
        for g, lst in self.archive.items():
            if len(lst) > 400:
                lst.sort(key=lambda x: x.time)
                del lst[300:]
        return n

    # -------------------------------------------------------------------------------------------- main loop
    def run(self):
        c = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        ctx = mp.get_context("spawn")
        labels = ctx.Manager().Queue()
        for lab in COPIES[:c.workers]:
            labels.put(lab)
        t0 = time.perf_counter()
        with ctx.Pool(c.workers, initializer=_init, initargs=(labels, self.eng, str(self.out), self.prefix)) as pool:
            self.reference(pool)
            last = len(self.gates) - 1
            for rnd in range(c.rounds):
                for g in range(-1, last):
                    nodes = [None] if g < 0 else select(self.archive.get(g, []), c.k, c.k_speed,
                                                             k_surv=c.k_surv, ahead=c.ahead)
                    if not nodes:
                        self.log(f"round {rnd} gate {g}: no branch")
                        continue
                    tasks = []
                    for nd in nodes:
                        for i in range(c.m if (nd is None or nd.expanded == 0) else max(c.m // 2, 1)):
                            tasks.append(self.task(nd, plain=(i == 0)))  # one unedited TAS continuation per branch
                        if nd is not None:
                            nd.expanded += 1
                    for r in pool.imap_unordered(_run, tasks):
                        self.absorb(r)
                    bB = self.best.get(last)
                    self.log(f"round {rnd} gate {g + 1}/{last + 1}: {len(tasks)} rollouts, best at gate {g + 1} "
                             f"{self.best.get(g + 1, float('nan')) - self.tas_time[g + 1]:+.1f} f vs TAS, at B "
                             f"{(bB - self.tas_time[last]) if bB is not None else float('nan'):+.1f} f, "
                             f"{self.n_rollouts} rollouts, {self.engine_s:.0f} engine-s, {time.perf_counter() - t0:.0f} s")
            self.wall = time.perf_counter() - t0
            return self.finish(pool)

    def chain(self, node: Node):
        """(times, positions) the search observed along a branch: each rollout's own samples, parents first."""
        parts = []
        nd = node
        hi = None
        while nd is not None:
            m = self.rollouts[nd.rollout]
            t, p = m["t"], m["pos"]
            lo = 0 if m["parent"] is None else m["parent"].end
            sel = (t >= lo) & (t <= (nd.end if hi is None else hi))
            parts.append((t[sel], p[sel]))
            hi = lo
            nd = m["parent"]
        parts.reverse()
        return np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts])

    def finish(self, pool):
        c = self.cfg
        last = len(self.gates) - 1
        cands = sorted(self.archive.get(last, []), key=lambda n: n.time)
        checked, diag, failed = [], [], []
        for nd in cands:
            got = {"pos": nd.pos, "vel": nd.vel, "type": nd.type, "counts": nd.counts}
            chk = c.end.check(self.ref_end, got)
            if len(diag) < 20:
                diag.append({"frames_vs_tas": nd.time - self.tas_time[last], "check": chk, "pos": nd.pos.tolist(),
                             "vel": nd.vel.tolist(), "counts": nd.counts})
            if chk["ok"] and len(checked) < c.verify_top:
                checked.append((nd, chk))
            elif not chk["ok"] and len(failed) < c.verify_failed and (not cands or nd.time < self.tas_time[last] - 0.5):
                failed.append((nd, chk))
        results = []
        for nd, chk in checked + failed:
            keys = self.node_keys(nd)
            v = pool.apply(_run, (dict(rid=-2, base=np.zeros(0, np.uint32), cont=keys, delta=self.delta, block=1,
                                       record_sr=True),))
            res = self.verified(nd, keys, v)
            res["kind"] = "passes_end_check" if chk["ok"] else "fails_end_check"
            results.append(res)
        best_any = cands[0].time - self.tas_time[last] if cands else None
        return {"results": results, "best_any_frames": best_any, "n_at_B": len(cands), "candidates": diag}

    def verified(self, nd: Node, keys, v) -> dict:
        last = len(self.gates) - 1
        t_s, p_s = self.chain(nd)
        vi = {int(t): i for i, t in enumerate(v["t"])}
        exact = all(int(t) in vi and np.array_equal(v["pos"][vi[int(t)]], p) for t, p in zip(t_s, p_s))
        cr = crossings(self.gates, 0, v["t"], v["pos"])
        ok_gate = last in cr
        res = {"keys_runs": to_runs(keys), "n_frames": int(len(keys)), "search_cross": nd.time,
               "deterministic": bool(exact), "n_compared": int(len(t_s)), "crossed_B": ok_gate}
        if ok_gate:
            i = cr[last][1]
            ev = [e for e in v["events"] if e[0] > 0]
            got = {"pos": v["pos"][i], "vel": v["vel"][i], "type": int(v["type"][i]),
                   "counts": pickup_counts(ev, v["t"][i]), "ball_off": ball_off(ev, v["t"][i])}
            res.update(frames=int(v["t"][i]), sr_ms=float(v["sr"][i] - v["sr"][0]), cross=float(cr[last][0]),
                       delta_frames=int(v["t"][i]) - self.ref_end["t"],
                       delta_ms=float(v["sr"][i] - v["sr"][0]) - self.ref_end["sr_ms"],
                       check=self.cfg.end.check(self.ref_end, got), pos=v["pos"][: i + 1].tolist())
        return res


def run_search(cfg: SearchConfig, dataset: Path, out: Path, eng: EngineSpec | None = None) -> dict:
    """Search A -> B of a tas-v1 dataset (run/framelog.bin = the record's executed frames); writes out/result.json."""
    frames = tas.load_raw(Path(dataset) / "run" / "framelog.bin")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "search.log", "a", encoding="utf-8")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    s = Search(cfg, frames, out, eng, log)
    fin = s.run()
    eng = s.eng
    res = {
        "a": cfg.a, "b": cfg.b, "dataset": Path(dataset).name,
        "framelog_sha256": sha256(Path(dataset) / "run" / "framelog.bin"),
        "engine": {"binary": Path(eng.binary).parent.name + "/" + Path(eng.binary).name, "ball_order": eng.ball_order,
                   "binary_sha256": sha256(eng.binary)},
        "config": {k: v for k, v in asdict(cfg).items() if k != "end"} | {"end": asdict(cfg.end)},
        "gates": [{"index": g.index, "frame": g.frame, "pos": g.pos.tolist(), "normal": g.normal.tolist(),
                   "tas_time": s.tas_time[g.index], "best_time": s.best.get(g.index),
                   "dropped_pickup": s.dropped.get(g.index, [0, None])} for g in s.gates],
        "tas": {**s.ref_end, "pos_path": s.ref["pos"][: s.ref_end["t"] + 1].tolist()},
        "stats": {"rollouts": s.n_rollouts, "engine_s": s.engine_s, "restore_s": s.start_s, "wall_s": s.wall,
                  "workers": cfg.workers, "ends": s.ends, "n_at_B": fin["n_at_B"],
                  "best_any_frames_vs_tas": fin["best_any_frames"]},
        "results": fin["results"],
        "candidates_at_B": fin["candidates"],  # earliest branches at B as the search saw them (4-frame samples), before verification
    }
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for r in fin["results"]:
        log(f"result ({r['kind']}): {r.get('delta_ms', float('nan')):+.1f} ms vs TAS, deterministic {r['deterministic']}, "
            f"check {r.get('check', {}).get('fails')}")
    logf.close()
    return res
