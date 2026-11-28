"""Single-edit scan: every single change of the TAS keys between A and B, run to B with the rest of the TAS keys
unchanged (docs/segsearch.md). Unlike the beam search nothing is sampled: the edits are enumerated.

Edits, on the TAS key runs (maximal stretches of one key) that start before B:
  shift   the boundary between run i and run i+1 by d frames, d = -max_shift..max_shift (both runs keep >= 1 frame)
  rotate  run i's direction by -90, -45, +45, +90, 180 degrees (a no-arrow run: each of the 8 directions instead)
  release run i's arrows (an arrow run only)
Every edit keeps the frame count, so the keys after it stay on their TAS frames (open loop). A branch is the TAS up to the
edit's first changed frame (restored by prefix replay) and the edited keys after it, run without the "behind" cut until
B, a ball-off / fall, 2.5 gate spacings without a gate, or TAS-at-B + `extra` frames."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import tas
from .core import RING, rotate, to_runs
from .engine import EngineSpec, pickup_counts
from .search import Search, SearchConfig, _run, logger, result_dict


def single_edits(keys, max_shift: int = 24) -> list[tuple[str, int, np.ndarray]]:
    """[(description, first changed frame, replacement keys from that frame)] for every single edit of `keys`."""
    keys = np.asarray(keys, np.uint32)
    runs = to_runs(keys)
    starts = np.cumsum([0] + [n for _, n in runs])
    out = []
    for i in range(len(runs) - 1):
        b = int(starts[i + 1])
        lo, hi = int(starts[i]), int(starts[i + 2])
        for d in range(-max_shift, max_shift + 1):
            if d == 0 or not lo < b + d < hi:
                continue
            if d > 0:
                seg = np.full(d, runs[i][0], np.uint32)
                out.append((f"shift r{i}/r{i + 1} {d:+d}", b, seg))
            else:
                seg = np.full(-d, runs[i + 1][0], np.uint32)
                out.append((f"shift r{i}/r{i + 1} {d:+d}", b + d, seg))
    for i, (k, n) in enumerate(runs):
        s = int(starts[i])
        if k & 15 in RING:
            for steps, name in ((-2, "-90"), (-1, "-45"), (1, "+45"), (2, "+90"), (4, "180")):
                out.append((f"rotate r{i} {name}", s, np.full(n, rotate(k, steps), np.uint32)))
            out.append((f"release r{i}", s, np.full(n, k & ~15, np.uint32)))
        else:
            for d in RING:
                out.append((f"press r{i} {tas.keys_to_str(d)}", s, np.full(n, (k & ~15) | d, np.uint32)))
    return out


def apply_edit(keys, start: int, seg) -> np.ndarray:
    out = np.asarray(keys, np.uint32).copy()
    out[start:start + len(seg)] = seg
    return out


def run_scan(cfg: SearchConfig, dataset: Path, out: Path, eng: EngineSpec | None = None, max_shift: int = 24,
             extra: int = 264) -> dict:
    frames = tas.load_raw(Path(dataset) / "run" / "framelog.bin")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log, logf = logger(out / "search.log")
    s = Search(cfg, frames, out, eng, log)
    t0 = time.perf_counter()
    with s.pool() as pool:
        s.reference(pool)
        last = len(s.gates) - 1
        tB = s.ref_end["t"]
        tk = s.tkeys[cfg.a:cfg.a + tB + extra]
        ref_ev = [e for e in s.ref["events"] if e[0] > 0]
        edits = single_edits(tk[:tB], max_shift)
        log(f"scan: {len(to_runs(tk[:tB]))} TAS key runs before B, {len(edits)} single edits")
        tasks, desc = [], {}
        for name, start, seg in edits:
            rid = s.next_rid
            s.next_rid += 1
            ed = apply_edit(tk, start, seg)
            s.rollouts[rid] = {"parent": None, "keys": None, "kind": name, "base_counts": pickup_counts(ref_ev, start)}
            desc[rid] = (name, start)
            tasks.append(dict(rid=rid, base=tk[:start], cont=ed[start:], delta=s.delta, block=cfg.block, gates=s.gates,
                              start_gate=sum(1 for g in s.tas_time.values() if g < start), best=None,
                              max_gap=2.5 * cfg.gate_every + 200, y_floor=s.y_floor))
        rows = []
        for n, r in enumerate(pool.imap_unordered(_run, tasks), 1):
            s.absorb(r)
            m = s.rollouts[r["rid"]]
            name, start = desc[r["rid"]]
            row = {"edit": name, "frame": cfg.a + start, "end": r["end"], "last_gate": max(r["cross"], default=-1),
                   "dropped_pickup_gate": m["dropped"]}
            if last in r["cross"] and m["dropped"] < 0:
                row["B_vs_tas_frames"] = r["cross"][last] - s.tas_time[last]
                j = min(int(np.searchsorted(r["t"], r["cross"][last])), len(r["t"]) - 1)
                ev = [e for e in r["events"] if e[0] > r["base_len"]]
                pc = pickup_counts(ev, r["t"][j])
                counts = {k: m["base_counts"].get(k, 0) + pc.get(k, 0) for k in m["base_counts"]}
                row["end_check"] = cfg.end.check(s.ref_end, {"pos": r["pos"][j], "vel": r["vel"][j],
                                                             "type": int(r["type"][j]), "counts": counts})["fails"]
            rows.append(row)
            if n % 200 == 0 or n == len(tasks):
                reached = [x for x in rows if "B_vs_tas_frames" in x]
                best = min((x["B_vs_tas_frames"] for x in reached if not x["end_check"]), default=None)
                log(f"{n}/{len(tasks)} edits, {len(reached)} reached B with the pickups, best valid "
                    f"{'%+.1f f' % best if best is not None else '-'} vs TAS, {s.engine_s:.0f} engine-s, "
                    f"{time.perf_counter() - t0:.0f} s")
        s.wall = time.perf_counter() - t0
        fin = s.finish(pool)
    s.save_rollouts(out / "rollouts.npz")
    res = result_dict(s, cfg, dataset, fin)
    res["scan"] = {"max_shift": max_shift, "extra": extra, "n_edits": len(edits),
                   "rows": sorted(rows, key=lambda x: (x["frame"], x["edit"]))}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for r in fin["results"]:
        log(f"result ({r['kind']}): {r.get('delta_ms', float('nan')):+.1f} ms vs TAS, deterministic {r['deterministic']}, "
            f"check {r.get('check', {}).get('fails')}")
    logf.close()
    return res
