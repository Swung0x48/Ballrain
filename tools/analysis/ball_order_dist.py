"""Retail ball-ball contact order per process (launches the reference game).

    uv run python tools/analysis/ball_order_dist.py --dll <x87-soft physics_RT.dll> --n 35 [--record SR01_1_14_457_264]
    uv run python tools/analysis/ball_order_dist.py --analyse-only

IVP orders the two balls of a ball-ball contact pair by heap address, and the pooled scene balls get a different
address order in every process. To measure the distribution: the x87-soft physics_RT.dll (bit-identical to the
2002 DLL on the TAS records; with BMMO_HX1_BB=<file> it logs every ball-ball pair with both addresses) is copied
into runtime/ref/BuildingBlocks for the duration of the measurement, the record is replayed by the real
TASSupport once per fresh process, and the 2002 DLL is restored afterwards (sha256 checked). Per process the
balls seen in any contact are ranked by address. Logs in runtime/analysis/ball_order/bb_<i>.txt (plus any
--extra logs in the same format); summary in runtime/analysis/ball_order/summary.json.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ballrain import instances  # noqa: E402

DLL_2002 = "e72e4afcfa5c33a7d3d2"
OUT = instances.runtime_dir() / "analysis" / "ball_order"
LINE = re.compile(r"BB t=\S+ (\S+) (\S+) lt=\d p0=([0-9A-Fa-f]+) p1=([0-9A-Fa-f]+)")


def order_of(path) -> list[str]:
    addr = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = LINE.match(line)
        if m:
            addr[m.group(1)] = int(m.group(3), 16)
            addr[m.group(2)] = int(m.group(4), 16)
    return [n for n, _ in sorted(addr.items(), key=lambda x: x[1])]


def measure(dll: Path, n: int, record: str) -> list[Path]:
    instances.LAUNCH_RECIPE = instances.launch_direct
    ref = instances.get_copy(instances.REFERENCE_COPY)
    target = ref.root / "BuildingBlocks" / "physics_RT.dll"
    backup = ref.root / "BuildingBlocks" / "physics_RT.dll.retail"
    if not instances.sha256_file(backup).startswith(DLL_2002):
        raise SystemExit(f"{backup} is not the 2002 DLL")
    if instances.running_in_copy(ref):
        raise SystemExit("the reference copy is running")
    tas = ref.records_dir / f"{record}.tas"
    OUT.mkdir(parents=True, exist_ok=True)
    logs = []
    verify = instances.verify_physics
    instances.verify_physics = lambda copy, expected=None: "x87-soft (measurement)"  # preflight's 2002 guard
    shutil.copyfile(dll, target)
    try:
        for i in range(n):
            log = OUT / f"bb_{i}.txt"
            log.unlink(missing_ok=True)
            os.environ["BMMO_HX1_BB"] = str(log)
            instances.run_reference(tas, 1, OUT / f"trace_{i}.txt", timeout=400)
            logs.append(log)
            print(i, order_of(log) if log.is_file() else "no log", flush=True)
    finally:
        os.environ.pop("BMMO_HX1_BB", None)
        instances.verify_physics = verify
        shutil.copyfile(backup, target)
        h = instances.sha256_file(target)
        print("restored", target, h[:20], "OK" if h.startswith(DLL_2002) else "MISMATCH", flush=True)
    return logs


def analyse(logs) -> dict:
    orders = [o for o in (order_of(f) for f in logs) if o]
    names = sorted({x for o in orders for x in o})
    pos = {nm: collections.Counter(o.index(nm) for o in orders if nm in o) for nm in names}
    pairs = {}
    for a, b in combinations(names, 2):
        both = [o for o in orders if a in o and b in o]
        if both:
            k = sum(o.index(a) < o.index(b) for o in both)
            pairs[f"{a}<{b}"] = f"{k}/{len(both)}"
    pooled = [nm for nm in names if nm.startswith("P_Ball_Stone_MF0")]
    perm = collections.Counter(tuple(x for x in o if x in pooled) for o in orders)
    return {"samples": len(orders), "names": names, "rank_counts": {k: dict(v) for k, v in pos.items()},
            "pair_order": pairs, "stone_mf00x_permutations": {" < ".join(k): v for k, v in perm.most_common()},
            "orders": [" < ".join(o) for o in orders]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dll", default=None)
    p.add_argument("--n", type=int, default=35)
    p.add_argument("--record", default="SR01_1_14_457_264")
    p.add_argument("--extra", nargs="*", default=[], help="earlier logs in the same format (glob patterns)")
    p.add_argument("--analyse-only", action="store_true")
    a = p.parse_args(argv)
    logs = [] if a.analyse_only else measure(Path(a.dll), a.n, a.record)
    if a.analyse_only:
        logs = sorted(OUT.glob("bb_*.txt"))
    for g in a.extra:
        logs += sorted(Path(x) for x in glob.glob(g))
    res = analyse(logs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "orders"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
