"""CLI contract and pure candidate selection."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

SPARES = tuple(f"hlw-{i}" for i in (91, 92, 93, 98, 99))
MATERIALS = ("wood", "paper", "stone")
UNVERIFIED = "unverified, search-build only"


def pose(text):
    try:
        xyz = tuple(map(float, text.split(",")))
    except ValueError:
        xyz = ()
    if len(xyz) not in (3, 4) or not all(map(math.isfinite, xyz)):
        raise argparse.ArgumentTypeError("expected finite x,y,z[,camera-yaw-degrees]")
    return xyz


def parser():
    p = argparse.ArgumentParser(prog="python -m ballrain.routescout")
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    src = plan.add_mutually_exclusive_group(required=True)
    src.add_argument("--level", type=int, choices=range(1, 14))
    src.add_argument("--map", type=Path)
    src.add_argument("--resume", type=Path)
    plan.add_argument("--out", type=Path)
    start = plan.add_mutually_exclusive_group()
    start.add_argument("--spawn", type=pose)
    start.add_argument("--from-checkpoint", type=int)
    plan.add_argument("--material", choices=MATERIALS)
    plan.add_argument("--to-checkpoint", type=int)
    plan.add_argument("--budget-s", type=float, default=120)
    plan.add_argument("--candidates", type=int, default=6)
    plan.add_argument(
        "--cold",
        action="store_true",
        help="ignore packaged scout-generated key proposals",
    )
    plan.add_argument("--verify", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("out", type=Path)
    runtime = Path("runtime/hl-win")
    for cmd in (plan, verify):
        cmd.add_argument(
            "--root",
            type=Path,
            default=Path(os.getenv("BALLRAIN_HL_ROOT", runtime / "game")),
        )
        cmd.add_argument("--second-instance", choices=SPARES, default=SPARES[-1])
        cmd.add_argument(
            "--second-build",
            type=Path,
            default=runtime / "bin-x86p2002d/BallanceMMOSimTool.exe",
        )
    plan.add_argument("--instance", choices=SPARES, default=SPARES[0])
    plan.add_argument(
        "--build",
        type=Path,
        default=Path(
            os.getenv("BALLRAIN_HL_BIN", runtime / "bin-x86soft/BallanceMMOSimTool.exe")
        ),
    )
    return p


def digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf8")
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            # Windows readers/AV can briefly hold a non-delete-sharing handle.
            if attempt == 5:
                raise
            time.sleep(0.02 * 2**attempt)


def difficulty(keys, balls, rail_fraction):
    changes = sum(a[0] != b[0] for a, b in zip(keys, keys[1:]))
    components = dict(
        steering=round(50 * min(1, changes * 32 / max(1, sum(n for _, n in keys))), 2),
        transforms=min(25, 10 * (len(balls) - 1)),
        rail=round(25 * rail_fraction, 2),
    )
    return dict(score=round(sum(components.values()), 2), components=components)


def order(r):
    return (
        -len(r["checkpoints"]),
        not r["complete"],
        0 if r["complete"] else r["distance_to_goal"],
        r["sr_ms"],
        r["id"],
    )


def select(rows, count):
    chosen = []
    seen = set()
    ranked = sorted(rows, key=order)
    finished = [r for r in ranked if r["complete"]]
    if finished:
        # Reserve a slot for the best payoff, not just the fastest families.
        ranked = [
            finished[0],
            min(finished, key=lambda r: (-r["hs"]["score_delta"], order(r))),
            *ranked,
        ]
    for r in ranked:
        if r["family"] not in seen:
            chosen.append(r)
            seen.add(r["family"])
            if len(chosen) == count:
                break
    for r in chosen:
        # Only compare like endpoints; unfinished fronts are progress bins, not records.
        peers = [
            q
            for q in chosen
            if [h[0] for h in q["checkpoints"]] == [h[0] for h in r["checkpoints"]]
            and q["complete"] == r["complete"]
            and q.get("start_keys") == r.get("start_keys")
            and (
                r["complete"]
                or int(q["distance_to_goal"] // 5) == int(r["distance_to_goal"] // 5)
            )
        ]

        def front(cost):
            a = (cost(r), r["difficulty"]["score"])
            return not any(
                (b := (cost(q), q["difficulty"]["score"])) != a
                and b[0] <= a[0]
                and b[1] <= a[1]
                for q in peers
            )

        r["pareto"] = dict(
            sr=front(lambda q: q["sr_ms"]), hs=front(lambda q: -q["hs"]["score_delta"])
        )
    return chosen
