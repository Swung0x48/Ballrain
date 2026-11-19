"""Returnable engine states, independent of the Pareto/family output list."""

import numpy as np


def cell(p, cp, ball, start=""):
    """Keep physically distinct start tapes in disjoint namespaces."""
    return start + ":" + ",".join(map(str, (cp, ball, *np.floor(p / 4).astype(int))))


def exits(field, p, sector, ball, start):
    """Adjacent cells with a continuous short ground march (not a feasibility proof)."""
    result = {}
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        m = field.march(p, [dx, dz], lmax=4, ball_type=ball)
        if min(m["edge_s"], m["obst_s"]) >= 4:
            q = p + [4 * dx, m["y_last"] - p[1], 4 * dz]
            result[cell(q, sector, ball, start)] = q.tolist()
    return result


def store(cells, key, parent, tail, witness, remaining):
    if key in cells:
        return False
    cells[key] = dict(
        parent=parent,
        tail=tail,
        witness=witness,
        distance=remaining,
        visits=0,
        failures=0,
    )
    return True


def tape(cells, key):
    chunks = []
    while key is not None:
        r = cells[key]
        chunks.append(r["tail"])
        key = r["parent"]
    return [step for chunk in reversed(chunks) for step in chunk]


def choose(cells, rng, start=""):
    if not cells or rng.random() < 0.06:
        return None
    pool = [
        k
        for k, r in cells.items()
        if not r.get("diverged") and k.startswith(start + ":")
    ]
    if not pool:
        return None
    guided = rng.random() < 0.78

    def rank(k):
        r = cells[k]
        missing = sum(key not in cells for key in r.get("exits", ()))
        return (
            (r["distance"] if guided else 0)
            + 9 * r["visits"]
            + 12 * r["failures"]
            - 2 * missing
            + rng.uniform(0, 12)
        )

    key = min(rng.sample(pool, min(64, len(pool))), key=rank)
    cells[key]["visits"] += 1
    return key


def failed_positions(cells, start, sector):
    prefix = f"{start}:{sector},"
    return [
        (np.array(list(map(int, key.split(",")[-3:]))) + 0.5) * 4
        for key, row in cells.items()
        if row["failures"] and key.startswith(prefix)
    ]


def restore(trace, cells, key, expired):
    """None means out of budget, not evidence that this cell's tape diverged."""
    for k, n in tape(cells, key):
        if expired():
            return None
        if not trace.step(k, n) or trace.keys[-1][1] != n:
            return False
    return trace.witness.hexdigest() == cells[key]["witness"]
