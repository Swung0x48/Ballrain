import json
import math
import random
import time
from pathlib import Path
import numpy as np
from . import engine as e, frontier
from .mechanisms import Rules
from .contract import atomic, parser, select, order


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(path, data):
    atomic(path, json.dumps(data, allow_nan=False, separators=(",", ":")))


def publish(out, archive, n, triangles):
    rows = select(archive["rows"], n)
    if (out / "candidates.json").is_file():
        old = read(out / "candidates.json")
        checked = (
            {(r["id"], r["witness"]): r["verification"] for r in old["candidates"]}
            if old["config"] == archive["config"]
            else {}
        )
        for r in rows:
            r["verification"] = checked.get((r["id"], r["witness"]), r["verification"])
    write(
        out / "candidates.json",
        dict(schema=2, config=archive["config"], candidates=rows),
    )
    summary(out, rows)
    if triangles is not None:
        e.draw(out, triangles, rows)


def summary(out, rows):
    atomic(
        out / "summary.md",
        "# Route scout\n\n| Candidate | Target reached | SR ms | HS delta | Difficulty | Verification |\n|---|---|---:|---:|---:|---|\n"
        + "".join(
            f"| {r['id']} | {r['complete']} | {r['sr_ms']:.1f} | {r['hs']['score_delta']} | {r['difficulty']['score']} | {r['verification']['status']} |\n"
            for r in rows
        ),
    )


def admit(rows, r):
    # Keep two tradeoffs per family and end sector: closest to goal and highest HS.
    peers = [
        q
        for q in rows
        if q["family"] == r["family"]
        and q["segments"] == r["segments"]
        and q["complete"] == r["complete"]
    ]
    if any(q["id"] == r["id"] for q in peers):
        return
    group = peers + [r]
    best = {
        min(group, key=order)["id"],
        min(group, key=lambda q: (-q["hs"]["score_delta"], order(q)))["id"],
    }
    rows[:] = [q for q in rows if q not in peers or q["id"] in best]
    if r["id"] in best:
        rows.append(r)


def search(a):
    if a.map:
        raise ValueError(
            "custom maps need an agent-mode file loader, ordered checkpoint/reset and floor exports, and a placement-matched reach-mechanism asset; retail rules cannot be substituted"
        )
    if not math.isfinite(a.budget_s) or a.budget_s <= 0 or a.candidates < 1:
        raise ValueError("budget and candidate count must be positive")
    out = a.resume or a.out
    if not out:
        raise ValueError("new search needs --out")
    if a.resume and (
        a.cold
        or a.out
        or any(
            v is not None
            for v in (a.spawn, a.material, a.from_checkpoint, a.to_checkpoint)
        )
    ):
        raise ValueError("resume uses the archived output and start/target settings")
    build = e.build_id(a.build)
    if a.verify and (
        a.instance == a.second_instance or build == e.build_id(a.second_build)
    ):
        raise ValueError(
            "verification needs a different instance and executable digest"
        )
    if a.resume:
        arc = read(out / "archive.json")
        cfg = arc["config"]
        if arc.get("schema") != 2:
            raise ValueError("archive version mismatch; use a fresh output directory")
        if a.instance != cfg["instance"]:
            raise ValueError(
                f"resume requires the archived search instance ({cfg['instance']})"
            )
        if build != cfg["build"]:
            raise ValueError("resume needs the original search build")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("use an empty output directory, or --resume")
        out.mkdir(parents=True, exist_ok=True)
        cp = a.from_checkpoint or 0
        target = a.to_checkpoint if a.to_checkpoint is not None else cp + 1
        if cp < 0 or target <= cp:
            raise ValueError("target must follow start checkpoint")
        cfg = dict(
            level=a.level,
            spawn=a.spawn,
            material=a.material,
            from_checkpoint=cp,
            target=target,
            build=build,
            instance=a.instance,
            cold=a.cold,
        )
        arc = dict(schema=2, config=cfg, episode=0, rows=[], prefix=None)
    cells = arc.setdefault("cells", {})
    stats = arc.setdefault("exploration", {"restores": 0, "diverged": 0, "deaths": 0})
    paths = triangles = None
    start = time.monotonic()
    deadline = start + a.budget_s
    report = start
    seeds = (
        read(Path(__file__).with_name("scout_seeds.json"))
        if not cfg.get("cold", True)
        else {}
    )
    try:
        while time.monotonic() < deadline:
            rng = random.Random(arc["episode"])
            arc["episode"] += 1
            with e.launch(cfg["level"], a.root, a.build, a.instance, out) as s:
                tr = s.tracker
                rules = Rules(tr)
                pin = tr.level.static_key()
                if (
                    cfg.setdefault("map_key", pin) != pin
                    or cfg.setdefault("rules_pin", rules.pin) != rules.pin
                ):
                    raise ValueError("map or mechanism assets changed")
                if paths is None:
                    paths, triangles, goals, grid = e.guide(s, cfg["target"])
                field = grid.field
                if cfg["target"] > len(paths):
                    raise ValueError("target checkpoint outside map")
                proposals = sorted(
                    (x for x in seeds.get(pin, []) if x["sector"] >= cfg["target"]),
                    key=lambda x: x["sector"],
                )
                idx = arc["episode"] - 1
                tape = proposals[idx]["keys"] if idx < len(proposals) else []
                cursor = left = 0

                def proposed():
                    nonlocal cursor, left
                    if cursor == len(tape):
                        return None
                    k, n = tape[cursor]
                    left = left or n
                    return k, min(left, 12)

                def advance(n):
                    nonlocal cursor, left
                    left -= n
                    if left == 0:
                        cursor += 1

                prefix = []
                look = rng.uniform(3, 8)
                speed = rng.uniform(7, 14)
                block = rng.choice((8, 12, 20, 28))
                if arc["prefix"] is not None and not tape:
                    for k, n in arc["prefix"]:
                        if time.monotonic() >= deadline:
                            break
                        if e.act(s, k, n) != n:
                            raise ValueError("archived start prefix diverged")
                        prefix.append([k, n])
                else:
                    for _ in range(20000 if tape else 700):
                        if e.ready(s, cfg) or time.monotonic() >= deadline:
                            break
                        if tr.checkpoints >= len(paths):
                            break
                        path = paths[tr.checkpoints]
                        if cfg["spawn"]:
                            path = e.ribbon([e.position(s), cfg["spawn"][:3]])
                        elif (
                            cfg["material"] and tr.checkpoints == cfg["from_checkpoint"]
                        ):
                            ts = [
                                p["matrix"][3][:3]
                                for p in rules.rows
                                if "transform_" + cfg["material"]
                                in rules.data["rules"][p["type"]]["rules"]
                            ]
                            if not ts:
                                break
                            path = e.ribbon(
                                [
                                    e.position(s),
                                    min(
                                        ts,
                                        key=lambda p: np.linalg.norm(e.position(s) - p),
                                    ),
                                ]
                            )
                        proposal = proposed()
                        k, n = proposal or (e.steer(s, path, look, speed, rng), block)
                        n = e.act(s, k, n)
                        if proposal:
                            advance(n)
                        if not n:
                            break
                        prefix.append([k, n])
                if e.ready(s, cfg):
                    arc["prefix"] = prefix
                    trace = e.Trace(s, rules)
                    if tape:
                        trace.proposal = "scout archive"
                    start_id = e.digest(prefix)[:16]
                    parent = frontier.choose(cells, rng, start_id) if not tape else None
                    if parent is not None:
                        restored = frontier.restore(
                            trace, cells, parent, lambda: time.monotonic() >= deadline
                        )
                        if not restored:
                            if restored is False:
                                stats["diverged"] += 1
                                cells[parent]["diverged"] = True
                            continue
                        stats["restores"] += 1
                    mark = len(trace.keys)
                    stale = 0
                    escape = None
                    until = 0
                    detours = e.Detours(rules, rng)
                    if detours.commit:
                        look = rng.uniform(7, 14)
                        speed = rng.uniform(15, 28)
                    explore = rng.choice((0.0, 0.5, 3.0, 12.0))
                    best = float("inf")
                    sector = tr.checkpoints
                    alternate = None
                    if parent and goals[sector] is not None and rng.random() < 0.5:
                        hazards = frontier.failed_positions(cells, start_id, sector)
                        alternate = grid.path(
                            e.position(s),
                            goals[sector],
                            rng.sample(hazards, min(16, len(hazards))),
                        )
                    for i in range(20000 if tape else 900):
                        if time.monotonic() >= deadline:
                            break
                        if time.monotonic() >= report:
                            print(
                                f"L{cfg['level']} episode {arc['episode']} cells {len(cells)} candidates {len(arc['rows'])} elapsed {time.monotonic() - start:.1f}s",
                                flush=True,
                            )
                            write(out / "archive.json", arc)
                            report = time.monotonic() + 10
                        path = paths[tr.checkpoints]
                        if alternate is not None:
                            path = alternate
                        # Try a reachable unvisited neighbour when the guide is absent
                        # or the same patch keeps absorbing the controller's proposals.
                        if (path is None or stale >= 10) and parent and i >= until:
                            exits = cells[parent].get("exits", {})
                            options = [q for k, q in exits.items() if k not in cells]
                            if options:
                                escape = rng.choice(options)
                                until = i + 7
                        drive = e.ribbon([e.position(s), escape]) if i < until else path
                        proposal = proposed()
                        if proposal:
                            k, n = proposal
                        else:
                            k, n = detours.action(
                                s, drive, look, speed, rng, block, field, explore
                            )
                        if not trace.step(k, n):
                            if s.obs is not None and s.obs.has_event(
                                e.Ev.BallOff, e.Ev.GameOver
                            ):
                                stats["deaths"] += 1
                            if parent:
                                cells[parent]["failures"] += 1
                            break
                        if proposal:
                            advance(trace.keys[-1][1])
                        if not s.obs.ball["found"]:
                            continue
                        next_goal = goals[sector]
                        goal = goals[cfg["target"] - 1]
                        p = e.position(s)
                        crossed = sector != tr.checkpoints
                        sector = tr.checkpoints
                        station, lateral = (
                            e.project(path, p) if path is not None else (0.0, 0.0)
                        )
                        dist = (
                            path[3][-1] - station + lateral if path is not None else 0.0
                        )
                        if path is not None and next_goal is not None:
                            dist += np.linalg.norm(path[0][-1] - next_goal)
                        supported = (
                            tr.flags.ball_nav
                            and bool(len(s.obs.fric["partner"]))
                            and field.support(p)[2] in (1, 2)
                        )
                        new = False
                        if supported and sector < cfg["target"]:
                            key = frontier.cell(p, sector, tr.ball_type(), start_id)
                            new = frontier.store(
                                cells,
                                key,
                                parent,
                                trace.keys[mark:],
                                trace.witness.hexdigest(),
                                float(dist),
                            )
                            if new:
                                cells[key]["exits"] = frontier.exits(
                                    field, p, sector, tr.ball_type(), start_id
                                )
                                parent, mark = key, len(trace.keys)
                        if crossed:
                            alternate = None
                        stale = 0 if new else stale + 1
                        if crossed or (supported and (new or dist < best - 0.75)):
                            candidate = trace.candidate(cfg, prefix, goal)
                            candidate["guide_remaining"] = (
                                round(float(dist), 3) if path is not None else None
                            )
                            admit(arc["rows"], candidate)
                            best = min(best, dist)
                        if not tape and stale > 30:
                            if parent:
                                cells[parent]["failures"] += 1
                            break
                        if sector >= cfg["target"] or s.ended:
                            break
            write(out / "archive.json", arc)
            if time.monotonic() >= report:
                print(
                    f"episode {arc['episode']} cells {len(cells)} candidates {len(arc['rows'])}",
                    flush=True,
                )
                report = time.monotonic() + 10
    finally:
        write(out / "archive.json", arc)
        publish(out, arc, a.candidates, triangles)
    print(
        f"search {time.monotonic() - start:.2f}s; {len(read(out / 'candidates.json')['candidates'])} candidates",
        flush=True,
    )
    if arc["prefix"] is None:
        raise ValueError("requested physical start was not reached within budget")
    if a.verify:
        verify(out, a)


def verify(out, a):
    doc = read(out / "candidates.json")
    cfg = doc["config"]
    build = e.build_id(a.second_build)
    if doc.get("schema") != 2:
        raise ValueError("unsupported candidate witness version")
    if build == cfg["build"] or a.second_instance == cfg["instance"]:
        raise ValueError("verification requires independent build and instance")
    for r in doc["candidates"]:
        mismatch = []
        with e.launch(
            cfg["level"], a.root, a.second_build, a.second_instance, out
        ) as s:
            if cfg["map_key"] != s.tracker.level.static_key():
                mismatch.append("map/rules")
            else:
                try:
                    rules = Rules(s.tracker)
                except ValueError:
                    mismatch.append("map/rules")
                else:
                    if rules.pin != cfg["rules_pin"]:
                        mismatch.append("map/rules")
            if not mismatch and (
                any(e.act(s, k, n) != n for k, n in r["start_keys"])
                or not e.ready(s, cfg)
            ):
                mismatch.append("start")
            if not mismatch:
                t = e.Trace(s, rules)
                if any(not t.step(k, n) or t.keys[-1][1] != n for k, n in r["keys"]):
                    mismatch.append("key stream interrupted")
                for field, x in [
                    ("witness", t.witness.hexdigest()),
                    ("ball_sequence", t.balls),
                    ("checkpoints", t.cp),
                    ("transformer_events", t.transform),
                    ("mechanisms", t.hit),
                    ("sr_ms", s.tracker.sr_ms - t.sr),
                    (
                        "hs",
                        dict(
                            pickups=t.pick,
                            esc_banks=t.banks,
                            score_delta=s.tracker.hs_partial() - t.hs,
                        ),
                    ),
                ]:
                    if r[field] != x:
                        mismatch.append(field)
            if not mismatch and r["complete"]:
                expected = list(range(cfg["from_checkpoint"] + 1, cfg["target"] + 1))
                if (
                    [cp for cp, _ in t.cp] != expected
                    or not t.cp
                    or t.cp[-1][1] != s.frames_done - t.frame
                ):
                    mismatch.append("missing ordered terminal PostCheckpoint")
        status = (
            "mismatch"
            if mismatch
            else (
                "certified"
                if r["complete"]
                else "matching partial replay; not certified"
            )
        )
        r["verification"] = dict(status=status, build=build, mismatches=mismatch)
        write(out / "candidates.json", doc)
        summary(out, doc["candidates"])


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        if a.command == "verify":
            verify(a.out, a)
        else:
            search(a)
    except (ValueError, OSError) as ex:
        p.exit(2, f"route-scout: {ex}\n")


if __name__ == "__main__":
    main()
