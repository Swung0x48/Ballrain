"""Version-two manifest: ordered single-line bundles and declarative phase gates."""

import json
import re
import numpy as np
from .path import RouteError


def span(value, shape):
    try:
        a = np.asarray(value)
        if a.dtype.kind not in "iuf":
            return False
        return a.shape == shape and np.isfinite(a).all() and (a[0] <= a[1]).all()
    except (ValueError, TypeError):
        return False


def manifest(raw):
    try:
        return _manifest(json.loads(raw))
    except (TypeError, KeyError, AttributeError, ValueError) as exc:
        raise RouteError(f"invalid branch manifest: {exc}") from exc


def _manifest(doc):
    entries = doc.get("branches")
    if doc.get("version") != 2 or not isinstance(entries, list) or not entries:
        raise RouteError("expected version 2 branch manifest")
    seen = set()
    for item in entries:
        name = item["id"]
        if (
            not re.fullmatch("[a-z][a-z0-9-]*", name)
            or name in seen
            or re.fullmatch(r"con|prn|aux|nul|(?:com|lpt)[1-9]", name)
        ):
            raise RouteError("invalid/duplicate branch id")
        seen.add(name)
        if item.get("legacy") is True and set(item) == {"id", "legacy"}:
            continue
        phases = item["phases"]
        if (
            not isinstance(phases, list)
            or not phases
            or set(item) != {"id", "phases", "merge"}
        ):
            raise RouteError("invalid branch phases")
        prev = None
        for idx, phase in enumerate(phases):
            if "from" in phase:
                src = phase["from"]
                if (
                    not isinstance(src, list)
                    or not src
                    or any(type(x) is not int or not 0 <= x < idx for x in src)
                    or idx - 1 not in src
                    or len(set(src)) != len(src)
                ):
                    raise RouteError("invalid phase predecessor list")
            if set(phase) - {"when", "local_s", "from"} or "ball" not in phase["when"]:
                raise RouteError("invalid phase")
            bounds = phase["local_s"]
            if (
                not span(bounds, (2,))
                or bounds[0] >= bounds[1]
                or prev not in (None, bounds[0])
            ):
                raise RouteError("phase spans must partition the diversion")
            prev = bounds[1]
        for gate in [p["when"] for p in phases] + [item["merge"]]:
            if (
                set(gate) - {"sector", "ball", "box", "s"}
                or not set(gate) & {"sector", "box", "s"}
                or type(gate.get("sector", 1)) is not int
                or type(gate.get("ball", 0)) is not int
                or gate.get("ball", 0) not in (0, 1, 2)
                or ("box" in gate and not span(gate["box"], (2, 3)))
                or ("s" in gate and not span(gate["s"], (2,)))
            ):
                raise RouteError("invalid phase gate")
    return entries


def matches(g, pos, sector, ball, arc):
    return (
        g.get("sector", sector) == sector
        and g.get("ball", ball) == ball
        and ("s" not in g or g["s"][0] <= arc <= g["s"][1])
        and (
            "box" not in g
            or (np.asarray(g["box"][0]) <= pos).all()
            and (pos <= np.asarray(g["box"][1])).all()
        )
    )


class MultiFollower:
    def __init__(self, main, members):
        self.main, self.members = main, members
        self.by_id = {e["id"]: (e, f) for e, f in members}
        self.primary = members[0][1].routes[0]
        self.clock = members[0][1].clocks[0]
        for e, f in members:
            if not e.get("legacy"):
                cuts = [p["local_s"][0] for p in e["phases"]] + [
                    e["phases"][-1]["local_s"][1]
                ]
                if (
                    cuts[0] != f.lo
                    or cuts[-1] != f.hi
                    or not all(np.any(f.alt.cum == x) for x in cuts)
                ):
                    raise RouteError(
                        "phase spans must cover branch_s and end at line vertices"
                    )
        self.reset()

    def reset(self):
        self.key, self.phase, self.done = None, 0, set()
        self.peak = self.credit = self.anchor = None
        self.clock_ms, self.timing = 0.0, {}
        self.primary.reset()
        for _, f in self.members:
            f.reset()

    def restore(self, key, phase, local):
        e, f = self.by_id[key]
        self.key, self.phase = key, phase
        f.routes[1].s_prev = local

    def project(self, e, f, pos, sector):
        route = f.routes[1]
        if e.get("legacy"):
            return route.update(pos, sector)
        lo, hi = e["phases"][self.phase]["local_s"]
        path = route.path

        # Do not reacquire across overlapping outbound and inbound legs.
        def near(a, b):
            return path.nearest(pos, path._segments(a, b))

        pr = near(lo, hi)
        jump = True
        if route.s_prev is not None:
            a = max(lo, route.s_prev - route.kw.get("back", 10))
            b = min(hi, route.s_prev + route.kw.get("ahead", 30))
            if a < b:
                local = near(a, b)
                if local["dist"] <= pr["dist"] + route.kw.get("reacquire", 5):
                    pr, jump = local, False
        route.s_prev = pr["s"]
        pr.update(jump=jump, mode="reacquire" if jump else "window")
        return pr

    def update(self, pos, sector, bt):
        pos = np.asarray(pos)
        old = self.key
        changed = False
        if old is None:
            pr = self.primary.update(pos, sector)
            for e, f in self.members:
                if e["id"] in self.done:
                    continue
                if e.get("legacy"):
                    if sector != f.sector or bt != f.ball_type:
                        continue
                    alt = self.project(e, f, pos, sector)
                    if not f.lo <= alt["s"] <= f.hi:
                        continue
                else:
                    if not matches(e["phases"][0]["when"], pos, sector, bt, pr["s"]):
                        continue
                    self.phase = 0
                    alt = self.project(e, f, pos, sector)
                self.phase = 0
                self.key = e["id"]
                pr = alt
                break
        else:
            e, f = self.by_id[old]
            pr = self.project(e, f, pos, sector)
            if e.get("legacy"):
                leave = (
                    sector != f.sector
                    or bt != f.ball_type
                    or not f.lo <= pr["s"] <= f.hi
                )
            else:
                phases = e["phases"]
                left_sector = phases[self.phase]["when"].get("sector", sector) != sector
                leave = self.phase == len(phases) - 1 and matches(
                    e["merge"], pos, sector, bt, pr["s"]
                )
                if leave and not left_sector:
                    self.account(pr["s"], f.alt, f.clocks[1], *f.stations, False)
                target = next(
                    (
                        i
                        for i in range(self.phase + 1, len(phases))
                        if self.phase in phases[i].get("from", (i - 1,))
                        and matches(phases[i]["when"], pos, sector, bt, pr["s"])
                    ),
                    None,
                )
                if not leave and target is not None:
                    self.account(pr["s"], f.alt, f.clocks[1], *f.stations, False)
                    self.phase = target
                    pr = self.project(e, f, pos, sector)
                    changed = True
                elif left_sector:
                    leave = True
            if leave:
                if not e.get("legacy"):
                    self.done.add(old)
                self.key = None
                self.primary.reset()
                f.routes[1].reset()
                pr = self.primary.update(pos, sector)
        local = pr["s"]
        key = self.key
        if key is not None:
            e, f = self.by_id[key]
            ms, bs = f.stations
            path = f.alt
            clock = f.clocks[1]

        else:
            path = self.main
            clock = self.clock
            ms = bs = (
                self.by_id[old][1].stations[0]
                if old is not None
                else [0.0, path.length]
            )
        self.account(local, path, clock, ms, bs, key != old or changed)
        frontier = np.interp(self.peak, ms, bs) if key else self.peak
        self.timing = dict(
            branch=key is not None,
            branch_id=key,
            phase=self.phase if key else None,
            ref_elapsed_ms=self.clock_ms,
            ref_remaining_ms=float(clock[-1] - np.interp(frontier, path.cum, clock)),
        )
        pr.update(s=self.credit, local_s=local, **self.timing)
        return path, pr

    def account(self, local, path, clock, ms, bs, switched):
        station = float(np.interp(local, bs, ms))
        if self.peak is None:
            self.peak = self.credit = station
        elif switched:
            self.peak = max(self.peak, station)
            end = ms[min(np.searchsorted(ms, self.peak, side="right"), len(ms) - 1)]
            self.anchor = self.peak, self.credit, end
        elif station > self.peak:
            start = np.interp(self.peak, ms, bs)
            self.clock_ms += np.interp(local, path.cum, clock) - np.interp(
                start, path.cum, clock
            )
            self.peak = station
            if self.anchor and station < self.anchor[2]:
                x, y, end = self.anchor
                self.credit = y + (station - x) * (end - y) / (end - x)
            else:
                self.credit = station
                self.anchor = None
