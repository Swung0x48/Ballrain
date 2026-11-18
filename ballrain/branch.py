"""Optional ball-type diversion. Physical lookahead is never in curriculum coordinates."""

from __future__ import annotations

import json
import numpy as np

from .branch_rules import BranchRules
from .path import (
    PathFollower,
    RouteError,
    asset_path,
    file_sha256,
    load_tas_path,
    resolve_path_dir,
)


class BranchFollower:
    def __init__(self, main, alt, meta, **kw):
        self.rules = BranchRules.from_desc(meta)
        self.main, self.alt = main, alt
        self.ball_type, self.sector = meta["ball_type"], meta["sector"]
        self.lo, self.hi = meta["branch_s"]
        self.stations = np.asarray(
            [
                (0, 0),
                (meta["main_s"][0], self.lo),
                *meta["stations"],
                (meta["main_s"][1], self.hi),
                (main.length, alt.length),
            ]
        ).T
        if not np.all(np.diff(self.stations) > 0):
            raise RouteError("branch stations must increase on both lines")
        self.routes = [PathFollower(main, **kw), PathFollower(alt, **kw)]
        self.clocks = [main.t_ms, alt.t_ms]
        if main.t_ms is None:
            # Older TAS assets have no t_ms. The generator supplies the replaced samples;
            # before and after the splice both clocks differ only by a constant.
            i, j = np.searchsorted(main.cum, meta["main_s"])
            k, l = np.searchsorted(alt.cum, meta["branch_s"])
            span = meta["main_span_t_ms"]
            self.clocks[0] = np.r_[
                alt.t_ms[:k], span, alt.t_ms[l + 1 :] - alt.t_ms[l] + span[-1]
            ]
        for path, clock in zip((main, alt), self.clocks):
            if clock is None or len(clock) != len(path.P) or not np.all(np.diff(clock) > 0):
                raise RouteError("branch timing must cover every point and increase strictly")
        self.reset()

    def reset(self):
        self.active = self.peak = self.credit = None
        self.clock_ms = 0.0
        self.anchor = None
        self.timing = {}
        for f in self.routes:
            f.reset()

    def update(self, pos, sector, bt):
        use = sector == self.sector and bt == self.ball_type
        if use:
            br = self.routes[1].update(pos, sector)
            use = self.lo <= br["s"] <= self.hi
        else:
            self.routes[1].reset()
        if use:
            self.routes[0].reset()
        pr = br if use else self.routes[0].update(pos, sector)
        path = self.routes[use].path
        local = pr["s"]
        ms, bs = self.stations
        station = float(np.interp(local, bs, ms)) if use else local
        if self.active is None:
            self.credit = self.peak = station
        elif self.active != use:
            # Consume the projection discontinuity without paying for it. Only never-before-
            # reached stations earn credit, even after repeated switches at the same gate.
            self.peak = max(self.peak, station)
            end = ms[min(np.searchsorted(ms, self.peak, side="right"), len(ms) - 1)]
            self.anchor = self.peak, self.credit, end
        elif station > self.peak:
            before = self.peak
            self.peak = station
            if self.anchor and station < self.anchor[2]:
                x, y, end = self.anchor
                self.credit = y + (station - x) * (end - y) / (end - x)
            else:
                self.anchor = None
                self.credit = station
            start = np.interp(before, ms, bs) if use else before
            self.clock_ms += np.diff(
                np.interp([start, local], path.cum, self.clocks[use])
            )[0]
        self.active = use
        frontier = np.interp(self.peak, ms, bs) if use else self.peak
        remaining = self.clocks[use][-1] - np.interp(frontier, path.cum, self.clocks[use])
        self.timing = dict(branch=use, ref_elapsed_ms=self.clock_ms, ref_remaining_ms=float(remaining))
        pr.update(
            s=self.credit,
            local_s=local,
            branch=use,
            ref_elapsed_ms=self.clock_ms,
            ref_remaining_ms=float(remaining),
        )
        return path, pr


def load_branch(main, level, root, main_root, **kw):
    asset = asset_path(level, resolve_path_dir(root))
    desc = json.loads(asset.with_suffix(".json").read_text(encoding="utf-8-sig"))
    if desc["level"] != level or desc["main_sha256"] != file_sha256(
        asset_path(level, resolve_path_dir(main_root))
    ):
        raise RouteError("branch is not for this main reference asset")
    alt = load_tas_path(level, root=root, strict=True, sha256=desc["branch_sha256"])
    return BranchFollower(main, alt, desc, **kw)
