"""Collect points from a start record's own pauses (EnvConfig.collect_points, scripts/train_m2.py --collect-at tas-pauses).

The L1 TAS pauses once per sector, shortly before each checkpoint: frames 8921, 14821 and 18243, at 345, 180 and 400
frames before checkpoints 9266, 15001 and 18645. It never pauses in the last sector, and a player does the same. Every
PauseLevel of the record becomes one [sector, route s] point: the sector of the record at that frame, and the route arc
length of the record's ball there (nearest point of the reference line within that sector)."""
from __future__ import annotations

import numpy as np


def pause_frames(starts) -> list[int]:
    """Frames of the record's PauseLevel events (events.txt in the record directory: "<frame> <Name>" per line)."""
    out = []
    for line in (starts.dir / "events.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "PauseLevel":
            out.append(int(parts[0]))
    return out


def points(starts, path, margin: float = 5.0) -> list[list]:
    """[[sector, route s], ...] for each pause of the record after its first controllable frame, capped at margin route
    units before the record's next checkpoint (its ball at the first frame of the next sector), so a life that reaches
    the checkpoint a little earlier than the record still collects before it."""
    fd = np.asarray(starts.valid)
    sec_all = np.asarray(starts.sector)
    out = []
    for f in pause_frames(starts):
        if f < int(starts.control_frame):
            continue
        i = min(int(np.searchsorted(fd, f)), len(fd) - 1)
        sec = int(starts.sector[i])
        p = np.asarray(starts.pos[i], np.float64)
        m = np.asarray(path.point_sector) == sec
        idx = np.flatnonzero(m) if m.any() else np.arange(len(path.P))
        j = idx[int(np.argmin(np.linalg.norm(path.P[idx] - p, axis=1)))]
        s_pause = float(path.cum[j])
        nxt = np.flatnonzero(sec_all > sec)
        if len(nxt):
            q = np.asarray(starts.pos[nxt[0]], np.float64)
            k = int(np.argmin(np.linalg.norm(path.P - q, axis=1)))
            s_pause = min(s_pause, float(path.cum[k]) - margin)
        out.append([sec, round(s_pause, 2)])
    return out
