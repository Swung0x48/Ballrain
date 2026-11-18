"""TAS start states and a reverse curriculum (docs/observations.md, "TAS 起点").

An episode can start from frame k of a TAS record: the env writes the record's first k frames as the
BALLRAIN_PREFIX of a new game process, the mod replays them (no round trips) and hands control to Python at
frame k (docs/agent-protocol.md 5). The exported .tas of that process is prefix + agent frames.

TasStarts(record) loads the record's frames and its valid start frames from the TAS dataset
(<runtime>/datasets/tas-v1/<record>/, cached in starts.npz there): k = 0 (a normal start from level load;
level 1 then presses the scripted Q) or a frame where the replayed record has control (BallNav active, IVP
ball found, not dead, not transforming, not paused). k counts the record's own frames (a 264 fps record has
264 frames per game second), not 132 fps env steps.

ReverseCurriculum samples k: it keeps a frontier (an index into the valid frames, starting near the end) and
draws k from the window [frontier, end] (a fraction `recent` of the draws from the last `band` frames above the
frontier, the rest uniformly over the whole window, so earlier-solved starts keep being visited). The trainer
reports outcomes of TAS-start episodes with update(k, success) (or update_many); when the success rate over
the last `min_episodes` episodes started within `band` frames above the frontier reaches `advance_at`, the
frontier moves back by `step` frames (never below the first valid frame; k = 0 is added once the frontier has
reached the first control frame, with probability `p_level_start`).

The state is a plain dict (state_dict / load_state_dict), so a trainer keeps one ReverseCurriculum, feeds it
vector_outcomes(infos) after every vector step (cu.update_many(...)) and pushes its state to the envs: gymnasium vector envs `venv.call("set_start_state", state)`, stable-baselines3
`venv.env_method("set_start_state", state)`. With the warm pool (EnvConfig.warm_pool) the start frame of the
next process is drawn when that process is pre-launched, i.e. one process ahead of the state update.
"""
from __future__ import annotations

import json
import math
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import dataset, instances, tas

STARTS_VERSION = 1
CAMERA_QUARTER_TOL = 1e-4  # radians; retail Cam_OrientRef at rest is within ~1.1e-6 of a quarter turn (SR01 replay)
CAMERA_TURN_GRACE_MS = 400.0  # a shift-turn animates ~250 ms (Cam_Orient); Cam_OrientRef may be off a quarter meanwhile


def dataset_dir() -> Path:
    return instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"


class TasStarts:
    """Frames and valid start frames of one dataset record."""

    def __init__(self, record: str, root=None, camera_tol: float | None = CAMERA_QUARTER_TOL,
                 camera_grace_ms: float = CAMERA_TURN_GRACE_MS):
        """record: a dataset record name (in root, default <runtime>/datasets/tas-v1) or a record directory.
        camera_tol: drop start frames whose Cam_OrientRef is off a quarter turn by more than this (radians) outside
        camera_grace_ms after a camera turn press of the record (the env's camera guard; None = keep all)."""
        self._camera_filter = (camera_tol, camera_grace_ms)
        self.record = Path(record).name if Path(record).is_dir() else Path(record).stem
        self.dir = Path(record) if Path(record).is_dir() else Path(root or dataset_dir()) / self.record
        if not (self.dir / "raw.npz").is_file():
            raise FileNotFoundError(f"{self.dir} has no raw.npz (build the TAS dataset: scripts/m1_gates.py --gates D)")
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.level = int(self.meta["level"])
        with np.load(self.dir / "raw.npz") as z:
            self.frames = z["record_frames"].copy()
        cache = self.dir / "starts.npz"
        if cache.is_file():
            c = np.load(cache)
            if int(c["version"]) == STARTS_VERSION:
                self._set(c)
                return
        tr = dataset.record_trajectory(self.dir, self.level)
        valid = tr["control"] & ~tr["transforming"]
        lf = tr["levelfinish_frame"]
        fd = tr["frames_done"]
        if lf is not None:
            valid &= fd < lf
        c = {"version": np.array(STARTS_VERSION), "frames_done": fd[valid], "sector": tr["sector"][valid],
             "sr_double": tr["sr_double"][valid], "points": tr["points"][valid], "lives": tr["lives"][valid],
             "pos": tr["pos"][valid], "checkpoints": tr["checkpoints"][valid],
             "levelfinish_frame": np.array(-1 if lf is None else lf),
             "sr_final": np.array(self.meta.get("sr_double_final") or np.nan)}
        tmp = cache.with_name(f"starts.{os.getpid()}.{threading.get_ident()}.tmp.npz")
        np.savez(tmp, **c)
        os.replace(tmp, cache)  # several workers may build it at once
        self._set(c)

    def _apply_camera_filter(self):
        tol, grace_ms = self._camera_filter
        if tol is None:
            return
        d = self.frames["delta"]
        ms = float(np.median(d)) if len(d) else 1000 / 264
        ok = self.camera_ok(tol, int(grace_ms / ms))
        self.camera_dropped = int((~ok).sum())
        if self.camera_dropped:
            self.restrict(ok)

    def _set(self, c):
        self.valid = np.asarray(c["frames_done"], np.int64)  # observation after k frames = state at frame k
        self.sector = np.asarray(c["sector"])
        self.sr_double = np.asarray(c["sr_double"])
        self.points, self.lives = np.asarray(c["points"]), np.asarray(c["lives"])
        self.pos, self.checkpoints = np.asarray(c["pos"]), np.asarray(c["checkpoints"])
        lf = int(c["levelfinish_frame"])
        self.levelfinish_frame = None if lf < 0 else lf
        self.sr_final = float(c["sr_final"])
        self.camera_dropped = 0
        self._apply_camera_filter()

    def camera(self) -> dict:
        """Cam_OrientRef of the replayed record per frames_done (the state after that many frames): "quarter" (yaw in
        quarter turns 0..3, -1 before the first CAMS section) and "dev" (radians off the nearest quarter turn, 0 there).
        Cached in camera.npz in the record directory."""
        if getattr(self, "_cam", None) is not None:
            return self._cam
        cache = self.dir / "camera.npz"
        if cache.is_file():
            with np.load(cache) as z:
                self._cam = {"quarter": z["quarter"], "dev": z["dev"]}
            return self._cam
        from .features import yaw_of
        fds, qs, ds = [], [], []
        for o in dataset.load_observations(self.dir / "raw.npz"):
            if o.cams is None:
                continue
            y = yaw_of(o.cams["cams"][0]["matrix"])
            fds.append(int(o.frames_done))
            qs.append(int(round(y / (np.pi / 2))) % 4)
            ds.append(abs(math.remainder(y, np.pi / 2)))
        n = (max(fds) + 1) if fds else 0
        q, d = np.full(n, -1, np.int8), np.zeros(n, np.float32)
        q[np.asarray(fds, np.int64)], d[np.asarray(fds, np.int64)] = qs, ds
        for i in range(1, n):  # frames without a stored observation keep the last known camera
            if q[i] < 0 and q[i - 1] >= 0:
                q[i], d[i] = q[i - 1], d[i - 1]
        tmp = cache.with_name(f"camera.{os.getpid()}.{threading.get_ident()}.tmp.npz")
        np.savez(tmp, quarter=q, dev=d)
        os.replace(tmp, cache)
        self._cam = {"quarter": q, "dev": d}
        return self._cam

    def camera_quarters(self) -> np.ndarray:
        return self.camera()["quarter"]

    def camera_ok(self, tol: float, grace_frames: int) -> np.ndarray:
        """Per valid start frame: Cam_OrientRef within tol of a quarter turn, or within grace_frames after a camera turn
        press (shift + left / right) of the record (ballrain.env camera guard)."""
        dev = self.camera()["dev"]
        keys = self.frames["keys"]
        turn = ((keys & tas.KEY_SHIFT) != 0) & ((keys & (tas.KEY_LEFT | tas.KEY_RIGHT)) != 0)
        edge = turn & ~np.r_[False, turn[:-1]]
        last = np.maximum.accumulate(np.where(edge, np.arange(len(keys)), -10 ** 9))
        k = self.valid
        d = dev[np.clip(k, 0, len(dev) - 1)] if len(dev) else np.zeros(len(k))
        recent = (k - last[np.clip(k - 1, 0, len(last) - 1)]) <= grace_frames
        return (d <= tol) | recent

    def restrict(self, mask: np.ndarray) -> None:
        """Keep only the valid start frames where mask is True."""
        mask = np.asarray(mask, bool)
        self.valid, self.sector, self.sr_double = self.valid[mask], self.sector[mask], self.sr_double[mask]
        self.points, self.lives, self.pos, self.checkpoints = (self.points[mask], self.lives[mask], self.pos[mask],
                                                               self.checkpoints[mask])

    def exclude(self, mask: np.ndarray) -> None:
        """Like restrict (the curriculum, rehearsal starts and nearest_valid see only the frames where mask is True), but
        the excluded frames stay hand-over states: nearest_any() can pick them and expected() knows their state (at-rest
        starts with a drive target on a stretch no TAS start may cover, e.g. L1's re-routed 4450-6018)."""
        if getattr(self, "_all", None) is None:
            self._all = (self.valid, self.sector, self.sr_double, self.points, self.lives, self.pos, self.checkpoints)
        self.restrict(mask)

    def nearest_any(self, k: int) -> int:
        """nearest_valid over the frames before exclude() (camera-filtered start states included, excluded ones too)."""
        v = self._all[0] if getattr(self, "_all", None) is not None else self.valid
        if k <= 0:
            return 0
        i = int(np.clip(np.searchsorted(v, k), 0, len(v) - 1))
        j = max(i - 1, 0)
        return int(v[i] if abs(v[i] - k) <= abs(v[j] - k) else v[j])

    def _state_arrays(self, k: int):
        """(index, arrays) of frame k among the kept frames, else among the excluded ones (exclude), else (None, None)."""
        i = self.index_of(k)
        if i is not None:
            return i, (self.valid, self.sector, self.sr_double, self.points, self.lives, self.pos, self.checkpoints)
        a = getattr(self, "_all", None)
        if a is not None:
            j = int(np.searchsorted(a[0], k))
            if j < len(a[0]) and a[0][j] == k:
                return j, a
        return None, None

    def index_of(self, k: int) -> int | None:
        i = int(np.searchsorted(self.valid, k))
        return i if i < len(self.valid) and self.valid[i] == k else None

    def nearest_valid(self, k: int) -> int:
        """The valid start frame closest to k (0 stays 0)."""
        if k <= 0:
            return 0
        i = int(np.clip(np.searchsorted(self.valid, k), 0, len(self.valid) - 1))
        j = max(i - 1, 0)
        return int(self.valid[i] if abs(self.valid[i] - k) <= abs(self.valid[j] - k) else self.valid[j])

    def expected(self, k: int) -> dict | None:
        """The record's state at frame k (for hand-over checks): sector, checkpoints, points, lives, SR, ball."""
        i, a = self._state_arrays(k)
        if i is None:
            return None
        _, sector, sr, points, lives, pos, cps = a
        return {"frame": int(k), "sector": int(sector[i]), "checkpoints": int(cps[i]), "points": int(points[i]),
                "lives": int(lives[i]), "sr_double": float(sr[i]), "pos": pos[i].tolist()}

    def prefix(self, k: int) -> np.ndarray:
        return self.frames[:k]


@dataclass
class CurriculumConfig:
    start_back: int = 1500  # initial frontier: this many record frames before the last valid frame
    step: int = 1000  # frames the frontier moves back on success
    band: int = 1500  # frames above the frontier that count as "the frontier"
    recent: float = 0.7  # fraction of draws from the band; the rest uniform over [frontier, end]
    advance_at: float = 0.5  # success rate needed to move back
    min_episodes: int = 20  # outcomes in the band before a decision
    p_level_start: float = 0.1  # probability of k = 0 once the frontier is at the first control frame
    # failure-weighted starts: this fraction of the draws (after the k = 0 draw) picks a bin of bin_frames record frames in
    # [frontier, end] with probability floor + (1 - n_bins * floor) * fail_b / sum(fail) (fail_b = the trainer's recent
    # failure rate of episodes started in bin b, set_bin_fail; bins without data count as 1), then a frame in the bin
    # uniformly. Counters forgetting: mastered bins keep the floor share, failing bins get the rest
    weighted: float = 0.0
    bin_frames: int = 2000
    bin_floor: float = 0.05
    seed: int | None = None


class ReverseCurriculum:
    def __init__(self, starts: TasStarts | None = None, config: CurriculumConfig | None = None,
                 valid=None):
        self.cfg = config or CurriculumConfig()
        self.valid = np.asarray(valid if valid is not None else starts.valid, np.int64)
        if len(self.valid) == 0:
            raise ValueError("no valid start frames")
        last = int(self.valid[-1])
        self.frontier = self._snap(last - self.cfg.start_back)
        self.outcomes: deque = deque(maxlen=max(self.cfg.min_episodes, 1))
        self.rng = np.random.default_rng(self.cfg.seed)
        self.history: list = []  # (frontier, success rate) at each move
        self.bin_fail: dict = {}  # bin index -> recent failure rate (set_bin_fail; failure-weighted starts)

    def _snap(self, k: int) -> int:
        i = int(np.clip(np.searchsorted(self.valid, k), 0, len(self.valid) - 1))
        return int(self.valid[i])

    @property
    def at_start(self) -> bool:
        return self.frontier <= int(self.valid[0])

    def bin_of(self, k: int) -> int:
        """Bin of a start frame for the failure-weighted draws (bins of bin_frames from the first valid frame)."""
        return int((k - int(self.valid[0])) // max(self.cfg.bin_frames, 1))

    def set_bin_fail(self, fail: dict) -> None:
        self.bin_fail = {int(b): float(v) for b, v in fail.items()}

    def bin_probs(self) -> tuple[np.ndarray, np.ndarray]:
        """(bins, probabilities) of the failure-weighted draw over the bins of [frontier, end]."""
        b0, b1 = self.bin_of(self.frontier), self.bin_of(int(self.valid[-1]))
        bins = np.arange(b0, b1 + 1)
        fail = np.array([self.bin_fail.get(int(b), 1.0) for b in bins], np.float64)
        n = len(bins)
        floor = min(self.cfg.bin_floor, 1.0 / n)
        tot = fail.sum()
        rest = (fail / tot) if tot > 0 else np.full(n, 1.0 / n)
        return bins, floor + (1 - n * floor) * rest

    def sample(self, rng=None) -> int:
        rng = rng or self.rng
        if self.at_start and rng.random() < self.cfg.p_level_start:
            return 0
        lo = int(np.searchsorted(self.valid, self.frontier))
        if self.cfg.weighted and rng.random() < self.cfg.weighted:
            bins, p = self.bin_probs()
            b = int(bins[rng.choice(len(bins), p=p)])
            base = int(self.valid[0])
            blo = int(np.searchsorted(self.valid, max(self.frontier, base + b * self.cfg.bin_frames)))
            bhi = int(np.searchsorted(self.valid, base + (b + 1) * self.cfg.bin_frames, side="left"))
            if bhi > blo:
                return int(self.valid[rng.integers(blo, bhi)])
        if rng.random() < self.cfg.recent:
            hi = int(np.searchsorted(self.valid, self.frontier + self.cfg.band, side="right"))
        else:
            hi = len(self.valid)
        hi = max(hi, lo + 1)
        return int(self.valid[rng.integers(lo, hi)])

    def in_band(self, k: int) -> bool:
        return (k == 0 and self.at_start) or self.frontier <= k <= self.frontier + self.cfg.band

    def update(self, k: int, success: bool) -> bool:
        """Report the outcome of an episode that started at TAS frame k. Returns True when the frontier moved."""
        if not self.in_band(k):
            return False
        self.outcomes.append(bool(success))
        if len(self.outcomes) >= self.cfg.min_episodes:
            rate = float(np.mean(self.outcomes))
            if rate >= self.cfg.advance_at and not self.at_start:
                self.frontier = self._snap(self.frontier - self.cfg.step)
                self.history.append((self.frontier, rate))
                self.outcomes.clear()
                return True
        return False

    def update_many(self, results) -> int:
        """results: iterable of (k, success). Returns how often the frontier moved."""
        return sum(self.update(k, s) for k, s in results)

    def state_dict(self) -> dict:
        return {"frontier": int(self.frontier), "config": asdict(self.cfg), "outcomes": list(self.outcomes),
                "history": list(self.history), "bin_fail": dict(self.bin_fail)}

    def load_state_dict(self, state: dict) -> None:
        self.frontier = self._snap(int(state["frontier"]))
        if "config" in state:
            cfg = dict(state["config"])
            seed = self.cfg.seed
            self.cfg = CurriculumConfig(**cfg)
            self.cfg.seed = seed
            self.outcomes = deque(state.get("outcomes", []), maxlen=max(self.cfg.min_episodes, 1))
        self.history = list(state.get("history", []))
        self.bin_fail = {int(b): float(v) for b, v in (state.get("bin_fail") or {}).items()}


def vector_outcomes(infos) -> list:
    """(k, success) of every TAS-start episode that ended in one vector-env step. infos: gymnasium's
    dict of arrays (with the "_key" masks) or stable-baselines3's list of per-env dicts."""
    if isinstance(infos, dict):
        ends, starts = infos.get("episode_end"), infos.get("start")
        if ends is None or starts is None:
            return []
        m_end = infos.get("_episode_end", [e is not None for e in ends])
        m_start = infos.get("_start", [True] * len(ends))

        def row(d, i):  # gymnasium merges nested dicts key by key ("start" -> {"kind": array, ...})
            if isinstance(d, dict):
                return {k: row(v, i) for k, v in d.items() if not k.startswith("_") and d.get("_" + k, [True] * len(ends))[i]}
            return d[i]

        def start_of(i):
            return row(starts, i)
        rows = [{"episode_end": ends[i], "start": start_of(i)} for i in range(len(ends)) if m_end[i] and m_start[i]]
    else:
        rows = list(infos)
    return [r for r in (episode_outcome(x) for x in rows) if r is not None]


def episode_outcome(info: dict) -> tuple[int, bool] | None:
    """(k, success) of a finished TAS-start episode from the env's last step info, None for other episodes
    (respawn episodes, starts from level load without a curriculum). Success = LevelFinish."""
    st = info.get("start")
    if not st or st.get("kind") != "tas" or "episode_end" not in info:
        return None
    return int(st["frame"]), info["episode_end"] == "LevelFinish"
