"""The f7 "reach" observation block: ballrain.fastreach.perceive of the current ball state, scaled to [-1, 1].

    rb = ReachBlock()
    block, raw = rb.compute(tracker, pos, vel, quarter, ball_type)   # block (N_REACH,) float64 in [-1, 1]

Fields: FastReach.fields_for_tracker (static geometry only, content-addressed in <runtime>/fastreach or
$BALLRAIN_FASTREACH_DIR, one build per host under a lock, atomic writes); the unknown zones (rest-pose boxes of the
physicalized moving entities, player balls excluded) are recomputed from the tracker whenever its level data changes,
and the moving bodies of the BODY section within the extractor's body_radius set unknown_zone near them. Both caches
are per process (module level): every env of a process shares the fields of one geometry.

Scaling (docs/observations.md, f7): flags and normals as they are; times (air_t_first, air_t_near, in s, horizon 3) / 3;
lengths and speed deficits (fastreach caps them at +-L_MAX = +-CLIP_V = 20) / 20; then clipped to [-1, 1].
Directions: the 8 key directions of fastreach (DIR_NAMES: up, down, left, right, up_left, up_right, down_left,
down_right = reach.KEY_COMBOS = env.ARROWS9[1:]) under the camera quarter of the policy's actions (0 with world actions).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

from . import fastreach as FR
from .state import BALL_NAME_TYPES, G_STOPPER

N_REACH = FR.N_FEATURES
REACH_NAMES = FR.FEATURE_NAMES
T_SCALE = FR.N_PSI / 132.0  # air times are n PSI / 132 (s): the 3 s horizon -> 1
L_SCALE = FR.L_MAX  # 20: lengths (edge, brake margin, obstacle, landing offsets / drops) -> [-1, 1]
V_SCALE = FR.CLIP_V  # 20: speed deficits


def _scale_vector() -> np.ndarray:
    s = np.ones(N_REACH, np.float64)
    i = {n: k for k, n in enumerate(REACH_NAMES)}
    for n in ("air_t_first", "air_t_near"):
        s[i[n]] = 1.0 / T_SCALE
    for n in ("air_first_dy", "air_near_r", "air_near_f", "air_near_dy"):
        s[i[n]] = 1.0 / L_SCALE
    for d in FR.DIR_NAMES:
        for f in ("edge", "brake_margin", "obst"):
            s[i[f"{d}_{f}"]] = 1.0 / L_SCALE
        for f in ("gap_deficit", "obst_deficit"):
            s[i[f"{d}_{f}"]] = 1.0 / V_SCALE
    return s


SCALE = _scale_vector()

# process-wide caches: static fields by fields key (two geometries per level at most on L2; LRU), and per level data
# version the fields key and the unknown zones
_FIELDS: "OrderedDict[str, FR.FastReach]" = OrderedDict()
_FIELDS_MAX = 2
_STATE: dict = {}  # (id(LevelData), version) -> (LevelData, fields key, zone array)


def require_numba() -> None:
    """The reach block runs the fastreach kernels on every env step: refuse without numba (the plain-Python path is
    ~1000x slower) instead of degrading silently."""
    if not FR.HAVE_NUMBA:
        raise RuntimeError("FeatureConfig.reach (f7) needs numba for ballrain.fastreach (install the optional "
                           "dependency group: uv sync --extra fastreach, or pip install numba); BALLRAIN_NO_NUMBA is "
                           f"{'set' if __import__('os').environ.get('BALLRAIN_NO_NUMBA') else 'not set'}")


def _zone_boxes(level) -> np.ndarray:
    boxes = FR.FastReach.zone_boxes(level, exclude_names=BALL_NAME_TYPES)
    return np.array([np.r_[np.asarray(lo, np.float64) - 4.0, np.asarray(hi, np.float64) + 4.0] for lo, hi in boxes],
                    np.float64).reshape(-1, 6)


def fields_for(tracker, cache_dir=None):
    """(FastReach with the tracker's current unknown zones, fields key)."""
    from .features import level_geometry
    lvl = tracker.level
    vk = (id(lvl), lvl.version)
    st = _STATE.get(vk)
    if st is None or st[0] is not lvl:
        geo = level_geometry(tracker)  # cached by the extractor already: its key is the static key
        key = FR.FastReach.fields_key(tracker, geo.key)
        if len(_STATE) > 16:
            _STATE.clear()
        st = (lvl, key, _zone_boxes(lvl))
        _STATE[vk] = st
    _, key, zones = st
    fr = _FIELDS.get(key)
    if fr is None:
        fr = FR.FastReach.fields_for_tracker(tracker, cache_dir=cache_dir, static_key=level_geometry(tracker).key)
        _FIELDS[key] = fr
        while len(_FIELDS) > _FIELDS_MAX:
            _FIELDS.popitem(last=False)
    else:
        _FIELDS.move_to_end(key)
    if fr.uzones is not zones:
        fr.uzones = zones
    return fr, key


def moving_bodies(tracker, pos, radius: float) -> np.ndarray:
    """(k, 3) positions of the BODY section's moving bodies within radius of pos (stoppers excluded, as the bodies
    block); the section never lists the player ball itself."""
    obs = tracker.obs
    if obs is None or not len(obs.body):
        return np.zeros((0, 3))
    lvl = tracker.level
    out = []
    for b in obs.body:
        if lvl.in_group(int(b["ckid"]), G_STOPPER):
            continue
        p = np.asarray(b["pos"], np.float64)
        if np.linalg.norm(p - pos) <= radius:
            out.append(p)
    return np.array(out, np.float64).reshape(-1, 3)


def scale(raw) -> np.ndarray:
    v = np.asarray(raw, np.float64) * SCALE
    np.nan_to_num(v, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(v, -1.0, 1.0)


class ReachBlock:
    def __init__(self, cache_dir=None, body_radius: float = 60.0):
        require_numba()
        self.cache_dir = cache_dir
        self.body_radius = body_radius

    def compute(self, tracker, pos, vel, quarter: int, ball_type: int):
        """(scaled block (N_REACH,), raw perceive vector (N_REACH,), fields key)."""
        fr, key = fields_for(tracker, self.cache_dir)
        p = np.asarray(pos, np.float64)
        bodies = moving_bodies(tracker, p, self.body_radius)
        raw, _ = fr.perceive(p, vel, int(ball_type), int(quarter), bodies=bodies)
        return scale(raw), raw, key


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]
