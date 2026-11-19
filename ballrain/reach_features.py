"""The f7 "reach" observation block: ballrain.fastreach.perceive of the current ball state, scaled to [-1, 1].

    rb = ReachBlock()
    block, raw = rb.compute(tracker, pos, vel, quarter, ball_type)   # block (N_REACH,) float64 in [-1, 1]

Fields: FastReach.fields_for_tracker (static geometry only, content-addressed in
<runtime>/fastreach ($BALLRAIN_RUNTIME), one build per host under a lock, atomic writes); the unknown zones (rest-pose boxes of the
physicalized moving entities, player balls excluded) are recomputed from the tracker whenever its level data changes,
and the moving bodies of the BODY section within the extractor's body_radius set unknown_zone near them. Both caches
are per process (module level): every env of a process shares the fields of one geometry.

Scaling (docs/observations.md, f7): flags and normals as they are; times (air_t_first, air_t_near, in s, horizon 3) / 3;
lengths and speed deficits (fastreach caps them at +-L_MAX = +-CLIP_V = 20) / 20; then clipped to [-1, 1].
Directions: the 8 key directions of fastreach (DIR_NAMES: up, down, left, right, up_left, up_right, down_left,
down_right = reach.KEY_COMBOS = env.ARROWS9[1:]) under the camera quarter of the policy's actions (0 with world actions).

Version pin: reach_version() names everything that defines the block's values (fastreach BUILD_VERSION: the fields;
QUERY_VERSION: perceive; BLOCK_VERSION here: names, order, scaling, zones, bodies); FeatureConfig.reach_version records
it and ReachBlock refuses another (tests/test_reach_obs.py pins a digest of the names and scales to BLOCK_VERSION).
"""
from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict

import numpy as np

from . import fastreach as FR
from .state import BALL_NAME_TYPES, G_STOPPER

BLOCK_VERSION = 1  # bump with any change of this module's output for the same perceive vector (scaling, zones, bodies)
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
_STATE: dict = {}  # (id(LevelData), version) -> (LevelData, fields key, zone array, canonical triangles or None)
# Thread safety: a FastReach query writes its results into the instance's own scratch arrays (and fields_for sets its
# unknown zones), and the numba kernels run without the GIL. Several threads of one process (the pool verifier's
# replay threads) therefore each get their own view of a cached instance: a shallow copy sharing the loaded read-only
# field arrays (memory does not grow per thread), with private copies of the scratch arrays below. A thread's view is
# created from the instance on first use and then reused, so single-threaded results are exactly today's.
_SCRATCH = ("yy", "xf", "zf", "D", "fins", "yl", "ya", "stack", "stamp", "work", "out", "dbg", "pdd", "_nocollect")
_LOCAL = threading.local()
_LOCK = threading.Lock()  # the shared caches above (OrderedDict moves / evictions)


def _thread_view(fr: "FR.FastReach") -> "FR.FastReach":
    """This thread's view of the cached instance fr (see _SCRATCH)."""
    views = getattr(_LOCAL, "views", None)
    if views is None:
        views = _LOCAL.views = {}
    hit = views.get(id(fr))
    if hit is not None and hit[0] is fr:
        return hit[1]
    view = copy.copy(fr)
    for k in _SCRATCH:
        setattr(view, k, getattr(fr, k).copy())
    if len(views) >= 2 * _FIELDS_MAX:  # instances evicted from _FIELDS: drop their views
        views.clear()
    views[id(fr)] = (fr, view)
    return view


def reach_version(mechanisms: bool = False) -> str:
    """The f7 block version of this code (FeatureConfig.reach_version)."""
    base = f"fastreach{FR.BUILD_VERSION}.{FR.QUERY_VERSION}-block{BLOCK_VERSION}"
    if mechanisms:
        from .reach_mechanisms import asset_pin
        return f"{base}-mech1-{asset_pin()}"
    return base


def is_mechanism_version(version) -> bool:
    return isinstance(version, str) and "-mech1-" in version


# One bundle per level-data revision per process, with thread-local scratch as for legacy fields.
_MECH_FIELDS: OrderedDict = OrderedDict()


def _mechanism_fields(tracker, cache_dir, version):
    from pathlib import Path
    from .reach_mechanisms import add_geometry, fan_layer, layers_for_tracker
    level = tracker.level
    ck = (id(level), level.version, tracker.level_no, Path(cache_dir).resolve() if cache_dir else None, version)
    with _LOCK:
        hit = _MECH_FIELDS.get(ck)
        if hit is None:
            base, key, zones = _shared_fields(tracker, cache_dir)
            rows = [(base, key + "|" + version)] * 3
            built = {}
            geom = None
            layers, probe = layers_for_tracker(tracker)
            for k, addition in enumerate(layers):
                if addition is not None:
                    if id(addition) not in built:
                        if geom is None:
                            geom = FR.FastReach.canonical_geometry(tracker)
                        augmented = add_geometry(geom, addition)
                        fr = FR.FastReach.fields_for_tracker(tracker, cache_dir=cache_dir,
                                                             geometry=augmented, variant=version)
                        built[id(addition)] = (fr, fr.info["key"])
                    rows[k] = built[id(addition)]
            fans, recovered = fan_layer(tracker)
            hit = (level, tuple(rows), zones, probe, fans if recovered else None)
            _MECH_FIELDS[ck] = hit
            while len(_MECH_FIELDS) > _FIELDS_MAX:
                _MECH_FIELDS.popitem(last=False)
        _MECH_FIELDS.move_to_end(ck)
    _, rows, zones, probe, fans = hit
    suffix = "|fans-" + hashlib.sha256(fans.astype("<f8").tobytes()).hexdigest()[:16] if fans is not None else ""
    views = {}
    out = []
    for fr, key in rows:
        if id(fr) not in views:
            view = copy.copy(fr)
            for attr in _SCRATCH:
                setattr(view, attr, getattr(fr, attr).copy())
            view.uzones = zones
            if fans is not None:
                view.fans = fans
            views[id(fr)] = view
        out.append((views[id(fr)], key + suffix))
    return tuple(out), probe


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
    with _LOCK:
        fr, key, zones = _shared_fields(tracker, cache_dir)
    view = _thread_view(fr)
    if view.uzones is not zones:
        view.uzones = zones
    return view, key


def _shared_fields(tracker, cache_dir=None):
    """(the process-wide cached FastReach, fields key, the tracker's unknown zones); called under _LOCK."""
    lvl = tracker.level
    vk = (id(lvl), lvl.version)
    st = _STATE.get(vk)
    if st is None or st[0] is not lvl:
        geom = FR.FastReach.canonical_geometry(tracker)
        key = FR.FastReach.fields_key(tracker, geom)
        if len(_STATE) > 16:
            _STATE.clear()
        st = (lvl, key, _zone_boxes(lvl), geom)
        _STATE[vk] = st
    _, key, zones, geom = st
    fr = _FIELDS.get(key)
    if fr is None:
        fr = FR.FastReach.fields_for_tracker(tracker, cache_dir=cache_dir, geometry=geom)
        _FIELDS[key] = fr
        while len(_FIELDS) > _FIELDS_MAX:
            _FIELDS.popitem(last=False)
    else:
        _FIELDS.move_to_end(key)
    if geom is not None:
        _STATE[vk] = (lvl, key, zones, None)  # the triangles are only needed for a build
    return fr, key, zones


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


class ReachVersionError(RuntimeError):
    pass


class ReachBlock:
    def __init__(self, cache_dir=None, body_radius: float = 60.0, expect_version: str | None = None):
        """expect_version: FeatureConfig.reach_version (None: unchecked); another version than this code's is refused
        before anything is computed."""
        self.mechanisms = is_mechanism_version(expect_version)
        self.version = reach_version(self.mechanisms)
        if expect_version is not None and expect_version != self.version:
            raise ReachVersionError(f"FeatureConfig.reach_version {expect_version!r}: this worker's reach block is "
                                    f"{self.version!r} (ballrain/fastreach.py, ballrain/reach_features.py); the "
                                    "policy was trained on another block, update the worker's code")
        require_numba()
        self.cache_dir = cache_dir
        self.body_radius = body_radius
        self._last_fields = None
        if self.mechanisms:
            from .reach_mechanisms import uses_lowered_plate
            self._uses_lowered_plate = uses_lowered_plate

    def compute(self, tracker, pos, vel, quarter: int, ball_type: int):
        """(scaled block (N_REACH,), raw perceive vector (N_REACH,), fields key)."""
        p = np.asarray(pos, np.float64)
        if not (np.isfinite(p).all() and np.isfinite(np.asarray(vel, np.float64)).all()):
            return np.zeros(N_REACH), np.full(N_REACH, np.nan), None  # perceive refuses it; the f6 entries nan_to_num
        if self.mechanisms:
            level = tracker.level
            cached = self._last_fields
            if (cached is None or cached[0] is not level or cached[1] != level.version
                    or cached[2] != tracker.level_no or cached[3] != threading.get_ident()):
                cached = (level, level.version, tracker.level_no, threading.get_ident(),
                          _mechanism_fields(tracker, self.cache_dir, self.version))
                self._last_fields = cached
            rows, probe = cached[4]
            k = ball_type if 0 <= ball_type <= 2 else 0
            if probe is not None and k == probe[0] and self._uses_lowered_plate(tracker, p, probe[2]):
                k = probe[1]  # change fields, not the true ball type passed to perceive
            fr, key = rows[k]
        else:
            fr, key = fields_for(tracker, self.cache_dir)
        bodies = moving_bodies(tracker, p, self.body_radius)
        raw, _ = fr.perceive(p, vel, int(ball_type), int(quarter), bodies=bodies)
        return scale(raw), raw, key


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]
