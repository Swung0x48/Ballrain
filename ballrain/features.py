"""Observation features, computed entirely in Python from the raw state (docs/observations.md).

FeatureExtractor(config).compute(tracker) -> (vector float32, info dict). The layout (name, size)
of the vector is fixed by the config and returned by layout(); every block is documented in
docs/observations.md. FEATURE_VERSION changes whenever the meaning of an existing entry changes.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

from . import geometry as G
from .path import PathFollower, ReferencePath, RouteError, is_synthetic_level, load_tas_path
from .state import G_RAILS, G_STOPPER, GameTracker

FEATURE_VERSION = "f6"  # f6: optional look-ahead block (far_waypoints, ref_speed_at; off by default = the f5 layout)
# f5: edge fan referenced to the floor under the ball when airborne / in side contact
# f4: TAS reference line + continuity projection, arc waypoints, slope-robust edge fan
# f3: transforming from the ball (un)physicalize events; HS block (followers, pause)

GEOM_EPS = 0.05  # state-spec.md 3.5: geometric contact threshold on the gap
NO_CONTACT_GAP = 2.0  # value used for "no mindist"
SUPPORT_GAP = 1.0  # a floor-facing mindist within this gap is the ball's support (edge-fan reference)
SURFACE_SLOTS = ("none", "stone", "wood", "metal", "other")


@dataclass
class FeatureConfig:
    version: str = FEATURE_VERSION
    camera: str = "orient_ref"  # frame for directional features: "orient_ref" | "orient" | "world"
    pos_scale: float = 50.0  # positions are divided by this
    vel_scale: float = 20.0
    angvel_scale: float = 20.0
    absolute_position: bool = False  # include the ball's world position (level-specific policies)
    height_grid: int = 7  # G x G vertical rays around the ball (0 = off)
    height_spacing: float = 1.0
    height_clip: float = 5.0
    height_up: float = 3.0  # rays start this far above the ball centre
    fan_dirs: int = 8  # edge distance per direction (ray fan, 0 = off)
    fan_step: float = 0.25
    fan_max: float = 8.0
    fan_rule: str = "step"  # "step" (f4): drop between consecutive samples | "contact" (f3): below the contact
    fan_step_drop: float = 0.75  # "step": a sample lower than the previous one (the contact for the first) by more
    #                              than this (per fan_step; 0.75 per 0.25 = a 72 degree descent) is the edge
    fan_drop: float = 1.0  # "contact": a sample lower than the contact height by this is "off the road"
    fan_up: float = 1.0  # fan rays start this far above the ball centre
    mesh_edge: bool = True  # distance from the contact point to the walkable boundary (mesh method)
    rail_radius: float = 6.0
    path_source: str = "tas"  # "tas" (TAS line; the polyline only for levels without an asset) | "checkpoints"
    path_dir: str | None = None  # directory of the L<nn>.npz assets, repo-relative (default benchmarks/refpaths)
    path_sha256: str | None = None  # sha256 the level's asset must have (set by the learner; None = unchecked)
    branch_dir: str | None = None  # optional paired Lnn.json / Lnn.npz diversion
    path_lateral_scale: float = 10.0  # path_lateral = distance to the line / this
    path_back: float = 10.0  # projection window behind the previous arc length
    path_ahead: float = 30.0  # ... and ahead of it
    path_margin: float = 20.0  # the window may leave the current sector's arc range by this much
    path_reacquire: float = 5.0  # switch to the sector-wide nearest point when it is closer by more than this
    waypoints: int = 3
    waypoint_spacing: float = 5.0  # waypoint j (1..waypoints) is the line point j * spacing units of arc ahead
    # f6 look-ahead (appended at the end of the vector, so a policy can be extended with zero weights):
    # far_waypoints: line points this many units of arc ahead, relative to the ball (frame of the directional features);
    # ref_speed_at: the reference trajectory's speed along the line at arc s + offset (units / s / vel_scale; 0 without
    # reference times, e.g. the checkpoint polyline)
    far_waypoints: tuple = ()
    ref_speed_at: tuple = ()
    bodies: int = 4
    body_radius: float = 60.0
    dynamic_geometry: bool = True  # include moving bodies in the height grid / fan
    counters: bool = True
    hs_block: bool = True  # small score balls following, pause, frames since the last checkpoint
    follower_sources: int = 4  # per-instance follower counts (largest first)
    checkpoint_frames_scale: float = 10000.0
    clip: float = 10.0  # final clip of every entry

    def to_dict(self):
        d = asdict(self)
        if self.branch_dir is None:
            del d["branch_dir"]  # preserve serialized configurations when off
        return d


# ------------------------------------------------------------------------------- geometry cache

class LevelGeometry:
    """Static collision triangles (stoppers excluded), walkable boundary edges, rail triangles."""

    def __init__(self, tracker: GameTracker):
        lvl = tracker.level
        items, rail_items = [], []
        for ck in lvl.static_entities():
            for m in lvl.meshes_of(ck):
                V = G.ck_to_world(m.vertices, m.matrix)
                items.append((ck, V, m.faces))
                if lvl.in_group(ck, G_RAILS):
                    rail_items.append((ck, V, m.faces))
        self.tris = G.TriangleSet.from_meshes(items)
        rails = {ck for ck, _, _ in rail_items}
        self.edges = G.BoundaryEdges(self.tris, owner_filter=lambda o: ~np.isin(o, list(rails)))
        self.rails = G.TriangleSet.from_meshes(rail_items)
        self.key = lvl.static_key()


_GEOM_CACHE: "OrderedDict[str, LevelGeometry]" = OrderedDict()
_KEY_OF: dict = {}  # (id(LevelData), version) -> content key


def level_geometry(tracker: GameTracker, cache_size: int = 4) -> LevelGeometry:
    lvl = tracker.level
    vk = (id(lvl), lvl.version)
    hit = _KEY_OF.get(vk)
    # the entry holds the object itself: a freed LevelData's id can be reused by a new one with the same version,
    # and an id-only key would then hand back the old level's content key (and its geometry)
    if hit is not None and hit[0] is lvl:
        key = hit[1]
    else:
        key = lvl.static_key()
        if len(_KEY_OF) > 16:
            _KEY_OF.clear()
        _KEY_OF[vk] = (lvl, key)
    g = _GEOM_CACHE.get(key)
    if g is None:
        g = LevelGeometry(tracker)
        _GEOM_CACHE[key] = g
        while len(_GEOM_CACHE) > cache_size:
            _GEOM_CACHE.popitem(last=False)
    else:
        _GEOM_CACHE.move_to_end(key)
    return g


_MESH_BOUND: dict = {}  # id(mesh) -> (mesh, max |local vertex|)


def _mesh_bound(m) -> float:
    e = _MESH_BOUND.get(id(m))
    if e is None or e[0] is not m:
        if len(_MESH_BOUND) > 4096:
            _MESH_BOUND.clear()
        v = np.asarray(m.vertices, np.float64)
        e = (m, float(np.sqrt((v * v).sum(1).max())) if len(v) else 0.0)
        _MESH_BOUND[id(m)] = e
    return e[1]


def dynamic_triangles(tracker: GameTracker, center, radius: float, xz_box=None) -> G.TriangleSet:
    """Moving bodies of the BODY section near center (body position within radius), posed at the last PSI.
    xz_box = (lo (2,), hi (2,)): keep only what can be hit by a vertical ray inside that (x, z) box (the footprint
    of the terrain rays): a mesh is skipped when its bounding sphere (|local vertex| + |shift| around the body
    position; rotations keep lengths) misses the box, a triangle when its (x, z) bounds miss it. Exact for rays
    inside the box; brute-force ray casts against ~1000 triangles of nearby platforms cost 10-35 ms per frame."""
    obs = tracker.obs
    items = []
    if xz_box is not None:
        blo = np.asarray(xz_box[0], np.float64) - 1e-6
        bhi = np.asarray(xz_box[1], np.float64) + 1e-6
    if obs is not None:
        lvl = tracker.level
        for b in obs.body:
            ck = int(b["ckid"])
            if lvl.in_group(ck, G_STOPPER) or not lvl.is_physicalized(ck):
                continue
            bp = np.asarray(b["pos"], np.float64)
            if np.linalg.norm(bp - center) > radius:
                continue
            for m in lvl.meshes_of(ck):
                if xz_box is not None:
                    r = _mesh_bound(m) + float(np.linalg.norm(np.asarray(b["shift"], np.float64)))
                    if (bp[0] + r < blo[0] or bp[0] - r > bhi[0] or bp[2] + r < blo[1] or bp[2] - r > bhi[1]):
                        continue
                V = G.body_to_world(m.vertices, b["pos"], b["q"], b["q_cfo"], b["shift"])
                F = np.asarray(m.faces, np.int64).reshape(-1, 3)
                if xz_box is not None and len(F):
                    X, Z = V[F, 0], V[F, 2]
                    keep = ((X.max(1) >= blo[0]) & (X.min(1) <= bhi[0]) & (Z.max(1) >= blo[1])
                            & (Z.min(1) <= bhi[1]))
                    F = F[keep]
                items.append((ck, V, F))
    return G.TriangleSet.from_meshes(items, grid_cell=None)


IVP_ST_BALL = 3  # synapse status of a sphere (IVP_ST_POINT 0, EDGE 1, TRIANGLE 2, BALL 3)


def _closest_on(tracker: GameTracker, ck: int, p, r: float):
    """Closest point to p on the collision mesh of entity ck (static: the level geometry; moving: its BODY pose),
    or None when nothing is within r."""
    lvl = tracker.level
    if lvl.is_fixed(ck):
        geo = level_geometry(tracker)
        idx = geo.tris.candidates(p, r)
        idx = idx[geo.tris.owner[idx] == ck] if len(idx) else idx
        if not len(idx):
            return None
        A, B, C = geo.tris.A[idx], geo.tris.B[idx], geo.tris.C[idx]
    else:
        body = next((b for b in tracker.obs.body if int(b["ckid"]) == ck), None) if tracker.obs is not None else None
        if body is None:
            return None
        items = [(ck, G.body_to_world(m.vertices, body["pos"], body["q"], body["q_cfo"], body["shift"]), m.faces)
                 for m in lvl.meshes_of(ck)]
        ts = G.TriangleSet.from_meshes(items, grid_cell=None)
        if not len(ts):
            return None
        A, B, C = ts.A, ts.B, ts.C
    Q, _ = G.closest_point_triangles(p, A, B, C)
    d = np.linalg.norm(Q - p, axis=1)
    k = int(np.argmin(d))
    return Q[k] if d[k] <= r else None


def mind_normals(tracker: GameTracker, pos, mind) -> np.ndarray:
    """Contact normals of the MIND entries (n, 3), oriented from the partner's surface towards the ball.
    The mindist plane has that orientation whenever the ball's synapse is a sphere (self_status 3: wood and stone
    balls). For the paper ball (a polyhedron: point / edge / triangle synapses) the plane's sign depends on the
    synapse order inside the mindist and is reversed in most of its entries (checked against the geometry: SR11 351 of 481, SR06 107 of 188);
    those normals are oriented with the geometry: n . (ball centre - closest point of the partner) > 0."""
    N = np.array(mind["plane"][:, :3], np.float64)
    fix = np.flatnonzero(mind["self_status"] != IVP_ST_BALL)
    if not len(fix):
        return N
    p = np.asarray(pos, np.float64)
    near: dict = {}
    for j in fix:
        ck = int(mind["partner"][j])
        if ck not in near:
            near[ck] = _closest_on(tracker, ck, p, float(mind["gap"][j]) + 4.0)
        q = near[ck]
        if q is not None and float(N[j] @ (p - q)) < 0:
            N[j] = -N[j]
    return N


# ------------------------------------------------------------------------------------- frames

def yaw_basis(matrix43) -> tuple[np.ndarray, np.ndarray]:
    """(right, forward) horizontal unit vectors of a CK matrix (rows: x axis, y axis, z axis, pos)."""
    m = np.asarray(matrix43, np.float64)
    f = m[2].copy()
    f[1] = 0
    n = np.linalg.norm(f)
    f = f / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    r = np.array([f[2], 0.0, -f[0]])
    return r, f


def to_frame(v, right, fwd) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return np.stack([v[..., 0] * right[0] + v[..., 2] * right[2], v[..., 1],
                     v[..., 0] * fwd[0] + v[..., 2] * fwd[2]], -1)


def yaw_of(matrix43) -> float:
    m = np.asarray(matrix43, np.float64)
    return math.atan2(m[2, 0], m[2, 2])


# --------------------------------------------------------------------------------- extractor

class FeatureExtractor:
    def __init__(self, config: FeatureConfig | None = None):
        self.cfg = config or FeatureConfig()
        self._layout = self._make_layout()
        self.size = sum(n for _, n in self._layout)
        self.path: ReferencePath | None = None
        self.follower: PathFollower | None = None
        self.branch = None
        self._path_key = None

    # layout -------------------------------------------------------------------------------
    def _make_layout(self):
        c = self.cfg
        L = [("valid", 1), ("ball_type", 3), ("vel_cam", 3), ("vel_path", 3), ("speed", 1), ("angvel", 3),
             ("cam_yaw_ref", 2), ("cam_yaw", 2), ("cam_rotating", 1)]
        if c.absolute_position:
            L.append(("pos_world", 3))
        L += [("geom_contact", 1), ("fric_contact", 1), ("n_fric", 1), ("n_mind", 1), ("min_gap", 1),
              ("normal_cam", 3), ("surface", len(SURFACE_SLOTS)), ("partner_moving", 1), ("partner_rail", 1),
              ("rel_vel_partner", 3)]
        if c.mesh_edge:
            L.append(("edge_mesh", 1))
        if c.fan_dirs:
            L.append(("edge_fan", c.fan_dirs))
        L += [("rail_dist", 1), ("rail_dir", 2)]
        if c.height_grid:
            L += [("heights", c.height_grid ** 2), ("height_hit", c.height_grid ** 2)]
        L += [("path_progress", 1), ("path_remaining", 1), ("path_lateral", 1), ("path_tangent_cam", 2),
              ("waypoints", 3 * c.waypoints)]
        if c.bodies:
            L += [("bodies", 7 * c.bodies)]
        if c.counters:
            L += [("lives", 1), ("points", 1), ("sr_s", 1), ("sector", 1), ("transforming", 1), ("ball_nav", 1),
                  ("depth_margin", 1)]
        if c.hs_block:
            L += [("followers", 1), ("follower_sources", c.follower_sources), ("follower_near", 1), ("paused", 1),
                  ("since_checkpoint", 1)]
        if c.far_waypoints:
            L.append(("far_waypoints", 3 * len(c.far_waypoints)))
        if c.ref_speed_at:
            L.append(("ref_speed", len(c.ref_speed_at)))
        return L

    def layout(self) -> list:
        return list(self._layout)

    def slices(self) -> dict:
        out, o = {}, 0
        for n, k in self._layout:
            out[n] = slice(o, o + k)
            o += k
        return out

    # helpers ------------------------------------------------------------------------------
    def reference_path(self, tracker: GameTracker) -> ReferencePath | None:
        """TAS line of the level when path_source is "tas" and the asset matches the level (start and
        checkpoint positions), else the checkpoint polyline. Re-chosen when the level data changes; the
        follower (projection state) is reset when the path object changes."""
        key = (id(tracker), tracker.level.version, tracker.arrays.rows("Checkpoints"))
        if key != self._path_key:
            path = None
            tas_pending = False
            if self.cfg.path_source == "tas" and not is_synthetic_level(tracker):
                path = load_tas_path(tracker.level_no, tracker, self.cfg.path_dir, strict=True,
                                     sha256=self.cfg.path_sha256)
                # the level has an asset but its data is still incomplete: no line yet (never the polyline)
                tas_pending = path is None and self._needs_asset(tracker)
            if path is None and not tas_pending:
                path = ReferencePath.from_tracker(tracker)
            new_tracker = self._path_key is None or key[0] != self._path_key[0]
            same_line = (path is not None and self.path is not None and path.source == self.path.source
                         and path.P.shape == self.path.P.shape and np.array_equal(path.P, self.path.P))
            if new_tracker or not same_line:  # keep the projection state when only the level data version moved
                self.follower = None if path is None else PathFollower(
                    path, back=self.cfg.path_back, ahead=self.cfg.path_ahead, margin=self.cfg.path_margin,
                    reacquire=self.cfg.path_reacquire)
                self.branch = None
                if self.cfg.branch_dir and path is not None and path.source == "tas":
                    from .branch import load_branch
                    self.branch = load_branch(path, tracker.level_no, self.cfg.branch_dir, self.cfg.path_dir,
                                              **self.follower.kw)
            self.path = path
            self._path_key = key
        return self.path

    def _has_asset(self, level) -> bool:
        from .path import asset_path, resolve_path_dir
        return asset_path(level, resolve_path_dir(self.cfg.path_dir)).is_file()

    def _needs_asset(self, tracker: GameTracker) -> bool:
        return self.cfg.path_source == "tas" and not is_synthetic_level(tracker) and self._has_asset(tracker.level_no)

    def check_route(self, tracker: GameTracker) -> None:
        """RouteError when the ball is in play while the line a "tas" configuration needs is not loaded."""
        if (self.path is None or self.path.source != "tas") and self._needs_asset(tracker):
            raise RouteError(f"level {tracker.level_no}: ball in play without its reference line "
                             f"(path_dir {self.cfg.path_dir!r}); refusing to run without the route")

    def reset_episode(self) -> None:
        """Forget the projection state (call at every episode start: new process, respawn, TAS hand-over)."""
        if self.follower is not None:
            self.follower.reset()
        if self.branch is not None:
            self.branch.reset()

    def frame(self, tracker: GameTracker):
        obs = tracker.obs
        cams = obs.cams if obs is not None else None
        if self.cfg.camera == "world" or cams is None:
            return np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
        k = 0 if self.cfg.camera == "orient_ref" else 1
        return yaw_basis(cams["cams"][k]["matrix"])

    # main ---------------------------------------------------------------------------------
    def compute(self, tracker: GameTracker):
        c = self.cfg
        obs = tracker.obs
        F = OrderedDict((n, np.zeros(k)) for n, k in self._layout)
        info = {}
        right, fwd = self.frame(tracker)
        ball = obs.ball if obs is not None else None
        have_ball = ball is not None and bool(ball["found"]) and int(ball["type"]) in (0, 1, 2)
        valid = have_ball and not tracker.transforming
        F["valid"][0] = float(valid)
        bt = tracker.ball_type()
        if bt in (0, 1, 2):
            F["ball_type"][bt] = 1
        cams = obs.cams if obs is not None else None
        if cams is not None:
            y0, y1 = yaw_of(cams["cams"][0]["matrix"]), yaw_of(cams["cams"][1]["matrix"])
            F["cam_yaw_ref"][:] = (math.cos(y0), math.sin(y0))
            F["cam_yaw"][:] = (math.cos(y1), math.sin(y1))
            F["cam_rotating"][0] = float(abs(math.remainder(y0 - y1, 2 * math.pi)) > 1e-3)
        path = self.reference_path(tracker)
        if valid:
            self.check_route(tracker)
        if have_ball:
            pos = np.asarray(ball["pos"], np.float64)
            vel = np.asarray(ball["speed"], np.float64) + np.asarray(ball["speed_change"], np.float64)
            info["pos"], info["vel"] = pos, vel
            F["vel_cam"][:] = to_frame(vel, right, fwd) / c.vel_scale
            F["speed"][0] = np.linalg.norm(vel) / c.vel_scale
            F["angvel"][:] = (np.asarray(ball["rot_speed"]) + np.asarray(ball["rot_speed_change"])) / c.angvel_scale
            if c.absolute_position:
                F["pos_world"][:] = pos / c.pos_scale
            radius = float(ball["extra_radius"]) or 2.0
            self._contacts(tracker, F, info, pos, vel, radius, right, fwd)
            info["supported"] = "support_point" in info  # floor-facing contact within SUPPORT_GAP (SR reward hold)
            self._terrain(tracker, F, info, pos, radius, right, fwd)
            if path is not None and (self.branch is None or valid):
                if self.branch is None:
                    pr = self.follower.update(pos, tracker.sector)
                    local_s = pr["s"]
                else:
                    path, pr = self.branch.update(pos, tracker.sector, bt)
                    local_s = pr["local_s"]
                info["path"] = pr
                info["path_source"] = "branch" if pr.get("branch") else path.source
                F["path_progress"][0] = pr["s"] / max(self.path.length, 1e-9)
                F["path_remaining"][0] = (path.length - local_s) / (20 * c.pos_scale)
                F["path_lateral"][0] = pr["dist"] / c.path_lateral_scale
                tc = to_frame(pr["tangent"], right, fwd)
                F["path_tangent_cam"][:] = (tc[0], tc[2])
                pr_r = np.array([pr["tangent"][2], 0, -pr["tangent"][0]])
                F["vel_path"][:] = to_frame(vel, pr_r, pr["tangent"]) / c.vel_scale
                if c.waypoints:
                    wp = path.points_ahead(local_s, c.waypoints, c.waypoint_spacing) - pos
                    F["waypoints"][:] = (to_frame(wp, right, fwd) / c.pos_scale).ravel()
                if c.far_waypoints:
                    fw = path.points_at(local_s, c.far_waypoints) - pos
                    F["far_waypoints"][:] = (to_frame(fw, right, fwd) / c.pos_scale).ravel()
                if c.ref_speed_at:
                    for j, o in enumerate(c.ref_speed_at):
                        v = path.speed_at(local_s + float(o))
                        F["ref_speed"][j] = (v or 0.0) / c.vel_scale
            if c.bodies:
                self._bodies(tracker, F, pos, vel, right, fwd)
        if c.counters:
            F["lives"][0] = tracker.lives / 10
            F["points"][0] = tracker.points / 1000
            sr = tracker.sr_ms
            F["sr_s"][0] = (sr or 0.0) / 100000.0
            npath = path.n_sectors if path is not None else 1
            F["sector"][0] = tracker.sector / max(npath, 1)
            F["transforming"][0] = float(tracker.transforming)
            F["ball_nav"][0] = float(tracker.flags.ball_nav)
            if have_ball:
                F["depth_margin"][0] = self._depth_margin(tracker, info["pos"]) / c.pos_scale
        if c.hs_block:
            n = tracker.followers()
            info["followers"] = n
            F["followers"][0] = n / 6
            st = tracker.follower_state()
            per = sorted(st["per_source"].values(), reverse=True)[:c.follower_sources]
            F["follower_sources"][:len(per)] = np.asarray(per, np.float64) / 6
            if have_ball and st["positions"]:
                d = tracker.nearest_follower(info["pos"])
                info["follower_near"] = d
                F["follower_near"][0] = min(d, c.pos_scale) / c.pos_scale
            F["paused"][0] = float(tracker.flags.paused)
            F["since_checkpoint"][0] = tracker.frames_since_checkpoint() / c.checkpoint_frames_scale
        vec = np.concatenate(list(F.values())).astype(np.float32)
        np.clip(vec, -c.clip, c.clip, out=vec)
        np.nan_to_num(vec, copy=False, nan=0.0, posinf=c.clip, neginf=-c.clip)
        return vec, info

    # blocks -------------------------------------------------------------------------------
    def _contacts(self, tracker, F, info, pos, vel, radius, right, fwd):
        obs, lvl = tracker.obs, tracker.level
        mind, fric = obs.mind, obs.fric
        F["n_mind"][0] = len(mind) / 10
        F["n_fric"][0] = len(fric) / 4
        F["fric_contact"][0] = float(len(fric) > 0)
        partner = None
        if len(mind):
            N = mind_normals(tracker, pos, mind)
            info["mind_normals"] = N
            k = int(np.argmin(mind["gap"]))
            gap = float(mind["gap"][k])
            normal = N[k]
            F["min_gap"][0] = min(gap, NO_CONTACT_GAP)
            F["geom_contact"][0] = float(gap <= GEOM_EPS)
            F["normal_cam"][:] = to_frame(normal, right, fwd)
            partner = int(mind["partner"][k])
            info.update(gap=gap, normal=normal, contact_point=pos - normal * (radius + gap))
            # floor support (edge fan reference, f5): the nearest mindist whose plane faces up (n_y >= 0.5)
            up = np.flatnonzero(N[:, 1] >= G.UP_WALKABLE)
            if len(up):
                j = int(up[np.argmin(mind["gap"][up])])
                gj = float(mind["gap"][j])
                if gj <= SUPPORT_GAP:
                    info.update(support_gap=gj, support_point=pos - N[j] * (radius + gj))
        else:
            F["min_gap"][0] = NO_CONTACT_GAP
        if len(fric):
            partner = int(fric["partner"][int(np.argmin(fric["gap"]))])
        info["partner"] = partner
        slot = "none"
        if partner is not None:
            cid = lvl.contact_id(partner)
            slot = {1: "stone", 2: "wood", 3: "metal"}.get(cid, "other")
            moving = not lvl.is_fixed(partner)
            F["partner_moving"][0] = float(moving)
            F["partner_rail"][0] = float(lvl.in_group(partner, G_RAILS))
            if moving:
                for b in obs.body:
                    if int(b["ckid"]) == partner:
                        rel = vel - np.asarray(b["speed"], np.float64)
                        F["rel_vel_partner"][:] = to_frame(rel, right, fwd) / self.cfg.vel_scale
                        break
        F["surface"][SURFACE_SLOTS.index(slot)] = 1
        info["surface"] = slot

    def _terrain(self, tracker, F, info, pos, radius, right, fwd):
        c = self.cfg
        geo = level_geometry(tracker)
        bottom = pos[1] - radius
        contact = info.get("contact_point")
        # edge-fan reference (f5): the floor support (a floor-facing mindist within SUPPORT_GAP: base = its contact
        # point, the same as f4 whenever the nearest contact faces up); otherwise (airborne, or only side / overhead
        # contacts: walls, pushable boxes, prop balls) the floor under the ball centre (base = the first hit of a
        # vertical ray at the centre, from fan_up above it), or no reference when there is none (over the void)
        support = info.get("support_point")
        # every ray of this frame first (centre, height grid, fan), so that moving bodies are culled to the footprint
        Pg = Pf = None
        if c.height_grid:
            g = c.height_grid
            o = (np.arange(g) - (g - 1) / 2) * c.height_spacing
            gx, gz = np.meshgrid(o, o, indexing="ij")  # gx: right, gz: forward
            off = gx[..., None] * right + gz[..., None] * fwd
            Pg = (pos + off).reshape(-1, 3)
            Pg[:, 1] = pos[1] + c.height_up
        if c.fan_dirs:
            ang = np.arange(c.fan_dirs) * (2 * math.pi / c.fan_dirs)
            dirs = np.cos(ang)[:, None] * fwd + np.sin(ang)[:, None] * right  # 0 = forward, then clockwise
            steps = np.arange(c.fan_step, c.fan_max + 1e-9, c.fan_step)
            base = support if support is not None else pos
            Pf = (base + steps[None, :, None] * dirs[:, None, :]).reshape(-1, 3)
            Pf[:, 1] = pos[1] + c.fan_up
        dyn = None
        allp = [q for q in (Pg, Pf) if q is not None] + [pos[None, :]]  # pos: the centre ray of the fan reference
        if c.dynamic_geometry and allp:
            xz = np.concatenate(allp)[:, [0, 2]]
            dyn = dynamic_triangles(tracker, pos, c.body_radius, (xz.min(0), xz.max(0)))

        def rays(P):
            y, t = geo.tris.ray_down(P)
            if dyn is not None and len(dyn):
                y2, t2 = dyn.ray_down(P)
                y, _, _ = G.merge_ray(y, t, y2, t2)
            return y

        if Pg is not None:
            y = rays(Pg)
            hit = ~np.isnan(y)
            F["heights"][:] = np.where(hit, np.clip(y - bottom, -c.height_clip, c.height_clip), -c.height_clip) / c.height_clip
            F["height_hit"][:] = hit
        if Pf is not None:
            if support is not None:
                h_ref, info["fan_ref"] = float(support[1]), "support"
            else:
                yc = rays(np.array([[pos[0], pos[1] + c.fan_up, pos[2]]]))[0]
                h_ref, info["fan_ref"] = (float(yc), "below") if not np.isnan(yc) else (float("nan"), "none")
            y = rays(Pf).reshape(c.fan_dirs, len(steps))
            first = fan_edges(y, h_ref, steps, c)
            F["edge_fan"][:] = first / c.fan_max
            info["edge_fan"] = first
        if c.mesh_edge:
            q = contact if contact is not None else pos - np.array([0, radius, 0])
            d = geo.edges.distance(q, 12.0)
            F["edge_mesh"][0] = min(d, 12.0) / 12.0
            info["edge_mesh"] = d
        rc = geo.rails.closest(pos, c.rail_radius) if len(geo.rails) else None
        if rc is not None:
            F["rail_dist"][0] = (rc["d"] - radius) / c.rail_radius
            dv = to_frame(rc["q"] - pos, right, fwd)
            h = math.hypot(dv[0], dv[2])
            if h > 1e-9:
                F["rail_dir"][:] = (dv[0] / h, dv[2] / h)
        else:
            F["rail_dist"][0] = 1.0

    def _bodies(self, tracker, F, pos, vel, right, fwd):
        c = self.cfg
        lvl = tracker.level
        rows = []
        for b in tracker.obs.body:
            ck = int(b["ckid"])
            if lvl.in_group(ck, G_STOPPER):
                continue
            d = np.asarray(b["pos"], np.float64) - pos
            dist = float(np.linalg.norm(d))
            if dist <= c.body_radius:
                rows.append((dist, d, np.asarray(b["speed"], np.float64) - vel))
        rows.sort(key=lambda r: r[0])
        out = F["bodies"].reshape(c.bodies, 7)
        for i, (dist, d, rv) in enumerate(rows[:c.bodies]):
            out[i, 0] = 1
            out[i, 1:4] = to_frame(d, right, fwd) / c.pos_scale
            out[i, 4:7] = to_frame(rv, right, fwd) / c.vel_scale

    def _depth_margin(self, tracker, pos) -> float:
        """Height of the ball centre above the highest death box under it (state-spec.md 2.7)."""
        best = None
        for _ck, lo, hi in tracker.level.depth:
            if lo[0] <= pos[0] <= hi[0] and lo[2] <= pos[2] <= hi[2]:
                m = pos[1] - float(hi[1])
                best = m if best is None else min(best, m)
        return best if best is not None else 20 * self.cfg.pos_scale


def fan_edges(y, h_ref: float, steps, cfg: FeatureConfig) -> np.ndarray:
    """Edge distance per fan direction from the sampled heights y (dirs x steps, nan = no hit).
    "step" (f4): the first sample with no hit, or lower than the previous sample (the reference height h_ref for
    the first: the support height, or the floor under the ball centre; nan = no reference, only misses count) by
    more than fan_step_drop. Slope-robust: a slope descends by tan(angle) * fan_step per sample
    (0.08 per 0.25 at 18 degrees), a cliff by >= 1 at once (docs/observations.md, edge fan histogram).
    "contact" (f3): no hit, or lower than h_ref by more than fan_drop (reads long gentle descents as edges).
    Directions without an edge get fan_max."""
    y = np.asarray(y, np.float64)
    if cfg.fan_rule == "contact":
        drop = np.isnan(y) | (y < h_ref - cfg.fan_drop)
    elif cfg.fan_rule == "step":
        prev = np.concatenate([np.full((len(y), 1), h_ref), y[:, :-1]], 1)
        with np.errstate(invalid="ignore"):
            drop = np.isnan(y) | (prev - y > cfg.fan_step_drop)
    else:
        raise ValueError(f"unknown fan_rule {cfg.fan_rule!r}")
    return np.where(drop.any(1), np.asarray(steps)[np.argmax(drop, 1)], cfg.fan_max)


def feature_names(extractor: FeatureExtractor) -> list:
    names = []
    for n, k in extractor.layout():
        names += [n] if k == 1 else [f"{n}[{i}]" for i in range(k)]
    return names
