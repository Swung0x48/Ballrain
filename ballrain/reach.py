"""Analytic reachability estimate: where can the ball get to in the next ~1-3 s under the best key inputs?

From the current ball state alone (PSI position, velocity, ball type, camera quarter) and the static level geometry,
roll out a small set of key plans with the measured dynamics and tell, for each plan, where the ball lands / ends
and whether it falls into a death zone. Derived outputs: the support point per key direction, the furthest arc
length along a reference path that a surviving plan reaches, "doomed" (no plan survives the horizon), and
"can still make it" flags for the next gap and the next turn of the reference path.

Measured dynamics (tas-v1, retail 2002 physics; docs in the report of this module, numbers repeated here):

* Time base. One 132 fps frame = one PSI = 1/66 s of *physics* time = 1/132 s of game time (the physics clock runs
  at twice game time, state-spec.md 1). Every velocity here is in units per physics second, exactly what BALL.speed
  holds; multiply by 2 for units per game second. Positions follow p[k+1] = p[k] + v[k] / 66 (explicit Euler).
* Air (no contact), exact to 1e-6 per PSI for wood and stone on 30k / 2.4k samples:
      v[k+1] = (1 - c_lin / 66) v[k] + (F / m) u + (0, -20 / 66, 0)
  u = summed horizontal unit pushes of the held arrow keys (two keys = two independent unit forces, |u| = sqrt 2),
  F = Force Value, m = mass, c_lin = linear damping of the Physicalize_GameBall row. Gravity 20 units / phys-s^2.
  Air control exists and is the same force as on the ground.
* Ground (rolling, no slip; omega * r / |v| median 1.00-1.02): a solid sphere (I = 2/5 m r^2) rolling on a plane
      v[k+1] = (1 - c_roll / 66) v[k] + (F / (1.4 m)) u + (5/7) g_tangential / 66,
      c_roll = (c_lin + 0.4 c_rot) / 1.4
  Flat-floor fit (wood, 3.7k samples): F/(1.4m) 0.159 vs 0.1617, c_roll/66 0.0101 vs 0.0102; residual sd 0.025 per
  PSI, no speed trend up to |v| = 18. No angular-speed cap was seen; the flat-road plateau is the damping limit
  (F / (1.4 m)) / (c_roll / 66) = 15.9 (one key), 22.5 (two keys) units / phys-s for wood.

Model (what is captured and what is not):

* Point-mass rollouts, per plan, in steps of `step_psi` PSIs, each step integrated in closed form (constant input,
  linear damping). Plans: the 9 constant key sets and the 72 two-segment plans per switch time in `switch_s`
  (225 by default). Ground / air switch by a sphere-vs-heightfield support test (SupportMap): the ball centre's rest
  height is max over surface samples within the radius of h + sqrt(r^2 - d^2), which also handles rails (a vertical
  ray between two rails misses, a sphere does not).
* Ground: rolling dynamics above; the tangential gravity comes from the support normal (face normal of the touched
  face, or sample -> centre over an edge, averaged over near-tied contacts), so slopes and banking both act. Leaves
  the ground when the support falls away faster than a ballistic drop.
* Hard barriers (SupportMap.query): walkable samples hold the ball when they keep it at its height or lift it by
  no more than their face slope over the move (floors, ramps whatever their total rise); an edge ahead of the
  motion h above the ball bottom is mounted only at a speed into it of at least mount_speed(h) (speed-dependent
  edge mount: a rolling sphere pivoting about the edge with its angular momentum about the edge conserved; never at
  h >= MOUNT_CAP 1.95; a MODEL ESTIMATE from the user's correction, not validated in game); edges beside the ball
  (two rails) hold it. Every other sample the sphere penetrates (as a solid column over its steep faces) is a wall:
  the motion into it is removed (inelastic slide), and while pressed against it steep edges cannot lift the ball.
  probe_directions walks the support in given directions at a given speed and reports the first barrier, its height
  and the speed it would need, or the first drop.
* Not modelled: wall-assisted turning force, bounces (landing kills the vertical speed), moving bodies (bridges,
  boxes, flippers, the airship platform), camera turns, more than one key switch per plan.
* Death: the centre within a death box's x-z footprint and below its top + `death_margin`, or falling with nothing
  under it. The small DepthTest boxes in the DEPT list (a few units wide, at road height on L1) are ignored: only
  boxes with a horizontal extent of at least `death_min_extent` count.
* Paper ball: unsupported (a 315-vertex polyhedron that bounces while it rolls; its air fit is also anomalous).
  Stone: parameterised by its Physicalize row (rot damping 0.1, as BallanceMMO read it), air dynamics measured
  exact, ground not validated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import tas

PSI_HZ = 66.0  # PSIs per physics second
GAME_PSI_HZ = 132.0  # PSIs per game second (132 fps: one PSI per frame)
GRAVITY = 20.0  # units / physics-s^2
ROLL_INERTIA = 1.4  # 1 + I / (m r^2) for a solid sphere
UP_FACE = 0.5  # a face with n_y at least this is a face the ball can sit on (its normal is the contact normal)
# Edge mount (user correction 2026-11-17; MODEL ESTIMATE, not validated in game): a rolling solid sphere that runs into
# an edge h above its bottom pivots about the edge, keeping its angular momentum about it; the rotational energy left
# must lift the centre by h. v_min = sqrt(g h / (0.7 f^2)), f = 1 - 5 h / (7 r), in physics-second units. Steps of at
# least MOUNT_CAP are never mounted; steps up to MOUNT_FREE (floor seams, sampling noise) always are.
MOUNT_CAP = 1.95
MOUNT_FREE = 0.1


def mount_speed(h, r: float = 2.0, g: float = GRAVITY):
    """Minimum speed into an edge h above the ball bottom for a rolling ball to mount it (inf at or above
    MOUNT_CAP, 0 up to MOUNT_FREE). Units / physics second."""
    h = np.asarray(h, np.float64)
    f = 1.0 - 5.0 * h / (7.0 * r)
    v = np.sqrt(g * np.maximum(h, 0.0) / (0.7 * np.maximum(f, 1e-6) ** 2))
    return np.where(h <= MOUNT_FREE, 0.0, np.where((h >= MOUNT_CAP) | (h >= r), np.inf, v))


@dataclass(frozen=True)
class BallParams:
    name: str
    force: float
    mass: float
    lin_damp: float
    rot_damp: float
    radius: float = 2.0
    supported: bool = True
    note: str = ""

    @property
    def air_accel(self) -> float:  # per PSI, per unit key push
        return self.force / self.mass

    @property
    def ground_accel(self) -> float:
        return self.force / (ROLL_INERTIA * self.mass)

    @property
    def air_lambda(self) -> float:  # velocity factor per PSI
        return 1.0 - self.lin_damp / PSI_HZ

    @property
    def ground_lambda(self) -> float:
        return 1.0 - (self.lin_damp + (ROLL_INERTIA - 1.0) * self.rot_damp) / ROLL_INERTIA / PSI_HZ

    def top_speed(self, keys: int = 1, ground: bool = True) -> float:
        """Damping-limited plateau speed (units / phys-s) on flat ground (or in the air) with 1 or 2 keys held."""
        a = (self.ground_accel if ground else self.air_accel) * math.sqrt(keys)
        lam = self.ground_lambda if ground else self.air_lambda
        return a / (1.0 - lam)


# Physicalize_GameBall (docs/state-spec.md 2.1); rot damping from the OnPhysicalize records (0.1 for wood and paper)
BALLS = {
    0: BallParams("wood", 0.43, 1.9, 0.9, 0.1),
    1: BallParams("paper", 0.065, 0.2, 1.5, 0.1, radius=2.004, supported=False,
                  note="polyhedron, bounces while rolling: not modelled"),
    2: BallParams("stone", 0.92, 10.0, 0.3, 0.1, note="rot damping 0.1 per the BallanceMMO source reading; ground not validated"),
}

# the 8 arrow combinations in env.ARROWS9 order (without the empty action)
KEY_COMBOS = (tas.KEY_UP, tas.KEY_DOWN, tas.KEY_LEFT, tas.KEY_RIGHT, tas.KEY_UP | tas.KEY_LEFT,
              tas.KEY_UP | tas.KEY_RIGHT, tas.KEY_DOWN | tas.KEY_LEFT, tas.KEY_DOWN | tas.KEY_RIGHT)


def key_push(keys, quarter) -> np.ndarray:
    """World (x, z) push of arrow bits under a Cam_OrientRef quarter turn (env.key_push_dirs at a quarter turn: UP
    along the z row, RIGHT along the x row; two keys are summed). keys and quarter broadcast; returns (..., 2)."""
    keys = np.asarray(keys, np.int64)
    yaw = np.asarray(quarter, np.float64) * (np.pi / 2)
    f = np.stack([np.sin(yaw), np.cos(yaw)], -1)
    r = np.stack([np.cos(yaw), -np.sin(yaw)], -1)
    b = lambda bit: ((keys & bit) != 0).astype(np.float64)[..., None]  # noqa: E731
    return (b(tas.KEY_UP) - b(tas.KEY_DOWN)) * f + (b(tas.KEY_RIGHT) - b(tas.KEY_LEFT)) * r


# ----------------------------------------------------------------------------------------------- support map

class SupportMap:
    """Layered x-z raster of surface samples for sphere support / wall queries.

    Every static triangle is sampled (its edges every `edge_step`, its interior every `fill_step`). Each node of a
    `res` grid keeps `layers` walkable samples (faces with n_y >= UP_FACE): the highest of each height cluster
    (samples within `layer_gap` below the cluster's top), with its exact (x, z) and upward face normal; and
    `wall_layers` wall columns: runs of steep-face samples with no vertical gap over `wall_gap`, stored as the run's
    top sample with HB = the run's foot (a solid column HB..H that blocks but never supports). Walkable samples are
    points (HB = H).
    Queries look at the nodes within the ball radius and treat each stored sample as a point the sphere can rest on
    or bump into. Storage is block-sparse (tiles of `tile` x `tile` nodes, only tiles that hold samples).
    """

    FIELDS = ("H", "X", "Z", "NX", "NY", "NZ", "HB")  # HB: foot of a wall column (HB..H); = H for a walkable point

    def __init__(self, A, B, C, res: float = 0.5, layers: int = 3, layer_gap: float = 4.0,
                 edge_step: float = 0.2, fill_step: float = 0.4, death_boxes=(),
                 tile: int = 16, wall_layers: int = 2, wall_gap: float = 1.0):
        A, B, C = (np.asarray(x, np.float64).reshape(-1, 3) for x in (A, B, C))
        self.res, self.tile = float(res), int(tile)
        self.K_walk = int(layers)
        self.K = int(layers) + int(wall_layers)  # walkable layers, then wall layers
        n = np.cross(B - A, C - A)
        ln = np.linalg.norm(n, axis=1)
        N = n / np.where(ln == 0, 1, ln)[:, None]
        N = np.where(N[:, 1:2] < 0, -N, N)  # upward
        pts, nrm = [], []
        e, ei = self._edge_samples(A, B, C, edge_step)
        pts.append(e)
        nrm.append(N[ei])
        # every triangle is fill-sampled, steep ones too: a wall must be a continuous run of heights in its nodes so
        # that it forms one solid column (an edge-only wall leaves holes where its face spans more than layer_gap)
        f, fi = self._fill_samples(A, B, C, fill_step)
        pts.append(f)
        nrm.append(N[fi])
        S = np.concatenate(pts)
        SN = np.concatenate(nrm)
        lo = S[:, [0, 2]].min(0) - 4 * res if len(S) else np.zeros(2)
        hi = S[:, [0, 2]].max(0) + 4 * res if len(S) else np.ones(2)
        self.origin = lo
        tl = self.tile
        nt = (np.ceil((hi - lo) / res).astype(np.int64) + 1 + tl - 1) // tl
        self.shape = (int(nt[0]) * tl, int(nt[1]) * tl)
        ij = np.floor((S[:, [0, 2]] - lo) / res + 0.5).astype(np.int64)
        tid_full = (ij[:, 0] // tl) * nt[1] + ij[:, 1] // tl
        used = np.unique(tid_full) if len(tid_full) else np.zeros(1, np.int64)  # an empty map keeps one empty tile
        self.tiles = np.full((int(nt[0]), int(nt[1])), -1, np.int32)
        self.tiles.flat[used] = np.arange(len(used), dtype=np.int32)
        T = len(used)
        data = {f: np.full((T, self.K, tl, tl), np.nan if f == "H" else 0.0, np.float32) for f in self.FIELDS}
        tid = self.tiles.flat[tid_full].astype(np.int64)
        key = (tid * tl + ij[:, 0] % tl) * tl + ij[:, 1] % tl
        h = S[:, 1]
        # highest first; at (almost) the same height prefer the most upward face (a slab's top edge is shared by its
        # top face and its side faces)
        order = np.lexsort((-SN[:, 1], -np.round(h, 2), key))
        key, S, SN, h = key[order], S[order], SN[order], h[order]
        walk_s = SN[:, 1] >= UP_FACE

        def put(sel, k_of, st, hb):
            kk = key[sel][st]
            t_, rem = kk // (tl * tl), kk % (tl * tl)
            li, lj = rem // tl, rem % tl
            Ss, Ns, hs = S[sel][st], SN[sel][st], h[sel][st]
            for f, val in zip(self.FIELDS, (hs, Ss[:, 0], Ss[:, 2], Ns[:, 0], Ns[:, 1], Ns[:, 2], hb)):
                data[f][t_, k_of, li, lj] = val

        # walkable layers 0..layers-1: per node, the highest walkable sample of each height cluster (samples within
        # layer_gap below the cluster's top), a point (HB = H)
        sel = walk_s
        kw, hw = key[sel], h[sel]
        for k in range(self.K_walk):
            if not len(kw):
                break
            first = np.r_[True, kw[1:] != kw[:-1]]
            st = np.flatnonzero(first)
            idx = np.flatnonzero(sel)
            sub = np.zeros(len(key), bool)
            sub[idx] = True
            put(sub, np.full(len(st), k), st, hw[st])
            grp = np.cumsum(first) - 1
            keep = hw < hw[st][grp] - layer_gap
            sel = sub.copy()
            sel[idx[~keep]] = False
            kw, hw = kw[keep], hw[keep]
        # wall layers: per node, runs of steep-face samples without a vertical gap of more than wall_gap (fill-
        # sampled, so a continuous face is one run): a solid column HB..H at the run top's (x, z), never a support
        steep_s = ~walk_s
        ks, hs_ = key[steep_s], h[steep_s]
        if len(ks):
            newnode = np.r_[True, ks[1:] != ks[:-1]]
            rstart = newnode | np.r_[False, (hs_[:-1] - hs_[1:]) > wall_gap]
            rid = np.cumsum(rstart) - 1
            node_r0 = np.maximum.accumulate(np.where(newnode, rid, 0))
            st = np.flatnonzero(rstart)
            k_of = (rid - node_r0)[st]
            hb = np.minimum.reduceat(hs_, st)
            ok = k_of < self.K - self.K_walk
            put(steep_s, self.K_walk + k_of[ok], st[ok], hb[ok])
        # one packed (tiles, tile, tile, layers, fields) array: a query gathers every field in one indexing op
        self.pack = np.ascontiguousarray(np.stack([data[f] for f in self.FIELDS], -1).transpose(0, 2, 3, 1, 4))
        self.min_y = float(np.nanmin(data["H"])) if np.isfinite(data["H"]).any() else 0.0
        del data
        self.death = [(np.asarray(a, np.float64), np.asarray(b, np.float64)) for a, b in death_boxes]
        self._stencils: dict = {}

    @property
    def nbytes(self) -> int:
        return int(self.pack.nbytes + self.tiles.nbytes)

    @staticmethod
    def _edge_samples(A, B, C, step):
        out, idx = [], []
        tri = np.arange(len(A))
        for P0, P1 in ((A, B), (B, C), (C, A)):
            L = np.linalg.norm(P1 - P0, axis=1)
            n = np.maximum(1, np.ceil(L / step).astype(np.int64))
            for m in np.unique(n):
                sel = n == m
                t = np.linspace(0.0, 1.0, m + 1)
                out.append((P0[sel, None, :] + (P1 - P0)[sel, None, :] * t[None, :, None]).reshape(-1, 3))
                idx.append(np.repeat(tri[sel], m + 1))
        if not out:
            return np.zeros((0, 3)), np.zeros(0, np.int64)
        return np.concatenate(out), np.concatenate(idx)

    @staticmethod
    def _fill_samples(A, B, C, step):
        L = np.maximum(np.linalg.norm(B - A, axis=1), np.linalg.norm(C - A, axis=1))
        n = np.maximum(1, np.ceil(L / step).astype(np.int64))
        tri = np.arange(len(A))
        out, idx = [], []
        for m in np.unique(n):
            sel = n == m
            i, j = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
            ok = i + j <= m
            u, v = (i[ok] / m), (j[ok] / m)
            a, b, c = A[sel], B[sel], C[sel]
            out.append((a[:, None, :] + (b - a)[:, None, :] * u[None, :, None]
                        + (c - a)[:, None, :] * v[None, :, None]).reshape(-1, 3))
            idx.append(np.repeat(tri[sel], len(u)))
        if not out:
            return np.zeros((0, 3)), np.zeros(0, np.int64)
        return np.concatenate(out), np.concatenate(idx)

    @classmethod
    def from_level(cls, geometry, depth=(), death_min_extent: float = 20.0, **kw) -> "SupportMap":
        """From features.LevelGeometry (static triangles, stoppers excluded) and tracker.level.depth."""
        boxes = []
        for item in depth:
            lo, hi = np.asarray(item[-2], np.float64), np.asarray(item[-1], np.float64)
            if min(hi[0] - lo[0], hi[2] - lo[2]) >= death_min_extent:
                boxes.append((lo, hi))
        T = geometry.tris
        return cls(T.A, T.B, T.C, death_boxes=boxes, **kw)

    def _stencil(self, r: float):
        st = self._stencils.get(r)
        if st is None:
            m = int(math.ceil((r + self.res) / self.res))
            di, dj = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
            keep = np.hypot(di, dj) * self.res <= r + self.res * 0.75
            st = (di[keep].ravel(), dj[keep].ravel())
            self._stencils[r] = st
        return st

    def _gather(self, xz, r):
        di, dj = self._stencil(r)
        c = np.floor((xz - self.origin) / self.res + 0.5).astype(np.int64)
        I = c[:, 0:1] + di[None, :]
        J = c[:, 1:2] + dj[None, :]
        nx, nz = self.shape
        ok = (I >= 0) & (I < nx) & (J >= 0) & (J < nz)
        I, J = np.where(ok, I, 0), np.where(ok, J, 0)
        tl = self.tile
        t = self.tiles[I // tl, J // tl]
        ok &= t >= 0
        t = np.where(ok, t, 0)
        G = self.pack[t, I % tl, J % tl]  # (N, S, K, fields)
        out = {f: G[..., i] for i, f in enumerate(self.FIELDS)}
        out["H"] = np.where(ok[..., None], out["H"], np.nan)
        return out

    def query(self, xz, y, r: float, climb_max: float, normals: bool = False, normal_eps: float = 0.2,
              step_len=None, ramp_tol: float = 0.15, wall_tol: float = 0.05, vel=None, pen_tol: float = 0.15,
              face_tol: float = 0.6, pressed=None):
        """Sphere of radius r centred at (x, y, z) per row. Returns (rest, wall_n, wall) or, with normals=True,
        (rest, wall_n, wall, support_n, wall_top).

        Hard barriers (climb rule): a sample within the radius is something the ball can stand on when
          * it is walkable (face n_y >= UP_FACE) and resting on it keeps the centre within a tolerance + the face's
            climb along the move (step_len = horizontal move of this step; None = the face's steepest climb) of its
            current height: holding contact (floors, ramps whatever their total rise, rising rails, sampling noise).
            For a steep edge contact (contact normal n_y < UP_FACE) ahead of the motion or while the ball is pressed
            against a wall (pressed) the tolerance is ~0, so pushing against a lip does not let it creep up, or
          * with vel (N,2) given (horizontal velocity, units / phys-s): it is a walkable edge ahead of the motion
            (within 60 degrees) h above the ball bottom and the speed into it reaches mount_speed(h) (speed-dependent
            edge mount, a model estimate; never at h >= MOUNT_CAP), or it is any sample beside / behind the ball below
            its centre that lifts it at most 0.15 more than a holding contact may (two rails, slanted rail faces; no
            sideways mounting) and the ball is not pressed against a wall;
            without vel (static queries): it is at most climb_max above the ball bottom.
        Anything else that the sphere, lifted onto its support, penetrates by more than pen_tol is a wall: a wall
        column HB..H (a run of steep-face samples) or a walkable point overlapping the shrunken sphere at that
        distance. So a guardrail or pit wall (h >= MOUNT_CAP) always blocks, a long ramp does not, a lip blocks unless
        the ball runs into it fast enough, and a steep face is never climbed as a staircase.

        rest = highest centre height on the supporting samples (nan = none within r); wall_n = horizontal unit
        push-out normal of the blocking samples; support_n = unit contact normal at the rest height (face normal on a
        face, sample -> centre over an edge; averaged over the samples within normal_eps of the top, face contacts
        taking precedence, so two rails give a vertical net normal); wall_top = height of the highest blocking sample above the ball bottom (nan = none)."""
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        N = len(xz)
        D = self._gather(xz, r)
        # float32 arithmetic relative to the query point (heights relative to y keep float32 precise)
        y64 = np.asarray(y, np.float64).reshape(-1)
        y = np.zeros(N, np.float32)
        xz = xz.astype(np.float32)
        D["H"] = (D["H"] - y64[:, None, None]).astype(np.float32)
        D["HB"] = (D["HB"] - y64[:, None, None]).astype(np.float32)
        H = D["H"]
        dx = D["X"] - xz[:, 0, None, None]
        dz = D["Z"] - xz[:, 1, None, None]
        d2 = dx * dx + dz * dz
        r2 = np.float32(r * r)
        near = (d2 < r2) & np.isfinite(H)
        lift = np.sqrt(np.maximum(r2 - d2, 0.0))
        bottom = (y - r)[:, None, None]
        nh = np.sqrt(D["NX"] ** 2 + D["NZ"] ** 2)
        dist = np.sqrt(d2)
        walk = (D["NY"] >= UP_FACE) & np.isfinite(H)
        # face contact: the sample sits where the sphere touches its face (centre r along the face normal from it, up
        # to the sampling spacing)
        face = walk & (np.hypot(dx + r * D["NX"], dz + r * D["NZ"]) <= face_tol) & (H < 0)
        ny_s = np.maximum(D["NY"], 1e-3)
        # rest height on each sample: the sphere touching that point (the face plane through it would be exact on
        # a slope, but near a face's edge it extends past the edge and lands a ball that just took off)
        rest_s = H + lift
        # holding contact: a walkable sample below the centre that keeps the ball at (about) its height, the face
        # slope allowing for the move (floors, ramps, rails, two rails at once: no speed needed)
        h = H - bottom  # sample height above the ball bottom
        pr_ = (np.zeros((N, 1, 1), bool) if pressed is None
               else np.broadcast_to(np.asarray(pressed, bool).reshape(-1), (N,))[:, None, None])
        steep = ~face & (-H < UP_FACE * r)  # edge contact steeper than a walkable face (contact normal n_y = -H / r)
        ahead = np.zeros((N, 1, 1), bool)
        if vel is not None:
            vv = np.asarray(vel, np.float32).reshape(-1, 2)
            sp = np.sqrt((vv * vv).sum(1))[:, None, None]
            v_into = (vv[:, 0, None, None] * dx + vv[:, 1, None, None] * dz) / np.maximum(dist, 1e-6)
            ahead = (v_into > 0.5 * sp) & (sp > 0.5)  # an edge within 60 degrees of the motion
        # holding contact: keeps the ball at its height or lifts it by at most a tolerance + the face's climb along
        # the move (sampling noise, ramps, rising rails). For a steep edge contact ahead of the motion (a lip being
        # approached), or any steep edge contact while the ball is pressed against a wall, the tolerance is ~0:
        # otherwise the ball would creep up a lip in small moves instead of mounting it by the speed rule
        tight = steep & (pr_ | ahead)
        tol = np.where(tight, np.float32(0.02), np.float32(ramp_tol))
        allow = tol
        if step_len is not None:
            sl = np.broadcast_to(np.asarray(step_len, np.float32).reshape(-1), (N,))[:, None, None]
            if vel is None:
                rise = nh / ny_s  # the face's steepest climb
            else:  # the face's climb along the motion (a rail rising ahead; not a rim that tilts back at the ball)
                ux = vv[:, 0, None, None] / np.maximum(sp, 1e-6)
                uz = vv[:, 1, None, None] / np.maximum(sp, 1e-6)
                rise = np.maximum(-(D["NX"] * ux + D["NZ"] * uz), 0.0) / ny_s
            allow = tol + sl * rise
        hold = walk & (H < 0) & (rest_s <= allow)
        if vel is None:  # static: any edge up to climb_max above the bottom (default: below the centre)
            edge = h <= climb_max
        else:
            # an edge ahead is mounted by the speed rule onto its walkable top (a step's top edge keeps its top face's
            # normal; a steep plane rasterises into node tops that would otherwise read as a staircase); an edge
            # beside or behind the ball (two rails, a rising rail) holds it when below the centre and lifting it
            # little more than a holding contact may (no sideways mounting), unless the ball is pressed against a wall
            # (also a steep face beside the ball: rails with slanted inner faces hold it in the groove between them)
            edge = ((walk & ahead & (v_into >= mount_speed(h, r).astype(np.float32)))
                    | (~ahead & (H < 0) & ~pr_ & (rest_s <= allow + np.float32(0.15))))
        sup = (hold | edge | (h <= MOUNT_FREE)) & near & np.isfinite(H)
        cand = np.where(sup, rest_s, -np.inf)
        rest = cand.max(axis=(1, 2)).astype(np.float64)
        rest = np.where(np.isfinite(rest), rest + y64, np.nan)
        if normals:
            top = cand >= (rest - y64)[:, None, None].astype(np.float32) - normal_eps
            # where the sphere sits on a face, that face's normal (edge contacts near it are sampling neighbours);
            # otherwise the edge contacts' sample -> centre directions (two rails: both sides, a vertical net normal)
            top_face = top & face
            top = np.where(top_face.any(axis=(1, 2), keepdims=True), top_face, top)
            # sphere on a face: the centre is r along the face normal from the contact; over an edge: sample -> centre
            ux = np.where(face, D["NX"], -dx / r)
            uy = np.where(face, D["NY"], lift / r)
            uz = np.where(face, D["NZ"], -dz / r)
            nx = np.where(top, ux, 0.0).sum(axis=(1, 2))
            nyv = np.where(top, uy, 0.0).sum(axis=(1, 2))
            nz = np.where(top, uz, 0.0).sum(axis=(1, 2))
            ln = np.sqrt(nx * nx + nyv * nyv + nz * nz)
            ln = np.where(ln > 1e-9, ln, 1.0)
            sn = np.stack([nx / ln, nyv / ln, nz / ln], -1).astype(np.float64)
            sn[~np.isfinite(rest)] = (0.0, 1.0, 0.0)
        # wall: a sample column (HB..H) too high to climb that overlaps the sphere's vertical extent at its distance
        # walls are tested with the sphere where it ends up: lifted onto its support when that is higher than y
        # (the face below a step's edge does not block a ball that has mounted the edge)
        rest_rel = (rest - y64).astype(np.float32)
        lift_to = np.where(np.isfinite(rest_rel), np.maximum(rest_rel, 0.0), 0.0)[:, None, None]
        Hw, HBw = H - lift_to, D["HB"] - lift_to
        lp = np.sqrt(np.maximum(np.float32((r - pen_tol) ** 2) - d2, 0.0))
        inside = near & ~sup & (d2 < (r - pen_tol) ** 2) & (HBw < lp - wall_tol) & (Hw > -lp + wall_tol)
        wall = inside.any(axis=(1, 2))
        wn = np.zeros((N, 2))
        if wall.any():
            inv = 1.0 / np.sqrt(np.maximum(d2, 1e-6))
            wx = np.where(inside, -dx * inv, 0.0).sum(axis=(1, 2))
            wz = np.where(inside, -dz * inv, 0.0).sum(axis=(1, 2))
            ln = np.hypot(wx, wz)
            wn = np.stack([wx, wz], -1) / np.where(ln > 1e-9, ln, 1.0)[:, None]
            wall &= ln > 1e-9
        if normals:
            wtop = np.where(inside, H, -np.inf).max(axis=(1, 2)).astype(np.float64) + r
            return rest, wn, wall, sn, np.where(wall, wtop, np.nan)
        return rest, wn, wall

    def death_top(self, xz) -> np.ndarray:
        """Top of the highest death box under each x-z point (nan = none)."""
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        out = np.full(len(xz), np.nan)
        for lo, hi in self.death:
            m = (xz[:, 0] >= lo[0]) & (xz[:, 0] <= hi[0]) & (xz[:, 1] >= lo[2]) & (xz[:, 1] <= hi[2])
            out = np.where(m & ~(out > hi[1]), hi[1], out)
        return out


# ------------------------------------------------------------------------------------------------- rollout

@dataclass
class ReachConfig:
    horizon_s: float = 3.0  # game seconds
    step_psi: int = 4  # integration step (closed form inside a step)
    switch_s: tuple = (0.25, 0.5, 1.0)  # game seconds at which two-segment plans switch keys
    # static step limit, used where no velocity is known (the start state's own support, probe_directions without a
    # speed): any walkable edge below the centre; a moving ball mounts edges ahead of it by the speed rule (mount_speed)
    climb_max: float = 2.0
    death_margin: float = 2.0  # BallOff fired with the centre 2.2-3.0 above the box top (SR09); empirical
    void_drop: float = 30.0  # falling this far below the start with no support under it = dead
    ground_tol: float = 0.25  # centre within this of the rest height = in contact
    path_back: float = 10.0
    path_ahead: float = 160.0
    gap_min: float = 3.0  # a stretch of the reference line without support under it at least this long is a gap
    gap_drop: float = 3.0  # ... "without support" = no rest height within this below the line
    turn_angle: float = 45.0  # heading change (deg) over turn_window units of arc that makes a turn
    turn_window: float = 10.0
    look: float = 60.0  # how far ahead along the line to look for the next gap / turn
    route_tol: float = 4.0  # a plan is "on the route" within this 3-D distance of the reference line ...
    route_ahead: float = 3.0  # ... at an arc length at least this far ahead of the start
    probe: bool = True  # also run probe_directions for the 8 key directions (blocked directions / nearest wall)
    probe_dist: float = 8.0
    extra: dict = field(default_factory=dict)


def _plans(cfg: ReachConfig):
    """(first push (P,2) unit world directions or 0, second push (P,2), switch PSI (P,), combo index of the first
    segment (-1 none)). World directions: the 8 key pushes at quarter 0 (any quarter is a rotation of these)."""
    dirs = key_push(np.array(KEY_COMBOS), 0)
    nine = np.vstack([np.zeros((1, 2)), dirs])
    first, second, sw, idx = [], [], [], []
    H = int(round(cfg.horizon_s * GAME_PSI_HZ))
    for a in range(9):
        first.append(nine[a]); second.append(nine[a]); sw.append(H); idx.append(a - 1)
    for t in cfg.switch_s:
        k = int(round(t * GAME_PSI_HZ))
        for a in range(9):
            for b in range(9):
                if a != b:
                    first.append(nine[a]); second.append(nine[b]); sw.append(k); idx.append(-2)
    return np.array(first), np.array(second), np.array(sw), np.array(idx)


def _lin_step(v, a, lam, n):
    """n PSIs of (p <- p + v / 66, then v <- lam v + a): retail moves the position with the velocity it had before
    the PSI's update (p[k+1] = p[k] + v[k] / 66, checked on tas-v1). Returns (v_n, displacement)."""
    ln = lam ** n
    geo = (1.0 - ln) / (1.0 - lam)  # sum_{k=0}^{n-1} lam^k
    v_n = ln * v + a * geo
    # sum_{j=0}^{n-1} v_j = v * geo + a * (n - geo) / (1 - lam)
    disp = (v * geo + a * (n - geo) / (1.0 - lam)) / PSI_HZ
    return v_n, disp


def plan_pushes(push1, push2, switch_psi, steps: int, step_psi: int) -> np.ndarray:
    """(P, steps, 2) per-step pushes of two-segment plans (push1 before switch_psi, push2 from it)."""
    t0 = np.arange(steps) * step_psi
    return np.where((t0[None, :] < np.asarray(switch_psi)[:, None])[..., None], np.asarray(push1)[:, None, :],
                    np.asarray(push2)[:, None, :])


def rollout(smap: SupportMap, pos, vel, ball: BallParams, pushes, cfg: ReachConfig, grounded=None,
            step_psi: int | None = None, fan=None):
    """Simulate P plans. pos, vel: (3,) one PSI state for all plans (BALL.pos, BALL.speed + speed_change), or (P,3)
    one state per plan (batching several states in one call); grounded: None (from the support map), a bool, or (P,).
    pushes (P, steps, 2) world key push per step (key_push; |push| = 1 or sqrt 2); a step is step_psi PSIs
    (cfg.step_psi by default), so the horizon is steps * step_psi PSIs.
    Returns dict of per-plan arrays: alive, t_death (PSI, inf), traj (P, steps+1, 3) (frozen after death),
    grounded (P, steps+1), landed_at (first ground contact after being airborne: PSI index or -1), land_pos."""
    pushes = np.asarray(pushes, np.float64)
    P, steps = pushes.shape[0], pushes.shape[1]
    n = int(step_psi or cfg.step_psi)
    r = ball.radius
    pos, vel = np.asarray(pos, np.float64), np.asarray(vel, np.float64)
    if pos.ndim == 1:  # one state: one support query, broadcast
        rest0, _, _, sn0, _ = smap.query(pos[None, [0, 2]], pos[None, 1], r, cfg.climb_max, normals=True)
        rest0, sn0 = np.repeat(rest0, P), np.repeat(sn0, P, axis=0)
    else:
        rest0, _, _, sn0, _ = smap.query(pos[:, [0, 2]], pos[:, 1], r, cfg.climb_max, normals=True)
    p = np.array(np.broadcast_to(pos, (P, 3)))
    v = np.array(np.broadcast_to(vel, (P, 3)))
    has0 = np.isfinite(rest0)
    if grounded is None:
        g = has0 & (np.abs(p[:, 1] - np.where(has0, rest0, np.inf)) <= cfg.ground_tol)
    else:
        g = np.array(np.broadcast_to(np.asarray(grounded, bool), (P,)))
    snap = g & has0
    p[snap, 1] = rest0[snap]  # on the ground the height is the support field's (vertical speed = its change)
    sn = sn0.copy()
    alive = np.ones(P, bool)
    t_death = np.full(P, np.inf)
    traj = np.zeros((P, steps + 1, 3))
    gr = np.zeros((P, steps + 1), bool)
    traj[:, 0], gr[:, 0] = p, g
    was_air = ~g.copy()
    landed_at = np.full(P, -1)
    land_pos = np.full((P, 3), np.nan)
    y0 = p[:, 1].copy()
    pressed = np.zeros(P, np.int64)  # steps left in the "ran into a wall" state (set on a wall, held for a while)
    lam_g, lam_a = ball.ground_lambda, ball.air_lambda
    g_psi = GRAVITY / PSI_HZ
    for s in range(steps):
        t0 = s * n
        a = np.flatnonzero(alive)  # dead plans are frozen
        if len(a) == 0:
            traj[:, s + 1:], gr[:, s + 1:] = traj[:, s:s + 1], gr[:, s:s + 1]
            break
        pa, va, ga, sna = p[a], v[a], g[a], sn[a]
        u = pushes[a, s]
        hv = va[:, [0, 2]]
        # ground: rolling on the support plane; horizontal part of the tangential gravity g n_y (n_x, n_z), x 5/7
        a_g = ball.ground_accel * u + (g_psi / ROLL_INERTIA) * sna[:, 1:2] * sna[:, [0, 2]]
        vg, dg = _lin_step(hv, a_g, lam_g, n)
        # air: full key force, gravity, linear damping on all three components
        lift = None
        vertical = np.full(len(a), -g_psi)
        if fan is not None:
            boxes, impulse, radius = fan
            # Same oriented volumes and nominal centre distance as fastreach.
            # Reference rollout samples force once per integration step (one PSI
            # with ReachConfig(step_psi=1)); it does not share the envelope scan.
            d = pa[:, None, :] - boxes[None, :, :3]
            local = np.einsum("pfi,fji->pfj", d, boxes[:, 6:15].reshape(-1, 3, 3))
            delta = np.maximum(np.abs(local) - boxes[None, :, 3:6], 0.0)
            inside = np.any(np.sum(delta * delta, axis=2) <= radius**2, axis=1)
            vertical[inside] += impulse
            if impulse > g_psi:
                lift = inside & (va[:, 1] >= 0.0)
                ga = ga & ~inside
        a_air = np.c_[ball.air_accel * u[:, 0], vertical, ball.air_accel * u[:, 1]]
        vair, dair = _lin_step(va, a_air, lam_a, n)
        new_h = np.where(ga[:, None], dg, dair[:, [0, 2]])
        new_vh = np.where(ga[:, None], vg, vair[:, [0, 2]])
        xz = pa[:, [0, 2]] + new_h
        # ballistic height after the step (also for a grounded ball the floor may drop away from)
        vyb_n, dyb = _lin_step(va[:, 1], vertical if fan is not None else -g_psi, lam_a, n)
        y_ball = pa[:, 1] + dyb
        y_ref = np.maximum(pa[:, 1], y_ball)
        step_len = np.linalg.norm(new_h, axis=1)
        vel_q = new_vh
        rest, wn, wall, sn_new, _ = smap.query(xz, y_ref, r, cfg.climb_max, normals=True, step_len=step_len,
                                               vel=vel_q, pressed=pressed[a] > 0)
        # walls: drop the motion into the wall, keep the slide (the re-query keeps the velocity the ball ran into the
        # wall with, so the wall it could not mount does not become a step it creeps up)
        vh_in = vel_q
        if wall.any():
            into = (new_h * wn).sum(1)
            fix = wall & (into < 0)
            new_h = np.where(fix[:, None], new_h - into[:, None] * wn, new_h)
            vin = (new_vh * wn).sum(1)
            new_vh = np.where((wall & (vin < 0))[:, None], new_vh - vin[:, None] * wn, new_vh)
            xz = pa[:, [0, 2]] + new_h
            if fix.any():
                rest2, _, _, sn2, _ = smap.query(xz[fix], y_ref[fix], r, cfg.climb_max, normals=True,
                                                  step_len=step_len[fix], vel=vh_in[fix], pressed=True)
                rest[fix], sn_new[fix] = rest2, sn2
        pressed[a] = np.where(wall, 8, np.maximum(pressed[a] - 1, 0))
        has = np.isfinite(rest)
        # a grounded ball follows its support down to ground_tol below the ballistic height (else it takes off); an
        # airborne one lands only on a support that reaches its ballistic height
        on = has & (rest >= y_ball - np.where(ga, cfg.ground_tol, 0.0))
        if lift is not None:
            on &= ~lift  # permit the first upward impulse even before position moves
        y_new = np.where(on, rest, y_ball)
        # on the ground the vertical speed follows the support; upwards it is capped at what the support plane gives
        # for this horizontal velocity (a mounted step or sampling noise must not turn into a launch)
        vy_follow = (rest - pa[:, 1]) * PSI_HZ / n
        sn_y = np.maximum(sn_new[:, 1], 0.2)
        vy_plane = -(sn_new[:, 0] * new_vh[:, 0] + sn_new[:, 2] * new_vh[:, 1]) / sn_y
        vy_follow = np.minimum(vy_follow, np.maximum(vy_plane, 0.0))
        vy_new = np.where(on, np.where(ga, vy_follow, 0.0), vyb_n)
        first_land = on & ~ga & (landed_at[a] < 0) & was_air[a]
        fl = a[first_land]
        landed_at[fl] = t0 + n
        land_pos[fl] = np.c_[xz[first_land, 0], y_new[first_land], xz[first_land, 1]]
        p[a] = np.c_[xz[:, 0], y_new, xz[:, 1]]
        v[a] = np.c_[new_vh[:, 0], vy_new, new_vh[:, 1]]
        g[a] = on
        sn[a] = sn_new
        was_air[a] |= ~on
        top = smap.death_top(xz)
        dead = ((np.isfinite(top) & (y_new < top + cfg.death_margin))
                | (~has & (y_new < y0[a] - cfg.void_drop))
                | (y_new < smap.min_y - cfg.void_drop))
        t_death[a[dead]] = t0 + n
        alive[a[dead]] = False
        traj[:, s + 1], gr[:, s + 1] = p, g
    # end of horizon: airborne with nothing under it (no sample below the centre within the radius) = dead
    a = np.flatnonzero(alive & ~g)
    if len(a):
        below, _, _ = smap.query(p[a][:, [0, 2]], p[a, 1], r, r)
        void = a[~np.isfinite(below)]
        t_death[void] = steps * n + 1
        alive[void] = False
    return {"alive": alive, "t_death": t_death, "traj": traj, "grounded": gr, "landed_at": landed_at,
            "land_pos": land_pos, "step_psi": n}


# --------------------------------------------------------------------------------------- path utilities

def project_many(path, pts, s_lo: float, s_hi: float, chunk: int = 1024):
    """Projection of each point (N,3) onto the path segments with arc in [s_lo, s_hi] (as ReferencePath.nearest):
    returns (arc length of the nearest point on the polyline, 3-D distance to it)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    seg = path._segments(max(0.0, s_lo), min(path.length, s_hi))
    A, D, L2 = path.P[seg], path.seg[seg], path.L2[seg]
    cum, sl = path.cum[seg], path.seg_len[seg]
    s = np.empty(len(pts))
    dist = np.empty(len(pts))
    for a in range(0, len(pts), chunk):
        q = pts[a:a + chunk]
        rel = q[:, None, :] - A[None]
        t = np.clip((rel * D[None]).sum(-1) / L2[None], 0.0, 1.0)
        d2 = ((rel - t[..., None] * D[None]) ** 2).sum(-1)
        k = np.argmin(d2, axis=1)
        r = np.arange(len(q))
        s[a:a + chunk] = cum[k] + t[r, k] * sl[k]
        dist[a:a + chunk] = np.sqrt(d2[r, k])
    return s, dist


def next_features(smap: SupportMap, path, s0: float, cfg: ReachConfig, radius: float = 2.0) -> dict:
    """Next gap (line stretch with no support under it) and next turn of the reference line ahead of s0."""
    ss = np.arange(s0, min(path.length, s0 + cfg.look), 1.0)
    if len(ss) < 2:
        return {"gap": None, "turn": None}
    pts = np.array([path.point_at(s) for s in ss])
    # supports at most 1 above the line (a sphere centred 1 above it, edges up to its centre): a walkable surface
    # overhead (a beam, an upper level) must not hide a gap
    rest, _, _ = smap.query(pts[:, [0, 2]], pts[:, 1] + 1.0, radius, radius)
    nosup = ~np.isfinite(rest) | (rest < pts[:, 1] - cfg.gap_drop)
    gap = None
    i = 0
    while i < len(ss):
        if nosup[i]:
            j = i
            while j + 1 < len(ss) and nosup[j + 1]:
                j += 1
            if ss[j] - ss[i] + 1.0 >= cfg.gap_min:
                gap = (float(ss[i]), float(ss[j] + 1.0))
                break
            i = j + 1
        else:
            i += 1
    d = np.diff(pts[:, [0, 2]], axis=0)
    head = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    w = max(1, int(round(cfg.turn_window)))
    turn = None
    if len(head) > w:
        ch = np.abs(head[w:] - head[:-w])
        k = np.flatnonzero(ch >= math.radians(cfg.turn_angle))
        if len(k):
            a = int(k[0])
            b = a
            while b + 1 < len(ch) and ch[b + 1] >= math.radians(cfg.turn_angle) * 0.5:
                b += 1
            turn = (float(ss[a]), float(min(ss[-1], ss[b + w] if b + w < len(ss) else ss[-1])))
    return {"gap": gap, "turn": turn}


def probe_directions(smap: SupportMap, pos, dirs, radius: float = 2.0, climb_max: float = 2.0,
                     max_dist: float = 8.0, step: float = 0.5, drop_tol: float = 1.0,
                     speed: float | None = None) -> dict:
    """Geometric look-around: walk a sphere of the given radius from pos along each horizontal unit direction
    (dirs (D,2) world x-z) in steps of `step`, following its support with the climb rule of SupportMap.query: with
    `speed` (units / phys-s) the walk runs into edges at that speed (speed-dependent mount), without it edges up to
    climb_max are steps. Per direction: blocked_at = distance walked before a barrier it cannot mount (inf = none within
    max_dist), wall_top = that barrier's height above the ball bottom, mount_speed = the speed it would need
    (inf = never: at or above MOUNT_CAP), wall_n = its push-out normal; drop_at / drop = where the support falls away
    by more than drop_tol (an edge or a pit; drop = height lost, inf = nothing under the sphere). The walk starts on
    the support under the ball (airborne = the ball is more than 0.25 above it); over the void everything is nan."""
    dirs = np.asarray(dirs, np.float64).reshape(-1, 2)
    n = len(dirs)
    nrm = np.linalg.norm(dirs, axis=1)
    dirs = dirs / np.where(nrm > 1e-9, nrm, 1.0)[:, None]
    pos = np.asarray(pos, np.float64)
    rest0, _, _, _, _ = smap.query(pos[None, [0, 2]], pos[None, 1], radius, climb_max, normals=True)
    if not np.isfinite(rest0[0]):
        # nothing under the ball at all (over the void): no walk to start (all nan)
        nan = np.full(n, np.nan)
        return {"dirs": dirs, "blocked_at": nan.copy(), "wall_top": nan.copy(), "mount_speed": nan.copy(),
                "wall_n": np.zeros((n, 2)), "drop_at": nan.copy(), "drop": nan.copy(), "airborne": True}
    y0 = rest0[0]  # the walk starts on the support under the ball, also when the ball is in the air above it
    airborne = bool(pos[1] - y0 > 0.25)
    xz = np.tile(pos[[0, 2]], (n, 1))
    y = np.full(n, y0)
    blocked = np.full(n, np.inf)
    wall_top = np.full(n, np.nan)
    wall_n = np.zeros((n, 2))
    drop_at = np.full(n, np.inf)
    drop = np.full(n, np.nan)
    live = np.ones(n, bool)
    for j in range(1, int(round(max_dist / step)) + 1):
        a = np.flatnonzero(live)
        if len(a) == 0:
            break
        nxt = xz[a] + dirs[a] * step
        vel = None if speed is None else dirs[a] * speed
        rest, wn, wall, _, wt = smap.query(nxt, y[a], radius, climb_max, normals=True, step_len=step, vel=vel)
        hit = wall & ((wn * dirs[a]).sum(1) < -0.2)  # a barrier facing the walk
        blocked[a[hit]] = (j - 1) * step
        wall_top[a[hit]] = wt[hit]
        wall_n[a[hit]] = wn[hit]
        fall = ~hit & (~np.isfinite(rest) | (rest < y[a] - drop_tol))
        drop_at[a[fall]] = j * step
        drop[a[fall]] = np.where(np.isfinite(rest[fall]), y[a[fall]] - rest[fall], np.inf)
        go = ~hit & ~fall
        xz[a[go]] = nxt[go]
        y[a[go]] = rest[go]
        live[a[hit | fall]] = False
    return {"dirs": dirs, "blocked_at": blocked, "wall_top": wall_top,
            "mount_speed": np.where(np.isfinite(wall_top), mount_speed(np.nan_to_num(wall_top), radius), np.nan),
            "wall_n": wall_n, "drop_at": drop_at, "drop": drop, "airborne": airborne}


# ------------------------------------------------------------------------------------------- estimator

class ReachEstimator:
    """reach = ReachEstimator(smap).estimate(pos, vel, ball_type, quarter, path=..., s0=...)."""

    def __init__(self, smap: SupportMap, config: ReachConfig | None = None):
        self.smap = smap
        self._mechanism_maps = {}
        self._plate_probe = None
        self._paper_regions = np.empty((0, 2, 3))
        self._fans = None
        self.cfg = config or ReachConfig()
        self._plans = _plans(self.cfg)
        steps = int(math.ceil(self.cfg.horizon_s * GAME_PSI_HZ / self.cfg.step_psi))
        self._pushes = plan_pushes(self._plans[0], self._plans[1], self._plans[2], steps, self.cfg.step_psi)

    def _ball_at(self, ball_type, pos):
        ball = BALLS.get(int(ball_type))
        if ball is not None and not ball.supported and len(self._paper_regions):
            # The legacy estimator does not model paper. Enable its reference
            # probe/rollout near mapped plates/fans, not for untouched remote states.
            p = np.asarray(pos, float).reshape(-1, 1, 3)
            lo, hi = self._paper_regions[:, 0], self._paper_regions[:, 1]
            radius = max(self.cfg.probe_dist, self.cfg.horizon_s * ball.top_speed(2)) + ball.radius
            if np.all(np.any(np.all((p >= lo - radius) & (p <= hi + radius), axis=2), axis=1)):
                from dataclasses import replace
                return replace(ball, supported=True)
        return ball

    def _map_at(self, ball_type, pos):
        k = int(ball_type)
        if self._plate_probe is not None and k == self._plate_probe[0]:
            from .reach_mechanisms import uses_lowered_plate
            pts = np.asarray(pos, float).reshape(-1, 3)
            if all(uses_lowered_plate(self._mechanism_tracker, p, self._plate_probe[2]) for p in pts):
                k = self._plate_probe[1]
        return self._mechanism_maps.get(k, self.smap)

    def estimate(self, pos, vel, ball_type: int = 0, quarter: int = 0, path=None, s0: float | None = None,
                 grounded=None) -> dict:
        ball = self._ball_at(ball_type, pos)
        smap = self._map_at(ball_type, pos)
        if ball is None or not ball.supported:
            return {"supported": False, "ball": None if ball is None else ball.name}
        c = self.cfg
        p1, p2, sw, idx = self._plans
        # world pushes of the plans are fixed; key combos per direction depend on the camera quarter
        R = rollout(smap, pos, vel, ball, self._pushes, c, grounded, fan=self._fans if ball_type == 1 else None)
        alive, traj, n = R["alive"], R["traj"], R["step_psi"]
        out = {"supported": True, "ball": ball.name, "doomed": not bool(alive.any()),
               "n_alive": int(alive.sum()), "n_plans": len(alive), "alive": alive, "t_death": R["t_death"],
               "traj": traj, "grounded": R["grounded"], "step_psi": n}
        # per key combo (constant plans 1..8): where the ball is supported first / at the end, and its fate
        dirs = key_push(np.array(KEY_COMBOS), quarter)
        const = np.arange(1, 9)
        per = []
        for j, keys in enumerate(KEY_COMBOS):
            # the constant plan whose world push equals this combo's push under the camera quarter
            k = int(const[np.argmin(np.linalg.norm(p1[const] - dirs[j], axis=1))])
            lp = R["land_pos"][k]
            end = traj[k, -1]
            td = float(R["t_death"][k])
            per.append({"keys": int(keys), "plan": k, "alive": bool(alive[k]),
                        "t_death_s": td / GAME_PSI_HZ if np.isfinite(td) else None,
                        "landing": None if not np.isfinite(lp).all() else lp, "end": end,
                        "support": lp if np.isfinite(lp).all() else end})
        out["keys"] = per
        if c.probe:
            # blocked = cannot be mounted even at the two-key flat-road top speed
            pr = probe_directions(smap, pos, dirs, ball.radius, c.climb_max, c.probe_dist,
                                  speed=ball.top_speed(2, ground=True))
            for j, k in enumerate(per):
                k["blocked_at"] = float(pr["blocked_at"][j])
                k["wall_top"] = None if not np.isfinite(pr["wall_top"][j]) else float(pr["wall_top"][j])
                k["mount_speed"] = None if not np.isfinite(pr["wall_top"][j]) else float(pr["mount_speed"][j])
                k["drop_at"] = float(pr["drop_at"][j])
            out["probe"] = pr
            j = int(np.argmin(pr["blocked_at"]))
            out["nearest_wall"] = (None if not np.isfinite(pr["blocked_at"][j]) else
                                   {"keys": int(KEY_COMBOS[j]), "dist": float(pr["blocked_at"][j]),
                                    "height": float(pr["wall_top"][j])})
        if path is not None:
            if s0 is None:
                s0 = float(project_many(path, np.asarray(pos)[None], 0.0, path.length)[0][0])
            # every 4th trajectory point and the last one, each with its true PSI time
            idx = np.unique(np.r_[np.arange(0, traj.shape[1], 4), traj.shape[1] - 1])
            sub = traj[:, idx]
            P, S = sub.shape[0], sub.shape[1]
            s, dl = project_many(path, sub.reshape(-1, 3), s0 - c.path_back, s0 + c.path_ahead)
            s, dl = s.reshape(P, S), dl.reshape(P, S)
            # a dead plan only counts the part of its trajectory before it left the road for good
            tstep = idx * n
            s = np.where(tstep[None, :] <= np.where(np.isfinite(R["t_death"]), R["t_death"], np.inf)[:, None], s, -np.inf)
            # a dead plan counts up to its death; furthest arc along the line among surviving plans
            reach = np.where(alive, s.max(1), -np.inf)
            out["s0"] = s0
            out["max_safe_s"] = float(reach.max() - s0) if alive.any() else 0.0
            out["s_plan_max"] = s.max(1)
            # on the route: a surviving plan that passes within route_tol of the line (3-D distance: a floor below
            # the line, e.g. after dropping off the platform the line continues on, does not count) at least
            # route_ahead units of arc ahead of s0
            near = (dl <= c.route_tol) & (s >= s0 + c.route_ahead) & alive[:, None]
            out["route_ok"] = bool(near.any())
            out["max_route_s"] = float(np.where(near, s, -np.inf).max() - s0) if near.any() else None
            out["start_route_dist"] = float(project_many(path, np.asarray(pos, np.float64)[None], s0 - c.path_back,
                                                         s0 + c.path_back)[1][0])
            # the furthest any plan travels (horizontal path length, dead plans up to their death): a feature further
            # along the line than this is out of the horizon, its flag is None (unknown), not False
            seg = np.linalg.norm(np.diff(traj[:, :, [0, 2]], axis=1), axis=2)
            tt = (np.arange(1, traj.shape[1]) * n)[None, :]
            seg = np.where(tt <= np.where(np.isfinite(R["t_death"]), R["t_death"], np.inf)[:, None], seg, 0.0)
            travel = float(seg.sum(1).max())
            out["travel"] = travel
            feats = next_features(smap, path, s0, c, ball.radius)
            for name in ("gap", "turn"):
                f = feats[name]
                out[name] = f
                target = None if f is None else f[1] + 2.0
                out[f"{name}_in_horizon"] = f is not None and target - s0 <= travel
                if f is None or target - s0 > travel:
                    out[f"can_make_{name}"] = None
                else:  # a surviving plan on the route (route_tol of the line) past the feature
                    out[f"can_make_{name}"] = bool((near & (s >= target)).any())
        return out


    def estimate_batch(self, pos, vel, ball_type: int = 0, grounded=None) -> dict:
        """Physics part only, for N states in one rollout call: doomed (N,), n_alive (N,), alive (N, plans)."""
        ball = self._ball_at(ball_type, pos)
        smap = self._map_at(ball_type, pos)
        if ball is None or not ball.supported:
            raise ValueError("unsupported ball type")
        pos = np.asarray(pos, np.float64).reshape(-1, 3)
        vel = np.asarray(vel, np.float64).reshape(-1, 3)
        N, P = len(pos), len(self._pushes)
        g = None if grounded is None else np.repeat(np.broadcast_to(np.asarray(grounded, bool).reshape(-1), (N,)), P)
        R = rollout(smap, np.repeat(pos, P, axis=0), np.repeat(vel, P, axis=0), ball,
                    np.tile(self._pushes, (N, 1, 1)), self.cfg, g, fan=self._fans if ball_type == 1 else None)
        alive = R["alive"].reshape(N, P)
        return {"doomed": ~alive.any(1), "n_alive": alive.sum(1), "alive": alive}


def estimator_for_tracker(tracker, config: ReachConfig | None = None, *, mechanisms=False, **kw) -> ReachEstimator:
    from .features import level_geometry
    geo = level_geometry(tracker)
    depth = tracker.level.depth
    result = ReachEstimator(SupportMap.from_level(geo, depth, **kw), config)
    if mechanisms:
        from types import SimpleNamespace
        from .reach_mechanisms import add_geometry, fan_layer, layers_for_tracker
        T = geo.tris
        geom = (T.A, T.B, T.C, T.owner, np.zeros(len(T.A), dtype=bool))
        maps = {}
        layers, result._plate_probe = layers_for_tracker(tracker)
        result._mechanism_tracker = tracker
        for k, addition in enumerate(layers):
            if addition is None:
                continue
            if id(addition) not in maps:
                A, B, C, _, _ = add_geometry(geom, addition)
                maps[id(addition)] = SupportMap.from_level(SimpleNamespace(tris=SimpleNamespace(A=A, B=B, C=C)),
                                                           depth, **kw)
            result._mechanism_maps[k] = maps[id(addition)]
        if layers[1] is not None:
            t = layers[1]  # per-triangle AABBs for the reference-only locality bound
            result._paper_regions = np.stack((t.min(1), t.max(1)), axis=1)
        from .fastreach import FAN_R_NOM, fan_dv
        boxes, _ = fan_layer(tracker)
        if len(boxes):
            result._fans = (boxes, fan_dv(BALLS[1]), FAN_R_NOM)
            ctr, extent = boxes[:, :3], boxes[:, 15:18] + FAN_R_NOM
            result._paper_regions = np.concatenate((result._paper_regions,
                                                    np.stack((ctr - extent, ctr + extent), axis=1)))
    return result
