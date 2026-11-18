"""Fast field-based reachability perception (prototype): precomputed per-level fields + an analytic per-state query.

reach.py rolls key plans out step by step (0.3-0.7 s per state). This module answers the same questions from fields
built once per level and closed-form flight / rolling formulas, in tens of microseconds per state:

    fr = FastReach.for_tracker(tracker)                  # builds or loads runtime/fastreach/L01.npz
    vec, info = fr.perceive(pos, vel, ball_type, quarter, debug=True)

Fields (all static geometry, stoppers excluded; moving bodies are not modelled):

* H, layered heightfield: 0.5-unit x-z cells in 8 x 8 tiles (block-sparse: only tiles near walkable surface or
  rails). Per cell up to 3 layers (surfaces closer than 4 in one cell keep only the top one: the ball cannot fit
  between): the highest walkable (n_y >= 0.5) surface sample of the layer and its face normal (int8 nx, nz).
  Rails are not in H (they are analytic, below).
* Min/max pyramid over H: 2 x 2 merges per level; levels 1-2 inside each tile, level 3 = the tile, levels 4+ dense
  grids. Empty nodes are (inf, -inf).
* R, sphere-rest field (derived from H and the rails, ball radius 2): per cell up to 4 clusters of supports
  (sources grouped by height, a new cluster below a 1.0 gap): the highest centre height a ball resting on that
  cluster has at this column (point contact h + sqrt(r^2 - d^2) or face contact on the source's plane, or a rail
  tube), the lowest centre height at which the cluster's sources still intersect the sphere ("low": a ball at low <
  y < rest penetrates it, i.e. it blocks), the contact normal and the contact kind (face / edge / rail). Ground
  queries (support, edges, steps, braking paths) are O(1) lookups in R.
* SDF: sparse narrow-band distance field of the static mesh (rails excluded): 0.5 voxels in 8^3-voxel bricks
  (stored as 9^3 corner samples so trilinear sampling never leaves a brick), bricks within 3 of a triangle,
  distances clamped to 3. Unsigned: L1's meshes are open in places, so a sign from the nearest face's winding is
  inconsistent between neighbouring corners, and the queries only need the distance from outside. int16 fixed
  point (1/4096): numba cannot index float16, and int16 has the same size. Distance and gradient (normal) from
  trilinear sampling.
* Rails: analytic tube segments (axis segments + radius) extracted from the Phys_FloorRails meshes (vertex
  normals: axis point = vertex - R n, R fitted per mesh component); a component that does not fit a tube falls
  back into H / R / SDF as ordinary triangles. Per-tile segment lists (CSR).
* Death boxes: the large DepthTestCubes (horizontal extent >= 20; the small DEPT entries are movable props); a
  ball dies with its centre in a box footprint below its top + 2 (reach.ReachConfig.death_margin), or anywhere
  30 below the lowest surface.
* Unknown zones: rest-pose bounding boxes (inflated by 4) of the physicalized non-fixed entities (moving
  platforms, boxes, bridges): the static fields do not see them, the query flags a ball inside one.

Query (perceive): no key enumeration, no trajectory search.
* Air: the height after n PSIs does not depend on the keys, y(n) = y0 + vy S1(n) - g S2(n); the key-free
  horizontal position is x0 + v S1(n); keys add at most D(n) = (F/m) S2(n) per world axis (keys are world-axis
  aligned at every camera quarter), so the reachable centres at PSI n form a square. A surface at height h is a
  reachable landing if at some PSI on the descending branch (centre between h + r and h) it is within the sphere
  contact radius sqrt(r^2 - (y - h)^2) of that square. A depth-first walk of the pyramid prunes nodes by their
  time window (binary search on y(n)) and the square swept over it. Outputs: any landing (none = doomed), the
  earliest and the least-effort landing (closest to the key-free flight), whether the ball can stop after the
  least-effort landing (brake distance vs the support ahead in R) and whether its flight path is clear (SDF and
  rails at 10 points).
* Ground, 8 key directions (camera frame): march R and the SDF along the direction: distance to the edge (support
  lost), brake distance at full reverse push (closed form per world axis, slope gravity from the support normal)
  and the margin, for a gap ahead the takeoff speed needed to clear it (flight formula inversely: bisection, ramp
  lift from the takeoff face, full air push) vs the speed a full push reaches at the edge (closed form), for a
  step / wall ahead its height and mount_speed vs the reachable speed.
* Contacts: support normal (R) and side normal (SDF probes inside the ball, rails), camera frame.
* Doomed: airborne with no reachable landing before the centre falls below the death height (no landing within the
  3 s horizon while the death height is not reached either = unknown: unknown_zone is set, doomed is not); grounded
  (not on rails): the full brake path (closed form, sampled
  on R) leaves the support and neither the brake departure nor a full-push takeoff along the velocity has a
  reachable landing. On rails a grounded ball is never doomed (a straight brake path leaves a curved rail: unknown).

Numerics: the kernels are scalar functions compiled by numba when it is installed (optional dependency group
`fastreach`); with BALLRAIN_NO_NUMBA=1 or without numba the same functions run as plain Python on the same numpy
arrays (identical float64 arithmetic, ~1000x slower; for tests).

Constants: reach.py (air: v' = (1 - c/66) v + (F/m) u + (0, -20/66, 0), p[k+1] = p[k] + v[k]/66; ground: rolling
sphere F/(1.4 m), (c_lin + 0.4 c_rot)/1.4; mount_speed; BALLS).
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np

from . import reach, tas
from .reach import BALLS, KEY_COMBOS

BUILD_VERSION = 4

# Physics constants of reach.py, as literals HERE on purpose: numba (cache=True) freezes globals into the compiled
# kernels and invalidates its on-disk cache only when this file changes, so a constant imported from reach.py would
# go stale in the numba path (not in the plain-Python path) after an edit to reach.py. Changing one of these means
# editing this file too, which recompiles; the assertion makes a mismatch fail at import.
GRAVITY = 20.0
PSI_HZ = 66.0
ROLL_INERTIA = 1.4
UP_FACE = 0.5
MOUNT_CAP = 1.95
MOUNT_FREE = 0.1
CONSTS = (GRAVITY, PSI_HZ, ROLL_INERTIA, UP_FACE, MOUNT_CAP, MOUNT_FREE)
if CONSTS != (reach.GRAVITY, reach.PSI_HZ, reach.ROLL_INERTIA, reach.UP_FACE, reach.MOUNT_CAP, reach.MOUNT_FREE):
    raise ImportError("ballrain.fastreach: physics constants differ from ballrain.reach; update both "
                      f"(fastreach {CONSTS})")

try:
    if os.environ.get("BALLRAIN_NO_NUMBA"):
        raise ImportError("disabled by BALLRAIN_NO_NUMBA")
    import numba
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    numba = None
    HAVE_NUMBA = False


def _jit(fn):
    return numba.njit(cache=True)(fn) if HAVE_NUMBA else fn


NAN = math.nan
INF = math.inf

# ------------------------------------------------------------------------------------------------ constants
RES = 0.5  # x-z cell size (H, R)
TL = 8  # tile edge in cells (4 units)
KH = 3  # H layers per cell
KR = 4  # R clusters per cell
H_LAYER_GAP = 4.0  # surfaces closer than this in one cell: only the top one (the ball cannot fit between)
R_SRC_GAP = 1.0  # R: sources more than this apart in height start a new cluster
R_BALL = 2.0  # R is built for this radius (wood, stone; paper 2.004)
VRES = 0.5  # SDF voxel
BRICK = 8  # SDF brick edge in voxels (4 units)
SDF_BAND = 3.0  # SDF bricks within this of a triangle; distances clamped to +-SDF_BAND
SDF_SCALE = 4096.0  # int16 fixed point
N_PSI = 396  # airborne horizon: 3 game s
GPSI = GRAVITY / PSI_HZ  # gravity per PSI (units / phys-s per PSI)
DEATH_MARGIN = 2.0
VOID_DROP = 30.0
GROUND_TOL = 0.3  # centre within this of a rest height = on the ground
DROP_TOL = 1.0  # support falling away by more than this = edge
UP_TOL = 0.15  # rise a holding contact may take per step (plus the face's climb)
PEN_TOL = 0.2  # SDF penetration that counts as a wall
DS = 0.5  # march step
L_MAX = 20.0  # march range (edge / obstacle distances are clipped to it)
GAP_MAX = 40.0  # how far past an edge to look for the far side
GAP_DROP = 40.0  # far side at most this below the takeoff (a lower platform after a drop counts)
CONTACT_TOL = 0.1  # landing contact radius r + this
V_MAX = 60.0  # takeoff speed search range
CLIP_V = 20.0  # speed deficits are clipped to +-CLIP_V (+ = impossible, - = nothing to clear)
FALL_TOL = 3.0  # grounded needs vy >= the support plane's vertical speed - this
UNKNOWN_R = 8.0  # a moving body this close to the ball: unknown zone
EXCL_R = 2.5  # takeoff scans ignore landings within this of the takeoff point (the edge just left)

DIR_NAMES = ("up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right")
DIR_FIELDS = ("edge", "brake_margin", "gap_deficit", "obst", "obst_deficit")
HEAD_NAMES = ("grounded", "doomed", "unknown_zone",
              "air_any", "air_t_first", "air_first_dy", "air_near_r", "air_near_f", "air_near_dy", "air_t_near",
              "air_near_effort", "air_near_can_stop", "air_near_clear",
              "sup_n_r", "sup_n_y", "sup_n_f", "side_n_r", "side_n_y", "side_n_f")
FEATURE_NAMES = HEAD_NAMES + tuple(f"{d}_{f}" for d in DIR_NAMES for f in DIR_FIELDS)
N_FEATURES = len(FEATURE_NAMES)
N_HEAD = len(HEAD_NAMES)
# debug array layout (float64)
DBG_NAMES = ("grounded", "sup_rest", "sup_flag", "airborne_any", "n_apex", "n_stop", "n_first", "first_x", "first_h",
             "first_z", "n_near", "near_x", "near_h", "near_z", "near_effort", "land_cx", "land_cy", "land_cz",
             "land_vx", "land_vz", "land_brake", "land_edge", "clear_min", "doom_reason", "brake_ok", "brake_takeoff",
             "jump_takeoff", "jump_edge", "jump_v", "visited", "speed")
DBG = {n: i for i, n in enumerate(DBG_NAMES)}
N_DBG = len(DBG_NAMES)
PER_DIR_DBG = ("edge_s", "obst_s", "obst_h", "mount_v", "brake_along", "v_can_edge", "v_can_obst", "s_near", "s_far",
               "h_far", "v_need", "v_need_coast", "takeoff_k")
N_PDD = len(PER_DIR_DBG)


def air_tables(ball, n: int = N_PSI):
    """S1(n) = sum_{k<n} lam^k / 66 (gain of the initial velocity), S2(n) = sum_{k<n} (1 - lam^k)/(1 - lam) / 66
    (gain of a constant per-PSI acceleration of 1), n = 0..N."""
    lam = ball.air_lambda
    k = np.arange(n + 1, dtype=np.float64)
    geo = (1.0 - lam ** k) / (1.0 - lam)
    return geo / PSI_HZ, (k - geo) / (1.0 - lam) / PSI_HZ


def mount_speed_scalar(h, r=R_BALL):
    """reach.mount_speed for one value (python float)."""
    return float(reach.mount_speed(h, r))


# ======================================================================================== scalar kernels

@_jit
def _mount_speed(h, r):
    if h <= MOUNT_FREE:
        return 0.0
    if h >= MOUNT_CAP or h >= r:
        return INF
    f = 1.0 - 5.0 * h / (7.0 * r)
    if f < 1e-6:
        f = 1e-6
    return math.sqrt(GRAVITY * h / (0.7 * f * f))


@_jit
def _cidx(x, o):
    return int(math.floor((x - o) / RES + 0.5))


@_jit
def _tile_of(tiles, i, j):
    if i < 0 or j < 0:
        return -1
    ti = i // TL
    tj = j // TL
    if ti >= tiles.shape[0] or tj >= tiles.shape[1]:
        return -1
    return int(tiles[ti, tj])


@_jit
def _h_top(tiles, Hh, ox, oz, x, z, lo, hi):
    """Highest H layer in [lo, hi] at the cell of (x, z) (nan none)."""
    i = _cidx(x, ox)
    j = _cidx(z, oz)
    t = _tile_of(tiles, i, j)
    if t < 0:
        return NAN
    li = i % TL
    lj = j % TL
    for k in range(KH):
        h = float(Hh[t, li, lj, k])
        if h != h:
            break
        if lo <= h <= hi:
            return h
    return NAN


@_jit
def _rail_axis_near(seg, rstart, ritems, ntz, ox, oz, x, y, z):
    """Nearest rail axis point to (x, y, z) among the segments listed in the tile of (x, z): (distance to the axis,
    axis point x, y, z, tube radius); distance inf when none listed."""
    ti = int(math.floor((x - ox + 0.5 * RES) / (RES * TL)))
    tj = int(math.floor((z - oz + 0.5 * RES) / (RES * TL)))
    best = INF
    bx = 0.0
    by = 0.0
    bz = 0.0
    br = 0.0
    if ti < 0 or tj < 0 or tj >= ntz or (ti * ntz + tj + 1) >= len(rstart):
        return best, bx, by, bz, br
    c = ti * ntz + tj
    for q in range(int(rstart[c]), int(rstart[c + 1])):
        s = int(ritems[q])
        ax = seg[s, 0]
        ay = seg[s, 1]
        az = seg[s, 2]
        dx = seg[s, 3] - ax
        dy = seg[s, 4] - ay
        dz = seg[s, 5] - az
        L2 = dx * dx + dy * dy + dz * dz
        t = 0.0
        if L2 > 1e-12:
            t = ((x - ax) * dx + (y - ay) * dy + (z - az) * dz) / L2
            t = min(max(t, 0.0), 1.0)
        px = ax + t * dx
        py = ay + t * dy
        pz = az + t * dz
        d = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
        if d < best:
            best = d
            bx = px
            by = py
            bz = pz
            br = seg[s, 6]
    return best, bx, by, bz, br


@_jit
def _rail_top_near(seg, rstart, ritems, ntz, ox, oz, x, z, rad, lo, hi):
    """Highest rail top (axis y + radius) in [lo, hi] among segments whose x-z projection passes within rad of
    (x, z) (nan none)."""
    sx = x - ox + 0.5 * RES
    sz = z - oz + 0.5 * RES
    ti = int(math.floor(sx / (RES * TL)))
    tj = int(math.floor(sz / (RES * TL)))
    best = NAN
    if ti < 0 or tj < 0 or tj >= ntz or (ti * ntz + tj + 1) >= len(rstart):
        return best
    c = ti * ntz + tj
    for q in range(int(rstart[c]), int(rstart[c + 1])):
        s = int(ritems[q])
        ax = seg[s, 0]
        az = seg[s, 2]
        dx = seg[s, 3] - ax
        dz = seg[s, 5] - az
        L2 = dx * dx + dz * dz
        t = 0.0
        if L2 > 1e-12:
            t = ((x - ax) * dx + (z - az) * dz) / L2
            t = min(max(t, 0.0), 1.0)
        px = ax + t * dx
        pz = az + t * dz
        if (x - px) ** 2 + (z - pz) ** 2 <= rad * rad:
            top = seg[s, 1] + t * (seg[s, 4] - seg[s, 1]) + seg[s, 6]
            if lo <= top <= hi and (best != best or top > best):
                best = top
    return best


@_jit
def _rail_dist(seg, rstart, ritems, ntz, ox, oz, x, y, z):
    """Distance from (x, y, z) to the nearest rail tube surface and the unit normal (surface -> point)."""
    d, px, py, pz, rr = _rail_axis_near(seg, rstart, ritems, ntz, ox, oz, x, y, z)
    if d == INF:
        return INF, 0.0, 1.0, 0.0
    nx = x - px
    ny = y - py
    nz = z - pz
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln < 1e-9:
        return d - rr, 0.0, 1.0, 0.0
    return d - rr, nx / ln, ny / ln, nz / ln


@_jit
def _sdf(bidx, bricks, sx0, sy0, sz0, x, y, z):
    """Trilinear SDF and its gradient at (x, y, z): (d, gx, gy, gz); (SDF_BAND, 0, 0, 0) outside the bricks."""
    fx = (x - sx0) / VRES
    fy = (y - sy0) / VRES
    fz = (z - sz0) / VRES
    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    iz = int(math.floor(fz))
    bx = ix // BRICK
    by = iy // BRICK
    bz = iz // BRICK
    if bx < 0 or by < 0 or bz < 0 or bx >= bidx.shape[0] or by >= bidx.shape[1] or bz >= bidx.shape[2]:
        return SDF_BAND, 0.0, 0.0, 0.0
    b = int(bidx[bx, by, bz])
    if b < 0:
        return SDF_BAND, 0.0, 0.0, 0.0
    lx = ix - bx * BRICK
    ly = iy - by * BRICK
    lz = iz - bz * BRICK
    tx = fx - ix
    ty = fy - iy
    tz = fz - iz
    c000 = float(bricks[b, lx, ly, lz]) / SDF_SCALE
    c100 = float(bricks[b, lx + 1, ly, lz]) / SDF_SCALE
    c010 = float(bricks[b, lx, ly + 1, lz]) / SDF_SCALE
    c110 = float(bricks[b, lx + 1, ly + 1, lz]) / SDF_SCALE
    c001 = float(bricks[b, lx, ly, lz + 1]) / SDF_SCALE
    c101 = float(bricks[b, lx + 1, ly, lz + 1]) / SDF_SCALE
    c011 = float(bricks[b, lx, ly + 1, lz + 1]) / SDF_SCALE
    c111 = float(bricks[b, lx + 1, ly + 1, lz + 1]) / SDF_SCALE
    c00 = c000 + (c100 - c000) * tx
    c10 = c010 + (c110 - c010) * tx
    c01 = c001 + (c101 - c001) * tx
    c11 = c011 + (c111 - c011) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    d = c0 + (c1 - c0) * tz
    # analytic gradient of the trilinear interpolant
    gx = ((c100 - c000) * (1 - ty) * (1 - tz) + (c110 - c010) * ty * (1 - tz)
          + (c101 - c001) * (1 - ty) * tz + (c111 - c011) * ty * tz)
    gy = ((c010 - c000) * (1 - tx) * (1 - tz) + (c110 - c100) * tx * (1 - tz)
          + (c011 - c001) * (1 - tx) * tz + (c111 - c101) * tx * tz)
    gz = c1 - c0
    ln = math.sqrt(gx * gx + gy * gy + gz * gz)
    if ln > 1e-12:
        gx /= ln
        gy /= ln
        gz /= ln
    return d, gx, gy, gz


@_jit
def _udist(bidx, bricks, sx0, sy0, sz0, x, y, z):
    """Unsigned distance to the static mesh and its gradient (pointing away from the nearest surface): the queries
    never need the inside, and the sign of the stored field depends on the mesh winding."""
    d, gx, gy, gz = _sdf(bidx, bricks, sx0, sy0, sz0, x, y, z)
    if d < 0.0:
        return -d, -gx, -gy, -gz
    return d, gx, gy, gz


@_jit
def _r_read(Rr, Rl, Rn, Rf, t, li, lj, k):
    rest = float(Rr[t, li, lj, k])
    low = float(Rl[t, li, lj, k])
    nx = float(Rn[t, li, lj, k, 0]) / 127.0
    nz = float(Rn[t, li, lj, k, 1]) / 127.0
    ny = math.sqrt(max(0.0, 1.0 - nx * nx - nz * nz))
    fl = int(Rf[t, li, lj, k]) & 3
    return rest, low, nx, ny, nz, fl


@_jit
def _r_ground(tiles, Rr, Rl, Rn, Rf, ox, oz, x, z, y, tol):
    """Rest cluster at the ball's column within tol of its height (the highest such): (rest, nx, ny, nz, flag);
    rest nan when none."""
    i = _cidx(x, ox)
    j = _cidx(z, oz)
    t = _tile_of(tiles, i, j)
    if t < 0:
        return NAN, 0.0, 1.0, 0.0, -1
    li = i % TL
    lj = j % TL
    for k in range(KR):
        rest, low, nx, ny, nz, fl = _r_read(Rr, Rl, Rn, Rf, t, li, lj, k)
        if rest != rest:
            break
        if abs(rest - y) <= tol:
            return rest, nx, ny, nz, fl
    return NAN, 0.0, 1.0, 0.0, -1


@_jit
def _r_below(tiles, Rr, Rl, Rn, Rf, ox, oz, x, z, y):
    """Highest rest at most y + GROUND_TOL at the column (nan none)."""
    i = _cidx(x, ox)
    j = _cidx(z, oz)
    t = _tile_of(tiles, i, j)
    if t < 0:
        return NAN
    li = i % TL
    lj = j % TL
    for k in range(KR):
        rest = float(Rr[t, li, lj, k])
        if rest != rest:
            break
        if rest <= y + GROUND_TOL:
            return rest
    return NAN


@_jit
def _r_hold(tiles, Rr, Rl, Rn, Rf, ox, oz, x, z, y, drop_tol, ux, uz, step):
    """A ball at height y moves to column (x, z) along (ux, uz) by `step`. Returns (hold, nx, ny, nz, flag, block):
    hold = the highest cluster rest it keeps contact with (in [y - drop_tol, y + allow]; allow = UP_TOL plus the
    face's climb along the move for face contacts, plus 0.5 per unit for rails, nothing more for edges); block = the
    lowest rest of a cluster that would penetrate the ball at its new height (rest above it by more than the
    allowance and low below it), nan when none; wall = the holding cluster's "non-walkable geometry within reach"
    bit (the SDF is consulted only there)."""
    i = _cidx(x, ox)
    j = _cidx(z, oz)
    t = _tile_of(tiles, i, j)
    if t < 0:
        return NAN, 0.0, 1.0, 0.0, -1, NAN, False
    li = i % TL
    lj = j % TL
    hold = NAN
    hnx = 0.0
    hny = 1.0
    hnz = 0.0
    hfl = -1
    hwall = False
    for k in range(KR):
        rest, low, nx, ny, nz, fl = _r_read(Rr, Rl, Rn, Rf, t, li, lj, k)
        if rest != rest:
            break
        if fl == 1:
            allow = UP_TOL + step * max(0.0, -(nx * ux + nz * uz)) / max(ny, 0.2)
        elif fl == 2:
            allow = UP_TOL + 0.5 * step
        else:
            allow = UP_TOL
        if y - drop_tol <= rest <= y + allow:
            if hold != hold or rest > hold:
                hold = rest
                hnx = nx
                hny = ny
                hnz = nz
                hfl = fl
                hwall = (int(Rf[t, li, lj, k]) & 4) != 0
    ynew = hold if hold == hold else y
    block = NAN
    for k in range(KR):
        rest, low, nx, ny, nz, fl = _r_read(Rr, Rl, Rn, Rf, t, li, lj, k)
        if rest != rest:
            break
        if fl == 1:
            allow = UP_TOL + step * max(0.0, -(nx * ux + nz * uz)) / max(ny, 0.2)
        elif fl == 2:
            allow = UP_TOL + 0.5 * step
        else:
            allow = UP_TOL
        if rest > ynew + allow and low < ynew - 1e-3:
            if block != block or rest < block:
                block = rest
    return hold, hnx, hny, hnz, hfl, block, hwall


@_jit
def _h_top_disc(tiles, Hh, seg, rstart, ritems, ntz, ox, oz, x, z, rad, lo, hi):
    """Highest H layer or rail top in [lo, hi] within rad of (x, z) (nan none)."""
    m = int(math.ceil(rad / RES))
    i0 = _cidx(x, ox)
    j0 = _cidx(z, oz)
    best = NAN
    for di in range(-m, m + 1):
        for dj in range(-m, m + 1):
            if (di * di + dj * dj) * RES * RES > rad * rad + 1e-9:
                continue
            t = _tile_of(tiles, i0 + di, j0 + dj)
            if t < 0:
                continue
            li = (i0 + di) % TL
            lj = (j0 + dj) % TL
            for k in range(KH):
                h = float(Hh[t, li, lj, k])
                if h != h:
                    break
                if lo <= h <= hi and (best != best or h > best):
                    best = h
    rt = _rail_top_near(seg, rstart, ritems, ntz, ox, oz, x, z, rad, lo, hi)
    if rt == rt and (best != best or rt > best):
        best = rt
    return best


@_jit
def _death_y(boxes, void_y, x, z):
    d = void_y
    for b in range(boxes.shape[0]):
        if boxes[b, 0] <= x <= boxes[b, 3] and boxes[b, 2] <= z <= boxes[b, 5]:
            d = max(d, boxes[b, 4] + DEATH_MARGIN)
    return d


@_jit
def _death_floor_box(boxes, void_y, x0, x1, z0, z1):
    """Lowest death height over an x-z box (boxes it touches; the void floor unless it is inside one box)."""
    d = INF
    inside = False
    for b in range(boxes.shape[0]):
        if x1 >= boxes[b, 0] and x0 <= boxes[b, 3] and z1 >= boxes[b, 2] and z0 <= boxes[b, 5]:
            d = min(d, boxes[b, 4] + DEATH_MARGIN)
            if x0 >= boxes[b, 0] and x1 <= boxes[b, 3] and z0 >= boxes[b, 2] and z1 <= boxes[b, 5]:
                inside = True
    if not inside:
        d = min(d, void_y)
    return d


@_jit
def _first_le(y, a, b, v):
    """First n in [a, b] with y[n] <= v (y non-increasing on [a, b]); b + 1 when none."""
    if a > b or y[b] > v:
        return b + 1
    lo = a
    hi = b
    while lo < hi:
        m = (lo + hi) // 2
        if y[m] <= v:
            hi = m
        else:
            lo = m + 1
    return lo


@_jit
def _last_ge(y, a, b, v):
    """Last n in [a, b] with y[n] >= v (y non-increasing on [a, b]); a - 1 when none."""
    if a > b or y[a] < v:
        return a - 1
    lo = a
    hi = b
    while lo < hi:
        m = (lo + hi + 1) // 2
        if y[m] >= v:
            lo = m
        else:
            hi = m - 1
    return lo


@_jit
def _brake_axis(v0, g_t, a_g, lam, t):
    """One world axis under full reverse push (a_g per PSI) plus a constant slope term g_t: (displacement after
    min(t, stop) PSIs, velocity then, stop PSI). Stop PSI inf when the slope overpowers the brake."""
    if abs(v0) < 1e-9:
        return 0.0, 0.0, 0.0
    s = 1.0 if v0 > 0 else -1.0
    a = -s * a_g + g_t
    one = 1.0 - lam
    if a * s >= 0.0:
        k = t
        kstop = INF
    else:
        A = a / one
        kstop = math.log(-A / (v0 - A)) / math.log(lam)
        k = min(t, kstop)
    lk = lam ** k
    geo = (1.0 - lk) / one
    d = (v0 * geo + a * (k - geo) / one) / PSI_HZ
    v = lk * v0 + a * geo
    if k >= kstop:
        v = 0.0
    return d, v, kstop


@_jit
def _disp1(v0, a, lam, k):
    one = 1.0 - lam
    lk = lam ** k
    geo = (1.0 - lk) / one
    return (v0 * geo + a * (k - geo) / one) / PSI_HZ, lk * v0 + a * geo


@_jit
def _v_reach(v0, a, lam, L):
    """1-D rolling with v' = lam v + a: speed when the displacement first reaches L (> 0); 0 when it never does."""
    if L <= 0.0:
        return max(v0, 0.0)
    one = 1.0 - lam
    if a < 0.0 or (a == 0.0 and v0 > 0.0):
        if v0 <= 0.0:
            return 0.0
        if a < 0.0:
            A = a / one
            ks = math.log(-A / (v0 - A)) / math.log(lam)
        else:
            ks = 1e6
        dmax, _ = _disp1(v0, a, lam, ks)
        if dmax < L:
            return 0.0
        lo = 0.0
        hi = ks
    elif a == 0.0:
        return 0.0
    else:
        hi = 1.0
        while True:
            d, _ = _disp1(v0, a, lam, hi)
            if d >= L:
                break
            hi *= 2.0
            if hi > 1e6:
                return 0.0
        lo = hi * 0.5 if hi > 1.0 else 0.0
    k = hi
    for _ in range(40):  # safeguarded Newton (d disp / dk = v / 66)
        d, v = _disp1(v0, a, lam, k)
        f = d - L
        if abs(f) < 1e-7:
            break
        if f > 0.0:
            hi = k
        else:
            lo = k
        kn = k - f * PSI_HZ / v if v > 1e-9 else -1.0
        if not (lo < kn < hi):
            kn = 0.5 * (lo + hi)
        k = kn
    _, v = _disp1(v0, a, lam, k)
    return v


@_jit
def _clear_ok(S1, S2, N, v, k, vy_abs, a_p, x_need, dy_t):
    """Takeoff at horizontal speed v (lift k v): is the centre at least dy_t above the takeoff height when it has
    travelled x_need horizontally (full air push a_p along the motion)?"""
    if v * S1[N] + a_p * S2[N] < x_need:
        return False
    lo = 0
    hi = N
    while lo < hi:
        m = (lo + hi) // 2
        if v * S1[m] + a_p * S2[m] >= x_need:
            hi = m
        else:
            lo = m + 1
    yv = (k * v + vy_abs) * S1[lo] - GPSI * S2[lo]
    return yv >= dy_t


@_jit
def _v_need(S1, S2, N, k, vy_abs, a_p, x_need, dy_t):
    """Smallest horizontal takeoff speed v that clears a gap of x_need to a far side dy_t above the takeoff (inf:
    none <= V_MAX); takeoff vertical speed k v + vy_abs (k: lift of the takeoff face), a_p: air push along the
    motion (per-axis keys: 1 along a world axis, sqrt 2 along a diagonal, |ux| + |uz| in general)."""
    if _clear_ok(S1, S2, N, 0.0, k, vy_abs, a_p, x_need, dy_t):
        return 0.0
    if not _clear_ok(S1, S2, N, V_MAX, k, vy_abs, a_p, x_need, dy_t):
        return INF
    lo = 0.0
    hi = V_MAX
    for _ in range(30):
        m = 0.5 * (lo + hi)
        if _clear_ok(S1, S2, N, m, k, vy_abs, a_p, x_need, dy_t):
            hi = m
        else:
            lo = m
    return hi


# --------------------------------------------------------------------------------------------- the march

@_jit
def _march(tiles, Hh, Rr, Rl, Rn, Rf, bidx, bricks, seg, rstart, ritems, ntz, ox, oz, sx0, sy0, sz0, r,
           x0, y0, z0, ux, uz, lmax, want_far, k0, out):
    """Walk the ball from (x0, y0, z0) along the horizontal unit direction (ux, uz) in steps of DS following R.
    k0: the start support's lift along the motion (nan: 0 until a face is held).
    out (float64, >= 9): edge_s (support lost; inf none), y_edge (centre height on the plateau before it: the last
    face or rail contact), k_edge (lift per unit
    of the support face along the motion at the last held column), obst_s (distance walked before a barrier; inf
    none), obst_h (barrier top above the ball bottom; inf = no top within 4.5), s_near (raw near edge), s_far (raw
    far side start; inf none), h_far (its surface height), y_last (height at the end)."""
    y = y0
    edge_s = INF
    y_edge = NAN
    k_edge = 0.0
    cur_k = k0 if k0 == k0 else 0.0
    y_face = y0
    obst_s = INF
    obst_h = NAN
    s_near = NAN
    s_far = INF
    h_far = NAN
    nsteps = int(lmax / DS + 1e-9)
    for st in range(1, nsteps + 1):
        s = st * DS
        x = x0 + ux * s
        z = z0 + uz * s
        hold, nx, ny, nz, fl, block, wall = _r_hold(tiles, Rr, Rl, Rn, Rf, ox, oz, x, z, y, DROP_TOL, ux, uz, DS)
        ynew = hold if hold == hold else y
        obst = block == block
        if not obst and wall:
            d, gx, gy, gz = _udist(bidx, bricks, sx0, sy0, sz0, x, ynew, z)
            if d < r - PEN_TOL and gx * ux + gz * uz < -0.3:
                obst = True
        if obst:
            obst_s = s - DS
            ht = _h_top_disc(tiles, Hh, seg, rstart, ritems, ntz, ox, oz, x, z, r, y - r + 0.05, y - r + 8.0)
            obst_h = ht - (y - r) if ht == ht else INF
            break
        if hold != hold:
            edge_s = s
            y_edge = y_face
            k_edge = cur_k
            break
        y = hold
        if fl != 0:  # the plateau: the last face or rail contact (an edge contact is the ball rolling over a corner)
            y_face = y
        if fl == 1:  # the last face's lift along the motion; kept over the edge contacts at a ramp's lip
            cur_k = -(nx * ux + nz * uz) / max(ny, 0.2)
        elif fl == 2:
            cur_k = 0.0
    out[0] = edge_s
    out[1] = y_edge
    out[2] = k_edge
    out[3] = obst_s
    out[4] = obst_h
    out[8] = y
    if want_far and edge_s < INF:
        hb = y_edge - r
        # raw near edge: last column (from the edge back up to 3 units) whose surface is the one held
        s_near = edge_s - 1.73
        sb = edge_s
        while sb >= max(0.0, edge_s - 3.0):
            h = _h_top(tiles, Hh, ox, oz, x0 + ux * sb, z0 + uz * sb, hb - 0.8, hb + 0.8)
            if h == h:
                s_near = sb
                break
            sb -= DS
        # far side: first column past the near edge with a surface (or rail) in the takeoff window
        sf = s_near + DS
        while sf <= edge_s + GAP_MAX:
            xx = x0 + ux * sf
            zz = z0 + uz * sf
            h = _h_top(tiles, Hh, ox, oz, xx, zz, hb - GAP_DROP, hb + 4.5)
            rt = _rail_top_near(seg, rstart, ritems, ntz, ox, oz, xx, zz, DS + 0.05, hb - GAP_DROP, hb + 4.5)
            if rt == rt and (h != h or rt > h):
                h = rt
            if h == h:
                s_far = sf
                h_far = h
                break
            sf += 2.0 * DS  # 1-unit steps: the gap length is known to +-1
    out[5] = s_near
    out[6] = s_far
    out[7] = h_far


# ------------------------------------------------------------------------------------- airborne landing scan

@_jit
def _pyr_node(Hh, P1, P2, Pd, Poff, Pdim, tiles, lev, a, b):
    """(hmin, hmax) of pyramid node (lev, a, b); (inf, -inf) when empty or out of range."""
    if lev >= 3:
        L = lev - 3
        if a < 0 or b < 0 or a >= Pdim[L, 0] or b >= Pdim[L, 1]:
            return INF, -INF
        o = Poff[L] + a * Pdim[L, 1] + b
        return float(Pd[o, 0]), float(Pd[o, 1])
    sh = 3 - lev
    ti = a >> sh
    tj = b >> sh
    if ti < 0 or tj < 0 or ti >= tiles.shape[0] or tj >= tiles.shape[1]:
        return INF, -INF
    t = int(tiles[ti, tj])
    if t < 0:
        return INF, -INF
    la = a - (ti << sh)
    lb = b - (tj << sh)
    if lev == 2:
        return float(P2[t, la, lb, 0]), float(P2[t, la, lb, 1])
    if lev == 1:
        return float(P1[t, la, lb, 0]), float(P1[t, la, lb, 1])
    mn = INF
    mx = -INF
    for k in range(KH):
        h = float(Hh[t, la, lb, k])
        if h != h:
            break
        mn = min(mn, h)
        mx = max(mx, h)
    return mn, mx


@_jit
def _contact_window(yy, xf, zf, D, na, ns, cx, cz, h, rc, hc):
    """Landing on a surface point (cx, h, cz) (hc: half cell, inflates it to a square for the reachability test):
    over the PSIs in [na, ns] at which the centre falls from h + rc to h and the reachable square can bring the sphere
    of contact radius rc onto the point: (first such PSI, least effort, the PSI of that least effort). Effort at PSI n
    = max(0, horizontal distance of the point from the key-free centre - contact radius at that height), i.e. how far
    the keys must move the centre (0 = the key-free flight lands on it); ties go to the earlier PSI.
    (-1, inf, -1) when never."""
    n0 = _first_le(yy, na, ns, h + rc)
    n1 = _last_ge(yy, na, ns, h)
    first = -1
    emin = INF
    ne = -1
    for n in range(n0, n1 + 1):
        dy = yy[n] - h
        rho2 = rc * rc - dy * dy
        if rho2 < 0.0:
            continue
        ddx = max(abs(cx - xf[n]) - D[n] - hc, 0.0)
        ddz = max(abs(cz - zf[n]) - D[n] - hc, 0.0)
        if ddx * ddx + ddz * ddz <= rho2:
            if first < 0:
                first = n
            e = max(0.0, math.sqrt((cx - xf[n]) ** 2 + (cz - zf[n]) ** 2) - math.sqrt(rho2))
            if e < emin:
                emin = e
                ne = n
    return first, emin, ne


@_jit
def _air_scan(tiles, Hh, P1, P2, Pd, Poff, Pdim, seg, rstart, ritems, ntz, ox, oz, boxes, void_y,
              S1, S2, x0, y0, z0, vx, vy, vz, a_air, r, filt, excl, mode, yy, xf, zf, D, stack, stamp, stamp_id,
              collect, res):
    """Reachable landings from an airborne state. filt (4,): x0, x1, z0, z1 of the only surface points that count
    (filt[0] > filt[1]: no filter); excl (3,): x, z, radius of a disc whose surface points do not count (radius 0:
    none); mode 0 = full (earliest and least-effort landing), 1 = stop at the first landing found. collect (M, 5):
    reachable landing cells (x, h, z, n, effort) are written while there is room (len 0: none).
    res (float64, >= 16): found, n_first, first x, h, z, n_near (the PSI of the least effort), near x, h, z, effort,
    n_apex, n_stop, visited nodes, the near landing's contact radius and contact centre height (surface height for a
    cell, the tube axis for a rail), death floor crossed within the horizon (0: the horizon ran out first).
    Returns the number of collected cells."""
    N = S1.shape[0] - 1
    for n in range(N + 1):
        yy[n] = y0 + vy * S1[n] - GPSI * S2[n]
        xf[n] = x0 + vx * S1[n]
        zf[n] = z0 + vz * S1[n]
        D[n] = a_air * S2[n]
    na = 0
    while na < N and yy[na + 1] >= yy[na]:
        na += 1
    na = max(na, 1)
    ns = N
    crossed = False
    for n in range(na, N + 1):
        dfl = _death_floor_box(boxes, void_y, xf[n] - D[n] - r, xf[n] + D[n] + r, zf[n] - D[n] - r, zf[n] + D[n] + r)
        if yy[n] < dfl:
            ns = n - 1
            crossed = True
            break
    best_n = N + 1
    best_e = INF
    fx = NAN
    fh = NAN
    fz = NAN
    ex_ = NAN
    eh_ = NAN
    ez_ = NAN
    en_ = -1
    erc_ = NAN
    eyc_ = NAN
    ncol = 0
    visited = 0
    rc = r + CONTACT_TOL
    hc = 0.5 * RES
    use_f = filt[0] <= filt[1]
    use_x = excl[2] > 0.0
    done = False
    if ns >= na:
        # ---- pyramid walk
        top = Pdim.shape[0] - 1 + 3
        sp = 0
        for a in range(int(Pdim[Pdim.shape[0] - 1, 0])):
            for b in range(int(Pdim[Pdim.shape[0] - 1, 1])):
                stack[sp, 0] = top
                stack[sp, 1] = a
                stack[sp, 2] = b
                sp += 1
        while sp > 0 and not done:
            sp -= 1
            lev = int(stack[sp, 0])
            a = int(stack[sp, 1])
            b = int(stack[sp, 2])
            visited += 1
            hmin, hmax = _pyr_node(Hh, P1, P2, Pd, Poff, Pdim, tiles, lev, a, b)
            if hmin > hmax:
                continue
            span = 1 << lev
            bx0 = ox + (a * span - 0.5) * RES
            bx1 = ox + ((a + 1) * span - 0.5) * RES
            bz0 = oz + (b * span - 0.5) * RES
            bz1 = oz + ((b + 1) * span - 0.5) * RES
            if use_f and (bx1 < filt[0] or bx0 > filt[1] or bz1 < filt[2] or bz0 > filt[3]):
                continue
            n_lo = _first_le(yy, na, ns, hmax + rc)
            n_hi = _last_ge(yy, na, ns, hmin)
            if n_lo > n_hi:
                continue
            Dm = D[n_hi] + rc + hc
            xl = min(xf[n_lo], xf[n_hi])
            xh = max(xf[n_lo], xf[n_hi])
            zl = min(zf[n_lo], zf[n_hi])
            zh = max(zf[n_lo], zf[n_hi])
            if xh + Dm < bx0 or xl - Dm > bx1 or zh + Dm < bz0 or zl - Dm > bz1:
                continue
            if mode == 0 and len(collect) == 0:
                # can this node improve the earliest or the least-effort landing?
                gx = max(0.0, max(bx0 - xh, xl - bx1))
                gz = max(0.0, max(bz0 - zh, zl - bz1))
                e_lb = max(0.0, math.sqrt(gx * gx + gz * gz) - rc - hc)
                # (least effort, ties to the earlier PSI: a node can still win a tie at an earlier PSI)
                if n_lo >= best_n and en_ >= 0 and (e_lb > best_e or (e_lb == best_e and n_lo >= en_)):
                    continue
            if lev > 0:
                for ca in range(2):
                    for cb in range(2):
                        stack[sp, 0] = lev - 1
                        stack[sp, 1] = 2 * a + ca
                        stack[sp, 2] = 2 * b + cb
                        sp += 1
                continue
            # ---- a cell: its layers
            t = _tile_of(tiles, a, b)
            cx = ox + a * RES
            cz = oz + b * RES
            if use_x and (cx - excl[0]) ** 2 + (cz - excl[1]) ** 2 <= excl[2] * excl[2]:
                continue
            for k in range(KH):
                h = float(Hh[t, a % TL, b % TL, k])
                if h != h:
                    break
                n1, e1, ne1 = _contact_window(yy, xf, zf, D, na, ns, cx, cz, h, rc, hc)
                if n1 < 0:
                    continue
                if yy[n1] < _death_y(boxes, void_y, cx, cz):
                    continue
                if ncol < collect.shape[0]:
                    collect[ncol, 0] = cx
                    collect[ncol, 1] = h
                    collect[ncol, 2] = cz
                    collect[ncol, 3] = n1
                    collect[ncol, 4] = e1
                    ncol += 1
                if n1 < best_n:
                    best_n = n1
                    fx = cx
                    fh = h
                    fz = cz
                if e1 < best_e or (e1 == best_e and ne1 < en_):
                    best_e = e1
                    ex_ = cx
                    eh_ = h
                    ez_ = cz
                    en_ = ne1
                    erc_ = rc
                    eyc_ = h
                if mode == 1:
                    done = True
                    break
        # ---- rails: axis samples every <= 0.5 with contact radius r + tube radius
        if not done and seg.shape[0] > 0:
            Dm = D[ns] + r + 1.0
            xl = min(xf[na], xf[ns]) - Dm
            xh = max(xf[na], xf[ns]) + Dm
            zl = min(zf[na], zf[ns]) - Dm
            zh = max(zf[na], zf[ns]) + Dm
            if use_f:
                xl = max(xl, filt[0] - 1.0)
                xh = min(xh, filt[1] + 1.0)
                zl = max(zl, filt[2] - 1.0)
                zh = min(zh, filt[3] + 1.0)
            ti0 = int(math.floor((xl - ox + 0.5 * RES) / (RES * TL)))
            ti1 = int(math.floor((xh - ox + 0.5 * RES) / (RES * TL)))
            tj0 = int(math.floor((zl - oz + 0.5 * RES) / (RES * TL)))
            tj1 = int(math.floor((zh - oz + 0.5 * RES) / (RES * TL)))
            ntx = (len(rstart) - 1) // ntz
            for ti in range(max(ti0, 0), min(ti1, ntx - 1) + 1):
                if done:
                    break
                for tj in range(max(tj0, 0), min(tj1, ntz - 1) + 1):
                    if done:
                        break
                    c = ti * ntz + tj
                    for q in range(int(rstart[c]), int(rstart[c + 1])):
                        s = int(ritems[q])
                        if stamp[s] == stamp_id:
                            continue
                        stamp[s] = stamp_id
                        L = math.sqrt((seg[s, 3] - seg[s, 0]) ** 2 + (seg[s, 4] - seg[s, 1]) ** 2
                                      + (seg[s, 5] - seg[s, 2]) ** 2)
                        m = int(math.ceil(L / RES))
                        rcr = r + seg[s, 6] + CONTACT_TOL
                        for u in range(m + 1):
                            f = u / max(m, 1)
                            px = seg[s, 0] + f * (seg[s, 3] - seg[s, 0])
                            py = seg[s, 1] + f * (seg[s, 4] - seg[s, 1])
                            pz = seg[s, 2] + f * (seg[s, 5] - seg[s, 2])
                            if use_f and (px < filt[0] or px > filt[1] or pz < filt[2] or pz > filt[3]):
                                continue
                            if use_x and (px - excl[0]) ** 2 + (pz - excl[1]) ** 2 <= excl[2] * excl[2]:
                                continue
                            n1, e1, ne1 = _contact_window(yy, xf, zf, D, na, ns, px, pz, py, rcr, 0.0)
                            if n1 < 0:
                                continue
                            if yy[n1] < _death_y(boxes, void_y, px, pz):
                                continue
                            if ncol < collect.shape[0]:
                                collect[ncol, 0] = px
                                collect[ncol, 1] = py + seg[s, 6]
                                collect[ncol, 2] = pz
                                collect[ncol, 3] = n1
                                collect[ncol, 4] = e1
                                ncol += 1
                            if n1 < best_n:
                                best_n = n1
                                fx = px
                                fh = py + seg[s, 6]
                                fz = pz
                            if e1 < best_e or (e1 == best_e and ne1 < en_):
                                best_e = e1
                                ex_ = px
                                eh_ = py + seg[s, 6]
                                ez_ = pz
                                en_ = ne1
                                erc_ = rcr
                                eyc_ = py
                            if mode == 1:
                                done = True
                                break
                        if done:
                            break
    found = best_n <= N
    res[0] = 1.0 if found else 0.0
    res[1] = best_n if found else -1.0
    res[2] = fx
    res[3] = fh
    res[4] = fz
    res[5] = en_
    res[6] = ex_
    res[7] = eh_
    res[8] = ez_
    res[9] = best_e
    res[10] = na
    res[11] = ns
    res[12] = visited
    res[13] = erc_
    res[14] = eyc_
    res[15] = 1.0 if crossed else 0.0
    return ncol


# ------------------------------------------------------------------------------------------- the full query

@_jit
def _perceive(tiles, Hh, P1, P2, Pd, Poff, Pdim, Rr, Rl, Rn, Rf, bidx, bricks, seg, rstart, ritems, ntz,
              ox, oz, sx0, sy0, sz0, boxes, void_y, uzones,
              S1, S2, lam_a, a_air, lam_g, a_g, r,
              px, py, pz, vx, vy, vz, dirs, rightv, fwdv,
              yy, xf, zf, D, stack, stamp, stamp_id, collect, work, out, dbg, pdd):
    """One state -> out (N_FEATURES), dbg (N_DBG), pdd (8, N_PDD). dirs (8, 3): world x, z of each key direction
    (unit) and the push magnitude along it (1 or sqrt 2); rightv, fwdv: camera right / forward (world x, z)."""
    N = S1.shape[0] - 1
    for i in range(out.shape[0]):
        out[i] = 0.0
    for i in range(dbg.shape[0]):
        dbg[i] = NAN
    for i in range(pdd.shape[0]):
        for j in range(pdd.shape[1]):
            pdd[i, j] = NAN
    nofilt = np.empty(4)
    nofilt[0] = 1.0
    nofilt[1] = 0.0
    nofilt[2] = 0.0
    nofilt[3] = 0.0
    noex = np.zeros(3)
    res = np.empty(16)
    mres = np.empty(9)
    speed = math.sqrt(vx * vx + vz * vz)
    dbg[DBG_SPEED] = speed
    # ---- unknown zones (moving bodies)
    unk = 0.0
    for u in range(uzones.shape[0]):
        if (uzones[u, 0] <= px <= uzones[u, 3] and uzones[u, 1] <= py <= uzones[u, 4]
                and uzones[u, 2] <= pz <= uzones[u, 5]):
            unk = 1.0
    out[2] = unk
    # ---- support
    rest, snx, sny, snz, sfl = _r_ground(tiles, Rr, Rl, Rn, Rf, ox, oz, px, pz, py, GROUND_TOL)
    grounded = rest == rest
    if grounded:
        # a ball falling onto (or bouncing off) its support much faster than the support plane lets it is not rolling
        vy_plane = -(snx * vx + snz * vz) / max(sny, 0.2)
        if vy < vy_plane - FALL_TOL:
            grounded = False
            rest = NAN
    dbg[DBG_GROUNDED] = 1.0 if grounded else 0.0
    dbg[DBG_SUP_REST] = rest
    dbg[DBG_SUP_FLAG] = sfl
    out[0] = 1.0 if grounded else 0.0
    if grounded:
        out[13] = snx * rightv[0] + snz * rightv[1]
        out[14] = sny
        out[15] = snx * fwdv[0] + snz * fwdv[1]
    # ---- side contact: SDF probes 1.2 from the centre in 8 directions, rails
    best = INF
    bnx = 0.0
    bny = 0.0
    bnz = 0.0
    for q in range(8):
        ang = q * (math.pi / 4.0)
        qx = px + 1.2 * math.cos(ang)
        qz = pz + 1.2 * math.sin(ang)
        d, gx, gy, gz = _udist(bidx, bricks, sx0, sy0, sz0, qx, py, qz)
        gap = d - (r - 1.2)
        if gap < 0.15 and abs(gy) < UP_FACE and gap < best:
            best = gap
            bnx = gx
            bny = gy
            bnz = gz
    dr, rnx, rny, rnz = _rail_dist(seg, rstart, ritems, ntz, ox, oz, px, py, pz)
    if dr - r < 0.15 and rny < UP_FACE and dr - r < best:
        best = dr - r
        bnx = rnx
        bny = rny
        bnz = rnz
    if best < INF:
        hn = math.sqrt(bnx * bnx + bny * bny + bnz * bnz)
        if hn > 1e-9:
            out[16] = (bnx * rightv[0] + bnz * rightv[1]) / hn
            out[17] = bny / hn
            out[18] = (bnx * fwdv[0] + bnz * fwdv[1]) / hn
    # ---- airborne landing reachability
    out[4] = N / 132.0
    if not grounded:
        stamp_id += 1
        _air_scan(tiles, Hh, P1, P2, Pd, Poff, Pdim, seg, rstart, ritems, ntz, ox, oz, boxes, void_y,
                  S1, S2, px, py, pz, vx, vy, vz, a_air, r, nofilt, noex, 0, yy, xf, zf, D, stack, stamp, stamp_id,
                  collect, res)
        dbg[DBG_AIR_ANY] = res[0]
        dbg[DBG_N_APEX] = res[10]
        dbg[DBG_N_STOP] = res[11]
        dbg[DBG_VISITED] = res[12]
        if res[0] > 0.0:
            out[3] = 1.0
            nf = int(res[1])
            out[4] = nf / 132.0
            out[5] = yy[nf] - py
            dbg[DBG_N_FIRST] = nf
            dbg[DBG_FIRST_X] = res[2]
            dbg[DBG_FIRST_H] = res[3]
            dbg[DBG_FIRST_Z] = res[4]
            nn = int(res[5])
            cx = res[6]
            ch = res[7]
            cz = res[8]
            dbg[DBG_N_NEAR] = nn
            dbg[DBG_NEAR_X] = cx
            dbg[DBG_NEAR_H] = ch
            dbg[DBG_NEAR_Z] = cz
            dbg[DBG_NEAR_EFFORT] = res[9]
            # landing centre: the key-free point at nn moved toward the surface point just far enough for the
            # contact (contact radius at that height), within the reachable square
            rho = math.sqrt(max(res[13] ** 2 - (yy[nn] - res[14]) ** 2, 0.0))
            ddx = cx - xf[nn]
            ddz = cz - zf[nn]
            dd = math.sqrt(ddx * ddx + ddz * ddz)
            need = max(0.0, dd - rho)
            ofx = ddx / dd * need if dd > 1e-9 else 0.0
            ofz = ddz / dd * need if dd > 1e-9 else 0.0
            tx = xf[nn] + min(max(ofx, -D[nn]), D[nn])
            tz = zf[nn] + min(max(ofz, -D[nn]), D[nn])
            ux_ = (tx - xf[nn]) / D[nn] if D[nn] > 1e-9 else 0.0
            uz_ = (tz - zf[nn]) / D[nn] if D[nn] > 1e-9 else 0.0
            dxl = tx - px
            dzl = tz - pz
            out[6] = dxl * rightv[0] + dzl * rightv[1]
            out[7] = dxl * fwdv[0] + dzl * fwdv[1]
            out[8] = yy[nn] - py
            out[9] = nn / 132.0
            out[10] = max(abs(ux_), abs(uz_))
            dbg[DBG_LAND_CX] = tx
            dbg[DBG_LAND_CY] = yy[nn]
            dbg[DBG_LAND_CZ] = tz
            # landing velocity (constant fractional push), then: can it stop? (brake vs support ahead)
            g66 = PSI_HZ * S1[nn]
            lamn = 1.0 - (1.0 - lam_a) * g66
            lvx = lamn * vx + a_air * ux_ * g66
            lvz = lamn * vz + a_air * uz_ * g66
            dbg[DBG_LAND_VX] = lvx
            dbg[DBG_LAND_VZ] = lvz
            lsp = math.sqrt(lvx * lvx + lvz * lvz)
            sx_, _, _ = _brake_axis(lvx, 0.0, a_g, lam_g, 1e9)
            sz_, _, _ = _brake_axis(lvz, 0.0, a_g, lam_g, 1e9)
            bl = math.sqrt(sx_ * sx_ + sz_ * sz_)
            dbg[DBG_LAND_BRAKE] = bl
            can_stop = 1.0
            if lsp > 0.05:
                ysup = _r_below(tiles, Rr, Rl, Rn, Rf, ox, oz, tx, tz, yy[nn] + 1.0)
                if ysup != ysup:
                    ysup = yy[nn]
                _march(tiles, Hh, Rr, Rl, Rn, Rf, bidx, bricks, seg, rstart, ritems, ntz, ox, oz, sx0, sy0, sz0, r,
                       tx, ysup, tz, sx_ / max(bl, 1e-9), sz_ / max(bl, 1e-9), min(bl + 2.0, 40.0), False, NAN, mres)
                dbg[DBG_LAND_EDGE] = mres[0]
                if mres[0] < INF and mres[0] < bl and not (mres[3] < mres[0]):
                    can_stop = 0.0
            out[11] = can_stop
            # flight path clearance: 10 points of the centre path to the landing
            cmin = INF
            for j in range(1, 11):
                k = int(round(j * (nn - 2) / 10.0))
                if k < 1:
                    continue
                qx = xf[k] + ux_ * D[k]
                qz = zf[k] + uz_ * D[k]
                d, _, _, _ = _udist(bidx, bricks, sx0, sy0, sz0, qx, yy[k], qz)
                d2, _, _, _ = _rail_dist(seg, rstart, ritems, ntz, ox, oz, qx, yy[k], qz)
                cmin = min(cmin, d, d2)
            dbg[DBG_CLEAR_MIN] = cmin
            out[12] = 1.0 if cmin >= r - 0.15 else 0.0
        elif res[15] > 0.0:
            out[1] = 1.0
            dbg[DBG_DOOM_REASON] = 1.0  # airborne, no landing before the death height
        else:
            out[2] = 1.0  # no landing within the horizon, but the death height is not reached in it either: unknown
            dbg[DBG_DOOM_REASON] = 4.0
    # ---- ground: 8 directions
    for q in range(8):
        base = N_HEAD + q * 5
        out[base + 0] = L_MAX
        out[base + 1] = L_MAX
        out[base + 2] = -CLIP_V
        out[base + 3] = L_MAX
        out[base + 4] = -CLIP_V
    if grounded:
        gtx = (GPSI / ROLL_INERTIA) * sny * snx
        gtz = (GPSI / ROLL_INERTIA) * sny * snz
        # brake vector (full reverse push on both world axes)
        bxd, _, kbx = _brake_axis(vx, gtx, a_g, lam_g, 1e9)
        bzd, _, kbz = _brake_axis(vz, gtz, a_g, lam_g, 1e9)
        for q in range(8):
            ux = dirs[q, 0]
            uz = dirs[q, 1]
            pmag = dirs[q, 2]
            base = N_HEAD + q * 5
            k_sup = -(snx * ux + snz * uz) / max(sny, 0.2) if sfl == 1 else 0.0
            _march(tiles, Hh, Rr, Rl, Rn, Rf, bidx, bricks, seg, rstart, ritems, ntz, ox, oz, sx0, sy0, sz0, r,
                   px, rest, pz, ux, uz, L_MAX, True, k_sup, mres)
            edge_s = mres[0]
            obst_s = mres[3]
            obst_h = mres[4]
            if kbx == INF or kbz == INF:
                brake_along = INF
            else:
                brake_along = max(0.0, bxd * ux + bzd * uz)
            v0d = vx * ux + vz * uz
            a_eff = a_g * pmag + gtx * ux + gtz * uz
            pdd[q, 0] = edge_s
            pdd[q, 1] = obst_s
            pdd[q, 2] = obst_h
            pdd[q, 4] = brake_along
            if edge_s < INF:
                out[base + 0] = min(edge_s, L_MAX)
                out[base + 1] = max(-L_MAX, min(L_MAX, edge_s - brake_along))
                s_near = mres[5]
                s_far = mres[6]
                pdd[q, 7] = s_near
                pdd[q, 8] = s_far
                pdd[q, 9] = mres[7]
                pdd[q, 12] = mres[2]
                vcan = _v_reach(v0d, a_eff, lam_g, max(s_near, 0.0))
                pdd[q, 5] = vcan
                if s_far < INF:
                    x_need = s_far - s_near
                    # the air scan's contact model: the sphere lands on the far side when its centre is still at or
                    # above the far top as it reaches the far edge (it touches the corner from above)
                    dy_t = mres[7] - mres[1]
                    vn = _v_need(S1, S2, N, mres[2], 0.0, a_air * pmag, x_need, dy_t)
                    pdd[q, 10] = vn
                    pdd[q, 11] = _v_need(S1, S2, N, mres[2], 0.0, 0.0, x_need, dy_t)
                    out[base + 2] = CLIP_V if vn == INF else max(-CLIP_V, min(CLIP_V, vn - vcan))
                else:
                    out[base + 2] = CLIP_V
            elif brake_along == INF:
                out[base + 1] = -L_MAX
            if obst_s < INF:
                out[base + 3] = min(obst_s, L_MAX)
                mv = _mount_speed(obst_h, r) if obst_h == obst_h else INF
                pdd[q, 3] = mv
                vco = _v_reach(v0d, a_eff, lam_g, max(obst_s, 0.0))
                pdd[q, 6] = vco
                out[base + 4] = CLIP_V if mv == INF else max(-CLIP_V, min(CLIP_V, mv - vco))
        # ---- grounded doomed: brake path, else a takeoff with a reachable landing
        dbg[DBG_DOOM_REASON] = 0.0
        if sfl != 2 and speed > 0.5:
            safe = False
            tko = False  # brake path left the support
            if kbx < INF and kbz < INF:
                kmax = min(max(kbx, kbz), 1320.0)
                yb = rest
                ok = True
                lx = px
                lz = pz
                lvx_ = vx
                lvz_ = vz
                for j in range(1, 13):
                    t = kmax * j / 12.0
                    dxj, vxj, _ = _brake_axis(vx, gtx, a_g, lam_g, t)
                    dzj, vzj, _ = _brake_axis(vz, gtz, a_g, lam_g, t)
                    qx = px + dxj
                    qz = pz + dzj
                    stl = math.sqrt((qx - lx) ** 2 + (qz - lz) ** 2)
                    mx = (qx - lx) / max(stl, 1e-9)
                    mz = (qz - lz) / max(stl, 1e-9)
                    hold, _, _, _, _, block, _ = _r_hold(tiles, Rr, Rl, Rn, Rf, ox, oz, qx, qz, yb, DROP_TOL, mx, mz,
                                                      stl)
                    if block == block:
                        break  # a barrier stops it
                    if hold != hold:
                        ok = False
                        lx = qx
                        lz = qz
                        lvx_ = vxj
                        lvz_ = vzj
                        break
                    yb = hold
                    lx = qx
                    lz = qz
                if ok:
                    safe = True
                else:
                    tko = True
                    dbg[DBG_BRAKE_OK] = 0.0
                    sp_ = math.sqrt(lvx_ * lvx_ + lvz_ * lvz_)
                    excl = np.empty(3)
                    excl[0] = lx
                    excl[1] = lz
                    excl[2] = EXCL_R
                    stamp_id += 1
                    empty = collect[:0]
                    _air_scan(tiles, Hh, P1, P2, Pd, Poff, Pdim, seg, rstart, ritems, ntz, ox, oz, boxes, void_y,
                              S1, S2, lx, yb, lz, lvx_, 0.0, lvz_, a_air, r, nofilt, excl, 1, yy, xf, zf, D, stack,
                              stamp, stamp_id, empty, res)
                    dbg[DBG_BRAKE_TAKEOFF] = res[0]
                    if res[0] > 0.0 or res[15] == 0.0 or sp_ < 0.05:
                        safe = True
            if not safe:
                # full push along the velocity to the edge, then the flight from there
                ux = vx / speed
                uz = vz / speed
                k_sup = -(snx * ux + snz * uz) / max(sny, 0.2) if sfl == 1 else 0.0
                _march(tiles, Hh, Rr, Rl, Rn, Rf, bidx, bricks, seg, rstart, ritems, ntz, ox, oz, sx0, sy0, sz0, r,
                       px, rest, pz, ux, uz, 40.0, False, k_sup, mres)
                dbg[DBG_JUMP_EDGE] = mres[0]
                if mres[0] == INF or mres[3] < mres[0]:
                    safe = True  # no edge ahead within 40, or a barrier first
                else:
                    pm = abs(ux) + abs(uz)
                    a_eff = a_g * pm + gtx * ux + gtz * uz
                    s_to = max(mres[0] - DS, 0.0)
                    vc = _v_reach(speed, a_eff, lam_g, s_to)
                    dbg[DBG_JUMP_V] = vc
                    excl = np.empty(3)
                    excl[0] = px + ux * s_to
                    excl[1] = pz + uz * s_to
                    excl[2] = EXCL_R
                    stamp_id += 1
                    empty = collect[:0]
                    _air_scan(tiles, Hh, P1, P2, Pd, Poff, Pdim, seg, rstart, ritems, ntz, ox, oz, boxes, void_y,
                              S1, S2, excl[0], mres[1], excl[1], vc * ux, vc * mres[2], vc * uz, a_air, r, nofilt,
                              excl, 1, yy, xf, zf, D, stack, stamp, stamp_id, empty, res)
                    dbg[DBG_JUMP_TAKEOFF] = res[0]
                    if res[0] > 0.0 or res[15] == 0.0:
                        safe = True
            if not safe:
                out[1] = 1.0
                dbg[DBG_DOOM_REASON] = 2.0 if tko else 3.0
    return stamp_id


# dbg indices as module constants (numba freezes globals)
DBG_GROUNDED = DBG["grounded"]
DBG_SUP_REST = DBG["sup_rest"]
DBG_SUP_FLAG = DBG["sup_flag"]
DBG_AIR_ANY = DBG["airborne_any"]
DBG_N_APEX = DBG["n_apex"]
DBG_N_STOP = DBG["n_stop"]
DBG_N_FIRST = DBG["n_first"]
DBG_FIRST_X = DBG["first_x"]
DBG_FIRST_H = DBG["first_h"]
DBG_FIRST_Z = DBG["first_z"]
DBG_N_NEAR = DBG["n_near"]
DBG_NEAR_X = DBG["near_x"]
DBG_NEAR_H = DBG["near_h"]
DBG_NEAR_Z = DBG["near_z"]
DBG_NEAR_EFFORT = DBG["near_effort"]
DBG_LAND_CX = DBG["land_cx"]
DBG_LAND_CY = DBG["land_cy"]
DBG_LAND_CZ = DBG["land_cz"]
DBG_LAND_VX = DBG["land_vx"]
DBG_LAND_VZ = DBG["land_vz"]
DBG_LAND_BRAKE = DBG["land_brake"]
DBG_LAND_EDGE = DBG["land_edge"]
DBG_CLEAR_MIN = DBG["clear_min"]
DBG_DOOM_REASON = DBG["doom_reason"]
DBG_BRAKE_OK = DBG["brake_ok"]
DBG_BRAKE_TAKEOFF = DBG["brake_takeoff"]
DBG_JUMP_TAKEOFF = DBG["jump_takeoff"]
DBG_JUMP_EDGE = DBG["jump_edge"]
DBG_JUMP_V = DBG["jump_v"]
DBG_VISITED = DBG["visited"]
DBG_SPEED = DBG["speed"]


# ============================================================================================ build kernels

@_jit
def _closest_tri(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """Closest point on triangle ABC to P (Ericson)."""
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    acx = cx - ax
    acy = cy - ay
    acz = cz - az
    apx = px - ax
    apy = py - ay
    apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return ax, ay, az
    bpx = px - bx
    bpy = py - by
    bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bx, by, bz
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3) if d1 != d3 else 0.0
        return ax + v * abx, ay + v * aby, az + v * abz
    cpx = px - cx
    cpy = py - cy
    cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cx, cy, cz
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6) if d2 != d6 else 0.0
        return ax + w * acx, ay + w * acy, az + w * acz
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        den = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / den if den != 0.0 else 0.0
        return bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz)
    den = va + vb + vc
    if den == 0.0:
        return ax, ay, az
    v = vb / den
    w = vc / den
    return ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w


@_jit
def _sdf_bricks(T, TN, bcoord, bstart, bitems, sx0, sy0, sz0, out):
    """Unsigned distance at the 9^3 corners of each brick to its candidate triangles, clamped to SDF_BAND, int16 fixed
    point. (Signs from the nearest face's winding were inconsistent on L1's open meshes and created false zero
    crossings under trilinear interpolation; the queries only need the distance from outside.)"""
    for b in range(bcoord.shape[0]):
        for i in range(BRICK + 1):
            x = sx0 + (bcoord[b, 0] * BRICK + i) * VRES
            for j in range(BRICK + 1):
                y = sy0 + (bcoord[b, 1] * BRICK + j) * VRES
                for k in range(BRICK + 1):
                    z = sz0 + (bcoord[b, 2] * BRICK + k) * VRES
                    best = INF
                    for q in range(int(bstart[b]), int(bstart[b + 1])):
                        t = int(bitems[q])
                        qx, qy, qz = _closest_tri(x, y, z, T[t, 0], T[t, 1], T[t, 2], T[t, 3], T[t, 4], T[t, 5],
                                                  T[t, 6], T[t, 7], T[t, 8])
                        d = math.sqrt((x - qx) ** 2 + (y - qy) ** 2 + (z - qz) ** 2)
                        if d < best:
                            best = d
                    v = min(best, SDF_BAND)
                    out[b, i, j, k] = int(round(v * SDF_SCALE))


@_jit
def _rest_field(tiles, tile_ij, Hh, Hn, seg, rstart, ritems, ntz, ox, oz, r, Rr, Rl, Rn, Rf, buf):
    """R from H (point and face contacts) and the rail tubes; see the module doc."""
    m = int(math.ceil(r / RES))
    for t in range(tile_ij.shape[0]):
        for li in range(TL):
            for lj in range(TL):
                i = tile_ij[t, 0] * TL + li
                j = tile_ij[t, 1] * TL + lj
                cx = ox + i * RES
                cz = oz + j * RES
                nc = 0
                for di in range(-m, m + 1):
                    for dj in range(-m, m + 1):
                        d2 = (di * di + dj * dj) * RES * RES
                        tt = _tile_of(tiles, i + di, j + dj)
                        if tt < 0:
                            continue
                        sx = cx + di * RES
                        sz = cz + dj * RES
                        ai = (i + di) % TL
                        aj = (j + dj) % TL
                        for k in range(KH):
                            h = float(Hh[tt, ai, aj, k])
                            if h != h:
                                break
                            nx = float(Hn[tt, ai, aj, k, 0]) / 127.0
                            nz = float(Hn[tt, ai, aj, k, 1]) / 127.0
                            ny = math.sqrt(max(0.0, 1.0 - nx * nx - nz * nz))
                            rest = -INF
                            fl = 0
                            cnx = 0.0
                            cnz = 0.0
                            if d2 <= r * r:
                                rest = h + math.sqrt(r * r - d2)
                                cnx = (cx - sx) / r
                                cnz = (cz - sz) / r
                            # face contact: the sphere touches the source's plane inside its cell
                            fx = cx - r * nx
                            fz = cz - r * nz
                            if abs(fx - sx) <= 0.5 * RES + 1e-6 and abs(fz - sz) <= 0.5 * RES + 1e-6 and ny >= UP_FACE:
                                rf = h - (nx * (fx - sx) + nz * (fz - sz)) / ny + r * ny
                                if rf >= rest - 0.02:
                                    rest = max(rest, rf)
                                    fl = 1
                                    cnx = nx
                                    cnz = nz
                            if rest == -INF:
                                continue
                            if nc < buf.shape[0]:
                                buf[nc, 0] = rest
                                buf[nc, 1] = 2.0 * h - rest
                                buf[nc, 2] = h
                                buf[nc, 3] = cnx
                                buf[nc, 4] = cnz
                                buf[nc, 5] = fl
                                nc += 1
                # rails
                sxr = cx - ox + 0.5 * RES
                szr = cz - oz + 0.5 * RES
                ti = int(math.floor(sxr / (RES * TL)))
                tj = int(math.floor(szr / (RES * TL)))
                if ti >= 0 and tj >= 0 and tj < ntz and ti * ntz + tj + 1 < len(rstart):
                    c = ti * ntz + tj
                    for q in range(int(rstart[c]), int(rstart[c + 1])):
                        s = int(ritems[q])
                        ax = seg[s, 0]
                        az = seg[s, 2]
                        dx = seg[s, 3] - ax
                        dz = seg[s, 5] - az
                        L2 = dx * dx + dz * dz
                        tq = 0.0
                        if L2 > 1e-12:
                            tq = ((cx - ax) * dx + (cz - az) * dz) / L2
                            tq = min(max(tq, 0.0), 1.0)
                        qx = ax + tq * dx
                        qz = az + tq * dz
                        yc = seg[s, 1] + tq * (seg[s, 4] - seg[s, 1])
                        R2 = (r + seg[s, 6]) ** 2
                        dh2 = (cx - qx) ** 2 + (cz - qz) ** 2
                        if dh2 > R2:
                            continue
                        rest = yc + math.sqrt(R2 - dh2)
                        if nc < buf.shape[0]:
                            rr_ = r + seg[s, 6]
                            buf[nc, 0] = rest
                            buf[nc, 1] = 2.0 * yc - rest
                            buf[nc, 2] = yc + seg[s, 6]
                            buf[nc, 3] = (cx - qx) / rr_
                            buf[nc, 4] = (cz - qz) / rr_
                            buf[nc, 5] = 2.0
                            nc += 1
                if nc == 0:
                    continue
                # sort candidates by source height, descending (insertion sort on a small buffer)
                for a in range(1, nc):
                    row0 = buf[a, 0]
                    row1 = buf[a, 1]
                    row2 = buf[a, 2]
                    row3 = buf[a, 3]
                    row4 = buf[a, 4]
                    row5 = buf[a, 5]
                    bb = a - 1
                    while bb >= 0 and buf[bb, 2] < row2:
                        for c6 in range(6):
                            buf[bb + 1, c6] = buf[bb, c6]
                        bb -= 1
                    buf[bb + 1, 0] = row0
                    buf[bb + 1, 1] = row1
                    buf[bb + 1, 2] = row2
                    buf[bb + 1, 3] = row3
                    buf[bb + 1, 4] = row4
                    buf[bb + 1, 5] = row5
                # clusters: a new one where the source height drops by more than R_SRC_GAP
                k = 0
                a = 0
                while a < nc and k < KR:
                    best = a
                    low = buf[a, 1]
                    bb = a + 1
                    while bb < nc and buf[bb - 1, 2] - buf[bb, 2] <= R_SRC_GAP:
                        # best: the highest rest, faces preferred within 0.02
                        if buf[bb, 0] > buf[best, 0] + 0.02 or (buf[bb, 0] > buf[best, 0] - 0.02 and buf[bb, 5] == 1.0
                                                                 and buf[best, 5] != 1.0):
                            best = bb
                        low = min(low, buf[bb, 1])
                        bb += 1
                    Rr[t, li, lj, k] = buf[best, 0]
                    Rl[t, li, lj, k] = low
                    Rn[t, li, lj, k, 0] = int(round(max(-1.0, min(1.0, buf[best, 3])) * 127.0))
                    Rn[t, li, lj, k, 1] = int(round(max(-1.0, min(1.0, buf[best, 4])) * 127.0))
                    Rf[t, li, lj, k] = int(buf[best, 5])
                    k += 1
                    a = bb
                # clusters are produced in descending source height; order them by rest (descending) for lookups
                for a in range(1, k):
                    bb = a
                    while bb > 0 and Rr[t, li, lj, bb - 1] < Rr[t, li, lj, bb]:
                        for arr_k in range(1):
                            tmp = Rr[t, li, lj, bb - 1]
                            Rr[t, li, lj, bb - 1] = Rr[t, li, lj, bb]
                            Rr[t, li, lj, bb] = tmp
                            tmp = Rl[t, li, lj, bb - 1]
                            Rl[t, li, lj, bb - 1] = Rl[t, li, lj, bb]
                            Rl[t, li, lj, bb] = tmp
                            for c2 in range(2):
                                ti8 = Rn[t, li, lj, bb - 1, c2]
                                Rn[t, li, lj, bb - 1, c2] = Rn[t, li, lj, bb, c2]
                                Rn[t, li, lj, bb, c2] = ti8
                            tf = Rf[t, li, lj, bb - 1]
                            Rf[t, li, lj, bb - 1] = Rf[t, li, lj, bb]
                            Rf[t, li, lj, bb] = tf
                        bb -= 1


@_jit
def _wall_bits(tile_ij, Rr, Rf, bidx, bricks, ox, oz, sx0, sy0, sz0, r):
    """R flag bit 4: a ball resting on the cluster at this cell centre comes within r - PEN_TOL + 0.36 (half a cell
    diagonal of slack) of the static mesh (SDF): only there does the march consult the SDF for walls."""
    for t in range(tile_ij.shape[0]):
        for li in range(TL):
            for lj in range(TL):
                cx = ox + (tile_ij[t, 0] * TL + li) * RES
                cz = oz + (tile_ij[t, 1] * TL + lj) * RES
                for k in range(KR):
                    rest = float(Rr[t, li, lj, k])
                    if rest != rest:
                        break
                    d, _, _, _ = _udist(bidx, bricks, sx0, sy0, sz0, cx, rest, cz)
                    if d < r - PEN_TOL + 0.36:
                        Rf[t, li, lj, k] = Rf[t, li, lj, k] | 4


# ============================================================================================ build (python)

def _weld(P, tol=1e-4):
    key = np.round(P / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return P[idx], inv.reshape(-1)


def _cluster_points(P, tol):
    """Single-linkage groups of points closer than tol (grid hashing + union-find). Returns group ids (0..k-1)."""
    key = np.floor(P / tol).astype(np.int64)
    cells = {}
    for i, k in enumerate(map(tuple, key)):
        cells.setdefault(k, []).append(i)
    par = list(range(len(P)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for k, idx in cells.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    other = cells.get((k[0] + dx, k[1] + dy, k[2] + dz))
                    if not other:
                        continue
                    for i in idx:
                        for j in other:
                            if j > i and np.sum((P[i] - P[j]) ** 2) <= tol * tol:
                                ri, rj = find(i), find(j)
                                if ri != rj:
                                    par[ri] = rj
    roots = np.array([find(i) for i in range(len(P))])
    return np.unique(roots, return_inverse=True)[1]


def _circle_fit(Q):
    """Least-squares circle through 3-D points (plane by PCA, Kasa fit in it): (centre (3,), radius) or None."""
    c0 = Q.mean(0)
    X = Q - c0
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    if len(sv) < 2 or sv[1] < 1e-6:
        return None
    u, w = vt[0], vt[1]
    x, y = X @ u, X @ w
    M = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(M, x * x + y * y, rcond=None)
    a, b, c = sol
    rad2 = c + a * a + b * b
    if rad2 <= 0:
        return None
    return c0 + a * u + b * w, float(np.sqrt(rad2))


def extract_rails(A, B, C, owner, max_resid: float = 0.08):
    """Tube axis segments from rail triangles. Returns (segments (S, 7): x0 y0 z0 x1 y1 z1 radius, fallback mask
    over the triangles (True = not a tube: keep as ordinary geometry), stats dict)."""
    A, B, C = (np.asarray(x, np.float64).reshape(-1, 3) for x in (A, B, C))
    nT = len(A)
    stats = {"tris": nT}
    if nT == 0:
        return np.zeros((0, 7)), np.zeros(0, bool), stats
    V, inv = _weld(np.concatenate([A, B, C]))
    F = inv.reshape(3, nT).T
    n = np.cross(B - A, C - A)
    area2 = np.linalg.norm(n, axis=1)
    fn = n / np.where(area2 > 0, area2, 1)[:, None]
    # angle-weighted vertex normals
    vn = np.zeros_like(V)
    P = (A, B, C)
    for k in range(3):
        e1 = P[(k + 1) % 3] - P[k]
        e2 = P[(k + 2) % 3] - P[k]
        cosang = (e1 * e2).sum(1) / np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-12)
        ang = np.arccos(np.clip(cosang, -1, 1))
        np.add.at(vn, F[:, k], fn * ang[:, None])
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]
    # components of the welded mesh
    par = np.arange(len(V))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for a, b, c in F:
        for x, y in ((a, b), (b, c)):
            rx, ry = find(x), find(y)
            if rx != ry:
                par[rx] = ry
    comp = np.array([find(i) for i in range(len(V))])
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    segs = []
    tube_vertex = np.zeros(len(V), bool)
    resid_all = []
    radii = []
    for cid in np.unique(comp):
        vi = np.flatnonzero(comp == cid)
        sel = np.isin(E[:, 0], vi)
        Ec = E[sel]
        if len(Ec) == 0:
            continue
        na, nb = vn[Ec[:, 0]], vn[Ec[:, 1]]
        dn = na - nb
        l2 = (dn * dn).sum(1)
        ring = l2 > 0.09  # normals at least ~17 degrees apart: a ring edge
        if ring.sum() < 3:
            continue
        Re = ((V[Ec[ring, 0]] - V[Ec[ring, 1]]) * dn[ring]).sum(1) / l2[ring]
        cmap = dict(zip(vi.tolist(), range(len(vi))))
        # rings: vertices whose centre estimates (vertex - R normal) lie within 0.15 of each other (union-find on a
        # grid); each ring's axis point is a least-squares circle fit of its vertices (exact for open rings, whose
        # boundary vertices have skewed normals). R is the candidate (ring-edge estimates' quartiles) that puts the
        # most vertices into rings whose fitted radius agrees with it.
        best = None
        for Rc in np.unique(np.round(np.percentile(Re, [10, 25, 40, 50, 60, 75, 90]), 4)):
            if not (0.02 < Rc < 2.0):
                continue
            kin_c = _cluster_points(V[vi] - Rc * vn[vi], 0.15)
            nodes_c = np.full((kin_c.max() + 1, 3), np.nan)
            rads = []
            covered = 0
            for g in range(kin_c.max() + 1):
                m = kin_c == g
                if m.sum() >= 3:
                    fit = _circle_fit(V[vi][m])
                    if fit is not None and abs(fit[1] - Rc) < 0.05:
                        nodes_c[g] = fit[0]
                        rads.append(fit[1])
                        covered += int(m.sum())
            if rads and (best is None or covered > best[0]):
                best = (covered, float(np.median(rads)), kin_c, nodes_c)
        if best is None:
            continue
        _, R, kin, nodes = best
        ea = np.array([cmap[x] for x in Ec[:, 0]])
        eb = np.array([cmap[x] for x in Ec[:, 1]])
        na_, nb_ = kin[ea], kin[eb]
        keep = (na_ != nb_) & np.isfinite(nodes[na_, 0]) & np.isfinite(nodes[nb_, 0])
        pairs = np.unique(np.sort(np.c_[na_[keep], nb_[keep]], axis=1), axis=0)
        if len(pairs) == 0:
            continue
        p0, p1 = nodes[pairs[:, 0]], nodes[pairs[:, 1]]
        L = np.linalg.norm(p1 - p0, axis=1)
        good = L > 0.05
        p0, p1 = p0[good], p1[good]
        if len(p0) == 0:
            continue
        # residual: every vertex's distance to the nearest axis segment minus R
        q = V[vi]
        d = np.full(len(q), np.inf)
        for a0, a1 in zip(p0, p1):
            dv = a1 - a0
            t = np.clip(((q - a0) @ dv) / max(dv @ dv, 1e-12), 0, 1)
            d = np.minimum(d, np.linalg.norm(q - (a0 + t[:, None] * dv), axis=1))
        res = np.abs(d - R)
        if np.percentile(res, 95) > max_resid:
            continue
        resid_all.append(res)
        radii.append(R)
        tube_vertex[vi] = True
        segs.append(np.c_[p0, p1, np.full(len(p0), R)])
    fallback = ~tube_vertex[F].all(axis=1)
    seg = np.concatenate(segs) if segs else np.zeros((0, 7))
    ra = np.concatenate(resid_all) if resid_all else np.zeros(0)
    stats.update(segments=len(seg), fallback_tris=int(fallback.sum()),
                 fallback_frac=float(fallback.mean()),
                 resid_p50=float(np.median(ra)) if len(ra) else np.nan,
                 resid_p95=float(np.percentile(ra, 95)) if len(ra) else np.nan,
                 resid_max=float(ra.max()) if len(ra) else np.nan,
                 radius_min=float(min(radii)) if radii else np.nan, radius_max=float(max(radii)) if radii else np.nan)
    return seg, fallback, stats


def _tri_samples(A, B, C, edge_step=0.2, fill_step=0.25):
    e, ei = reach.SupportMap._edge_samples(A, B, C, edge_step)
    f, fi = reach.SupportMap._fill_samples(A, B, C, fill_step)
    return np.concatenate([e, f]), np.concatenate([ei, fi])


class FastReach:
    """Fields of one level + the query. Build with FastReach.build(...) / for_tracker(...), or load(path)."""

    ARRAYS = ("tiles", "tile_ij", "Hh", "Hn", "P1", "P2", "Pd", "Poff", "Pdim", "Rr", "Rl", "Rn", "Rf", "bidx",
              "bricks", "seg", "rstart", "ritems", "boxes", "uzones", "meta")
    COMPONENTS = {"heightfield": ("tiles", "tile_ij", "Hh", "Hn"), "pyramid": ("P1", "P2", "Pd", "Poff", "Pdim"),
                  "rest_field": ("Rr", "Rl", "Rn", "Rf"), "sdf": ("bidx", "bricks"),
                  "rails": ("seg", "rstart", "ritems"), "death_unknown": ("boxes", "uzones", "meta")}

    def __init__(self, arrays: dict, info: dict | None = None):
        for k in self.ARRAYS:
            setattr(self, k, arrays[k])
        m = self.meta
        self.ox, self.oz, self.sx0, self.sy0, self.sz0, self.void_y = (float(x) for x in m[:6])
        self.ntz = int(self.tiles.shape[1])
        self.info = dict(info or {})
        self._tables = {}
        self.yy = np.empty(N_PSI + 1)
        self.xf = np.empty(N_PSI + 1)
        self.zf = np.empty(N_PSI + 1)
        self.D = np.empty(N_PSI + 1)
        self.stack = np.zeros((512, 3), np.int64)
        self.stamp = np.zeros(max(len(self.seg), 1), np.int64)
        self.stamp_id = 0
        self.work = np.zeros(16)
        self.out = np.zeros(N_FEATURES)
        self.dbg = np.zeros(N_DBG)
        self.pdd = np.zeros((8, N_PDD))
        self._nocollect = np.zeros((0, 5))

    # ------------------------------------------------------------------------------------------------ build
    @classmethod
    def build(cls, tris, rails, depth=(), unknown_boxes=(), death_min_extent: float = 20.0, r: float = R_BALL,
              verbose: bool = False) -> "FastReach":
        """tris: features.LevelGeometry.tris (all static triangles, rails included, owner ids); rails: its .rails
        (rail triangles with owners); depth: tracker.level.depth; unknown_boxes: [(lo, hi)] of moving bodies."""
        t_all = time.perf_counter()
        times = {}
        A, B, C = (np.asarray(x, np.float64).reshape(-1, 3) for x in (tris.A, tris.B, tris.C))
        owner = np.asarray(tris.owner, np.int64)
        rail_owners = np.unique(np.asarray(rails.owner, np.int64)) if len(rails) else np.zeros(0, np.int64)
        is_rail = np.isin(owner, rail_owners) if len(rail_owners) else np.zeros(len(A), bool)
        t0 = time.perf_counter()
        seg, fb, rstats = extract_rails(A[is_rail], B[is_rail], C[is_rail], owner[is_rail])
        times["rails_extract"] = time.perf_counter() - t0
        keep = ~is_rail
        keep[np.flatnonzero(is_rail)[fb]] = True  # rail parts that are not tubes stay ordinary geometry
        A, B, C = A[keep], B[keep], C[keep]
        n = np.cross(B - A, C - A)
        ln = np.linalg.norm(n, axis=1)
        N = n / np.where(ln == 0, 1, ln)[:, None]
        Nup = np.where(N[:, 1:2] < 0, -N, N)
        # ---- grid
        allx = np.concatenate([A[:, 0], B[:, 0], C[:, 0], seg[:, 0], seg[:, 3]]) if len(A) else seg[:, [0, 3]].ravel()
        allz = np.concatenate([A[:, 2], B[:, 2], C[:, 2], seg[:, 2], seg[:, 5]]) if len(A) else seg[:, [2, 5]].ravel()
        ally = np.concatenate([A[:, 1], B[:, 1], C[:, 1], seg[:, 1], seg[:, 4]]) if len(A) else seg[:, [1, 4]].ravel()
        pad = 8.0
        ox, oz = float(allx.min() - pad), float(allz.min() - pad)
        nx_c = int(math.ceil((allx.max() + pad - ox) / RES)) + 1
        nz_c = int(math.ceil((allz.max() + pad - oz) / RES)) + 1
        ntx, ntz = (nx_c + TL - 1) // TL, (nz_c + TL - 1) // TL
        # ---- H samples
        t0 = time.perf_counter()
        walk = Nup[:, 1] >= UP_FACE
        S, si = _tri_samples(A[walk], B[walk], C[walk])
        SN = Nup[walk][si]
        ci = np.floor((S[:, 0] - ox) / RES + 0.5).astype(np.int64)
        cj = np.floor((S[:, 2] - oz) / RES + 0.5).astype(np.int64)
        occ = np.zeros((ntx, ntz), bool)
        occ[ci // TL, cj // TL] = True
        # rail tiles
        for s in seg:
            lo = np.minimum(s[[0, 2]], s[[3, 5]]) - (r + s[6] + 1.0)
            hi = np.maximum(s[[0, 2]], s[[3, 5]]) + (r + s[6] + 1.0)
            i0, j0 = (np.floor((lo - [ox, oz]) / RES + 0.5).astype(int) // TL)
            i1, j1 = (np.floor((hi - [ox, oz]) / RES + 0.5).astype(int) // TL)
            occ[max(i0, 0):i1 + 1, max(j0, 0):j1 + 1] = True
        dil = occ.copy()
        dil[1:] |= occ[:-1]
        dil[:-1] |= occ[1:]
        d2 = dil.copy()
        d2[:, 1:] |= dil[:, :-1]
        d2[:, :-1] |= dil[:, 1:]
        tiles = np.full((ntx, ntz), -1, np.int32)
        tij = np.argwhere(d2)
        tiles[tij[:, 0], tij[:, 1]] = np.arange(len(tij), dtype=np.int32)
        T = len(tij)
        Hh = np.full((T, TL, TL, KH), np.nan, np.float32)
        Hn = np.zeros((T, TL, TL, KH, 2), np.int8)
        key = (tiles[ci // TL, cj // TL].astype(np.int64) * TL + ci % TL) * TL + cj % TL
        h = S[:, 1]
        # the top sample's face plane evaluated at the cell centre (the highest sample of a sloped face in a 0.5
        # cell sits up to 0.35 x slope above the plane height at the centre)
        ny_ = np.maximum(SN[:, 1], UP_FACE)
        hc_ = np.minimum(h, h - (SN[:, 0] * (ox + ci * RES - S[:, 0]) + SN[:, 2] * (oz + cj * RES - S[:, 2])) / ny_)
        order = np.lexsort((-SN[:, 1], -np.round(h, 2), key))
        key, h, SN, hc_ = key[order], h[order], SN[order], hc_[order]
        sel = np.ones(len(key), bool)
        for k in range(KH):
            kk, hh, nn, hcc = key[sel], h[sel], SN[sel], hc_[sel]
            if not len(kk):
                break
            first = np.r_[True, kk[1:] != kk[:-1]]
            st = np.flatnonzero(first)
            tt, rem = kk[st] // (TL * TL), kk[st] % (TL * TL)
            Hh[tt, rem // TL, rem % TL, k] = hcc[st]
            Hn[tt, rem // TL, rem % TL, k, 0] = np.round(np.clip(nn[st, 0], -1, 1) * 127).astype(np.int8)
            Hn[tt, rem // TL, rem % TL, k, 1] = np.round(np.clip(nn[st, 2], -1, 1) * 127).astype(np.int8)
            grp = np.cumsum(first) - 1
            below = hh < hh[st][grp] - H_LAYER_GAP
            idx = np.flatnonzero(sel)
            sel[idx[~below]] = False
        times["heightfield"] = time.perf_counter() - t0
        # ---- pyramid
        t0 = time.perf_counter()
        with np.errstate(all="ignore"):
            cmin = np.where(np.isnan(Hh), np.inf, Hh).min(-1)
            cmax = np.where(np.isnan(Hh), -np.inf, Hh).max(-1)
        P1 = np.stack([cmin.reshape(T, 4, 2, 4, 2).min((2, 4)), cmax.reshape(T, 4, 2, 4, 2).max((2, 4))], -1)
        P2 = np.stack([P1[..., 0].reshape(T, 2, 2, 2, 2).min((2, 4)), P1[..., 1].reshape(T, 2, 2, 2, 2).max((2, 4))],
                      -1)
        lev = np.empty((ntx, ntz, 2), np.float32)
        lev[..., 0], lev[..., 1] = np.inf, -np.inf
        lev[tij[:, 0], tij[:, 1], 0] = P2[..., 0].min((1, 2))
        lev[tij[:, 0], tij[:, 1], 1] = P2[..., 1].max((1, 2))
        levels = [lev]
        while levels[-1].shape[0] > 1 or levels[-1].shape[1] > 1:
            L = levels[-1]
            a, b = (L.shape[0] + 1) // 2, (L.shape[1] + 1) // 2
            Lp = np.empty((2 * a, 2 * b, 2), np.float32)
            Lp[..., 0], Lp[..., 1] = np.inf, -np.inf
            Lp[:L.shape[0], :L.shape[1]] = L
            levels.append(np.stack([Lp[..., 0].reshape(a, 2, b, 2).min((1, 3)),
                                    Lp[..., 1].reshape(a, 2, b, 2).max((1, 3))], -1))
        Pdim = np.array([L.shape[:2] for L in levels], np.int64)
        Poff = np.r_[0, np.cumsum([L.shape[0] * L.shape[1] for L in levels])].astype(np.int64)
        Pd = np.concatenate([L.reshape(-1, 2) for L in levels]).astype(np.float32)
        times["pyramid"] = time.perf_counter() - t0
        # ---- rails CSR over tiles (in tile coordinates shifted by half a cell: tile of x = floor((x-ox+0.25)/4))
        t0 = time.perf_counter()
        lists = [[] for _ in range(ntx * ntz)]
        for s_i, s in enumerate(seg):
            padr = r + s[6] + 1.0
            lo = np.minimum(s[[0, 2]], s[[3, 5]]) - padr
            hi = np.maximum(s[[0, 2]], s[[3, 5]]) + padr
            i0 = int(math.floor((lo[0] - ox + 0.5 * RES) / (RES * TL)))
            i1 = int(math.floor((hi[0] - ox + 0.5 * RES) / (RES * TL)))
            j0 = int(math.floor((lo[1] - oz + 0.5 * RES) / (RES * TL)))
            j1 = int(math.floor((hi[1] - oz + 0.5 * RES) / (RES * TL)))
            for i in range(max(i0, 0), min(i1, ntx - 1) + 1):
                for j in range(max(j0, 0), min(j1, ntz - 1) + 1):
                    lists[i * ntz + j].append(s_i)
        rstart = np.r_[0, np.cumsum([len(x) for x in lists])].astype(np.int32)
        ritems = np.array([v for x in lists for v in x], np.int32) if rstart[-1] else np.zeros(0, np.int32)
        times["rails_index"] = time.perf_counter() - t0
        # ---- R
        t0 = time.perf_counter()
        Rr = np.full((T, TL, TL, KR), np.nan, np.float32)
        Rl = np.full((T, TL, TL, KR), np.nan, np.float32)
        Rn = np.zeros((T, TL, TL, KR, 2), np.int8)
        Rf = np.zeros((T, TL, TL, KR), np.uint8)
        buf = np.zeros(((2 * int(math.ceil(r / RES)) + 1) ** 2 * KH + 256, 6))
        _rest_field(tiles, tij.astype(np.int64), Hh, Hn, seg, rstart, ritems, ntz, ox, oz, float(r), Rr, Rl, Rn, Rf,
                    buf)
        times["rest_field"] = time.perf_counter() - t0
        # ---- SDF
        t0 = time.perf_counter()
        sy0 = float(ally.min() - pad)
        sx0, sz0 = ox, oz
        bsz = BRICK * VRES
        nbx = int(math.ceil((allx.max() + pad - sx0) / bsz)) + 1
        nby = int(math.ceil((ally.max() + pad - sy0) / bsz)) + 1
        nbz = int(math.ceil((allz.max() + pad - sz0) / bsz)) + 1
        lo = np.minimum(np.minimum(A, B), C) - SDF_BAND
        hi = np.maximum(np.maximum(A, B), C) + SDF_BAND
        org = np.array([sx0, sy0, sz0])
        b0 = np.floor((lo - org) / bsz).astype(np.int64)
        b1 = np.floor((hi - org) / bsz).astype(np.int64)
        cnt = (b1 - b0 + 1).prod(1)
        tid = np.repeat(np.arange(len(A)), cnt)
        off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ex = (b1 - b0 + 1)[tid]
        bx = b0[tid, 0] + off // (ex[:, 1] * ex[:, 2])
        by = b0[tid, 1] + (off // ex[:, 2]) % ex[:, 1]
        bz = b0[tid, 2] + off % ex[:, 2]
        lin = (bx * nby + by) * nbz + bz
        ub, binv = np.unique(lin, return_inverse=True)
        bidx = np.full((nbx, nby, nbz), -1, np.int32)
        bidx.reshape(-1)[ub] = np.arange(len(ub), dtype=np.int32)
        bcoord = np.c_[ub // (nby * nbz), (ub // nbz) % nby, ub % nbz].astype(np.int64)
        o2 = np.argsort(binv, kind="stable")
        bitems = tid[o2].astype(np.int64)
        bstart = np.r_[0, np.cumsum(np.bincount(binv, minlength=len(ub)))].astype(np.int64)
        Tarr = np.c_[A, B, C]
        bricks = np.zeros((len(ub), BRICK + 1, BRICK + 1, BRICK + 1), np.int16)
        _sdf_bricks(Tarr, N, bcoord, bstart, bitems, sx0, sy0, sz0, bricks)
        _wall_bits(tij.astype(np.int64), Rr, Rf, bidx, bricks, ox, oz, sx0, sy0, sz0, float(r))
        times["sdf"] = time.perf_counter() - t0
        # ---- death boxes, unknown zones
        bxs = []
        for item in depth:
            lo_, hi_ = np.asarray(item[-2], np.float64), np.asarray(item[-1], np.float64)
            if min(hi_[0] - lo_[0], hi_[2] - lo_[2]) >= death_min_extent:
                bxs.append(np.r_[lo_, hi_])
        boxes = np.array(bxs, np.float64).reshape(-1, 6)
        uz = np.array([np.r_[np.asarray(lo_) - 4.0, np.asarray(hi_) + 4.0] for lo_, hi_ in unknown_boxes],
                      np.float64).reshape(-1, 6)
        min_y = float(np.nanmin(Hh)) if np.isfinite(Hh).any() else float(ally.min())
        meta = np.array([ox, oz, sx0, sy0, sz0, min_y - VOID_DROP, r, BUILD_VERSION], np.float64)
        arrays = dict(tiles=tiles, tile_ij=tij.astype(np.int32), Hh=Hh, Hn=Hn, P1=P1.astype(np.float32),
                      P2=P2.astype(np.float32), Pd=Pd, Poff=Poff, Pdim=Pdim, Rr=Rr, Rl=Rl, Rn=Rn, Rf=Rf, bidx=bidx,
                      bricks=bricks, seg=seg.astype(np.float64), rstart=rstart, ritems=ritems, boxes=boxes,
                      uzones=uz, meta=meta)
        times["total"] = time.perf_counter() - t_all
        info = {"build_s": times, "rails": rstats, "tiles": T, "bricks": len(ub), "grid_cells": (nx_c, nz_c),
                "numba": HAVE_NUMBA}
        if verbose:
            print("fastreach build", {k: round(v, 2) for k, v in times.items()}, "tiles", T, "bricks", len(ub))
        return cls(arrays, info)

    @staticmethod
    def cache_key(static_key: str, depth, unknown_boxes, death_min_extent: float = 20.0, r: float = R_BALL) -> str:
        """Everything a build depends on: the static geometry hash, the death boxes it keeps, the moving-body zones
        (they arrive with the Physicalize increments: a tracker at level load has none), the build parameters, the
        physics constants and BUILD_VERSION."""
        import hashlib
        h = hashlib.sha256()
        for item in depth:
            lo, hi = np.asarray(item[-2], np.float64), np.asarray(item[-1], np.float64)
            if min(hi[0] - lo[0], hi[2] - lo[2]) >= death_min_extent:
                h.update(np.r_[lo, hi].astype("<f8").tobytes())
        h.update(b"|")
        for lo, hi in unknown_boxes:
            h.update(np.r_[np.asarray(lo, np.float64), np.asarray(hi, np.float64)].astype("<f8").tobytes())
        h.update(np.array([death_min_extent, r, *CONSTS], "<f8").tobytes())
        return f"{static_key}:{h.hexdigest()[:16]}:{BUILD_VERSION}"

    @classmethod
    def tracker_key(cls, tracker):
        """(cache key, moving-body boxes [(lo, hi)]) of the tracker's level."""
        from .state import G_STOPPER
        lvl = tracker.level
        ub = []
        for ck, p in lvl.phys_by_ent.items():
            if p.unphysicalize or p.fixed or lvl.in_group(ck, G_STOPPER):
                continue
            bb = lvl.mesh_world_bbox(ck)
            if bb is not None:
                ub.append(bb)
        return cls.cache_key(lvl.static_key(), lvl.depth, ub), ub

    @classmethod
    def for_tracker(cls, tracker, cache_dir=None, name: str | None = None, rebuild: bool = False,
                    verbose: bool = False) -> "FastReach":
        """Fields of the tracker's level, cached in runtime/fastreach/<name>.npz (keyed by tracker_key: static
        geometry, death boxes, moving-body zones, build parameters, physics constants, BUILD_VERSION)."""
        from .features import level_geometry
        lvl = tracker.level
        key, ub = cls.tracker_key(tracker)
        if cache_dir is None:
            from . import instances
            cache_dir = instances.runtime_dir() / "fastreach"
        cache_dir = Path(cache_dir)
        name = name or f"L{int(getattr(tracker, 'level_no', 0) or 0):02d}"
        path = cache_dir / f"{name}.npz"
        if path.is_file() and not rebuild:
            fr = cls.load(path)
            if fr.info.get("key") == key:
                return fr
        geo = level_geometry(tracker)
        fr = cls.build(geo.tris, geo.rails, lvl.depth, ub, verbose=verbose)
        fr.info["key"] = key
        cache_dir.mkdir(parents=True, exist_ok=True)
        fr.save(path)
        return fr

    def save(self, path) -> None:
        """Atomic: written to a unique temporary file next to `path`, then renamed over it, so a reader (another
        worker on the same host) never sees a partial file. When the rename fails (Windows: the target is open in
        another process) and `path` already exists, the existing file is kept (a concurrent writer of the same key
        produced it)."""
        import json
        import uuid
        path = Path(path)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.npz")
        try:
            np.savez(tmp, info=np.array(json.dumps(self.info, default=float)),
                     **{k: getattr(self, k) for k in self.ARRAYS})
            for attempt in range(20):
                try:
                    os.replace(tmp, path)
                    return
                except PermissionError:
                    if attempt == 19:
                        if path.is_file():
                            return
                        raise
                    time.sleep(0.1)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    @classmethod
    def load(cls, path) -> "FastReach":
        import json
        with np.load(path) as z:  # closed here: an open handle would block another process's rename on Windows
            info = json.loads(str(z["info"])) if "info" in z.files else {}
            arrays = {k: z[k] for k in cls.ARRAYS}
        return cls(arrays, info)

    # ------------------------------------------------------------------------------- fields keyed by static geometry
    @staticmethod
    def zone_boxes(level, exclude_names=()) -> list:
        """Rest-pose boxes [(lo, hi)] of the physicalized non-fixed entities (moving platforms, boxes, bridges, ball
        pieces) of a state.LevelData, stoppers and the entities named in exclude_names left out (e.g. the player
        balls: their Physicalize record is where the ball was transformed or respawned, not an obstacle)."""
        from .state import G_STOPPER
        ex = set(exclude_names)
        out = []
        for ck, p in level.phys_by_ent.items():
            if p.unphysicalize or p.fixed or level.in_group(ck, G_STOPPER) or level.name(ck) in ex:
                continue
            bb = level.mesh_world_bbox(ck)
            if bb is not None:
                out.append(bb)
        return out

    def set_unknown_boxes(self, boxes) -> None:
        """Replace the unknown zones (rest-pose boxes [(lo, hi)], inflated by 4 as in build). The static fields do
        not depend on them, so one build serves every moving-body state of the same static geometry."""
        self.uzones = np.array([np.r_[np.asarray(lo, np.float64) - 4.0, np.asarray(hi, np.float64) + 4.0]
                                for lo, hi in boxes], np.float64).reshape(-1, 6)

    @classmethod
    def fields_key(cls, tracker, static_key: str | None = None) -> str:
        """Cache key of the static fields only (static geometry hash, death boxes, build parameters, physics constants,
        BUILD_VERSION): cache_key without moving-body zones. tracker_key also hashes the moving-body boxes, which change
        at every sector start, transformation and respawn (L2 TAS: 7 distinct tracker keys over 2 static geometries)."""
        lvl = tracker.level
        return cls.cache_key(static_key if static_key is not None else lvl.static_key(), lvl.depth, ())

    @classmethod
    def fields_for_tracker(cls, tracker, cache_dir=None, static_key: str | None = None, lock_timeout_s: float = 120.0,
                           verbose: bool = False) -> "FastReach":
        """Static fields of the tracker's current geometry, cached content-addressed in
        <cache_dir>/L<nn>-<sha16 of fields_key>.npz (default <runtime>/fastreach, or $BALLRAIN_FASTREACH_DIR), with
        no unknown zones (set them with set_unknown_boxes). Safe when many processes on one host need the same file
        at once: one builds under a lock file (O_EXCL; a lock older than lock_timeout_s is taken over), the others
        wait and load its result; the file is written atomically (save). Identical keys on every host: the key is a
        hash of the level content, not of the host or the time."""
        import hashlib
        from .features import level_geometry
        key = cls.fields_key(tracker, static_key)
        if cache_dir is None:
            env = os.environ.get("BALLRAIN_FASTREACH_DIR")
            if env:
                cache_dir = Path(env)
            else:
                from . import instances
                cache_dir = instances.runtime_dir() / "fastreach"
        cache_dir = Path(cache_dir)
        name = f"L{int(getattr(tracker, 'level_no', 0) or 0):02d}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"
        path = cache_dir / f"{name}.npz"

        def try_load():
            if not path.is_file():
                return None
            try:
                fr = cls.load(path)
            except Exception:  # noqa: BLE001 - unreadable (should not happen with atomic writes): rebuild
                return None
            return fr if fr.info.get("key") == key else None

        fr = try_load()
        if fr is not None:
            return fr
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock = cache_dir / f"{name}.lock"
        t0 = time.time()
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                fr = try_load()
                if fr is not None:
                    return fr
                try:
                    stale = time.time() - lock.stat().st_mtime > lock_timeout_s
                except FileNotFoundError:
                    continue
                if stale:
                    try:
                        lock.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                time.sleep(0.25)
                continue
            # heartbeat: the lock's mtime is refreshed while this process builds, so a lock left by a killed
            # process (a worker restart during a build) goes stale after lock_timeout_s, a live build never does
            import threading
            done = threading.Event()

            def beat():
                while not done.wait(5.0):
                    try:
                        os.utime(lock)
                    except OSError:
                        pass
            hb = threading.Thread(target=beat, daemon=True)
            try:
                os.write(fd, f"{os.getpid()}\n".encode())
                os.close(fd)
                hb.start()
                fr = try_load()  # built by another process between our check and the lock
                if fr is None:
                    lvl = tracker.level
                    geo = level_geometry(tracker)
                    fr = cls.build(geo.tris, geo.rails, lvl.depth, (), verbose=verbose)
                    fr.info["key"] = key
                    fr.info["wait_s"] = time.time() - t0
                    fr.info["built_by"] = os.getpid()
                    fr.save(path)
                return fr
            finally:
                done.set()
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass

    def sizes(self) -> dict:
        """Bytes per component (in memory = on disk, uncompressed npz)."""
        return {c: int(sum(getattr(self, k).nbytes for k in ks)) for c, ks in self.COMPONENTS.items()}

    # ------------------------------------------------------------------------------------------------ query
    def tables(self, ball_type: int):
        t = self._tables.get(ball_type)
        if t is None:
            b = BALLS[int(ball_type)]
            S1, S2 = air_tables(b)
            t = (S1, S2, b)
            self._tables[ball_type] = t
        return t

    _DIRS: dict = {}

    @classmethod
    def directions(cls, quarter: int):
        """(8, 3): world x, z unit direction and push magnitude of each key combo; camera right and forward."""
        q = int(quarter) % 4
        hit = cls._DIRS.get(q)
        if hit is not None:
            return hit
        hit = cls._directions(q)
        cls._DIRS[q] = hit
        return hit

    @staticmethod
    def _directions(quarter: int):
        u = reach.key_push(np.array(KEY_COMBOS), int(quarter) % 4)
        mag = np.linalg.norm(u, axis=1)
        dirs = np.c_[u / mag[:, None], mag]
        return dirs, reach.key_push(tas.KEY_RIGHT, quarter), reach.key_push(tas.KEY_UP, quarter)

    def _args(self):
        return (self.tiles, self.Hh, self.P1, self.P2, self.Pd, self.Poff, self.Pdim, self.Rr, self.Rl, self.Rn,
                self.Rf, self.bidx, self.bricks, self.seg, self.rstart, self.ritems, self.ntz, self.ox, self.oz,
                self.sx0, self.sy0, self.sz0, self.boxes, self.void_y, self.uzones)

    def perceive(self, pos, vel, ball_type: int = 0, quarter: int = 0, debug: bool = False, collect: int = 0,
                 bodies=None):
        """pos (3,), vel (3,) (BALL.pos, BALL.speed + speed_change: units / phys-s), ball type, camera quarter;
        bodies: optional (k, 3) positions of the moving bodies (tracker.obs.body pos): unknown_zone is also set when
        one is within UNKNOWN_R of the ball. Returns (vector (N_FEATURES,) float64 (a copy), info dict or None).
        collect > 0: info["landings"] holds up to that many reachable landing cells (x, h, z, n, effort) of an
        airborne state."""
        S1, S2, b = self.tables(int(ball_type))
        dirs, rv, fv = self.directions(quarter)
        col = np.zeros((int(collect), 5)) if collect else self._nocollect
        p = np.asarray(pos, np.float64)
        v = np.asarray(vel, np.float64)
        if not (np.isfinite(p).all() and np.isfinite(v).all()):
            raise ValueError(f"fastreach.perceive: non-finite state pos={p} vel={v}")
        self.stamp_id = _perceive(*self._args(), S1, S2, b.air_lambda, b.air_accel, b.ground_lambda, b.ground_accel,
                                  b.radius, float(p[0]), float(p[1]), float(p[2]), float(v[0]), float(v[1]),
                                  float(v[2]), dirs, rv, fv, self.yy, self.xf, self.zf, self.D, self.stack, self.stamp,
                                  self.stamp_id, col, self.work, self.out, self.dbg, self.pdd)
        vec = self.out.copy()
        if bodies is not None and len(bodies):
            bp = np.asarray(bodies, np.float64).reshape(-1, 3)
            if (np.linalg.norm(bp - p, axis=1) < UNKNOWN_R).any():
                vec[2] = 1.0
        if not debug:
            return vec, None
        info = {"named": dict(zip(FEATURE_NAMES, vec.tolist())), "dbg": dict(zip(DBG_NAMES, self.dbg.tolist())),
                "dirs": {DIR_NAMES[q]: dict(zip(PER_DIR_DBG, self.pdd[q].tolist())) for q in range(8)},
                "ball": b.name, "supported": b.supported}
        if collect:
            m = col[:, 3] > 0
            info["landings"] = col[m].copy()
        return vec, info

    def landing_reachable(self, pos, vel, target_xz, half: float = RES, ball_type: int = 0):
        """Is a surface point within `half` (x-z square) of target_xz a reachable landing? (found, first PSI)."""
        if not (np.isfinite(np.asarray(pos, float)).all() and np.isfinite(np.asarray(vel, float)).all()):
            raise ValueError("fastreach.landing_reachable: non-finite state")
        S1, S2, b = self.tables(int(ball_type))
        filt = np.array([target_xz[0] - half, target_xz[0] + half, target_xz[1] - half, target_xz[1] + half])
        res = np.zeros(16)
        self.stamp_id += 1
        _air_scan(self.tiles, self.Hh, self.P1, self.P2, self.Pd, self.Poff, self.Pdim, self.seg, self.rstart,
                  self.ritems, self.ntz, self.ox, self.oz, self.boxes, self.void_y, S1, S2, float(pos[0]),
                  float(pos[1]), float(pos[2]), float(vel[0]), float(vel[1]), float(vel[2]), b.air_accel, b.radius,
                  filt, np.zeros(3), 0, self.yy, self.xf, self.zf, self.D, self.stack, self.stamp, self.stamp_id,
                  self._nocollect, res)
        return bool(res[0] > 0), int(res[1])

    def support(self, pos, tol: float = GROUND_TOL):
        """(rest height or nan, contact normal (3,), flag 0 edge / 1 face / 2 rail / -1 none) at the ball's column."""
        rest, nx, ny, nz, fl = _r_ground(self.tiles, self.Rr, self.Rl, self.Rn, self.Rf, self.ox, self.oz,
                                         float(pos[0]), float(pos[2]), float(pos[1]), float(tol))
        return rest, np.array([nx, ny, nz]), fl

    def rest_layers(self, x, z):
        """All R clusters at the column of (x, z): array (k, 4) rest, low, flag, n_y."""
        i, j = _cidx(float(x), self.ox), _cidx(float(z), self.oz)
        t = _tile_of(self.tiles, i, j)
        if t < 0:
            return np.zeros((0, 4))
        rows = []
        for k in range(KR):
            rest, low, nx, ny, nz, fl = _r_read(self.Rr, self.Rl, self.Rn, self.Rf, t, i % TL, j % TL, k)
            if rest != rest:
                break
            rows.append((rest, low, fl, ny))
        return np.array(rows).reshape(-1, 4)

    def sdf(self, p):
        """Distance to the static mesh (unsigned) and its gradient at p."""
        return _sdf(self.bidx, self.bricks, self.sx0, self.sy0, self.sz0, float(p[0]), float(p[1]), float(p[2]))

    def rail_dist(self, p):
        return _rail_dist(self.seg, self.rstart, self.ritems, self.ntz, self.ox, self.oz, float(p[0]), float(p[1]),
                          float(p[2]))

    def march(self, pos, direction, y=None, lmax: float = L_MAX, ball_type: int = 0, k0: float = NAN):
        """The ground march (debug / validation): dict of edge_s, y_edge, k_edge, obst_s, obst_h, s_near, s_far,
        h_far, y_last."""
        d = np.asarray(direction, np.float64)
        d = d / np.linalg.norm(d)
        if y is None:
            y = self.support(pos)[0]
        out = np.zeros(9)
        _march(self.tiles, self.Hh, self.Rr, self.Rl, self.Rn, self.Rf, self.bidx, self.bricks, self.seg, self.rstart,
               self.ritems, self.ntz, self.ox, self.oz, self.sx0, self.sy0, self.sz0, BALLS[int(ball_type)].radius,
               float(pos[0]), float(y), float(pos[2]), float(d[0]), float(d[1]), float(lmax), True, float(k0), out)
        return dict(zip(("edge_s", "y_edge", "k_edge", "obst_s", "obst_h", "s_near", "s_far", "h_far", "y_last"),
                        out.tolist()))


def gap_takeoff_speed(fr: FastReach, x_need: float, dy_target: float, k: float = 0.0, ball_type: int = 0,
                      push: float = 1.0, vy0: float = 0.0):
    """Minimum horizontal takeoff speed (units / phys-s) to travel x_need horizontally before the centre has fallen
    below dy_target (relative to the takeoff); takeoff vertical speed k v + vy0; push = per-axis key push along the
    motion in the air (0 coasting, 1 one key along a world axis; keys are independent unit pushes per world axis, so
    a diagonal adds nothing along a world axis)."""
    S1, S2, b = fr.tables(ball_type)
    return _v_need(S1, S2, len(S1) - 1, float(k), float(vy0), b.air_accel * push, float(x_need), float(dy_target))
