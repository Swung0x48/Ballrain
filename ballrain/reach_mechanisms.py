"""Retail reach-mechanism assets and cold-path geometry compilation.

The JSON beside this module is distributed with the Python checkout, not discovered
from a machine-local routes directory. Only `layers` and plate meshes are active;
the rest is an inventory, not an assertion of traversability.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).with_name("reach_mechanism_levels")
CAPABILITY = "reach_mechanisms_v1"


@lru_cache(maxsize=1)
def asset_pin():
    files = tuple(ASSETS / f"L{k:02d}.json" for k in range(1, 14))
    h = hashlib.sha256()
    for path in files:
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:16]


@lru_cache(maxsize=13)
def level_asset(level):
    data = json.loads((ASSETS / f"L{level:02d}.json").read_bytes())
    if data["schema"] != 1 or data["level"] != level:
        raise ValueError(f"invalid reach-mechanism asset for level {level}")
    return data


def layers_for_tracker(tracker):
    """Cold path: (wood, paper, stone) read-only triangle additions, None = baseline.

    Require an exported physical mesh at the pinned placement: phase increments can
    instantiate plates later, and a renamed/moved custom placement must not inherit
    the retail overlay. This does not infer the instantaneous angle from BODY.
    """
    n = tracker.level_no
    if not 1 <= n <= 13:
        raise ValueError(f"no retail reach-mechanism asset for level {n}")
    data = level_asset(n)
    level = tracker.level
    matched = []
    for plate in data["plates"]:
        ck = level.by_name.get(plate["object"])
        if ck is None or ck not in level.phys_by_ent:
            continue
        bbox = level.mesh_world_bbox(ck)
        # The full body's bbox includes the small mounting tab; compare against
        # the measured full-body bbox, not the main plate's synthetic triangles.
        source = data["physical_parts"]["P_Modul_25"][plate["object"]]["bbox"]
        if bbox is not None and np.allclose(bbox, source, atol=0.15, rtol=0):
            matched.append(plate)
    by_layer = {name: None for name in data["layers"].values()}
    for name in by_layer:
        pieces = [np.asarray(p[name], dtype=np.float64).reshape(-1, 3, 3) for p in matched if name in p]
        if pieces:
            by_layer[name] = np.concatenate(pieces)
            by_layer[name].flags.writeable = False
    layers = tuple(by_layer[data["layers"][str(k)]] for k in range(3))
    # We can select a whole alternative field for one matched plate; multiple
    # independent switches would need a different (local overlay) field design.
    plate = matched[0] if len(matched) == 1 else None
    probe = plate.get("state_probe") if plate is not None else None
    if probe is not None:
        pose = (level.by_name[plate["object"]], *probe["pivot"], *probe["down_axis"], probe["min_alignment"])
        probe = (probe["from_ball"], probe["field_of_ball"], pose)
    return layers, probe


def fan_layer(tracker):
    """Compile the normal and recovered fan OBBs once, only on matching retail PH data."""
    from .fastreach import fan_boxes, fan_obbs

    existing = fan_boxes(tracker.level)
    data = level_asset(tracker.level_no)
    arrays = tracker.arrays
    additions = []
    for candidate in data["recovered_fans"]:
        i = candidate["ph_row"]
        placement = data["placements"][i]
        mat = np.asarray(candidate["matrix"], dtype=np.float64)
        live_matrix = arrays.get("PH", i, 2)
        if (arrays.get("PH", i, 0) == placement["sector"] and arrays.get("PH", i, 1) == placement["type"]
                and live_matrix is not None and np.allclose(live_matrix, placement["matrix"], atol=0.02, rtol=0)
                and not any(np.linalg.norm(old[3] - mat[3]) < 0.5 for _, old in existing)):
            additions.append((f"PH[{i}]", mat))
    return fan_obbs(existing + additions), bool(additions)


def uses_lowered_plate(tracker, pos, probe):
    """One pose check, localized to the one retail plate with a verified down geometry.

    CK's first row is the world-space long axis of this panel. The transform
    was checked against the composed IVP pose in SR02 near the hinge. There is
    no extrapolation, allocation of a mesh, or held-over state after a reset.
    """
    ckid, x, y, z, dx, dy, dz, alignment = probe
    if abs(pos[0] - x) > 32.0 or abs(pos[2] - z) > 32.0 or abs(pos[1] - y) > 12.0:
        return False
    obs = tracker.obs
    if obs is None or not len(obs.body):
        return False
    ids = obs.body["ckid"]
    # BODY has no index. Only check for the one preselected CKID, in native
    # array code; never search the level's mechanisms in the step path.
    matches = np.flatnonzero(ids == ckid)
    if len(matches) != 1:
        return False
    mat = obs.body[matches[0]]["ck_matrix"]
    a, b, c = mat[0]
    px, py, pz = mat[3]
    return (a * dx + b * dy + c * dz >= alignment and abs(px - x) <= 0.75
            and abs(py - y) <= 0.75 and abs(pz - z) <= 0.75)


def add_geometry(geometry, triangles):
    """Append the same world-space triangles to either reach implementation."""
    a, b, c, owner, rail = geometry
    new_owner = int(owner.max()) + 1 if len(owner) else 0
    return (*[np.concatenate((v, triangles[:, i])) for i, v in enumerate((a, b, c))],
            np.concatenate((owner, np.full(len(triangles), new_owner, dtype=owner.dtype))),
            np.concatenate((rail, np.zeros(len(triangles), dtype=bool))))
