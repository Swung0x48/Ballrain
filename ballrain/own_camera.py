"""The agent's own camera in the observation (EnvConfig.own_camera).

With world actions and the "world" feature frame the camera only enters the observation through cam_yaw_ref, cam_yaw and
cam_rotating: the keys are mapped through Cam_OrientRef (world_to_camera_keys), so the camera does not change what an
action does. The agent has no camera actions, so a life played from spawn keeps the spawn camera to the end. A TAS
hand-over carries the camera the TAS had at that frame instead. The L1 TAS turns the camera at frame 7052 (quarter 2 ->
1), so every hand-over after it showed a camera that lives from spawn never see, and what those lives learned did not
carry over (frozen u96000, 小节4 gap crossings of cp4 arrivals: 小节3 hand-overs 84 % -> 41 % with the spawn camera
written in; lives from spawn 58 % -> 75 % with the TAS camera written in after the turn point).

own_camera writes the spawn camera (the record's camera at its first start frame) into those inputs on every step of
every life. Lives from spawn are unchanged. view() rebuilds the camera that frozen teachers trained before this option
saw on TAS hand-overs: the TAS camera at the row's route position."""
from __future__ import annotations

import math

import numpy as np

CAM_FIELDS = ("cam_yaw_ref", "cam_yaw", "cam_rotating")


def quarter_vec(q: int) -> tuple[float, float]:
    """(cos, sin) of a yaw of q quarter turns, as features.compute writes cam_yaw_ref / cam_yaw (yaw_of -> cos, sin)."""
    y = (int(q) % 4) * math.pi / 2
    return (round(math.cos(y)), round(math.sin(y)))


def write(obs: np.ndarray, slices: dict, q: int) -> np.ndarray:
    """Write a resting camera of q quarter turns into the camera inputs of obs (one row or a batch), in place."""
    c, s = quarter_vec(q)
    for k in ("cam_yaw_ref", "cam_yaw"):
        obs[..., slices[k].start] = c
        obs[..., slices[k].start + 1] = s
    obs[..., slices["cam_rotating"].start] = 0.0
    return obs


def spawn_quarter(starts) -> int:
    """Camera quarter of the record at its first start frame (the first controllable frame: the spawn camera)."""
    q = np.asarray(starts.camera_quarters())
    f0 = int(starts.control_frame)
    v = int(q[min(f0, len(q) - 1)])
    if v < 0:
        raise ValueError("own_camera: the start record has no camera at its first start frame")
    return v


def turn_points(starts, path) -> list[tuple[float, int]]:
    """[(route s, quarter)] of the record's camera from its first start frame on: the first entry is (0, spawn quarter),
    then one per quarter change, at the route s of the TAS ball (nearest line point) at that frame."""
    q = np.asarray(starts.camera_quarters())
    f0 = int(starts.control_frame)
    fd = np.asarray(starts.valid)
    out = [(0.0, spawn_quarter(starts))]
    for f in range(f0 + 1, len(q)):
        if q[f] != q[f - 1] and q[f] >= 0:
            i = min(int(np.searchsorted(fd, f)), len(fd) - 1)  # nearest start frame at or after the turn
            p = np.asarray(starts.pos[i], np.float64)
            j = int(np.argmin(np.linalg.norm(path.P - p, axis=1)))
            out.append((float(path.cum[j]), int(q[f])))
    return out


def view(raw: np.ndarray, slices: dict, s_col: int, length: float, points: list[tuple[float, int]]) -> np.ndarray:
    """Copy of raw rows with the camera inputs the TAS had at each row's route s (raw[:, s_col] * length)."""
    out = np.array(raw, copy=True)
    s = out[:, s_col].astype(np.float64) * length
    for k, (s0, q) in enumerate(points):
        s1 = points[k + 1][0] if k + 1 < len(points) else math.inf
        m = (s >= s0) & (s < s1)
        if m.any():
            out[m] = write(out[m], slices, q)
    return out
