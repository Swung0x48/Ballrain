"""Pinned moving branch starts (opt-in, no pose writes or braking)."""
from pathlib import Path

import numpy as np

from . import tas
from .branch_assets import rest_member
from .features import FeatureExtractor, yaw_of
from .protocol import Ev


POSITION_TOLERANCE = 0.10
VELOCITY_TOLERANCE = 0.20
CURSOR_TOLERANCE = 2.0
HANDOVER_TABLE = 'moving_handover_v1'


def uses_branch_move(cfg):
    return bool((getattr(cfg, 'start_rest', None) or {}).get('branch_move') or any(
        len(start) == 3 and isinstance(start[2], dict) and start[2].get('branch_move')
        for start in getattr(cfg, "rest_starts", None) or ()
    ))


def select_handover(drive, k, station, record_sha, branch_sha):
    if (k != int(drive['prefix_frame']) or str(drive['record_sha256']) != record_sha
            or str(drive['branch_sha256']) != branch_sha):
        raise ValueError('moving replay does not match the prefix, recording or branch line')
    if HANDOVER_TABLE not in drive:
        raise ValueError('branch drive has no pinned moving handovers')
    samples = drive['samples']
    ix = np.flatnonzero(samples[:, 1] == station)
    if len(ix) != 1:
        raise ValueError('moving start needs a unique exact recorded station (no snapping)')
    offset = samples[ix[0], 0]
    table = drive[HANDOVER_TABLE]
    if table.ndim != 2 or table.shape[1] != 10 or not np.isfinite(table).all():
        raise ValueError('invalid moving handover table')
    rows = table[table[:, 0] == offset]
    frames = drive['frames']
    if len(rows) != 1 or not 0 < offset <= len(frames) or int(offset) != offset:
        raise ValueError('moving start not approved at this recorded frame')
    expected = rows[0]
    if np.any(expected[:4] != expected[:4].astype(int)):
        raise ValueError('non-integral moving handover metadata')
    step = drive['moving_replay_step_v1'].item()
    if type(step) is not int or not 1 <= step <= 16:
        raise ValueError('invalid moving replay observation cadence')
    return frames[:int(offset)], expected, step


def replay(env, prepared, k, station, key):
    from .env import ARROW_MASK
    from .path import asset_path, file_sha256

    if not env.cfg.branch_assets or env.cfg.branch_assets.get('version') != 2 or not key:
        raise ValueError('moving start needs a named phased v2 branch')
    root = Path(env._branch_temp.name)
    key, _ = rest_member(env.cfg.branch_assets, key)
    session = prepared['session']
    tracker = session.tracker
    fx = FeatureExtractor(env.cfg.features, branch_root=root)
    fx.set_branch_entry_context("drill")  # Explicit replay must use the same entrances as the sampler.
    fx.reference_path(tracker)
    follower = fx.branch
    entry, branch = follower.by_id[key]
    if entry.get('legacy'):
        raise ValueError('moving starts require a phased branch')
    with np.load(root/key/'drive.npz', allow_pickle=False) as drive:
        frames, expected, step = select_handover(
            drive, k, station, prepared['start']['record_sha256'],
            file_sha256(asset_path(env.cfg.level, root/key)),
        )
    offset, phase, ball_type, sector = map(int, expected[:4])
    if (not 0 <= phase < len(entry['phases'])
            or entry['phases'][phase]['when']['ball'] != ball_type
            or not entry['phases'][phase]['local_s'][0] <= station <= entry['phases'][phase]['local_s'][1]):
        raise ValueError('moving handover inconsistent with pinned phase')
    if np.any(frames['keys'] & np.uint32(0xffffffff ^ ARROW_MASK)):
        raise ValueError('moving handover currently supports only arrow-only drives')

    def quarter():
        cams = tracker.obs.cams
        if cams is None:
            raise RuntimeError('moving replay lacks camera orientation')
        return round(yaw_of(cams['cams'][0]['matrix']) / (np.pi / 2)) % 4

    initial_quarter = quarter()
    origin = session.frames_done
    pr = None
    if tracker.control_valid():
        _, pr = follower.update(tracker.obs.ball['pos'], tracker.sector, tracker.ball_type())
    for n in range(0, len(frames), step):
        session.act(frames[n:n+step], stop=(Ev.BallOff, Ev.GameOver, Ev.LevelFinish))
        if session.ended or tracker.flags.ball_off or session.frames_done != origin + min(n+step, len(frames)):
            raise RuntimeError('moving replay interrupted before the exact handover frame')
        if quarter() != initial_quarter:
            raise RuntimeError('camera orientation changed during an arrow-only moving replay')
        if tracker.control_valid():
            b = tracker.obs.ball
            _, pr = follower.update(b['pos'], tracker.sector, tracker.ball_type())
    if not tracker.control_valid() or tracker.sector != sector or tracker.ball_type() != ball_type:
        raise RuntimeError('moving handover has wrong material, sector or control state')
    if (not pr or pr.get('branch_id') != key or pr.get('phase') != phase
            or abs(pr['local_s']-station) > CURSOR_TOLERANCE):
        raise RuntimeError('moving replay did not naturally acquire the expected phase and station')
    b = tracker.obs.ball
    pos = np.asarray(b['pos'])
    vel = b['speed'] + b['speed_change']
    pe = float(np.linalg.norm(pos-expected[4:7]))
    ve = float(np.linalg.norm(vel-expected[7:10]))
    if not (pe <= POSITION_TOLERANCE and ve <= VELOCITY_TOLERANCE):
        raise RuntimeError(f'moving handover exceeds position/velocity tolerance: {pe:.4g}/{ve:.4g}')
    handover = dict(kind='branch_moving', branch_id=key, phase=phase, branch_s=float(pr['local_s']),
                    control_frame=session.frames_done, sr_ms=tracker.sr_ms, branch_move_offset=offset,
                    branch_move_station=station, branch_move_keys=int(frames['keys'][-1]),
                    branch_move_position_error=pe, branch_move_velocity_error=ve)
    prepared['start'].update(handover)
    prepared['recorder'].meta.start_state.update(handover)
