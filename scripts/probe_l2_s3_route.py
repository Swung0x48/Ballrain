"""Run the sector-3 controller in an isolated, local headless session."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain import tas
from ballrain.client import AgentSession, SessionConfig
from ballrain.headless import HeadlessConfig, HeadlessLauncher
from ballrain.env import world_to_camera_keys, EPISODE_STOP
from ballrain.protocol import Ev
from ballrain.path import ReferencePath

BUILDS = {'soft': 'x86soft', '2002': 'x86p2002f-respawnfix'}
MAIN = Path(__file__).resolve().parents[1] / 'benchmarks/refpaths-v2/L02.npz'
RECORD = Path('datasets/tas-v1') / ReferencePath.load(MAIN).meta['record']
APPROACH = [(906, -346.8), (916.35, -346.74, 'paper'), (988, -346.74),
            (991.4, -366.74), (1016.67, -366.74, 61), (1006.67, -366.74, 72),
            (1026.68, -366.74, 'stone'), (1015, -366.7), (992, -366.7)]
EXIT = [(927, -320), (926.5, -300), (927.07, -261.52, 'paper'),
        (941, -206), (935, -145), (929, -96)]
MIDDLE = [(948, -363.98), (934, -363.98), (932, -357.5)]
RAIL = (970, 21, -363.98)
MATERIAL = {'wood': 0, 'paper': 1, 'stone': 2}


def engine(runtime, build):
    hl = runtime / 'hl-win'
    return HeadlessConfig(binary=str(hl / f'bin-{BUILDS[build]}/BallanceMMOSimTool.exe'),
                          root=str(hl / 'game'), socket='tcp')


def drive(session, obs, route, stage=0, exit_start=12, repeat=6, max_steps=1700, rail=None, plate_release=-328, plate_speed=27):
    origin = session.frames_done
    samples, contacts, events, poses = [], [], [], []
    stop = (*EPISODE_STOP, Ev.PostCheckpoint)
    for _ in range(max_steps):
        if obs is None:
            break
        t = session.tracker
        f = session.frames_done - origin
        events += [(f, int(c)) for c in obs.events['code']]
        for body in obs.body:
            name = t.level.name(int(body['ckid']))
            if name == 'P_Modul_25_Bridge':
                poses.append((f, name, body['pos'].tolist(), body['q'].tolist()))
        b = obs.ball
        keys = 0
        if b is not None and b['found']:
            p = np.asarray(b['pos'])
            v = b['speed'] + b['speed_change']
            bt = int(b['type'])
            goal = route[stage]
            d = np.array(goal[:2]) - p[[0, 2]]
            norm = np.linalg.norm(d)
            samples.append((f, *p, bt, stage, t.sector, t.sr_ms, *v))
            contacts.extend((f, t.level.name(int(c['partner'])), bt) for c in obs.fric)
            if obs.has_event(Ev.PostCheckpoint) and t.sector >= 4:
                break
            if session.ended or obs.has_event(Ev.BallOff) or p[1] < -5:
                break
            if len(goal) == 2:
                hit = norm < 1.5
                if rail and 9 <= stage < exit_start - 1:
                    hit = p[0] <= goal[0]
                if stage >= exit_start:
                    hit = hit or p[2] >= goal[1] - 1
            elif isinstance(goal[2], str):
                hit = bt == MATERIAL[goal[2]]
            else:
                hit = p[1] >= goal[2]
            if hit and stage + 1 < len(route):
                stage += 1
            # Keep the previous goal for the rest of this tick at a waypoint boundary.
            fast = stage >= exit_start
            desired = d * min(2, 14 / max(0.01, norm))
            if rail and 9 <= stage < exit_start - 1:
                lead, speed, line = rail
                target = -366.7 if p[0] > lead else line
                desired = np.array([-speed, np.clip((target-p[2])*3, -4, 4)])
            if rail and stage == exit_start - 1:
                desired = np.array([np.clip((932.0-p[0])*3, -8, 8),
                                    np.clip((-357.5-p[2])*2, -7, 7)])
            if fast:
                if stage != exit_start + 2:  # keep the proportional approach to the downstream trafo
                    desired[1] = plate_speed if stage == exit_start and p[2] > plate_release else 27
                x = 931.4 if stage == exit_start and p[2] < plate_release else goal[0]
                desired[0] = np.clip((x - p[0]) * 2, -18, 18)
            w = desired - v[[0, 2]]
            threshold = .5 if fast else .10 if rail and 9 <= stage < exit_start else .28 * np.linalg.norm(w)
            mask = ((tas.KEY_RIGHT if w[0] > 0 else tas.KEY_LEFT)
                    if abs(w[0]) > threshold else 0)
            if abs(w[1]) > threshold:
                mask |= tas.KEY_UP if w[1] > 0 else tas.KEY_DOWN
            if obs.cams is not None:
                keys = world_to_camera_keys(mask, obs.cams['cams'][0]['matrix'])
        if session.ended or obs.has_event(Ev.BallOff):
            break
        obs = session.act(tas.make_frames([1000/132], [keys]), repeat=repeat, stop=stop)
    return np.asarray(samples), session.frame_log()[origin:], dict(contacts=contacts, events=events, poses=poses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runtime', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--build', choices=BUILDS, required=True)
    p.add_argument('--copy', required=True, help='reserved local engine slot label')
    p.add_argument('--frame', type=int, default=10557)
    p.add_argument('--repeat', type=int, default=6)
    p.add_argument('--middle', default='948:-363.98,934:-363.98,932:-357.5')
    p.add_argument('--rail', type=float, nargs=3, metavar=('LEAD_X', 'SPEED', 'Z'), default=RAIL)
    p.add_argument('--prefix-drive', type=Path)
    p.add_argument('--drive-frames', type=int, default=0)
    p.add_argument('--stage', type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'{a.build}-', dir=a.out))
    print(folder, flush=True)
    record = a.runtime / RECORD
    prefix = tas.load_raw(record / 'run/framelog.bin')[:a.frame]
    if a.prefix_drive:
        with np.load(a.prefix_drive) as z:
            if a.frame != int(z['prefix_frame']):
                p.error('drive and TAS prefix frames must match')
            prefix = np.concatenate((prefix, z['frames'][:a.drive_frames]))
    tas.save_raw(folder / 'prefix.bin', prefix)
    middle = [tuple(map(float, v.split(':'))) for v in a.middle.split(',')]
    route = APPROACH + middle + EXIT
    s = AgentSession(SessionConfig(level=2, copy=a.copy, out_dir=folder, prefix=folder/'prefix.bin',
                                   hello_timeout_s=30, obs_timeout_s=30),
                     HeadlessLauncher(engine(a.runtime, a.build)))
    try:
        obs = s.start()
        assert s.frames_done == len(prefix)
        if not a.prefix_drive:
            with np.load(record / 'raw.npz') as z:
                i = np.searchsorted(z['frames_done'], a.frame)
                assert int(z['frames_done'][i]) == a.frame
                assert s.tracker.sector == 3
                np.testing.assert_array_equal(obs.ball['pos'], z['ball_pos'][i])
                assert s.tracker.sr_ms == z['sr_double'][i]
        samples, frames, info = drive(s, obs, route, a.stage, len(APPROACH)+len(middle), a.repeat, rail=a.rail)
        np.savez_compressed(folder/'trace.npz', samples=samples, frames=frames, prefix_frame=len(prefix))
        (folder/'events.json').write_text(json.dumps(info), encoding='utf8')
        print(samples[0], samples[-1], flush=True)
    finally:
        s.close(kill=True)


if __name__ == '__main__':
    main()
