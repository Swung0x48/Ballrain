"""Check physical branch-rest handovers and single-life continuations on a local build."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from build_l2_s3_route import MAIN
from probe_l2_s3_route import APPROACH, BUILDS, EXIT, MIDDLE, RAIL, RECORD, drive, engine
from ballrain.branch_assets import read_bundle, validate
from ballrain.env import BallanceEnv, EnvConfig, FeatureConfig, RewardConfig
from ballrain.path import file_sha256
from ballrain.protocol import Ev
from scripts.train_m2 import parse_rest_starts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for param in ('runtime', 'checkpoint', 'bundle', 'menu', 'out'):
        p.add_argument(param, type=Path)
    p.add_argument('--copy', required=True, help='reserved local engine slot label')
    p.add_argument('--build', choices=BUILDS, required=True)
    p.add_argument('--only', nargs='+')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR', str(a.out/'numba'))
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['env_config'].copy()
    assets = read_bundle(2, a.bundle)
    pin = validate(assets, 2, file_sha256(MAIN))
    fx = FeatureConfig(**cfg['features'])
    assert fx.path_sha256 == file_sha256(MAIN)
    fx.path_dir = str(MAIN.parent)
    fx.branch_sha256 = pin
    local = engine(a.runtime, a.build)
    headless = dict(cfg['headless'], binary=local.binary, root=local.root, socket='tcp')
    cfg.update(features=fx, reward=RewardConfig(**cfg['reward']), branch_assets=assets,
               headless=headless, copy=a.copy, start_record=str(a.runtime/RECORD), curriculum=None,
               p_rest=1., p_rehearse=0, rehearse_always=[], seed=1)
    menu = json.loads(a.menu.read_text(encoding='utf8'))
    for name, entry in menu.items():
        if a.only and name not in a.only:
            continue
        stage = entry['stage']
        spec = entry['spec']
        rest = parse_rest_starts(spec)
        out = a.out / f'{a.build}-{name}'
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out, prefix='engine-') as tmp:
            env = BallanceEnv(EnvConfig(**dict(cfg, start_frame=rest[0][0], start_rest=None,
                                               rest_starts=rest, out_root=tmp)))
            try:
                _, info = env.reset(seed=1)
                s = env.session
                obs = s.tracker.obs
                b = obs.ball
                r = dict(pin=pin, spec=spec, supported=bool(info['supported']), start=info['start'],
                         pos=b['pos'].tolist(), vel=(b['speed']+b['speed_change']).tolist(),
                         ball=int(b['type']), contacts=[s.tracker.level.name(int(c['partner'])) for c in obs.fric])
                r['valid_rest'] = bool(r['supported'] and r['start']['kind']=='tas_rest' and
                                   not r['start']['handover_mismatches'] and not r['start'].get('branch_rest_error') and
                                   r['start'].get('branch_id')=='east-platform' and s.tracker.sector == 3 and
                                   np.linalg.norm(np.asarray(r['vel'])[[0,2]]) <= .501)
                if r['valid_rest']:
                    route = APPROACH+MIDDLE+EXIT
                    for idx, xy in entry.get('waypoints', {}).items():
                        i = int(idx)
                        route[i] = (*xy, *route[i][2:])
                    samples, frames, log = drive(s, obs, route, stage, 12,
                                                 max_steps=1500, rail=RAIL, **entry.get('controller', {}))
                    r.update(post_checkpoint=bool(samples[-1,6] == 4 and
                                                 any(f>0 and c==Ev.PostCheckpoint for f,c in log['events'])),
                             end=samples[-1].tolist(), elapsed_ms=float(samples[-1,7]-samples[0,7]))
                    np.savez_compressed(out/'trace.npz', samples=samples, frames=frames)
                    (out/'events.json').write_text(json.dumps(log), encoding='utf8')
                (out/'result.json').write_text(json.dumps(r, indent=2, default=str)+'\n', encoding='utf8')
                print(a.build, name, r['valid_rest'], r.get('post_checkpoint'), r['pos'], r['vel'],
                      r.get('end', [])[1:7], r['start'].get('branch_rest_error'), flush=True)
            finally:
                env.close()


if __name__ == '__main__':
    main()
