"""Evaluate first lives from branch rest starts using a frozen local checkpoint."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from probe_l2_s3_route import BUILDS, RECORD, engine
from ballrain.branch_assets import validate
from ballrain.env import BallanceEnv, EnvConfig, FeatureConfig, RewardConfig
from ballrain.train.ppo import ActorCritic, normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runtime', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--copy', required=True, help='reserved local engine slot label')
    p.add_argument('--build', choices=BUILDS, required=True)
    p.add_argument('--seeds', type=int, default=6)
    p.add_argument('--cap', type=int, default=1500)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR', str(a.out / 'numba'))
    torch.set_num_threads(1)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['env_config'].copy()
    args = ck['args']
    fx = FeatureConfig(**cfg['features'])
    fx.path_dir = str(a.runtime.parent / fx.path_dir)
    assert validate(ck['branch_assets'], cfg['level'], fx.path_sha256) == fx.branch_sha256
    headless = cfg['headless'].copy()
    local = engine(a.runtime, a.build)
    headless.update(binary=local.binary, root=local.root, socket='tcp')
    cfg.update(features=fx, reward=RewardConfig(**cfg['reward']), branch_assets=ck['branch_assets'],
               headless=headless, copy=a.copy, start_record=str(a.runtime / RECORD), start_frame=10660,
               curriculum=None, p_rest=0, p_rehearse=0, rehearse_always=[], seed=1)
    model = ActorCritic(ck['n_obs'], ck['n_act'], tuple(map(int, args['hidden'].split(','))),
                        args['activation'], args['actor_heads'])
    model.load_state_dict(ck['model'])
    model.eval()
    for label, station in [('s1118', 1207.456418722111), ('s1129', 1323.9255119553718)]:
        cfg['start_rest'] = dict(branch_id='east-platform', branch_s=station)
        for seed in range(1, a.seeds + 1):
            gen = torch.Generator().manual_seed(seed)
            dest = a.out / f'{a.build}-{label}-{seed}.jsonl'
            with tempfile.TemporaryDirectory(dir=a.out, prefix='engine-') as tmp, dest.open('w', encoding='utf8') as stream:
                env = BallanceEnv(EnvConfig(**dict(cfg, out_root=tmp)))
                try:
                    obs, info = env.reset(seed=1)
                    start = info['start']
                    assert start['kind'] == 'tas_rest' and start['frame'] == 10660 and info['supported']
                    assert not start['handover_mismatches'] and not start.get('branch_rest_error')
                    for i in range(a.cap + 1):
                        t = env.session.tracker if env.session else None
                        b = t.obs.ball if t else None
                        r = {k: info.get(k) for k in ('frames_done','sector','supported','branch_id','phase','local_s','events','episode_end')}
                        if i == 0:
                            r['start'] = start
                        if i == a.cap and not r['episode_end']:
                            r['episode_end'] = 'probe_cap'
                        if b is not None and b['found']:
                            r.update(bt=int(b['type']), pos=b['pos'].tolist(),
                                     vel=(b['speed']+b['speed_change']).tolist(),
                                     contacts=list(dict.fromkeys(t.level.name(int(x)) for x in t.obs.fric['partner'])))
                        z = normalize(obs, ck['rms'])
                        if not r['episode_end'] and r['sector'] < 4:
                            with torch.inference_mode():
                                action, _, _ = model.act(z, temperature=1., gen=gen)
                            r['action'] = int(action)
                        print(json.dumps(r, default=str), file=stream)
                        if 'action' not in r:
                            break
                        obs, _, _, _, info = env.step(int(action))
                    print(a.build, label, seed, r['sector'], r['episode_end'], r.get('pos'), flush=True)
                finally:
                    env.close()


if __name__ == '__main__':
    main()
