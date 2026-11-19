"""Verify real sampler handovers and continuations; never trains or deploys."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('workspace', 'runtime', 'checkpoint', 'out'):
        p.add_argument(k, type=Path)
    p.add_argument('--build', choices=('soft', '2002'), required=True)
    p.add_argument('--copy', required=True)
    p.add_argument('--seeds', type=int, default=8)
    p.add_argument('--cap', type=int, default=1800)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--menu', nargs='+', type=Path, required=True)
    p.add_argument('--bad-station', action='store_true')
    p.add_argument('--script-check', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR', str(a.out/'numba'))
    sys.path[:0] = [str(a.workspace), str(a.workspace/'scripts')]
    from ballrain.branch_assets import read_bundle, validate
    from ballrain.env import BallanceEnv, EnvConfig, FeatureConfig, RewardConfig
    from ballrain.features import to_frame
    from ballrain.protocol import Ev
    from ballrain.train.ppo import ActorCritic, normalize
    from probe_l2_s3_route import APPROACH, EXIT, MIDDLE, RAIL, RECORD, drive, engine

    torch.set_num_threads(1)
    from scripts.train_m2 import parse_rest_starts
    bundle_dir = a.bundle
    menu = {}
    for source in a.menu:
        menu.update(json.loads(source.read_text(encoding='utf8')))
    if a.bad_station:
        name, entry = next((n,e) for n,e in menu.items() if 'velocity' in e)
        band, member, station = entry['spec'].split(':')
        menu = {name:dict(entry, spec=f'{band}:{member}:{float(station)+0.001}')}
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    c, args = ck['env_config'].copy(), ck['args']
    fx = FeatureConfig(**c['features'])
    fx.path_dir = str(a.workspace/fx.path_dir)
    assets = read_bundle(2, bundle_dir)
    pin = validate(assets, c['level'], fx.path_sha256)
    ckpin = validate(ck['branch_assets'], c['level'], fx.path_sha256)
    assert ckpin == fx.branch_sha256
    fx.branch_sha256 = pin
    local = engine(a.runtime, a.build)
    headless = dict(c['headless'], binary=local.binary, root=local.root, socket='tcp')
    c.update(features=fx, reward=RewardConfig(**c['reward']), branch_assets=assets,
             headless=headless, copy=a.copy, start_record=str(a.runtime/RECORD),
             curriculum=None, start_rest=None, p_rest=1., p_rehearse=0., warm_pool=False, seed=1)
    model = ActorCritic(ck['n_obs'], ck['n_act'], tuple(map(int,args['hidden'].split(','))),
                        args['activation'], args['actor_heads'])
    model.load_state_dict(ck['model']); model.eval()
    def row(env, obs, info, z):
        b = env.session.tracker.obs.ball if env.session is not None else None
        t = env.session.tracker if env.session else None
        r = {k:info.get(k) for k in ('frames_done','sr_ms','sector','supported','branch_id','phase',
                                      'local_s','events','episode_end','surface','gap')}
        if b is None or not b['found']:
            return r
        sl=env.fx.slices(); bt=int(b['type']); pos=np.asarray(b['pos']); v=b['speed']+b['speed_change']
        r.update(pos=pos.tolist(), vel=v.tolist(), bt=bt, active=float(obs[sl['valid']][0]),
                 head=int(model.context(torch.as_tensor(z))), bit=obs[sl['ball_type']].tolist(),
                 contacts=list(dict.fromkeys(t.level.name(int(x)) for x in t.obs.fric['partner'])),
                 norm_max=float(np.max(abs(z))), norm_clip=int(np.count_nonzero(abs(z)>=10)))
        assert np.isfinite(obs).all() and np.isfinite(z).all()
        if r['active'] and bt in (0,1,2):
            assert r['head']==bt
        if r['active'] and r['branch_id']=='east-platform':
            _, branch=env.fx.branch.by_id['east-platform']; path=branch.alt; s=r['local_s']
            near=path.points_ahead(s,fx.waypoints,fx.waypoint_spacing)-pos
            far=path.points_at(s,fx.far_waypoints)-pos
            right,fwd=env.fx.frame(t)
            for k,w in [('waypoints',near),('far_waypoints',far)]:
                np.testing.assert_allclose(obs[sl[k]], (to_frame(w,right,fwd)/fx.pos_scale).ravel(), atol=2e-6, rtol=0)
            r.update(near=near.tolist(), far=far.tolist(), reference_error=float(np.linalg.norm(path.point_at(s)-pos)),
                     ref_speed=(obs[sl['ref_speed']]*fx.vel_scale).tolist(),
                     reach=obs[sl['reach']].tolist() if 'reach' in sl else None)
        return r

    for label, entry in menu.items():
        moving = 'velocity' in entry
        spec = entry['spec'].replace('@branch:', '@branch-move:') if moving else entry['spec']
        parsed = parse_rest_starts(spec)
        k, _, target = parsed[0]
        key, station = target['id'], target['branch_s']
        for seed in range(1,(1 if a.script_check else a.seeds)+1):
            dest=a.out/f'{a.build}-{label}-{seed}.jsonl'
            if dest.exists():
                raise FileExistsError(dest)
            gen=torch.Generator().manual_seed(seed)
            with tempfile.TemporaryDirectory(dir=a.out,prefix='engine-') as tmp,dest.open('w',encoding='utf8') as stream:
                cfg = dict(c, start_frame=None, rest_starts=parsed, out_root=tmp)
                env = BallanceEnv(EnvConfig(**cfg))
                try:
                    obs,info=env.reset(seed=1)
                    start=info['start']
                    kind='branch_moving' if moving else 'tas_rest'
                    assert start['frame']==k and not start['handover_mismatches']
                    if a.bad_station:
                        assert start['kind']=='branch_moving_fallback' and start['branch_move_fallback']
                        assert start['branch_rest_failures']==env.branch_rest_failures==1
                        assert info['frames_done']==k and info['branch_id'] is None
                        print(json.dumps(dict(kind=start['kind'], failures=env.branch_rest_failures,
                                              error=start['branch_rest_error'], frame=k)),file=stream)
                        print(a.build, 'marked fallback at original TAS frame', k, flush=True)
                        continue
                    assert not start.get('branch_rest_error') and not env.branch_rest_failures
                    assert start['kind']==kind and info['branch_id']==key
                    if moving:
                        assert start['branch_move_offset']==entry['offset'] and info['phase']==entry['phase']
                        assert env._held == start['branch_move_keys']
                    else:
                        assert info['supported']
                    initial=row(env,obs,info,normalize(obs,ck['rms']))
                    initial.update(kind=kind,spec=spec,station=station,pin=pin,ckpin=ckpin,
                                   fallback_count=env.branch_rest_failures,held=env._held,
                                   position_error=start.get('branch_move_position_error'),
                                   velocity_error=start.get('branch_move_velocity_error'),
                                   offset=start.get('branch_move_offset'))
                    if a.script_check:
                        route=APPROACH+MIDDLE+EXIT
                        for ii,xy in entry.get('waypoints',{}).items():
                            i=int(ii);route[i]=(*xy,*route[i][2:])
                        sess=env.session
                        samples,_,log=drive(sess,sess.tracker.obs,route,entry['stage'],12,rail=RAIL,
                                            **entry.get('controller',{}))
                        passed=bool(samples[-1,6]==4 and any(f>0 and c==Ev.PostCheckpoint for f,c in log['events']))
                        initial.update(script_cp=passed,end=samples[-1].tolist(),elapsed=(samples[-1,7]-samples[0,7])/1000)
                        print(json.dumps(initial,default=str),file=stream)
                        print(a.build,label,'script',passed,flush=True)
                        continue
                    print(json.dumps(initial,default=str),file=stream)
                    for step in range(a.cap+1):
                        z=normalize(obs,ck['rms']); r=row(env,obs,info,z)
                        r['step']=step
                        if step==a.cap and not r['episode_end']:
                            r['episode_end']='probe_cap'
                        if not r['episode_end'] and r['sector']<4:
                            with torch.inference_mode():
                                aidx,_,_=model.act(z,temperature=1.,gen=gen)
                                if step==0:
                                    logits,_=model(torch.as_tensor(z)[None]); r['prob']=torch.softmax(logits,dim=-1)[0].tolist()
                            r['action']=int(aidx); r['keys']=int(env.actions[int(aidx)])
                        print(json.dumps(r,default=str),file=stream)
                        if 'action' not in r:
                            break
                        obs,_,_,_,info=env.step(int(aidx))
                    print(a.build,label,seed,r['sector'],r['episode_end'],r.get('pos'),flush=True)
                finally:
                    env.close()


if __name__=='__main__':
    main()
