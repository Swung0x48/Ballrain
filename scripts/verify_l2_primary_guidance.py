"""Offline headless verification of the primary entrance and legacy drills."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain import tas
from ballrain.branch_assets import read_bundle, validate
from ballrain.env import BallanceEnv, EnvConfig, FeatureConfig, RewardConfig
from ballrain.protocol import Ev
from ballrain.train.ppo import ActorCritic, normalize
from probe_l2_s3_route import BUILDS, MAIN, RECORD, engine
from train_m2 import parse_rest_starts


BANDS = (10357, 10457, 10557)
PRIMARY = 'east-primary'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','checkpoint','bundle','out'):
        p.add_argument(name,type=Path)
    p.add_argument('--build',required=True,choices=BUILDS)
    p.add_argument('--copy',required=True)
    p.add_argument('--mode',required=True,choices=('script','actor','parity'))
    p.add_argument('--trace',type=Path,help='verified full trace; required in script mode')
    p.add_argument('--old-screen',action='store_true')
    p.add_argument('--full-rest-list',action='store_true')
    p.add_argument('--rehearse-only',action='store_true')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR',str(a.out/'numba'))
    torch.set_num_threads(1)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    original=ck['env_config']; args=ck['args']; fx=FeatureConfig(**original['features']); fx.path_dir=str(MAIN.parent)
    base=ck['branch_assets']; prior=validate(base,2,fx.path_sha256); assert prior==fx.branch_sha256
    bundle=read_bundle(2,a.bundle); pin=validate(bundle,2,fx.path_sha256)
    cfg=dict(original,features=fx,reward=RewardConfig(**original['reward']),start_record=str(a.runtime/RECORD),
             headless=dict(original['headless'],binary=engine(a.runtime,a.build).binary,
                           root=engine(a.runtime,a.build).root,socket='tcp'),
             copy=a.copy,start_frame=None,start_rest=None,curriculum=None,p_rest=0.,p_rehearse=0.,warm_pool=False,
             rest_starts=None,rehearse_starts=None,rehearse_always=None,seed=1)
    model=ActorCritic(ck['n_obs'],ck['n_act'],tuple(map(int,args['hidden'].split(','))),
                      args['activation'],args['actor_heads']); model.load_state_dict(ck['model']); model.eval()
    print('checkpoint update',ck['counters']['update'],'n_obs',ck['n_obs'],'verified pins',prior,pin,flush=True)

    def make(assets, sha, folder, **kwargs):
        f=FeatureConfig(**dict(fx.to_dict(),branch_sha256=sha))
        return BallanceEnv(EnvConfig(**dict(cfg,**kwargs,features=f,branch_assets=assets,out_root=folder)))

    if a.mode=='script':
        with np.load(a.bundle/PRIMARY/'drive.npz',allow_pickle=False) as drive:
            offset_ref=int(drive['prefix_frame']); tape=drive['frames']
        # The bundle drive ends at the merge. Use the same verified input stream past merge, to PostCheckpoint.
        if not a.trace:
            p.error('--trace required in script mode')
        with np.load(a.trace, allow_pickle=False) as z:
            raw=tas.load_raw(a.runtime/RECORD/'run/framelog.bin')
            tape_all=np.r_[raw[offset_ref:int(z['prefix_frame'])], z['frames']]
        assert np.array_equal(tape, tape_all[:len(tape)])
        for k in (offset_ref,*BANDS,None):
            with tempfile.TemporaryDirectory(dir=a.out,prefix='engine-') as tmp:
                kw = dict(start_frame=k) if k is not None else dict(curriculum=dict(recent=1.,band=200))
                env=make(bundle,pin,tmp,**kw)
                if k is None:
                    env.curriculum.frontier = 10357
                    assert not ((env.curriculum.valid>=10825)&(env.curriculum.valid<=11278)).any()
                try:
                    obs,info=env.reset(seed=1); sess=env.session; t=sess.tracker
                    got=info['frames_done']; assert got>=offset_ref and got<=BANDS[-1] and not info['start']['handover_mismatches']
                    assert info['branch_id']==PRIMARY and info['phase']==0
                    seq=[]; last=None
                    rail_contacts=0; plate_contacts=0
                    with (a.out/f'{a.build}-{k or "sampled"}.jsonl').open('w',encoding='utf8') as out:
                        for n in range(got-offset_ref,len(tape_all)+1,6):
                            # Keep the actor's actual feature extractor, not a mock projection.
                            if t.control_valid():
                                obs,fi=env.fx.compute(t); pr=fi.get('path',{}); state=(pr.get('branch_id'),pr.get('phase'))
                                if state!=last:
                                    seq.append(state); last=state
                                d=dict(frame=sess.frames_done,sector=t.sector,branch=state[0],phase=state[1],
                                       arc=pr.get('local_s'),pos=fi['pos'].tolist(),ball=t.ball_type(),
                                       **features(env,obs))
                                names = [t.level.name(int(c['partner'])) for c in sess.obs.fric]
                                x,y,z = fi['pos']
                                if t.ball_type()==2 and 940<x<960 and y>49.7 and abs(z+363.98)<.2:
                                    rail_contacts += any(name.startswith('A02_Gel') for name in names)
                                plate_contacts += t.ball_type()==2 and 'P_Modul_25_Bridge' in names
                                print(json.dumps(d),file=out)
                            if sess.obs.has_event(Ev.PostCheckpoint) and t.sector==4:
                                break
                            left=tape_all[n:n+6]
                            if not len(left): break
                            sess.act(left,stop=(Ev.BallOff,Ev.PostCheckpoint))
                            assert not sess.ended and not t.flags.ball_off
                        assert t.sector==4 and sess.obs.has_event(Ev.PostCheckpoint), (k,t.sector)
                        assert [ph for ident,ph in seq if ident==PRIMARY]==list(range(5)),seq
                        assert rail_contacts and plate_contacts, (k,rail_contacts,plate_contacts)
                        print(json.dumps(dict(CP3=True,frame=sess.frames_done,sr_ms=t.sr_ms,transitions=seq,
                                              rail_samples=rail_contacts,stone_plate_samples=plate_contacts)),file=out)
                    print(a.build,k,'actual',got,'CP3',sess.frames_done,seq,flush=True)
                finally: env.close()
        return

    if a.mode=='actor':
        for k in BANDS:
            for seed in range(1,5):
                assets,sha=(base,prior) if a.old_screen else (bundle,pin)
                with tempfile.TemporaryDirectory(dir=a.out,prefix='engine-') as tmp, (a.out/f'{a.build}-{k}-{seed}.jsonl').open('w',encoding='utf8') as out:
                    env=make(assets,sha,tmp,start_frame=k)
                    try:
                        obs,info=env.reset(seed=1); gen=torch.Generator().manual_seed(seed)
                        assert not info['start']['handover_mismatches']
                        if not a.old_screen: assert info['branch_id']==PRIMARY and info['phase']==0
                        assert obs.shape==(ck['n_obs'],) and len(env.actions)==ck['n_act']
                        for step in range(1501):
                            t=env.session.tracker if env.session else None; b=t.obs.ball if t else None
                            z=normalize(obs,ck['rms']); row=dict(step=step,frame=info['frames_done'],sector=info['sector'],
                                branch=info.get('branch_id'),phase=info.get('phase'),arc=info.get('local_s'),
                                supported=info['supported'],end=info.get('episode_end'),**features(env,obs))
                            if b is not None and b['found']:
                                row.update(pos=b['pos'].tolist(),vel=(b['speed']+b['speed_change']).tolist(),ball=int(b['type']))
                            if step==1500 and not row['end']:row['end']='probe_cap'
                            if row['sector']<4 and not row['end']:
                                action,_,_=model.act(z,temperature=1.,gen=gen);row['act']=int(action)
                            print(json.dumps(row),file=out)
                            if 'act' not in row: break
                            obs,_,_,_,info=env.step(row['act'])
                        print(a.build,k,seed,row['branch'],row['phase'],row['arc'],row.get('pos'),row['end'],flush=True)
                    finally:env.close()
        return

    cases=retention(original,a.full_rest_list)
    if a.rehearse_only:
        cases=[case for case in cases if case[1]=='rehearse']
    for label,style,target in cases:
        records=[]
        for assets,sha in [(base,prior),(bundle,pin)]:
            with tempfile.TemporaryDirectory(dir=a.out,prefix='engine-') as tmp:
                kw=(dict(p_rehearse=1.,rehearse_always=[target]) if style=='rehearse'
                    else dict(p_rest=1.,rest_starts=[target]))
                env=make(assets,sha,tmp,**kw)
                try:
                    obs,info=env.reset(seed=1); start=info['start']; assert not start['handover_mismatches']
                    assert not start.get('branch_rest_error'), start.get('branch_rest_error')
                    assert env.fx.branch_entry_context=='drill' and info['branch_id']!=PRIMARY
                    start_branch=info['branch_id']; start_phase=info['phase']; start_arc=info.get('local_s')
                    start_ball=env.session.tracker.ball_type(); h=hashlib.sha256(); active=env.session
                    origin=start['kind']; last=0; gen=torch.Generator().manual_seed(1)
                    for n in range(129):
                        digest_state(h,obs,info,env)
                        last=n
                        if n==128 or info.get('episode_end'): break
                        control,_,_=model.act(normalize(obs,ck['rms']),temperature=1.,gen=gen)
                        obs,reward,te,tr,info=env.step(int(control)); h.update(struct.pack('<id??',int(control),reward,te,tr))
                    h.update(active.frame_log().tobytes())
                    records.append(dict(sha=h.hexdigest(),steps=last,kind=origin,k=start['frame'],branch=start_branch,
                                        phase=start_phase,arc=start_arc,ball=start_ball,end=info.get('episode_end')))
                finally:env.close()
        assert records[0]==records[1],(label,records)
        with (a.out/f'{a.build}-parity.jsonl').open('a',encoding='utf8') as f:
            print(json.dumps(dict(label=label,style=style,target=target,**records[0])),file=f)
        print(a.build,label,records[0],flush=True)


def features(env,obs):
    sl=env.fx.slices(); c=env.fx.cfg
    return {k:(obs[sl[k]]*v).tolist() for k,v in [('waypoints',c.pos_scale),('far_waypoints',c.pos_scale),('ref_speed',c.vel_scale)]}


def digest_state(h,obs,info,env):
    t=env.session.tracker if env.session else None; b=t.obs.ball if t else None
    h.update(obs.tobytes())
    h.update(json.dumps({k:info.get(k) for k in ('frames_done','sr_ms','sector','events','supported','branch_id','phase','local_s','episode_end')},
                        default=str,sort_keys=True).encode())
    if b is not None:h.update(b.tobytes())


def retention(original,full):
    starts=original['rest_starts']
    cases=[('back-to-wood','rest',starts[10]),('junction','rest',parse_rest_starts('14730-14730@933.8:-57:0:1200')[0]),
           ('1462','rest',parse_rest_starts('14535-14535@branch:south-loop:1460.589978')[0]),
           ('south-loop','rest',starts[0]),('stone-arrival','rest',parse_rest_starts('13900-13900@branch:stone-arrival:1447')[0]),
           ('11080-brake','rest',[11080,11080]),('11080-target','rest',[11080,11080,921.,-348.6,0.])]
    if full:
        used=[t for _,_,t in cases]
        for i,t in enumerate(starts):
            if t not in used:
                used.append(t);cases.append((f'rest-{i}','rest',t))
    else:
        cases.extend((f'east-{i}','rest',starts[i]) for i in (15,17,19,20,21))
    cases.extend((f'dodge-{lo}-{hi}','rehearse',[lo,hi]) for lo,hi in dict.fromkeys(map(tuple,original['rehearse_always']))
                 if 10825<=lo<=11278)
    cases.append(('same-frame-earlier-rehearsal','rehearse',[10357,10357]))
    return cases


if __name__=='__main__':main()
