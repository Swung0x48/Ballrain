"""Build a contextual approach member without altering the old member files."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain import tas
from ballrain.branch import load_branch
from ballrain.branch_assets import read_bundle, validate
from ballrain.path import ReferencePath, file_sha256
from probe_l2_s3_route import MAIN


def assemble(base, target, record, trace, entry, entrance_box, drill_only):
    # The splice has an independently reproducible physical approach and reuses the existing measured return.
    primary = 'east-primary'
    source = 'east-platform'
    main = ReferencePath.load(MAIN)
    document = json.loads((base/'branches.json').read_bytes())
    original = next(m for m in document['branches'] if m['id'] == source)
    desc = json.loads((base/source/'L02.json').read_bytes())
    old = ReferencePath.load(base/source/'L02.npz')
    with np.load(base/source/'drive.npz', allow_pickle=False) as z:
        drive = {n: z[n] for n in z.files}
    with np.load(trace, allow_pickle=False) as z:
        samples, inputs = z['samples'], z['frames']
        prefix = int(z['prefix_frame'])
    assert prefix == int(drive['prefix_frame']) and entry < prefix
    assert np.array_equal(inputs[:len(drive['frames'])], drive['frames'])
    with np.load(record/'raw.npz', allow_pickle=False) as z:
        ix = np.flatnonzero((z['frames_done']>=entry) & (z['frames_done']<prefix))
        assert z['ball_found'][ix].all()
        f = z['frames_done'][ix]
        p = z['ball_pos'][ix]
        times = z['sr_double'][ix]
    first = np.flatnonzero(samples[:,4]==1)[0]
    wood = samples[:first]
    assert np.all(wood[:,4]==0) and np.all(wood[:,6]==3)
    p = np.vstack([p, wood[:,1:4]])
    f = np.r_[f, prefix+wood[:,0]].astype(int)
    times = np.r_[times, wood[:,7]]
    orig_begin = np.flatnonzero(old.cum == desc['branch_s'][0])[0]
    np.testing.assert_allclose(old.P[orig_begin+1], samples[first,1:4], atol=1e-4, rtol=0)
    sec = desc['sector']
    splice = np.flatnonzero(main.point_sector == sec)[0]
    assert splice < orig_begin and np.linalg.norm(main.P[splice]-p[0]) < 3
    # Distance sampling (also preserve the last wood point) with recorded clocks.
    keep = [0]
    for n in range(1, len(p)):
        if times[n] > times[keep[-1]] and (np.linalg.norm(p[n]-p[keep[-1]]) > .42 or n == len(p)-1):
            keep.append(n)
    p = p[keep]
    inserted_t = main.t_ms[splice] + max(1., np.linalg.norm(main.P[splice]-p[0])*100) + times[keep]-times[0]
    tail = old.t_ms[orig_begin+1:] - old.t_ms[orig_begin+1] + inserted_t[0] + samples[first,7]-times[0]
    rebuilt = ReferencePath(np.vstack((main.P[:splice+1],p,old.P[orig_begin+1:])),
                            np.r_[main.point_sector[:splice+1], np.full(len(p),sec), old.point_sector[orig_begin+1:]],
                            meta=dict(main.meta,generator='scripts/build_l2_east_primary.py'),
                            t_ms=np.r_[main.t_ms[:splice+1], inserted_t, tail])
    shutil.copytree(base,target)
    where = target/primary
    where.mkdir()
    asset = where/'L02.npz'
    rebuilt.save(asset)
    rebuilt = ReferencePath.load(asset)
    first_paper_idx = splice+len(p)+1
    delta = first_paper_idx-orig_begin-1
    old_s = lambda v: float(rebuilt.cum[np.flatnonzero(old.cum == v)[0]+delta])
    new_lo, paper_lo, new_hi = float(rebuilt.cum[splice]), float(rebuilt.cum[first_paper_idx]), old_s(desc['branch_s'][1])
    data = dict(level=desc['level'],ball_type=0,sector=sec,main_sha256=desc['main_sha256'],
                branch_sha256=file_sha256(asset),branch_s=[new_lo,new_hi],
                main_s=[float(main.cum[splice]),desc['main_s'][1]],
                stations=[[desc['main_s'][0],paper_lo],*[[m,old_s(b)] for m,b in desc['stations']]],rules=desc['rules'])
    put(where/'L02.json',data)
    phases = [dict(local_s=[new_lo,paper_lo],when=dict(ball=0,sector=sec,
               box=entrance_box))]
    for i,src in enumerate(original['phases']):
        row=copy.deepcopy(src)
        row['local_s']=[paper_lo if i==0 else old_s(src['local_s'][0]),old_s(src['local_s'][1])]
        if 's' in row['when']:
            shift=row['local_s'][0]-src['local_s'][0]
            row['when']['s']=[x+shift for x in row['when']['s']]
        if 'from' in row:
            row['from']=[x+1 for x in row['from']]
        phases.append(row)
    merge=copy.deepcopy(original['merge'])
    merge['s']=[x+new_hi-desc['branch_s'][1] for x in merge['s']]
    document['branches'].append(dict(id=primary,entry_context='ordinary',phases=phases,merge=merge,
                                     ordinary_exclude=dict(record_sha256=str(drive['record_sha256']),ranges=drill_only)))
    put(target/'branches.json',document)
    rest = np.column_stack((f[keep]-entry,rebuilt.cum[splice+1:first_paper_idx]))
    tail_samples = [[int(n)+prefix-entry,old_s(v)] for n,v in drive['samples']]
    rest=np.vstack((rest,tail_samples))
    tape=tas.load_raw(record/'run/framelog.bin')
    np.savez_compressed(where/'drive.npz',frames=np.r_[tape[entry:prefix],drive['frames']],
                        prefix_frame=entry,samples=rest,branch_sha256=data['branch_sha256'],
                        record_sha256=drive['record_sha256'])
    assets=read_bundle(desc['level'],target)
    print(validate(assets,desc['level'],file_sha256(MAIN)))
    load_branch(main,desc['level'],target,MAIN.parent)


def put(path, data):
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('base','target','record','trace'):
        p.add_argument(k,type=Path)
    p.add_argument('--entry',type=int,required=True)
    p.add_argument('--entrance-box',type=json.loads,required=True)
    p.add_argument('--drill-only',type=json.loads,required=True)
    assemble(**vars(p.parse_args()))
