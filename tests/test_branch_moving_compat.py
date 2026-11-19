"""Teacher and snapshot compatibility for additive moving-handover metadata."""
import copy
import hashlib
import io
import json

import numpy as np
import pytest

from ballrain.branch_assets import contextual_append_end, read_bundle, validate
from ballrain.path import REFPATH_DIR, RouteError, file_sha256
from ballrain.train.anchor import snapshot_branch_ignore

ROOT = REFPATH_DIR.parent
MAIN = file_sha256(ROOT/'refpaths-v2/L02.npz')


def pin(b):
    return validate(b,2,MAIN)


def update(child, arrays):
    blob=io.BytesIO(); np.savez_compressed(blob,**arrays)
    child['drive_npz']=blob.getvalue()
    child['sha256']['drive_npz']=hashlib.sha256(child['drive_npz']).hexdigest()


@pytest.fixture
def bundles():
    teacher=read_bundle(2,ROOT/'refpaths-alt/L02-multi-s3route')
    old=read_bundle(2,ROOT/'refpaths-alt/L02-multi-s3-primary')
    new=copy.deepcopy(old)
    with np.load(io.BytesIO(old['members'][1]['drive_npz']),allow_pickle=False) as a:
        arrays={k:a[k] for k in a.files}
    sample=arrays['samples'][(arrays['samples'][:,1]>1384)&(arrays['samples'][:,1]<1385)][0]
    arrays['moving_handover_v1']=np.array([[sample[0],2,2,3,958.,49.91,-363.98,-6.,0.,0.]])
    arrays['moving_replay_step_v1']=np.asarray(6)
    update(new['members'][1],arrays)
    return teacher,old,new,arrays


def test_old_explicit_teacher_and_live_snapshot_both_compatible(bundles):
    teacher,old,new,_=bundles
    assert contextual_append_end(old,new,2,MAIN,pin(old),pin(new)) is None
    assert contextual_append_end(teacher,new,2,MAIN,pin(teacher),pin(new)) == pytest.approx(1157.924316740663)
    f={'branch_sha256':pin(new),'path_sha256':MAIN}
    for source in (teacher,old):
        snapshot={'features':{'branch_sha256':pin(source),'path_sha256':MAIN}}
        assert snapshot_branch_ignore(snapshot,{pin(source):source},new,2,f,True)==frozenset({'branch_sha256'})
        assert snapshot['branch_assets'] is source
        assert snapshot_branch_ignore(snapshot,{},new,2,f,True)==frozenset({'branch_sha256'})
    with pytest.raises(RouteError):
        snapshot_branch_ignore({'features':{'branch_sha256':pin(old)}},{},new,2,f,True)
    assert not snapshot_branch_ignore({'features':{'branch_sha256':pin(old)}},{},new,2,f,False)


@pytest.mark.parametrize('mutation',['keys','clock','station','prefix','dtype','new key','phase','duplicate','rewrite'])
def test_unsafe_drive_changes_refused(bundles,mutation):
    _,old,new,arrays=bundles
    if mutation=='keys': arrays['frames']['keys'][0]^=1
    if mutation=='clock': arrays['frames']['delta'][0]+=1
    if mutation=='station': arrays['samples'][0,1]+=.1
    if mutation=='prefix': arrays['prefix_frame']=arrays['prefix_frame']+1
    if mutation=='dtype': arrays['samples']=arrays['samples'].astype(np.float32)
    if mutation=='new key': arrays['not_moving']=np.zeros(1)
    if mutation=='phase': arrays['moving_handover_v1'][0,1]=0
    if mutation=='duplicate': arrays['moving_handover_v1']=np.repeat(arrays['moving_handover_v1'],2,axis=0)
    if mutation=='rewrite': old=copy.deepcopy(new); arrays['moving_handover_v1'][0,4]+=1
    update(new['members'][1],arrays)
    with pytest.raises(RouteError,match='moving-only'):
        contextual_append_end(old,new,2,MAIN,pin(old),pin(new))


def test_scope_changes_and_unvalidated_pins_refused(bundles):
    _,old,new,_=bundles
    with pytest.raises(RouteError):
        contextual_append_end(old,new,2,MAIN,pin(old),'fake')
    doc=json.loads(new['manifest']);doc['branches'][1]['phases'][0]['when']['box'][0][0]-=.1
    new['manifest']=json.dumps(doc).encode()
    with pytest.raises(RouteError):
        contextual_append_end(old,new,2,MAIN,pin(old),pin(new))
