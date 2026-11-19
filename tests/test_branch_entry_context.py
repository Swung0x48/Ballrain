"""Generic contextual entrances and append-only teacher compatibility."""
import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ballrain.branch import load_branch
from ballrain.branch_assets import contextual_append_end, ordinary_start_frames, install, read_bundle, uses_entry_context, validate
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.multi_branch import manifest
from ballrain.path import REFPATH_DIR, ReferencePath, RouteError, file_sha256
from ballrain.train.remote import PROTOCOL, RemoteHub


MAIN = REFPATH_DIR.parent / 'refpaths-v2'
ALT = REFPATH_DIR.parent / 'refpaths-alt/L02-multi'
MAIN_SHA = file_sha256(MAIN/'L02.npz')


@pytest.fixture
def appended():
    before = read_bundle(2, ALT)
    after = copy.deepcopy(before)
    doc = json.loads(before['manifest'])
    member = copy.deepcopy(doc['branches'][1])
    member.update(id='test-ordinary', entry_context='ordinary')
    # Scope test: deliberately use a different first material to isolate the two entrances geometrically.
    member['phases'][0]['when']['ball'] = 0
    doc['branches'].append(member)
    after['members'].append(before['members'][1])
    after['manifest'] = json.dumps(doc).encode()
    return before, after


def test_context_checks_new_entrance_only(appended):
    before, bundle = appended
    assert not uses_entry_context(before) and uses_entry_context(bundle)
    pin = validate(bundle, 2, MAIN_SHA)
    root = install(bundle, 2, MAIN_SHA, pin)
    f = load_branch(ReferencePath.load(MAIN/'L02.npz'), 2, root.name, MAIN)
    e, child = f.by_id['test-ordinary']
    p = child.alt.point_at(child.lo + 2)
    assert f.update(p, 3, 0)[1]['branch_id'] == 'test-ordinary'
    f.entry_context = 'drill'
    f.reset()
    assert f.update(p, 3, 0)[1]['branch_id'] is None
    assert f.update(p, 3, 1)[1]['branch_id'] == 'east-platform'
    # A deliberately selected branch is not subject to its natural entrance predicate.
    f.reset()
    f.restore('test-ordinary', 0, child.lo + 2)
    assert f.update(p, 3, 0)[1]['branch_id'] == 'test-ordinary'
    e['entry_context'] = 'anything'
    with pytest.raises(RouteError):
        manifest(json.dumps({'version':2,'branches':[e]}))


def test_real_selector_distinguishes_the_same_frame_by_source():
    env = object.__new__(BallanceEnv)
    env.cfg = EnvConfig(start_frame=10357)
    env.starts = SimpleNamespace(nearest_valid=lambda n:n)
    env.curriculum = None
    env._start_rng = np.random.default_rng(8)
    assert env._draw_start()==10357 and env._entry_context_next=='ordinary'
    env.cfg.p_rehearse = 1.0
    env.cfg.rehearse_always = [[10357,10357]]
    assert env._draw_start()==10357 and env._entry_context_next=='drill'
    env.cfg.p_rehearse = 0.0
    env.cfg.p_rest = 1.0
    env.cfg.rest_starts = [[10357,10357]]
    assert env._draw_start()==10357 and env._entry_context_next=='drill' and env._rest_next
    env.cfg.p_rest = 0.0
    assert env._draw_start()==10357 and env._entry_context_next=='ordinary'


def test_teacher_exception_is_append_only_and_validates_pins(appended):
    old, new = appended
    op, npin = (validate(b, 2, MAIN_SHA) for b in appended)
    hi = float(json.loads(new['members'][-1]['desc_json'])['main_s'][1])
    assert contextual_append_end(old, new, 2, MAIN_SHA, op, npin)==hi
    with pytest.raises(RouteError,match='append-only'):
        contextual_append_end(old, new, 2, MAIN_SHA, op, 'fake')
    bad = copy.deepcopy(new)
    doc = json.loads(new['manifest']); doc['branches'][0]['phases'][0]['when']['box'][0][0] -= 1.
    bad['manifest'] = json.dumps(doc).encode()
    with pytest.raises(RouteError,match='append-only'):
        contextual_append_end(old, bad, 2, MAIN_SHA, op, validate(bad,2,MAIN_SHA))


def test_old_python_bridge_rejects_contextual_bundle(appended):
    hub = object.__new__(RemoteHub)
    hub.template = {'env':EnvConfig(branch_assets=appended[1])}
    hub.stats = {'disconnects':0};hub.log=Mock();hub._cond=threading.Condition()
    conn=Mock();conn.recv.return_value=dict(type='hello',protocol=PROTOCOL,multi_branch_v2=True,
                                           branch_assets_version=1,branch_rules_version=1)
    hub._serve(conn)
    conn.send.assert_called_once_with(dict(type='error',error='contextual branches require branch_entry_context_v1'))


def test_ordinary_prefix_filter_leaves_drill_states_and_requires_matching_record(appended):
    _, bundle = appended
    data = json.loads(bundle['manifest'])
    sha = 'a'*64
    item = data['branches'][-1]
    item['ordinary_exclude'] = dict(record_sha256=sha, ranges=[[3, 6]])
    bundle['manifest'] = json.dumps(data).encode()
    starts = SimpleNamespace(valid=np.arange(1,8), meta={'record_sha256':sha})
    np.testing.assert_array_equal(ordinary_start_frames(starts, bundle), [1,2,6,7])
    np.testing.assert_array_equal(starts.valid, np.arange(1,8))
    with pytest.raises(RouteError, match='different pinned TAS record'):
        ordinary_start_frames(SimpleNamespace(valid=starts.valid,meta={}), bundle)
    item['ordinary_exclude']['ranges'] = [[9,1]]
    with pytest.raises(RouteError,match='ordinary prefix exclusion'):
        manifest(json.dumps(data))
