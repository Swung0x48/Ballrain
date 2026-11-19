import pickle
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ballrain import env, tas
from ballrain.branch_move import HANDOVER_TABLE, select_handover
from ballrain.branch_rules import BranchRules
from ballrain.train.remote import PROTOCOL, RemoteHub
from scripts.train_m2 import parse_rest_starts


MOVING = '10557-10557@branch-move:east-platform:1290.827804369602'
STATION = 1290.827804369602


def test_parse_preserves_ordinary_starts_and_tags_moving():
    assert parse_rest_starts('2-8,2-3@1:2:0:12,9-9@branch:10,9-9@branch:test:12') == [
        [2, 8], [2, 3, 1., 2., 0., 12], [9, 9, {'branch_s': 10.}],
        [9, 9, {'branch_s': 12., 'id': 'test'}],
    ]
    assert parse_rest_starts(MOVING) == [[10557, 10557, {'branch_move': True,
                                                       'id': 'east-platform', 'branch_s': STATION}]]


@pytest.mark.parametrize('entry', ['0-0@branch-move:foo:3', '1-2@branch-move:foo:3',
    '2@branch-move:foo:3', 'bad@branch-move:foo:3', '2-2@branch-move:3',
    '2-2@branch-move:X:3', '2-2@branch-move:foo:nan', '2-2@branch-move:foo:-1'])
def test_parse_rejects_invalid_moving_form(entry):
    with pytest.raises(SystemExit, match='recorded station'):
        parse_rest_starts(entry)


def test_sampler_does_not_snap_moving_but_snaps_ordinary():
    e = object.__new__(env.BallanceEnv)
    e.cfg = env.EnvConfig(p_rest=1., rest_starts=parse_rest_starts(MOVING))
    e.curriculum = None
    e.starts = SimpleNamespace(nearest_any=Mock(return_value=10556))
    e._start_rng = np.random.default_rng(34)
    assert e._draw_start() == 10557
    assert e._rest_next['branch_move']
    e.starts.nearest_any.assert_not_called()
    e.cfg.rest_starts = parse_rest_starts(MOVING.replace('branch-move:', 'branch:'))
    assert e._draw_start() == 10556
    assert 'branch_move' not in e._rest_next
    e.starts.nearest_any.assert_called_once_with(10557)


def test_moving_selects_exact_approved_frame_and_matching_pin():
    frames = tas.make_frames([10.] * 8, [0] * 5 + [tas.KEY_LEFT] * 3)
    handover = np.array([[6, 0, 1, 3, 100., 50., -25., 1., 0., 0.]])
    drive = dict(prefix_frame=10557, record_sha256='record', branch_sha256='line',
                 frames=frames, samples=np.array([[4, STATION-2], [6, STATION]]),
                 moving_replay_step_v1=np.int64(6), **{HANDOVER_TABLE: handover})
    selected, expected, step = select_handover(drive, 10557, STATION, 'record', 'line')
    np.testing.assert_array_equal(selected, frames[:6]); np.testing.assert_array_equal(expected, handover[0])
    assert step == 6
    with pytest.raises(ValueError, match='exact recorded station'):
        select_handover(drive, 10557, STATION+1e-7, 'record', 'line')
    with pytest.raises(ValueError, match='not approved'):
        select_handover(drive, 10557, STATION-2, 'record', 'line')
    with pytest.raises(ValueError, match='does not match'):
        select_handover(drive, 10556, STATION, 'record', 'line')


def preparation(monkeypatch):
    from ballrain import branch_move
    z = object.__new__(env.BallanceEnv)
    z.cfg = env.EnvConfig(start_check=True)
    z.branch_rest_failures = 0
    z._handover_check = Mock(return_value=None)
    z._rules_for_rest = Mock(return_value=BranchRules())
    sess = Mock(); fresh = Mock()
    def launch(s):
        return dict(session=s, start={'kind':'tas','frame':10557},
                    recorder=SimpleNamespace(meta=SimpleNamespace(start_state={})),run_dir=Path('.'))
    orig = launch(sess); retry = launch(fresh)
    z._launch = Mock(side_effect=[orig,retry])
    replay = Mock(side_effect=RuntimeError('wrong velocity'))
    monkeypatch.setattr(branch_move, 'replay', replay)
    return z, orig, retry, replay, parse_rest_starts(MOVING)[0][2]


def test_failed_moving_handover_is_counted_and_explicitly_marked(monkeypatch):
    z, orig, retry, replay, target = preparation(monkeypatch)
    assert z._prepare('slot', 10557, rest=target) is retry
    orig['session'].close.assert_called_once_with(kill=True)
    replay.assert_called_once_with(z, orig, 10557, STATION, 'east-platform')
    st=retry['start']
    assert st['kind']=='branch_moving_fallback' and st['frame']==10557
    assert st['branch_move_fallback'] and st['branch_move_station']==STATION
    assert st['branch_rest_error']=='wrong velocity' and st['branch_rest_failures']==z.branch_rest_failures==1
    assert retry['recorder'].meta.start_state['kind']==st['kind']


def test_existing_error_fallback_policy_is_respected(monkeypatch):
    z, _, _, _, target = preparation(monkeypatch)
    z._rules_for_rest.return_value = BranchRules(rest_fallback='error')
    with pytest.raises(RuntimeError, match='wrong velocity'):
        z._prepare('slot', 10557, rest=target)
    assert z.branch_rest_failures==1 and z._launch.call_count==1


def test_old_worker_rejects_moving_config_before_rollout(monkeypatch):
    cfg = env.EnvConfig(rest_starts=parse_rest_starts(MOVING))
    wire = pickle.dumps(cfg)
    assert pickle.loads(wire).rest_starts == cfg.rest_starts
    hub=object.__new__(RemoteHub); hub.template={'env':cfg}
    hub.stats={'disconnects':0}; hub.log,hub._cond=Mock(),threading.Condition()
    conn=Mock();conn.recv.return_value={'type':'hello','protocol':PROTOCOL,'multi_branch_v2':True}
    hub._serve(conn)
    conn.send.assert_called_once_with({'type':'error','error':'moving branch starts require branch_move_v1'})
    monkeypatch.delattr(env, '_restore_branch_move_env_v1')
    with pytest.raises(AttributeError, match='_restore_branch_move_env_v1'):
        pickle.loads(wire)
