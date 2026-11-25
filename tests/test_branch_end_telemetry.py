from types import SimpleNamespace as NS

import pytest

from ballrain.branch_rules import DEFAULT_RULES
from ballrain.env import BallanceEnv
from ballrain.train.worker import _Episode
from scripts.train_m2 import EP_COLS, Logger


def test_branch_end_uses_latest_in_tolerance_sample_not_terminal_projection():
    env = NS(reward=NS(s=8.0), action_space=NS(n=1), branch_rules=DEFAULT_RULES)
    branch = dict(branch=True, branch_id='test-route')
    ep = _Episode(dict(branch, branch_progress_sample=8.0), env, 0, None)
    ep.step(0, 0.0, dict(branch, branch_progress_sample=9.0), None)
    ep.step(0, 0.0, dict(branch, branch_progress_sample=8.5), None)
    info = dict(branch, episode_end='BallOff', ball_pos=(1.0, -20.0, 3.0))
    env.reward.s = 99.0
    ep.step(0, 0.0, info, None)
    result = ep.finish(info, 'BallOff', env, 10)
    assert result['s_end'] == 8.5
    assert tuple(result[k] for k in ('end_x', 'end_y', 'end_z')) == info['ball_pos']
    assert env.reward.s == 99.0


@pytest.mark.parametrize('switch_to, expected', [(False, 99.0), (True, None)])
def test_switch_discards_previous_member_sample(switch_to, expected):
    env = NS(reward=NS(s=99.0), action_space=NS(n=1), branch_rules=DEFAULT_RULES)
    ep = _Episode(dict(branch=True, branch_id='old', branch_progress_sample=8.0), env, 0, None)
    info = dict(branch=switch_to, branch_id='new' if switch_to else None)
    ep.step(0, 0.0, info, None)
    result = ep.finish(info, 'stall', env, 10)
    assert result['s_end'] == expected
    assert all(result[k] is None for k in ('end_x', 'end_y', 'end_z'))


@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('off,dist,expected', [(False, 2.0, 15.0), (False, 2.01, None), (True, 0.0, None)])
def test_info_qualifies_branch_samples(multi, off, dist, expected):
    tracker = NS(flags=NS(ball_off=off, ball_nav=True, finished=False, paused=False), sr_ms=0,
                 points=0, lives=1, sector=1, control_valid=lambda: not off, followers=lambda: 0)
    rules = NS(rest_position_tolerance=2.0)
    follower = NS(rules=rules, stations=([10.0, 20.0], [100.0, 200.0]))
    route = dict(branch=True, dist=dist, local_s=150.0, s=99.0, ref_elapsed_ms=0, ref_remaining_ms=0)
    if multi:
        route.update(branch_id='member', phase=0)
    branch = NS(by_id={'member': (None, follower)}) if multi else follower
    env = NS(session=NS(tracker=tracker, frames_done=0), _ep_pause_frames=0, reward=NS(held=False),
             recorder=None, _start={}, fx=NS(branch=branch), branch_rules=rules)
    info = BallanceEnv._info(env, {'path': route, 'pos': [1.0, 2.0, 3.0]}, [])
    assert info.get('branch_progress_sample') == expected
    assert info['ball_pos'] == (1.0, 2.0, 3.0)


def test_episode_csv_appends_columns_without_rewriting_old_header(tmp_path):
    import csv

    assert EP_COLS.index('end_x') + 2 == EP_COLS.index('end_z')  # end_x / end_y / end_z stay together
    fields = EP_COLS[-3:]  # the newest columns (appended after an existing file's header)
    old = tmp_path / 'episodes.csv'
    header = ','.join(EP_COLS[:-3]) + '\n'
    old.write_text(header)
    # Exercise the writer without starting TensorBoard.
    log = Logger.__new__(Logger)
    log.dir, log.files, log.writers, log.cols = tmp_path, {}, {}, {'episodes': EP_COLS}
    log.row('episodes', dict(end='BallOff', **dict(zip(fields, (1, 2, 3)))))
    log.files['episodes'].close()
    assert old.read_text() == header
    with (tmp_path / 'episodes.1.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == EP_COLS
        row = next(reader)
        assert [row[k] for k in fields] == ['1', '2', '3']

