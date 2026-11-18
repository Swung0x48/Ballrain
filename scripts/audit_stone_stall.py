"""Replay recorded controller observations through the training odometer and stall rule."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain.branch import load_branch
from ballrain.path import REFPATH_DIR, ReferencePath
from ballrain.train.worker import stall_age


def audit(run):
    rows = [json.loads(line) for line in (run / 'controller.jsonl').read_text().splitlines()
            if line.startswith('{')]
    main = ReferencePath.load(REFPATH_DIR / 'L02.npz')
    route = load_branch(main, 2, 'benchmarks/refpaths-alt/L02-stone', None)
    ms, bs = route.stations
    print('arc budgets, main vs branch:', ms[-2]-ms[1], bs[-2]-bs[1],
          'between transforms:', ms[-3]-ms[2], bs[-3]-bs[2], 'credit/m:', np.diff(ms)/np.diff(bs))
    assert rows[-1]['sector'] == 5
    first, last = rows[0], rows[-1]
    dt = (last['sr'] - first['sr']) / (last['frame'] - first['frame'])
    win, hard = int(4000/dt), int(15000/dt)
    ep = SimpleNamespace(stall_branch=False, mark_frame=0, move_frame=0, mark_s=-np.inf)
    age_max = [0, 0]
    stalls = 0
    n_slow_pauses = 0
    prev = None
    for row in rows:
        frame = row['frame'] - first['frame']
        # Check missing frames conservatively: no new metres and no speed pause.
        if prev is not None:
            for ef in range(prev + 2, frame, 2):
                age = stall_age(ep, ef, None, ep.stall_branch, None, 5, 3, hard, win)
                age_max[ep.stall_branch] = max(age_max[ep.stall_branch], age)
                stalls += age >= win
        _, pr = route.update(row['pos'], row['sector'], row['bt'])
        br = pr['branch']
        age = stall_age(ep, frame, pr['local_s'] if br else pr['s'], br, row['speed'], 5, 3, hard, win)
        age_max[br] = max(age_max[br], age)
        n_slow_pauses += br and 1 < row['speed'] <= 3
        stalls += age >= win
        prev = frame
    print(dict(rows=len(rows), last_frame=last['frame'], frame_ms=dt, slow_pause_samples=n_slow_pauses,
               max_age_ms=(np.array(age_max)*dt).tolist(), stalls=stalls))
    assert not stalls


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
