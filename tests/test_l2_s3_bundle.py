import json
from pathlib import Path

import numpy as np

from ballrain.branch import load_branch
from ballrain.branch_assets import read_bundle, validate
from ballrain.path import ReferencePath, file_sha256


ROOT = Path(__file__).resolve().parents[1] / 'benchmarks'
MAIN = ROOT / 'refpaths-v2/L02.npz'
ASSETS = ROOT / 'refpaths-alt/L02-multi-s3route'
PIN = '4e9fc293ffdaff7a6c32bed9751b41e1e67ae4bafb47f0f8a0147c737d8e8093'


def test_s3_bundle_pin_and_rest_targets():
    assert validate(read_bundle(2, ASSETS), 2, file_sha256(MAIN)) == PIN
    menu = json.loads((ASSETS / 'verified-s3-rest.json').read_text(encoding='ascii'))
    doc = json.loads((ASSETS / 'branches.json').read_text(encoding='utf8'))
    phases = next(e['phases'] for e in doc['branches'] if e['id'] == 'east-platform')
    child = ASSETS / 'east-platform'
    with np.load(child / 'drive.npz', allow_pickle=False) as z:
        for entry in menu.values():
            band, member, station = entry['spec'].split(':')
            assert band == f'{int(z["prefix_frame"])}-{int(z["prefix_frame"])}@branch'
            assert member == 'east-platform'
            assert np.any(z['samples'][:, 1] == float(station))
    rail_s = float(menu['rail-top']['spec'].rsplit(':', 1)[1])
    assert phases[2]['local_s'][0] < rail_s < phases[2]['local_s'][1]
    crown = ReferencePath.load(child / 'L02.npz').point_at(rail_s)
    assert 956 < crown[0] < 960 and crown[1] > 49.8 and abs(crown[2]+363.98) < .15


def test_s3_measured_reference_latches_each_phase_and_merges():
    main = ReferencePath.load(MAIN)
    f = load_branch(main, 2, ASSETS, MAIN.parent)
    phase = []
    entry, follower = f.by_id['east-platform']
    lo, hi = follower.lo, follower.hi
    alt = follower.alt
    ix = np.flatnonzero((alt.cum > lo) & (alt.cum <= hi))
    stone = entry['phases'][1]['local_s'][0]
    for j in ix:
        bt = 1 if alt.cum[j] < stone else 2
        _, p = f.update(alt.P[j], 3, bt)
        if not phase or p['phase'] != phase[-1]:
            phase.append(p['phase'])
    assert phase[-5:] == [0, 1, 2, 3, None]
    assert 'east-platform' in f.done
    f.reset()
    for pos in main.P[main.point_sector == 3]:
        _, p = f.update(pos, 3, 0)
        assert p['branch_id'] is None
