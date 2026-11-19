"""Add an allowlist of exact moving handovers from a verified runtime trace."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain.branch import load_branch
from ballrain.branch_assets import read_bundle, validate
from ballrain.branch_move import HANDOVER_TABLE
from ballrain.path import ReferencePath, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('trace', type=Path)
    p.add_argument('menus', nargs='+', type=Path)
    a = p.parse_args()
    route = Path(__file__).resolve().parents[1] / 'benchmarks/refpaths-v2/L02.npz'
    main = ReferencePath.load(route)
    follower = load_branch(main, 2, a.bundle, route.parent)
    member = 'east-platform'
    dst = a.bundle/member/'drive.npz'
    with np.load(dst, allow_pickle=False) as z:
        data = {n: z[n] for n in z.files}
    with np.load(a.trace, allow_pickle=False) as z:
        assert int(z['prefix_frame']) == int(data['prefix_frame'])
        assert np.array_equal(z['frames'][:len(data['frames'])], data['frames'])
        rows = z['samples']
    projections = {}
    for row in rows:
        _, projection = follower.update(row[1:4], int(row[6]), int(row[4]))
        projections[int(row[0])] = projection
    allowlist = {}
    for file in a.menus:
        for spec in json.loads(file.read_text(encoding='utf8')).values():
            if 'velocity' not in spec:
                continue
            prefix, ident, s = spec['spec'].split(':')
            k = int(data['prefix_frame'])
            assert prefix == f'{k}-{k}@branch' and ident == member
            s = float(s)
            f, = data['samples'][data['samples'][:,1] == s, 0]
            assert f == spec['offset'] and f > 0 and int(f)==f
            row, = rows[rows[:,0] == f]
            projection = projections[int(f)]
            assert projection['branch_id'] == member and projection['phase'] == spec['phase']
            assert row[4]==spec['ball'] and row[6]==3
            np.testing.assert_array_equal(row[1:4], spec['position'])
            np.testing.assert_array_equal(row[8:11], spec['velocity'])
            assert int(f) not in allowlist
            allowlist[int(f)] = [f, projection['phase'], row[4], row[6], *row[1:4], *row[8:11]]
    if not allowlist:
        raise RuntimeError('no moving targets')
    temp = dst.with_name('drive.pending.npz')
    np.savez_compressed(temp, **dict(data, **{HANDOVER_TABLE: np.asarray([allowlist[k] for k in sorted(allowlist)]),
                                            'moving_replay_step_v1': 6}))
    temp.replace(dst)
    print(validate(read_bundle(2, a.bundle), 2, file_sha256(route)))


if __name__ == '__main__':
    main()
