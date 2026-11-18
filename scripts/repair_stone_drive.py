"""Re-sign a measured drive after a timing-only rebuild (never after geometry changes)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain.path import file_sha256


def repair(old_dir, new_dir):
    old, new = (d / 'L02.npz' for d in (old_dir, new_dir))
    with np.load(old) as a, np.load(new) as b:
        for key in ('points', 'sector'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f'{key} changed: re-probe instead')
    with np.load(old_dir / 'drive.npz') as source:
        data = dict(source)
    if str(data['branch_sha256']) != file_sha256(old):
        raise ValueError('original drive is not paired')
    data['branch_sha256'] = file_sha256(new)
    np.savez_compressed(new_dir / 'drive.npz', **data)


if __name__ == '__main__':
    repair(*map(Path, sys.argv[1:]))
