"""Run in fresh interpreters against the baseline and the scout checkout. Use --expected to assert bit parity."""
from pathlib import Path
import sys, tempfile, json, hashlib, numpy as np
import argparse
ap = argparse.ArgumentParser(description=__doc__)
for n in ('repo', 'root', 'binary', 'output'):
    ap.add_argument('--' + n, type=Path, required=True)
ap.add_argument('--copy', required=True)
ap.add_argument('--expected', type=Path)
args = ap.parse_args()
repo = args.repo
sys.path.insert(0, str(repo))
from ballrain.reach_features import ReachBlock, reach_version
from ballrain.client import SessionConfig, AgentSession, DEFAULT_DELTA
from ballrain.headless import HeadlessLauncher
from ballrain import tas
from ballrain.protocol import Ev
out = args.output.parent
out.mkdir(parents=True, exist_ok=True)
rows = []
rb = ReachBlock(cache_dir=out / 'fields')
digest = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
with tempfile.TemporaryDirectory(dir=out) as tmp:
    for l in (2, 13):
        s = AgentSession(SessionConfig(l, Path(tmp) / str(l), copy=args.copy), HeadlessLauncher(binary=str(args.binary.resolve()), root=str(args.root.resolve()), socket='tcp'))
        try:
            s.start()
            s.run_until(0, [Ev.BallNavActive], limit=3000) if not s.tracker.flags.ball_nav else None
            for keys in (0, 8, 9, 1, 0):
                s.act(tas.make_frames([DEFAULT_DELTA], [keys]), repeat=24)
                b = s.obs.ball
                p = b['pos']
                v = b['speed'] + b['speed_change']
                cam = s.obs.cams['cams'][0]['matrix']
                q = int(round(np.arctan2(cam[2, 0], cam[2, 2]) / (np.pi / 2))) % 4
                scaled, raw, key = rb.compute(s.tracker, p, v, q, s.tracker.ball_type())
                rows.append([l, s.frames_done, digest(p), digest(v), q, key, digest(scaled), digest(raw)])
        finally:
            s.close(kill=True)
r = dict(version=reach_version(), rows=rows)
args.output.write_text(json.dumps(r))
print(repo.name, reach_version(), len(rows))
if args.expected and r != json.loads(args.expected.read_text()):
    raise AssertionError('f7 pin, inputs or feature bytes differed')
