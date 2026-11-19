"""Build a geometry-only gate guide from a local level-message snapshot."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain import protocol as P
from ballrain.state import GameTracker
from ballrain.routescout.flap_graph import build
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ('level-dir', 'atlas', 'output', 'cache'):
        ap.add_argument('--' + k, type=Path, required=True)
    for k in ('sector', 'level'):
        ap.add_argument('--' + k, type=int, required=True)
    ap.add_argument('--mode',choices=['layers','tubes','groove'],default='layers')
    a = ap.parse_args()
    tr = GameTracker(a.level)
    tr.merge_level(P.parse_level_message((a.level_dir / 'level.msg').read_bytes())[0])
    inc = a.level_dir / 'level.msg.inc'
    if inc.is_file():
        for msg in P.parse_level_stream(inc.read_bytes())[0]:
            tr.merge_level(msg)
    atlas = json.loads(a.atlas.read_text(encoding='utf8'))['map']
    if tr.level.static_key() != atlas['static_key']:
        raise ValueError('snapshot differs from atlas')
    doc = build(tr, atlas, a.sector, a.cache)
    if a.mode!='layers':
        from ballrain.routescout.tube_graph import build_tubes,groove
        doc=build_tubes(tr,atlas,a.sector,doc)
        if a.mode=='groove':doc=groove(tr,doc)
    doc.update(level=a.level, sector=a.sector, static_key=atlas['static_key'])
    a.output.write_text(json.dumps(doc), encoding='utf8')
