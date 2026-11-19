"""Generate the packaged retail reach mechanism inventory from tas-v1 captures.

    python tools/build_reach_mechanisms.py <runtime>/datasets/tas-v1

Use the first sorted HS record for each level, fixed by the generated file's
`record` field. PH rows are placements, not a promise that a body was spawned
or that the mesh for each placement was exported by this route.
"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ballrain import protocol as P  # noqa: E402
from ballrain.fastreach import fan_boxes  # noqa: E402
from ballrain.reach_mechanisms import ASSETS  # noqa: E402
from ballrain.state import GameTracker  # noqa: E402

# Unknown is intentional; the inventory is not an allowlist of traversable routes.
UNKNOWN = ('unknown',) * 3
RULES = {
    'P_Modul_01': ('physics-driven pusher; route-dependent', UNKNOWN, 'level, scout L2'),
    'P_Modul_03': ('gate/floor assembly; trigger or phase unresolved', UNKNOWN, 'level'),
    'P_Modul_08': ('hinged swing; phase/pose', UNKNOWN, 'level name'),
    'P_Modul_17': ('hinge assembly; behavior unresolved', UNKNOWN, 'level name'),
    'P_Modul_18': ('oriented fan volume; existing FastReach lift', ('slows_fall', 'lift', 'ignored'), 'level, scout L2, existing fan calibration'),
    'P_Modul_19': ('physics-driven flaps; state and approach dependent', ('unknown', 'unknown', 'observed_crossing'), 'level, scout L9'),
    'P_Modul_25': ('hinged push plate; L2 has a live near-flat pose check', ('pushable', 'blocked_when_raised', 'pushable'), 'user, wood TAS, scout and paper probe L2; transfer assumed'),
    'P_Modul_26': ('rope/sack; physics-driven assumed', UNKNOWN, 'level name'),
    'P_Modul_29': ('hinged panel bank; physics-driven', ('unknown', 'observed_crossing', 'unknown'), 'level, scout L2'),
    'P_Modul_30': ('seesaw; physics-driven, depends on angle/velocity', ('unknown', 'unknown', 'observed_crossing'), 'level, scout L2'),
    'P_Modul_34': ('sliding stone/crate assembly; motion rule unresolved', UNKNOWN, 'level name'),
    'P_Modul_37': ('hinged bridge; opening rule unresolved', UNKNOWN, 'level name'),
    'P_Modul_41': ('hinged panel; material and breakage unresolved', UNKNOWN, 'level name'),
    'P_Box': ('loose physics body', ('pushable_assumed', 'unknown', 'pushable_assumed'), 'level, user; exact thresholds unverified'),
    'P_Dome': ('obstacle placement; motion/use unresolved', UNKNOWN, 'level PH'),
    'P_Extra_Point': ('pickup trigger, not support', ('pickup',) * 3, 'state-spec'),
    'P_Extra_Life': ('pickup trigger, not support', ('pickup',) * 3, 'state-spec'),
    'PS_Levelstart': ('start marker', ('spawn',) * 3, 'state-spec'),
    'PE_Levelende': ('final goal/airship assembly', ('goal',) * 3, 'state-spec'),
}
for material in ('Wood', 'Paper', 'Stone'):
    RULES[f'P_Trafo_{material}'] = ('trigger replaces ball and interrupts control',
                                   (f'transform_{material.lower()}',) * 3, 'state-spec, PH')
    RULES[f'P_Ball_{material}'] = ('loose physics body; collision response mass-dependent', UNKNOWN, 'PH, scout')


def compile_level(dataset, n):
    rec = next(iter(sorted(dataset.glob(f'HS{n:02d}*'))))
    tracker = GameTracker(n)
    tracker.merge_level(P.parse_level_message((rec / 'level.msg').read_bytes())[0])
    for lm in P.parse_level_stream((rec / 'level.msg.inc').read_bytes())[0]:
        tracker.merge_level(lm)
    level, arrays = tracker.level, tracker.arrays
    parts, plates, placements = defaultdict(dict), [], []
    for c, ent in sorted(level.entities.items(), key=lambda item: item[1].name or ''):
        name = ent.name or ''
        m = re.match(r'(P_Modul_\d{2}|P_Trafo_Paper|P_Trafo_Stone|P_Trafo_Wood|P_Box|P_Ball_Wood|'
                     r'P_Ball_Paper|P_Ball_Stone|P_Dome|PE_Balloon|P_Extra_Point|P_Extra_Life)', name)
        if not m or c not in level.phys_by_ent:
            continue
        bb = level.mesh_world_bbox(c)
        if bb is not None:
            parts[m[1]][name] = {'bbox': np.round(bb, 5).tolist(), 'fixed': level.is_fixed(c)}
        if not re.fullmatch(r'P_Modul_25_Bridge\d*', name):
            continue
        meshes = [m for (owner, _), (_, m) in level.phys_meshes.items() if owner == c] or level.meshes_of(c)
        if not meshes:
            continue
        # Main panel only, not the mounting tab. Transform the Physicalize mesh.
        mesh = max(meshes, key=lambda m: np.ptp(m.vertices[:, 0]))
        M = np.asarray(mesh.matrix, float)
        V = np.asarray(mesh.vertices, float)
        raised = V @ M[:3, :3] + M[3, :3]
        row = {'object': name, 'type': 'P_Modul_25', 'bbox': np.round([raised.min(0), raised.max(0)], 5).tolist(),
               'raised': np.round(raised[mesh.faces], 5).tolist()}
        if n == 2 and name == 'P_Modul_25_Bridge':
            down = V @ np.array([M[1, :3], -M[0, :3], M[2, :3]]) + M[3, :3]
            row['down'] = np.round(down[mesh.faces], 5).tolist()
            row['state_probe'] = {'pivot': np.round(M[3, :3], 5).tolist(), 'min_alignment': 0.996,
                                  'down_axis': np.round(M[1, :3] / np.linalg.norm(M[1, :3]), 6).tolist(),
                                  'from_ball': 1, 'field_of_ball': 0}
        plates.append(row)
    for i in range(arrays.rows('PH')):
        sector, kind, matrix, ck = (arrays.get('PH', i, j) for j in range(4))
        placements.append({'type': kind, 'sector': sector, 'matrix': np.round(matrix, 6).tolist(),
                           'object': level.name(ck) if ck is not None else None})
    kinds = Counter(p['type'] for p in placements)
    rule_table = {k: dict(zip(('state', 'rules', 'source'), RULES[k])) for k in sorted(kinds)}
    fans = [{'object': name, 'type': 'P_Modul_18', 'matrix': np.round(mat, 6).tolist()}
            for name, mat in fan_boxes(level)]
    # Pool instances were exported at their L2 sector-5 poses. Recover only
    # the two sector-3 lift placements that have replay evidence. The measured
    # child-box offset from all nine L2 sector-5 placements is +1.1399 world-y.
    recovered = []
    for i, p in enumerate(placements):
        ph = np.asarray(p['matrix'], float)
        if n != 2 or p['type'] != 'P_Modul_18' or p['sector'] != 3 or ph[3, 0] < 1000:
            continue
        mat = ph[:, :3].copy()
        mat[3, 1] += 1.1399
        recovered.append({'ph_row': i, 'matrix': np.round(mat, 6).tolist(), 'source': 'L2 PH and scout fan witness'})
    return {'schema': 1, 'level': n, 'record': rec.name, 'rules': rule_table, 'ph_counts': dict(kinds),
            'placements': placements, 'physical_parts': parts, 'layers': {'0': 'down', '1': 'raised', '2': 'down'},
            'plates': plates, 'fans': fans, 'recovered_fans': recovered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    ASSETS.mkdir(exist_ok=True)
    for n in range(1, 14):
        output = ASSETS / f'L{n:02d}.json'
        output.write_text(json.dumps(compile_level(args.dataset, n), separators=(',', ':')) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
