"""State-conditioned mechanism evidence. No cross-build or out-of-sample promotion."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MechanismEntry:
    static_key: str
    owner: str
    sector: int
    ball: int
    # Exact reproducible prefix pin (captures hidden state), in addition to observable state.
    prefix_pin: str
    position: tuple
    velocity: tuple
    body_position: tuple
    body_rotation: tuple
    body_velocity: tuple


def verification(entry, observations):
    """Rows are (build, entry, exit_label, observed_contact_or_trigger, reached_exit).

    A drive through an 8-unit proximity zone is not engine evidence. Nor is another
    ball moving a plate that was already open at entry. No nearest-cell interpolation.
    """
    if not entry.static_key or not entry.prefix_pin or not entry.owner or len(entry.position)!=3 or len(entry.velocity)!=3:
        return frozenset()
    if entry.owner.startswith('P_Modul_25') and entry.ball not in (0,2):
        return frozenset()
    if entry.owner.startswith(('P_Modul_', 'P_Box', 'P_Ball_')) and not entry.owner.startswith('P_Modul_18'):
        if not (len(entry.body_position)==3 and len(entry.body_rotation)==4 and len(entry.body_velocity)==3):
            return frozenset()
    seen={}
    for build,state,exit_label,touched,reached in observations:
        if entry==state and touched and reached and exit_label:
            seen.setdefault(exit_label,set()).add(build)
    return frozenset(exit_label for exit_label,builds in seen.items() if {'x86soft','2002'}<=builds)


def pose_bins(tracker,p,radius=16.):
    """Small, deterministic pose suffix for an archive cell near moving mechanisms."""
    import numpy as np
    result=[]
    for row in sorted(tracker.obs.body,key=lambda x:int(x['ckid'])):
        name=tracker.level.name(row['ckid']) or ''
        if not name.startswith(('P_Modul_','P_Box','P_Ball_')) or np.linalg.norm(row['pos']-p)>radius:continue
        quat=np.asarray(row['q']);quat=quat*np.sign(quat[np.argmax(abs(quat))]);result.extend((int(row['ckid']),*np.rint(4*quat).astype(int),*np.floor(row['pos']/2).astype(int)))
    return tuple(result)
