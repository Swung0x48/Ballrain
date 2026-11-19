"""Finite beam over reach rollouts. Radius gates are hypotheses, never engine events."""
from dataclasses import dataclass
import numpy as np
from ballrain.reach import BALLS, KEY_COMBOS, ReachConfig, key_push, rollout, PSI_HZ, GAME_PSI_HZ
from .map import MATERIAL
from .graph import guide_distance

KEYS = (0, *KEY_COMBOS)
STEP, STEPS = 4, 12
FRAMES = STEP * STEPS
PUSH = np.repeat(key_push(KEYS, 0)[:, None, :], STEPS, axis=1)


def scout_rollout(field,p,v,ball,push,cfg):
    from ballrain.fastreach import fan_dv,FAN_R_NOM
    from ballrain.reach_mechanisms import uses_lowered_plate
    rules=getattr(field,'scout_rules',None);k=next(i for i,b in BALLS.items() if b.name==ball.name);layer=k;fan=None
    if rules is not None:
        if field.scout_live and rules.probe and k==rules.probe[0] and all(uses_lowered_plate(rules.tr,q,rules.probe[2]) for q in np.asarray(p).reshape(-1,3)):layer=rules.probe[1]
        dv=fan_dv(ball);fan=(rules.fans,dv,FAN_R_NOM) if len(rules.fans) and dv else None
    r=rollout(getattr(field,'scout_maps',{}).get(layer,field),p,v,ball,push,cfg,fan=fan) if rules is not None else rollout(field,p,v,ball,push,cfg)
    if 'final_velocity' not in r:raise ValueError('scout requires the terminal-velocity return on the reachmech rollout')
    return r


def cell(pos, vel, ball, sector):
    return (sector, ball, *np.floor(pos / 3).astype(int), *np.floor(vel / 4).astype(int))


def contact(line, center, radius):
    """First segment intersecting a sphere (including tangent or stationary segments)."""
    a, d = line[:-1], np.diff(line, axis=0)
    t = np.clip(np.sum((center-a)*d, axis=1) / np.maximum(1e-12, (d*d).sum(1)), 0, 1)
    hit = np.flatnonzero(np.linalg.norm(a+t[:,None]*d-center, axis=1) <= radius)
    return int(hit[0])+1 if len(hit) else None


def boundary(line, ball, sector, objects, goal):
    hits = []
    for o in objects:
        if o['sector'] not in (0, sector):
            continue
        k=o['kind'];rule=o.get('ball_rules',[None]*3)[ball];material=({'transform_wood':0,'transform_paper':1,'transform_stone':2}.get(rule,ball) if rule is not None else MATERIAL.get(o['name'],ball))
        if k == 'trafo' and material == ball:
            continue
        if k in ('pickup', 'start', 'finish', 'checkpoint', 'reset'):
            continue
        idx = contact(line, np.asarray(o['pos']), 4 if k == 'trafo' else 8)
        if idx is not None:
            hits.append((idx, 0, o['name'] if rule is None else o['name']+':'+rule+' (engine gate)', material))
    idx = contact(line, goal, 4)
    if idx is not None:
        hits.append((idx, 1, 'goal region', ball))
    return min(hits) if hits else None


@dataclass
class Node:
    p: np.ndarray
    v: np.ndarray
    keys: tuple = ()
    ends: tuple = ()
    status: str = 'frontier'
    ball: int = 0
    frames: int = 0


def search(smap, p, v, ball, sector, goal, objects, depth=12, width=5, corridor=()):
    root = Node(p, v, ball=ball)
    if not BALLS[ball].supported:
        root.status = 'paper: engine required'
        return [root], 0
    frontier, leaves, visits = [root], {}, {}
    distance = guide_distance(corridor, goal)
    expanded = 0
    for _ in range(depth):
        candidates = {}
        for n in frontier:
            r = scout_rollout(smap, n.p, n.v, BALLS[ball], PUSH, ReachConfig(step_psi=STEP))
            expanded += len(KEYS)
            for j in np.flatnonzero(r['alive']):
                line = r['traj'][j]; rules=getattr(smap,'scout_rules',None)
                if rules is not None and any(rules.gate(x,y,ball,allow_pose=smap.scout_live) for x,y in zip(line,line[1:])):continue
                hit = boundary(line, ball, sector, objects, goal)
                i, _, status, b = hit or (STEPS, 0, 'frontier', ball)
                end = line[i]; speed = r['final_velocity'][j]
                child = Node(end, speed, n.keys+(KEYS[j],), n.ends+(end.tolist(),), status, b, n.frames+i*STEP)
                score = distance(end) + child.frames/GAME_PSI_HZ
                if hit:
                    tag = (status, b, tuple(np.floor(end/4)))
                    if tag not in leaves or child.frames < leaves[tag].frames:
                        leaves[tag] = child
                    continue
                k = cell(end, speed, b, sector)
                rank = score + 2 * visits.get(k, 0)
                if k not in candidates or rank < candidates[k][0]:
                    candidates[k] = rank, child
        if not candidates:
            break
        frontier = []
        for k, (_, n) in sorted(candidates.items(), key=lambda item:item[1][0])[:width]:
            frontier.append(n); visits[k] = visits.get(k, 0)+1
    return sorted([*leaves.values(), *frontier], key=lambda n:(n.status != 'goal region', distance(n.p), n.frames)), expanded
