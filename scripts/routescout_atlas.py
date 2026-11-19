"""Scout a stock level from cold load; the second executable independently tests the same prefix."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain import tas
from ballrain.client import AgentSession, DEFAULT_DELTA, SessionConfig
from ballrain.env import world_to_camera_keys
from ballrain.headless import HeadlessLauncher
from ballrain.protocol import Ev, EVENT_NAMES
from ballrain.reach import BALLS, ReachConfig, SupportMap, rollout
from ballrain.routescout.map import inventory
from ballrain.routescout.graph import SurfaceGraph
from ballrain.routescout.sensitivity import section_scan
from ballrain.routescout.evidence import aggregate_rows
from ballrain.routescout.search import FRAMES, KEYS, PUSH, STEP, search


def load(a, level, binary, copy, directory):
    s = AgentSession(SessionConfig(level, directory, copy=copy, hello_timeout_s=20, obs_timeout_s=30),
                     HeadlessLauncher(binary=str(binary.resolve()), root=str(a.root.resolve()), socket='tcp'))
    try:
        s.start()
        if not s.tracker.flags.started:
            s.run_until(0, [Ev.StartLevel], limit=3000)
        if level == 1:
            for _ in range(200):
                if s.tracker.flags.cam_nav or s.tracker.flags.ball_nav:
                    break
                s.act(tas.make_frames([DEFAULT_DELTA]*2, [tas.KEY_Q,0]), stop=[Ev.BallNavActive,Ev.CamNavActive])
        if not s.tracker.flags.ball_nav:
            s.run_until(0, [Ev.BallNavActive], limit=3000)
        if not s.tracker.flags.ball_nav:
            raise RuntimeError('no controllable start')
        return s
    except BaseException:
        s.close(kill=True)
        raise


def state(s):
    b = s.obs.ball
    return np.array(b['pos']), np.array(b['speed'], float) + b['speed_change'], s.tracker.ball_type()


def verify(s, smap, n, ball):
    if hasattr(smap,'scout_rules') and smap.scout_rules.tr is not s.tracker:
        from ballrain.routescout.mechanisms import Rules
        smap.scout_rules=Rules(s.tracker)
    rows = []
    for i, target in enumerate(n.ends):
        p,v,b = state(s)
        if b != ball:
            break
        count = min(FRAMES, n.frames-i*FRAMES)
        from ballrain.routescout.search import scout_rollout
        r = scout_rollout(smap,p,v,BALLS[b],PUSH[:,:count//STEP], ReachConfig(step_psi=STEP))
        options = np.flatnonzero(r['alive'])
        if not len(options):
            break
        j = min(options, key=lambda k: np.linalg.norm(r['traj'][k,-1]-target))
        pred = r['traj'][j,-1]; key = world_to_camera_keys(KEYS[j], s.obs.cams['cams'][0]['matrix'])
        before = s.frames_done
        o = s.act(tas.make_frames([DEFAULT_DELTA], [key]), repeat=count,
                  stop=[Ev.BallOff, Ev.PreCheckpoint, Ev.LevelFinish, Ev.BallNavInactive])
        if o is None:
            break
        actual = state(s)[0]; events = [EVENT_NAMES.get(e[1],str(e[1])) for e in s.tracker.events if e[0] > before]
        alive = not o.has_event(Ev.BallOff); full = s.frames_done-before == count
        err = float(np.linalg.norm(pred-actual)); dev = float(np.linalg.norm(actual-target))
        rows.append(dict(start=p.tolist(),actual=actual.tolist(),prediction=pred.tolist(),world_key=int(KEYS[j]),
                         frames=s.frames_done-before, error=err, deviation=dev, events=events,
                         match=bool(full and alive and state(s)[2]==b and err<=2)))
        if not full or not alive or dev>4 or s.ended or not s.tracker.flags.ball_nav or o.has_event(Ev.PreCheckpoint):
            break
    return rows


def draw(path, level, g, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, axes = plt.subplots(2,1,figsize=(10,8),height_ratios=[1.4,1], layout='constrained')
    t = g.tris; V = np.stack([t.A,t.B,t.C],1)[:,:,[0,2]]
    walk = abs(t.N[:,1]) >= .5
    for ax in axes:
        ax.add_collection(PolyCollection(V[walk], facecolors='#cdd8df',edgecolors='none'))
        ax.add_collection(PolyCollection(np.stack([g.rails.A,g.rails.B,g.rails.C],1)[:,:,[0,2]],facecolors='#708493',edgecolors='none'))
        for i,p in enumerate(data['map']['anchors']):
            if p is not None:
                ax.plot(p[0],p[2],'ko',ms=3); ax.annotate(str(i),(p[0],p[2]),xytext=(3,4),textcoords='offset points')
        for segment in data['segments']:
            if segment['guide']:
                guide=np.array(segment['guide']); ax.plot(guide[:,0],guide[:,2],color='#648d76',lw=.8)
        for segment in data['segments'][1:]:
            for n in segment.get('hypothetical_from_reset',[]):
                line=np.array([n['start'],*n['ends']]); ax.plot(line[:,0],line[:,2],color='#bb711f',lw=1)
        for family in data['prefixes']:
            line=np.array([data['start'],*family['ends']]); ax.plot(line[:,0],line[:,2],color='#dd6e1b',lw=1)
        for build,rows in data['checks'].items():
            if rows:
                line=np.array([rows[0]['start'],*[r['actual'] for r in rows]]); ax.plot(line[:,0],line[:,2],'.:',ms=3,label=build)
        ax.set_aspect('equal',adjustable='datalim'); ax.autoscale(); ax.set_xlabel('world x'); ax.set_ylabel('world z')
    axes[0].set_title(f'L{level:02} | static support and ordered goals (0 = start)')
    ax=axes[1]; line=np.array([data['start'],*data['prefixes'][0]['ends']]); ax.set_aspect('equal',adjustable='box'); ax.set_xlim(line[:,0].min()-8,line[:,0].max()+8); ax.set_ylim(line[:,2].min()-8,line[:,2].max()+8)
    ax.set_title('Start detail | orange: model prefixes; dotted: engine observations'); ax.legend(loc="upper right"); fig.savefig(path,dpi=140); plt.close(fig)


def local_field(t, depth, p, rules=None, live=False):
    radius=180
    sel=((t.hi[:,[0,2]]>=p[[0,2]]-radius)&(t.lo[:,[0,2]]<=p[[0,2]]+radius)).all(1)
    boxes=[(x[-2],x[-1]) for x in depth if min((x[-1]-x[-2])[[0,2]])>=20]
    field=SupportMap(t.A[sel],t.B[sel],t.C[sel],death_boxes=boxes)
    if rules is not None:
        field.scout_rules=rules;field.scout_live=live;field.scout_maps={};cache={}
        for k,layer in enumerate(rules.layers):
            if layer is None:continue
            ex=layer[((layer.max(1)[:,[0,2]]>=p[[0,2]]-radius)&(layer.min(1)[:,[0,2]]<=p[[0,2]]+radius)).all(1)]
            if not len(ex):continue
            if id(layer) not in cache:cache[id(layer)]=SupportMap(*[np.concatenate([a[sel],ex[:,j]]) for j,a in enumerate((t.A,t.B,t.C))],death_boxes=boxes)
            field.scout_maps[k]=cache[id(layer)]
    return field


def describe_node(n,ball):
    return dict(ends=n.ends,keys=n.keys,frames=n.frames,estimated_game_seconds=n.frames/132,
                status=n.status,materials=[ball] if n.ball==ball else [ball,n.ball],terminal_ball=n.ball)


def run(a, level):
    a.out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out) as tmp:
        s=load(a,level,a.soft,a.copy,Path(tmp)/'first')
        try:
            from ballrain.routescout.mechanisms import Rules
            rules=Rules(s.tracker);p,v,b=state(s); g,m=inventory(s.tracker);m['mechanism_pin']=rules.pin;m['fan_obbs']=rules.fans.tolist()
            for ob in m['placements']:ob['ball_rules']=[rules.rule(ob['name'],k) for k in range(3)]
            goal=np.asarray(m['anchors'][1]); t=g.tris
            # Each 48-frame plan can travel only within a bounded local search; use full intersecting triangles.
            graph=SurfaceGraph(t); anchors=m['anchors']
            segments=[dict(sector=i+1,guide=graph.corridor(src,dst),verified=False) for i,(src,dst) in enumerate(zip(anchors,anchors[1:]))]
            for row in segments: row['uncertain_edges']=graph.gates(row['guide'])
            smap=local_field(t,s.tracker.level.depth,p,rules,True)
            nodes,expanded=search(smap,p,v,b,1,goal,m['placements'],a.depth,a.width,segments[0]['guide'])
            if segments[0]['guide']:
                # A welded-graph shortcut can point into a wall. Retain a second beam
                # independent of the graph and compare actual predicted progress.
                alternate,more=search(smap,p,v,b,1,goal,m['placements'],a.depth,a.width)
                nodes=sorted(nodes+alternate,key=lambda n:(n.status!='goal region',np.linalg.norm(n.p-goal),n.frames))
                expanded+=more
            chosen=[]
            for n in nodes:
                if all(np.linalg.norm(n.p-other.p)>4 or n.status!=other.status for other in chosen):
                    chosen.append(n)
                if len(chosen)==3: break
            data=dict(segments=segments,map=m,start=p.tolist(),expanded=expanded,initial_ball=b,prefixes=[describe_node(n,b) for n in chosen],checks={})
            for desc,n in zip(data['prefixes'],chosen):
                desc['model_sections']=section_scan(smap,g,p,v,b,1,m['placements'],goal,n)
            data['checks']['x86soft']=verify(s,smap,chosen[0],b)
            for row in segments[1:]:
                i=row['sector']-1
                if i>=len(m['resets']) or anchors[i+1] is None: continue
                seed=np.array(m['resets'][i]); field=local_field(t,s.tracker.level.depth,seed,rules,False)
                row['hypothetical_from_reset']=[]
                for material in (0,2):
                    ns,attempts=search(field,seed,np.zeros(3),material,i+1,np.array(anchors[i+1]),m['placements'],a.depth,a.width,row['guide'])
                    row['hypothetical_from_reset'].append(dict(start=seed.tolist(),expanded=attempts,**describe_node(ns[0],material)))
        finally: s.close(kill=True)
        for i,n in enumerate(chosen):
            checks=data['checks'] if i==0 else {}
            for name,binary,copy in [('x86soft',a.soft,a.copy),('2002',a.retail,a.second_copy)]:
                if name in checks: continue
                s=load(a,level,binary,copy,Path(tmp)/'replay')
                try: checks[name]=verify(s,smap,n,b)
                finally: s.close(kill=True)
            data['prefixes'][i]['verification']=checks
            data['prefixes'][i]['supported_rest_starts']=[]
    if a.human_table and a.levels==[level]:
        evidence=aggregate_rows(a.human_table.read_text(encoding='utf8'))
        for row in segments: row['human_evidence']=evidence.get(str(row['sector']),[])
    stem=a.out/f'L{level:02}'
    stem.with_suffix('.json').write_text(json.dumps(data,allow_nan=False),encoding='utf8'); draw(stem.with_suffix('.png'),level,g,data)
    print(level,expanded,len(chosen[0].keys),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['root','soft','retail','out']: ap.add_argument('--'+k,type=Path,required=True)
    for k in ['copy','second-copy']: ap.add_argument('--'+k,required=True)
    ap.add_argument('--levels',nargs='+',type=int,required=True); ap.add_argument('--width',type=int,default=5); ap.add_argument('--depth',type=int,default=12)
    ap.add_argument('--human-table',type=Path,help='post-search aggregate evidence; single level only')
    args=ap.parse_args()
    if args.human_table and len(args.levels)!=1: ap.error('human evidence must identify a single level')
    if not 1<=args.depth<=12 or not 1<=args.width<=32: ap.error('depth 1..12, width 1..32')
    for level in args.levels: run(args,level)
