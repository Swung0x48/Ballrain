"""Rank verified representatives, not search seeds. Manifest is a post-search family classification."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from routescout_payoff import inventory,DEFAULT_DELTA
from routescout_scores import window,difficulty,frontier
from ballrain.path import ReferencePath


def read(p):return json.loads(p.read_text(encoding='utf8'))


from ballrain.routescout.mech_scores import evidence,asset_pin

def summarize(o,atlases,manifest,mechanisms):
    removed={};maps={};routes=[];groups={};anchors={}; refs=o/'refs';refs.mkdir(exist_ok=True)
    for l in manifest['levels']:
        m=read(atlases/f'L{l:02}.json');anchors[l]=m['map']['anchors'];maps[str(l)]=dict(inventory=inventory(m),geometry=m['map']['static_key']);groups[l]=len(m['map']['anchors'])-1
    for c in manifest['members']:
        tag=c['id'];d=read(mechanisms/'replays'/(tag+'.json'));level=d['level'];s=len(d['boundaries'])-1;a,b=d['boundaries'][-2:];inv=maps[str(level)]['inventory'];keys={k for k in inv if inv[k]['sector']==s};picked=sorted(k for k in d['picked'] if k in keys);assert not d['unresolved'] and not d['end']['deaths'];rows=[r for r in d['rows'] if a['frame']<=r[0]<=b['frame']];balls=[k for k,_ in itertools.groupby(r[7] for r in rows)];ev=[(t,n) for t,n in d['events'] if a['frame']<t<=b['frame']];pauses=[t for t,n in ev if n=='PauseLevel'];ends=[t for t,n in ev if n=='UnpauseLevel'];assert len(pauses)==len(ends)
        proof=evidence(d,a['frame'],b['frame'])
        if proof['blocked']:removed[tag]=proof;continue
        gain=0;prev=a['points'];banks=[0]*len(pauses)
        def followers(t):return next((r[3] for r in reversed(d['score_log']) if r[0]<=t),d['start']['followers'])
        bank_proof=[dict(pause=st,unpause=en,before=followers(st),after=followers(en)) for st,en in zip(pauses,ends)]
        for t,pts,_,_,_ in d['score_log']:
            if not a['frame']<t<=b['frame']:continue
            delta=pts-prev;prev=pts;credit=max(0,round(delta/20)*20);gain+=credit
            for i,(st,en) in enumerate(zip(pauses,ends)):
                if st<t<=en:banks[i]+=credit
        for br,credit in zip(bank_proof,banks):
            br['points']=credit;assert br['after']==0 and credit==20*(br['before']-br['after'])
        mains=sum(inv[k]['kind']=='point' for k in picked);small=gain-100*mains;assert 0<=small<=120*mains and small%20==0
        assert not d['finished'], 'Sector exporter does not silently certify whole-level runs'
        pm=b['potential']-a['potential'];assert pm==(b['points']-a['points'])+(b['lives']*b['life_bonus']-a['lives']*a['life_bonus']);assert prev==b['points']; assert sum(inv[k]['kind']=='life' for k in picked)==b['lives']-a['lives'];assert gain-(b['points']-a['points'])>=0
        w=[window(read(f)) for dirname in ('probes','probes-extra','probes-critical') if (f:=o/dirname/(tag+'.json')).exists()]
        extra=mechanisms/'probes'/(tag+'.json')
        if extra.exists():w.append(window(read(extra)))
        strip=read(o/'widths.json').get(tag,{}).get('minimum') if (o/'widths.json').exists() else None;di=difficulty(w,len(c['precision_events']),len(balls)-1,strip['width'] if strip else None);ref=refs/(tag+'.npz');p=ReferencePath([r[1:4] for r in rows],[s]*len(rows),source='scout',meta={'source':'scout','scope':'sector','id':tag,'frame_sha256':d['frame_sha256']},t_ms=(np.array([r[0] for r in rows])-a['frame'])*DEFAULT_DELTA);p.save(ref);z=ReferencePath.load(ref);assert z.point_sector.tolist()==[s]*len(rows) and np.allclose(p.P,z.P)
        routes.append(dict(**c,mechanism=proof,level=level,sector=s,first=a,last=b,ball_sequence=balls,windows=w,static_strip=strip,difficulty=di,sr_ms=int(b['sr_ms'])-int(a['sr_ms']),game_ms=(b['frame']-a['frame'])*float(DEFAULT_DELTA),hs_delta=pm,picked=picked,missed=sorted(keys-set(picked)),main_points=100*mains,follower_points=small,esc_credit=banks,esc_evidence=bank_proof,clock_cost=gain-(b['points']-a['points']),life_value_delta=pm-b['points']+a['points'],all_sector_banked=(set(picked)==keys and small==120*mains and sum(banks)==small and b['followers']==0),level_hs=None,ref=ref.relative_to(o).as_posix(),sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),key_sha256=d['frame_sha256'],contacts=sorted(set(n for r in rows for n in r[8])),rest_starts=[]))
    for c in manifest.get('whole_members',[]):
        tag=c['id'];d=read(mechanisms/'replays'/(tag+'.json'));level=d['level'];assert d['finished'] and d['finish_sr_ms'] is not None and d['official_hs'] is not None and not d['end']['deaths'] and not d['unresolved'] and len(d['boundaries'])==groups[level];rows=d['rows'];miss=sorted(set(maps[str(level)]['inventory'])-set(d['picked']));last=d['boundaries'][-1]['frame'];goal=np.array(anchors[level][-1]);pauses=[f for f,e in d['events'] if e=='PauseLevel' and f>last];final_bank=False
        proof=evidence(d,d['start']['frame'],d['end']['frame'])
        if proof['blocked']:removed[tag]=proof;continue
        for f in pauses:
            resume=next((t for t,e in d['events'] if e=='UnpauseLevel' and t>f),None);near=min(rows,key=lambda r:abs(r[0]-f));followers=next((r[3] for r in reversed(d['score_log']) if resume is not None and r[0]<=resume),None)
            if resume and f>=max((v['frame'] for v in d['picked'].values()),default=0) and np.linalg.norm(near[1:4]-goal)<=40 and followers==0 and d['end']['followers']==0:final_bank=True
        sector_at=np.searchsorted([x['frame'] for x in d['boundaries'][1:]], [r[0] for r in rows],side='left')+1;ref=refs/(tag+'.npz');p=ReferencePath([r[1:4] for r in rows],sector_at,source='scout',meta={'source':'scout','scope':'whole','id':tag},t_ms=(np.array([r[0] for r in rows])-d['start']['frame'])*float(DEFAULT_DELTA));p.save(ref);assert len(ReferencePath.load(ref).P)==len(rows)
        # Do not borrow sector difficulty from an unrelated start. Supply remeasured windows for this tape.
        w=[window(read(o/f)) for f in c.get('probe_files',[])];balls=[b for b,_ in itertools.groupby(r[7] for r in rows)];di=difficulty(w,len(c['precision_events']),len(balls)-1) if len({int(np.searchsorted([x['frame'] for x in d['boundaries'][1:]],v['entry'])) for v in w})==groups[level] else None
        routes.append(dict(**c,mechanism=proof,level=level,sector='level',picked=list(d['picked']),missed=miss,finish_esc=final_bank,all_sector_banked=not miss and final_bank,sr_ms=int(d['finish_sr_ms']),hs_delta=d['official_hs'],level_hs=d['official_hs'],difficulty=di,windows=w,ref=ref.relative_to(o).as_posix(),sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),key_sha256=d['frame_sha256']))
    ranks={}
    for l,last in groups.items():
        for s in [*range(1,last+1),'level']:
            rs=[r for r in routes if r['level']==l and r['sector']==s];eligible=[r for r in rs if r['all_sector_banked']];ranks[f'L{l:02}/{s}']=dict(sr=[r['id'] for r in sorted(rs,key=lambda r:r['sr_ms'])],hs_partial=[r['id'] for r in sorted(rs,key=lambda r:-r['hs_delta'])],hs_eligible=[r['id'] for r in sorted(eligible,key=lambda r:-r['hs_delta'])],pareto_sr=frontier(rs,'sr_ms',1),pareto_hs_partial=frontier(rs,'hs_delta',-1),pareto_hs_eligible=frontier(eligible,'hs_delta',-1))
    previous=read(mechanisms/'before-scores.json') if (mechanisms/'before-scores.json').is_file() else None
    if previous:
        old={v['id']:v for v in previous['members']}
        for r in routes:
            before=old.get(r['id']);r['revision']={'status':'newly_verified' if before is None else ('difficulty_revised' if r['difficulty']!=before['difficulty'] else 'retained'),'previous_difficulty':before['difficulty'] if before else None,'sr_delta_ms':r['sr_ms']-before['sr_ms'] if before else None,'partial_hs_delta':r['hs_delta']-before['hs_delta'] if before else None}
            scope=f"L{r['level']:02}/{r['sector']}";r['revision']['pareto_changes']={k:[r['id'] in previous['rankings'].get(scope,{}).get(k,[]),r['id'] in ranks[scope][k]] for k in ('pareto_sr','pareto_hs_partial','pareto_hs_eligible')}
    (o/'scores.json').write_text(json.dumps(dict(version=2,mechanism_pin=asset_pin(),removed=removed,maps=maps,members=routes,rankings=ranks),indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--atlases',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--mechanisms',type=Path,required=True);a=p.parse_args();summarize(a.out,a.atlases,read(a.manifest),a.mechanisms)
