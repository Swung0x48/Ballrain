"""Replay scout keys to a measured hinge state and steer using only legal arrows."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from routescout_atlas import load,state,tas,DEFAULT_DELTA,world_to_camera_keys,Ev
from ballrain.routescout.mechanisms import Rules
from ballrain.reach_mechanisms import uses_lowered_plate


def main(a):
    src=json.loads(a.input.read_text());keys=np.repeat([k for k,n in src['spawn_keys']],[n for k,n in src['spawn_keys']]);
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        s=load(a,src['level'],a.bin,a.copy,Path(tmp));i=a.cut-s.frames_done
        try:
            import itertools
            from routescout_pickups import append
            seq=[[int(k),len(list(g))] for k,g in itertools.groupby(keys[:i])];s.act(tas.make_frames([DEFAULT_DELTA]*i,keys[:i]));m=Rules(s.tracker);plate=next(x for x in m.plates if 'state_probe' in x);probe=m.probe[2];pivot=np.array(plate['state_probe']['pivot']);axis=-np.array(plate['state_probe']['down_axis'])[[0,2]];side=axis[::-1]*[1,-1];hinge=pivot[[0,2]];paper=min((q for q in m.rows if m.rule(q['type'],0)=='transform_paper'),key=lambda q:np.linalg.norm(np.array(q['matrix'][3][:3])-pivot));approach=hinge-14*axis+a.offset*side;end=hinge+14*axis+a.offset*side
            stops=([hinge-15*axis,np.array(paper['matrix'][3])[[0,2]]] if a.return_for_paper else [])+[approach,end];j=0;rows=[];hits=0;reason='budget'
            for _ in range(850):
                p,v,b=state(s);contact=any(int(n)==probe[0] for n in s.obs.fric['partner']);down=uses_lowered_plate(s.tracker,p,probe);ck=next((x['ck_matrix'][0].tolist() for x in s.obs.body if int(x['ckid'])==probe[0]),None);rows.append([s.frames_done,*p.tolist(),*v.tolist(),int(b),bool(contact),bool(down),j,ck]);h=stops[j]-p[[0,2]];dist=np.linalg.norm(h)
                if dist<1.8 and (not a.return_for_paper or j!=1 or b==1):
                    j+=1
                    if j==len(stops):reason='far lip';break
                    h=stops[j]-p[[0,2]];dist=np.linalg.norm(h)
                want=h*min(2.0,(a.speed if j==len(stops)-1 else 8.)/max(dist,.01));u=want-v[[0,2]];k=(8 if u[0]>.3 else 4 if u[0]<-.3 else 0)|(1 if u[1]>.3 else 2 if u[1]<-.3 else 0);key=world_to_camera_keys(k,s.obs.cams['cams'][0]['matrix']) if s.tracker.flags.ball_nav else 0;before=s.frames_done;o=s.act(tas.make_frames([DEFAULT_DELTA],[key]),repeat=4,stop=[Ev.BallOff]);append(seq,key,s.frames_done-before)
                if o is None or o.has_event(Ev.BallOff):reason='fall';break
                hits+=int(contact and b==1 and j==len(stops)-1)
                if hits>=140:reason='sustained contact';break
            a.out.write_text(json.dumps(dict(spawn_keys=seq,reason=reason,cut=a.cut,offset=a.offset,speed=a.speed,asset_pin=m.pin,pivot=pivot.tolist(),rows=rows)));print(a.out.name,reason,j,hits,rows[-1][:10],flush=True)
        finally:s.close(kill=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('bin','root','input','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--copy',required=True);p.add_argument('--cut',type=int,required=True);p.add_argument('--speed',type=float,default=14);p.add_argument('--offset',type=float,default=0);p.add_argument('--return-for-paper',action='store_true');main(p.parse_args())
