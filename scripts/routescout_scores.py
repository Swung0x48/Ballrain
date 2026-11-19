"""Route-local sensitivity index. Missing trials are unknown, never easy."""
from math import hypot
import numpy as np


def window(p):
    base,*tests=p['trials'];assert base['passed'] and tests
    u=np.array(base['entry']['vel'])[::2]; side=u[::-1]*[1,-1]/max(np.linalg.norm(u),1e-10)
    good=[t['entry'] for t in [base,*tests] if t['passed'] and t['entry']];speed=[hypot(*e['vel'][::2]) for e in good];offset=[float((np.array(e['pos'])[::2]-np.array(base['entry']['pos'])[::2])@side) for e in good]; failures=[t for t in tests if not t['passed']];dead=sum(t['death'] for t in failures);rec=sum(t['recovered'] for t in failures);unknown=len(failures)-dead-rec
    scorebase=base.get('end_score'); pay=sum(t['passed'] and t['end_score'][1:]==scorebase[1:] and abs(t['end_score'][0]-scorebase[0])<=1 for t in tests) if scorebase else None
    return dict(payoff_matched_passes=pay,entry=p['entry'],end=p['end'],followup=p['followup'],passes=len(tests)-len(failures),n=len(tests),deaths=dead,rejoins=rec,unresolved=unknown,speed=[min(speed),max(speed)],offset=[min(offset),max(offset)],width=max(offset)-min(offset),speed_span=max(speed)-min(speed))


def difficulty(w,events,changes,strip_width=None):
    if not w:return None
    n=sum(x['n'] for x in w);rate=sum(x['passes'] for x in w)/n; width=min(x['width'] for x in w);width=min(width,strip_width) if strip_width is not None else width;band=min(x['speed_span'] for x in w);cost=sum(x['deaths']+.5*x['unresolved']+.25*x['rejoins'] for x in w)/n
    return round(100*(.4*(1-rate)+.15*(1-min(1,width/4))+.12*(1-min(1,band/8))+.15*min(1,events/8)+.08*min(1,changes/3)+.10*cost),2)


def frontier(rows,payoff,sign):
    rows=[r for r in rows if r['difficulty'] is not None and r[payoff] is not None];return [r['id'] for r in rows if not any(q['difficulty']<=r['difficulty'] and sign*q[payoff]<=sign*r[payoff] and (q['difficulty']<r['difficulty'] or sign*q[payoff]<sign*r[payoff]) for q in rows)]


