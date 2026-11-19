"""Publish rankings only from fresh, pinned, two-engine receipts; retain rejected proposals separately."""
from collections import Counter
from ballrain.reach_mechanisms import level_asset,asset_pin


def evidence(d,a,b):
    if d.get('mechanism_pin')!=asset_pin():raise ValueError('missing or different mechanism asset pin')
    obs=d['mechanism_observations'];counts=Counter();sources=level_asset(d['level'])['rules'];fan=Counter();blocked=[]
    for r in obs:
        if r['gate']:blocked.append([r['frame'],r['gate']]) # reject entire key prefix, not only last sector
        if not a<r['frame']<=b:continue
        for n,k,rr in r['contacts']:
            if sources[k]['rules'][r['ball']]!=rr:raise ValueError('receipt and source rules disagree')
            counts[n,k,r['ball'],rr,bool(r.get('aligned_down') and rr=='blocked_when_raised')]+=1
        if r['fan']:fan[r['ball']]+=1
    return dict(pin=asset_pin(),blocked=blocked,source='reach_mechanism_levels',contact_rules=[dict(owner=n,kind=k,ball=bt,rule=v,source=sources[k]['source'],observed_down=down,samples=num) for (n,k,bt,v,down),num in counts.items()],in_compiled_fan_samples=dict(fan))
