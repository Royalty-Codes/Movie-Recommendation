import numpy as np
import pandas as pd

def bootstrap_ci(values, n=10000, seed=7):
    values=np.asarray(values,float)
    if not len(values): return (float('nan'),float('nan'))
    rng=np.random.default_rng(seed)
    means=[]
    for start in range(0,n,200):
        means.extend(values[rng.integers(len(values),size=(min(200,n-start),len(values)))].mean(1))
    return tuple(np.quantile(means,[.025,.975]))

def paired_test(a,b,n=10000,seed=7):
    d=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(seed)
    count=0
    for start in range(0,n,200):
        null=(rng.choice([-1,1],size=(min(200,n-start),len(d)))*d).mean(1)
        count += np.count_nonzero(np.abs(null)>=abs(d.mean())-1e-15)
    low,high=bootstrap_ci(d,n,seed)
    return {'difference':float(d.mean()),'ci_low':low,'ci_high':high,'p_value':(count+1)/(n+1)}

def comparison_table(raw,top,config_hash,n=10000):
    rows=[]
    for (method,protocol),g in raw.groupby(['method','protocol']):
        for metric in ['HR@10','NDCG@10','MRR','Recall@10']:
            # Sampling/model seeds are repeated measures, never independent users.
            per_user=g.groupby('user_id')[metric].mean()
            low,high=bootstrap_ci(per_user,n)
            rows.append(dict(method=method,protocol=protocol,metric=metric,mean=per_user.mean(),
                std=g.groupby('init_seed')[metric].mean().std(ddof=0),ci_low=low,ci_high=high,
                users=len(per_user),config_hash=config_hash))
    return pd.DataFrame(rows)
