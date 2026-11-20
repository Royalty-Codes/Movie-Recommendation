import re
import numpy as np
import pandas as pd
from .metrics import rank_items
from .stats import bootstrap_ci
from ..explain import explain
from ..fusion import COMPONENTS

def assess(model,candidates,targets,k=10,n_bootstrap=10000,seed=7):
    rng=np.random.default_rng(seed)
    records=[]; explanations=[]; suff=[]
    target_map=dict(zip(targets.user_id,targets.item_id))
    for (_,u),(items,ties) in candidates.items():
        z=model.components(u,items); contrib=z*model.weights
        scores=contrib.sum(1); order=rank_items(scores,ties)
        original_rank=np.empty(len(items),int); original_rank[order]=np.arange(1,len(items)+1)
        for j in order[:k]:
            ex=explain(model,u,int(items[j])); explanations.append(ex)
            cited=[COMPONENTS.index(c) for c in ex['cited_components']]
            others=[c for c in range(len(COMPONENTS)) if c not in cited]
            if not cited or not others: continue
            for primary in cited:
                control=int(rng.choice(others))
                # Neutralize only this item's component; other candidates retain scores.
                deleted=scores.copy(); deleted[j]-=contrib[j,primary]
                random_deleted=scores.copy(); random_deleted[j]-=contrib[j,control]
                rank_del=int(np.flatnonzero(rank_items(deleted,ties)==j)[0])+1
                rank_random=int(np.flatnonzero(rank_items(random_deleted,ties)==j)[0])+1
                records.append(dict(user_id=u,item_id=int(items[j]),component=COMPONENTS[primary],
                    control=COMPONENTS[control],rank_drop=rank_del-original_rank[j],
                    control_rank_drop=rank_random-original_rank[j],
                    faithful=float(rank_del>rank_random),evidence=bool(ex['evidence_items'])))
        # Sufficiency uses the target's own explanation, with its top two components for the whole pool.
        j=int(np.flatnonzero(items==target_map[u])[0])
        cited=[COMPONENTS.index(c) for c in explain(model,u,int(items[j]))['cited_components']]
        retained=rank_items(contrib[:,cited].sum(1) if cited else np.zeros(len(items)),ties)
        r=int(np.flatnonzero(retained==j)[0])+1; original=original_rank[j]
        suff.append((1/np.log2(r+1) if r<=k else 0,1/np.log2(original+1) if original<=k else 0))
    details=pd.DataFrame(records)
    rows=[]
    for col in ['rank_drop','control_rank_drop','faithful','evidence']:
        means=details.groupby('user_id')[col].mean()
        low,high=bootstrap_ci(means,n_bootstrap,seed)
        rows.append(dict(metric=col,mean=means.mean(),ci_low=low,ci_high=high))
    texts=[ex['text'] for ex in explanations]
    sets=[set(re.findall(r'\w+',text.lower())) for text in texts]
    similarities=[]
    for _ in range(min(10000,len(sets)*2)):
        a,b=rng.choice(len(sets),2,replace=False)
        if explanations[a]['user_id']!=explanations[b]['user_id']:
            similarities.append(len(sets[a]&sets[b])/max(1,len(sets[a]|sets[b])))
    for metric,value in [('distinct_strings',len(set(texts))),('explanation_count',len(texts)),
                         ('mean_jaccard',np.mean(similarities)),
                         ('evidence_coverage',np.mean([bool(e['evidence_items']) for e in explanations])),
                         ('sufficiency_ndcg_retained',np.mean(np.array(suff)[:,0])/max(1e-12,np.mean(np.array(suff)[:,1])))]:
        rows.append(dict(metric=metric,mean=value,ci_low=np.nan,ci_high=np.nan))
    return pd.DataFrame(rows),details,explanations
