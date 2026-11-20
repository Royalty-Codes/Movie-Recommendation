import numpy as np
import pandas as pd
from .metrics import rank_items, ranking_metrics
from ..data import histories

def candidate_sets(all_ratings, movies, targets, seeds, negatives=99, full=False):
    """History is used ONLY to exclude negatives; no rating values enter scoring."""
    watched = {u:set(h) for u,h in histories(all_ratings).items()}
    items = np.asarray(movies.index, dtype=int)
    result = {}
    for seed in seeds:
        for row in targets.itertuples():
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(row.user_id)]))
            pool = items[~np.isin(items, list(watched[row.user_id]))]
            if not full:
                pool = rng.choice(pool, min(negatives,len(pool)), replace=False)
            cands = np.concatenate(([row.item_id],pool))
            result[(int(seed),int(row.user_id))] = (cands, rng.random(len(cands)))
    return result

def evaluate(model, candidates, targets, popularity, item_map, k=10, method='', protocol='P1', init_seed=42):
    target_map = dict(zip(targets.user_id,targets.item_id))
    rows, top_rows = [], []
    for (seed,u),(items,ties) in candidates.items():
        scores = model.score(u,items)
        order = rank_items(scores,ties)
        rank = int(np.flatnonzero(items[order] == target_map[u])[0])+1
        rows.append(dict(method=method,protocol=protocol,init_seed=init_seed,seed=seed,user_id=u,
                         target=int(target_map[u]),rank=rank,**ranking_metrics(rank,k)))
        top = items[order[:k]]
        top_rows.append(dict(method=method,protocol=protocol,init_seed=init_seed,seed=seed,user_id=u,
            items=top.tolist(),mean_popularity=float(np.mean([popularity[item_map[i]] for i in top]))))
    return pd.DataFrame(rows), pd.DataFrame(top_rows)

def validation_hr(model, candidates, targets, k=10):
    values=[]
    for (_,u),(items,ties) in candidates.items():
        order=rank_items(model.score(u,items),ties)
        values.append(float(0 in order[:k]))
    return float(np.mean(values))
