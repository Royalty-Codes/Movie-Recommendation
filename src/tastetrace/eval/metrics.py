import numpy as np

def ranking_metrics(rank, k=10):
    hit = float(rank <= k)
    return {'HR@10': hit, 'NDCG@10': hit/np.log2(rank+1), 'MRR': 1/rank, 'Recall@10': hit}

def rank_items(scores, tie_keys):
    scores = np.asarray(scores)
    if np.isnan(scores).any():
        raise ValueError('NaN scores are not valid')
    return np.lexsort((tie_keys, -scores))
