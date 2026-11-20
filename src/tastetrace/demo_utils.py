from pathlib import Path
import pickle
import numpy as np
from .explain import explain

def load_bundle(path='results/demo_bundle.pkl'):
    # Only load local experiment artifacts produced by this project.
    with Path(path).open('rb') as f: return pickle.load(f)

def recommend(model,user_id,k=10,history=None):
    index=model.index
    h=index.histories.get(user_id,[]) if history is None else history
    items=index.items[~np.isin(index.items,h)]
    scores=model.score(user_id,items) if history is None else model.components(user_id,items,history)@model.weights
    order=np.lexsort((items,-scores))[:k]
    return [(int(items[j]),float(scores[j])) for j in order]

def demo_report(bundle,user_id):
    models=bundle['models']; tt=models['TasteTrace-v2']
    target=int(bundle['split'].test.set_index('user_id').at[user_id,'item_id'])
    lines=[f'# TasteTrace demo: user {user_id}',f'Config: `{bundle["config_hash"]}`',
           f'Held-out movie: {tt.movies.at[target,"title"]}',
           'Current inferred state: '+str(int(tt.posteriors[user_id].argmax())),
           'Next-state probabilities: '+str(np.round(tt.next_probs[user_id],3).tolist())]
    for name in [bundle['stronger_paper'],'SVD-MF','TasteTrace-v2']:
        model=models[name]; recs=recommend(model,user_id)
        lines.append(f'\n## {name}')
        rank=next((r+1 for r,(i,_) in enumerate(recs) if i==target),None)
        lines.append(f'Held-out movie in top 10: {rank if rank else "absent"}')
        for r,(i,score) in enumerate(recs,1):
            lines.append(f'{r}. {tt.movies.at[i,"title"]} ({score:.4f})')
            if name=='TasteTrace-v2': lines.append('   '+explain(tt,user_id,i)['text'])
    return '\n\n'.join(lines)
