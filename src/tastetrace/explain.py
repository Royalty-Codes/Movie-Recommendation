import numpy as np
from .fusion import COMPONENTS

def explain(model,user_id,item_id,history=None):
    idx=model.index
    h=idx.histories.get(user_id,[]) if history is None else np.asarray(history,int)
    j=idx.imap[item_id]
    components=model.components(user_id,[item_id],history)[0]
    contribution=components*model.weights
    positive=[int(k) for k in np.argsort(-contribution) if contribution[k]>0][:2]
    watched=[idx.imap[i] for i in h]
    evidence=[]
    if watched:
        sims=model.knn.similarity[j,watched]
        evidence=[int(idx.items[watched[k]]) for k in np.argsort(-sims,kind='stable')[:3] if sims[k]>0]
    if history is None:
        probs=model.next_probs[user_id]; state=int(model.posteriors[user_id].argmax())
    else:
        from .hmm import user_states
        _,post,nxt=user_states(model.hmm,idx,model.labels,{user_id:h})
        probs=nxt[user_id]; state=int(post[user_id].argmax())
    cluster=int(model.labels[j]); state_cluster=int(model.hmm.emissionprob_[state].argmax())
    name=model.names[state_cluster]['name']
    destination=int(model.dominant[cluster])
    titles=[model.movies.at[i,'title'] for i in evidence]
    text={
        'c_phase':f'Its attributes fit your predicted mixture of taste phases; your current inferred phase is {name} (state {state}), and the next-phase mixture assigns this cluster {(probs@model.hmm.emissionprob_)[cluster]:.1%} probability.',
        'c_recent':f'It matches the cluster, genre and era of your last {min(len(h),model.config["fusion"]["recent_window"])} watched movies.',
        'c_item':('It is similar to '+', '.join(titles)+'.') if titles else 'Its collaborative item similarity contributes to this score.',
        'c_mf':'The collaborative factor model finds it compatible with your viewing history.',
        'c_pop':f'It has {int(idx.popularity[j])} training-set viewers.',
        'c_trans':f'The inferred probability of moving next to its dominant phase (state {destination}) is {probs[destination]:.1%}.'}
    reasons=[text[COMPONENTS[k]] for k in positive]
    if not reasons: reasons=['This is the highest available relative fit; no component has a positive standardized contribution.']
    if titles and 'c_item' not in [COMPONENTS[k] for k in positive]:
        reasons.append('Related watched titles: '+', '.join(titles)+'.')
    return dict(user_id=int(user_id),item_id=int(item_id),title=model.movies.at[item_id,'title'],
        text=' '.join(reasons),components=dict(zip(COMPONENTS,map(float,components))),
        contributions=dict(zip(COMPONENTS,map(float,contribution))),cited_components=[COMPONENTS[k] for k in positive],
        evidence_items=evidence,state=state,next_phase_probabilities=probs.tolist(),
        bn_edges_used=sorted([list(e) for e in set(edge for bn in model.networks for edge in bn.edges())]) if 'c_phase' in [COMPONENTS[k] for k in positive] else [])
