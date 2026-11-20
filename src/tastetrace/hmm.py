import numpy as np
from hmmlearn.hmm import CategoricalHMM

def fit_hmm(index,labels,states,config,seed=42):
    sequences=[labels[[index.imap[i] for i in h]] for h in index.histories.values() if len(h)]
    lengths=[len(s) for s in sequences]
    x=np.concatenate(sequences).reshape(-1,1)
    best=None
    diagnostics=[]
    # Lower state counts are tried if all requested fits contain a collapsed state.
    for n in range(states,1,-1):
        candidates=[]
        for restart in range(config['restarts']):
            model=CategoricalHMM(n_components=n,n_features=int(labels.max())+1,
                n_iter=config['iterations'],random_state=seed+restart,implementation='scaling')
            model.fit(x,lengths)
            ll=model.score(x,lengths)
            decoded=model.predict(x,lengths)
            occupancy=np.bincount(decoded,minlength=n)/len(decoded)
            valid=bool(occupancy.min()>=config['min_occupancy'])
            diagnostics.append(dict(requested_states=states,states=n,restart=restart,
                                    log_likelihood=float(ll),min_occupancy=float(occupancy.min()),valid=valid))
            if valid: candidates.append((ll,model))
        if candidates:
            best=max(candidates,key=lambda v:v[0])[1]
            break
    if best is None:
        raise RuntimeError('No non-degenerate HMM fit; inspect state occupancy diagnostics')
    return best,diagnostics

def user_states(model,index,labels,history=None):
    paths,posteriors,next_probs={},{},{}
    for u,h in (index.histories if history is None else history).items():
        x=labels[[index.imap[i] for i in h]].reshape(-1,1)
        if len(x):
            paths[u]=model.predict(x)
            posteriors[u]=model.predict_proba(x)[-1]
        else:
            paths[u]=np.array([],int)
            posteriors[u]=model.startprob_
        next_probs[u]=posteriors[u]@model.transmat_
    return paths,posteriors,next_probs
