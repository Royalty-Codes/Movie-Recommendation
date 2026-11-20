import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import KFold

COMPONENTS=['c_phase','c_recent','c_item','c_mf','c_pop','c_trans']

def standardize(raw,mask=None):
    pool=raw if mask is None else raw[mask]
    return (raw-pool.mean(axis=0))/np.maximum(pool.std(axis=0),1e-8)

def fit_weights(differences,l2,active=None):
    active=np.ones(differences.shape[1],bool) if active is None else np.asarray(active,bool)
    x=differences[:,active]
    def objective(w):
        margin=x@w
        return np.logaddexp(0,-margin).mean()+l2*np.dot(w,w)/2, -x.T@expit(-margin)/len(x)+l2*w
    fit=minimize(objective,np.ones(x.shape[1])/max(1,x.shape[1]),jac=True,method='L-BFGS-B',bounds=[(0,None)]*x.shape[1])
    if not fit.success: raise RuntimeError(f'Fusion optimization failed: {fit.message}')
    weights=np.zeros(differences.shape[1]); weights[active]=fit.x
    return weights

def learn_fusion(components,validation_candidates,config,seed=42,active=None):
    # Each row-group is one user: CV never splits one user's pairs across folds.
    groups=[]
    for (_,u),(items,_) in validation_candidates.items():
        z=components(u,items)
        groups.append(z[0]-z[1:])
    splitter=KFold(min(config['cv_folds'],len(groups)),shuffle=True,random_state=seed)
    trials=[]
    for l2 in config['l2']:
        losses=[]
        for tr,va in splitter.split(groups):
            w=fit_weights(np.concatenate([groups[i] for i in tr]),l2,active)
            x=np.concatenate([groups[i] for i in va])
            losses.append(float(np.logaddexp(0,-x@w).mean()))
        trials.append({'l2':l2,'cv_loss':float(np.mean(losses))})
    chosen=min(trials,key=lambda r:r['cv_loss'])['l2']
    return fit_weights(np.concatenate(groups),chosen,active),trials
