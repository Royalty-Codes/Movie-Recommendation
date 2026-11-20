import numpy as np
from ..features import item_features
from ..clusters import taste_clusters
from ..hmm import fit_hmm,user_states
from ..bn import fit_state_bns
from ..fusion import standardize

class TasteTrace:
    def __init__(self,index,movies,mf,knn,config,seed=42,clusters=12,states=4,ess=10,labels=None,names=None,hmm=None,independent=False,static=False):
        self.index=index; self.movies=movies; self.mf=mf; self.knn=knn
        self.config=config; self.seed=seed; self.cluster_count=clusters; self.states=states; self.ess=ess
        self.labels=labels; self.names=names; self.hmm=hmm; self.independent=independent; self.static=static
        self.weights=np.array([1.,0,0,0,0,0]); self.mode='fusion'

    def fit(self,train_data,val_data=None):
        if self.labels is None:
            self.labels,self.names=taste_clusters(self.index,self.movies,self.mf,self.cluster_count,self.seed,self.config['clusters']['n_init'])
        self.features=item_features(train_data,self.movies,self.labels)
        if self.hmm is None:
            self.hmm,self.hmm_diagnostics=fit_hmm(self.index,self.labels,self.states,self.config['hmm'],self.seed)
        self.paths,self.posteriors,self.next_probs=user_states(self.hmm,self.index,self.labels)
        self.networks,self.joint,self.global_bn,self.global_joint=fit_state_bns(self.index,self.features,self.paths,
            self.hmm.n_components,self.config['bn'],self.ess,self.independent)
        self.dominant=self.hmm.emissionprob_.argmax(0)
        if self.static:
            self.joint[:]=self.global_joint
            prior=self.hmm.get_stationary_distribution()
            self.next_probs={u:prior for u in self.next_probs}
        return self

    def raw_components(self,user_id,history=None):
        index=self.index
        h=index.histories.get(user_id,[]) if history is None else np.asarray(history,int)
        probs=self.next_probs[user_id] if history is None else user_states(self.hmm,index,self.labels,{user_id:h})[2][user_id]
        phase=np.log(np.maximum(probs@self.joint,1e-300))
        recent_ids=[index.imap[i] for i in h[-self.config['fusion']['recent_window']:]]
        recent=np.zeros(len(index.items))
        for attr in ['taste_cluster','primary_genre','era']:
            values=self.features[attr].to_numpy(int)
            counts=np.bincount(values[recent_ids],minlength=int(values.max())+1)+1
            recent+=np.log(counts[values]/counts.sum())
        if history is None:
            item=self.knn.scores[index.umap[user_id]]
            mf=self.mf.scores[index.umap[user_id]]
        else:
            ids=[index.imap[i] for i in h]
            item=self.knn.similarity[:,ids].sum(1) if ids else np.zeros(len(index.items))
            # SVD fold-in for an ordered custom profile.
            factors=self.mf.item_factors
            profile=factors[ids].sum(0) if ids else np.zeros(factors.shape[1])
            mf=factors@profile
        return np.column_stack([phase,recent,item,mf,np.log1p(index.popularity),probs[self.dominant[self.labels]]])

    def components(self,user_id,item_ids,history=None):
        raw=self.raw_components(user_id,history)
        h=self.index.histories.get(user_id,[]) if history is None else history
        mask=~np.isin(self.index.items,h)
        z=standardize(raw,mask if mask.any() else None)
        return z[[self.index.imap[int(i)] for i in item_ids]]

    def score(self,user_id,item_ids):
        if self.mode=='hmm':
            return (self.next_probs[user_id]@self.hmm.emissionprob_)[self.labels[[self.index.imap[int(i)] for i in item_ids]]]
        if self.mode=='core':
            return self.raw_components(user_id)[[self.index.imap[int(i)] for i in item_ids],0]
        return self.components(user_id,item_ids)@self.weights
