import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from numba import njit

class MatrixModel:
    def __init__(self,index,**params):
        self.index=index
        self.params=params

    def fit(self,train_data=None,val_data=None):
        return self

    def score(self,user_id,item_ids):
        return self.scores[self.index.umap[user_id], [self.index.imap[int(i)] for i in item_ids]]

class Popularity(MatrixModel):
    def score(self,user_id,item_ids):
        return self.index.popularity[[self.index.imap[int(i)] for i in item_ids]]

class ItemKNN(MatrixModel):
    def fit(self,train_data=None,val_data=None):
        x=normalize(self.index.matrix.T)
        self.similarity=(x@x.T).toarray()
        np.fill_diagonal(self.similarity,0)
        k=min(self.params.get('neighbors',50),len(self.similarity)-1)
        keep=np.argpartition(self.similarity,-k,axis=1)[:,-k:]
        limited=np.zeros_like(self.similarity)
        np.put_along_axis(limited,keep,np.take_along_axis(self.similarity,keep,axis=1),axis=1)
        self.similarity=limited
        self.scores=np.asarray(self.index.matrix@limited.T)
        return self

class UserKNN(MatrixModel):
    def fit(self,train_data=None,val_data=None):
        x=normalize(self.index.matrix)
        sim=(x@x.T).toarray()
        np.fill_diagonal(sim,0)
        k=min(self.params.get('neighbors',50),len(sim)-1)
        keep=np.argpartition(sim,-k,axis=1)[:,-k:]
        limited=np.zeros_like(sim)
        np.put_along_axis(limited,keep,np.take_along_axis(sim,keep,axis=1),axis=1)
        self.scores=np.asarray(limited@self.index.matrix)
        return self

class SVDMF(MatrixModel):
    def fit(self,train_data=None,val_data=None):
        k=min(self.params.get('dim',32),min(self.index.matrix.shape)-1)
        self.svd=TruncatedSVD(k,random_state=self.params.get('seed',42))
        self.user_factors=self.svd.fit_transform(self.index.matrix)
        self.item_factors=self.svd.components_.T
        self.scores=self.user_factors@self.item_factors.T
        return self

@njit(cache=True)
def _bpr(indptr,indices,nusers,nitems,dim,epochs,lr,reg,seed):
    np.random.seed(seed)
    p=np.random.normal(0,.1,(nusers,dim))
    q=np.random.normal(0,.1,(nitems,dim))
    for epoch in range(epochs):
        for step in range(len(indices)):
            u=np.random.randint(nusers)
            lo,hi=indptr[u],indptr[u+1]
            if hi==lo or hi-lo==nitems: continue
            i=indices[np.random.randint(lo,hi)]
            j=np.random.randint(nitems)
            while j in indices[lo:hi]: j=np.random.randint(nitems)
            diff=0.
            for d in range(dim): diff+=p[u,d]*(q[i,d]-q[j,d])
            grad=1/(1+np.exp(min(40.,diff)))
            for d in range(dim):
                pu,qi,qj=p[u,d],q[i,d],q[j,d]
                p[u,d]+=lr*(grad*(qi-qj)-reg*pu)
                q[i,d]+=lr*(grad*pu-reg*qi)
                q[j,d]+=lr*(-grad*pu-reg*qj)
    return p,q

class BPRMF(MatrixModel):
    def fit(self,train_data=None,val_data=None):
        x=self.index.matrix
        self.user_factors,self.item_factors=_bpr(x.indptr,x.indices,*x.shape,
            self.params.get('dim',32),self.params.get('epochs',30),self.params.get('lr',.03),
            self.params.get('reg',.002),self.params.get('seed',42))
        self.scores=self.user_factors@self.item_factors.T
        return self

class HistoryReplay(MatrixModel):
    """Deliberately invalid recommender for demonstrating the P0 protocol flaw."""
    def score(self,user_id,item_ids):
        return np.isin(item_ids,self.index.histories[user_id]).astype(float)
