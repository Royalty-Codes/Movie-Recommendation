"""Source-reviewed Keudjeu HMM reproduction. See docs/base_paper_assumptions.md."""
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist
from .baselines import MatrixModel
from ..data import GENRES, ordered

@njit(cache=True)
def feature_vector(symbols,ratings,n_symbols,iterations,power,epsilon):
    n=len(symbols); states=5
    pi=np.zeros(states)
    a=np.zeros((states,states))
    b=np.zeros((states,n_symbols))
    if n:
        pi[ratings[0]]+=1
        for t in range(n):
            b[ratings[t],symbols[t]]+=1
            if t: a[ratings[t-1],ratings[t]]+=1
    # Equations 1-4: divide counts by total+epsilon, then distribute missing mass.
    pi/=1+epsilon
    pi+=(1-pi.sum())/states
    for s in range(states):
        a[s]/=a[s].sum()+epsilon
        a[s]+=(1-a[s].sum())/states
        b[s]/=b[s].sum()+epsilon
        b[s]+=(1-b[s].sum())/n_symbols
    # Scaled forward-backward; tiny feature HMMs compiled with numba.
    for iteration in range(iterations if n>1 else 0):
        forward=np.zeros((n,states)); backward=np.ones((n,states)); scale=np.zeros(n)
        forward[0]=pi*b[:,symbols[0]]
        scale[0]=forward[0].sum(); forward[0]/=scale[0]
        for t in range(1,n):
            for j in range(states):
                for s in range(states): forward[t,j]+=forward[t-1,s]*a[s,j]
                forward[t,j]*=b[j,symbols[t]]
            scale[t]=max(forward[t].sum(),1e-300); forward[t]/=scale[t]
        for t in range(n-2,-1,-1):
            for s in range(states):
                backward[t,s]=0
                for j in range(states): backward[t,s]+=a[s,j]*b[j,symbols[t+1]]*backward[t+1,j]
                backward[t,s]/=scale[t+1]
        gamma=forward*backward
        for t in range(n): gamma[t]/=max(gamma[t].sum(),1e-300)
        anew=np.ones((states,states))*1e-8
        bnew=np.ones((states,n_symbols))*1e-8
        for t in range(n):
            for s in range(states): bnew[s,symbols[t]]+=gamma[t,s]
            if t<n-1:
                for s in range(states):
                    for j in range(states):
                        anew[s,j]+=forward[t,s]*a[s,j]*b[j,symbols[t+1]]*backward[t+1,j]/scale[t+1]
        pi=gamma[0]
        for s in range(states):
            a[s]=anew[s]/anew[s].sum(); b[s]=bnew[s]/bnew[s].sum()
    stationary=np.ones(states)/states
    for step in range(power): stationary=stationary@a
    return stationary@b

def frequency_bins(values,n):
    """Frequency-ranked category quantiles; deterministic lexical tie-break."""
    unique,counts=np.unique(values.astype(str),return_counts=True)
    order=sorted(range(len(unique)),key=lambda i:(-counts[i],unique[i]))
    mapping={unique[j]:min(n-1,rank*n//len(unique)) for rank,j in enumerate(order)}
    return np.array([mapping[str(v)] for v in values],int)

def threshold_frequency_bins(values,boundaries):
    unique,counts=np.unique(values.astype(str),return_counts=True)
    mapping={key:int(np.digitize(count,boundaries,right=True)) for key,count in zip(unique,counts)}
    return np.array([mapping[str(v)] for v in values],int)

def paper_symbols(train,movies,users):
    stats=train.groupby('item_id').rating.agg(['mean','count']).reindex(movies.index)
    years=np.digitize(movies.year,[1980,1986,1991,1996,2001,2006,2011,2016,2021])
    means=np.maximum(0,np.ceil(stats['mean'].fillna(train.rating.mean()).to_numpy()*2).astype(int)-1).clip(0,9)
    counts=np.digitize(stats['count'].fillna(0),[25,50,75,100,125,150,200,250,300],right=True)
    flags=movies[GENRES].to_numpy(int)
    genre_counts=flags.T@stats['count'].fillna(0).to_numpy()
    bins=3-np.digitize(genre_counts,[100,400,700])
    # Fixed paper thresholds may leave an empty class: use frequency ranks then.
    if len(np.unique(bins))<4:
        bins=np.empty(len(GENRES),int)
        bins[np.argsort(-genre_counts,kind='stable')]=np.minimum(3,np.arange(len(GENRES))*4//len(GENRES))
    membership=np.column_stack([flags[:,bins==c].any(1) for c in range(4)])
    word=membership@np.array([8,4,2,1]); word=np.maximum(word,1)-1
    item_symbols=np.column_stack([years,means,counts,word]).astype(int)
    age=np.digitize(users.age,[12,19,31,46,61,76])
    gender=(users.gender=='M').astype(int)
    user_symbols=np.column_stack([age,gender,threshold_frequency_bins(users.occupation.to_numpy(),[10,20,40,70,100,150]),
                                 threshold_frequency_bins(users['zip'].fillna('?').str[0].to_numpy(),[10,50,70,100,150])]).astype(int)
    return item_symbols,user_symbols

class KeudjeuHMM(MatrixModel):
    def fit(self,train_data,val_data=None):
        index=self.index; cfg=self.params['config']; window=self.params.get('window',60)
        item_symbols,user_symbols=paper_symbols(train_data,self.params['movies'],self.params['users'])
        self.user_vectors=np.tile(np.concatenate([np.full(m,1/m) for m in [10,10,10,15]]),(len(index.users),1))
        self.item_vectors=np.tile(np.concatenate([np.full(m,1/m) for m in [7,2,7,6]]),(len(index.items),1))
        tr=ordered(train_data)
        for uid,group in tr.groupby('user_id'):
            group=group.tail(window)
            ids=np.array([index.imap[i] for i in group.item_id])
            ratings=group.rating.to_numpy(dtype=int)-1
            self.user_vectors[index.umap[uid]]=np.concatenate([feature_vector(item_symbols[ids,f],ratings,m,
                cfg['iterations'],cfg['stationary_power'],cfg['epsilon']) for f,m in enumerate([10,10,10,15])])
        by_item=train_data.sort_values(['item_id','timestamp','row_order'],kind='stable')
        for item,group in by_item.groupby('item_id'):
            group=group.tail(window)
            ids=np.array([index.umap[u] for u in group.user_id]); ratings=group.rating.to_numpy(dtype=int)-1
            self.item_vectors[index.imap[item]]=np.concatenate([feature_vector(user_symbols[ids,f],ratings,m,
                cfg['iterations'],cfg['stationary_power'],cfg['epsilon']) for f,m in enumerate([7,2,7,6])])
        self.user_distances=cdist(self.user_vectors,self.user_vectors,metric='cityblock')
        np.fill_diagonal(self.user_distances,np.inf)
        self.item_distances=cdist(self.item_vectors,self.item_vectors,metric='cityblock')
        self.user_ratings={int(u):dict(zip(g.item_id,g.rating)) for u,g in tr.groupby('user_id')}
        self.set_neighbors(self.params.get('neighbors',50))
        return self

    def set_neighbors(self,k):
        idx=self.index
        neighbors=np.argsort(self.user_distances,axis=1,kind='stable')[:,:k]
        self.neighbor_count=np.asarray(idx.matrix[neighbors.ravel()].toarray().reshape(len(idx.users),k,-1).sum(1))
        scores=[]; fallback=[]
        eps=self.params['config']['distance_epsilon']
        for u in idx.users:
            h=idx.histories.get(u,[]); ids=np.array([idx.imap[i] for i in h],int)
            ratings=np.array([self.user_ratings[u][i] for i in h])
            inverse=1/(self.item_distances[:,ids]+eps)
            importance=inverse@ratings
            count=self.neighbor_count[idx.umap[u]]
            beta=np.log(np.maximum(count,1))+np.log(np.maximum(importance,eps))
            scores.append(np.where(count>0,beta,-np.inf))
            estimate=self.predict_ratings(int(u),idx.items)
            fallback.append(np.where(count>0,beta,estimate))
        self.scores=np.array(scores); self.fallback_scores=np.array(fallback)
        return self

    def predict_ratings(self,user_id,item_ids):
        """Equation 9: mean rating of closest history items, times distance penalty."""
        idx=self.index; h=idx.histories.get(user_id,[])
        if not len(h): return np.zeros(len(item_ids))
        columns=np.array([idx.imap[i] for i in h],int)
        rows=np.array([idx.imap[int(i)] for i in item_ids],int)
        distances=self.item_distances[np.ix_(rows,columns)]
        minimum=distances.min(1)
        nearest=distances==minimum[:,None]
        ratings=np.array([self.user_ratings[user_id][i] for i in h])
        average=(nearest@ratings)/nearest.sum(1)
        maximum=self.item_distances.max()
        return average*(1-minimum/maximum) if maximum>0 else average

    def score(self,user_id,item_ids):
        values=self.fallback_scores if self.params.get('fallback',False) else self.scores
        return values[self.index.umap[user_id],[self.index.imap[int(i)] for i in item_ids]]
