import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from .data import GENRES

def taste_clusters(index,movies,mf,count=12,seed=42,n_init=10):
    embedding=normalize(mf.item_factors)
    warm=index.popularity>0
    count=min(count,int(warm.sum()))
    km=KMeans(count,n_init=n_init,random_state=seed).fit(embedding[warm])
    labels=np.zeros(len(movies),int)
    labels[warm]=km.labels_
    genres=movies[GENRES].to_numpy(float)
    profiles=np.vstack([genres[(labels==c)&warm].mean(0) for c in range(count)])
    labels[~warm]=(normalize(genres[~warm])@normalize(profiles).T).argmax(1) if (~warm).any() else []
    global_genres=genres[warm].mean(0)+1e-6
    names={}
    for c in range(count):
        ids=np.flatnonzero((labels==c)&warm)
        lift=(genres[ids].mean(0)+1e-6)/global_genres
        top=np.argsort(-lift)[:3]
        # Select representative titles from the popular half of each cluster.
        popular=ids[index.popularity[ids]>=np.median(index.popularity[ids])]
        order=np.argsort(np.linalg.norm(embedding[popular]-km.cluster_centers_[c],axis=1))[:3]
        titles=movies.iloc[popular[order]].title.tolist()
        names[c]={'name':' / '.join(GENRES[j] for j in top),'representative_titles':titles,
                  'size':int(len(ids))}
    return labels,names
