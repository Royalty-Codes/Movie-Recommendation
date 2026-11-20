import numpy as np
from .data import GENRES

ATTRS = ['taste_cluster','primary_genre','era','popularity_tier','avg_rating_tier']

def item_features(train, movies, labels=None):
    """Only train supplies item counts and means; metadata is public catalogue data."""
    f = movies.copy()
    stats = train.groupby('item_id').rating.agg(['count','mean']).reindex(f.index)
    f['count'] = stats['count'].fillna(0).astype(int)
    f['mean'] = stats['mean'].fillna(train.rating.mean())
    genres = f[GENRES].to_numpy()
    # Eight interpretable genre families, with a deterministic priority.
    families = [['Crime','Thriller','Horror','Mystery','Film-Noir'], ['Action','Adventure','War','Western'],
                ['Sci-Fi','Fantasy'], ['Animation',"Children's"], ['Comedy','Musical'],
                ['Romance'], ['Drama'], ['Documentary','unknown']]
    counts = np.column_stack([genres[:,[GENRES.index(g) for g in fam]].sum(1) for fam in families])
    f['primary_genre'] = counts.argmax(1)
    f['era'] = np.digitize(f.year, [1970,1980,1990,1995])
    positive = f.loc[f['count']>0,'count']
    thresholds = positive.quantile([1/3,2/3]).to_numpy() if len(positive) else [0,0]
    f['popularity_tier'] = np.digitize(f['count'], thresholds, right=True)
    f['avg_rating_tier'] = np.digitize(f['mean'], [3,4])
    if labels is not None:
        f['taste_cluster'] = labels
    return f
