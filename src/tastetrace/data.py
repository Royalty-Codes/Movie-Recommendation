from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

GENRES = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

@dataclass
class Dataset:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame

def load_data(root='data', dataset='ml-100k'):
    root = Path(root) / dataset
    if dataset == 'ml-100k':
        ratings = pd.read_csv(root/'u.data', sep='\t', names=['user_id','item_id','rating','timestamp'])
        movies = pd.read_csv(root/'u.item', sep='|', encoding='latin-1',
            names=['item_id','title','release_date','video_date','url']+GENRES).set_index('item_id')
        movies['year'] = pd.to_datetime(movies.release_date, format='%d-%b-%Y', errors='coerce').dt.year.fillna(0).astype(int)
        users = pd.read_csv(root/'u.user', sep='|', names=['user_id','age','gender','occupation','zip'], dtype={'zip':str}).set_index('user_id')
    elif dataset == 'ml-1m':
        ratings = pd.read_csv(root/'ratings.dat', sep='::', engine='python', names=['user_id','item_id','rating','timestamp'])
        movies = pd.read_csv(root/'movies.dat', sep='::', engine='python', encoding='latin-1', names=['item_id','title','genres']).set_index('item_id')
        movies['year'] = movies.title.str.extract(r'\((\d{4})\)')[0].fillna(0).astype(int)
        for genre in GENRES:
            movies[genre] = movies.genres.str.split('|').apply(lambda gs: int(genre in gs))
        users = pd.read_csv(root/'users.dat', sep='::', engine='python', names=['user_id','gender','age','occupation','zip'], dtype={'zip':str}).set_index('user_id')
    else:
        raise ValueError('Supported datasets: ml-100k, ml-1m')
    ratings['row_order'] = np.arange(len(ratings))
    return Dataset(ratings, movies.sort_index(), users.sort_index())

def ordered(frame):
    return frame.sort_values(['user_id','timestamp','row_order'], kind='stable')

def histories(frame):
    return {int(u): g.item_id.to_numpy(dtype=int) for u,g in ordered(frame).groupby('user_id')}

class Index:
    def __init__(self, train, movies, users):
        self.items = movies.index.to_numpy(dtype=int)
        self.users = users.index.to_numpy(dtype=int)
        self.imap = {int(i): j for j,i in enumerate(self.items)}
        self.umap = {int(u): j for j,u in enumerate(self.users)}
        self.histories = histories(train)
        rows = train.user_id.map(self.umap).to_numpy()
        cols = train.item_id.map(self.imap).to_numpy()
        self.matrix = csr_matrix((np.ones(len(train)), (rows,cols)), shape=(len(self.users),len(self.items)))
        self.matrix.data[:] = 1
        self.popularity = np.asarray(self.matrix.sum(axis=0)).ravel()

def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
