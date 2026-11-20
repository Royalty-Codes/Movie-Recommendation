"""Compatibility adapter preserving the supplied v1 implementation."""
import numpy as np
from data_utils import load_movies,add_popularity_tier,build_user_sequences
from hmm_model import train_taste_hmm,decode_all_users
from bn_model import learn_all_state_bns
from recommend import TasteTraceRecommender

class LegacyV1:
    def __init__(self,config): self.config=config
    def fit(self,train_data,val_data=None):
        tr=train_data.rename(columns={'item_id':'movie_id'})
        movies=add_popularity_tier(tr,load_movies())
        seq=build_user_sequences(tr,movies)
        hmm=train_taste_hmm(seq,n_states=4,random_state=self.config['model_seeds'][0],
            n_iter=self.config['hmm']['iterations'],n_restarts=self.config['hmm']['restarts'])
        paths=decode_all_users(hmm,seq)
        bns,tables=learn_all_state_bns(seq,paths,movies)
        self.rec=TasteTraceRecommender(hmm,bns,tables,movies,seq,paths)
        return self
    def score(self,user_id,item_ids):
        return np.array([self.rec.score_movie(user_id,int(i)) for i in item_ids])
