import numpy as np
import pandas as pd
from tastetrace.data import Index,Dataset,GENRES
from tastetrace.features import item_features
from tastetrace.splits import leave_last_out
from tastetrace.eval.protocol import candidate_sets,evaluate
from tastetrace.eval.stats import paired_test,bootstrap_ci
from tastetrace.eval.metrics import rank_items

def fixture_data(nusers=50,nitems=160):
    rng=np.random.default_rng(7); rows=[]
    for u in range(1,nusers+1):
        for t,i in enumerate(rng.choice(np.arange(1,nitems+1),8,replace=False)):
            rows.append((u,int(i),4,t//2,len(rows)))
    ratings=pd.DataFrame(rows,columns=['user_id','item_id','rating','timestamp','row_order'])
    movies=pd.DataFrame({'title':[f'Movie {i}' for i in range(nitems)],'year':1995},index=np.arange(1,nitems+1))
    for genre in GENRES: movies[genre]=int(genre=='Drama')
    users=pd.DataFrame({'age':25,'gender':'M','occupation':'student','zip':'12345'},index=np.arange(1,nusers+1))
    return Dataset(ratings,movies,users)

def test_split_ties_and_no_leakage():
    d=fixture_data(); s=leave_last_out(d.ratings)
    assert set(s.train.row_order).isdisjoint(s.test.row_order)
    assert set(s.train.row_order).isdisjoint(s.val.row_order)
    assert len(s.train)+len(s.val)+len(s.test)==len(d.ratings)
    for u,g in d.ratings.groupby('user_id'):
        assert s.test[s.test.user_id==u].row_order.item()==g.row_order.iloc[-1]
        assert s.val[s.val.user_id==u].row_order.item()==g.row_order.iloc[-2]

def test_short_users_all_train():
    d=fixture_data(1); s=leave_last_out(d.ratings.iloc[:4])
    assert len(s.train)==4 and s.test.empty and s.val.empty

def test_training_features_do_not_use_heldout_values():
    d=fixture_data(); s=leave_last_out(d.ratings)
    before=item_features(s.train,d.movies)
    changed=d.ratings.copy(); changed.loc[changed.row_order.isin(s.test.row_order),'rating']=1
    after=item_features(leave_last_out(changed).train,d.movies)
    pd.testing.assert_frame_equal(before,after)
    counts=s.train.groupby('item_id').size().reindex(d.movies.index,fill_value=0)
    np.testing.assert_array_equal(before['count'],counts)

def test_candidates_deterministic_exclude_all_watched():
    d=fixture_data(); s=leave_last_out(d.ratings)
    a=candidate_sets(d.ratings,d.movies,s.test,[7,17]); b=candidate_sets(d.ratings,d.movies,s.test,[7,17])
    for key,(items,ties) in a.items():
        u=key[1]; watched=set(d.ratings[d.ratings.user_id==u].item_id)
        assert not (set(items[1:])&watched)
        assert len(items)==100 and len(set(items))==100
        np.testing.assert_array_equal(items,b[key][0]); np.testing.assert_array_equal(ties,b[key][1])

def test_oracle_random_and_tie_behavior():
    d=fixture_data(1000); s=leave_last_out(d.ratings); index=Index(s.train,d.movies,d.users)
    c=candidate_sets(d.ratings,d.movies,s.test,[7])
    targets=dict(zip(s.test.user_id,s.test.item_id))
    class Oracle:
        def score(self,u,items): return (items==targets[u]).astype(float)
    raw,_=evaluate(Oracle(),c,s.test,index.popularity,index.imap)
    assert raw['HR@10'].mean()==1 and raw['NDCG@10'].mean()==1
    class Ties:
        def score(self,u,items): return np.full(len(items),-np.inf)
    raw,_=evaluate(Ties(),c,s.test,index.popularity,index.imap)
    assert .07<raw['HR@10'].mean()<.13
    assert len(rank_items([0,-np.inf],[.5,.2]))==2

def test_paired_statistics():
    result=paired_test(np.ones(100),np.zeros(100),n=1000)
    assert result['ci_low']==1 and result['p_value']<.01
    assert bootstrap_ci(np.zeros(100),n=100)==(0,0)
