import numpy as np
import pandas as pd
from tastetrace.models.baselines import ItemKNN,UserKNN,SVDMF,BPRMF,Popularity
from tastetrace.models.base_paper import feature_vector,threshold_frequency_bins,KeudjeuHMM
from tastetrace.data import Index
from tastetrace.bn import learn_network,joint_lookup
from tastetrace.features import ATTRS
from tastetrace.fusion import fit_weights
from test_protocol import fixture_data

def test_baselines_common_interface():
    d=fixture_data(12,60); idx=Index(d.ratings,d.movies,d.users)
    for cls in [ItemKNN,UserKNN,SVDMF,BPRMF,Popularity]:
        model=cls(idx,dim=4,epochs=1,neighbors=3).fit(d.ratings,None)
        scores=model.score(1,[1,3,9]); assert scores.shape==(3,) and np.isfinite(scores).all()

def test_tiny_hmm_probability_vector():
    vec=feature_vector(np.array([0,1,0,2]),np.array([3,4,3,2]),3,5,100,1.)
    assert np.isfinite(vec).all() and (vec>=0).all()
    assert np.isclose(vec.sum(),1)

def test_paper_frequency_thresholds_and_equation9():
    values=np.array(['a']*10+['b']*11+['c']*41)
    bins=threshold_frequency_bins(values,[10,20,40,70,100,150])
    assert bins[0]==0 and bins[10]==1 and bins[-1]==3
    d=fixture_data(4,20); idx=Index(d.ratings,d.movies,d.users)
    model=KeudjeuHMM(idx)
    # A constructed catalogue-distance matrix isolates Eq.9 from model fitting.
    model.item_distances=np.abs(np.arange(20)[:,None]-np.arange(20)[None,:]).astype(float)
    model.user_ratings={u:dict(zip(g.item_id,g.rating)) for u,g in d.ratings.groupby('user_id')}
    item=int(idx.histories[1][0])
    assert model.predict_ratings(1,[item])[0]==4

def test_bn_lookup_with_unobserved_state_values():
    table=pd.DataFrame({a:[0,0,1,1]*8 for a in ATTRS})
    domains={a:[0,1,2] for a in ATTRS}
    bn=learn_network(table,domains,ess=10)
    catalog=pd.DataFrame({a:[0,1,2] for a in ATTRS})
    scores=joint_lookup(bn,catalog)
    assert (scores>0).all() and np.isfinite(scores).all()
    assert bn.check_model()

def test_fusion_constraint_and_ablation():
    x=np.column_stack([np.ones(50),-np.ones(50)])
    w=fit_weights(x,.01)
    assert w[0]>0 and w[1]==0
    ab=fit_weights(x,.01,[False,True]); assert ab[0]==0
