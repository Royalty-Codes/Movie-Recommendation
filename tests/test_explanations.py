import copy
import yaml
import numpy as np
from pathlib import Path
from tastetrace.data import Index
from tastetrace.splits import leave_last_out
from tastetrace.models.baselines import ItemKNN,SVDMF
from tastetrace.models.tastetrace import TasteTrace
from tastetrace.explain import explain
from test_protocol import fixture_data

def test_explanations_are_grounded_deterministic_and_change_with_profile():
    d=fixture_data(30,80); s=leave_last_out(d.ratings)
    idx=Index(s.train,d.movies,d.users)
    config=yaml.safe_load(Path('config.yaml').read_text())
    config['hmm'].update(iterations=5,restarts=1,min_occupancy=0)
    mf=SVDMF(idx,dim=4).fit(s.train); knn=ItemKNN(idx,neighbors=10).fit(s.train)
    model=TasteTrace(idx,d.movies,mf,knn,config,clusters=4,states=2).fit(s.train)
    model.weights=np.array([.1,.2,.3,.4,.1,.2])
    ex=explain(model,1,1)
    assert ex==explain(model,1,1)
    assert set(ex['evidence_items'])<=set(idx.histories[1])
    assert '{' not in ex['text'] and '}' not in ex['text']
    assert np.isclose(sum(ex['contributions'].values()),model.score(1,[1])[0])
    custom=list(idx.histories[1])[:2]
    profile_ex=explain(model,1,1,custom)
    assert set(profile_ex['evidence_items'])<=set(custom)
    assert profile_ex['components']!=ex['components']
