import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from .features import ATTRS

def learn_network(table,domains,ess=10,max_indegree=2,independent=False):
    table=table.astype({a:pd.CategoricalDtype(categories=domains[a]) for a in ATTRS})
    if independent or len(table)<20:
        edges=[]
    else:
        structure=HillClimbSearch(table).estimate(scoring_method='bic-d',max_indegree=max_indegree,show_progress=False)
        edges=list(structure.edges())
    model=DiscreteBayesianNetwork(edges)
    model.add_nodes_from(ATTRS)
    cpds=BayesianEstimator(model,table,state_names=domains).get_parameters(prior_type='BDeu',equivalent_sample_size=ess)
    model.add_cpds(*cpds)
    if not model.check_model(): raise ValueError('Invalid fitted Bayesian network')
    return model

def joint_lookup(model,attributes):
    """Catalogue-vectorized CPD lookup, preserving each CPD's parent order."""
    probability=np.ones(len(attributes))
    for node in model.nodes():
        cpd=model.get_cpds(node)
        coordinates=[]
        for var in cpd.variables:
            mapping={value:i for i,value in enumerate(cpd.state_names[var])}
            coordinates.append(attributes[var].map(mapping).to_numpy(dtype=int))
        probability*=cpd.values[tuple(coordinates)]
    return np.maximum(probability,1e-300)

def fit_state_bns(index,features,paths,nstates,config,ess=10,independent=False):
    attrs=features[ATTRS].astype(int)
    domains={a:sorted(attrs[a].unique().tolist()) for a in ATTRS}
    all_items=[]; all_states=[]
    for u,h in index.histories.items():
        all_items.extend(index.imap[i] for i in h)
        all_states.extend(paths[u])
    table=attrs.iloc[all_items].reset_index(drop=True)
    state_array=np.asarray(all_states)
    global_bn=learn_network(table,domains,ess,config['max_indegree'],independent)
    global_joint=joint_lookup(global_bn,attrs)
    networks=[]; joint=[]
    for s in range(nstates):
        sub=table[state_array==s]
        bn=learn_network(sub,domains,ess,config['max_indegree'],independent) if len(sub) else global_bn
        if len(sub)<config['small_state']:
            weight=len(sub)/(len(sub)+config['shrinkage'])
            # Fit global CPDs on the same DAG so parent domains and array shapes agree.
            categorical=table.astype({a:pd.CategoricalDtype(categories=domains[a]) for a in ATTRS})
            global_cpds=BayesianEstimator(bn,categorical,state_names=domains).get_parameters(prior_type='BDeu',equivalent_sample_size=ess)
            for global_cpd in global_cpds:
                cpd=bn.get_cpds(global_cpd.variable)
                cpd.values=weight*cpd.values+(1-weight)*global_cpd.values
            bn.check_model()
        values=joint_lookup(bn,attrs)
        networks.append(bn); joint.append(values)
    return networks,np.array(joint),global_bn,global_joint
