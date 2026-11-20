"""One-command, resumable experiment driver. Every exported number is computed."""
import argparse
import copy
import hashlib
import json
import logging
import importlib.metadata
import os
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from download_data import download
from tastetrace.data import load_data,Index,fingerprint
from tastetrace.splits import leave_last_out
from tastetrace.models.baselines import Popularity,ItemKNN,UserKNN,SVDMF,BPRMF,HistoryReplay
from tastetrace.models.base_paper import KeudjeuHMM
from tastetrace.models.tastetrace import TasteTrace
from tastetrace.clusters import taste_clusters
from tastetrace.hmm import fit_hmm
from tastetrace.features import item_features
from tastetrace.fusion import learn_fusion,COMPONENTS
from tastetrace.eval.protocol import candidate_sets,validation_hr,evaluate
from tastetrace.eval.stats import comparison_table,paired_test
from tastetrace.eval.faithfulness import assess
from tastetrace.runtime import Cache,digest,write_json

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(ROOT/'config.yaml'))
    p.add_argument('--smoke',action='store_true',help='Small integration run; outputs are explicitly separate from full results')
    p.add_argument('--force',action='store_true')
    p.add_argument('--no-refit',action='store_true')
    p.add_argument('--model',choices=['all','tastetrace_v2'],default='all',help='Compatibility selector; dependencies and comparisons still run')
    return p.parse_args()

def run(args):
    os.chdir(ROOT)
    cfg=yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        cfg['results_dir']='results/smoke'; cfg['figures_dir']='figures/smoke'
        cfg['seeds']=cfg['seeds'][:1]; cfg['model_seeds']=cfg['model_seeds'][:1]
        cfg['baselines'].update(svd_dims=[16,32],knn_neighbors=[20],bpr_dims=[16],bpr_epochs=[5])
        cfg['clusters'].update(embedding_dims=[16],counts=[8])
        cfg['hmm'].update(states=[3],restarts=1,iterations=8)
        cfg['bn']['ess']=[10]; cfg['base_paper'].update(windows=[20],neighbors=[10],iterations=5)
        cfg['evaluation'].update(bootstrap=500,permutations=500)
    out=Path(cfg['results_dir']); out.mkdir(parents=True,exist_ok=True)
    (out/'logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=out/'logs/library.log',level=logging.WARNING)
    download(cfg['dataset'],cfg['data_dir'])
    data=load_data(cfg['data_dir'],cfg['dataset'])
    if args.smoke:
        data.ratings=data.ratings[data.ratings.user_id.isin(data.users.index[:100])].copy()
        data.users=data.users.iloc[:100].copy()
    split=leave_last_out(data.ratings,cfg['evaluation']['min_interactions'])
    source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src/tastetrace').rglob('*.py')))+Path(__file__).read_bytes()).hexdigest()
    run_hash=digest({'config':cfg,'train':fingerprint(split.train),'data':fingerprint(data.ratings),
                     'movies':fingerprint(data.movies),'users':fingerprint(data.users),'source':source_hash,'refit':not args.no_refit})
    cache=Cache(out,run_hash,args.force)
    write_json(out/'run_manifest.json',dict(config_hash=run_hash,config=cfg,source_hash=source_hash,
        smoke=args.smoke,refit=not args.no_refit,train_events=len(split.train),validation_events=len(split.val),test_events=len(split.test),
        training_fingerprint=fingerprint(split.train),python=sys.version,
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','hmmlearn','pgmpy','streamlit','numba']}))
    for name,frame in [('train',split.train),('val',split.val),('test',split.test)]:
        frame[['user_id','item_id','row_order']].assign(config_hash=run_hash).to_csv(out/f'split_{name}.csv',index=False)
    idx=Index(split.train,data.movies,data.users)
    val_candidates=candidate_sets(data.ratings,data.movies,split.val,[cfg['seeds'][0]],cfg['fusion']['negatives'])
    k=cfg['evaluation']['k']; seed=cfg['model_seeds'][0]
    selection=[]
    def tune(name,cls,grid):
        fits=[]
        for params in grid:
            params=dict(params,seed=seed)
            model=cache.get(name,params,lambda:cls(idx,**params).fit(split.train,split.val))
            hr=validation_hr(model,val_candidates,split.val,k)
            selection.append(dict(method=name,parameters=json.dumps(params),validation_hr=hr,config_hash=run_hash))
            fits.append((hr,params,model))
        return max(fits,key=lambda a:a[0])[1:]
    svd_params,svd=tune('SVD-MF',SVDMF,[dict(dim=d) for d in cfg['baselines']['svd_dims']])
    item_params,knn=tune('ItemKNN',ItemKNN,[dict(neighbors=n) for n in cfg['baselines']['knn_neighbors']])
    user_params,_=tune('UserKNN',UserKNN,[dict(neighbors=n) for n in cfg['baselines']['knn_neighbors']])
    bpr_params,_=tune('BPR-MF',BPRMF,[dict(dim=d,epochs=e,lr=cfg['baselines']['bpr_lr'],reg=cfg['baselines']['bpr_reg'])
                                for d in cfg['baselines']['bpr_dims'] for e in cfg['baselines']['bpr_epochs']])
    # Select cluster embedding dimension by validation MF, then jointly select C and states.
    embed_candidates=[]
    for dim in cfg['clusters']['embedding_dims']:
        embedding=cache.get('embedding',{'dim':dim,'seed':seed},lambda:SVDMF(idx,dim=dim,seed=seed).fit(split.train))
        embed_candidates.append((validation_hr(embedding,val_candidates,split.val,k),dim,embedding))
    _,embed_dim,embedding=max(embed_candidates,key=lambda x:x[0])
    hmm_selection=[]; best=None
    for count in cfg['clusters']['counts']:
        labels,names=cache.get('clusters',{'dim':embed_dim,'count':count,'seed':seed},
            lambda:taste_clusters(idx,data.movies,embedding,count,seed,cfg['clusters']['n_init']))
        for states in cfg['hmm']['states']:
            hmm,diagnostics=cache.get('hmm',{'dim':embed_dim,'count':count,'states':states,'seed':seed},
                lambda:fit_hmm(idx,labels,states,cfg['hmm'],seed))
            for ess in cfg['bn']['ess']:
                model=cache.get('core',{'count':count,'states':states,'ess':ess,'seed':seed},
                    lambda:TasteTrace(idx,data.movies,svd,knn,cfg,seed,count,states,ess,labels,names,hmm).fit(split.train))
                model.mode='core'
                hr=validation_hr(model,val_candidates,split.val,k)
                ll=np.mean([np.log(max(1e-300,model.next_probs[row.user_id]@hmm.emissionprob_[:,labels[idx.imap[row.item_id]]])) for row in split.val.itertuples()])
                entry=dict(clusters=count,states=states,effective_states=hmm.n_components,ess=ess,validation_hr=hr,
                           validation_log_likelihood=ll,config_hash=run_hash)
                hmm_selection.append(entry)
                pd.DataFrame(hmm_selection).to_csv(out/'hmm_selection.csv',index=False)
                if best is None or hr>best[0]: best=(hr,count,states,ess,model,diagnostics)
    _,chosen_c,chosen_s,chosen_ess,train_tt,diagnostics=best
    train_tt.mode='fusion'
    weights,fusion_trials=learn_fusion(train_tt.components,val_candidates,cfg['fusion'],seed)
    train_tt.weights=weights
    sensitivity=[]
    for window in cfg['fusion']['history_windows']:
        variant=copy.copy(train_tt); variant.config=copy.deepcopy(cfg)
        variant.config['fusion']['recent_window']=window
        _,trials=learn_fusion(variant.components,val_candidates,cfg['fusion'],seed)
        sensitivity.append(dict(parameter='history_window',value=window,cv_loss=min(t['cv_loss'] for t in trials),config_hash=run_hash))
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv',index=False)
    write_json(out/'fusion_weights.json',dict(config_hash=run_hash,weights=dict(zip(COMPONENTS,map(float,weights))),cv=fusion_trials))
    write_json(out/'hmm_diagnostics.json',dict(config_hash=run_hash,restarts=diagnostics))
    pd.DataFrame(selection).to_csv(out/'baseline_selection.csv',index=False)
    # Tune faithful candidate-supported and fallback versions separately.
    paper_best={False:None,True:None}
    for window in cfg['base_paper']['windows']:
        paper=cache.get('paper_train',{'window':window},lambda:KeudjeuHMM(idx,config=cfg['base_paper'],movies=data.movies,users=data.users,window=window).fit(split.train))
        for neighbors in cfg['base_paper']['neighbors']:
            paper.set_neighbors(neighbors)
            for fallback in [False,True]:
                paper.params['fallback']=fallback
                hr=validation_hr(paper,val_candidates,split.val,k)
                selection.append(dict(method='KeudjeuHMM+fallback' if fallback else 'KeudjeuHMM',parameters=json.dumps(dict(window=window,neighbors=neighbors)),validation_hr=hr,config_hash=run_hash))
                if paper_best[fallback] is None or hr>paper_best[fallback][0]: paper_best[fallback]=(hr,window,neighbors)
    pd.DataFrame(selection).to_csv(out/'baseline_selection.csv',index=False)
    stronger=max(paper_best,key=lambda f:paper_best[f][0])
    final_train=pd.concat([split.train,split.val]) if not args.no_refit else split.train
    final_idx=Index(final_train,data.movies,data.users)
    sampled=candidate_sets(data.ratings,data.movies,split.test,cfg['seeds'],cfg['evaluation']['negatives'])
    full=candidate_sets(data.ratings,data.movies,split.test,[cfg['seeds'][0]],full=True)
    raw=[]; tops=[]; ablations=[]; bundles=[]
    for init_seed in cfg['model_seeds']:
        fitted={}
        for name,cls,params in [('Popularity',Popularity,{}),('HistoryReplay',HistoryReplay,{}),('ItemKNN',ItemKNN,item_params),('UserKNN',UserKNN,user_params),('SVD-MF',SVDMF,svd_params),('BPR-MF',BPRMF,bpr_params)]:
            params=dict(params,seed=init_seed)
            fitted[name]=cache.get('final_'+name,params,lambda cls=cls,params=params:cls(final_idx,**params).fit(final_train))
        # SVD for clusters is distinct from the tuned baseline if embedding dimensions differ.
        em=cache.get('final_embedding',{'seed':init_seed,'dim':embed_dim},lambda:SVDMF(final_idx,dim=embed_dim,seed=init_seed).fit(final_train))
        def build_final():
            labels,names=taste_clusters(final_idx,data.movies,em,chosen_c,init_seed,cfg['clusters']['n_init'])
            return TasteTrace(final_idx,data.movies,fitted['SVD-MF'],fitted['ItemKNN'],cfg,init_seed,chosen_c,chosen_s,chosen_ess,labels,names).fit(final_train)
        tt=cache.get('final_tt',{'seed':init_seed},build_final)
        # Fusion coefficients are frozen before final refit and test evaluation.
        tt.weights=weights.copy(); fitted['TasteTrace-v2']=tt
        for fallback,name in [(False,'KeudjeuHMM'),(True,'KeudjeuHMM+fallback')]:
            _,window,neighbors=paper_best[fallback]
            fitted[name]=cache.get('final_paper',{'window':window,'neighbors':neighbors,'fallback':fallback},
                lambda window=window,neighbors=neighbors,fallback=fallback:KeudjeuHMM(final_idx,config=cfg['base_paper'],movies=data.movies,users=data.users,window=window,neighbors=neighbors,fallback=fallback).fit(final_train))
        core=copy.copy(tt); core.mode='core'; fitted['TasteTrace-core']=core
        hmmonly=copy.copy(tt); hmmonly.mode='hmm'; fitted['HMM-only']=hmmonly
        for name,model in fitted.items():
            for protocol,cands in [('P1',sampled),('P2',full)]:
                r,t=evaluate(model,cands,split.test,final_idx.popularity,final_idx.imap,k,name,protocol,init_seed)
                raw.append(r); tops.append(t)
        # Each component ablation is refit on validation with exactly the same CV rule.
        for component in range(len(COMPONENTS)):
            active=np.ones(len(COMPONENTS),bool); active[component]=False
            w,_=learn_fusion(train_tt.components,val_candidates,cfg['fusion'],seed,active)
            ab=copy.copy(tt); ab.weights=w
            r,_=evaluate(ab,sampled,split.test,final_idx.popularity,final_idx.imap,k,'without_'+COMPONENTS[component],'P1',init_seed)
            ablations.append(r)
        for variant,flags in [('static',dict(static=True)),('independent_bn',dict(independent=True))]:
            tr_variant=cache.get('train_'+variant,{},lambda flags=flags:TasteTrace(idx,data.movies,svd,knn,cfg,seed,chosen_c,chosen_s,chosen_ess,train_tt.labels,train_tt.names,train_tt.hmm,**flags).fit(split.train))
            w,_=learn_fusion(tr_variant.components,val_candidates,cfg['fusion'],seed)
            final_variant=cache.get('final_'+variant,{'seed':init_seed},lambda flags=flags:TasteTrace(final_idx,data.movies,fitted['SVD-MF'],fitted['ItemKNN'],cfg,init_seed,chosen_c,chosen_s,chosen_ess,tt.labels,tt.names,tt.hmm,**flags).fit(final_train))
            final_variant.weights=w
            r,_=evaluate(final_variant,sampled,split.test,final_idx.popularity,final_idx.imap,k,variant,'P1',init_seed)
            ablations.append(r)
        if init_seed==cfg['model_seeds'][0]:
            r,_=evaluate(train_tt,sampled,split.test,idx.popularity,idx.imap,k,'no_refit','P1',init_seed)
            ablations.append(r)
        if init_seed==cfg['model_seeds'][0]:
            fitted['train_tt']=train_tt
            bundle=dict(models=fitted,split=split,data=data,config=cfg,config_hash=run_hash,stronger_paper='KeudjeuHMM+fallback' if stronger else 'KeudjeuHMM')
            with (out/'demo_bundle.pkl').open('wb') as f: pickle.dump(bundle,f,pickle.HIGHEST_PROTOCOL)
            faith,details,explanations=assess(tt,full,split.test,k,cfg['evaluation']['bootstrap'])
            faith.assign(config_hash=run_hash).to_csv(out/'faithfulness.csv',index=False)
            details.assign(config_hash=run_hash).to_csv(out/'faithfulness_details.csv',index=False)
            with (out/'explanations.jsonl').open('w',encoding='utf-8') as f:
                for e in explanations: f.write(json.dumps(dict(e,config_hash=run_hash))+'\n')
            write_json(out/'cluster_names.json',dict(config_hash=run_hash,clusters=tt.names))
            np.savez(out/'state_semantics.npz',emissions=tt.hmm.emissionprob_,transitions=tt.hmm.transmat_,config_hash=run_hash)
            write_json(out/'bn_structures.json',dict(config_hash=run_hash,edges={s:list(bn.edges()) for s,bn in enumerate(tt.networks)}))
            pd.DataFrame([dict(user_id=u,event=t,state=int(s),item_id=int(i),config_hash=run_hash) for u in list(tt.paths)[:3]
                         for t,(s,i) in enumerate(zip(tt.paths[u],final_idx.histories[u]))]).to_csv(out/'state_timelines.csv',index=False)
            start=time.perf_counter()
            ties=[]
            for u in final_idx.histories:
                values=tt.raw_components(u)[:,0]
                unseen=~np.isin(final_idx.items,final_idx.histories[u])
                top_values=np.sort(values[unseen])[-k:]
                ties.append(1-len(np.unique(top_values))/max(1,len(top_values)))
            write_json(out/'core_diagnostics.json',dict(config_hash=run_hash,bn_all_user_catalog_seconds=time.perf_counter()-start,
                mean_top10_duplicate_fraction=float(np.mean(ties)),ties_under_5_percent=bool(np.mean(ties)<.05)))
    raw=pd.concat(raw,ignore_index=True); top=pd.concat(tops,ignore_index=True)
    raw.assign(config_hash=run_hash).to_csv(out/'per_user_metrics.csv',index=False)
    comparison_table(raw,top,run_hash,cfg['evaluation']['bootstrap']).to_csv(out/'main_comparison.csv',index=False)
    pd.concat(ablations).assign(config_hash=run_hash).to_csv(out/'ablation_per_user.csv',index=False)
    ab=pd.concat(ablations).groupby(['method','user_id'])[['HR@10','NDCG@10']].mean().reset_index()
    ab.groupby('method')[['HR@10','NDCG@10']].mean().assign(config_hash=run_hash).to_csv(out/'ablation.csv')
    beyond=[]
    for (name,protocol,init,seed_),g in top.groupby(['method','protocol','init_seed','seed']):
        beyond.append(dict(method=name,protocol=protocol,init_seed=init,seed=seed_,coverage=len(set(i for items in g['items'] for i in items))/len(final_idx.items),mean_popularity=g.mean_popularity.mean(),config_hash=run_hash))
    pd.DataFrame(beyond).to_csv(out/'beyond_accuracy.csv',index=False)
    sig=[]
    for protocol in ['P1','P2']:
        for metric in ['HR@10','NDCG@10']:
            pivot=raw[raw.protocol==protocol].groupby(['user_id','method'])[metric].mean().unstack()
            for other in ['KeudjeuHMM','KeudjeuHMM+fallback','SVD-MF','BPR-MF','TasteTrace-core','HMM-only']:
                sig.append(dict(method='TasteTrace-v2',baseline=other,protocol=protocol,metric=metric,
                    **paired_test(pivot['TasteTrace-v2'],pivot[other],cfg['evaluation']['permutations']),config_hash=run_hash))
    significance=pd.DataFrame(sig); significance.to_csv(out/'significance.csv',index=False)
    ablation_tests=[]
    full_users=raw[(raw.method=='TasteTrace-v2')&(raw.protocol=='P1')].groupby('user_id')[['HR@10','NDCG@10']].mean()
    for name,g in pd.concat(ablations).groupby('method'):
        by_user=g.groupby('user_id')[['HR@10','NDCG@10']].mean()
        for metric in ['HR@10','NDCG@10']:
            common=full_users.index.intersection(by_user.index)
            ablation_tests.append(dict(ablation=name,metric=metric,**paired_test(full_users.loc[common,metric],by_user.loc[common,metric],cfg['evaluation']['permutations']),config_hash=run_hash))
    pd.DataFrame(ablation_tests).to_csv(out/'ablation_significance.csv',index=False)
    # Stratification is descriptive: no extra test-time tuning.
    lengths=split.train.groupby('user_id').size(); f=item_features(final_train,data.movies)
    grouped=raw.copy(); grouped['activity']=grouped.user_id.map(lambda u:'cold-ish (<20)' if lengths[u]<20 else 'heavy (>=20)')
    grouped['target_tier']=grouped.target.map(f.popularity_tier)
    subgroup=grouped.groupby(['method','protocol','activity','target_tier'])[['HR@10','NDCG@10']].agg(['mean','count'])
    subgroup.columns=['_'.join(c) for c in subgroup.columns]
    subgroup.assign(config_hash=run_hash).to_csv(out/'subgroups.csv')
    # P0 uses models fitted on the full dataset. Never re-use these in held-out evaluation.
    p0=run_p0(data,cfg,cache,paper_best,svd_params,run_hash)
    p0.to_csv(out/'paper_protocol.csv',index=False)
    protocol_rows=[]
    for name,p0_name in [('Popularity','Popularity'),('SVD-MF','SVD-MF'),('KeudjeuHMM',f'KeudjeuHMM T={paper_best[False][1]}'),('HistoryReplay','HistoryReplay')]:
        protocol_rows.append(dict(method=name,p0_precision_at10=p0[(p0.method==p0_name)&(p0.n==10)].precision.mean(),
            p1_hr_at10=raw[(raw.method==name)&(raw.protocol=='P1')]['HR@10'].mean(),config_hash=run_hash))
    pd.DataFrame(protocol_rows).to_csv(out/'protocol_comparison.csv',index=False)
    # v1 is preserved in the repository; evaluate it under the new fair split too.
    run_v1(data,split,final_train,final_idx,cfg,cache,sampled,run_hash,out)
    summary(out,cfg,significance,run_hash,stronger,args.smoke)
    pd.DataFrame(cache.timings).assign(config_hash=run_hash).to_csv(out/'timings.csv',index=False)
    from make_figures import make_figures
    make_figures(out,Path(cfg['figures_dir']))
    print(f'Finished. Read {out / "SUMMARY.md"}',flush=True)

def run_p0(data,cfg,cache,paper_best,svd_params,run_hash):
    idx=Index(data.ratings,data.movies,data.users)
    models={'Popularity':Popularity(idx),'SVD-MF':SVDMF(idx,**svd_params).fit(data.ratings),'HistoryReplay':HistoryReplay(idx)}
    rows=[]
    for window in cfg['base_paper']['windows']:
        paper=cache.get('paper_p0',{'window':window},lambda:KeudjeuHMM(idx,config=cfg['base_paper'],movies=data.movies,users=data.users,window=window,neighbors=max(cfg['base_paper']['neighbors'])).fit(data.ratings))
        models[f'KeudjeuHMM T={window}']=paper
    for name,model in models.items():
        for n in [10,20]:
            for u,h in idx.histories.items():
                ids=idx.items
                if name.startswith('Keudjeu'):
                    ids=ids[model.neighbor_count[idx.umap[u]]>0]
                scores=model.score(u,ids)
                rng=np.random.default_rng(int(u))
                order=np.lexsort((rng.random(len(ids)),-scores))[:n]
                hit=np.isin(ids[order],h).astype(int)
                dcg=(hit/np.log2(np.arange(len(hit))+2)).sum()
                ideal=(1/np.log2(np.arange(min(n,len(h)))+2)).sum()
                rows.append(dict(method=name,user_id=u,n=n,precision=hit.sum()/n,recall=hit.sum()/len(h),ndcg=dcg/max(ideal,1e-12),config_hash=run_hash))
    return pd.DataFrame(rows)

def run_v1(data,split,final_train,index,cfg,cache,candidates,run_hash,out):
    if cfg['dataset']!='ml-100k': return
    from tastetrace.models.legacy import LegacyV1
    model=cache.get('legacy_v1',{},lambda:LegacyV1(cfg).fit(final_train))
    r,_=evaluate(model,candidates,split.test,index.popularity,index.imap,method='TasteTrace-v1',k=cfg['evaluation']['k'])
    r.assign(config_hash=run_hash).to_csv(out/'legacy_v1.csv',index=False)
    tr_model=cache.get('legacy_v1_train',{},lambda:LegacyV1(cfg).fit(split.train))
    vc=candidate_sets(data.ratings,data.movies,split.val,[cfg['seeds'][0]],cfg['evaluation']['negatives'])
    val_hr=validation_hr(tr_model,vc,split.val,cfg['evaluation']['k'])
    selected=pd.read_csv(out/'hmm_selection.csv')
    write_json(out/'v1_validation_comparison.json',dict(config_hash=run_hash,v1_core_validation_hr=val_hr,
        v2_core_validation_hr=float(selected.validation_hr.max()),improved=bool(selected.validation_hr.max()>val_hr),
        historical_number_reproduction='Not verified: original 5-seed benchmark script and exact candidate pools were not supplied. New val/test split differs.'))

def summary(out,cfg,significance,run_hash,stronger,smoke):
    main=pd.read_csv(out/'main_comparison.csv'); faith=pd.read_csv(out/'faithfulness.csv').set_index('metric')
    paper='KeudjeuHMM+fallback' if stronger else 'KeudjeuHMM'
    checks=significance[(significance.baseline==paper)&(significance.protocol=='P1')]
    t1=bool(((checks.difference>0)&(checks.p_value<.01)).all())
    svd=significance[(significance.baseline=='SVD-MF')&(significance.protocol=='P1')]
    t2=bool((svd.ci_high>=0).all())
    t3=bool(faith.loc['faithful','ci_low']>.5)
    lines=['# TasteTrace results',f'Config hash: `{run_hash}`',
        '**SMOKE RUN: integration evidence only; not the final experiment.**' if smoke else 'Full configured experiment.',
        f'T1 — beats validation-selected base-paper variant ({paper}), both P1 metrics with p < .01: **{t1}**.',
        f'T2 — difference CI versus SVD includes zero or favors TasteTrace on both P1 metrics: **{t2}**.',
        f'T3 — faithfulness rate lower 95% bound > .5: **{t3}**.',
        'T4 — cached Streamlit and CLI demo, comparison figures: generated; see README for commands.',
        'Core acceptance diagnostics: '+(out/'core_diagnostics.json').read_text(),
        '\n## Held-out comparison (computed)',main[main.metric.isin(['HR@10','NDCG@10'])].to_string(index=False),
        '\n## Interpretation','No superiority claim is supported unless its paired difference and permutation test pass. '
        'A zero or harmful ablation delta means that component has not demonstrated added accuracy. '
        'This is one dataset, with one held-out positive per user; Recall@10 therefore equals HR@10. '
        'P0 overlap precision is not comparable to P1 hit rate. Source equations were reviewed against the author-posted paper; remaining discretization choices are documented. '
        'Deletion faithfulness is expected to favor components selected for large positive contributions; it establishes arithmetic consistency, not human usefulness. '
        'Sufficiency is a ratio and can exceed one when the reduced score happens to rank held-out positives better.',
        '\n## Missing external evidence','Human study not conducted; no responses fabricated. MovieLens-1M, SASRec and non-linear ranker are stretch work, not included in this run.',
        '\n## Evidence','main_comparison.csv; significance.csv; ablation.csv; faithfulness.csv; paper_protocol.csv; run_manifest.json.']
    (out/'SUMMARY.md').write_text('\n\n'.join(lines),encoding='utf-8')
    paper_scores=pd.read_csv(out/'paper_protocol.csv')
    paper_mean=paper_scores[(paper_scores.method.str.startswith('Keudjeu'))&(paper_scores.n==10)].groupby('method').precision.mean()
    report='# Reproduction report\n\nComputed P0 precision@10 by history window:\n\n'+paper_mean.to_string()+'\n\n'
    if (abs(paper_mean-.997)>.1).any():
        report+='The >0.1 reproduction gap acceptance criterion FAILED. Source equations have been reviewed, but headline paper numbers are NOT reproduced. Do not claim otherwise; see documented data/discretization assumptions.\n'
    else:
        report+='P0 overlap precision is in the plan\'s expected ballpark; this alone does not verify all source equations.\n'
    report+='\nSource review: https://github.com/HmmsMovieRecoms/HMMs-Movie-Recommendation and the author-posted paper linked in docs/base_paper_assumptions.md. The plan URL omitted a hyphen. Equations 1-4 and 9, plus occupation/zip thresholds, were corrected from that paper.\n'
    report+='\nFull per-user evidence: '+str(out/'paper_protocol.csv')+'\nNo exact RMSE/MAE headline reproduction claimed.\n'
    (out/'reproduction_report.md').write_text(report,encoding='utf-8')
    if not smoke:
        Path('docs/reproduction_report.md').write_text(report,encoding='utf-8')

if __name__=='__main__':
    args=parse_args()
    with threadpool_limits(limits=4): run(args)
