import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

def make_figures(results=Path('results'),figures=Path('figures')):
    results=Path(results); figures=Path(figures); figures.mkdir(parents=True,exist_ok=True)
    main=pd.read_csv(results/'main_comparison.csv')
    plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
    def save(fig,name):
        fig.tight_layout(); fig.savefig(figures/f'{name}.png',bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for row,protocol in enumerate(['P1','P2']):
        for col,metric in enumerate(['HR@10','NDCG@10']):
            ax=axes[row,col]; g=main[(main.protocol==protocol)&(main.metric==metric)].sort_values('mean')
            colors=['#df6c3b' if n=='TasteTrace-v2' else '#6f529c' if n.startswith('Keudjeu') else '#80a7a0' for n in g.method]
            errors=np.vstack([g['mean']-g.ci_low,g.ci_high-g['mean']]).clip(0)
            ax.barh(g.method,g['mean'],xerr=errors,color=colors,capsize=2)
            ax.set_title(f'{protocol}: {metric} (user-bootstrap 95% CI)'); ax.set_xlim(0,max(.1,g.ci_high.max()*1.1))
    save(fig,'hero_comparison')
    p0=pd.read_csv(results/'paper_protocol.csv')
    g=p0[p0.n==10].groupby('method').precision.mean()
    selection_baselines=pd.read_csv(results/'baseline_selection.csv')
    paper_choice=selection_baselines[selection_baselines.method=='KeudjeuHMM'].sort_values('validation_hr',ascending=False,kind='stable').iloc[0]
    paper='KeudjeuHMM T='+str(json.loads(paper_choice.parameters)['window'])
    names=['Popularity','SVD-MF',paper,'HistoryReplay']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(['Popularity','SVD-MF','KeudjeuHMM','History replay'],[g[n] for n in names],color='#6f529c')
    axes[0].set(title='P0: precision@10 against own history',ylim=(0,1.05))
    p1=main[(main.protocol=='P1')&(main.metric=='HR@10')].set_index('method')['mean']
    names_p1=['Popularity','SVD-MF','KeudjeuHMM']+(['HistoryReplay'] if 'HistoryReplay' in p1 else [])
    axes[1].bar(names_p1,[p1[n] for n in names_p1],color='#80a7a0')
    axes[1].set(title='P1: held-out HR@10 (different metric)',ylim=(0,1.05))
    fig.suptitle('History overlap and held-out prediction answer different questions')
    save(fig,'protocol_flaw')
    ab=pd.read_csv(results/'ablation.csv').set_index('method')
    base=float(p1['TasteTrace-v2'])
    fig,ax=plt.subplots(figsize=(9,4)); ax.barh(ab.index,base-ab['HR@10'],color='#df6c3b'); ax.axvline(0,color='black',lw=.8)
    ax.set_title('Ablations: full model HR@10 minus refitted ablation HR@10'); save(fig,'ablations')
    weights=json.loads((results/'fusion_weights.json').read_text())['weights']
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(weights.keys(),weights.values(),color='#df6c3b'); ax.set_title('Nonnegative fusion weights (validation fit)'); save(fig,'fusion_weights')
    semantics=np.load(results/'state_semantics.npz')
    fig,ax=plt.subplots(figsize=(10,4)); im=ax.imshow(semantics['emissions'],aspect='auto',cmap='YlGnBu'); fig.colorbar(im,ax=ax,label='Emission probability')
    ax.set(xlabel='Collaborative taste cluster',ylabel='Hidden phase',title='HMM state semantics'); save(fig,'state_semantics')
    timelines=pd.read_csv(results/'state_timelines.csv')
    fig,axes=plt.subplots(3,1,figsize=(12,5),squeeze=False)
    for ax,(u,g) in zip(axes.ravel(),timelines.groupby('user_id')):
        ax.scatter(g.event,[u]*len(g),c=g.state,cmap='tab10',vmin=0,vmax=9,s=12); ax.set(ylabel=f'User {u}',yticks=[])
    axes[-1,0].set_xlabel('Chronological interaction'); save(fig,'state_timelines')
    structures=json.loads((results/'bn_structures.json').read_text())['edges']
    fig,axes=plt.subplots(1,len(structures),figsize=(5*len(structures),4),squeeze=False)
    for ax,(state,edges) in zip(axes.ravel(),structures.items()):
        graph=nx.DiGraph(); graph.add_nodes_from(['taste_cluster','primary_genre','era','popularity_tier','avg_rating_tier']); graph.add_edges_from(edges)
        nx.draw_networkx(graph,pos=nx.circular_layout(graph),ax=ax,node_color='#d6e8e3',node_size=2300,font_size=7,arrows=True)
        ax.set_title(f'Phase {state}'); ax.axis('off')
    save(fig,'bn_structures')
    faith=pd.read_csv(results/'faithfulness.csv').set_index('metric')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    g=faith.loc[['rank_drop','control_rank_drop']]
    axes[0].bar(['Cited component','Random other'],g['mean'],yerr=np.vstack([g['mean']-g.ci_low,g.ci_high-g['mean']]).clip(0),capsize=3,color=['#df6c3b','#80a7a0'])
    axes[0].set_title('Mean item rank drop after deletion')
    r=faith.loc['faithful']; axes[1].bar(['Cited hurts more'],[r['mean']],yerr=[[max(0,r['mean']-r.ci_low)],[max(0,r.ci_high-r['mean'])]],capsize=3,color='#df6c3b')
    axes[1].axhline(.5,ls='--',color='grey'); axes[1].set(title='Faithfulness win rate',ylim=(0,1)); save(fig,'faithfulness')
    selection=pd.read_csv(results/'hmm_selection.csv')
    fig,ax=plt.subplots(figsize=(8,4))
    for c,g in selection.groupby('clusters'):
        best=g.groupby('states').validation_hr.max(); ax.plot(best.index,best.values,marker='o',label=f'C={c}')
    ax.legend(); ax.set(xlabel='Requested HMM states',ylabel='Validation HR@10',title='Validation model selection'); save(fig,'hmm_selection')
    capabilities=pd.DataFrame([
        ['KeudjeuHMM','Yes','Yes','No','P0 original; P1/P2 here'],
        ['SVD-MF','No','Yes','No','P1/P2'],['TasteTrace-v2','Yes','Yes','Yes','P1/P2']],
        columns=['Method','Sequential','Collaborative','Explanation','Evaluation'])
    run_hash=json.loads((results/'run_manifest.json').read_text())['config_hash']
    capabilities.assign(config_hash=run_hash).to_csv(results/'capabilities.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,2)); ax.axis('off'); table=ax.table(cellText=capabilities.values,colLabels=capabilities.columns,loc='center'); table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.8); save(fig,'capabilities')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,column in zip(axes,['clusters','states','ess']):
        g=selection.groupby(column).validation_hr.max()
        ax.plot(g.index,g.values,marker='o'); ax.set(xlabel=column,ylabel='Best validation HR@10')
    save(fig,'sensitivity')
    if (results/'sensitivity.csv').exists():
        g=pd.read_csv(results/'sensitivity.csv')
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(g.value,g.cv_loss,marker='o'); ax.set(xlabel='Recent history window',ylabel='Cross-validated validation loss'); save(fig,'history_sensitivity')
    write_readable_summary(results,main)
    from verify_artifacts import verify
    verify(results)

def write_readable_summary(results,main):
    manifest=json.loads((results/'run_manifest.json').read_text())
    sig=pd.read_csv(results/'significance.csv')
    selection=pd.read_csv(results/'baseline_selection.csv')
    paper_rows=selection[selection.method.isin(['KeudjeuHMM','KeudjeuHMM+fallback'])]
    paper=paper_rows.sort_values('validation_hr',ascending=False,kind='stable').iloc[0].method
    faith=pd.read_csv(results/'faithfulness.csv').set_index('metric')
    diagnostics=json.loads((results/'core_diagnostics.json').read_text())
    ab_tests=pd.read_csv(results/'ablation_significance.csv')
    hsel=pd.read_csv(results/'hmm_selection.csv').sort_values('validation_hr',ascending=False,kind='stable').iloc[0]
    t1=sig[(sig.baseline==paper)&(sig.protocol=='P1')]
    t1=bool(((t1.difference>0)&(t1.p_value<.01)).all())
    t2=bool((sig[(sig.baseline=='SVD-MF')&(sig.protocol=='P1')].ci_high>=0).all())
    t3=bool(faith.loc['faithful','ci_low']>.5)
    def supported(value): return 'MET' if value else 'NOT MET'
    lines=['# TasteTrace experiment results',f'Run hash: `{manifest["config_hash"]}`. Dataset: {manifest["config"]["dataset"]}.',
        '**SMOKE: integration evidence only; not final research results.**' if manifest['smoke'] else
        f'{len(manifest["config"]["seeds"])} candidate seeds × {len(manifest["config"]["model_seeds"])} model initialization seeds; {manifest["test_events"]} evaluated users. P2 uses the complete candidate catalogue once per initialization.',
        '## Held-out accuracy','Mean with 95% user-bootstrap interval. All methods receive the same chronological history and candidate sets.',
        '| Method | P1 HR@10 | P1 NDCG@10 | P2 HR@10 | P2 NDCG@10 |','|---|---:|---:|---:|---:|']
    for method in ['TasteTrace-v2',paper,'SVD-MF','BPR-MF','ItemKNN','UserKNN','Popularity','TasteTrace-core','HMM-only','HistoryReplay']:
        cells=[]
        for protocol,metric in [('P1','HR@10'),('P1','NDCG@10'),('P2','HR@10'),('P2','NDCG@10')]:
            g=main[(main.method==method)&(main.protocol==protocol)&(main.metric==metric)]
            if len(g):
                r=g.iloc[0]; cells.append(f'{r["mean"]:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]')
            else: cells.append('—')
        lines.append('| '+method+' | '+' | '.join(cells)+' |')
    lines+=['## Targets',f'- T1: {supported(t1)} — both P1 metrics exceed the validation-selected paper variant ({paper}) with paired permutation p < .01.',
        f'- T2: {supported(t2)} — the paired difference versus SVD on P1 is favorable or includes zero on both metrics; this is not an equivalence test.',
        f'- T3: {supported(t3)} — cited-component deletion win-rate lower 95% bound exceeds .5.',
        '- T4: cached Streamlit and CLI demos, structured explanations and comparison figures are generated; see README for launch commands.',
        '## Paired comparisons','P-values are two-sided and unadjusted. Differences average sampling and initialization seeds within each user before inference.',
        '| Comparator | Protocol / metric | Difference | Paired 95% CI | p |','|---|---|---:|---|---:|']
    for row in sig[sig.baseline.isin([paper,'SVD-MF','BPR-MF'])].itertuples():
        lines.append(f'| {row.baseline} | {row.protocol} / {row.metric} | {row.difference:+.4f} | [{row.ci_low:+.4f}, {row.ci_high:+.4f}] | {row.p_value:.5f} |')
    lines+=['## Component evidence',
        'The static-phase ablation removes temporal conditioning; the independent-BN ablation tests dependencies. Each ranker is refitted on validation. No-refit is a first-initialization diagnostic.',
        '| Ablation | Metric | Full minus ablation | 95% CI | p |','|---|---|---:|---|---:|']
    for r in ab_tests.itertuples():
        lines.append(f'| {r.ablation} | {r.metric} | {r.difference:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.p_value:.5f} |')
    inconclusive=ab_tests[(ab_tests.ci_low<=0)|(ab_tests.p_value>=.05)]
    if len(inconclusive): lines.append('Added accuracy is **not demonstrated** for: '+', '.join(f'{r.ablation} ({r.metric})' for r in inconclusive.itertuples())+'.')
    lines+=['## Explanations',
        f'Cited-component deletion mean rank drop: {faith.loc["rank_drop","mean"]:.2f}; random-other control: {faith.loc["control_rank_drop","mean"]:.2f}.',
        f'Deletion win rate: {faith.loc["faithful","mean"]:.3%}, 95% CI [{faith.loc["faithful","ci_low"]:.3%}, {faith.loc["faithful","ci_high"]:.3%}]. Both positive cited components are assessed when present.',
        f'Concrete watched-title evidence coverage: {faith.loc["evidence_coverage","mean"]:.3%}. Distinct texts: {int(faith.loc["distinct_strings","mean"])} / {int(faith.loc["explanation_count","mean"])}. Cross-user mean text Jaccard: {faith.loc["mean_jaccard","mean"]:.3f}.',
        f'Sufficiency NDCG ratio using only the target explanation\'s cited components: {faith.loc["sufficiency_ndcg_retained","mean"]:.3f}. Ratios above one mean the reduced score happened to rank targets better.',
        'Deletion favors components selected for large positive contributions. This checks arithmetic consistency, not causal validity or human usefulness.',
        '## Acceptance checks and limitations',
        f'Selected validation core: {int(hsel.clusters)} clusters, {int(hsel.states)} requested states ({int(hsel.effective_states)} effective), ESS={int(hsel.ess)}; validation HR@10={hsel.validation_hr:.3f}.',
        f'Catalogue BN scoring over all users: {diagnostics["bn_all_user_catalog_seconds"]:.3f} seconds. Mean duplicate fraction in core top-10 scores: {diagnostics["mean_top10_duplicate_fraction"]:.1%}. The <5% core-tie target is {supported(diagnostics["ties_under_5_percent"])}. Fusion retains continuous collaborative signals; discrete BN attributes still tie.',
        'Historical v1 ±.01 reproduction is not verified because exact original candidate pools and benchmark code were not supplied. The new split is different. The preserved v1 core is evaluated on this split; see v1_validation_comparison.json and legacy_v1.csv.',
        'The base implementation follows reviewed paper equations, with genre-frequency and held-out adapter choices documented in docs/base_paper_assumptions.md. Matching P0 overlap precision alone does not verify exact authors-code parity. Read reproduction_report.md.',
        'One dataset family and one held-out positive per user; Recall@10 equals HR@10. P0 watched-set precision and P1 held-out HR measure different tasks. Bayesian edges express statistical dependencies, not identified causal effects.',
        'Human study has not been conducted. Optional materials are available through scripts/human_study.py; no responses are fabricated. MovieLens-1M experiments, SASRec and a non-linear ranker remain optional stretch work.',
        '## Files','main_comparison.csv · significance.csv · ablation.csv · ablation_significance.csv · faithfulness.csv · protocol_comparison.csv · run_manifest.json. Figures are in the configured figures directory.']
    formatted=[]
    for position,line in enumerate(lines):
        previous=lines[position-1] if position else ''
        if line.startswith('|') or line.startswith('- '):
            formatted.append(line+'\n')
        else:
            formatted.append(('\n' if previous.startswith(('|','- ')) else '')+line+'\n\n')
    (results/'SUMMARY.md').write_text(''.join(formatted),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,default=Path('results')); p.add_argument('--figures',type=Path,default=Path('figures'))
    a=p.parse_args(); make_figures(a.results,a.figures)
