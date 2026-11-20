"""Export blinded study stimuli, or analyze real paired human responses.

This script never invents responses and never sends forms to respondents.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tastetrace.demo_utils import load_bundle,recommend
from tastetrace.explain import explain

CONDITIONS=['tastetrace','generic','none']
DIMENSIONS=['understandability','trust','usefulness']

def export(results,seed=7):
    bundle=load_bundle(results/'demo_bundle.pkl'); model=bundle['models']['TasteTrace-v2']
    rng=np.random.default_rng(seed)
    users=rng.choice(sorted(bundle['split'].test.user_id),size=10,replace=False)
    rows=[]
    for trial,u in enumerate(users,1):
        item=recommend(model,int(u))[int(rng.integers(10))][0]
        history=' | '.join(model.movies.at[int(i),'title'] for i in model.index.histories[u][-10:])
        ex=explain(model,int(u),item)['text']
        for group in range(3):
            condition=CONDITIONS[(trial+group)%3]
            rows.append(dict(group=chr(65+group),trial=trial,user_id=u,item_id=item,history=history,
                recommendation=model.movies.at[item,'title'],condition=condition,
                explanation=ex if condition=='tastetrace' else 'Popular among similar viewers.' if condition=='generic' else '',
                config_hash=bundle['config_hash']))
    pd.DataFrame(rows).to_csv(results/'human_study_stimuli.csv',index=False)
    pd.DataFrame(columns=['respondent_id','group','trial','condition']+DIMENSIONS).to_csv(results/'human_response_template.csv',index=False)
    print('Exported stimuli and an EMPTY response template. No human study has been conducted.')

def analyze(path,output):
    frame=pd.read_csv(path)
    required={'respondent_id','condition',*DIMENSIONS}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Supply non-empty real responses with respondent_id, condition and the three Likert dimensions.')
    if not frame.condition.isin(CONDITIONS).all(): raise ValueError('Unknown condition')
    if frame[DIMENSIONS].isna().any().any() or not frame[DIMENSIONS].isin([1,2,3,4,5]).all().all():
        raise ValueError('Responses must be complete integer Likert ratings from 1 to 5.')
    rows=[]
    for dimension in DIMENSIONS:
        pivot=frame.groupby(['respondent_id','condition'])[dimension].mean().unstack()
        for control in ['generic','none']:
            pair=pivot.reindex(columns=['tastetrace',control]).dropna()
            if len(pair)<2: continue
            delta=pair.tastetrace-pair[control]
            p=1.0 if (delta==0).all() else float(wilcoxon(delta,alternative='two-sided').pvalue)
            rows.append(dict(dimension=dimension,control=control,paired_respondents=len(pair),
                tastetrace_mean=pair.tastetrace.mean(),control_mean=pair[control].mean(),p_value=p))
    if not rows: raise ValueError('Not enough paired respondents across conditions.')
    pd.DataFrame(rows).to_csv(output,index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--results',type=Path,default=Path('results')); parser.add_argument('--responses',type=Path)
    args=parser.parse_args()
    if args.responses: analyze(args.responses,args.results/'human_study_analysis.csv')
    else: export(args.results)
