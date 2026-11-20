"""Check saved experiment evidence against scores, histories and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from tastetrace.demo_utils import load_bundle

def verify(results=Path('results')):
    results=Path(results)
    manifest=json.loads((results/'run_manifest.json').read_text())
    bundle=load_bundle(results/'demo_bundle.pkl')
    config_hash=manifest['config_hash']
    assert bundle['config_hash']==config_hash,'Bundle belongs to another experiment'
    source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src/tastetrace').rglob('*.py')))+(ROOT/'scripts/run_all.py').read_bytes()).hexdigest()
    assert source_hash==manifest['source_hash'],'Training source changed since the run'
    raw=pd.read_csv(results/'per_user_metrics.csv')
    main=pd.read_csv(results/'main_comparison.csv')
    for name in ['per_user_metrics.csv','main_comparison.csv','significance.csv','ablation.csv',
                 'faithfulness.csv','ablation_significance.csv','protocol_comparison.csv']:
        frame=pd.read_csv(results/name)
        assert set(frame.config_hash)=={config_hash},f'Mixed provenance in {name}'
    for row in main.itertuples():
        group=raw[(raw.method==row.method)&(raw.protocol==row.protocol)]
        actual=group.groupby('user_id')[row.metric].mean().mean()
        assert np.isclose(actual,row.mean),f'Aggregated metric mismatch: {row.method} {row.metric}'
    users=manifest['test_events']; methods=raw.method.nunique()
    expected=users*methods*len(manifest['config']['model_seeds'])*(len(manifest['config']['seeds'])+1)
    assert len(raw)==expected,'Missing model/seed/user evaluations'
    tt=bundle['models']['TasteTrace-v2']
    for row in bundle['split'].test.itertuples():
        assert row.item_id not in tt.index.histories[row.user_id],'Test interaction in fitted user history'
    for probs in tt.next_probs.values(): assert np.isclose(probs.sum(),1)
    for bn in tt.networks: assert bn.check_model()
    count=0; by_user={}; distinct=set()
    with (results/'explanations.jsonl').open(encoding='utf-8') as f:
        for line in f:
            ex=json.loads(line); u=ex['user_id']; i=ex['item_id']
            assert ex['config_hash']==config_hash
            assert set(ex['evidence_items'])<=set(tt.index.histories[u]),'Ungrounded evidence title'
            assert i not in tt.index.histories[u],'Watched item recommended in held-out demo'
            assert all(ex['contributions'][c]>0 for c in ex['cited_components'])
            assert ex['text'] and '{' not in ex['text'] and '}' not in ex['text']
            if count%100==0:
                assert np.isclose(sum(ex['contributions'].values()),tt.score(u,[i])[0]),'Explanation score mismatch'
            assert (u,i) not in distinct; distinct.add((u,i))
            by_user[u]=by_user.get(u,0)+1; count+=1
    assert len(by_user)==users and all(n==manifest['config']['evaluation']['k'] for n in by_user.values())
    ab=pd.read_csv(results/'ablation_per_user.csv')
    assert ab.method.nunique()==9
    report=dict(config_hash=config_hash,status='passed',evaluated_users=users,methods=int(methods),
        metric_rows=len(raw),verified_explanations=count,source_matches=True,grounded_evidence=True,
        test_history_leakage=False,all_metric_aggregates_match=True,ablation_variants=int(ab.method.nunique()))
    (results/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,default=Path('results'))
    verify(p.parse_args().results)
