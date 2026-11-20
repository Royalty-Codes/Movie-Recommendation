import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from tastetrace.demo_utils import load_bundle,demo_report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--user',type=int,default=42); p.add_argument('--results',default='results')
    a=p.parse_args(); root=Path(a.results)
    if not (root/'demo_bundle.pkl').exists(): p.error('Run python scripts/run_all.py first, or use --results results/smoke.')
    bundle=load_bundle(root/'demo_bundle.pkl')
    if a.user not in set(bundle['split'].test.user_id): p.error('Choose an eligible existing user from the evaluated dataset.')
    report=demo_report(bundle,a.user); print(report)
    (root/f'demo_user_{a.user}.md').write_text(report,encoding='utf-8')
