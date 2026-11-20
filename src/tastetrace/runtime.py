import hashlib
import json
import pickle
import time
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()[:16]

class Cache:
    def __init__(self,root,run_hash,force=False):
        self.root=Path(root)/'cache'/run_hash; self.root.mkdir(parents=True,exist_ok=True)
        self.force=force; self.timings=[]

    def get(self,name,parameters,build):
        path=self.root/f'{name}_{digest(parameters)}.pkl'
        start=time.perf_counter(); hit=path.exists() and not self.force
        print(f'{"CACHE" if hit else "BUILD"} {name} {parameters}',flush=True)
        if hit:
            with path.open('rb') as f: value=pickle.load(f)
        else:
            value=build()
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as f: pickle.dump(value,f,protocol=pickle.HIGHEST_PROTOCOL)
            temp.replace(path)
        elapsed=time.perf_counter()-start
        self.timings.append(dict(stage=name,seconds=elapsed,cached=hit))
        print(f'DONE {name} ({elapsed:.1f}s)',flush=True)
        return value

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)),encoding='utf-8')
