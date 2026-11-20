"""Fetch the original GroupLens archives; never redistribute them."""
import argparse
from pathlib import Path
import urllib.request
import zipfile

def download(dataset='ml-100k',data_dir='data'):
    root=Path(data_dir); root.mkdir(parents=True,exist_ok=True)
    sentinel='u.data' if dataset=='ml-100k' else 'ratings.dat'
    if (root/dataset/sentinel).exists(): return
    archive=root/f'{dataset}.zip'
    urllib.request.urlretrieve(f'https://files.grouplens.org/datasets/movielens/{dataset}.zip',archive)
    with zipfile.ZipFile(archive) as z:
        resolved=root.resolve()
        for member in z.infolist():
            if not (root/member.filename).resolve().is_relative_to(resolved):
                raise ValueError('Unsafe archive path')
        z.extractall(root)
    archive.unlink()
    print(f'Downloaded {dataset} from GroupLens. See its README for license terms.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['ml-100k','ml-1m'],default='ml-100k'); p.add_argument('--data-dir',default='data')
    a=p.parse_args(); download(a.dataset,a.data_dir)
