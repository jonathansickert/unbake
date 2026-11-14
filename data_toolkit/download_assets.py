"""
Download every object in data_toolkit/selected_assets.json (written by select_assets.py) to <root>/<sha256>/pbr.glb and
write <root>/metadata.csv. Objects already in the dataset are skipped.

Run from the repo root (pip install objaverse pandas trimesh tqdm):
    python data_toolkit/download_assets.py baked_env
"""
import os
import json
import shutil
import argparse
import tempfile
import pandas as pd
from select_assets import download


def download_objects(opt):
    records = json.load(open(opt.list))
    os.makedirs(opt.root, exist_ok=True)
    todo = [r for r in records if not os.path.isfile(os.path.join(opt.root, r['sha256'], 'pbr.glb'))]
    print(f'Downloading {len(todo)} of {len(records)} objects to {opt.root}')
    if todo:
        objects = pd.DataFrame([{'fileIdentifier': r['file_identifier'], 'source': r['source'],
                                 'sha256': r['sha256']} for r in todo])
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = download(objects, tmp)   # skips objects that fail instead of aborting
            for r in todo:
                path = downloaded.get(r['file_identifier'])
                if path is None or not os.path.exists(path):
                    print(f'  failed: {r["sha256"]}')
                    continue
                target = os.path.join(opt.root, r['sha256'], 'pbr.glb')
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.move(path, target + '.tmp')   # may be a slow copy across disks...
                os.replace(target + '.tmp', target)  # ...so pbr.glb only appears once complete
    present = [r for r in records if os.path.isfile(os.path.join(opt.root, r['sha256'], 'pbr.glb'))]
    pd.DataFrame(present).to_csv(os.path.join(opt.root, 'metadata.csv'), index=False)
    print(f'{len(present)} objects in {opt.root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default='baked_env', help='dataset folder')
    parser.add_argument('--list', default='data_toolkit/selected_assets.json', help='JSON list of objects to download')
    download_objects(parser.parse_args())
