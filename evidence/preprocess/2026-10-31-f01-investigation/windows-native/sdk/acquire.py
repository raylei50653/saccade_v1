#!/usr/bin/env python3
"""Download official Windows CUDA redistributables into an isolated SDK tree."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
MANIFEST_URL = BASE+'redistrib_13.4.2.json'
COMPONENTS = ('cuda_cudart','libnvjpeg','cuda_sanitizer_api','cuda_crt','cccl','cuda_nvcc','libnvvm')

def acquire(item):
    key, meta = item
    platform = meta['windows-x86_64']
    url = BASE+platform['relative_path']
    target = ROOT/'downloads'/Path(platform['relative_path']).name
    start = datetime.datetime.now().astimezone().isoformat()
    with urllib.request.urlopen(url, timeout=60) as response, target.open('wb') as stream:
        shutil.copyfileobj(response,stream)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    assert actual == platform['sha256'], key
    assert target.stat().st_size == int(platform['size']), key
    extracted = ROOT/'components'/key
    extracted.mkdir()
    with zipfile.ZipFile(target) as archive:
        for member in archive.namelist():
            assert not Path(member).is_absolute() and '..' not in Path(member).parts
        archive.extractall(extracted)
    top = list(extracted.iterdir())
    assert len(top)==1 and top[0].is_dir(),key
    record = {'component':key,'version':meta['version'],'url':url,
              'official_sha256':platform['sha256'],'actual_sha256':actual,
              'bytes':target.stat().st_size,'started':start,
              'finished':datetime.datetime.now().astimezone().isoformat(),
              'extracted_root':str(top[0].relative_to(ROOT))}
    (ROOT/'records'/f'{key}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return record,top[0]

def main():
    for name in ('downloads','components','records','cuda'):
        (ROOT/name).mkdir()
    blob=urllib.request.urlopen(MANIFEST_URL,timeout=60).read()
    (ROOT/'redistrib_13.4.2.json').write_bytes(blob)
    manifest=json.loads(blob)
    records=[]
    roots=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for record,root in pool.map(acquire,[(key,manifest[key]) for key in COMPONENTS]):
            records.append(record);roots.append((record['component'],root))
    collisions=[]
    for key,root in roots:
        for source in sorted(root.rglob('*')):
            if not source.is_file():continue
            relative=source.relative_to(root)
            target=ROOT/'cuda'/relative
            if target.exists():
                same=target.read_bytes()==source.read_bytes()
                collisions.append({'component':key,'path':str(relative),'byte_identical':same})
                if not same:
                    # License documents stay per-component; do not replace
                    # functional SDK files silently.
                    if relative.name in ('LICENSE','LICENSE.txt') or relative.name.startswith('LICENSE'):
                        continue
                    raise RuntimeError(f'nonidentical collision {key}: {relative}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
    report={'manifest_url':MANIFEST_URL,
            'manifest_sha256':hashlib.sha256(blob).hexdigest(),
            'release':manifest['release_label'],'components':records,'merge_collisions':collisions}
    (ROOT/'SDK.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SDK READY '+str(ROOT/'cuda'),flush=True)

if __name__=='__main__':main()
