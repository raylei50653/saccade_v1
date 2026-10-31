#!/usr/bin/env python3
"""Bounded normal-execution CUPTI diagnostic; never run alongside Sanitizer."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    out=args.out.resolve()
    if out.exists(): raise RuntimeError('refusing to overwrite output directory')
    out.mkdir(parents=True)
    capture=out/'capture';capture.mkdir()
    source=HERE/'cupti_trace.cpp'
    original=REPO/'build-f01-investigation/repro'
    inputs=[source,original,REPO/'fixtures/preprocess/baseline.jpg',REPO/'evidence/preprocess/2026-10-31-f01-investigation/baseline.rgb']
    identity={str(p):sha(p) for p in inputs}
    libraries=[Path(p).resolve() for p in ['/opt/cuda/lib64/libnvjpeg.so.13','/opt/cuda/lib64/libcupti.so','/usr/lib/wsl/lib/libcuda.so.1','/usr/lib/wsl/lib/libnvcuvid.so.1']]
    identity.update({str(p):sha(p) for p in libraries})
    (out/'pre-identity.json').write_text(json.dumps({'input_hashes':identity,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'worktree':subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True),'classification':'diagnostic; no sanitizer; capture not yet complete'},indent=2)+'\n')
    expected='58e2c0ea589fd64ef9e334793c2af8230f4e580d6a303888de769313b5da27c3'
    if sha(original)!=expected: raise RuntimeError('historical tested executable hash changed')
    shutil.copy2(source,out/source.name)
    def run(name,argv,env=None,limit=60):
        available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
        if available<4194304: raise RuntimeError('need 4 GiB MemAvailable')
        record={'argv':argv,'cwd':str(REPO),'environment_overrides':env or {},'time':datetime.datetime.now().astimezone().isoformat(),'mem_available_kib':available,'timeout_seconds':limit}
        (capture/f'{name}.command.json').write_text(json.dumps(record,indent=2)+'\n')
        with (capture/f'{name}.stdout').open('wb') as so,(capture/f'{name}.stderr').open('wb') as se:
            payload=(['env',*[f'{k}={v}' for k,v in env.items()]] if env else [])+argv
            proc=subprocess.run(['/usr/bin/time','-v','-o',str(capture/f'{name}.time'),'timeout','--signal=TERM','--kill-after=5s',str(limit),*payload],cwd=REPO,stdout=so,stderr=se)
        (capture/f'{name}.exit').write_text(str(proc.returncode)+'\n')
        print(name,proc.returncode,flush=True)
        return proc.returncode
    code=run('compile',['g++','-std=c++17','-shared','-fPIC','-O2','-g','-I/opt/cuda/include',str(out/source.name),'-L/opt/cuda/lib64','-Wl,-rpath,/opt/cuda/lib64','-lcupti','-L/usr/lib/wsl/lib','-lcuda','-ldl','-o',str(out/'trace.so')])
    if code: raise RuntimeError('compile failed; retained attempt')
    identity[str(out/'trace.so')]=sha(out/'trace.so')
    (out/'built-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    run('device',['nvidia-smi'])
    run('sanitizer-version',['/opt/cuda/bin/compute-sanitizer','--version'])
    run('ldd',['ldd',str(original)])
    for name,flags,preallocate in [('default','default','preallocate'),('no-pipeline','no-pipeline','preallocate'),('no-preallocate','default','no-preallocate')]:
        argv=[str(original),'hw','rgb',flags,preallocate,'uninit','2',str(inputs[2]),str(inputs[3]),str(out/f'{name}.rgb')]
        code=run(name,argv,{'LD_PRELOAD':str(out/'trace.so'),'F01_TRACE_LOG':str(out/f'{name}.jsonl'),'F01_MAPS_OUT':str(out/f'{name}.maps')})
        if code: raise RuntimeError(f'{name} failed; retained attempt')
        records=[json.loads(s) for s in (out/f'{name}.jsonl').read_text().splitlines()]
        launches=[r for r in records if r.get('domain')=='driver' and r.get('converter_launch')]
        surfaces=[r for r in records if r['kind']=='surface_identity']
        if len(launches)!=2 or len(surfaces)!=2: raise RuntimeError('missing authoritative converter/surface snapshots')
        if any(r['range_status'] or r['id_status'] for r in surfaces): raise RuntimeError('allocation query failed')
        if sha(out/f'{name}.rgb')!=sha(inputs[3]): raise RuntimeError('RGB mismatch')
    after={str(p):sha(p) for p in [*inputs,*libraries,out/'trace.so']}
    if any(identity[str(p)]!=after[str(p)] for p in [*inputs,*libraries,out/'trace.so']):raise RuntimeError('input/source/library/tracer drift')
    loaded={}
    for maps in out.glob('*.maps'):
        for line in maps.read_text().splitlines():
            fields=line.split(maxsplit=5)
            if len(fields)==6 and fields[5].startswith('/') and any(s in fields[5] for s in ('libcuda','libnvjpeg','libnvcuvid','libcupti')):
                path=Path(fields[5]);loaded[str(path)]=sha(path)
    (out/'identity.json').write_text(json.dumps({'inputs_before':identity,'inputs_after':after,'loaded_modules':loaded,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'worktree':subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True),'classification':'diagnostic; no sanitizer; same historical standalone executable; added CUPTI callbacks and host pointer queries'},indent=2)+'\n')

if __name__=='__main__':main()
