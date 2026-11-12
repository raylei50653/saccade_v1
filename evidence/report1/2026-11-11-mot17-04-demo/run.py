#!/usr/bin/env python3
"""Capture one bounded MOT17-04 replay source; no GT, scoring or runtime edits."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
WORK = ROOT / 'build/report1-mot04-demo-20261111'
SOURCE = ROOT / 'build/report1-live-20261111/source'
BINARY = ROOT / 'build/report1-live-20261111/cuda/track_sequence'
ENGINE = Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')
SEQ = Path('/home/ray/developer/ai/dataset/MOT17/train/MOT17-04-SDP')
BASE = '3cbaa518f0dc37d691404e95ed85c0109c81130d'
ENGINE_SHA = '8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def manifest(paths):
    return {str(p): {'symlink': os.readlink(p)} if p.is_symlink() else
            {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}


def source_manifest():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() == BASE
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE)
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=SOURCE).split(b'\0')
    return manifest([SOURCE / os.fsdecode(p) for p in names if p])


def main():
    WORK.mkdir(exist_ok=False)
    out = WORK / 'run1'
    out.mkdir()
    assert sha(ENGINE) == ENGINE_SHA
    assert not [k for k in os.environ if k.startswith('SACCADE_CFG_')]
    contract = {'schema': 'report1-mot17-04-demo-capture/1', 'runtime_base_commit': BASE,
        'runtime_clean': True, 'sequence': str(SEQ), 'frames': 1050, 'fps': 30,
        'engine_sha256': ENGINE_SHA, 'engine_rebuilt': False, 'model_profile': 'hr-1280x768',
        'motion': 'v1-motion/2 G1 NVOFA', 'cuda_graph': True, 'mode': 'pipelined',
        'result_slots': 4, 'nsa_enabled': False, 'measurement_box': False,
        'purpose': 'one complete source capture for prerecorded technical replay',
        'not_a_performance_benchmark': True, 'no_gt_no_scoring': True,
        'no_new_quality_or_sanitizer_acceptance': True, 'gpu_lock': str(Path.home()/'.cache/saccade-v1/gpu.lock')}
    dump(EVIDENCE/'contract.json', contract)
    before_source = source_manifest()
    inputs = [ENGINE, SEQ/'seqinfo.ini'] + [SEQ/'img1'/f'{i:06}.jpg' for i in range(1,1051)]
    binaries = [BINARY] + sorted(BINARY.parent.glob('*.a'))
    before_inputs, before_binaries = manifest(inputs), manifest(binaries)
    for name, value in [('source',before_source),('inputs',before_inputs),('binaries',before_binaries)]:
        dump(EVIDENCE/(name+'-before.json'),value)
    env_command = ['bash','-c','date -Is\nuname -srmo\n/opt/cuda/bin/nvcc --version\n/usr/bin/c++ --version\nldd "$1"\nnvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv','capture-env',str(BINARY)]
    dump(EVIDENCE/'environment.command.json',{'argv':env_command,'cwd':str(ROOT)})
    with (EVIDENCE/'environment.log').open('wb') as log:
        subprocess.run(env_command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    argv = [str(p) for p in [BINARY,ENGINE,SEQ,out/'mot.txt','--profile','hr-1280x768',
        '--set','execution.mode=pipelined','--set','execution.cuda_graph=true',
        '--set','execution.result_slots=4','--set','tracker.kalman.nsa_enabled=false',
        '--set','tracker.rows.measurement_box=false','--packets',out/'packets.bin',
        '--motion',out/'motion.scmo','--resolved-config',out/'resolved.config']]
    dump(EVIDENCE/'run1.command.json',{'argv':argv,'cwd':str(ROOT)})
    print('START full MOT17-04 source capture',flush=True)
    with (EVIDENCE/'run1.log').open('wb') as log:
        started=time.monotonic()
        result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        elapsed=time.monotonic()-started
    dump(EVIDENCE/'run1.result.json',{'returncode':result.returncode,'observed_process_wall_seconds':elapsed,
        'timing_scope':'one ordinary subprocess source capture; not formal performance evidence'})
    result.check_returncode()
    log=(EVIDENCE/'run1.log').read_text()
    assert 'source_commit='+BASE+'\n' in log and 'source_dirty=true\n' not in log
    assert 'kalman.nsa_enabled=0\n' in log and 'rows.measurement_box=0\n' in log
    match,=re.findall(r'track_sequence frames=(\d+) mode=(\S+) wall_ms=(\S+) ms_per_frame=(\S+)',log)
    assert int(match[0])==1050 and match[1]=='pipelined'
    after_source, after_inputs, after_binaries = source_manifest(), manifest(inputs), manifest(binaries)
    for name, value in [('source',after_source),('inputs',after_inputs),('binaries',after_binaries)]:
        dump(EVIDENCE/(name+'-after.json'),value)
    assert before_source==after_source and before_inputs==after_inputs and before_binaries==after_binaries
    raw=EVIDENCE/'raw/run1'
    raw.mkdir(parents=True)
    outputs={}
    for p in out.iterdir():
        outputs[p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
        if p.suffix in ('.bin','.scmo'):
            with gzip.GzipFile(filename=str(raw/(p.name+'.gz')),mode='wb',mtime=0) as target:
                target.write(p.read_bytes())
        else:
            (raw/p.name).write_bytes(p.read_bytes())
    summary={'schema':'report1-mot17-04-demo-summary/1','status':'PASS_COMPLETE_SOURCE_CAPTURE',
        'contract':contract,'driver_version':'621.14','source_inputs_binaries_unchanged':True,
        'outputs':outputs,'observed_process_wall_seconds':elapsed,
        'metrics':None,'limitations':['single source capture, not a latency benchmark',
            'no GT opened or quality scores recomputed','MOT17-04 is in-domain data; not a new held-out evaluation',
            'no full-session sanitizer gate or external camera/RTSP verification']}
    dump(EVIDENCE/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    with open(Path.home()/'.cache/saccade-v1/gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
