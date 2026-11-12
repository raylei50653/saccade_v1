#!/usr/bin/env python3
"""Three bounded full-process offline sequence runs; no runtime changes or GT reads."""
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
SOURCE = ROOT / 'build/report1-live-20261111/source'
BUILD = ROOT / 'build/report1-live-20261111/cuda'
OUTPUT = ROOT / 'build/report1-live-20261111/offline-e2e'
BASE = '3cbaa518f0dc37d691404e95ed85c0109c81130d'
ENGINE = Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')
SEQ = Path('/home/ray/developer/ai/dataset/MOT17/train/MOT17-10-SDP')
ENGINE_SHA = '8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876'

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

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

def run(name, argv, cwd=ROOT, timeout=300):
    argv = list(map(str, argv))
    dump(EVIDENCE / (name + '.command.json'), {'argv': argv, 'cwd': str(cwd)})
    started = datetime.now(timezone.utc).isoformat()
    print('START', name, flush=True)
    with (EVIDENCE / (name + '.log')).open('wb') as output:
        before = time.perf_counter_ns()
        process = subprocess.Popen(argv, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
        watchdog = threading.Timer(timeout, process.kill)
        watchdog.start()
        try:
            returncode = process.wait()
        finally:
            watchdog.cancel()
        after = time.perf_counter_ns()
        result = subprocess.CompletedProcess(argv, returncode)
    receipt = {'returncode': result.returncode, 'wall_ms': (after-before)/1e6,
               'start_utc': started, 'end_utc': datetime.now(timezone.utc).isoformat(),
               'perf_counter_before_ns': before, 'perf_counter_after_ns': after}
    dump(EVIDENCE / (name + '.result.json'), receipt)
    print('DONE', name, result.returncode, receipt['wall_ms'], flush=True)
    if result.returncode:
        print((EVIDENCE / (name + '.log')).read_text()[-6000:], flush=True)
        result.check_returncode()
    return receipt

def stats(values):
    return {'n': len(values), 'mean': statistics.fmean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)}

def main():
    OUTPUT.mkdir(exist_ok=False)
    contract = {'schema': 'report1-offline-process-e2e-contract/1', 'runtime_base_commit': BASE,
        'runtime_clean': True, 'benchmark_instrumentation': False, 'engine_sha256': ENGINE_SHA,
        'engine_rebuilt': False, 'sequence': str(SEQ), 'frames_per_process': 654, 'fresh_processes': 3,
        'model_profile': 'hr-1280x768', 'motion': 'v1-motion/2 G1 NVOFA', 'graph': True,
        'mode': 'pipelined', 'result_slots': 4, 'nsa_enabled': False, 'measurement_box': False,
        'outputs': ['final MOT', 'all frame packets', 'all frame MotionRecords', 'resolved config', 'stdout log'],
        'primary_boundary': 'external perf_counter immediately before Popen through blocking waitpid return after child process exit; watchdog timeout, no polling wait',
        'included': ['process startup', 'engine load and session constructor', 'all 654 JPEG file reads',
                     'all frame processing and lazy capture', 'drain and checked session close',
                     'sequence-tail interpolation and ID finalization', 'final MOT/packet/motion writes and checked file close',
                     'resolved-config write', 'stdout log write', 'process exit and parent wait'],
        'excluded': ['build', 'preflight hashing', 'post-run output verification/hash/compression',
                     'fsync or storage durability', 'camera/network/live acquisition'],
        'cache_policy': 'not cleared; preflight input hashing reads all files and may warm OS page cache',
        'latency_interpretation': 'whole offline sequence completion; wall/frames is amortized cost, not per-frame latency',
        'gpu_lock': '/home/ray/.cache/saccade-v1/gpu.lock', 'telemetry': 'nvidia-smi 1-second sampling over 3-run block',
        'clock_lock': False, 'order': 'three consecutive fresh processes', 'outlier_removal': False,
        'no_gt_no_scoring': True, 'no_sanitizer_or_quality_acceptance': True}
    dump(EVIDENCE / 'contract.json', contract)
    before_source = source_manifest(); dump(EVIDENCE / 'source-before.json', before_source)
    inputs = [ENGINE, SEQ/'seqinfo.ini'] + [SEQ/'img1'/f'{i:06}.jpg' for i in range(1,655)]
    before_inputs = manifest(inputs); dump(EVIDENCE/'inputs-before.json', before_inputs)
    assert sha(ENGINE) == ENGINE_SHA
    assert not [k for k in os.environ if k.startswith('SACCADE_CFG_')]
    run('environment', ['bash','-c','date -Is\nuname -srmo\n/opt/cuda/bin/nvcc --version\n/usr/bin/c++ --version\nnvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv'])
    run('build-track-sequence', ['cmake','--build',BUILD,'--parallel','8','--target','track_sequence'],timeout=600)
    binary = BUILD/'track_sequence'
    binaries = [binary]+list(BUILD.glob('*.a'))
    before_binaries = manifest(binaries);dump(EVIDENCE/'binaries-before.json',before_binaries)
    records = []
    with (EVIDENCE/'gpu-telemetry.csv').open('wb') as log:
        telemetry = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv','-l','1'],stdout=log,stderr=subprocess.STDOUT)
        try:
            for ordinal in range(1,4):
                out = OUTPUT/f'run{ordinal}';out.mkdir()
                receipt = run(f'run{ordinal}', [binary,ENGINE,SEQ,out/'mot.txt','--profile','hr-1280x768',
                    '--set','execution.mode=pipelined','--set','execution.cuda_graph=true',
                    '--set','execution.result_slots=4','--set','tracker.kalman.nsa_enabled=false',
                    '--set','tracker.rows.measurement_box=false','--packets',out/'packets.bin',
                    '--motion',out/'motion.scmo','--resolved-config',out/'resolved.config'])
                text=(EVIDENCE/f'run{ordinal}.log').read_text()
                match,=re.findall(r'track_sequence frames=(\d+) mode=(\S+) wall_ms=(\S+) ms_per_frame=(\S+)',text)
                assert int(match[0])==654 and match[1]=='pipelined'
                assert 'source_commit='+BASE+'\n' in text and 'source_dirty=true\n' not in text
                assert 'kalman.nsa_enabled=0\n' in text and 'rows.measurement_box=0\n' in text
                records.append({'ordinal':ordinal,**receipt,'app_frames':int(match[0]),'app_mode':match[1],
                    'app_loop_wall_ms':float(match[2]),'app_loop_ms_per_frame':float(match[3])})
        finally:
            telemetry.terminate();telemetry.wait(timeout=10)
    after_source=source_manifest();after_inputs=manifest(inputs);after_binaries=manifest(binaries)
    for kind,data in [('source',after_source),('inputs',after_inputs),('binaries',after_binaries)]:dump(EVIDENCE/(kind+'-after.json'),data)
    assert before_source==after_source and before_inputs==after_inputs and before_binaries==after_binaries
    parity={}
    for filename in ['mot.txt','packets.bin','motion.scmo','resolved.config']:
        payloads=[(OUTPUT/f'run{i}'/filename).read_bytes() for i in range(1,4)]
        assert all(data==payloads[0] for data in payloads),filename
        parity[filename]={'all_three_byte_equal':True,'bytes_each':len(payloads[0]),'sha256':hashlib.sha256(payloads[0]).hexdigest()}
    prior=(ROOT/'build/report1-live-20261111/runs/r0-2-graph-single.bin').read_bytes()
    assert (OUTPUT/'run1/packets.bin').read_bytes()[:len(prior)]==prior
    raw=EVIDENCE/'raw';raw.mkdir()
    for i in range(1,4):
        target=raw/f'run{i}';target.mkdir()
        for path in (OUTPUT/f'run{i}').iterdir():
            if path.name in ('packets.bin','motion.scmo'):
                with gzip.GzipFile(filename=str(target/(path.name+'.gz')),mode='wb',mtime=0) as out:out.write(path.read_bytes())
            else:(target/path.name).write_bytes(path.read_bytes())
    dump(EVIDENCE/'runs.json',records)
    telemetry_rows=list(csv.DictReader((EVIDENCE/'gpu-telemetry.csv').open(),skipinitialspace=True))
    telemetry_summary={'samples':len(telemetry_rows),'sampling_period_seconds':1,'scope':'three-run block including startup, output and gaps; not steady-state utilization'}
    for dest,key in [('temperature_c','temperature.gpu'),('power_w','power.draw [W]'),('utilization_percent','utilization.gpu [%]'),('memory_mib','memory.used [MiB]')]:
        telemetry_summary[dest]=stats([float(r[key].split()[0]) for r in telemetry_rows])
    process_times=[r['wall_ms'] for r in records]
    summary={'schema':'report1-offline-process-e2e-summary/1','status':'PASS_BOUNDED_PROCESS_COMPLETION_AND_OUTPUT_PARITY',
        'contract':contract,'driver_version':'621.14','process_wall_ms':stats(process_times),
        'amortized_process_ms_per_frame':stats([v/654 for v in process_times]),
        'aggregate_process_throughput_fps':1962*1000/sum(process_times),
        'app_loop_wall_ms':stats([r['app_loop_wall_ms'] for r in records]),
        'app_loop_ms_per_frame':stats([r['app_loop_ms_per_frame'] for r in records]),
        'process_minus_app_loop_ms':stats([r['wall_ms']-r['app_loop_wall_ms'] for r in records]),
        'outputs':parity,'packet_prefix_matches_prior_140_frame_benchmark':True,
        'source_inputs_binaries_unchanged':True,'run_records':records,'telemetry_observations':telemetry_summary,
        'limitations':['one complete sequence, three fresh processes','file reads use uncontrolled OS cache warmed by preflight',
            'process wall is full sequence completion, not per-frame latency','MOT is finalized after EOF',
            'includes packet/motion/resolved/stdout outputs as well as MOT','no fsync durability claim',
            'no live camera/network source, quality rescoring or sanitizer acceptance']}
    dump(EVIDENCE/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    with open('/home/ray/.cache/saccade-v1/gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
