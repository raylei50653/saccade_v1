"""Balanced fresh-process model-only TensorRT comparison.

Run only after the root agent authorizes performance measurement. The complete
series holds the shared GPU lock so engine builds and pipeline tests cannot
interleave with timed samples.
"""
from pathlib import Path
import argparse,csv,datetime,fcntl,hashlib,json,math,os,statistics,subprocess,time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=HERE.parents[2]
BIN=HERE/'model_compare'
IMAGES=Path('/home/ray/.cache/saccade-v1/hr-deploy/refs/model-images.f32')
ENGINES={
 'main':Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'),
 'official':ROOT/'artifacts/official/model.engine',
 'finetuned':ROOT/'artifacts/finetuned/model.engine',
 'official-default':ROOT/'artifacts/official-default/model.engine',
}
RAW_ORDER=['main','official','finetuned','finetuned','official','main']
DEFAULT_ORDER=['main','official-default','official-default','main']
LIBS=REPO/'build/report1-live-20261111/cuda'
SOURCES=[HERE/'model_compare.cpp',BIN,HERE/'build.json',HERE/'run_model.py',
 *[LIBS/('libsaccade_'+x+'.a') for x in ['scan_trt','selective_scan_cuda','selective_scan','file_io','sha256']]]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def percentile(x,q):return sorted(x)[max(1,math.ceil(len(x)*q))-1]
def describe(x):return {'mean':statistics.mean(x),'p50':percentile(x,.5),'p95':percentile(x,.95),'p99':percentile(x,.99),'samples':len(x)}
def aggregate(out):
 runs=json.loads((out/'runs.json').read_text())
 expected=[(comparison,cycle,ordinal,arm) for comparison,order in [('raw6',RAW_ORDER),('original-default',DEFAULT_ORDER)] for cycle in range(2) for ordinal,arm in enumerate(order)]
 assert [(r['comparison'],r['cycle'],r['ordinal'],r['arm']) for r in runs]==expected,'incomplete or reordered model session matrix'
 assert all(r['exit_code']==0 for r in runs)
 groups={}
 for rec in runs:
  p=Path(rec['output']);arm=rec['arm'];comparison=rec['comparison']
  summary=json.loads((p/'summary.json').read_text());assert summary['status']=='PASS' and not summary['verify_only']
  for row in csv.DictReader((p/'samples.csv').open()):
   g=groups.setdefault((comparison,arm,row['mode']),{'gpu':[],'wall':[],'queued_gpu_ms':[],'queued_wall_ms':[],'queued_replays':0,'processes':set(),'verification_hashes':set()})
   g['gpu'].append(float(row['gpu_event_ms']));g['wall'].append(float(row['cpu_wall_ms']));g['processes'].add(str(p))
  for row in csv.DictReader((p/'queued.csv').open()):
   g=groups[(comparison,arm,row['mode'])];g['queued_gpu_ms'].append(float(row['gpu_event_total_ms']));g['queued_wall_ms'].append(float(row['cpu_wall_total_ms']));g['queued_replays']+=int(row['replays'])
  hashes=tuple((x['input'],x['payload_sha256'],x['status_sha256']) for x in csv.DictReader((p/'verification.csv').open()))
  for mode in ['ordinary','graph']:groups[(comparison,arm,mode)]['verification_hashes'].add(hashes)
 result=[]
 for (comparison,arm,mode),g in groups.items():
  assert len(g['processes'])==4,(comparison,arm,mode,'fresh-process count')
  assert len(g['gpu'])==len(g['wall'])==1280 and len(g['queued_gpu_ms'])==len(g['queued_wall_ms'])==32 and g['queued_replays']==2048,(comparison,arm,mode,'measurement coverage')
  assert len(g['verification_hashes'])==1,(comparison,arm,'output changed across processes')
  result.append({'comparison':comparison,'arm':arm,'mode':mode,'fresh_processes':len(g['processes']),
   'gpu_event_single_ms':describe(g['gpu']),'cpu_wall_single_ms':describe(g['wall']),
   'queued_batches':len(g['queued_gpu_ms']),'queued_replays':g['queued_replays'],
   'queued_gpu_interval_ms':sum(g['queued_gpu_ms'])/g['queued_replays'],
   'queued_cpu_interval_ms':sum(g['queued_wall_ms'])/g['queued_replays'],
   'queued_gpu_fps':1000*g['queued_replays']/sum(g['queued_gpu_ms']),
   'queued_cpu_fps':1000*g['queued_replays']/sum(g['queued_wall_ms']),
   'all_four_input_output_bits_stable_across_processes':True})
 expected_groups={('raw6',a,m) for a in ['main','official','finetuned'] for m in ['ordinary','graph']}|{('original-default',a,m) for a in ['main','official-default'] for m in ['ordinary','graph']}
 assert set(groups)==expected_groups,'incomplete model group matrix'
 save(out/'summary.json',{'status':'PASS','groups':result,
  'single_scope':'one enqueue/Graph through CUDA event synchronization; model only',
  'queued_scope':'64 back-to-back replays per batch; throughput interval, not individual-frame latency',
  'excludes':'H2D,D2H,allocation,initialization,warmup,verification,file_IO',
  'default_comparison_scope':'official-default includes YOLO one2one decode and top-k; raw6 comparisons exclude F04',
  'formal_acceptance':False})
 with (out/'summary.csv').open('w') as f:
  cols=['comparison','arm','mode','fresh_processes','gpu_mean_ms','gpu_p50_ms','gpu_p95_ms','wall_mean_ms','wall_p50_ms','wall_p95_ms','queued_gpu_interval_ms','queued_cpu_interval_ms','queued_gpu_fps','queued_cpu_fps']
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for r in result:w.writerow({k:r[k] for k in cols if k in r}|{f'{prefix}_{k}_ms':r[key][k] for prefix,key in [('gpu','gpu_event_single_ms'),('wall','cpu_wall_single_ms')] for k in ['mean','p50','p95']})
 # Preserve every raw latency sample with external sequence identity.
 with (out/'all-samples.csv').open('w') as f:
  cols=['comparison','cycle','external_ordinal','arm','run','round','input','ordinal','mode','iteration','gpu_event_ms','cpu_wall_ms']
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for rec in runs:
   for row in csv.DictReader((Path(rec['output'])/'samples.csv').open()):
    w.writerow({'comparison':rec['comparison'],'cycle':rec['cycle'],'external_ordinal':rec['ordinal'],'arm':rec['arm'],'run':Path(rec['output']).name,**row})
 return result

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('output',type=Path);ap.add_argument('--aggregate-only',action='store_true');ap.add_argument('--external-lock-fd',type=int)
 args=ap.parse_args();out=args.output.resolve()
 assert out.is_relative_to(HERE), 'write outputs within benchmark-tools only'
 if args.aggregate_only:aggregate(out);return
 assert not out.exists();out.mkdir(parents=True)
 paths=[*ENGINES.values(),IMAGES,*SOURCES]
 before={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
 save(out/'inputs-before.json',before)
 contract={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'cycles':2,'raw_engine_order_per_cycle':RAW_ORDER,'default_engine_order_per_cycle':DEFAULT_ORDER,
  'all_inputs_timed':4,'iterations_per_ABBA_group':40,'warmup_per_group':12,'ABBA_rounds_per_process':1,
  'queued_replays_per_group':64,'fresh_processes_per_arm_per_comparison':4,
  'same_engine_under_modes':'ordinary then Graph then Graph then ordinary per input',
  'scope':'four fixed real HR images resident before measured model-only calls; no profiler',
  'default_scope':'separate comparison, original decoded end2end output; includes decode/top-k inside engine',
  'stats':'nearest-rank per-invocation percentiles; queued batch intervals are throughput',
  'inputs':before,'no_gpu_interference':'shared GPU flock held for entire series'}
 save(out/'PLAN.json',contract)
 runs=[]
 with (Path('/home/ray/.cache/saccade-v1/gpu.lock').open('a') if args.external_lock_fd is None else os.fdopen(os.dup(args.external_lock_fd),'a')) as lock:
  assert os.fstat(lock.fileno()).st_ino==Path('/home/ray/.cache/saccade-v1/gpu.lock').stat().st_ino
  fcntl.flock(lock,fcntl.LOCK_EX if args.external_lock_fd is None else fcntl.LOCK_EX|fcntl.LOCK_NB)
  for comparison,order in [('raw6',RAW_ORDER),('original-default',DEFAULT_ORDER)]:
   for cycle in range(2):
    for ordinal,arm in enumerate(order):
     dest=out/f'{comparison}-c{cycle}-o{ordinal}-{arm}'
     cmd=[str(BIN),str(ENGINES[arm]),str(IMAGES),str(dest),'--layout','default' if arm=='official-default' else 'raw6','--iterations','40','--warmup','12','--rounds','1','--queue','64']
     started=time.monotonic();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
     with (out/(dest.name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
     rec={'comparison':comparison,'cycle':cycle,'ordinal':ordinal,'arm':arm,'command':cmd,'output':str(dest),'started_utc':stamp,'wall_seconds_including_initialization':time.monotonic()-started,'exit_code':r.returncode}
     runs.append(rec);save(out/'runs.json',runs)
     print(json.dumps(rec),flush=True)
     assert r.returncode==0,(dest,'benchmark process failed')
  if args.external_lock_fd is None:fcntl.flock(lock,fcntl.LOCK_UN)
 after={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
 save(out/'inputs-after.json',after);assert before==after,'engine/input/harness changed'
 aggregate(out)

if __name__=='__main__':main()
