"""Balanced native detection and tracking measurements; root GO required.

Each process measures the same first 140 JPEGs (20 warmup, 120 samples).
Tracking uses deployed Graph/G1 and ring4; DetectionSession uses its existing
ordinary enqueue path. Prefix JPEGs are preloaded by the native harness.
"""
from pathlib import Path
import argparse,csv,datetime,fcntl,hashlib,json,math,os,statistics,subprocess,time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=HERE.parents[2]
BIN=ROOT/'pipeline-tools/pipeline_benchmark'
DATA=REPO.parent/'dataset/MOT17/train'
ENGINES={
 'main':Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'),
 'official':ROOT/'artifacts/official/model.engine',
 'finetuned':ROOT/'artifacts/finetuned/model.engine',
}
ORDER=['main','official','finetuned','finetuned','official','main']
SEQS=['MOT17-10-SDP','MOT17-02-SDP']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def pct(x,q):return sorted(x)[max(1,math.ceil(len(x)*q))-1]
def describe(x):return {'mean':statistics.mean(x),'p50':pct(x,.5),'p95':pct(x,.95),'p99':pct(x,.99),'samples':len(x)}

def aggregate(out):
 runs=json.loads((out/'runs.json').read_text());groups={};hashes={};all_samples=[];engine_hashes={a:sha(p) for a,p in ENGINES.items()}
 expected=[(seq,path,mode,cycle,ordinal,arm) for seq in SEQS for path in ['detect','tracking'] for mode in ['single','queued'] for cycle in range(2) for ordinal,arm in enumerate(ORDER)]
 assert [(r['sequence'],r['path'],r['mode'],r['cycle'],r['ordinal'],r['arm']) for r in runs]==expected,'incomplete or reordered pipeline session matrix'
 assert all(r['exit_code']==0 for r in runs)
 for r in runs:
  p=Path(r['output']);summary=json.loads((p/'summary.json').read_text())
  assert summary['status']=='PASS' and (summary['frames'],summary['warmup'],summary['samples'],summary['ring_slots'])==(140,20,120,4)
  assert summary['path']==r['path'] and summary['mode']==r['mode']
  assert (summary['candidate_slots'],summary['motion_slots'],summary['result_slots'])==(2,2 if r['path']=='tracking' else 0,4)
  assert summary['tracking_graph']==(r['path']=='tracking')
  assert summary['engine_sha256']==engine_hashes[r['arm']]
  key=(r['sequence'],r['path'],r['arm'],r['mode'])
  g=groups.setdefault(key,{'samples':[],'intervals':[],'initializations':[],'processes':[]})
  samples=[x for x in csv.DictReader((p/'delivery.csv').open()) if x['warmup']=='0']
  assert len(samples)==120 and [int(x['frame']) for x in samples]==list(range(20,140))
  for x in samples:
   assert math.isfinite(float(x['delivery_ms'])) and float(x['delivery_ms'])>0
   g['samples'].append(float(x['delivery_ms']))
   all_samples.append({'sequence':r['sequence'],'path':r['path'],'arm':r['arm'],'mode':r['mode'],'cycle':r['cycle'],'ordinal':r['ordinal'],'run':p.name,**x})
  g['intervals'].append(summary['interval_ms_per_frame']);g['initializations'].append(summary['init_ms']);g['processes'].append(str(p))
  hashes.setdefault((r['sequence'],r['path'],r['arm']),set()).add(summary['result_sha256'])
 for key,h in hashes.items():assert len(h)==1,(key,'result changes across repeats or queue modes')
 result=[]
 for (sequence,path,arm,mode),g in groups.items():
  assert len(g['processes'])==4 and len(g['samples'])==480
  interval=statistics.mean(g['intervals'])
  result.append({'sequence':sequence,'path':path,'arm':arm,'mode':mode,'fresh_processes':4,
   'delivery_ms':describe(g['samples']),'interval_ms_per_frame':interval,'throughput_fps':1000/interval,
   'process_interval_ms':describe(g['intervals']),'initialization_ms':describe(g['initializations']),
   'result_hash_identical_all_modes_and_repeats':True,
   'result_sha256':next(iter(hashes[(sequence,path,arm)]))})
 assert set(groups)=={(seq,path,arm,mode) for seq in SEQS for path in ['detect','tracking'] for arm in ENGINES for mode in ['single','queued']},'incomplete pipeline group matrix'
 save(out/'summary.json',{'status':'PASS','groups':result,'frames_per_session':140,'warmup':20,
  'samples_per_group':480,'prefixes':SEQS,'order':ORDER,'cycles':2,
  'delivery_scope':'enqueue call start to receive return; end before caller copy/check',
  'throughput_scope':'wall interval including same caller copy/check, ring fill/drain, after warmup',
  'excludes':'JPEG disk reads, model initialization, first20frames, output file writes',
  'tracking_graph':True,'detection_graph':False,'candidate_slots':2,'motion_slots_tracking':2,'result_slots':4,'ring_slots':4,'formal_acceptance':False})
 with (out/'summary.csv').open('w') as f:
  cols=['sequence','path','arm','mode','fresh_processes','mean_delivery_ms','p50_delivery_ms','p95_delivery_ms','p99_delivery_ms','interval_ms_per_frame','throughput_fps','result_sha256']
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for r in result:
   w.writerow({k:r[k] for k in cols if k in r}|{f'{k}_delivery_ms':r['delivery_ms'][k] for k in ['mean','p50','p95','p99']})
 with (out/'all-delivery.csv').open('w') as f:
  cols=['sequence','path','arm','mode','cycle','ordinal','run','frame','warmup','delivery_ms'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(all_samples)
 return result

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--aggregate-only',action='store_true');ap.add_argument('--external-lock-fd',type=int);args=ap.parse_args();out=args.output.resolve()
 assert out.is_relative_to(HERE),'write outputs under benchmark-tools only'
 if args.aggregate_only:aggregate(out);return
 assert not out.exists();out.mkdir(parents=True)
 paths=[*ENGINES.values(),BIN,HERE/'run_pipeline.py']
 paths+=sorted((ROOT/'pipeline-tools').glob('*.cpp'))+sorted((ROOT/'pipeline-tools').glob('*.cu'))
 for seq in SEQS:paths+=[DATA/seq/'seqinfo.ini',*[DATA/seq/'img1'/f'{i:06d}.jpg' for i in range(1,141)]]
 before={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths};save(out/'inputs-before.json',before)
 save(out/'PLAN.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'sequences':SEQS,'order':ORDER,'cycles':2,'paths':['detect','tracking'],'modes':['single','queued'],
  'frames_total':140,'warmup':20,'measured_per_session':120,'sessions':96,
  'fresh_processes_per_arm_path_mode_sequence':4,'candidate_slots':2,'motion_slots_tracking':2,'result_slots':4,'ring_slots':4,'tracking_graph':True,'detection_graph':False,
  'endpoint':'receive return before caller copy/check; input JPEG preloaded, initialization/warmup excluded',
  'statistics':'pooled per-frame nearest-rank percentiles, 480 delivery samples per group; interval is throughput',
  'GPU_lock':'held across the whole series; root approval after all builds and quality tests'})
 runs=[]
 with (Path('/home/ray/.cache/saccade-v1/gpu.lock').open('a') if args.external_lock_fd is None else os.fdopen(os.dup(args.external_lock_fd),'a')) as lock:
  assert os.fstat(lock.fileno()).st_ino==Path('/home/ray/.cache/saccade-v1/gpu.lock').stat().st_ino
  fcntl.flock(lock,fcntl.LOCK_EX if args.external_lock_fd is None else fcntl.LOCK_EX|fcntl.LOCK_NB)
  for seq in SEQS:
   for path in ['detect','tracking']:
    for mode in ['single','queued']:
     for cycle in range(2):
      for ordinal,arm in enumerate(ORDER):
       dest=out/f'{seq}-{path}-{mode}-c{cycle}-o{ordinal}-{arm}'
       cmd=[str(BIN),path,str(ENGINES[arm]),str(DATA/seq),str(dest),'--frames','140','--warmup','20','--mode',mode]
       started=time.monotonic();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
       with (out/(dest.name+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
       rec={'sequence':seq,'path':path,'mode':mode,'cycle':cycle,'ordinal':ordinal,'arm':arm,'command':cmd,'output':str(dest),'started_utc':stamp,'wall_seconds_including_initialization':time.monotonic()-started,'exit_code':p.returncode};runs.append(rec);save(out/'runs.json',runs);print(json.dumps(rec),flush=True)
       assert p.returncode==0,(dest,'pipeline process failed')
  if args.external_lock_fd is None:fcntl.flock(lock,fcntl.LOCK_UN)
 after={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths};save(out/'inputs-after.json',after);assert before==after,'engine/input/harness changed'
 aggregate(out)

if __name__=='__main__':main()
