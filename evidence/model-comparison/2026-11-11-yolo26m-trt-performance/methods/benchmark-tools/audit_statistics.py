"""Independent CPU audit of a completed fixed performance campaign."""
from pathlib import Path
import csv,hashlib,json,math

HERE=Path(__file__).resolve().parent

def load(p):return json.loads(p.read_text())
def read(p):return list(csv.DictReader(p.open()))
def close(a,b):assert abs(a-b)<=1e-7,(a,b)
def desc(values):
 values=sorted(values);n=len(values);assert n and all(math.isfinite(v) and v>0 for v in values)
 return {'mean':math.fsum(values)/n,**{k:values[max(1,math.ceil(n*q))-1] for k,q in [('p50',.5),('p95',.95),('p99',.99)]},'samples':n}
def verify_desc(actual,expected):
 assert actual['samples']==expected['samples']
 for k in ['mean','p50','p95','p99']:close(actual[k],expected[k])

model=HERE/'model-runs';mruns=load(model/'runs.json')
assert len(mruns)==20 and all(r['exit_code']==0 for r in mruns)
expected_model=[(comparison,cycle,ordinal,arm) for comparison,arms in [('raw6',['main','official','finetuned','finetuned','official','main']),('original-default',['main','official-default','official-default','main'])] for cycle in range(2) for ordinal,arm in enumerate(arms)]
assert [(r['comparison'],r['cycle'],r['ordinal'],r['arm']) for r in mruns]==expected_model
model_data={};memory={}
for r in mruns:
 p=Path(r['output']);s=load(p/'summary.json');assert s['status']=='PASS' and s['ordinary_repeat_and_graph_bits_equal'] and s['frames_verified']==4
 assert (s['iterations_per_group'],s['warmup_per_group'],s['queued_replays_per_group'],s['rounds'])==(40,12,64,1)
 mem=load(p/'memory.json');memory.setdefault(r['arm'],[]).append(mem)
 for row in read(p/'samples.csv'):
  g=model_data.setdefault((r['comparison'],r['arm'],row['mode']),{'gpu':[],'cpu':[],'qgpu':[],'qcpu':[],'replays':0,'runs':set()})
  g['gpu'].append(float(row['gpu_event_ms']));g['cpu'].append(float(row['cpu_wall_ms']));g['runs'].add(str(p))
 for row in read(p/'queued.csv'):
  g=model_data[(r['comparison'],r['arm'],row['mode'])];g['qgpu'].append(float(row['gpu_event_total_ms']));g['qcpu'].append(float(row['cpu_wall_total_ms']));g['replays']+=int(row['replays'])
mg=load(model/'summary.json')['groups'];assert len(mg)==len(model_data)==10
for r in mg:
 g=model_data[(r['comparison'],r['arm'],r['mode'])];assert len(g['runs'])==4 and len(g['gpu'])==len(g['cpu'])==1280 and len(g['qgpu'])==32 and g['replays']==2048
 verify_desc(r['gpu_event_single_ms'],desc(g['gpu']));verify_desc(r['cpu_wall_single_ms'],desc(g['cpu']))
 close(r['queued_gpu_interval_ms'],math.fsum(g['qgpu'])/g['replays']);close(r['queued_cpu_interval_ms'],math.fsum(g['qcpu'])/g['replays'])
 close(r['queued_gpu_fps'],1000*g['replays']/math.fsum(g['qgpu']));close(r['queued_cpu_fps'],1000*g['replays']/math.fsum(g['qcpu']))

pipeline=HERE/'pipeline-runs';pruns=load(pipeline/'runs.json');assert len(pruns)==96 and all(r['exit_code']==0 for r in pruns)
expected_pipeline=[(seq,path,mode,cycle,ordinal,arm) for seq in ['MOT17-10-SDP','MOT17-02-SDP'] for path in ['detect','tracking'] for mode in ['single','queued'] for cycle in range(2) for ordinal,arm in enumerate(['main','official','finetuned','finetuned','official','main'])]
assert [(r['sequence'],r['path'],r['mode'],r['cycle'],r['ordinal'],r['arm']) for r in pruns]==expected_pipeline
pipeline_data={};hashes={}
for r in pruns:
 p=Path(r['output']);s=load(p/'summary.json');assert s['status']=='PASS' and (s['frames'],s['warmup'],s['samples'],s['candidate_slots'],s['motion_slots'],s['result_slots'])==(140,20,120,2,2 if r['path']=='tracking' else 0,4)
 assert s['tracking_graph']==(r['path']=='tracking')
 g=pipeline_data.setdefault((r['sequence'],r['path'],r['arm'],r['mode']),{'samples':[],'intervals':[],'runs':set()})
 rows=[x for x in read(p/'delivery.csv') if x['warmup']=='0'];assert [int(x['frame']) for x in rows]==list(range(20,140))
 g['samples'] += [float(x['delivery_ms']) for x in rows];g['intervals'].append(s['interval_ms_per_frame']);g['runs'].add(str(p))
 hashes.setdefault((r['sequence'],r['path'],r['arm']),set()).add(s['result_sha256'])
pg=load(pipeline/'summary.json')['groups'];assert len(pg)==len(pipeline_data)==24
for r in pg:
 g=pipeline_data[(r['sequence'],r['path'],r['arm'],r['mode'])];assert len(g['runs'])==4 and len(g['samples'])==480 and len(g['intervals'])==4
 verify_desc(r['delivery_ms'],desc(g['samples']));close(r['interval_ms_per_frame'],math.fsum(g['intervals'])/4);close(r['throughput_fps'],1000/(math.fsum(g['intervals'])/4))
 assert len(hashes[(r['sequence'],r['path'],r['arm'])])==1
memory_summary={}
for arm,rows in memory.items():
 fields=['engine_plan_bytes','engine_getDeviceMemorySizeV2_bytes','explicit_IO_and_status_allocations_bytes']
 for k in fields:assert len({r[k] for r in rows})==1
 memory_summary[arm]={k:rows[0][k] for k in fields}

result={'status':'PASS','model_sessions':20,'model_groups':10,'model_single_samples':12800,'model_queued_batches':320,'model_queued_replays':20480,
 'pipeline_sessions':96,'pipeline_groups':24,'pipeline_measured_frames':11520,'pipeline_total_frames_including_warmup':13440,
 'all_percentiles':'nearest rank recomputed from pooled individual samples',
 'queued_intervals':'recomputed from batches or equal-frame process wall intervals; never labelled per-frame latency',
 'result_hashes':'same within model/sequence/path across single/queued and fresh repeats, not compared across detectors',
 'exact_memory_fields_stable':memory_summary,'memory_delta_caution':'cudaMemGetInfo free bytes are device-level staged observations, not private allocation attribution',
 'scope':'independent CPU reading only; no new inference or measurement'}
(HERE/'statistics-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
