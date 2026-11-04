from pathlib import Path
import subprocess, json, csv, hashlib, statistics, math, time
root=Path('build/f02-opt').resolve()
out=root/'pipeline-abba-final'
out.mkdir()
engine=Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')
sequence=Path('/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP')
binaries={'baseline':root/'baseline-tracking-benchmark','candidate':root/'cuda/tracking_benchmark'}
records=[]
commands=[]
started=time.time()
for round_id in range(2):
    for position,backend in enumerate(['baseline','candidate','candidate','baseline']):
        dest=out/f'r{round_id}-{position}-{backend}'
        cmd=[str(binaries[backend]),str(engine),str(sequence),str(dest),'--frames','120','--warmup','20','--rounds','1','--only','graph']
        commands.append(cmd)
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        print('START',round_id,position,backend,flush=True)
        with (out/f'r{round_id}-{position}-{backend}.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        (out/f'r{round_id}-{position}-{backend}.exit').write_text(str(result.returncode)+'\n')
        result.check_returncode()
        for row in csv.DictReader((dest/'runs.csv').open()):
            row.update(outer_round=round_id,outer_position=position,backend=backend,run_directory=dest.name)
            records.append(row)
        print('DONE',round_id,position,backend,json.loads((dest/'summary.json').read_text())['groups'],flush=True)
fields=list(records[0])
with (out/'runs.csv').open('w') as file:
    writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(records)
hashes={r['packet_sha256'] for r in records}
assert len(hashes)==1,hashes
input_manifests={hashlib.sha256((out/r['run_directory']/'inputs.sha256').read_bytes()).hexdigest() for r in records}
assert len(input_manifests)==1
summary={'capture_kind':'development-optimization','frames_per_session':140,'warmup_frames':20,'measured_frames':120,'outer_abba_rounds':2,'sessions':len(records),'packet_sha256':next(iter(hashes)),'packet_hashes_all_equal':True,'input_manifest_sha256':next(iter(input_manifests)),'engine_sha256':hashlib.sha256(engine.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-started,'measurements':{}}
for backend in binaries:
    single=[r for r in records if r['backend']==backend and r['style']=='single']
    queued=[r for r in records if r['backend']==backend and r['style']=='queued']
    latencies=[]
    for r in single:
        name=f"r{r['round']}-{r['ordinal']}-graph-single.csv"
        latencies += [float(x['enqueue_to_receive_us']) for x in csv.DictReader((out/r['run_directory']/name).open())]
    values=sorted(latencies)
    summary['measurements'][backend]={'single_mean_us':statistics.mean(latencies),'single_p50_us':values[math.ceil(.50*len(values))-1],'single_p95_us':values[math.ceil(.95*len(values))-1],'single_p99_us':values[math.ceil(.99*len(values))-1],'latency_samples':len(latencies),'queued_mean_us_per_frame':statistics.mean(float(r['wall_us_per_frame']) for r in queued),'queued_runs_us_per_frame':[float(r['wall_us_per_frame']) for r in queued]}
a=summary['measurements']['baseline'];b=summary['measurements']['candidate']
summary['comparison']={'single_mean_reduction_percent':100*(1-b['single_mean_us']/a['single_mean_us']),'queued_interval_reduction_percent':100*(1-b['queued_mean_us_per_frame']/a['queued_mean_us_per_frame'])}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
