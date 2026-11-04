import subprocess,pathlib,json,time,csv,statistics,math,hashlib
root=pathlib.Path('/home/ray/developer/ai/saccade_v1');b=root/'build/resource-schedule-20261104'
engine='/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine';seq='/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP'
results=[]
for depth in [1,2,4]:
 cmd=[str(b/'resource_schedule_probe'),engine,seq,str(b/('depth-'+str(depth))),'--frames','120','--warmup','20','--only','queued','--result-slots',str(depth)]
 (b/('depth-'+str(depth)+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (b/('depth-'+str(depth)+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise SystemExit(r.returncode)
 path=b/('depth-'+str(depth));rows=[r for r in csv.DictReader((path/'frames.csv').open()) if int(r['frame'])>=20]
 a=sorted(float(r['enqueue_to_receive_us']) for r in rows);runs=list(csv.DictReader((path/'runs.csv').open()));run=runs[0]
 sha=hashlib.sha256((path/'queued.bin').read_bytes()).hexdigest()
 results.append(dict(result_slots=depth,samples=len(a),mean_latency_us=statistics.mean(a),p50_latency_us=a[math.ceil(.5*len(a))-1],p95_latency_us=a[math.ceil(.95*len(a))-1],p99_latency_us=a[math.ceil(.99*len(a))-1],wall_us_per_frame=float(run['measured_wall_us'])/120,packet_sha256=sha,returncode=r.returncode))
 print(results[-1],flush=True)
 (b/'queue-depth-summary.json').write_text(json.dumps(dict(schema='queue-depth/1',status='PASS' if len({x['packet_sha256'] for x in results})==1 else 'FAIL',scope='one fresh unprofiled session per depth; observational comparison without ABBA; preloaded JPEG; excludes disk/output',results=results),indent=2)+'\n')
