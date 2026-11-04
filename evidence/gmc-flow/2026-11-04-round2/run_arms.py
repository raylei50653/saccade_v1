from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,shlex,datetime,csv,threading
root=Path.cwd();b=root/'build/gmc-flow2';p=b/'pp22';tasks=[]
for arm in ('OFF','G1','L2','LD','R','RC'):
 for s in (p/'included.txt').read_text().splitlines():
  tasks.append(('pp22',arm,s,p/'scdr'/(s+'.scdr'),p/'scmo'/(s+'.scmo'),p/'flow'/(s+'.scfl'),p/'arms'/arm/(s+'.txt'),p/'diagnostics'/arm/(s+'.csv')))
for arm in ('OFF','LD','R','RC'):
 for n in ('02','04','05','09','10','11','13'):
  s=f'MOT17-{n}-SDP';tasks.append(('mot17',arm,s,root.parent/'m4-formal/build-m4-evaluation/scdr'/(s+'.scdr'),root.parent/'m1-data'/(s+'.scmo'),root/'build/gmc-flow/flow'/(s+'.scfl'),b/'mot17'/arm/(s+'.txt'),b/'diagnostics/mot17'/arm/(s+'.csv')))
(b/'arm-logs').mkdir();lock=threading.Lock();failed=threading.Event()
with (b/'tracking-runs.csv').open('w',buffering=1) as csvfile:
 writer=csv.writer(csvfile,lineterminator='\n');writer.writerow(['dataset','arm','sequence','started','finished','exit','command'])
 def run(task):
  if failed.is_set():return
  dataset,arm,s,scdr,scmo,flow,mot,diag=task
  mot.parent.mkdir(parents=True,exist_ok=True);diag.parent.mkdir(parents=True,exist_ok=True)
  cmd=['timeout','3600',str(b/'gmc_flow_offline'),'track',str(scdr),str(scmo),str(flow),arm,str(mot),str(diag)]
  started=datetime.datetime.now().astimezone().isoformat()
  with (b/'arm-logs'/f'{dataset}-{arm}-{s}.log').open('w') as log:
   log.write('$ '+shlex.join(cmd)+'\n');log.flush();result=subprocess.run(cmd,stdout=log,stderr=log)
  if result.returncode:failed.set()
  with lock:
   writer.writerow([dataset,arm,s,started,datetime.datetime.now().astimezone().isoformat(),result.returncode,shlex.join(cmd)])
   print(f'{dataset} {arm} {s} exit={result.returncode}',flush=True)
  if result.returncode:raise RuntimeError('tracking failed: '+dataset+' '+arm+' '+s)
 with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(run,tasks))
print(f'PASS {len(tasks)} tracking commands; plus 69 C1 G0 and 21 C0 MOT17 outputs',flush=True)
