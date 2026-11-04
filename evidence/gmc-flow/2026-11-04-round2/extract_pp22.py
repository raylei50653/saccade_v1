from pathlib import Path
import csv,subprocess,shlex,datetime,shutil,hashlib,json
root=Path.cwd(); build=root/'build/gmc-flow2'; out=build/'pp22'; source=Path('/home/ray/developer/ai/saccade/datasets/PersonPath22/mot_test_kf')
engine=Path.home()/'.cache/saccade-v1/hr-deploy/model-hr-fp16.engine';lock=Path.home()/'.cache/saccade-v1/gpu.lock'
for d in ('scdr','scmo','flow','logs'): (out/d).mkdir(parents=True,exist_ok=True)
expected='8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876'
actual=hashlib.sha256(engine.read_bytes()).hexdigest()
assert actual==expected
(out/'engine.sha256').write_text(f'{actual}  {engine}\n')
failures=[]; included=[]
with (out/'extraction.csv').open('w',buffering=1) as table:
 writer=csv.writer(table);writer.writerow(['sequence','tool','exit','started','finished','free_bytes','command'])
 for i,seq in enumerate(sorted(source.iterdir()),1):
  s=seq.name
  commands=[('detect',[str(build/'detect_sequence'),str(engine),str(seq),str(out/'scdr'/(s+'.scdr')),'--profile','hr-1280x768']),('motion',[str(build/'motion_sequence'),str(seq),str(out/'scmo'/(s+'.scmo'))]),('flow',[str(build/'flow_sequence'),str(seq),str(out/'flow'/(s+'.scfl')),'--perf','slow'])]
  failed=False
  for tool,args in commands:
   cmd=['flock',str(lock),'timeout','--kill-after=30','1200',*args];started=datetime.datetime.now().astimezone().isoformat()
   print(f'{i}/74 {s} {tool} started',flush=True)
   with (out/'logs'/f'{s}-{tool}.log').open('w') as log:
    log.write('$ '+shlex.join(cmd)+'\n');log.flush()
    result=subprocess.run(cmd,stdout=log,stderr=log)
   writer.writerow([s,tool,result.returncode,started,datetime.datetime.now().astimezone().isoformat(),shutil.disk_usage(build).free,shlex.join(cmd)])
   if result.returncode:
    failures.append({'sequence':s,'tool':tool,'exit':result.returncode,'log':str(out/'logs'/f'{s}-{tool}.log')});failed=True
    print(f'EXCLUDE {s} {tool} exit={result.returncode}',flush=True);break
  if not failed: included.append(s)
  (out/'status.json').write_text(json.dumps({'processed':i,'included':included,'excluded':failures},indent=2)+'\n')
  if len(failures)>5:
   print('STOP: more than five excluded sequences',flush=True);raise SystemExit(1)
(out/'included.txt').write_text(''.join(s+'\n' for s in included))
print(f'Extraction complete included={len(included)} excluded={len(failures)}',flush=True)
