import subprocess,pathlib,json,time
root=pathlib.Path('/home/ray/developer/ai/saccade_v1')
b=root/'build/resource-schedule-20261104'
engine='/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'
seq='/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP'
probe=[str(b/'resource_schedule_probe'),engine,seq]
commands=[
 ('probe',probe+[str(b/'probe'),'--frames','120','--warmup','20']),
 ('trace',['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--cuda-memory-usage=true','--cuda-event-trace=true','-o',str(b/'trace')]+probe+[str(b/'trace-probe'),'--frames','120','--warmup','20']),
 ('export',['nsys','export','--type','sqlite','-o',str(b/'trace.sqlite'),str(b/'trace.nsys-rep')]),
]
results=[]
for name,cmd in commands:
 (b/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 sampler=None
 with (b/(name+'.log')).open('w') as log, (b/(name+'-gpu.csv')).open('w') as gpu:
  if name=='probe':
   sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,memory.used,utilization.gpu','--format=csv','-lms','100'],stdout=gpu,stderr=subprocess.DEVNULL)
  try:
   start=time.time()
   r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  finally:
   if sampler is not None:
    sampler.terminate();sampler.wait()
 results.append(dict(name=name,returncode=r.returncode,elapsed_seconds=time.time()-start))
 (b/'probe-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(results[-1],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
