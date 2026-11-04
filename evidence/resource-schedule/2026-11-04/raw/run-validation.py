import subprocess,pathlib,json,time
root=pathlib.Path('/home/ray/developer/ai/saccade_v1');b=root/'build/resource-schedule-20261104'
t01=str(root/'build/t01-f03-evidence/stage/model.engine')
seq='/home/ray/developer/ai/saccade/datasets/MOT17/train'
records='/home/ray/developer/ai/saccade_v1-worktrees/m1-data'
test=[str(b/'tracking_session_tests'),t01,seq,records,'MOT17-10-SDP']
hr=[str(b/'resource_schedule_probe'),'/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine',seq+'/MOT17-10-SDP']
commands=[('protocol',[str(b/'protocol_model'),'--self-test']),('session-sequence',test+['--limit','120']),('session-lifecycle',test+['--lifecycle']),('f04-hr',[str(b/'postprocess_cuda_tests'),'--profile','hr-1280x768']),('tracking',[str(b/'tracking_cuda_tests')]),('motion',[str(b/'motion_cuda_tests')])]
custom='_tracking_session_cu_|_tracking_cu_|_motion_cu_|_postprocess_cu_|_model_producer_cu_|_resize_cu_|_selective_scan_cu_'
for tool in ['memcheck','racecheck','synccheck']:
 cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99']
 if tool!='memcheck':cmd+=['--kernel-name','regex='+custom]
 commands.append(('hr-'+tool,cmd+hr+[str(b/('hr-'+tool)),'--frames','30','--warmup','2','--only','queued']))
results=[]
for name,cmd in commands:
 (b/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 start=time.time()
 with (b/(name+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 results.append(dict(name=name,returncode=r.returncode,elapsed_seconds=time.time()-start))
 (b/'validation-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(results[-1],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
