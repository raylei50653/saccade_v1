import subprocess,pathlib,json,time
root=pathlib.Path('/home/ray/developer/ai/saccade_v1');b=root/'build/resource-schedule-20261104'
test=[str(b/'tracking_session_tests'),str(root/'build/t01-f03-evidence/stage/model.engine'),'/home/ray/developer/ai/saccade/datasets/MOT17/train','/home/ray/developer/ai/saccade_v1-worktrees/m1-data','MOT17-10-SDP','--limit','10']
commands=[]
for tool in ['racecheck','synccheck']:
 base=['/opt/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99']
 if tool=='racecheck':base+=['--racecheck-num-workers','1']
 commands.append(('session-'+tool,base+['--kernel-name','regex=_tracking_session_cu_|_tracking_cu_|_motion_cu_']+test,0))
 commands.append(('f04-'+tool,base+[str(b/'postprocess_cuda_tests'),'--no-held-stream','--profile','hr-1280x768'],0))
 commands.append(('positive-'+tool,base+[str(b/'postprocess_sanitizer_control'),tool],99))
results=[]
for name,command,expected in commands:
 cmd=['systemd-run','--user','--wait','--pipe','--collect','--unit=saccade-resource-'+name,'--working-directory='+str(root),'-p','MemoryHigh=6G','-p','MemoryMax=8G','-p','MemorySwapMax=0','-p','RuntimeMaxSec=180','/usr/bin/timeout','-k','10','150']+command
 (b/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 start=time.time()
 with (b/(name+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 text=(b/(name+'.log')).read_text()
 marker=('Error: Race reported between' if name=='positive-racecheck' else 'Barrier error detected' if name=='positive-synccheck' else 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if name.endswith('racecheck') else 'ERROR SUMMARY: 0 errors')
 results.append(dict(name=name,returncode=r.returncode,expected=expected,marker_found=marker in text,status='PASS' if r.returncode==expected and marker in text else 'FAIL',elapsed_seconds=time.time()-start))
 (b/'bounded-validation-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(results[-1],flush=True)
