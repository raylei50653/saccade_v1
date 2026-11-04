import subprocess,pathlib,json,time
root=pathlib.Path('/home/ray/developer/ai/saccade_v1');b=root/'build/resource-schedule-20261104';results=[]
for name,binary,pattern,extra in [('tracking-racecheck','tracking_cuda_tests','_tracking_cu_',[]),('motion-racecheck','motion_cuda_tests','_motion_cu_',['--synthetic'])]:
 cmd=['systemd-run','--user','--wait','--pipe','--collect','--unit=saccade-resource-'+name,'--working-directory='+str(root),'-p','MemoryHigh=6G','-p','MemoryMax=8G','-p','MemorySwapMax=0','-p','RuntimeMaxSec=150','/usr/bin/timeout','-k','10','120','/opt/cuda/bin/compute-sanitizer','--tool','racecheck','--error-exitcode','99','--racecheck-num-workers','1','--kernel-name','regex='+pattern,str(b/binary)]+extra
 (b/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n');start=time.time()
 with (b/(name+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 log=(b/(name+'.log')).read_text();passed=r.returncode==0 and 'checks passed' in log and 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in log
 results.append(dict(name=name,status='PASS' if passed else 'INCOMPLETE',returncode=r.returncode,elapsed_seconds=time.time()-start))
 (b/'component-race-results.json').write_text(json.dumps(results,indent=2)+'\n');print(results[-1],flush=True)
