from pathlib import Path
import subprocess,json,time
root=Path('build/f02-opt').resolve()
out=root/'sanitizers'
out.mkdir()
results=[]
for name in ['memcheck','initcheck','racecheck','synccheck']:
    cmd=['timeout','--signal=TERM','--kill-after=10s','240s','/opt/cuda/bin/compute-sanitizer','--tool',name,'--report-api-errors','explicit','--error-exitcode','99','--kernel-name','regex=tiled_hr','--launch-count','3',str(root/'final-resize-benchmark'),str(out/name),'--verify-only','--shape','1920','1080','1280','768']
    (out/f'{name}.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    print('START',name,flush=True)
    start=time.time()
    with (out/f'{name}.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    results.append({'tool':name,'exit_code':result.returncode,'elapsed_seconds':time.time()-start,'instrumented_kernel':'tiled_hr','instrumented_launches':3,'scope':'first synthetic random case candidate ordinary/repeat/graph; baseline and other kernels excluded'})
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('DONE',results[-1],flush=True)
    result.check_returncode()
