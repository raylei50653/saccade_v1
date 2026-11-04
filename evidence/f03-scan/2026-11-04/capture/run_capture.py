import pathlib, subprocess, shlex, time, json, sys
root=pathlib.Path(__file__).resolve().parent
name=sys.argv[1]
args=sys.argv[2:]
(root/(name+'.command')).write_text(shlex.join(args)+'\n')
start=time.time()
with (root/(name+'.log')).open('wb') as log:
    result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
(root/(name+'.exit')).write_text(str(result.returncode)+'\n')
(root/(name+'.time.json')).write_text(json.dumps({'start':start,'end':time.time(),'exit':result.returncode},indent=2)+'\n')
print(name, 'exit', result.returncode, flush=True)
sys.exit(result.returncode)
