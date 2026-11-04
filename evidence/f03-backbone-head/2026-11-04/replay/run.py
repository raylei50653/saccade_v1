import pathlib, subprocess, sys, json, datetime, time, shlex
root=pathlib.Path(__file__).resolve().parent
name=sys.argv[1]; command=sys.argv[2:]
assert name and command
(root/(name+'.command')).write_text(shlex.join(command)+'\n')
start=time.time()
with (root/(name+'.log')).open('w') as f:
 result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
(root/(name+'.exit')).write_text(str(result.returncode)+'\n')
(root/(name+'.time.json')).write_text(json.dumps({'start':start,'elapsed_s':time.time()-start,'exit':result.returncode},indent=2)+'\n')
print(name,'exit',result.returncode,flush=True)
if result.returncode:
 print((root/(name+'.log')).read_text()[-5000:])
sys.exit(result.returncode)
