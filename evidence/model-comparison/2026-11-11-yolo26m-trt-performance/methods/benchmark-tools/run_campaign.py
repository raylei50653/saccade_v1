"""Execute the authorized fixed model and pipeline plans under one GPU lock."""
from pathlib import Path
import datetime,fcntl,json,subprocess,time

HERE=Path(__file__).resolve().parent
result=[]
with Path('/home/ray/.cache/saccade-v1/gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for tool,name in [('run_model.py','model-runs'),('run_pipeline.py','pipeline-runs')]:
  cmd=['python','-B',str(HERE/tool),str(HERE/name),'--external-lock-fd',str(lock.fileno())]
  started=time.monotonic()
  stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
  with (HERE/(name+'.driver.log')).open('w') as f:
   p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
  rec={'tool':tool,'output':str(HERE/name),'started_utc':stamp,'seconds':time.monotonic()-started,'command':cmd,'exit_code':p.returncode}
  result.append(rec);(HERE/'campaign.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(rec),flush=True)
  assert p.returncode==0,rec
 fcntl.flock(lock,fcntl.LOCK_UN)
