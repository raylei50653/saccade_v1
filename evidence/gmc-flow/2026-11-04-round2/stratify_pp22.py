from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,shlex,csv,json,datetime,hashlib,shutil
root=Path.cwd();b=root/'build/gmc-flow2';p=b/'pp22';e=root/'evidence/gmc-flow/2026-11-04-round2'
status=json.loads((p/'status.json').read_text());included=status['included'];excluded={x['sequence']:x for x in status['excluded']}
assert status['processed']==74 and len(excluded)<=5
(p/'stratification').mkdir()
def run(s):
 cmd=['timeout','600',str(b/'gmc_flow_offline'),'stratify',str(p/'scdr'/(s+'.scdr')),str(p/'scmo'/(s+'.scmo')),str(p/'flow'/(s+'.scfl')),str(p/'stratification'/(s+'.csv'))]
 result=subprocess.run(cmd,text=True,capture_output=True,check=True)
 (p/'stratification'/(s+'.log')).write_text('$ '+shlex.join(cmd)+'\n'+result.stdout+result.stderr)
 m=float(result.stdout.strip().split('=')[1]);return s,m
with ThreadPoolExecutor(max_workers=8) as pool: scores=dict(pool.map(run,included))
with (e/'stratification.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['sequence','m','stratum','exclusion_reason'])
 for s in sorted([*included,*excluded]):
  if s in excluded:
   x=excluded[s];reason=Path(x['log']).read_text().splitlines()[-1];w.writerow([s,'','EXCLUDED',reason])
  else:w.writerow([s,format(scores[s],'.17g'),'PP-moving' if scores[s]>=0.5 else 'PP-static',''])
counts={k:sum((v>=0.5)==(k=='PP-moving') for v in scores.values()) for k in ('PP-moving','PP-static')}
(e/'C3.log').write_text('C3 PASS\nwritten_before_any_arm_scoring=true\nwritten_at='+datetime.datetime.now().astimezone().isoformat()+'\nstratification_sha256='+hashlib.sha256((e/'stratification.csv').read_bytes()).hexdigest()+'\n'+json.dumps(counts)+'\n')
shutil.copytree(p/'stratification',e/'stratification-frames')
shutil.copytree(p/'logs',e/'extraction')
for src in (p/'extraction.csv',p/'status.json',b/'logs/extraction.log',b/'extract_pp22.py',b/'stratify_pp22.py'):shutil.copy2(src,e/src.name)
print('C3 PASS',counts,'excluded',len(excluded))
