from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,shlex,hashlib
root=Path.cwd();b=root/'build/gmc-flow2';p=b/'pp22';e=root/'evidence/gmc-flow/2026-11-04-round2'
seqs=(p/'included.txt').read_text().splitlines()
for d in ('arms/G0','stock-G0','diagnostics/G0','C1-logs'):(p/d).mkdir(parents=True,exist_ok=True)
def run(s):
 mot=p/'arms/G0'/(s+'.txt');stock=p/'stock-G0'/(s+'.txt')
 commands=[['timeout','600',str(b/'gmc_flow_offline'),'track',str(p/'scdr'/(s+'.scdr')),str(p/'scmo'/(s+'.scmo')),str(p/'flow'/(s+'.scfl')),'G0',str(mot),str(p/'diagnostics/G0'/(s+'.csv'))],['timeout','600',str(b/'track_offline'),str(p/'scdr'/(s+'.scdr')),str(p/'scmo'/(s+'.scmo')),str(stock)]]
 with (p/'C1-logs'/(s+'.log')).open('w') as log:
  for cmd in commands:
   log.write('$ '+shlex.join(cmd)+'\n');log.flush();subprocess.run(cmd,stdout=log,stderr=log,check=True)
  if mot.read_bytes()!=stock.read_bytes():raise RuntimeError('C1 FAIL '+s)
 return f'PASS {s} sha256={hashlib.sha256(mot.read_bytes()).hexdigest()}'
with (e/'C1.log').open('w',buffering=1) as log:
 with ThreadPoolExecutor(max_workers=8) as pool:
  for result in pool.map(run,seqs):log.write(result+'\n');print(result.split(' sha256=')[0],flush=True)
 log.write(f'C1 PASS {len(seqs)}/{len(seqs)} byte-identical MOT files; 5 excluded under PLAN-2 section 2.4\n')
