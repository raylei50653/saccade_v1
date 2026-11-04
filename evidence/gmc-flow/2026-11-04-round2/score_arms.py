from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,csv,shlex,shutil,datetime,threading,json
root=Path.cwd();b=root/'build/gmc-flow2';p=b/'pp22';e=root/'evidence/gmc-flow/2026-11-04-round2'
strata=list(csv.DictReader((e/'stratification.csv').open()));arms=('OFF','G0','G1','L2','LD','R','RC')
sets={k:[r['sequence'] for r in strata if r['stratum']==k] for k in ('PP-moving','PP-static')}
sets['PP-all']=[r['sequence'] for r in strata if r['stratum']!='EXCLUDED']
sets.update({'MOT17-10':['MOT17-10-SDP'],'in-domain':[f'MOT17-{n}-SDP' for n in ('04','05','09','11','13')],'MOT17-02':['MOT17-02-SDP']})
tasks=[(arm,s,seqs) for arm in arms for s,seqs in sets.items()];failed=threading.Event()
def run(task):
 if failed.is_set(): return
 arm,s,seqs=task;pp=s.startswith('PP-');out=b/'scoring'/arm/s;out.mkdir(parents=True)
 for seq in seqs:
  source=(p/'arms'/arm if pp else b/'mot17'/arm)/(seq+'.txt')
  (out/source.name).symlink_to(source)
 script=root/'scripts'/('eval_pp22.sh' if pp else 'eval_mot17.sh')
 cmd=['timeout','1800',str(script),str(out),*seqs]
 start=datetime.datetime.now().astimezone().isoformat();result=subprocess.run(cmd,text=True,capture_output=True)
 archive=e/'trackeval'/arm/s;archive.mkdir(parents=True)
 (archive/'command.txt').write_text('$ '+shlex.join(cmd)+'\nstarted='+start+'\nfinished='+datetime.datetime.now().astimezone().isoformat()+'\nexit='+str(result.returncode)+'\n')
 (archive/'score.log').write_text(result.stdout+result.stderr)
 te=out/'trackeval'
 if (te/'trackeval.log').exists():shutil.copy2(te/'trackeval.log',archive/'trackeval.log')
 if result.returncode:
  failed.set();raise RuntimeError('scoring failed '+arm+' '+s+': '+result.stderr)
 for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):shutil.copy2(te/'results/v1'/name,archive/name)
 if pp:shutil.copy2(te/'keyframe-filter.csv',archive/'keyframe-filter.csv')
 print(arm,s,result.stdout.strip(),flush=True)
 return arm,s,seqs,list(csv.DictReader((archive/'pedestrian_detailed.csv').open()))
with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(run,tasks))
fields=['arm','set','HOTA','DetA','AssA','IDF1','MOTA','IDSW']
def metrics(r):return {**{k:float(r[k+'___AUC'])*100 for k in ('HOTA','DetA','AssA')},**{k:float(r[k])*100 for k in ('IDF1','MOTA')},'IDSW':int(r['IDSW'])}
scores=[];seqscores=[]
for arm,s,seqs,rows in results:
 combined=next(r for r in rows if r['seq']=='COMBINED');scores.append({'arm':arm,'set':s,**metrics(combined)})
 if s=='PP-moving':
  for r in rows:
   if r['seq']!='COMBINED':seqscores.append({'arm':arm,'sequence':r['seq'],**metrics(r)})
for filename,header,rows in [('scores.csv',fields,scores),('pp-moving-per-sequence.csv',['arm','sequence',*fields[2:]],seqscores)]:
 with (e/filename).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=header,lineterminator='\n');w.writeheader();w.writerows(rows)
print('Scoring complete 42 arm/set combinations',flush=True)
