from pathlib import Path
import csv,json
root=Path.cwd();e=root/'evidence/gmc-flow/2026-11-04-round2'
rows=list(csv.DictReader((e/'scores.csv').open()));scores={(r['arm'],r['set']):{k:float(r[k]) for k in ('HOTA','AssA')} for r in rows}
def delta(arm,base,group,metric='HOTA'):return scores[arm,group][metric]-scores[base,group][metric]
a={'rule':'A','arm':'G1','moving_delta_hota':delta('G1','G0','PP-moving'),'moving_delta_assa':delta('G1','G0','PP-moving','AssA'),'static_delta_hota':delta('G1','G0','PP-static')}
a['pass']=a['moving_delta_hota']>=0 and a['moving_delta_assa']>=0 and a['static_delta_hota']>=-0.2
b=[]
for arm in ('L2','R','RC'):
 r={'rule':'B','arm':arm,'moving_delta_hota':delta(arm,'G1','PP-moving'),'moving_delta_assa':delta(arm,'G1','PP-moving','AssA'),'static_delta_hota':delta(arm,'G1','PP-static'),'mot10_delta_hota':delta(arm,'G1','MOT17-10')}
 r['condition_1']=r['moving_delta_hota']>=0.5 and r['moving_delta_assa']>=0
 r['condition_2']=r['static_delta_hota']>=-0.2;r['condition_3']=r['mot10_delta_hota']>=0;r['pass']=all(r[k] for k in ('condition_1','condition_2','condition_3'));b.append(r)
go=[r['arm'] for r in b if r['pass']];selected=None
if go:
 best=max(scores[arm,'PP-moving']['HOTA'] for arm in go)
 selected=next(arm for arm in ('R','RC','L2') if arm in go and best-scores[arm,'PP-moving']['HOTA']<0.2)
result={'A':a,'B':b,'selected_local':selected}
(e/'verdicts.json').write_text(json.dumps(result,indent=2)+'\n')
seq=list(csv.DictReader((e/'pp-moving-per-sequence.csv').open()));baseline={r['sequence']:float(r['HOTA']) for r in seq if r['arm']=='G1'}
with (e/'pp-moving-win-loss.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['arm','wins','losses','ties'])
 for arm in ('OFF','G0','G1','L2','LD','R','RC'):
  diffs=[float(r['HOTA'])-baseline[r['sequence']] for r in seq if r['arm']==arm];w.writerow([arm,sum(x>0 for x in diffs),sum(x<0 for x in diffs),sum(x==0 for x in diffs)])
print(json.dumps(result,indent=2))
