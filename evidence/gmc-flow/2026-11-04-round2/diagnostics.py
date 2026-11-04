from pathlib import Path
import csv,math,collections
root=Path.cwd();b=root/'build/gmc-flow2';p=b/'pp22';e=root/'evidence/gmc-flow/2026-11-04-round2'
strata={r['sequence']:r['stratum'] for r in csv.DictReader((e/'stratification.csv').open())}
arms=('OFF','G0','G1','L2','LD','R','RC')
class Stats:
 def __init__(self):
  self.n=0;self.x=self.y=0.;self.norm=[];self.l2=[];self.r=[];self.fb=self.rfb=self.neighbor=0;self.alpha=[0,0,0];self.active_alpha=[0,0,0];self.rc_fallback=0
 def add(self,row,arm):
  self.n+=1;dx=float(row['dx']);dy=float(row['dy']);self.x+=dx;self.y+=dy;self.norm.append(math.hypot(dx,dy))
  self.fb+=int(row['fallback']);self.rfb+=int(row['r_fallback']);self.neighbor+=int(row['row_offset'])!=0
  if arm=='RC':
   a=float(row['alpha']);i=0 if a==0 else 2 if a==1 else 1;self.alpha[i]+=1
   fallback=int(row['local_cells'])<8;self.rc_fallback+=fallback
   if not fallback:self.active_alpha[i]+=1
   self.l2.append(float(row['delta_l2']));self.r.append(float(row['delta_r']))
def q(v,p):
 if not v:return 0
 pos=(len(v)-1)*p;lo=math.floor(pos);hi=math.ceil(pos);return v[lo]+(v[hi]-v[lo])*(pos-lo)
stats=collections.defaultdict(Stats)
for arm in arms:
 for s in (p/'included.txt').read_text().splitlines():
  groups=[s,'PP-all',strata[s]]
  with (p/'diagnostics'/arm/(s+'.csv')).open() as f:
   for row in csv.DictReader(f):
    for g in groups:stats['PP22',arm,g].add(row,arm)
  for g in groups:stats['PP22',arm,g]
 for n in ('02','04','05','09','10','11','13'):
  s=f'MOT17-{n}-SDP';groups=[s]+(['MOT17-static'] if n in ('02','04','09') else [])
  with (b/'diagnostics/mot17'/arm/(s+'.csv')).open() as f:
   for row in csv.DictReader(f):
    for g in groups:stats['MOT17',arm,g].add(row,arm)
  for g in groups:stats['MOT17',arm,g]
 print('diagnostics parsed',arm,flush=True)
rows=[]
for (dataset,arm,scope),s in sorted(stats.items()):
 s.norm.sort();s.l2.sort();s.r.sort();den=s.n or 1
 row={'dataset':dataset,'arm':arm,'scope':scope,'calls':s.n,'mean_dx':s.x/den,'mean_dy':s.y/den,'mean_norm':sum(s.norm)/den,'p99_norm':q(s.norm,.99),'max_norm':max(s.norm,default=0),'fallback_count':s.fb,'fallback_fraction':s.fb/den}
 if arm in ('R','RC'):row.update(r_fallback_count=s.rfb,r_fallback_fraction=s.rfb/den,neighbor_count=s.neighbor,neighbor_fraction=s.neighbor/den)
 if arm=='RC':
  row.update(alpha_0=s.alpha[0],alpha_mid=s.alpha[1],alpha_1=s.alpha[2],rc_fallback_to_r=s.rc_fallback,active_alpha_0=s.active_alpha[0],active_alpha_mid=s.active_alpha[1],active_alpha_1=s.active_alpha[2])
  for name,values in (('delta_l2',s.l2),('delta_r',s.r)):
   row.update({name+'_mean':sum(values)/den,name+'_p50':q(values,.5),name+'_p90':q(values,.9),name+'_p99':q(values,.99),name+'_max':max(values,default=0)})
 rows.append(row)
fields=list(dict.fromkeys(k for r in rows for k in r))
with (e/'diagnostics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
with (e/'static-controls.csv').open('w',newline='') as f:
 fields=['dataset','arm','scope','calls','mean_dx','mean_dy','mean_norm','p99_norm','max_norm'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(r for r in rows if r['scope'] in ('PP-static','MOT17-static','MOT17-02-SDP','MOT17-04-SDP','MOT17-09-SDP'))
print('Diagnostics complete',flush=True)
