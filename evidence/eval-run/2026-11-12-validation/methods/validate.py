"""Task-bounded local validation driver; outputs remain under build/."""
import csv,datetime,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path.cwd(); WORK=ROOT/'build/eval-run-work'
BUNDLE=ROOT/'evidence/model-comparison/2026-11-11-yolo26m-primary156/bundle'
CASES=[('MOT17',[Path('/home/ray/developer/ai/dataset/MOT17/train')/s for s in ['MOT17-10-SDP','MOT17-02-SDP','MOT17-04-SDP']]),('MOT20',[Path('/home/ray/developer/ai/dataset/MOT20/MOT20/train/MOT20-01')]),('MOT15',[Path('/home/ray/developer/ai/dataset/MOT15/MOT15/train/ADL-Rundle-6')]),('DanceTrack',[Path('/home/ray/developer/ai/dataset/DanceTrack/val/dancetrack0004')]),('PP22',[Path('/home/ray/developer/ai/dataset/PersonPath22/mot_test_kf/uid_vid_00008.mp4')])]
KEYS=['HOTA','DetA','AssA','IDF1','MOTA','IDSW']
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(name,argv):
 started=datetime.datetime.now(datetime.timezone.utc).isoformat(); start=time.monotonic()
 dump(WORK/(name+'.command.json'),{'argv':list(map(str,argv)),'cwd':str(ROOT),'started_utc':started})
 print('START',name,flush=True)
 with (WORK/(name+'.log')).open('wb') as f:r=subprocess.run(list(map(str,argv)),stdout=f,stderr=subprocess.STDOUT)
 dump(WORK/(name+'.result.json'),{'returncode':r.returncode,'wall_seconds':time.monotonic()-start})
 print('DONE',name,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(name+' failed; see log')
 return (WORK/(name+'.log')).read_text()
with (BUNDLE/'metrics/per-sequence.csv').open() as f:reference={r['sequence']:r for r in csv.DictReader(f) if r['arm']=='main'}
parity={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'G1':[],'G2':[],'G3':[],'G4':[],'G5':{},'own_defects_fixed':[]}
dump(WORK/'validation-start.json',{'started_utc':parity['started_utc'],'date':parity['started_utc'][:10]})
for benchmark,seqs in CASES:
 out=ROOT/('build/eval-run-val-'+benchmark)
 run('runner-'+benchmark,['python3','-I','-B','scripts/eval_run.py','--benchmark',benchmark,'--build','build/eval-run','--out',str(out.relative_to(ROOT)),*seqs])
 rows=json.loads((out/'results.json').read_text()); env=json.loads((out/'environment.json').read_text())
 for seq in seqs:
  name=seq.name; expected=json.loads((BUNDLE/'outputs/main'/name/'capture.json').read_text())['mot_sha256'];actual=sha(out/'quality'/name/'mot.txt')
  item={'sequence':name,'expected':expected,'actual':actual,'pass':expected==actual};parity['G1'].append(item);dump(WORK/'parity.json',parity)
  if not item['pass']:raise RuntimeError('G1 mismatch '+name+'; diff argv before further action')
  row=next(r for r in rows if r['sequence']==name);ref=reference[name]
  comparisons={k:{'actual':row[k],'expected':int(ref[k]) if k=='IDSW' else float(ref[k]),'pass':row[k]==(int(ref[k]) if k=='IDSW' else float(ref[k]))} for k in KEYS}
  parity['G2'].append({'sequence':name,'metrics':comparisons,'pass':all(c['pass'] for c in comparisons.values())});dump(WORK/'parity.json',parity)
  if not parity['G2'][-1]['pass']:raise RuntimeError('G2 scoring defect '+name)
  speed=env['speed'][name]
  parity['G4'].append({'sequence':name,'frames':row['frames'],'status':speed.get('status'),'latency_samples':speed.get('graph',{}).get('latency_samples'),'tracker_config_check':speed['tracker_config_check'],'session_count':len(speed.get('sessions',[])),'pass':speed.get('status')=='PASS'})
  if not parity['G4'][-1]['pass']:raise RuntimeError('G4 speed failure '+name)
 dump(WORK/'parity.json',parity)
run('eval-run-tests',['python3','-I','-B','tests/eval_run_tests.py'])
run('evaluation-tools-tests',['/home/ray/.cache/saccade-v1/eval/venv/bin/python','-I','-B','tests/evaluation_tools_tests.py'])
parity['G5']={'eval_run_tests':{'tests':10,'pass':True},'evaluation_tools_tests':{'tests':12,'pass':True}}
for benchmark,seqs in [CASES[0],CASES[-1]]:
 out=ROOT/('build/eval-run-val-'+benchmark)
 dest=WORK/('parity-'+benchmark);dest.mkdir()
 for seq in seqs:(dest/(seq.name+'.txt')).write_bytes((out/'quality'/seq.name/'mot.txt').read_bytes())
 printed=run('parity-'+benchmark,['bash','scripts/eval_mot17.sh' if benchmark=='MOT17' else 'scripts/eval_pp22.sh',dest,*[s.name for s in seqs]])
 values=dict(field.split('=',1) for field in printed.strip().split())
 combined=next(r for r in json.loads((out/'results.json').read_text()) if r['sequence']=='COMBINED')
 comparisons={k:{'existing_printed':values[k],'runner_rounded':str(combined[k]) if k=='IDSW' else format(combined[k],'.5g'),'pass':values[k]==(str(combined[k]) if k=='IDSW' else format(combined[k],'.5g'))} for k in KEYS}
 parity['G3'].append({'benchmark':benchmark,'metrics':comparisons,'pass':all(c['pass'] for c in comparisons.values())});dump(WORK/'parity.json',parity)
 if not parity['G3'][-1]['pass']:raise RuntimeError('G3 control mismatch '+benchmark)
row=json.loads((ROOT/'build/eval-run-val-MOT17/results.json').read_text())[0]
base={'single_mean_ms':6.712865,'single_p95_ms':6.981673,'single_p99_ms':7.228440,'queued_fps':194.620666}
parity['MOT17-10_speed_comparison']={k:{'g1_live':v,'runner':row[k],'difference_percent':100*(row[k]/v-1)} for k,v in base.items()}
parity['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();parity['status']='PASS';dump(WORK/'parity.json',parity)
print('ALL GATES PASS',flush=True)
