import pathlib,json,csv,sys,hashlib,statistics,math
root=pathlib.Path(__file__).resolve().parent
trial=sys.argv[1]
runs=[]
reference=None
for i,implementation in enumerate(['baseline','candidate','candidate','baseline']):
 name=f'model-{trial}-{i}-{implementation}'
 directory=root/name
 assert (root/(name+'.exit')).read_text().strip()=='0'
 assert not (directory/'FAILURE.txt').exists()
 summary=json.loads((directory/'summary.json').read_text())
 assert summary['status']=='PASS'
 payloads={p.name:p.read_bytes() for p in directory.glob('input-*') if p.suffix in ['.f32','.status']}
 if reference is None: reference=payloads
 assert payloads==reference, name+' raw outputs or statuses changed'
 samples=list(csv.DictReader((directory/'samples.csv').open()))
 runs.append({'name':name,'implementation':implementation,'summary':summary,'samples':samples})
result={'status':'PASS','trial':trial,'formal_acceptance':False,'raw_outputs_and_statuses_exact':True,'frames_per_run':4,'runs':[]}
for r in runs: result['runs'].append({k:v for k,v in r.items() if k!='samples'})
for mode in ['ordinary','graph']:
 means={}
 for implementation in ['baseline','candidate']:
  values=[float(s['gpu_ms']) for r in runs if r['implementation']==implementation for s in r['samples'] if s['mode']==mode]
  ordered=sorted(values)
  means[implementation]={'samples':len(values),'mean_ms':statistics.mean(values),'p50_ms':ordered[math.ceil(len(values)*.5)-1],'p95_ms':ordered[math.ceil(len(values)*.95)-1]}
 means['change_percent']=(means['candidate']['mean_ms']/means['baseline']['mean_ms']-1)*100
 result[mode]=means
result['output_sha256']={name:hashlib.sha256(data).hexdigest() for name,data in reference.items()}
(root/(f'model-{trial}-comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['trial','raw_outputs_and_statuses_exact','ordinary','graph']},indent=2))
