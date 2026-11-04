import pathlib,csv,json,sys,math,statistics,hashlib
root=pathlib.Path(__file__).resolve().parent
runs=[]
reference=None
for i,implementation in enumerate(['baseline','candidate','candidate','baseline']):
 name=f'tracking-trial2-{i}-{implementation}'
 directory=root/name
 assert (root/(name+'.exit')).read_text().strip()=='0'
 summary=json.loads((directory/'summary.json').read_text())
 assert summary['status']=='PASS'
 packets={p.name:p.read_bytes() for p in directory.glob('*.bin')}
 if reference is None: reference=packets
 assert packets==reference,name+' packet bytes changed'
 samples=[float(r['enqueue_to_receive_us']) for p in directory.glob('*.csv') if p.name!='runs.csv' for r in csv.DictReader(p.open())]
 queued=[float(row['wall_us_per_frame']) for row in csv.DictReader((directory/'runs.csv').open()) if row['style']=='queued']
 runs.append({'name':name,'implementation':implementation,'summary':summary,'samples':samples,'queued':queued})
result={'status':'PASS','formal_acceptance':False,'packets_exact':True,'frames_per_session':140,'sessions':16,'packet_sha256':runs[0]['summary']['packet_sha256'],'runs':[{k:v for k,v in r.items() if k not in ['samples','queued']} for r in runs]}
for implementation in ['baseline','candidate']:
 values=[x for r in runs if r['implementation']==implementation for x in r['samples']]
 ordered=sorted(values)
 queued=[x for r in runs if r['implementation']==implementation for x in r['queued']]
 result[implementation]={'single_samples':len(values),'single_mean_us':statistics.mean(values),'single_p50_us':ordered[math.ceil(len(values)*.5)-1],'single_p95_us':ordered[math.ceil(len(values)*.95)-1],'queued_us_per_frame_mean':statistics.mean(queued)}
result['change_percent']={key:100*(result['candidate'][key]/result['baseline'][key]-1) for key in ['single_mean_us','single_p50_us','single_p95_us','queued_us_per_frame_mean']}
(root/'tracking-trial2-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
