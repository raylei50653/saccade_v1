from pathlib import Path
import subprocess,sys,json,hashlib
p=Path(__file__).resolve().parent
engine='/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'
images='/home/ray/.cache/saccade-v1/hr-deploy/refs/model-images.f32'
for case,options in [('cache4m-confirm',['--persistent-cache-bytes',str(4<<20)])]:
 dirs=[]
 for i,kind in enumerate(['baseline','candidate','candidate','baseline']):
  name=f'{case}-{i}-{kind}';out=p/name;dirs.append(out)
  args=[str(p/'model_benchmark'),engine,images,str(out),'--profile','hr-1280x768','--iterations','200','--warmup','40','--rounds','1']
  if kind=='candidate':args+=options
  subprocess.run([sys.executable,str(p/'run.py'),name,*args],check=True)
 records=[]
 for d in dirs:
  for f in sorted(dirs[0].glob('input-*')):
   same=f.read_bytes()==(d/f.name).read_bytes();assert same,(d.name,f.name)
   records.append({'run':d.name,'file':f.name,'equal':same,'sha256':hashlib.sha256((d/f.name).read_bytes()).hexdigest()})
 (p/(case+'-exact.json')).write_text(json.dumps({'status':'PASS','comparisons':records},indent=2)+'\n')
