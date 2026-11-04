from pathlib import Path
import subprocess, sys, json, hashlib
p=Path(__file__).resolve().parent
kind_name='block128'
label='block128-confirm'
engine='/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'
images='/home/ray/.cache/saccade-v1/hr-deploy/refs/model-images.f32'
for i,kind in enumerate(['baseline','candidate','candidate','baseline']):
 binary=p/'full-baseline/model_benchmark' if kind=='baseline' else p/('model_benchmark_'+kind_name)
 name=f'{label}-model-{i}-{kind}'
 cmd=[str(binary),engine,images,str(p/name),'--profile','hr-1280x768','--iterations','200','--warmup','40','--rounds','1']
 subprocess.run([sys.executable,str(p/'run.py'),name,*cmd],check=True)
ref=p/(label+'-model-0-baseline')
comparisons=[]
for i,kind in enumerate(['baseline','candidate','candidate','baseline']):
 d=p/f'{label}-model-{i}-{kind}'
 for f in sorted(ref.glob('input-*')):
  same=f.read_bytes()==(d/f.name).read_bytes()
  comparisons.append({'run':d.name,'file':f.name,'equal':same,'sha256':hashlib.sha256((d/f.name).read_bytes()).hexdigest()})
  assert same,(d,f.name)
(p/(label+'-model-exact-comparison.json')).write_text(json.dumps({'status':'PASS','comparisons':comparisons},indent=2)+'\n')
