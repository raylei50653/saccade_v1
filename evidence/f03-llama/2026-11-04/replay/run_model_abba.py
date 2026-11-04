from pathlib import Path
import subprocess, sys, json, hashlib
p=Path(__file__).resolve().parent
engine='/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'
images='/home/ray/.cache/saccade-v1/hr-deploy/refs/model-images.f32'
for i,kind in enumerate(['baseline','candidate','candidate','baseline']):
 binary=p/'full-baseline/model_benchmark' if kind=='baseline' else p/'model_benchmark_candidate'
 name=f'model-{i}-{kind}'
 cmd=[str(binary),engine,images,str(p/name),'--profile','hr-1280x768','--iterations','100','--warmup','30','--rounds','1']
 subprocess.run([sys.executable,str(p/'run.py'),name,*cmd],check=True)
ref=p/'model-0-baseline'
comparisons=[]
for i,kind in enumerate(['baseline','candidate','candidate','baseline']):
 d=p/f'model-{i}-{kind}'
 for f in sorted(ref.glob('input-*')):
  same=f.read_bytes()==(d/f.name).read_bytes()
  comparisons.append({'run':d.name,'file':f.name,'equal':same,'sha256':hashlib.sha256((d/f.name).read_bytes()).hexdigest()})
  assert same,(d,f.name)
(p/'model-exact-comparison.json').write_text(json.dumps({'status':'PASS','comparisons':comparisons},indent=2)+'\n')
