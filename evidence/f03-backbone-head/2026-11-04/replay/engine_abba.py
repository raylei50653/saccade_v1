from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
original=Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')
images='/home/ray/.cache/saccade-v1/hr-deploy/refs/model-images.f32'
for case,a,b in [('rebuild-control',original,p/'baseline.engine'),('causal-vs-rebuild',p/'baseline.engine',p/'causal-padding.engine')]:
 for i,kind in enumerate(['baseline','candidate','candidate','baseline']):
  name=f'{case}-{i}-{kind}'
  cmd=[str(p/'model_benchmark'),str(a if kind=='baseline' else b),images,str(p/name),'--profile','hr-1280x768','--iterations','160','--warmup','40','--rounds','1']
  subprocess.run([sys.executable,str(p/'run.py'),name,*cmd],check=True)
