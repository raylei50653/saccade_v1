import pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parent
engine='/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'
sequence='/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP'
for i,implementation in enumerate(['baseline','candidate','candidate','baseline']):
 name=f'tracking-trial2-{i}-{implementation}'
 args=[str(root/implementation/'tracking_benchmark'),engine,sequence,str(root/name),'--profile','hr-1280x768','--frames','120','--warmup','20','--rounds','1','--only','graph']
 subprocess.run([sys.executable,str(root/'run_capture.py'),name,*args],check=True)
