import pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parent
trial=sys.argv[1]
engine='/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'
images='/home/ray/.cache/saccade-v1/hr-deploy/refs/model-images.f32'
for i,implementation in enumerate(['baseline','candidate','candidate','baseline']):
    name=f'model-{trial}-{i}-{implementation}'
    args=[str(root/implementation/'model_benchmark'),engine,images,str(root/name),'--profile','hr-1280x768','--iterations','80','--warmup','20','--rounds','1']
    subprocess.run([sys.executable,str(root/'run_capture.py'),name,*args],check=True)
