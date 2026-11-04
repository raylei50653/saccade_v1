from pathlib import Path
import subprocess,sys,hashlib,json
p=Path(__file__).resolve().parent
art=Path('/home/ray/.cache/saccade-v1/hr-deploy')
onnx=art/'yolo26m-backbone-hr.onnx'; bundle=art/'hr-head.swb'
onnx_sha=hashlib.sha256(onnx.read_bytes()).hexdigest(); bundle_sha=hashlib.sha256(bundle.read_bytes()).hexdigest()
(p/'artifact-inputs.json').write_text(json.dumps({'onnx':str(onnx),'onnx_sha256':onnx_sha,'bundle':str(bundle),'bundle_sha256':bundle_sha},indent=2)+'\n')
for kind in ['baseline','opt5','workspace1g','tiling-fast','causal-padding']:
 engine=p/(kind+'.engine'); report=p/(kind+'.build.txt')
 ready=engine.exists() and report.exists()
 args=[str(p/('model_trt_'+kind)),'build',str(onnx),onnx_sha,str(bundle),bundle_sha,str(engine),str(report),'--profile','hr-1280x768','--precision','fp16']
 cmd=[sys.executable,str(p/'run.py'),'engine-build-'+kind,'flock','-w','300','/home/ray/.cache/saccade-v1/gpu.lock','timeout','--signal=TERM','--kill-after=10s','300',*args]
 result=subprocess.CompletedProcess(cmd,0) if ready else subprocess.run(cmd)
 if result.returncode:continue
 args=[str(p/'model_benchmark'),str(engine),str(art/'refs/model-images.f32'),str(p/('screen2-'+kind)),'--profile','hr-1280x768','--iterations','50','--warmup','25','--rounds','1','--layer-iterations','0']
 subprocess.run([sys.executable,str(p/'run.py'),'screen2-'+kind,'flock','-w','300','/home/ray/.cache/saccade-v1/gpu.lock','timeout','--signal=TERM','--kill-after=10s','120',*args],check=True)
