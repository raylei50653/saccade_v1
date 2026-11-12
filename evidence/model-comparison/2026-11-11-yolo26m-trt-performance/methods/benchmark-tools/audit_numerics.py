"""Independent CPU recomputation of the retained TensorRT numerical screen."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for arm in ['official','finetuned','official-fp32-diagnostic']:
 folder=ROOT/'artifacts'/arm
 ref_folder=ROOT/'artifacts'/('official' if arm=='official-fp32-diagnostic' else arm)
 onnx=json.loads((ref_folder/'onnx-verify.json').read_text())
 report=json.loads((folder/'trt-verify.json').read_text())
 native=json.loads((folder/'trt-verify/summary.json').read_text())
 build=json.loads((folder/'build.json').read_text())
 assert native['status']=='PASS' and native['ordinary_repeat_and_graph_bits_equal']
 assert native['engine_sha256']==sha(folder/'model.engine')
 assert native['scan_status_count']==0
 assert (build['tf32'],build['bf16'],build['int8'],build['optimization_level'],build['workspace_bytes'],build['actual_aux_streams'])==(False,False,False,3,268435456,0)
 assert build['fp16']==(arm!='official-fp32-diagnostic')
 worst=0.;all_finite=True;cases=[]
 for case in range(4):
  actual=np.fromfile(folder/'trt-verify'/f'input-{case}.f32',dtype='<f4')
  reference=np.fromfile(ref_folder/'refs'/f'pytorch-{case}.f32',dtype='<f4')
  assert actual.shape==reference.shape
  offset=0;checks={}
  for name,shape in zip(onnx['output_names'],onnx['output_shapes'],strict=True):
   size=int(np.prod(shape));a=actual[offset:offset+size].astype(np.float64);b=reference[offset:offset+size].astype(np.float64);offset+=size
   finite=bool(np.isfinite(a).all() and np.isfinite(b).all());all_finite&=finite
   rel=float(np.sqrt(np.sum((a-b)**2))/max(float(np.sqrt(np.sum(b*b))),1e-30))
   checks[name]={'finite':finite,'rel_l2':rel,'max_abs':float(np.max(np.abs(a-b)))};worst=max(worst,rel)
   recorded=(report['rows'] if arm=='official-fp32-diagnostic' else report['frames'])[case]['outputs'][name]
   assert abs(rel-recorded['rel_l2'])<1e-12
   assert checks[name]['max_abs']==recorded['max_abs']
  assert offset==actual.size
  cases.append({'case':case,'outputs':checks})
 limit=1e-4 if arm=='official-fp32-diagnostic' else .01
 expected='PASS' if all_finite and worst<=limit else 'FAIL'
 assert report['status']==expected
 rows.append({'arm':arm,'retained_screen_status':expected,'limit':limit,'independent_max_rel_l2':worst,
              'finite_repeat_graph_execution':'PASS','cases':cases})
default=ROOT/'artifacts/official-default'
default_report=json.loads((default/'trt-verify.json').read_text())
assert default_report['status']=='PASS'
assert default_report['limits']['default_decoded'].startswith('finite required')
for case in range(4):assert np.isfinite(np.fromfile(default/'trt-verify'/f'input-{case}.f32',dtype='<f4')).all()
result={'audit_status':'PASS','scope':'independent CPU numerical metric and flag/identity recomputation; no new inference, no gate changes',
 'raw_screens':rows,'official_default':'finite/repeat/Graph execution PASS; decoded order/set diagnostics are not an equivalence certificate',
 'conclusion':'FP16 raw6 1% screens remain FAIL; bounded official FP32/noTF32 control remains PASS. Actual detector/tracker quality is evaluated separately.'}
(HERE/'numerical-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'audit_status':'PASS','retained_screens':[{k:r[k] for k in ['arm','retained_screen_status','independent_max_rel_l2']} for r in rows]}))
