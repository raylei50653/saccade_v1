"""Independent, CPU-only audit of the completed primary156 scoring bundle.

Recombines full retained raw objects with the frozen official TrackEval metric
implementations. It never evaluates GT or changes scorer/capture files.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=ROOT.parent.parent
TOOLS=ROOT/'scoring-tools'
SCORING=ROOT/'scoring'
ARMS=['official','finetuned','main']
FLOAT_FIELDS=['HOTA','DetA','AssA','IDF1','MOTA']
COUNT_FIELDS=['IDSW','CLR_FP','CLR_FN','Frag','GT_Dets','GT_IDs','Dets','IDs','Frames']
TIE_TOLERANCE_PP=1e-9
RECOMPUTE_TOLERANCE_PP=2e-10

def load(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def save(path,payload):
 path=Path(path);assert path.resolve().is_relative_to(HERE.resolve())
 assert not path.exists(),('refuse to overwrite audit output',str(path))
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')

def initialize():
 global np,metrics
 freeze=load(TOOLS/'freeze.json')
 selection=load(TOOLS/'selection-frozen.json')
 assert sha(TOOLS/'selection-frozen.json')==freeze['selection_sha256']
 assert sha(ROOT/'selection-runtime.json')==freeze['selection_sha256']
 assert sha(TOOLS/'policy-frozen.json')==freeze['policy_sha256']
 for path,expected in freeze['core_source_sha256'].items():assert sha(path)==expected,('scorer source changed',path)
 vendor=TOOLS/'frozen-TrackEval'
 assert sha(vendor/'SOURCE.json')==freeze['vendor_source_json_sha256']
 manifest=load(vendor/'SOURCE.json')
 for rel,expected in manifest['unchanged_files_sha256'].items():
  assert sha(vendor/rel)==expected
  assert sha(REPO/'third_party/TrackEval'/rel)==expected
 for rel,row in manifest['modified_initializers'].items():assert sha(vendor/rel)==row['vendored_sha256']
 sys.path.insert(0,str(vendor))
 import numpy
 np=numpy
 for name,alias in [('float',float),('int',int),('bool',bool)]:
  if name not in np.__dict__:setattr(np,name,alias)
 import trackeval
 assert Path(trackeval.__file__).resolve().is_relative_to(vendor.resolve())
 metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),trackeval.metrics.Identity({'PRINT_CONFIG':False}),trackeval.metrics.Count()]
 return freeze,selection,vendor

def converted(row):
 out={name:dict(fields) for name,fields in row.items()}
 assert set(out)=={m.get_name() for m in metrics}
 for metric in metrics:
  for key in metric.float_array_fields+metric.integer_array_fields:
   out[metric.get_name()][key]=np.asarray(out[metric.get_name()][key],dtype=np.float64)
 return out

def recombine(raw,ids,arm):
 assert len(ids)==len(set(ids)) and ids
 return {m.get_name():m.combine_sequences({name:converted(raw[arm][name])[m.get_name()] for name in ids}) for m in metrics}

def simplified(row):
 result={k:100*float(np.mean(row['HOTA'][k])) for k in ['HOTA','DetA','AssA']}
 result.update(IDF1=100*float(row['Identity']['IDF1']),MOTA=100*float(row['CLEAR']['MOTA']))
 result.update({k:int(row['CLEAR'][k]) for k in ['IDSW','CLR_FP','CLR_FN','Frag']})
 result.update({k:int(row['Count'][k]) for k in ['GT_Dets','GT_IDs','Dets','IDs']})
 result['Frames']=int(row['CLEAR']['CLR_Frames'])
 assert all(math.isfinite(float(v)) for v in result.values())
 mota=100*(1-(result['CLR_FP']+result['CLR_FN']+result['IDSW'])/max(1,result['GT_Dets']))
 assert abs(result['MOTA']-mota)<2e-12
 # Exact per-alpha HOTA identity; no sqrt of averaged DetA/AssA shortcut.
 h=row['HOTA'];assert np.allclose(h['HOTA'],np.sqrt(h['DetA']*h['AssA']),rtol=1e-12,atol=1e-12)
 return result

def compare_simple(actual,expected,context):
 for k in FLOAT_FIELDS:assert abs(actual[k]-expected[k])<=RECOMPUTE_TOLERANCE_PP,(context,k,actual[k],expected[k])
 for k in COUNT_FIELDS:assert actual[k]==expected[k],(context,k,actual[k],expected[k])

def compare_raw(actual,expected,context):
 assert set(actual)==set(expected)
 for metric,fields in actual.items():
  assert set(fields)==set(expected[metric]),(context,metric,'field population')
  for key,value in fields.items():
   assert np.allclose(value,expected[metric][key],rtol=1e-12,atol=1e-12),(context,metric,key)

def checked_event(group,ids,selection,vendor):
 stage=SCORING/'groups'/group.replace('/','__')
 event=load(stage/'metrics.json');cfg=load(stage/'config.json');execution=load(stage/'execution.json')
 assert event['result']=='PASS' and event['group']==cfg['group']==group and execution['exit_code']==0
 assert set(event['per_sequence'])==set(ids)==set(cfg['dataset']['SEQ_INFO'])
 assert cfg['dataset']['TRACKERS_TO_EVAL']==ARMS
 adapter='MOT15' if group.startswith('MOT15/') else 'MOT20' if group.startswith('MOT20/') else 'MOT17'
 assert cfg['dataset']['BENCHMARK']==adapter and cfg['dataset']['DO_PREPROC'] is True
 assert cfg['dataset']['SKIP_SPLIT_FOL'] is True and cfg['dataset']['CLASSES_TO_EVAL']==['pedestrian']
 assert cfg['worker_sha256']==execution['worker_sha256']==sha(TOOLS/'worker.py')
 assert cfg['freeze_sha256']==sha(TOOLS/'freeze.json')
 assert execution['config_sha256']==sha(stage/'config.json')
 assert event['input_hashes']==cfg['input_hashes']
 # Read bindings only: never reopen GT for evaluation or alter prepared inputs.
 for path,expected in event['input_hashes'].items():assert sha(path)==expected
 assert len(event['opened_ground_truth'])==len(event['opened_metadata'])==len(ids)
 assert len(event['opened_trackers'])==3*len(ids)
 for rel,expected in event['trackeval_imports'].items():assert sha(vendor/rel)==expected
 for arm in ARMS:assert set(event['raw_components'][arm])=={*ids,'COMBINED_SEQ'}
 return event,cfg

def audit_one_group(group,selection,vendor):
 ids=selection['primary_groups'][group]
 event,cfg=checked_event(group,ids,selection,vendor)
 previous=load(REPO/'evidence/model-comparison/2026-11-11-yolo26m-trt-performance/quality-scores.json')['mot']
 rows=[]
 for arm in ARMS:
  for name in ids:
   r=simplified(converted(event['raw_components'][arm][name]))
   compare_simple(r,event['per_sequence'][name][arm],(group,arm,name))
   assert r['Frames']==cfg['expected_frames'][name]
   if group=='MOT17/all':
    for field,expected in previous[arm][name].items():assert abs(r[field]-expected)<2e-12,(arm,name,field)
  combined=recombine(event['raw_components'],ids,arm)
  compare_raw(combined,event['raw_components'][arm]['COMBINED_SEQ'],(group,arm))
  compare_simple(simplified(combined),event['combined'][arm],(group,arm))
  rows.append({'arm':arm,'genuine_recombination':'PASS','sequences':len(ids)})
 return rows

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--self-test-group',choices=['MOT17/all'])
 ap.add_argument('--out',type=Path,default=HERE/'scoring-audit.json')
 args=ap.parse_args();freeze,selection,vendor=initialize()
 groups=selection['primary_groups'];by={s['id']:s for s in selection['sequences']};all_ids=sorted(by)
 assert len(all_ids)==156 and sum(s['frames'] for s in by.values())==135503
 assert len(groups)==7 and len({x for ids in groups.values() for x in ids})==sum(map(len,groups.values()))==156
 if args.self_test_group:
  rows=audit_one_group(args.self_test_group,selection,vendor)
  save(args.out,{'result':'PASS_BOUNDED_MOT17_SELF_TEST_ONLY','scope':'seven imported MOT17 sequences only; not full primary156 acceptance','rows':rows})
  print(json.dumps({'result':'PASS_BOUNDED_MOT17_SELF_TEST_ONLY','arms':3,'sequences':7}));return
 progress=load(SCORING/'progress.json');assert progress['status']=='PASS','full scorer not complete'
 complete=load(REPO/'evidence/model-comparison/2026-11-11-yolo26m-primary156/capture-complete.json')
 assert complete['complete'] and complete['cells']==468 and complete['frames']==406509 and not complete['failures']
 summary=load(SCORING/'summary.json');components=load(SCORING/'raw-components.json')
 assert summary['result']=='PASS' and summary['sequence_count']==156 and summary['frames_per_arm']==135503
 assert summary['arms']==ARMS and set(summary['primary_groups'])==set(groups)
 assert set(summary['per_sequence'])==set(all_ids)
 assert set(summary['datasets'])=={s['benchmark'] for s in by.values()} and len(summary['datasets'])==5
 raw=components['per_sequence'];assert set(raw)==set(ARMS)
 assert all(set(raw[a])==set(all_ids) for a in ARMS)
 checked_groups=[];contexts=[]
 for group,ids in groups.items():
  event,cfg=checked_event(group,ids,selection,vendor)
  for arm in ARMS:
   for name in ids:assert raw[arm][name]==event['raw_components'][arm][name]
   recomputed=recombine(raw,ids,arm)
   compare_raw(recomputed,event['raw_components'][arm]['COMBINED_SEQ'],(group,arm))
   compare_simple(simplified(recomputed),summary['primary_groups'][group]['metrics'][arm],(group,arm))
  assert summary['primary_groups'][group]['sequences']==ids
  assert summary['primary_groups'][group]['sequence_count']==len(ids)
  assert summary['primary_groups'][group]['frames']==sum(by[n]['frames'] for n in ids)
  checked_groups.append(group);contexts.append(('primary_group',group,ids,summary['primary_groups'][group]['metrics']))
 for benchmark,row in summary['datasets'].items():
  ids=sorted(n for n in all_ids if by[n]['benchmark']==benchmark)
  assert row['sequences']==ids and row['sequence_count']==len(ids) and row['frames']==sum(by[n]['frames'] for n in ids)
  for arm in ARMS:
   recomputed=recombine(raw,ids,arm)
   compare_raw(recomputed,components['dataset_combined'][benchmark][arm],(benchmark,arm))
   compare_simple(simplified(recomputed),row['metrics'][arm],(benchmark,arm))
  contexts.append(('dataset',benchmark,ids,row['metrics']))
 for arm in ARMS:
  recomputed=recombine(raw,all_ids,arm)
  compare_raw(recomputed,components['pooled_combined'][arm],('all156',arm))
  compare_simple(simplified(recomputed),summary['pooled_all156']['metrics'][arm],('all156',arm))
 contexts.append(('pooled','all156',all_ids,summary['pooled_all156']['metrics']))
 previous=load(REPO/'evidence/model-comparison/2026-11-11-yolo26m-trt-performance/quality-scores.json')['mot']
 deltas=[];maximum_prior_difference=0.;validation_count=0;support={}
 for name,row in summary['per_sequence'].items():
  assert row['benchmark']==by[name]['benchmark'] and row['frames']==by[name]['frames']
  assert row['primary_group']==next(g for g,ids in groups.items() if name in ids)
  per_arm=[]
  for arm in ARMS:
   r=simplified(converted(raw[arm][name]));compare_simple(r,row['metrics'][arm],(name,arm))
   assert r['Frames']==by[name]['frames'];per_arm.append(tuple(r[k] for k in ['GT_Dets','GT_IDs','Frames']))
   capture=load(ROOT/'cells'/arm/name/'capture.json');assert capture['status']=='PASS' and capture['frames']==r['Frames']
   valpath=Path(capture['validation']['path']);assert sha(valpath)==capture['validation']['sha256']
   val=load(valpath);assert val['status']=='PASS' and val['frames']==r['Frames']
   assert val['packets']['sha256_raw']==capture['packets_sha256'] and val['motion']['sha256_of_file']==capture['motion_sha256']
   assert val['config_sha256']==capture['resolved_config_sha256']
   assert val['build_identity']['build.engine_sha256']==capture['engine_sha256']
   assert val['tracker_output_matches_frozen_baseline'] and val['packets']['canonical_all_frames'] and val['packets']['all_status_ok']
   validation_count+=1
   if row['benchmark']=='MOT17':
    for k,v in previous[arm][name].items():
     difference=abs(r[k]-v);assert difference<2e-12,(name,arm,k);maximum_prior_difference=max(maximum_prior_difference,difference)
  assert len(set(per_arm))==1,(name,'GT/frame support differs across arms')
  support[name]=per_arm[0]
  delta={k:row['metrics']['main'][k]-row['metrics']['finetuned'][k] for k in FLOAT_FIELDS}
  deltas.append({'sequence':name,'benchmark':row['benchmark'],'primary_group':row['primary_group'],'frames':row['frames'],
                 'GT_Dets':per_arm[0][0],'main':row['metrics']['main'],'finetuned':row['metrics']['finetuned'],'delta_pp':delta})
 statistics=[]
 for kind,label,ids,values in contexts:
  assert len({tuple(values[a][k] for k in ['GT_Dets','GT_IDs','Frames']) for a in ARMS})==1
  stats={}
  for metric in FLOAT_FIELDS:
   ds=[summary['per_sequence'][n]['metrics']['main'][metric]-summary['per_sequence'][n]['metrics']['finetuned'][metric] for n in ids]
   stats[metric]={'wins':sum(d>TIE_TOLERANCE_PP for d in ds),'ties':sum(abs(d)<=TIE_TOLERANCE_PP for d in ds),'losses':sum(d< -TIE_TOLERANCE_PP for d in ds)}
   assert sum(stats[metric].values())==len(ids)
  statistics.append({'kind':kind,'group':label,'sequence_count':len(ids),'per_sequence_win_tie_loss':stats,
                     'main':values['main'],'finetuned':values['finetuned'],
                     'combined_delta_pp':{k:values['main'][k]-values['finetuned'][k] for k in FLOAT_FIELDS},
                     'interpretation':'descriptive HOTA/DetA/AssA deltas; no causal attribution or adoption gate'})
 sorted_deltas=sorted(deltas,key=lambda x:(x['delta_pp']['HOTA'],x['sequence']))
 regressions=[r for r in sorted_deltas if r['delta_pp']['HOTA']< -TIE_TOLERANCE_PP]
 result={'result':'PASS','scope':'independent CPU audit of frozen full scoring objects; no GT reevaluation or GPU work',
   'primary_groups_checked':7,'dataset_groups_checked':5,'sequences_checked':156,'arms':ARMS,'capture_validation_receipts_checked':validation_count,
   'frames_per_arm':135503,'raw_metric_recombination':'genuine frozen official combine_sequences for HOTA/CLEAR/Identity/Count',
   'recompute_float_tolerance_pp':RECOMPUTE_TOLERANCE_PP,'win_tie_loss_tolerance_pp':TIE_TOLERANCE_PP,
   'tolerance_purpose':'floating roundoff only; no meaningful quality tie band or altered registered gate',
   'main_and_other_MOT17_parity_max_abs_difference':maximum_prior_difference,
   'context_comparisons':statistics,'per_sequence_deltas':deltas,
   'HOTA_regression_sequence_count':len(regressions),'maximum_HOTA_regression':regressions[0] if regressions else None,
   'worst_ten_HOTA_regressions':regressions[:10],
   'data_exposure':'all156 already exposed; MOT17 includes training/selection/postM4 reuse; SportsMOT excluded by prior user policy',
   'formal_acceptance':False,'adoption_changed':False,
   'input_hashes':{'selection':sha(TOOLS/'selection-frozen.json'),'freeze':sha(TOOLS/'freeze.json'),'summary':sha(SCORING/'summary.json'),'raw_components':sha(SCORING/'raw-components.json')}}
 save(args.out,result)
 csvpath=args.out.with_suffix('.per-sequence.csv');assert not csvpath.exists()
 with csvpath.open('w') as f:
  fields=['sequence','benchmark','primary_group','frames','GT_Dets',*['delta_'+k+'_pp' for k in FLOAT_FIELDS]]
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for r in sorted_deltas:writer.writerow({k:r[k] for k in fields if k in r}|{'delta_'+k+'_pp':r['delta_pp'][k] for k in FLOAT_FIELDS})
 print(json.dumps({'result':'PASS','primary_groups':7,'datasets':5,'sequences':156,'validation_receipts':validation_count,
                   'overall_main_vs_fine_HOTA_WTL':statistics[-1]['per_sequence_win_tie_loss']['HOTA'],
                   'worst_HOTA_regression':result['maximum_HOTA_regression']},indent=2))

if __name__=='__main__':main()
