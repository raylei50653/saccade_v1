"""Aggregate frozen per-sequence metric objects, never sequence score averages."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

sys.dont_write_bytecode=True
OUT=Path(sys.argv[1]).resolve();TOOLS=Path(sys.argv[2]).resolve();ROOT=TOOLS.parent
REPO=ROOT.parent.parent
sys.path.insert(0,str(TOOLS/'frozen-TrackEval'))
import numpy as np
for name,alias in [('float',float),('int',int),('bool',bool)]:
    if name not in np.__dict__:setattr(np,name,alias)
import trackeval

def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def native(value):
    if isinstance(value,dict):return {str(k):native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    if isinstance(value,np.ndarray):return native(value.tolist())
    if isinstance(value,np.generic):return native(value.item())
    if isinstance(value,float) and not math.isfinite(value):raise RuntimeError('nonfinite aggregate')
    return value
def simplified(row):
    result={key:100*float(np.mean(row['HOTA'][key])) for key in ['HOTA','DetA','AssA']}
    result.update(IDF1=100*float(row['Identity']['IDF1']),MOTA=100*float(row['CLEAR']['MOTA']))
    result.update({key:int(row['CLEAR'][key]) for key in ['IDSW','CLR_FP','CLR_FN','Frag']})
    result.update({key:int(row['Count'][key]) for key in ['GT_Dets','GT_IDs','Dets','IDs']})
    result['Frames']=int(row['CLEAR']['CLR_Frames'])
    return result

def main():
    selection=load(TOOLS/'selection-frozen.json');by={s['id']:s for s in selection['sequences']}
    groups=selection['primary_groups'];arms=['official','finetuned','main']
    metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),
             trackeval.metrics.Identity({'PRINT_CONFIG':False}),trackeval.metrics.Count()]
    raw={arm:{} for arm in arms};group_results={};source_events={};primary_checks=[]
    def convert(metric,row):
        result=dict(row)
        for field in metric.float_array_fields+metric.integer_array_fields:
            result[field]=np.asarray(result[field],dtype=np.float64)
        return result
    def combine(ids,arm):
        return {metric.get_name():metric.combine_sequences({name:convert(metric,raw[arm][name][metric.get_name()]) for name in ids})
                for metric in metrics}
    for group,ids in groups.items():
        stage=OUT/'groups'/group.replace('/','__');event=load(stage/'metrics.json')
        assert event['result']=='PASS' and event['group']==group
        assert load(stage/'execution.json')['exit_code']==0
        assert set(event['per_sequence'])==set(ids)
        source_events[group]={'path':str(stage/'metrics.json'),'sha256':sha(stage/'metrics.json'),
                              'config_sha256':sha(stage/'config.json')}
        for arm in arms:
            assert set(event['raw_components'][arm])=={*ids,'COMBINED_SEQ'}
            for name in ids:
                assert name not in raw[arm]
                raw[arm][name]=event['raw_components'][arm][name]
                assert simplified(convert_all(event['raw_components'][arm][name],metrics))==event['per_sequence'][name][arm]
        group_results[group]={'sequences':ids,'sequence_count':len(ids),'frames':sum(by[name]['frames'] for name in ids),
                              'adapter':load(stage/'config.json')['dataset']['BENCHMARK'],'metrics':event['combined']}
    assert all(set(raw[arm])==set(by) for arm in arms)
    # Check each official worker aggregate against direct metric recombination.
    for group,ids in groups.items():
        event=load(OUT/'groups'/group.replace('/','__')/'metrics.json')
        for arm in arms:
            combined=combine(ids,arm)
            expected=event['raw_components'][arm]['COMBINED_SEQ']
            for metric in metrics:
                name=metric.get_name()
                for field,value in combined[name].items():
                    assert np.allclose(value,expected[name][field],rtol=1e-12,atol=1e-12),(group,arm,name,field)
            primary_checks.append({'group':group,'arm':arm,'genuine_combination_matches_worker':True})
    dataset_results={};dataset_components={}
    for benchmark in sorted({s['benchmark'] for s in by.values()}):
        ids=sorted(name for name,s in by.items() if s['benchmark']==benchmark)
        dataset_components[benchmark]={arm:native(combine(ids,arm)) for arm in arms}
        dataset_results[benchmark]={'sequences':ids,'sequence_count':len(ids),'frames':sum(by[name]['frames'] for name in ids),
                                   'metrics':{arm:simplified(combine(ids,arm)) for arm in arms}}
    all_ids=sorted(by)
    pooled_components={arm:native(combine(all_ids,arm)) for arm in arms}
    pooled={'sequences':156,'frames':135503,'metrics':{arm:simplified(combine(all_ids,arm)) for arm in arms},
            'scope':'descriptive pool after each dataset-specific annotation policy; seven primary groups retain separate interpretation'}
    per_sequence={name:{'benchmark':by[name]['benchmark'],'split':by[name].get('split'),
                        'primary_group':next(group for group,ids in groups.items() if name in ids),
                        'frames':by[name]['frames'],'metrics':{arm:simplified(convert_all(raw[arm][name],metrics)) for arm in arms}}
                  for name in sorted(by)}
    # The imported seven MOT17 cells must reproduce the completed native TRT study.
    previous=load(REPO/'evidence/model-comparison/2026-11-11-yolo26m-trt-performance/quality-scores.json')['mot']
    for arm in arms:
        for name in groups['MOT17/all']:
            for field,expected in previous[arm][name].items():
                assert abs(per_sequence[name]['metrics'][arm][field]-expected)<2e-12,(arm,name,field)
        for field,expected in previous[arm]['COMBINED'].items():
            assert abs(group_results['MOT17/all']['metrics'][arm][field]-expected)<2e-12,(arm,field)
    for container in [*group_results.values(),*dataset_results.values(),pooled]:
        assert len({tuple(container['metrics'][arm][field] for field in ['GT_Dets','GT_IDs','Frames']) for arm in arms})==1
    summary={'schema':'primary156-threeway-quality/1','result':'PASS','sequence_count':156,'frames_per_arm':135503,
             'arms':arms,'primary_groups':group_results,'datasets':dataset_results,'pooled_all156':pooled,
             'per_sequence':per_sequence,'group_events':source_events,'aggregation_checks':primary_checks,
             'MOT17_imported_results_match_previous_native_TRT':True,
             'aggregation':'TrackEval metric.combine_sequences on full raw objects; no averages of sequence metrics',
             'data_exposure':'all156 already exposed; not a blind test; SportsMOT excluded by user policy',
             'GPU_used_by_scorer':False,'formal_acceptance':False}
    save(OUT/'summary.json',summary)
    save(OUT/'raw-components.json',{'per_sequence':raw,'dataset_combined':dataset_components,'pooled_combined':pooled_components})
    fields=['HOTA','DetA','AssA','IDF1','MOTA','IDSW','Frag','CLR_FP','CLR_FN','GT_Dets','GT_IDs','Dets','IDs','Frames']
    for filename,containers in [('primary-groups.csv',group_results),('datasets.csv',dataset_results)]:
        with (OUT/filename).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['group','arm','sequences',*fields]);writer.writeheader()
            for group,container in containers.items():
                for arm in arms:writer.writerow({'group':group,'arm':arm,'sequences':container['sequence_count'],**container['metrics'][arm]})
    with (OUT/'per-sequence.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sequence','benchmark','primary_group','arm',*fields]);writer.writeheader()
        for name,item in per_sequence.items():
            for arm in arms:writer.writerow({'sequence':name,'benchmark':item['benchmark'],'primary_group':item['primary_group'],
                                            'arm':arm,**item['metrics'][arm]})
    print(json.dumps({'result':'PASS','primary_groups':7,'datasets':5,'sequences':156,'frames_per_arm':135503,
                      'primary_metrics':{group:item['metrics'] for group,item in group_results.items()}},indent=2),flush=True)

def convert_all(row,metrics):
    result={name:dict(value) for name,value in row.items()}
    for metric in metrics:
        for field in metric.float_array_fields+metric.integer_array_fields:
            result[metric.get_name()][field]=np.asarray(result[metric.get_name()][field],dtype=np.float64)
    return result

if __name__=='__main__':main()
