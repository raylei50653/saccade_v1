"""One frozen primary group, three tracker arms, one genuine TrackEval event."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

sys.dont_write_bytecode=True
STAGE=Path(sys.argv[1]).resolve()
CONFIG=json.loads((STAGE/'config.json').read_text())
VENDOR=Path(CONFIG['frozen_vendor']).resolve()
DATASET=CONFIG['dataset']
NAMES=list(DATASET['SEQ_INFO'])
ARMS=DATASET['TRACKERS_TO_EVAL']
GT=Path(DATASET['GT_FOLDER'])
TRACKERS=Path(DATASET['TRACKERS_FOLDER'])
ALLOWED_GT={(GT/name/'gt/gt.txt').resolve() for name in NAMES}
ALLOWED_METADATA={(GT/name/'seqinfo.ini').resolve() for name in NAMES}
ALLOWED_TRACKERS={(TRACKERS/arm/(name+'.txt')).resolve() for arm in ARMS for name in NAMES}
OPENED_GT=set();OPENED_METADATA=set();OPENED_TRACKERS=set()

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()

def audit(event,values):
    if event!='open' or not values or not isinstance(values[0],(str,bytes)):return
    path=Path(values[0].decode() if isinstance(values[0],bytes) else values[0]).resolve()
    if path.name=='gt.txt':
        if path not in ALLOWED_GT:raise RuntimeError('unselected GT: '+str(path))
        OPENED_GT.add(path)
    if path.name=='seqinfo.ini':
        if path not in ALLOWED_METADATA:raise RuntimeError('unselected metadata: '+str(path))
        OPENED_METADATA.add(path)
    if path.name in {name+'.txt' for name in NAMES}:
        if path not in ALLOWED_TRACKERS:raise RuntimeError('unbound tracker: '+str(path))
        OPENED_TRACKERS.add(path)

sys.addaudithook(audit)
sys.path.insert(0,str(VENDOR))
import numpy as np
for name,alias in [('float',float),('int',int),('bool',bool)]:
    if name not in np.__dict__:setattr(np,name,alias)
import scipy
import trackeval
if not Path(trackeval.__file__).resolve().is_relative_to(VENDOR):raise RuntimeError('TrackEval import outside frozen vendor')

def serializable(value):
    if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
    if isinstance(value,np.ndarray):return serializable(value.tolist())
    if isinstance(value,np.generic):return serializable(value.item())
    if isinstance(value,float) and not math.isfinite(value):raise RuntimeError('nonfinite metric component')
    return value

def simplified(row):
    result={key:100*float(np.mean(row['HOTA'][key])) for key in ['HOTA','DetA','AssA']}
    result.update(IDF1=100*float(row['Identity']['IDF1']),MOTA=100*float(row['CLEAR']['MOTA']))
    result.update({key:int(row['CLEAR'][key]) for key in ['IDSW','CLR_FP','CLR_FN','Frag']})
    result.update({key:int(row['Count'][key]) for key in ['GT_Dets','GT_IDs','Dets','IDs']})
    result['Frames']=int(row['CLEAR']['CLR_Frames'])
    if any(not math.isfinite(float(v)) for v in result.values()):raise RuntimeError('nonfinite simplified result')
    mota=100*(1-(result['CLR_FP']+result['CLR_FN']+result['IDSW'])/max(1,result['GT_Dets']))
    if abs(mota-result['MOTA'])>2e-12:raise RuntimeError('MOTA formula mismatch')
    return result

def main():
    input_hashes={str(path):sha(path) for path in sorted(ALLOWED_GT|ALLOWED_METADATA|ALLOWED_TRACKERS)}
    for path,expected in CONFIG['input_hashes'].items():
        if input_hashes.get(str(Path(path).resolve()))!=expected:raise RuntimeError('prepared scoring input hash mismatch: '+path)
    # Count actual dataset/evaluator reads, excluding our binding hash reads.
    OPENED_GT.clear();OPENED_METADATA.clear();OPENED_TRACKERS.clear()
    dataset=trackeval.datasets.MotChallenge2DBox(DATASET)
    if set(dataset.seq_list)!=set(NAMES):raise RuntimeError('selected sequence population differs')
    metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),trackeval.metrics.Identity({'PRINT_CONFIG':False})]
    evaluator=trackeval.Evaluator({**trackeval.Evaluator.get_default_eval_config(),
        'USE_PARALLEL':False,'PRINT_CONFIG':False,'PLOT_CURVES':False,'TIME_PROGRESS':False,
        'BREAK_ON_ERROR':True,'RETURN_ON_ERROR':False,'OUTPUT_SUMMARY':True,'OUTPUT_DETAILED':True,
        'LOG_ON_ERROR':str(STAGE/'error.log')})
    results,messages=evaluator.evaluate([dataset],metrics)
    key=dataset.get_name()
    if messages[key]!={arm:'Success' for arm in ARMS}:raise RuntimeError('evaluation arm failed')
    per_sequence={name:{} for name in NAMES};combined={};components={}
    for arm in ARMS:
        table=results[key][arm]
        if set(table)!={*NAMES,'COMBINED_SEQ'}:raise RuntimeError('scored sequence population differs')
        combined[arm]=simplified(table['COMBINED_SEQ']['pedestrian'])
        components[arm]={name:serializable(table[name]['pedestrian']) for name in [*NAMES,'COMBINED_SEQ']}
        for name in NAMES:
            per_sequence[name][arm]=simplified(table[name]['pedestrian'])
            if per_sequence[name][arm]['Frames']!=CONFIG['expected_frames'][name]:raise RuntimeError('scored frame count differs')
        for metric in metrics:
            label=metric.get_name();actual=table['COMBINED_SEQ']['pedestrian'][label]
            expected=metric.combine_sequences({name:table[name]['pedestrian'][label] for name in NAMES})
            if set(actual)!=set(expected):raise RuntimeError('aggregate fields differ')
            for field in actual:
                if not np.allclose(actual[field],expected[field],rtol=1e-12,atol=1e-12):
                    raise RuntimeError('official combination differs: '+label+'/'+field)
        for field in ['IDSW','CLR_FP','CLR_FN','Frag','GT_Dets','GT_IDs','Frames','Dets','IDs']:
            if combined[arm][field]!=sum(per_sequence[name][arm][field] for name in NAMES):
                raise RuntimeError('combined count differs: '+field)
    for name in [*NAMES,'COMBINED_SEQ']:
        rows=combined if name=='COMBINED_SEQ' else per_sequence[name]
        support={tuple(rows[arm][k] for k in ['GT_Dets','GT_IDs','Frames']) for arm in ARMS}
        if len(support)!=1:raise RuntimeError('cross-arm GT/frame coverage differs')
    if OPENED_GT!=ALLOWED_GT or OPENED_METADATA!=ALLOWED_METADATA or OPENED_TRACKERS!=ALLOWED_TRACKERS:
        raise RuntimeError('not every selected GT/metadata/tracker file was read')
    if input_hashes!={str(path):sha(path) for path in sorted(ALLOWED_GT|ALLOWED_METADATA|ALLOWED_TRACKERS)}:
        raise RuntimeError('scoring input bytes changed')
    imports={str(Path(module.__file__).resolve().relative_to(VENDOR)):sha(Path(module.__file__))
             for module in tuple(sys.modules.values()) if getattr(module,'__file__',None)
             and Path(module.__file__).resolve().is_relative_to(VENDOR) and Path(module.__file__).is_file()}
    payload={'result':'PASS','group':CONFIG['group'],'combined':combined,'per_sequence':per_sequence,
             'raw_components':components,'numpy':np.__version__,'scipy':scipy.__version__,
             'python':sys.version,'trackeval_imports':imports,'input_hashes':input_hashes,
             'opened_ground_truth':sorted(map(str,OPENED_GT)),'opened_metadata':sorted(map(str,OPENED_METADATA)),
             'opened_trackers':sorted(map(str,OPENED_TRACKERS)),
             'aggregation':'genuine metric.combine_sequences from per-sequence raw components; no means of sequence scores'}
    target=STAGE/'metrics.json';target.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'result':'PASS','group':CONFIG['group'],'sequences':len(NAMES),'combined':combined}),flush=True)

if __name__=='__main__':main()
