"""Combine retained 02+10 and training5 arrays using TrackEval, without GT reread."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

RUN=Path(__file__).resolve().parent.parent
REPO=RUN.parent.parent
EVIDENCE=REPO/'evidence/model-comparison/2026-11-11-yolo26m-trt-performance'
VENDOR=REPO/'third_party/TrackEval'
sys.path.insert(0,str(VENDOR))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import numpy as np
    for name,alias in [('float',float),('int',int),('bool',bool)]:
        if name not in np.__dict__:setattr(np,name,alias)
    from trackeval.metrics import HOTA,CLEAR,Identity,Count
    metrics=[HOTA(),CLEAR({'PRINT_CONFIG':False}),Identity({'PRINT_CONFIG':False}),Count()]
    integer_fields={field for metric in metrics for field in metric.integer_fields}
    integer_array_fields={field+'___'+str(int(100*label)) for metric in metrics
                          for field in metric.integer_array_fields for label in metric.array_labels}
    sources=[VENDOR/'trackeval/metrics'/name for name in ['hota.py','clear.py','identity.py','count.py','_base_metric.py']]
    hashes={str(path):sha(path) for path in sources}
    tables={};results={};full_results={};controls=[];input_hashes={}
    def reconstruct(metric,row):
        result={field:float(row[field]) for field in metric.float_fields}
        result.update({field:int(float(row[field])) for field in metric.integer_fields})
        labels=[int(100*value) for value in metric.array_labels]
        for field in metric.float_array_fields+metric.integer_array_fields:
            result[field]=np.array([float(row[field+'___'+str(label)]) for label in labels],dtype=np.float64)
        return result
    def combine(table,sequences):
        detail={}
        for metric in metrics:
            rows={seq:reconstruct(metric,table[seq]) for seq in sequences}
            combined=metric.combine_sequences(rows)
            detail.update(metric.detailed_results({'COMBINED_SEQ':combined})['COMBINED_SEQ'])
        return {key:float(value) for key,value in detail.items()}
    def compare(actual,expected,label):
        assert set(actual)==set(expected)-{'seq'},label
        differences=[]
        for key in actual:
            reference=float(expected[key]);error=abs(actual[key]-reference)
            exact=key in integer_fields or key in integer_array_fields
            unit=math.ulp(max(1.0,abs(reference)))
            # Reconstructing printed per-sequence fields changes summation order
            # relative to the original evaluator. Counts remain exact; 32 ulps
            # cover the bounded f64 sum/product/divide/mean aggregation chain.
            bound=0.0 if exact else 32*unit
            assert error<=bound,(label,key,error,bound)
            if error:differences.append({'field':key,'absolute_error':error,'ulps_at_reference_scale':error/unit})
        return {'control':label,'numeric_fields':len(actual),'integer_counts_exact':True,
                'floating_bound':'32 binary64 ulps at max(1,abs(reference))',
                'max_absolute_error':max([item['absolute_error'] for item in differences] or [0]),
                'max_ULP_error':max([item['ulps_at_reference_scale'] for item in differences] or [0]),
                'nonzero_rounding_differences':differences,'result':'PASS'}
    group_defs={'02+10':['MOT17-02-SDP','MOT17-10-SDP'],
                'training5':['MOT17-04-SDP','MOT17-05-SDP','MOT17-09-SDP','MOT17-11-SDP','MOT17-13-SDP']}
    for mode in ['mot','fixed-motion']:
        tables[mode]={};results[mode]={group:{} for group in group_defs}
        full_results[mode]={group:{} for group in group_defs}
        for arm in ['main','official','finetuned']:
            path=RUN/'quality'/arm/mode/'trackeval/results/v1/pedestrian_detailed.csv'
            input_hashes[str(path)]=sha(path)
            table={row['seq']:row for row in csv.DictReader(path.open())}
            seqs=[key for key in table if key!='COMBINED']
            controls.append(compare(combine(table,seqs),table['COMBINED'],mode+'/'+arm+'/all7'))
            for group,selected in group_defs.items():
                detail=combine(table,selected)
                full_results[mode][group][arm]=detail
                results[mode][group][arm]={
                    **{key:100*detail[key+'___AUC'] for key in ['HOTA','DetA','AssA']},
                    **{key:100*detail[key] for key in ['MOTA','IDF1']},
                    **{key:int(detail[key]) for key in ['CLR_FP','CLR_FN','IDSW','Frag','GT_Dets','GT_IDs']},
                    'Frames':int(detail['CLR_Frames'])}
                assert results[mode][group][arm]['Frames']==(1254 if group=='02+10' else 4062)
                assert results[mode][group][arm]['GT_Dets']==(31420 if group=='02+10' else 80877)
                if arm=='main':
                    if group=='02+10':
                        prior=REPO/'build/yolo26m-threeway-20261111/main/mot/trackeval/results/v1/pedestrian_detailed.csv'
                    else:
                        prior=REPO/'evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/capture/scoring/in-domain/pedestrian_detailed.csv'
                    expected={row['seq']:row for row in csv.DictReader(prior.open())}['COMBINED']
                    input_hashes[str(prior)]=sha(prior)
                    controls.append(compare(detail,expected,mode+'/main/prior_actual_'+group))
    assert hashes=={str(path):sha(path) for path in sources}
    output={'result':'PASS','method':'vendored TrackEval metric.combine_sequences on reconstructed retained per-alpha arrays and integer counts',
            'new_GT_read':False,'new_scoring_assignments':False,'GPU_used':False,'groups':group_defs,
            'data_roles':'model-selection val10 plus post-M4 reused eval02; not independent blind test',
            'metric_source_hashes':hashes,'input_detailed_CSV_hashes':input_hashes,'controls':controls,
            'scores':results,'full_detailed_combined_fields':full_results}
    path=RUN/'pipeline-tools/quality-subgroups-validation.json'
    path.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    (RUN/'pipeline-tools/quality-subgroups.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'result':'PASS','controls':controls,'scores':results},indent=2),flush=True)

if __name__=='__main__':main()
