"""Independent post-campaign prefix equivalence and pooled-statistics audit."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import statistics
import struct

RUN=Path(__file__).resolve().parent.parent
BENCH=RUN/'benchmark-tools'
OUT=RUN/'pipeline-tools'
ARMS=['main','official','finetuned']
ORDER=['main','official','finetuned','finetuned','official','main']
SEQS=['MOT17-10-SDP','MOT17-02-SDP']

def rows(path):return list(csv.DictReader(path.open()))
def data(path):return json.loads(path.read_text())
def sha(payload):return hashlib.sha256(payload).hexdigest()
def close(a,b):assert abs(a-b)<1e-8,(a,b)
def p95(values):return sorted(values)[math.ceil(.95*len(values))-1]

def main():
    runs=data(BENCH/'pipeline-runs/runs.json')
    assert len(runs)==96
    matrix={(r['sequence'],r['path'],r['mode'],r['cycle'],r['ordinal']) for r in runs}
    assert matrix==set(itertools.product(SEQS,['detect','tracking'],['single','queued'],range(2),range(6)))
    reference={};checks=[];groups={};hashes={}
    for run in runs:
        seq,path,arm=run['sequence'],run['path'],run['arm']
        assert arm==ORDER[run['ordinal']] and run['exit_code']==0
        folder=Path(run['output']);summary=data(folder/'summary.json')
        key=(seq,path,arm)
        if key not in reference:
            quality=RUN/'quality'/arm/seq
            if path=='tracking':
                source=quality/'packets.bin'
                with source.open('rb') as f:expected=f.read(140*12368)
                assert len(expected)==140*12368
            else:
                source=quality/'detections.scdr'
                with source.open('rb') as f:
                    header=f.read(28);payload=f.read(140*9664)
                magic,version,size,epoch,count=struct.unpack('<4sIIQQ',header)
                assert (magic,version,size,epoch)==(b'SCDR',1,9664,1)
                assert count==(654 if seq=='MOT17-10-SDP' else 600) and len(payload)==140*9664
                expected=struct.pack('<4sIIQQ',magic,version,size,epoch,140)+payload
            reference[key]={'source':str(source),'expected':expected,'prefix_sha256':sha(expected)}
        artifact=folder/('packets.bin' if path=='tracking' else 'detections.scdr')
        actual=artifact.read_bytes()
        assert actual==reference[key]['expected'],(key,folder,'prefix differs from full native quality')
        actual_hash=sha(actual)
        assert actual_hash==summary['result_sha256']
        assert summary['status']=='PASS' and summary['frames']==140 and summary['warmup']==20 and summary['samples']==120
        assert summary['candidate_slots']==2 and summary['result_slots']==4
        assert summary['motion_slots']==(2 if path=='tracking' else 0)
        assert summary['tracking_graph']==(path=='tracking')
        checks.append({'sequence':seq,'path':path,'arm':arm,'mode':run['mode'],'output':str(folder),
                       'full_native_quality_prefix_exact':True,'result_sha256':actual_hash})
        sample=rows(folder/'delivery.csv');assert len(sample)==140
        assert [int(r['frame']) for r in sample]==list(range(140))
        assert [int(r['warmup']) for r in sample]==[1]*20+[0]*120
        values=[float(r['delivery_ms']) for r in sample[20:]]
        group=(seq,path,arm,run['mode'])
        g=groups.setdefault(group,{'latencies':[],'intervals':[],'folders':[]})
        g['latencies']+=values;g['intervals'].append(summary['interval_ms_per_frame']);g['folders'].append(str(folder))
        hashes.setdefault(key,set()).add(actual_hash)
    assert len(reference)==12 and all(len(value)==1 for value in hashes.values())
    prefix_report={'result':'PASS','pipeline_sessions_verified':96,'reference_prefixes':12,'frames_per_session':140,
                   'measured_native_quality_prefixes_exact':True,'all_single_queued_and_fresh_results_match':True,
                   'tracking_byte_scope':'all140complete12368-byte canonical packets, including unused/reserved bytes',
                   'detection_byte_scope':'all140complete9664-byte candidate frames, normalizing only SCDR fileheader count',
                   'references':{'/'.join(key):{k:v for k,v in value.items() if k!='expected'} for key,value in reference.items()},
                   'checks':checks,'GPU_used':False,'new_measurements':False}
    (OUT/'benchmark-prefix-review.json').write_text(json.dumps(prefix_report,indent=2)+'\n')
    summary=data(BENCH/'pipeline-runs/summary.json')
    assert summary['status']=='PASS' and len(summary['groups'])==24 and len(groups)==24
    pipeline_stats=[]
    for g in summary['groups']:
        key=tuple(g[k] for k in ['sequence','path','arm','mode'])
        actual=groups[key]
        assert len(actual['folders'])==4 and len(actual['latencies'])==480
        close(g['delivery_ms']['mean'],statistics.mean(actual['latencies']))
        close(g['delivery_ms']['p95'],p95(actual['latencies']))
        interval=statistics.mean(actual['intervals'])
        close(g['interval_ms_per_frame'],interval);close(g['throughput_fps'],1000/interval)
        pipeline_stats.append({'key':'/'.join(key),'samples':480,'fresh_processes':4,
                               'p95_delivery_ms':p95(actual['latencies']),'interval_ms_per_frame':interval,
                               'throughput_fps':1000/interval,'checks_pass':True})
    model_runs=data(BENCH/'model-runs/runs.json');assert len(model_runs)==20
    assert sum(r['comparison']=='raw6' for r in model_runs)==12
    assert sum(r['comparison']=='original-default' for r in model_runs)==8
    models={};model_hashes={}
    for run in model_runs:
        assert run['exit_code']==0
        folder=Path(run['output']);s=data(folder/'summary.json')
        assert s['status']=='PASS' and not s['verify_only'] and s['frames_verified']==4
        assert s['scan_status_count']==(9 if run['arm']=='main' else 0)
        assert s['aux_streams']==0 and s['persistent_cache_bytes']==0
        verification=tuple((r['input'],r['payload_sha256'],r['status_sha256']) for r in rows(folder/'verification.csv'))
        assert len(verification)==4
        model_hashes.setdefault((run['comparison'],run['arm']),set()).add(verification)
        for row in rows(folder/'samples.csv'):
            key=(run['comparison'],run['arm'],row['mode'])
            g=models.setdefault(key,{'gpu':[],'wall':[],'queued_gpu':[],'queued_wall':[],'replays':0,'folders':set()})
            g['gpu'].append(float(row['gpu_event_ms']));g['wall'].append(float(row['cpu_wall_ms']));g['folders'].add(str(folder))
        for row in rows(folder/'queued.csv'):
            g=models[run['comparison'],run['arm'],row['mode']]
            g['queued_gpu'].append(float(row['gpu_event_total_ms']));g['queued_wall'].append(float(row['cpu_wall_total_ms']))
            g['replays']+=int(row['replays'])
    expected={('raw6',arm,mode) for arm in ARMS for mode in ['ordinary','graph']}
    expected|={('original-default',arm,mode) for arm in ['main','official-default'] for mode in ['ordinary','graph']}
    assert set(models)==expected and all(len(values)==1 for values in model_hashes.values())
    model_summary=data(BENCH/'model-runs/summary.json');assert model_summary['status']=='PASS' and len(model_summary['groups'])==10
    model_stats=[]
    for g in model_summary['groups']:
        key=tuple(g[k] for k in ['comparison','arm','mode']);actual=models[key]
        assert len(actual['folders'])==4 and len(actual['gpu'])==1280
        close(g['gpu_event_single_ms']['mean'],statistics.mean(actual['gpu']))
        close(g['gpu_event_single_ms']['p95'],p95(actual['gpu']))
        close(g['cpu_wall_single_ms']['p95'],p95(actual['wall']))
        gpu_interval=sum(actual['queued_gpu'])/actual['replays'];wall_interval=sum(actual['queued_wall'])/actual['replays']
        close(g['queued_gpu_interval_ms'],gpu_interval);close(g['queued_cpu_interval_ms'],wall_interval)
        close(g['queued_gpu_fps'],1000/gpu_interval);close(g['queued_cpu_fps'],1000/wall_interval)
        model_stats.append({'key':'/'.join(key),'samples':1280,'fresh_processes':4,
                            'p95_gpu_event_ms':p95(actual['gpu']),'queued_gpu_interval_ms':gpu_interval,
                            'queued_gpu_fps':1000/gpu_interval,'checks_pass':True})
    for name in ['model-runs','pipeline-runs']:
        assert data(BENCH/name/'inputs-before.json')==data(BENCH/name/'inputs-after.json')
    stats_report={'result':'PASS','model_sessions':20,'model_groups':10,'model_invocations_per_group':1280,
                  'pipeline_sessions':96,'pipeline_groups':24,'pipeline_delivery_samples_per_group':480,
                  'full_expected_matrices_present':True,'engine_input_harness_hash_receipts_before_after_equal':True,
                  'pooled_nearest_rank_percentiles_recomputed':True,'weighted_queued_intervals_recomputed':True,
                  'per_arm_output_hashes_stable':True,'model_statistics':model_stats,'pipeline_statistics':pipeline_stats,
                  'GPU_used':False,'new_measurements':False}
    (OUT/'benchmark-statistics-review.json').write_text(json.dumps(stats_report,indent=2)+'\n')
    print(json.dumps({'prefix_result':'PASS','pipeline_sessions':96,'reference_prefixes':12,
                      'model_statistic_groups':10,'pipeline_statistic_groups':24},indent=2))

if __name__=='__main__':main()
