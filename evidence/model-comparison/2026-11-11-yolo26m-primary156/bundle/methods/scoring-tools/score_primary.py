"""Freeze, prepare and score the seven disjoint primary groups; no GPU usage."""
from pathlib import Path
import argparse
import concurrent.futures
import configparser
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=ROOT.parent.parent
OUT=ROOT/'scoring'
VENDOR=HERE/'frozen-TrackEval'
PYTHON=Path('/home/ray/.cache/saccade-v1/eval/venv/bin/python')
ARMS=['official','finetuned','main']
EXPECTED={'DanceTrack/all':(65,67304),'MOT15/moving':(5,3185),'MOT15/other':(6,2315),
          'MOT17/all':(7,5316),'MOT20/all':(4,8931),'PP22/moving':(9,4004),'PP22/static':(60,44448)}
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
     'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):result.update(block)
    return result.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.writing');temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)
def write_bound(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=payload:raise RuntimeError('immutable prepared output differs: '+str(path))
    else:path.write_bytes(payload)

def freeze(selection_path):
    selection=load(selection_path)
    seqs=selection['sequences'];by={s['id']:s for s in seqs};groups=selection['primary_groups']
    assert len(seqs)==len(by)==156 and sum(s['frames'] for s in seqs)==135503
    assert set(groups)==set(EXPECTED) and not any(s['benchmark']=='SportsMOT' for s in seqs)
    flat=[name for ids in groups.values() for name in ids]
    assert len(flat)==len(set(flat))==156 and set(flat)==set(by)
    for group,(count,frames) in EXPECTED.items():
        assert len(groups[group])==count and sum(by[name]['frames'] for name in groups[group])==frames
        assert all(by[name]['benchmark']==group.split('/')[0] for name in groups[group])
    policy_path=Path(selection['policy']['path'])
    assert sha(policy_path)==selection['policy']['sha256']
    policy=load(policy_path)
    assert not policy['excluded_from_default']['SportsMOT']['default_scoring']
    write_bound(HERE/'selection-frozen.json',Path(selection_path).read_bytes())
    write_bound(HERE/'policy-frozen.json',policy_path.read_bytes())
    source=REPO/'third_party/TrackEval';manifest=load(source/'SOURCE.json')
    if not VENDOR.exists():shutil.copytree(source,VENDOR,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    algorithm_checks={}
    historical=REPO.parent/'saccade_research/runs/bridge-pair-fullpool-replay/20261108-002/scoring/tool'
    for relative,expected in manifest['unchanged_files_sha256'].items():
        assert sha(source/relative)==sha(VENDOR/relative)==expected
        if relative.endswith('.py'):assert sha(historical/relative)==expected
        algorithm_checks[relative]={'sha256':expected,'matches_current_vendor':True,'matches_frozen_historical_algorithm':relative.endswith('.py')}
    for relative,item in manifest['modified_initializers'].items():
        assert sha(source/relative)==sha(VENDOR/relative)==item['vendored_sha256']
    core=[HERE/'score_primary.py',HERE/'worker.py',HERE/'aggregate_primary.py']
    lock={'schema':'primary156-scoring-freeze/1','selection_sha256':sha(HERE/'selection-frozen.json'),
          'policy_sha256':sha(HERE/'policy-frozen.json'),'core_source_sha256':{str(p):sha(p) for p in core},
          'algorithm_checks':algorithm_checks,'vendor_source_json_sha256':sha(VENDOR/'SOURCE.json'),
          'arms':ARMS,'primary_groups':{g:{'sequences':n,'frames':f} for g,(n,f) in EXPECTED.items()},
          'sequence_count':156,'frames_per_arm':135503,
          'GT_semantics':{'MOT15':'MOT15 adapter; no MOT17 class/distractor preprocessing',
                          'MOT20':'MOT20 adapter with class6 distractor rule',
                          'DanceTrack':'retain first7GT fields; set class1 and visibility1; MOT17 adapter',
                          'PP22':'score predictions only at GT-present original frame numbers; retain fullseqLength; MOT17 adapter',
                          'MOT17':'MOT17 adapter pedestrian DO_PREPROC=True'},
          'aggregation':'genuine TrackEval combine_sequences; no averaging sequence scores',
          'exposure':'all156 already exposed; MOT17 includes training/selection and postM4 reused eval',
          'GPU_used':False,'python':str(PYTHON)}
    if (HERE/'freeze.json').exists():assert load(HERE/'freeze.json')==lock,'scorer freeze changed'
    else:save(HERE/'freeze.json',lock)
    OUT.mkdir(exist_ok=True)
    return selection,by,groups,lock

def capture_status(ids,cells,by):
    waiting=[];failed=[]
    for arm in ARMS:
        for name in ids:
            path=cells/arm/name/'capture.json'
            if not path.exists():waiting.append(arm+'/'+name);continue
            try:record=load(path)
            except json.JSONDecodeError:waiting.append(arm+'/'+name);continue
            if record.get('status')!='PASS':failed.append({'cell':arm+'/'+name,'status':record.get('status')});continue
            if record.get('frames')!=by[name]['frames']:raise RuntimeError('capture frame count differs: '+str(path))
            if not (path.parent/'mot.txt').is_file():raise RuntimeError('PASS capture lacks MOT: '+str(path))
    return waiting,failed

def prepare_group(group,ids,by,cells,lock):
    stage=OUT/'groups'/group.replace('/','__');stage.mkdir(parents=True,exist_ok=True)
    gtroot=stage/'prepared/gt';trackroot=stage/'prepared/trackers'
    binding={};ground_truth=[];filters=[];capture_receipts={}
    for name in ids:
        seq=by[name];n=seq['frames'];rawgt=Path(seq['ground_truth']['path']);rawmeta=Path(seq['sequence_info']['path'])
        assert sha(rawgt)==seq['ground_truth']['sha256'] and sha(rawmeta)==seq['sequence_info']['sha256']
        gtbytes=rawgt.read_bytes();metabytes=rawmeta.read_bytes()
        info=configparser.ConfigParser();info.read_string(metabytes.decode())
        assert int(info['Sequence']['seqLength'])==n
        lines=gtbytes.decode().splitlines();gt_frames=set();transformed=[];rows=0
        for line in lines:
            if not line.strip():continue
            fields=line.split(',');assert len(fields)>=7
            fid=int(fields[0]);assert 1<=fid<=n
            gt_frames.add(fid);rows+=1
            if seq['benchmark']=='DanceTrack':transformed.append(','.join(fields[:7]+['1','1'])+'\n')
        prepared_gt=''.join(transformed).encode() if seq['benchmark']=='DanceTrack' else gtbytes
        destgt=gtroot/name/'gt/gt.txt';destmeta=gtroot/name/'seqinfo.ini'
        write_bound(destgt,prepared_gt);write_bound(destmeta,metabytes)
        binding[str(destgt.resolve())]=sha(destgt);binding[str(destmeta.resolve())]=sha(destmeta)
        ground_truth.append({'sequence':name,'benchmark':seq['benchmark'],'frames':n,'GT_rows':rows,
            'GT_present_original_frames':len(gt_frames),'source_GT':str(rawgt),'source_GT_sha256':sha(rawgt),
            'prepared_GT_sha256':sha(destgt),'source_metadata':str(rawmeta),'source_metadata_sha256':sha(rawmeta),
            'prepared_seqLength_unchanged':True,'DanceTrack_class_visibility_transform':seq['benchmark']=='DanceTrack'})
        for arm in ARMS:
            source=cells/arm/name/'mot.txt';capture=cells/arm/name/'capture.json';receipt=load(capture)
            assert receipt['status']=='PASS' and receipt['frames']==n and sha(source)==receipt['mot_sha256']
            payload=source.read_bytes();out=[];count=kept=0;previous=0;seen=set()
            for line in payload.splitlines(keepends=True):
                if not line.strip():continue
                fields=line.rstrip(b'\r\n').split(b',');assert len(fields)==10
                fid=int(fields[0]);uid=int(fields[1]);assert 1<=fid<=n and uid>0 and fid>=previous
                if fid!=previous:seen.clear();previous=fid
                assert uid not in seen;seen.add(uid)
                assert all(math.isfinite(float(value)) for value in fields)
                count+=1
                if seq['benchmark']!='PP22' or fid in gt_frames:out.append(line);kept+=1
            prepared=b''.join(out) if seq['benchmark']=='PP22' else payload
            target=trackroot/arm/(name+'.txt');write_bound(target,prepared)
            binding[str(target.resolve())]=sha(target)
            capture_receipts[arm+'/'+name]={'path':str(capture),'sha256':sha(capture),
                                          'engine_sha256':receipt['engine_sha256'],'mot_sha256':receipt['mot_sha256']}
            filters.append({'sequence':name,'arm':arm,'benchmark':seq['benchmark'],'input_rows':count,'kept_rows':kept,
                            'removed_rows':count-kept,'GT_present_original_frames':len(gt_frames),'full_seqLength':n,
                            'source':str(source),'source_sha256':sha(source),'prepared':str(target),'prepared_sha256':sha(target),
                            'frame_numbers_preserved':True,'filter_applied':seq['benchmark']=='PP22'})
    benchmark='MOT15' if group.startswith('MOT15/') else 'MOT20' if group.startswith('MOT20/') else 'MOT17'
    config={'group':group,'frozen_vendor':str(VENDOR),'worker_sha256':sha(HERE/'worker.py'),
            'freeze_sha256':sha(HERE/'freeze.json'),'input_hashes':binding,
            'capture_receipts':capture_receipts,'expected_frames':{name:by[name]['frames'] for name in ids},
            'dataset':{'GT_FOLDER':str(gtroot),'TRACKERS_FOLDER':str(trackroot),'OUTPUT_FOLDER':str(stage/'results'),
                       'SKIP_SPLIT_FOL':True,'TRACKER_SUB_FOLDER':'','TRACKERS_TO_EVAL':ARMS,'CLASSES_TO_EVAL':['pedestrian'],
                       'BENCHMARK':benchmark,'SPLIT_TO_EVAL':'train','DO_PREPROC':True,'PRINT_CONFIG':False,
                       'SEQ_INFO':{name:None for name in ids}}}
    for filename,value in [('config.json',config),('ground-truth-preparation.json',ground_truth),('keyframe-filter.json',filters)]:
        target=stage/filename
        if target.exists():assert load(target)==value,'prepared scoring config changed: '+str(target)
        else:save(target,value)
    assert lock==load(HERE/'freeze.json')
    return stage

def score_group(group,ids,by,cells,lock):
    stage=prepare_group(group,ids,by,cells,lock)
    if (stage/'metrics.json').exists():
        assert load(stage/'metrics.json')['result']=='PASS'
        assert load(stage/'execution.json')['exit_code']==0
        return group
    if (stage/'execution.json').exists():raise RuntimeError('failed prior scoring event retained; do not overwrite: '+group)
    command=[str(PYTHON),'-I','-B',str(HERE/'worker.py'),str(stage)]
    started=now();clock=time.monotonic()
    with (stage/'worker.log').open('x') as log:result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=ENV)
    save(stage/'execution.json',{'command':command,'started_utc':started,'completed_utc':now(),
                                'wall_seconds':time.monotonic()-clock,'exit_code':result.returncode,
                                'worker_sha256':sha(HERE/'worker.py'),'config_sha256':sha(stage/'config.json')})
    if result.returncode:raise RuntimeError('group scoring failed: '+group+' '+(stage/'worker.log').read_text()[-2500:])
    assert load(stage/'metrics.json')['result']=='PASS'
    print(json.dumps({'group':group,'status':'PASS','sequences':len(ids),'frames':sum(by[name]['frames'] for name in ids)}),flush=True)
    return group

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,default=ROOT/'selection-runtime.json')
    parser.add_argument('--cells',type=Path,default=ROOT/'cells')
    parser.add_argument('--workers',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args();selection,by,groups,lock=freeze(args.selection)
    if args.freeze_only:print(json.dumps({'freeze':'PASS','groups':list(groups),'frames':135503}));return
    complete=set();running={};pending=set(groups);failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending or running:
            for future,group in list(running.items()):
                if future.done():
                    future.result();complete.add(group);del running[future]
            status={}
            for group in groups:
                if group not in pending:continue
                waiting,failed=capture_status(groups[group],args.cells,by)
                status[group]={'waiting_cells':len(waiting),'failed_cells':failed,'sequence_count':len(groups[group])}
                if not waiting and not failed and len(running)<args.workers:
                    future=pool.submit(score_group,group,groups[group],by,args.cells,lock)
                    running[future]=group;pending.remove(group)
            save(OUT/'progress.json',{'updated_utc':now(),'status':'SCORING_OR_WAITING','complete_groups':sorted(complete),
                                     'running_groups':list(running.values()),'pending_groups':status,'workers':args.workers})
            if pending and not running and not args.watch:
                raise RuntimeError('capture groups not complete; use --watch to wait: '+json.dumps(status))
            if pending or running:time.sleep(5)
    command=[str(PYTHON),'-I','-B',str(HERE/'aggregate_primary.py'),str(OUT),str(HERE)]
    with (OUT/'aggregation.log').open('x') as log:result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=ENV)
    if result.returncode:raise RuntimeError('aggregation failed: '+(OUT/'aggregation.log').read_text()[-2500:])
    save(OUT/'progress.json',{'status':'PASS','completed_utc':now(),'complete_groups':sorted(complete),
                              'sequence_count':156,'frames_per_arm':135503,'GPU_used':False})
    print(json.dumps({'result':'PASS','groups':len(complete),'sequence_count':156,'frames_per_arm':135503}),flush=True)

if __name__=='__main__':main()
