"""Frozen three-model native TRT capture over the 156-sequence primary pool."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
EVIDENCE=REPO/'evidence/model-comparison/2026-11-11-yolo26m-primary156'
PRIOR=REPO/'build/yolo26m-trt-comparison-20261111'
SOURCE=REPO.parent/'saccade_research/experiments/common/primary-selection-20261111.json'
TRACK=PRIOR/'pipeline-tools/track_sequence'
ENGINES={'main':Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'),
         'official':PRIOR/'artifacts/official/model.engine','finetuned':PRIOR/'artifacts/finetuned/model.engine'}
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'0'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.writing');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');tmp.replace(p)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def validate(arm,s,folder,engine_sha):
    target=ROOT/'validation-tools/cells'/arm/(s['id']+'.json')
    cmd=['python','-B',ROOT/'validation-tools/validate_cell.py','--packets',folder/'packets.bin',
         '--motion',folder/'motion.scmo','--config',folder/'resolved.config','--frames',s['frames'],
         '--width',s['width'],'--height',s['height'],'--arm',arm,'--sequence',s['id'],
         '--engine-sha256',engine_sha,'--out',target]
    with (folder/'validation.log').open('w') as f:r=subprocess.run(list(map(str,cmd)),cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'cell validation failed {target}: '+(folder/'validation.log').read_text()[-2000:])
    return {'path':str(target),'sha256':sha(target)}

def import_mot17(arm,s,out,engine_sha):
    old=PRIOR/'quality'/arm/s['id']
    capture=json.loads((REPO/'evidence/model-comparison/2026-11-11-yolo26m-trt-performance/quality-capture.json').read_text())[arm+'/'+s['id']]
    assert capture['native_CPU_exact'] and capture['frames']==s['frames']
    assert sha(old/'mot.txt')==capture['mot_sha256'] and sha(old/'packets.bin')==capture['packets_sha256']
    assert sha(old/'motion.scmo')==capture['motion_sha256']
    out.mkdir(parents=True,exist_ok=False)
    for name in ['mot.txt','packets.bin','motion.scmo','resolved.config','track.log']:
        shutil.copy2(old/name,out/name)
    validation=validate(arm,s,out,engine_sha)
    return {'status':'PASS','imported':True,'source_cell':str(old),'source_capture':capture,
            'frames':s['frames'],'engine_sha256':engine_sha,'mot_sha256':sha(out/'mot.txt'),
            'packets_sha256':sha(out/'packets.bin'),'motion_sha256':sha(out/'motion.scmo'),
            'resolved_config_sha256':sha(out/'resolved.config'),'validation':validation}

def fresh(arm,s,out,engine_sha):
    out.mkdir(parents=True,exist_ok=False)
    command=['flock','/home/ray/.cache/saccade-v1/gpu.lock',str(TRACK),str(ENGINES[arm]),str(s['inference_path']),str(out/'mot.txt'),
       '--profile','hr-1280x768','--set','tracker.kalman.nsa_enabled=false','--set','tracker.rows.measurement_box=false',
       '--packets',str(out/'packets.bin'),'--motion',str(out/'motion.scmo'),'--resolved-config',str(out/'resolved.config')]
    started=time.monotonic();begin=utc()
    save(out/'command.json',{'argv':command,'cwd':str(REPO),'started_utc':begin})
    try:
        with (out/'track.log').open('w') as f:
            r=subprocess.run(command,cwd=REPO,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=900)
        record={'started_utc':begin,'completed_utc':utc(),'exit_code':r.returncode,'wall_seconds_including_lock_wait':time.monotonic()-started}
    except subprocess.TimeoutExpired:
        record={'started_utc':begin,'completed_utc':utc(),'exit_code':'TIMEOUT','wall_seconds_including_lock_wait':time.monotonic()-started}
    save(out/'exit.json',record)
    if record['exit_code']!=0:raise RuntimeError(f'native capture failed {out}: '+(out/'track.log').read_text()[-3000:])
    validation=validate(arm,s,out,engine_sha)
    assert sha(ENGINES[arm])==engine_sha
    return {'status':'PASS','imported':False,'benchmark':s['benchmark'],'frames':s['frames'],
            'engine_sha256':engine_sha,'binary_sha256':sha(TRACK),'command':command,**record,
            'mot_sha256':sha(out/'mot.txt'),'packets_sha256':sha(out/'packets.bin'),
            'motion_sha256':sha(out/'motion.scmo'),'resolved_config_sha256':sha(out/'resolved.config'),
            'validation':validation,'native_interval_not_a_controlled_performance_measurement':True}

def main():
    assert not any(k.startswith('SACCADE_CFG_') for k in os.environ),'unfrozen config environment'
    d=json.loads((ROOT/'selection-runtime.json').read_text())
    sequences=d['sequences'];assert len(sequences)==156 and sum(s['frames'] for s in sequences)==135503
    assert not any(s['benchmark']=='SportsMOT' for s in sequences)
    before={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in [TRACK,*ENGINES.values(),ROOT/'run_capture.py',ROOT/'selection-runtime.json']}
    save(EVIDENCE/'capture-binaries-before.json',before)
    engine_sha={a:sha(p) for a,p in ENGINES.items()}
    all_cells={};failures=[]
    started=time.monotonic()
    # Import only the exact, already validated seven-sequence native TRT cells.
    # All other 149 sequences are fresh native captures for every arm.
    order=sorted(sequences,key=lambda s:(s['benchmark']!='MOT17',s['benchmark'],s['id']))
    for s in order:
        for arm in ENGINES:
            out=ROOT/'cells'/arm/s['id'];receipt=out/'capture.json';key=arm+'/'+s['id']
            if receipt.exists():
                rec=json.loads(receipt.read_text());assert rec['status']=='PASS' and rec['engine_sha256']==engine_sha[arm]
                assert sha(out/'mot.txt')==rec['mot_sha256'] and sha(out/'packets.bin')==rec['packets_sha256'] and sha(out/'motion.scmo')==rec['motion_sha256']
            else:
                try:
                    rec=import_mot17(arm,s,out,engine_sha[arm]) if s['benchmark']=='MOT17' else fresh(arm,s,out,engine_sha[arm])
                    save(receipt,rec)
                except Exception as exc:
                    failure={'arm':arm,'sequence':s['id'],'benchmark':s['benchmark'],'error':str(exc),'at':utc()}
                    save(out/'FAILURE.json',failure);failures.append(failure)
                    save(EVIDENCE/'failures.json',failures)
                    print(json.dumps({'status':'FAIL',**failure}),flush=True)
                    continue
            all_cells[key]=rec
            by_arm={a:{'sequences':sum(k.startswith(a+'/') for k in all_cells),'frames':sum(v['frames'] for k,v in all_cells.items() if k.startswith(a+'/'))} for a in ENGINES}
            progress={'at':utc(),'completed_cells':len(all_cells),'expected_cells':468,'completed_frames':sum(v['frames'] for v in all_cells.values()),'expected_frames':406509,'failures':len(failures),'elapsed_seconds':time.monotonic()-started,'by_arm':by_arm,'last':{'arm':arm,'sequence':s['id'],'imported':rec['imported']}}
            save(ROOT/'progress.json',progress);print(json.dumps(progress),flush=True)
    after={p:{'bytes':Path(p).stat().st_size,'sha256':sha(Path(p))} for p in before}
    save(EVIDENCE/'capture-binaries-after.json',after);assert before==after
    save(EVIDENCE/'capture-index.json',all_cells)
    save(EVIDENCE/'capture-complete.json',{'at':utc(),'complete':len(all_cells)==468 and not failures,'cells':len(all_cells),'frames':sum(v['frames'] for v in all_cells.values()),'failures':failures})
    assert len(all_cells)==468 and not failures,'full scope incomplete; failures retained, no sequences excluded'
    print('ALL_468_CELLS_COMPLETE',flush=True)

if __name__=='__main__':main()
