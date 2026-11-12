"""Native TRT quality capture: full seven sequences, fixed tracker, no tuning."""
from pathlib import Path
import csv,hashlib,json,os,subprocess,time,concurrent.futures

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
EVIDENCE=REPO/'evidence/model-comparison/2026-11-11-yolo26m-trt-performance'
DATA=REPO.parent/'dataset/MOT17/train'
RECORDS=REPO.parent/'saccade_v1-worktrees/hr-records-v2'
SEQ={'MOT17-02-SDP':600,'MOT17-04-SDP':1050,'MOT17-05-SDP':837,'MOT17-09-SDP':525,'MOT17-10-SDP':654,'MOT17-11-SDP':900,'MOT17-13-SDP':750}
ENGINES={'main':Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'),
         'official':ROOT/'artifacts/official/model.engine','finetuned':ROOT/'artifacts/finetuned/model.engine'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')

def run(cmd,log,gpu=False):
    actual=['flock','/home/ray/.cache/saccade-v1/gpu.lock',*map(str,cmd)] if gpu else list(map(str,cmd))
    save(log.with_suffix('.command.json'),actual)
    started=time.monotonic()
    with log.open('w') as out:r=subprocess.run(actual,cwd=REPO,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    receipt={'command':actual,'exit_code':r.returncode,'wall_seconds_including_lock_wait':time.monotonic()-started}
    save(log.with_suffix('.exit.json'),receipt)
    if r.returncode:raise RuntimeError(f'{log}: exit{r.returncode}: {log.read_text()[-2000:]}')
    return receipt

def parse_score(path):
    results={}
    for row in csv.DictReader(path.open()):
        results[row['seq']]={**{k:100*float(row[k+'___AUC']) for k in ['HOTA','DetA','AssA']},
             **{k:100*float(row[k]) for k in ['MOTA','IDF1']},
             **{k:int(float(row[k])) for k in ['CLR_FP','CLR_FN','IDSW','Frag','GT_Dets','GT_IDs']},
             'Frames':int(float(row['CLR_Frames']))}
    return results

def main():
    # Adapter preflight is owned by pipeline-tools; require independent protocol/parity receipts before running.
    assert all(p.exists() for p in ENGINES.values())
    tools=ROOT/'pipeline-tools'
    identity={str(p):sha(p) for p in [*ENGINES.values(),tools/'detect_sequence',tools/'track_sequence',REPO/'build/cpu/track_offline']}
    save(EVIDENCE/'quality-binaries-before.json',identity)
    all_receipts={}
    for arm,engine in ENGINES.items():
        d=ROOT/'quality'/arm;d.mkdir(parents=True,exist_ok=True)
        for s,n in SEQ.items():
            out=d/s;out.mkdir(exist_ok=False)
            run([tools/'detect_sequence',engine,DATA/s,out/'detections.scdr','--profile','hr-1280x768'],out/'detect.log',True)
            run([tools/'track_sequence',engine,DATA/s,out/'mot.txt','--profile','hr-1280x768',
                 '--set','tracker.kalman.nsa_enabled=false','--set','tracker.rows.measurement_box=false',
                 '--packets',out/'packets.bin','--motion',out/'motion.scmo','--resolved-config',out/'resolved.config'],out/'track.log',True)
            # Canonical packet validation and supplemental identical-motion replay are CPU-only.
            validator=REPO/'build/yolo26m-threeway-all7-20261111/replay-tools/validate_packets'
            run([validator,out/'packets.bin',n],out/'packet-validation.log')
            fixed=d/'fixed-motion';fixed.mkdir(exist_ok=True)
            run([REPO/'build/cpu/track_offline',out/'detections.scdr',RECORDS/(s+'.scmo'),fixed/(s+'.txt'),
                 '--resolved-config',fixed/(s+'.config'),'--packets',fixed/(s+'.packets.bin')],out/'fixed-replay.log')
            run([validator,fixed/(s+'.packets.bin'),n],out/'fixed-packet-validation.log')
            native=d/'mot';native.mkdir(exist_ok=True)
            (native/(s+'.txt')).write_bytes((out/'mot.txt').read_bytes())
            # Each native session is required to equal CPU reference with its own captured motion.
            run([REPO/'build/cpu/track_offline',out/'detections.scdr',out/'motion.scmo',out/'cpu-native.txt',
                 '--packets',out/'cpu-native.packets.bin'],out/'cpu-native.log')
            assert sha(out/'cpu-native.txt')==sha(out/'mot.txt'),(arm,s,'native MOT differs from CPU')
            assert sha(out/'cpu-native.packets.bin')==sha(out/'packets.bin'),(arm,s,'native packets differ from CPU')
            r={'frames':n,'detection_sha256':sha(out/'detections.scdr'),'motion_sha256':sha(out/'motion.scmo'),
               'mot_sha256':sha(out/'mot.txt'),'packets_sha256':sha(out/'packets.bin'),'native_CPU_exact':True}
            if arm=='main':
                old=REPO/'build/yolo26m-threeway-all7-20261111/main/mot'
                assert r['detection_sha256']==sha(RECORDS/(s+'.scdr'))
                assert r['motion_sha256']==sha(RECORDS/(s+'.scmo'))
                assert r['mot_sha256']==sha(old/(s+'.txt'))
                assert r['packets_sha256']==sha(old/(s+'.packets.bin'))
                r['matches_frozen_deployment']=True
            all_receipts[arm+'/'+s]=r
            save(EVIDENCE/'quality-capture.json',all_receipts)
            print(json.dumps({'arm':arm,'sequence':s,'status':'PASS',**r}),flush=True)
    configs=[]
    for a in ENGINES:
        for s in SEQ:
            configs.append(sorted(x for x in (ROOT/'quality'/a/s/'resolved.config').read_text().splitlines() if x.startswith(('tracker.','output.'))))
    assert all(x==configs[0] for x in configs)
    save(EVIDENCE/'shared-native-tracker-config.json',configs[0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs=[]
        for a in ENGINES:
            for mode in ['mot','fixed-motion']:
                folder=ROOT/'quality'/a/mode
                futs.append(pool.submit(run,[REPO/'scripts/eval_mot17.sh',folder,*SEQ],folder/'score.log'))
        for f in futs:f.result()
    results={mode:{a:parse_score(ROOT/'quality'/a/mode/'trackeval/results/v1/pedestrian_detailed.csv') for a in ENGINES} for mode in ['mot','fixed-motion']}
    for mode in results:
        for a in ENGINES:
            assert results[mode][a]['COMBINED']['Frames']==sum(SEQ.values())
        for s in [*SEQ,'COMBINED']:
            assert len({tuple(results[mode][a][s][k] for k in ['Frames','GT_Dets','GT_IDs']) for a in ENGINES})==1
    save(EVIDENCE/'quality-scores.json',results)
    assert identity=={p:sha(p) for p in identity}
    save(EVIDENCE/'quality-binaries-after.json',identity)
    print(json.dumps({'quality_complete':True,'frames_per_arm':sum(SEQ.values()),'scores':{mode:{a:results[mode][a]['COMBINED'] for a in ENGINES} for mode in results}},indent=2),flush=True)

if __name__=='__main__':main()
