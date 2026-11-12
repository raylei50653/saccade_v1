"""Score frozen three-way outputs with the same vendored TrackEval and AP50."""
from pathlib import Path
import concurrent.futures, csv, hashlib, json, os, shutil, struct, subprocess, math
from run_main import ROOT, REPO, EVIDENCE, DATA, RECORDS, SEQS, sha, save

ARMS = ['official','finetuned','main']

def validate_scdr(p, n):
    b = p.read_bytes()
    magic,version,size,epoch,count = struct.unpack_from('<4sIIQQ',b)
    assert (magic,version,size,epoch,count)==(b'SCDR',1,9664,1,n)
    assert len(b)==28+size*count
    kept=suppressed=0
    for f in range(n):
        frame=b[28+f*size:28+(f+1)*size]
        h=struct.unpack_from('<QQ12I',frame)
        assert (h[0],h[1],h[2],h[4],h[5],h[6],h[13])==(1,f,1,300,0,0x34504f46,0)
        assert h[3]<=300 and h[3]==h[7]+h[8]
        assert h[9]+h[10]+h[3]+h[11]+h[12]==20160
        seen=set(); rows=[]; flags=[]
        for j in range(h[3]):
            x1,y1,x2,y2,s,c,cid,flag=struct.unpack_from('<5f3I',frame,64+32*j)
            assert all(math.isfinite(v) for v in (x1,y1,x2,y2,s))
            assert x2>x1 and y2>y1 and 0.001-1e-9<=s<=1 and c==0 and cid<20160
            assert cid not in seen and flag in (1,2)
            seen.add(cid); rows.append((-s,cid)); flags.append(flag)
        assert rows==sorted(rows)
        assert flags.count(1)==h[7] and flags.count(2)==h[8]
        assert frame[64+32*h[3]:]==bytes(size-64-32*h[3])
        kept+=h[7];suppressed+=h[8]
    return {'frames':n,'kept':kept,'suppressed':suppressed,'sha256':sha(p)}

def checked_run(cmd, log):
    with log.open('w') as f:
        r=subprocess.run([str(c) for c in cmd],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,
                         env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2'})
    assert r.returncode==0, (cmd,log.read_text()[-2000:])

def ap_job(arm,s):
    out=ROOT/arm/(s+'.ap50.json')
    checked_run(['/home/ray/.cache/saccade-v1/t01/venv/bin/python',REPO/'training/hr_deploy/eval_scdr.py',ROOT/arm/(s+'.scdr'),DATA/s],out)
    return arm,s,json.loads(out.read_text())

if __name__=='__main__':
    validations={}
    configurations=[]
    for arm in ARMS:
        (ROOT/arm/'mot').mkdir(exist_ok=True)
        for s,n in SEQS.items():
            if arm=='main':
                src=ROOT/'replay-tools'/s
                shutil.copy2(src/(s+'.txt'),ROOT/arm/'mot'/(s+'.txt'))
                shutil.copy2(src/'resolved.config',ROOT/arm/'mot'/(s+'.config'))
                shutil.copy2(src/'packets.bin',ROOT/arm/'mot'/(s+'.packets.bin'))
            validations[arm+'/'+s]=validate_scdr(ROOT/arm/(s+'.scdr'),n)
            config=ROOT/arm/'mot'/(s+'.config')
            # Only the tracker and formatter values affect the controlled replay.
            lines=sorted(x for x in config.read_text().splitlines() if x.startswith(('tracker.','output.')))
            assert lines
            configurations.append(lines)
            checked_run([ROOT/'replay-tools/validate_packets',ROOT/arm/'mot'/(s+'.packets.bin'),n],ROOT/arm/'mot'/(s+'.packet-validation.log'))
    assert all(x==configurations[0] for x in configurations), 'tracker/MOT config mismatch'
    save(EVIDENCE/'scdr-validation.json',validations)
    save(EVIDENCE/'shared-tracker-config.json',configurations[0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(checked_run,[REPO/'scripts/eval_mot17.sh',ROOT/arm/'mot',*SEQS],ROOT/arm/'mot/scoring.log') for arm in ARMS]
        for j in jobs:j.result()
    results={}
    for arm in ARMS:
        rows=list(csv.DictReader((ROOT/arm/'mot/trackeval/results/v1/pedestrian_detailed.csv').open()))
        results[arm]={}
        for row in rows:
            results[arm][row['seq']]={
                **{k:100*float(row[k+'___AUC']) for k in ['HOTA','DetA','AssA']},
                **{k:100*float(row[k]) for k in ['IDF1','MOTA']},
                **{k:int(float(row[k])) for k in ['IDSW','Frag','CLR_FP','CLR_FN','GT_Dets','GT_IDs']},
                'Frames':int(float(row['CLR_Frames'])),
            }
        # Confirm full sequence and GT support are identical for every arm.
        for s,n in SEQS.items(): assert results[arm][s]['Frames']==n
    for s in [*SEQS,'COMBINED']:
        assert len({tuple(results[a][s][k] for k in ('Frames','GT_Dets','GT_IDs')) for a in ARMS})==1
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for arm,s,result in pool.map(lambda args:ap_job(*args),[(a,s) for a in ARMS for s in SEQS]):
            results[arm][s]['AP50']=result['ap50_percent']
    save(EVIDENCE/'scores.json',results)
    table=EVIDENCE/'scores.csv'
    with table.open('w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','sequence','HOTA','DetA','AssA','IDF1','MOTA','IDSW','Frag','CLR_FP','CLR_FN','GT_Dets','GT_IDs','Frames','AP50'])
        w.writeheader()
        for a in ARMS:
            for s,r in results[a].items():w.writerow({'arm':a,'sequence':s,**r})
    print(json.dumps(results,indent=2))
