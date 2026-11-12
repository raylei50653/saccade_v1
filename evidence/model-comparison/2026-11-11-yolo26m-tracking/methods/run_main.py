"""Fresh deployed detector capture; fixed two-sequence quality comparison only."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
EVIDENCE = REPO / 'evidence/model-comparison/2026-11-11-yolo26m-tracking'
DATA = REPO.parent / 'dataset/MOT17/train'
RECORDS = REPO.parent / 'saccade_v1-worktrees/hr-records-v2'
ENGINE = Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')
DETECTOR = REPO / 'build/report1-live-20261111/cuda/detect_sequence'
SEQS = {'MOT17-10-SDP': 654, 'MOT17-02-SDP': 600}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')

def inventory():
    paths = [ENGINE, DETECTOR, Path('/home/ray/developer/ai/dataset/yolo26m.pt'),
             REPO.parent / 'saccade_v1_dev/build-hr/HR01-UA2-20261102/teacher/attempt-001/epoch_0004.ckpt']
    paths += sorted((REPO / 'third_party/TrackEval').rglob('*.py'))
    for s, n in SEQS.items():
        images = sorted((DATA/s/'img1').glob('*.jpg'))
        assert len(images) == n
        paths += images + [DATA/s/'seqinfo.ini', DATA/s/'gt/gt.txt', RECORDS/(s+'.scdr'), RECORDS/(s+'.scmo')]
    return {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}

if __name__ == '__main__':
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    save(EVIDENCE/'PLAN.json', {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Two-sequence, fixed-tracker/G1 detector comparison; development research, no deployment changes',
        'sequences': SEQS, 'arms': ['official', 'finetuned', 'main'],
        'detector_input': 'nvJPEG RGB, FP32 /255, float antialiased bilinear stretch 1280x768',
        'official_and_finetuned': 'original YOLO one2many head, gate disabled, strict FP32; not default Ultralytics one2one',
        'main': 'deployed native TensorRT FP16 engine; scan plugins and input/output FP32',
        'postprocess': 'native F04 reference for YOLO arms; native CUDA F04 for main; top300/conf0.001/NMS0.5, full kept and suppressed rows',
        'tracker': 'current native CPU track_offline, default tracker, NSA/measurement_box disabled; same MOT formatting and EOF interpolation',
        'motion': 'fixed G1 SCMO from main detector mask; identical motion records across all arms, not recomputed per detector',
        'scorer': 'vendored TrackEval MOT17 train pedestrian DO_PREPROC=True; per-sequence and COMBINED',
        'quality_metrics': ['HOTA','DetA','AssA','IDF1','MOTA','IDSW','Frag','FP','FN'],
        'limits': ['MOT17-10 was used for model selection', 'MOT17-02 is post-M4 reused evaluation, not a new independent evaluation',
                   'Detector precision/backend differs; this is a deployed-model comparison, not architecture-only ablation',
                   'No per-arm threshold tuning; no throughput/latency or sanitizer claims'],
    })
    before = inventory()
    save(EVIDENCE/'inputs-before.json', before)
    save(EVIDENCE/'source-before.json', {
        'head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'status': subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
        'native_detector_source_status': subprocess.check_output(['git','status','--short'],cwd=REPO/'build/report1-live-20261111/source',text=True),
    })
    results = []
    for s in SEQS:
        out = ROOT/'main'/(s+'.scdr')
        out.parent.mkdir(exist_ok=True)
        assert not out.exists(), out
        cmd = ['flock','/home/ray/.cache/saccade-v1/gpu.lock',str(DETECTOR),str(ENGINE),str(DATA/s),str(out),'--profile','hr-1280x768']
        save(ROOT/'main'/(s+'.command.json'),cmd)
        t = time.monotonic()
        with (ROOT/'main'/(s+'.log')).open('w') as f:
            r = subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        result = {'sequence':s,'exit_code':r.returncode,'seconds':time.monotonic()-t}
        if r.returncode == 0:
            result.update(sha256=sha(out), frozen_scdr_sha256=sha(RECORDS/(s+'.scdr')))
            result['matches_frozen'] = result['sha256'] == result['frozen_scdr_sha256']
        results.append(result)
        save(ROOT/'main/capture.json',results)
        print(json.dumps(result),flush=True)
        assert r.returncode == 0
        assert result['matches_frozen'], 'fresh main SCDR differs: inspect before scoring'
    after = inventory()
    save(EVIDENCE/'inputs-after-main.json',after)
    assert before == after, 'input custody changed'
