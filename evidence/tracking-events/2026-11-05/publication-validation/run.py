"""Publication checks; run from repo root with existing built publication-cpu binaries."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
EVIDENCE = ROOT/'evidence/tracking-events/2026-11-05'
CAPTURE = EVIDENCE/'publication-validation'
BUILD = ROOT/'build/tracking-events'
BINARY = BUILD/'publication-cpu/tracking_event_trace'
TRACE = BUILD/'publication-trace'
VIEWER = BUILD/'publication-viewer'
PYTHON = Path(sys.executable)
RECORDS = Path('/home/ray/developer/ai/saccade_v1-worktrees/hr-records-v2')
SCDR = RECORDS/'MOT17-10-SDP.scdr'
SCMO = RECORDS/'MOT17-10-SDP.scmo'
BASELINE = ROOT/'evidence/occlusion/2026-11-05/run/baseline/MOT17-10-SDP.txt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


sources = subprocess.check_output(['git','ls-files','--','CMakeLists.txt','cmake','src','include'],text=True).splitlines()
sources += ['tools/tracking_event_trace.cpp','scripts/tracking_events.py','tests/tracking_events_tests.py',
            'tools/tracking_event_viewer.html']
source_hashes = {p:sha(ROOT/p) for p in sources}
frozen = {str(p):sha(p) for p in [BINARY,SCDR,SCMO,BASELINE]}
original_evidence = {str(p.relative_to(EVIDENCE)):sha(p) for p in EVIDENCE.rglob('*')
                     if p.is_file() and not p.is_relative_to(CAPTURE) and p.name != 'REPORT.md'}
dump(CAPTURE/'source-files.sha256.json',source_hashes)
manifest = {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_dirty':True,'formal_acceptance':False,'binary_inputs_sha256':frozen,'commands':[]}


def run(label, command, success=True):
    cmd = list(map(str,command))
    with (CAPTURE/f'{label}.log').open('w') as f:
        result = subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    manifest['commands'].append({'label':label,'argv':cmd,'exit':result.returncode})
    dump(CAPTURE/'manifest.json',manifest)
    if success and result.returncode:
        raise RuntimeError(f'{label} exited {result.returncode}')
    if not success and result.returncode == 0:
        raise RuntimeError(f'{label} should have failed')


run('ctest',['ctest','--test-dir',BUILD/'publication-cpu','--output-on-failure'])
run('event-tests',[PYTHON,ROOT/'tests/tracking_events_tests.py'])
run('trace',[BINARY,SCDR,SCMO,TRACE])
run('extract',[PYTHON,ROOT/'scripts/tracking_events.py','--trace',TRACE,
               '--baseline',BASELINE,'--output',VIEWER])
for name in ('events.csv','events.json'):
    assert (VIEWER/name).read_bytes() == (EVIDENCE/name).read_bytes(),name
assert (TRACE/'MOT17-10-SDP.txt').read_bytes() == BASELINE.read_bytes()
frames = [json.loads(line) for line in (TRACE/'trace.jsonl').read_text().splitlines()]
for frame in frames:
    for group in ('before','predicted','after'):
        for t in frame[group]:
            assert 1 <= t['birth_frame'] <= t['last_observed_frame'] <= frame['frame']
    for t in frame['after']:
        if t['age'] == 0:
            assert t['last_observed_frame'] == frame['frame']
run('existing-trace',[BINARY,'missing.scdr','missing.scmo',TRACE],success=False)
run('existing-viewer',[PYTHON,ROOT/'scripts/tracking_events.py','--trace',TRACE,
                       '--baseline',BASELINE,'--output',VIEWER],success=False)
tampered = BUILD/'publication-tampered.txt'
tampered.write_bytes(BASELINE.read_bytes()+b'\n')
rejected = BUILD/'publication-rejected'
assert not rejected.exists()
run('tampered-baseline',[PYTHON,ROOT/'scripts/tracking_events.py','--trace',TRACE,
                         '--baseline',tampered,'--output',rejected],success=False)
assert not rejected.exists()
# Both processes use one new leaf. Exactly one must acquire it; its output stays whole.
concurrent = BUILD/'publication-concurrent-trace'
assert not concurrent.exists()
def race(i):
    with (CAPTURE/f'concurrent-{i}.log').open('w') as log:
        return subprocess.run(list(map(str,[BINARY,SCDR,SCMO,concurrent])),stdout=log,stderr=subprocess.STDOUT).returncode
with ThreadPoolExecutor(max_workers=2) as pool:
    race_exits = list(pool.map(race,range(2)))
assert sorted(race_exits) == [0,1],race_exits
assert (concurrent/'MOT17-10-SDP.txt').read_bytes() == BASELINE.read_bytes()
assert len((concurrent/'trace.jsonl').read_text().splitlines()) == 654
summary = json.loads((VIEWER/'summary.json').read_text())
assert summary['inputs_unchanged'] and summary['checks']['IDSW']['official']==56
assert source_hashes == {p:sha(ROOT/p) for p in sources}
assert frozen == {p:sha(Path(p)) for p in frozen}
assert original_evidence == {p:sha(EVIDENCE/p) for p in original_evidence}
manifest.update(status='complete',inputs_binary_sources_unchanged=True,original_evidence_unchanged=True,
                event_csv_json_unchanged=True,packet_to_mot_bytes_equal=True,
                one_based_track_frames_checked=True,concurrent_exits=race_exits,
                payload_sha256={str(p):sha(p) for p in [TRACE/'trace.jsonl',TRACE/'MOT17-10-SDP.txt',VIEWER/'index.html']})
dump(CAPTURE/'manifest.json',manifest)
original = json.loads((EVIDENCE/'source-files.sha256.json').read_text())
dump(CAPTURE/'source-delta.json',{p:{'original':original.get(p),'publication':h}
                               for p,h in source_hashes.items() if original.get(p)!=h})
print(json.dumps({k:v for k,v in manifest.items() if k not in ('binary_inputs_sha256','payload_sha256','commands')},indent=2))
