#!/usr/bin/env python3
"""Verify completed capture and preserve reviewable F04 evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
SOURCE = STAGE / 'source'
CUSTODY = STAGE / 'custody'
DEST = ROOT / 'evidence/f04/2026-10-31-formal-995bca5'
CAP = SOURCE / 'build-acceptance/capture'
MUT = SOURCE / 'build-formal-mutations/capture'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


status = json.loads((CUSTODY / 'formal-status.json').read_text())
assert status['exit'] == 0
assert (CAP / 'standalone-checks.txt').read_text() == 'PASS\n'
assert 'exit=0\n' in (CAP / 'capture-status.txt').read_text()
assert 'complete=true\n' in (MUT / 'status.txt').read_text()
assert 'exit=0\n' in (MUT / 'status.txt').read_text()
full = (CAP / 'full-replay.log').read_text()
assert re.search(r'^source_dirty=false$', full, re.M)
assert re.search(r'^source_commit=995bca50d41510611b3990c9f8a020a9d8b76074$', full, re.M)
assert 'checks=34915 max_score_abs_error=1.19209e-07 PASS' in full
assert 'streams=PASS held=PASS' in full
assert len(re.findall(r'^case=.* PASS$', full, re.M)) == 49
assert len(re.findall(r'^reuse_phase=.* PASS$', full, re.M)) == 10
assert len(re.findall(r'^f03=.* PASS$', full, re.M)) == 6
assert 'CPU fixture invariants=2805 PASS' in (CAP / 'fixtures.log').read_text()
assert '100% tests passed out of 4' in (CAP / 'ctest.log').read_text()
last_test = (SOURCE / 'build-acceptance/cuda/Testing/Temporary/LastTest.log').read_text()
assert 'postprocess_reference: 442 checks passed' in last_test
assert last_test.count('streams=PASS held=PASS') == 3
sanitizers = {}
for tool in ['memcheck', 'initcheck', 'racecheck', 'synccheck']:
    text = (CAP / f'sanitizer-{tool}.log').read_text()
    summary = ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
               if tool == 'racecheck' else '========= ERROR SUMMARY: 0 errors')
    assert text.splitlines().count(summary) == 1
    assert 'checks=34610 max_score_abs_error=1.19209e-07 PASS' in text
    assert 'streams=PASS held=UNEXECUTED' in text
    assert len(re.findall(r'^f03=.* PASS$', text, re.M)) == 6
    assert re.findall(r'^f03=.* PASS$', text, re.M) == re.findall(r'^f03=.* PASS$', full, re.M)
    assert (CAP / f'sanitizer-{tool}.exit').read_text().strip() == '0'
    assert (CAP / f'control-{tool}.exit').read_text().strip() == '99'
    diagnostic = {'memcheck': 'Invalid __global__ write',
                  'initcheck': 'Uninitialized __global__ memory read',
                  'racecheck': 'Error: Race reported between',
                  'synccheck': 'Barrier error detected'}[tool]
    assert diagnostic in (CAP / f'control-{tool}.log').read_text()
    sanitizers[tool] = {'checks': 34610, 'summary': summary, 'positive_control_exit': 99,
                        'callback_gates': 'UNEXECUTED'}
results = (MUT / 'results.tsv').read_text().splitlines()[1:]
assert len(results) == 13 and all('\tDETECTED\t1\t' in x for x in results)
for row in results:
    variant, result, exit_code, diagnostic = row.split('\t')
    assert diagnostic in (MUT / f'{variant}-run.stderr').read_text().splitlines()
    assert (MUT / f'{variant}-run.exit').read_text().strip() == '1'
    assert (MUT / f'{variant}-build.exit').read_text().strip() == '0'
assert 'checks=34610 max_score_abs_error=1.19209e-07 PASS' in (MUT / 'restore-baseline.stdout').read_text()
assert re.findall(r'^f03=.* PASS$', (MUT / 'restore-baseline.stdout').read_text(), re.M) == re.findall(r'^f03=.* PASS$', full, re.M)
original = json.loads((CUSTODY / 'source-before.json').read_text())
assert original == json.loads((CUSTODY / 'source-after.json').read_text())
old_evidence = {p: h for p, h in original['tracked_files'].items() if p.startswith('evidence/')}
assert all(sha(ROOT / p) == h for p, h in old_evidence.items())
tools = json.loads((CUSTODY / 'sanitizer-installation-hashes.json').read_text())
assert all(sha(Path(p)) == h for p, h in tools['files'].items())
assert json.loads((CUSTODY / 'binary-library-hashes-after-acceptance.json').read_text()) == json.loads((CUSTODY / 'binary-library-hashes-final.json').read_text())

# Preserve actual binary/library/raw bytes locally; commit-sized evidence holds hashes.
bundle = STAGE / 'formal-binaries-and-replay.tar.gz'
with tarfile.open(bundle, 'w:gz') as tar:
    for p in sorted((STAGE / 'inputs').glob('*.f32')):
        tar.add(p, arcname=str(p.relative_to(STAGE)))
    binaries = json.loads((CUSTODY / 'binary-library-hashes-final.json').read_text())
    for name, expected in binaries.items():
        p = SOURCE / name
        assert sha(p) == expected
        tar.add(p, arcname=name)
summary = {'prior_custody_attempt_exit': 1, 'date': datetime.datetime.now().astimezone().isoformat(), 'verdict': 'PASS',
           'scope': 'standalone F04 raw predictions to complete DetectionCandidates',
           'source_revision': original['revision'], 'source_tree': original['tree'], 'source_dirty': False,
           'cpu_reference_checks': 442, 'fixture_invariants': 2805, 'fixed_cases': 49,
           'reuse_phases': 10, 'ctest_passed': 4, 'normal_checks': 34915,
           'normal_callback_gates': 'PASS', 'lazy_stream_callback_gates': 'PASS',
           'replay_inputs': 6, 'max_score_abs_error': 1.19209e-7,
           'sanitizers': sanitizers, 'mutations_detected': 13, 'mutations_total': 13,
           'restored_baseline_checks': 34610, 'binary_bytes_fixed_through_acceptance_and_controls': True, 'reproducible_rebuild': 'not claimed',
           'historical_evidence_files_unchanged': len(old_evidence),
           'local_artifacts': [{'path': str(bundle), 'sha256': sha(bundle), 'bytes': bundle.stat().st_size},
               {'path': str(CUSTODY / 'source-995bca5.tar.gz'), 'sha256': sha(CUSTODY / 'source-995bca5.tar.gz'),
                'bytes': (CUSTODY / 'source-995bca5.tar.gz').stat().st_size}],
           'open': ['complete async session', 'F03 plugin/finite status aggregation', 'P01',
                    'F04b/GMC/tracker/ResultPacket/MOT writer integration', 'model quality']}
(CUSTODY / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
# Verify the immediate archive retains the actual original accepted bytes.
with tarfile.open(STAGE / 'accepted-binaries.tar.gz') as archive:
    for member in archive.getmembers():
        if member.isfile():
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == binaries[member.name]
DEST.mkdir()
shutil.copytree(CAP, DEST / 'capture')
shutil.copytree(MUT, DEST / 'mutations')
shutil.copytree(CUSTODY, DEST / 'custody', ignore=shutil.ignore_patterns('source-995bca5.tar.gz'))
previous = STAGE.parent / 'build-f04-formal-995bca5'
failed = DEST / 'custody-attempt-rebuild'
shutil.copytree(previous / 'source/build-acceptance/capture', failed / 'capture')
shutil.copytree(previous / 'source/build-formal-mutations/capture', failed / 'mutations')
shutil.copytree(previous / 'custody', failed / 'custody', ignore=shutil.ignore_patterns('source-995bca5.tar.gz'))
shutil.copy2(previous / 'run_formal.py', failed / 'custody/run_formal.py')
for name in ['run_formal.py', 'preserve_formal.py']:
    shutil.copy2(STAGE / name, DEST / 'custody' / name)
shutil.copy2(SOURCE / 'build-acceptance/cuda/Testing/Temporary/LastTest.log', DEST / 'capture/LastTest.log')
shutil.copy2(SOURCE / 'build-acceptance/cuda/CTestTestfile.cmake', DEST / 'capture/CTestTestfile.cmake')
with (DEST / 'capture/source-build-inputs.tar.gz').open('wb') as f:
    subprocess.run(['git', 'archive', '--format=tar.gz', original['revision'], '--',
        'CMakeLists.txt', 'cmake', 'include', 'src', 'cuda', 'tests', 'tools', 'fixtures',
        'docs/POSTPROCESS_POLICY.md', 'scripts/run_postprocess_acceptance.sh',
        'scripts/check_postprocess_mutations.sh'], cwd=SOURCE, stdout=f, check=True)
print(json.dumps(summary, indent=2))
