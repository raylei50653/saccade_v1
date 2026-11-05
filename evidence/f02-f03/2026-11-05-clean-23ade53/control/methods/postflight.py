import hashlib
import json
import re
import shlex
import subprocess
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
control = root / 'build/closeout-control'
cap = control / 'capture'
acceptance = root / 'build/clean-acceptance/capture'
evaluation = root / 'build/clean-evaluation/capture'

def require(ok, message):
    if not ok:
        raise SystemExit('REFUSED: ' + message)

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def manifest(path, cwd=root):
    rows = []
    for line in path.read_text().splitlines():
        expected, name = line.split('  ', 1)
        p = Path(name)
        if not p.is_absolute():
            p = cwd / p
        require(p.is_file() and digest(p) == expected, f'hash mismatch: {p}')
        rows.append((name, expected))
    require(rows, f'empty manifest: {path}')
    return rows

require((cap/'status.txt').read_text().startswith('exit=0\n'), 'campaign did not exit zero')
require((acceptance/'acceptance.txt').read_text().strip() == 'PASS', 'session gate not PASS')
require((evaluation/'gates.txt').read_text().strip() == 'PASS', 'evaluation gate not PASS')
for folder in (cap, acceptance, evaluation):
    for path in folder.glob('*.exit'):
        expected = 99 if path.name.startswith('control-') else 0
        require(path.read_text().strip() == str(expected), f'{path.name}: expected exit {expected}')
controls = {
    'memcheck': 'Invalid __global__ write',
    'initcheck': 'Uninitialized __global__ memory read',
    'racecheck': 'Error: Race reported between',
    'synccheck': 'Barrier error detected',
}
for tool, diagnostic in controls.items():
    require((acceptance/f'control-{tool}.exit').read_text().strip() == '99', f'{tool} control did not exit 99')
    require(diagnostic in (acceptance/f'control-{tool}.log').read_text(), f'{tool} control diagnostic missing')
for name, count in [('cpu-ctest', 16), ('ctest', 59)]:
    text = (acceptance/f'{name}.log').read_text()
    require(re.search(rf'^100% tests passed(?:, 0 tests failed)? out of {count}$', text, re.MULTILINE), f'{name} summary mismatch')
sanitizers = []
for tool in controls:
    for test in ('resize_cuda_tests', 'selective_scan_cuda_tests'):
        name = f'{test}-{tool}'
        require((cap/f'{name}.exit').read_text().strip() == '0', f'{name}: nonzero exit')
        text = (cap/f'{name}.log').read_text()
        expected = ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                    if tool == 'racecheck' else '========= ERROR SUMMARY: 0 errors')
        require(text.splitlines().count(expected) == 1, f'{name}: summary mismatch')
        require(re.search(r'^resize_cuda_tests PASS cases=72 ' if test.startswith('resize')
                          else rf'^selective_scan_cuda_tests: {360 if tool == "initcheck" else 368} checks passed$', text, re.MULTILINE), f'{name}: suite result missing')
        command = (cap/f'{name}.command').read_text()
        require('--kernel-name' not in command and '--launch-count' not in command, f'{name}: filtered')
        require(('--no-held-stream' in command) == (tool == 'initcheck'), f'{name}: held-stream coverage mismatch')
        if test == 'selective_scan_cuda_tests':
            require('build_dirty=false' in text, f'{name}: binary dirty identity')
        else:
            require(f'held_stream={"UNEXECUTED" if tool == "initcheck" else "PASS"}' in text,
                    f'{name}: held-stream result missing')
        sanitizers.append({'suite': test, 'tool': tool, 'exit': 0, 'kernel_filter': False})
require('26 checks passed' in (cap/'hr-python-reference.log').read_text()
        or 'checks=26' in (cap/'hr-python-reference.log').read_text(), 'HR reference summary missing')
require(len(re.findall(r'^case=\S+ RGB/FP32 bitwise PASS$', (cap/'hr-python-reference.log').read_text(), re.MULTILINE)) == 4,
        'expected four HR RGB/FP32 bitwise cases')
require((cap/'source-status-before.txt').read_bytes() == b'', 'initial dirty source')
require((cap/'source-status-after.txt').read_bytes() == b'', 'final dirty source')
require(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root) == b'', 'current dirty source')
for name in ('source-files.sha256', 'control.sha256', 'hr-reference-before.sha256', 'accepted-binaries.sha256'):
    manifest(cap/name)
refs = Path('/home/ray/.cache/saccade-v1/hr-deploy/refs')
reference_manifest = json.loads((refs/'manifest.json').read_text())
for name, expected in reference_manifest['hashes'].items():
    require(digest(refs/name) == expected, f'HR frozen manifest mismatch: {name}')
compile_commands = json.loads((root/'build/clean-acceptance/cuda/compile_commands.json').read_text())
for source in ('cuda/resize.cu', 'cuda/selective_scan.cu'):
    rows = [row for row in compile_commands if Path(row['file']) == root/source]
    require(len(rows) == 1, f'compile identity missing: {source}')
    args = shlex.split(rows[0]['command'])
    require('-DSACCADE_BUILD_DIRTY="false"' in args, f'dirty compile: {source}')
    require('-DSACCADE_BUILD_COMMIT="'+(cap/'revision.txt').read_text().strip()+'"' in args, f'wrong compile revision: {source}')
binaries = manifest(acceptance/'binaries.sha256')
require((acceptance/'binaries.sha256').read_bytes() == (acceptance/'binaries-after.sha256').read_bytes(), 'binary drift')
require((acceptance/'artifacts.sha256').read_bytes() == (acceptance/'artifacts-after.sha256').read_bytes(), 'input drift')
require((evaluation/'artifacts.sha256').read_bytes() == (evaluation/'artifacts-after.sha256').read_bytes(), 'evaluation input drift')
with tarfile.open(control/'accepted-binaries.tar.gz') as archive:
    for name, expected in binaries:
        member = archive.extractfile('./'+Path(name).name)
        require(member is not None and hashlib.file_digest(member, 'sha256').hexdigest() == expected,
                f'archived binary mismatch: {name}')
m4 = root/'evidence/m4/2026-11-03-formal-cad5298/evaluation/outputs.sha256'
outputs = manifest(m4, root/'build/clean-evaluation')
require(len(outputs) == 21, f'M4 output count {len(outputs)}')
equivalence = json.loads((cap/'component-equivalence.json').read_text())
for row in equivalence['files']:
    require(digest(root/row['path']) == row['sha256'], 'accepted component drift')
summary = {
    'status': 'PASS',
    'formal_acceptance': True,
    'source_dirty': False,
    'revision': (cap/'revision.txt').read_text().strip(),
    'motion_profile': 'v1-motion/1',
    'cpu_tests': 16,
    'cuda_tests': 59,
    'm4_identical_outputs': len(outputs),
    'sanitizer_positive_controls': {tool: {'exit': 99, 'diagnostic_matched': True} for tool in controls},
    'supplemental_component_sanitizers': sanitizers,
    'current_component_equivalence': equivalence,
    'g1_session_memcheck': 'OPEN; this certificate does not close v1-motion/2',
    'performance_abba': 'UNEXECUTED; original dirty-source measurements retain their original identity',
    'semantic_mutations': 'UNEXECUTED in this capture; not required by the unchanged v1-motion/1 gate',
    'full_hr_session_racecheck': 'UNEXECUTED; standalone component checks and scoped session checks only',
}
(cap/'closeout.json').write_text(json.dumps(summary, indent=2)+'\n')
print('PASS: strict control exits, 8 component sanitizers, source/binary custody, M4 21/21 outputs')
