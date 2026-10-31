#!/usr/bin/env python3
"""External custody wrapper; the tested checkout and its runners are unchanged."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

STAGE = Path(__file__).resolve().parent
SOURCE = STAGE / 'source'
CUSTODY = STAGE / 'custody'
REVISION = '995bca50d41510611b3990c9f8a020a9d8b76074'
ENV = dict(os.environ, SACCADE_F04_F03_DUMPS=str(STAGE / 'inputs'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now().astimezone().isoformat()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=SOURCE).decode()


def manifest(paths, base):
    return {str(p.relative_to(base)): digest(p) for p in sorted(paths) if p.is_file()}


def save(name, data):
    (CUSTODY / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def source_state():
    return {'revision': git('rev-parse', 'HEAD').strip(),
            'tree': git('rev-parse', 'HEAD^{tree}').strip(),
            'status': git('status', '--porcelain=v1', '--untracked-files=all'),
            'tracked_files': manifest([SOURCE / p for p in git('ls-files', '-z').split('\0') if p], SOURCE)}


def guard():
    assert source_state() == INITIAL, 'tested source changed'
    assert manifest((STAGE / 'inputs').glob('*.f32'), STAGE) == INPUTS, 'replay inputs changed'
    assert {p: digest(Path(p)) for p in TOOLS} == TOOLS, 'compiler/sanitizer tool changed'


def run(name, command):
    guard()
    started = now()
    print(name + ': started', flush=True)
    with (CUSTODY / (name + '.log')).open('w') as log:
        proc = subprocess.Popen(command, cwd=SOURCE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            with (CUSTODY / 'resource-samples.jsonl').open('a') as sample:
                mem = Path('/proc/meminfo').read_text().splitlines()
                cg = Path('/proc/self/cgroup').read_text()
                row = {'time': now(), 'stage': name, 'pid': proc.pid,
                       'mem_available': next(x for x in mem if x.startswith('MemAvailable:')),
                       'cgroup': cg.strip()}
                for line in cg.splitlines():
                    if line.startswith('0::'):
                        path = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
                        for key in ['memory.current', 'memory.peak', 'memory.max', 'memory.events']:
                            if (path / key).exists():
                                row[key] = (path / key).read_text().strip()
                sample.write(json.dumps(row) + '\n')
            time.sleep(2)
        status = proc.returncode
    save(name + '.result.json', {'command': command, 'cwd': str(SOURCE),
         'SACCADE_F04_F03_DUMPS': ENV['SACCADE_F04_F03_DUMPS'],
         'started': started, 'finished': now(), 'exit': status})
    print(name + ': exit=' + str(status), flush=True)
    guard()
    assert status == 0, name + ' failed; preserve attempt'


INITIAL = source_state()
assert INITIAL['revision'] == REVISION and INITIAL['status'] == ''
INPUTS = manifest((STAGE / 'inputs').glob('*.f32'), STAGE)
TOOLS = {}
for name in ['/opt/cuda/bin/nvcc', '/opt/cuda/bin/compute-sanitizer', 'c++', 'cmake', 'ctest', 'bash', 'as', 'ld']:
    p = Path(shutil.which(name)).resolve()
    TOOLS[str(p)] = digest(p)
save('source-before.json', INITIAL)
save('input-hashes-before.json', INPUTS)
save('tool-executable-hashes.json', TOOLS)
save('wrapper-identity.json', {'sha256': digest(Path(__file__)), 'started': now(),
     'resource_scope': 'systemd user scope MemoryMax=12G MemorySwapMax=0',
     'source': str(SOURCE), 'source_revision': REVISION})
with (CUSTODY / 'source-995bca5.tar.gz').open('wb') as f:
    subprocess.run(['git', 'archive', '--format=tar.gz', REVISION], cwd=SOURCE, stdout=f, check=True)
try:
    run('formal-acceptance', ['bash', 'scripts/run_postprocess_acceptance.sh', 'build-acceptance'])
    build = SOURCE / 'build-acceptance/cuda'
    binaries = [build / n for n in ['postprocess_tests', 'postprocess_cuda_tests',
                'postprocess_sanitizer_control', 'libsaccade_postprocess.a', 'libsaccade_postprocess_cuda.a',
                'libsaccade_sha256.a', 'libsaccade_file_io.a']]
    first = manifest(binaries, SOURCE)
    save('binary-library-hashes-after-acceptance.json', first)
    # Require the suite/control bytes to match the hashes captured before GPU work.
    for line in (SOURCE / 'build-acceptance/capture/binaries.log').read_text().splitlines():
        expected, file = line.split(maxsplit=1)
        assert digest(Path(file)) == expected, 'binary changed during formal acceptance'
    run('binary-runtime-links', ['ldd', str(build / 'postprocess_cuda_tests')])
    shutil.copy2(build / 'CMakeCache.txt', CUSTODY / 'CMakeCache.txt')
    run('semantic-mutations', ['bash', 'scripts/check_postprocess_mutations.sh',
                             str(build), 'build-formal-mutations'])
    assert manifest(binaries, SOURCE) == first, 'formal binaries changed during mutation controls'
    # Rebuild from unchanged source bytes and require identical executable/library bytes.
    for name in ['cuda/postprocess.cu', 'src/postprocess.cpp', 'tests/postprocess_cuda_tests.cpp',
                 'tests/postprocess_tests.cpp', 'tests/postprocess_sanitizer_control.cu']:
        path = SOURCE / name
        if path.exists():
            path.touch()
    run('unchanged-source-rebuild', ['cmake', '--build', str(build), '--parallel', '3', '--target',
        'postprocess_tests', 'postprocess_cuda_tests', 'postprocess_sanitizer_control'])
    final = manifest(binaries, SOURCE)
    save('binary-library-hashes-after-rebuild.json', final)
    assert final == first, 'rebuilt binaries/libraries differ'
    save('source-after.json', source_state())
    save('input-hashes-after.json', manifest((STAGE / 'inputs').glob('*.f32'), STAGE))
    guard()
    save('formal-status.json', {'exit': 0, 'finished': now(), 'source_clean_unchanged': True,
         'binary_checks': 'pre-GPU test/control hashes match final bytes; all seven executables/libraries unchanged across mutations and rebuild',
         'tool_hashes_unchanged': True, 'replay_bytes_unchanged': True})
except BaseException as e:
    save('formal-status.json', {'exit': 1, 'finished': now(), 'error': str(e)})
    raise
