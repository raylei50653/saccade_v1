#!/usr/bin/env python3
"""Build/run a bounded, GPU-locked diagnostic matrix, preserving exact commands."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CUDA = Path('/opt/cuda')
LOCK = Path('/home/ray/.cache/saccade-v1/gpu.lock')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = {'root': str(ROOT), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'runs': []}
    compile_cmd = ['g++', '-std=c++20', '-O0', '-g', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
                   '-I' + str(ROOT / 'third_party/nvof'), '-I' + str(CUDA / 'include'),
                   str(HERE / 'nvof_api_probe.cpp'), '-L' + str(CUDA / 'lib64'),
                   '-Wl,-rpath,' + str(CUDA / 'lib64'), '-L/usr/lib/wsl/lib',
                   '-Wl,-rpath,/usr/lib/wsl/lib', '-lcuda', '-lcupti', '-ldl', '-pthread',
                   '-o', str(HERE / 'nvof_api_probe')]
    with (HERE / 'compile.log').open('x') as output:
        compiled = subprocess.run(compile_cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    manifest['compile'] = {'command': compile_cmd, 'exit': compiled.returncode}
    if compiled.returncode:
        (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        raise SystemExit(compiled.returncode)
    for name in ('nvof_api_probe.cpp', 'array_trace.hpp', 'nvof_api_probe', 'run_probe.py'):
        manifest.setdefault('sha256', {})[str(HERE / name)] = digest(HERE / name)
    for path in (ROOT / 'third_party/nvof/nvOpticalFlowCuda.h', ROOT / 'third_party/nvof/nvOpticalFlowCommon.h',
                 Path('/usr/lib/wsl/lib/libnvidia-opticalflow.so.1'), Path('/usr/lib/wsl/lib/libnvcuvid.so.1'),
                 ROOT / 'build/development/track_sequence', ROOT / 'build/development/ofa_audit'):
        manifest['sha256'][str(path)] = digest(path)
    for name, command in (
        ('git-head', ['git', 'rev-parse', 'HEAD']),
        ('git-status', ['git', 'status', '--porcelain=v1']),
        ('gpu', ['nvidia-smi']),
        ('sanitizer-version', [str(CUDA / 'bin/compute-sanitizer'), '--version']),
        ('os', ['uname', '-a']),
    ):
        output = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        manifest[name] = {'command': command, 'exit': output.returncode, 'stdout': output.stdout, 'stderr': output.stderr}

    def run(name, arguments, sanitizer=True, executable=None):
        command = [str(executable or HERE / 'nvof_api_probe'), *arguments]
        if sanitizer:
            command = [str(CUDA / 'bin/compute-sanitizer'), '--tool', 'memcheck', '--report-api-errors', 'explicit',
                       '--print-limit', '0', '--error-exitcode', '99', *command]
        command = ['flock', str(LOCK), 'timeout', '-k', '10', '300', *command]
        started = time.monotonic()
        with (HERE / (name + '.log')).open('x') as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        item = {'name': name, 'command': command, 'exit': result.returncode, 'wall_seconds': time.monotonic() - started,
                'log': str(HERE / (name + '.log')), 'log_sha256': digest(HERE / (name + '.log'))}
        manifest['runs'].append(item)
        (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(name, 'exit=' + str(result.returncode), 'wall_s=' + str(round(item['wall_seconds'], 2)), flush=True)

    for name, args in (
        ('probe-init-only', ['--init-only']),
        ('probe-allocate-only', ['--executes', '0']),
        ('probe-execute-1', ['--executes', '1']),
        ('probe-execute-2', ['--executes', '2']),
        ('probe-execute-30', ['--executes', '30']),
        ('probe-sessions-2', ['--sessions', '2', '--executes', '2']),
        ('probe-default-stream', ['--default-stream', '--executes', '2']),
    ):
        run(name, args)
    run('probe-cupti', ['--executes', '2', '--cupti'], sanitizer=False)
    engine = '/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'
    sequence = str(ROOT / 'build/nvof-memcheck/prefix/MOT17-10-SDP')
    for name, sanitizer in (('hr-plain', False), ('hr-memcheck', True)):
        run(name, [engine, sequence, str(HERE / (name + '.txt')), '--profile', 'hr-1280x768',
                   '--packets', str(HERE / (name + '.packets.bin')), '--motion', str(HERE / (name + '.scmo'))],
            sanitizer=sanitizer, executable=ROOT / 'build/development/track_sequence')
    manifest['hr_output_comparison'] = {}
    for extension in ('.txt', '.packets.bin', '.scmo'):
        plain, checked = HERE / ('hr-plain' + extension), HERE / ('hr-memcheck' + extension)
        manifest['hr_output_comparison'][extension] = {'plain_sha256': digest(plain), 'memcheck_sha256': digest(checked),
                                                     'identical_bytes': plain.read_bytes() == checked.read_bytes(),
                                                     'size': plain.stat().st_size}
    manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
