#!/usr/bin/env python3
"""Separate Initcheck/NVTX diagnostic. prepare never runs a GPU executable."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SAN = '/opt/cuda/bin/compute-sanitizer'
MODES = [('none-yes', 'none', 'yes'), ('rgb-yes', 'rgb', 'yes'),
         ('alpha-yes', 'alpha', 'yes'), ('caller-yes', 'caller', 'yes'),
         ('rgb-no', 'rgb', 'no')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def state():
    return {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'worktree': subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)}


def pinned_paths(out):
    return [out / 'annotation.cu', out / 'nvtx_mark.hpp', out / 'capture.py', out / 'annotation',
            *sorted((out / 'inputs').glob('*')),
            Path('/opt/cuda/lib64/libnvjpeg.so').resolve(),
            Path('/usr/lib/wsl/lib/libcuda.so.1').resolve(),
            Path('/opt/cuda/bin/nvcc').resolve(), Path(SAN).resolve()]


def run(out, name, argv, *, output=None, timeout=60):
    cap = out / 'capture'
    target = cap / (name + '.command.json')
    if target.exists():
        raise RuntimeError(f'refusing to overwrite {name}')
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:')))
    if available < 4194304:
        raise RuntimeError('need at least 4 GiB MemAvailable')
    dump(target, {'argv': argv, 'cwd': str(REPO), 'timeout_seconds': timeout,
                  'time': datetime.datetime.now().astimezone().isoformat(),
                  'mem_available_kib': available,
                  'environment': {k: v for k, v in os.environ.items()
                                  if k.startswith(('CUDA_', 'CUPTI_', 'NVTX_', 'LD_'))}})
    stdout = Path(output) if output else cap / (name + '.stdout')
    with stdout.open('wb') as so, (cap / (name + '.stderr')).open('wb') as se:
        p = subprocess.run(['/usr/bin/time', '-v', '-o', str(cap / (name + '.time')),
                            'timeout', '--signal=TERM', '--kill-after=5s', str(timeout), *argv],
                           cwd=REPO, stdout=so, stderr=se)
    (cap / (name + '.exit')).write_text(str(p.returncode) + '\n')
    print(f'{name}: exit {p.returncode}', flush=True)
    return p.returncode


def planar(ppm):
    data = Path(ppm).read_bytes()
    m = re.match(rb'P6\s+35\s+27\s+255\n', data)
    if not m or len(data) - m.end() != 35 * 27 * 3:
        raise RuntimeError('unexpected independent djpeg PPM output')
    rgb = data[m.end():]
    return rgb[::3] + rgb[1::3] + rgb[2::3]


def prepare(out):
    if out.exists():
        raise RuntimeError('refusing to overwrite preparation directory')
    out.mkdir(parents=True)
    (out / 'capture').mkdir()
    inputs = out / 'inputs'
    inputs.mkdir()
    sources = [HERE / 'annotation.cu', HERE.parent / 'contracts/nvtx_mark.hpp', HERE / 'capture.py']
    dump(out / 'preparation-start.json', {'sources': {str(p): sha(p) for p in sources},
                                         'state': state(), 'gpu_execution': False})
    for p in sources:
        shutil.copy2(p, out / p.name)
    shutil.copy2(REPO / 'fixtures/preprocess/baseline.jpg', inputs / 'A.jpg')
    rgb = bytes(value for y in range(27) for x in range(35)
                for value in ((19*x + 7*y + 11) % 256,
                              (3*x + 23*y + 67) % 256,
                              (x*x + 13*y + 191) % 256))
    (inputs / 'B-source.ppm').write_bytes(b'P6\n35 27\n255\n' + rgb)
    if run(out, 'cjpeg-B', ['/usr/bin/cjpeg', '-baseline', '-quality', '91', '-dct', 'int',
                           '-sample', '2x2,1x1,1x1', str(inputs / 'B-source.ppm')],
           output=inputs / 'B.jpg'):
        raise RuntimeError('offline JPEG generation failed')
    for letter in 'AB':
        if run(out, 'djpeg-' + letter, ['/usr/bin/djpeg', '-dct', 'int', '-nosmooth', '-rgb',
                                      str(inputs / (letter + '.jpg'))],
               output=inputs / (letter + '.ppm')):
            raise RuntimeError('independent reference decode failed')
        (inputs / (letter + '.rgb')).write_bytes(planar(inputs / (letter + '.ppm')))
    frozen = REPO / 'evidence/preprocess/2026-10-31-f01-investigation/baseline.rgb'
    if sha(inputs / 'A.rgb') != sha(frozen):
        raise RuntimeError('baseline independent reference no longer matches original fixed bytes')
    if sha(inputs / 'A.rgb') == sha(inputs / 'B.rgb'):
        raise RuntimeError('references must be distinct')
    command = ['/opt/cuda/bin/nvcc', '-std=c++17', '-O2', '-lineinfo', '--allow-unsupported-compiler',
               '-arch=sm_120', str(out / 'annotation.cu'), '-I' + str(out),
               '-L/opt/cuda/lib64', '-lnvjpeg', '-L/usr/lib/wsl/lib', '-lcuda', '-ldl',
               '-o', str(out / 'annotation')]
    if run(out, 'compile', command):
        raise RuntimeError('compile failed; attempt retained')
    for name, argv in [('nvcc-version', ['/opt/cuda/bin/nvcc', '--version']),
                       ('sanitizer-version', [SAN, '--version']),
                       ('cjpeg-version', ['/usr/bin/cjpeg', '-version']),
                       ('djpeg-version', ['/usr/bin/djpeg', '-version']),
                       ('ldd', ['ldd', str(out / 'annotation')])]:
        if run(out, name, argv):
            raise RuntimeError(name + ' failed')
    paths = pinned_paths(out)
    dump(out / 'prepared-identity.json', {'files': {str(p): sha(p) for p in paths},
                                         'source_parent': str(REPO / 'evidence/preprocess/2026-10-31-f01-investigation/repro.cu'),
                                         'source_parent_sha256': sha(REPO / 'evidence/preprocess/2026-10-31-f01-investigation/repro.cu'),
                                         'cjpeg_sha256': sha('/usr/bin/cjpeg'),
                                         'djpeg_sha256': sha('/usr/bin/djpeg'),
                                         'gpu_execution': False,
                                         'notes': 'Fixed before GPU; A,B,A same dimensions; independent djpeg nosmooth RGB references.'})


def analyze(out, label, mode, nvtx, rc):
    path = out / label
    stdout = (out / 'capture' / (label + '.stdout')).read_text(errors='replace')
    stderr = (out / 'capture' / (label + '.stderr')).read_text(errors='replace')
    tool = (path / 'tool.log').read_text(errors='replace') if (path / 'tool.log').exists() else ''
    summaries = re.findall(r'^========= ERROR SUMMARY: (\d+) errors$', tool, re.M)
    headers = len(re.findall(r'^========= Uninitialized ', tool, re.M))
    reads = re.findall(r'Uninitialized __global__ memory read of size (\d+) bytes\n'
                       r'=+\s+at .*?rgba2rgb<\(nvjpegOutputFormat_t\)3,.*?\+0x170\n'
                       r'=+\s+by thread \((\d+),(\d+),(\d+)\) in block \((\d+),(\d+),(\d+)\)\n'
                       r'=+\s+Address (0x[0-9a-fA-F]+)', tool)
    outputs = re.findall(r'^output repeat=(\d+) input=([AB]) rgb_exact=(\d+) differences=(\d+)$', stdout, re.M)
    byte_checks = []
    for repeat, letter in enumerate('ABA'):
        actual, reference = path / f'repeat-{repeat}.rgb', out / 'inputs' / (letter + '.rgb')
        byte_checks.append({'repeat': repeat, 'input': letter,
                            'present': actual.is_file(),
                            'exact': actual.is_file() and actual.read_bytes() == reference.read_bytes(),
                            'sha256': sha(actual) if actual.is_file() else None})
    complete = 'result=complete warmup_fail_preserved=1' in stdout
    loaded = {}
    maps = path / 'loaded.maps'
    if maps.is_file():
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith('/') and any(
                    part in fields[5] for part in ('libnvjpeg', 'libcuda.so')):
                module = Path(fields[5])
                loaded[str(module)] = sha(module)
    expected_nvjpeg = sha(Path('/opt/cuda/lib64/libnvjpeg.so').resolve())
    expected_driver = sha(Path('/usr/lib/wsl/lib/libcuda.so.1').resolve())
    module_identity = (any('libnvjpeg' in p and h == expected_nvjpeg for p, h in loaded.items())
                       and any('libcuda.so' in p and h == expected_driver for p, h in loaded.items())
                       and all(h == (expected_nvjpeg if 'libnvjpeg' in p else expected_driver)
                               for p, h in loaded.items()))
    result = {'mode': mode, 'nvtx': nvtx, 'exit': rc,
              'application_complete': complete,
              'application_stdout_outputs': outputs,
              'application_failed': 'result=failed' in stderr,
              'loaded_modules': loaded, 'mapped_library_identity_matches_pins': module_identity,
              'summary_errors': int(summaries[0]) if len(summaries) == 1 else None,
              'uninitialized_headers': headers, 'complete_converter_records': len(reads),
              'all_printed_errors_classified_as_converter_3byte_reads': len(reads) == headers and all(r[0] == '3' for r in reads),
              'reference_byte_checks': byte_checks,
              'warmup_discovery_present': 'discovery current_process=1 complete_rgb_records=945 headers=945 valid_coverage=945 pixel_stride=4' in stdout,
              'postwarmup_aggregate_report_count': max(0, len(reads) - 945),
              'prediction_total_errors': 945 if mode == 'rgb' and nvtx == 'yes' else 2835,
              'acceptance': 'not assessed; original unsuppressed warmup FAIL retained',
              'added_sync': 'caller stream synchronized after each annotation and in none controls; private converter stream not identified here'}
    result['informative_diagnostic'] = (complete and rc == 99 and not result['application_failed']
                                       and result['warmup_discovery_present']
                                       and module_identity
                                       and len(outputs) == 3
                                       and all(x['exact'] for x in byte_checks)
                                       and result['summary_errors'] == headers == len(reads)
                                       and result['all_printed_errors_classified_as_converter_3byte_reads'])
    result['prediction_matched'] = result['summary_errors'] == result['prediction_total_errors']
    dump(path / 'analysis.json', result)
    return result


def execute(out, labels):
    if os.environ.get('LD_PRELOAD') or os.environ.get('CUDA_INJECTION64_PATH'):
        raise RuntimeError('refusing ambient preload/injection for separate sanitizer experiment')
    identity = json.loads((out / 'prepared-identity.json').read_text())
    before = {p: sha(p) for p in identity['files']}
    if before != identity['files']:
        raise RuntimeError('prepared identity drift')
    tag = '-'.join(labels)
    start_path, result_path = out / ('gpu-start-' + tag + '.json'), out / ('gpu-result-' + tag + '.json')
    if start_path.exists() or result_path.exists():
        raise RuntimeError('refusing to overwrite invocation record')
    dump(start_path, {'files': before, 'state': state(), 'labels': labels,
                                  'executing_runner': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                                  'time': datetime.datetime.now().astimezone().isoformat()})
    results = []
    error = None
    try:
        if not (out / 'capture/device.command.json').exists():
            run(out, 'device', ['nvidia-smi'])
        for label, mode, nvtx in MODES:
            if label not in labels:
                continue
            target = out / label
            target.mkdir()
            argv = [SAN, '--tool', 'initcheck', '--error-exitcode', '99', '--print-limit', '0',
                    '--show-backtrace', 'yes', '--nvtx', nvtx, '--log-file', str(target / 'tool.log'),
                    str(out / 'annotation'), mode, str(target / 'tool.log'),
                    str(out / 'inputs/A.jpg'), str(out / 'inputs/A.rgb'),
                    str(out / 'inputs/B.jpg'), str(out / 'inputs/B.rgb'), '35x27', str(target)]
            rc = run(out, label, argv)
            result = analyze(out, label, mode, nvtx, rc)
            results.append(result)
            if not result['informative_diagnostic']:
                raise RuntimeError(label + ': uninformative/failed capture retained; no dependent intervention')
    except BaseException as e:
        error = str(e)
        raise
    finally:
        after = {p: sha(p) for p in before}
        dump(result_path, {'results_this_invocation': results,
                                       'failure': error, 'files_after': after,
                                       'identity_unchanged': before == after,
                                       'state': state(), 'acceptance': 'not changed'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phase', choices=['prepare', 'run'], required=True)
    ap.add_argument('--labels', nargs='+', choices=[x[0] for x in MODES], default=[x[0] for x in MODES])
    args = ap.parse_args()
    out = args.out.resolve()
    if args.phase == 'prepare':
        prepare(out)
    else:
        execute(out, args.labels)


if __name__ == '__main__':
    main()
