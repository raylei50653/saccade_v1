#!/usr/bin/env python3
"""Capture HR lifecycle racecheck under explicit tool synchronization.

This evidence-only runner uses the preserved clean runtime and separately built
HR harness. The default checks all CUDA kernels. The explicit custom-only mode
selects the deployed saccade and CUB namespaces. Neither grants native acceptance.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
from pathlib import Path
import re
import signal
import subprocess
import sys


def discover_repo_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts/run_hr_sanitizer_validation.py").is_file() and (candidate / "CMakeLists.txt").is_file():
            return candidate
    raise RuntimeError("cannot discover repository ancestor for evidence rerun")


REPO = discover_repo_root()
RUNTIME_REVISION = 'a00ab3a6abd7cc2c1f823fd3fbac66a345f3ce48'
ENGINE_SHA256 = '8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876'
CAPTURE_RUNNER = REPO / 'scripts/run_hr_sanitizer_validation.py'
CUSTOM_REGEX = r'_ZN7saccade|_ZN3cub'
CONTROL_REGEX = CUSTOM_REGEX + r'|^_Z11shared_racePf$'
spec = importlib.util.spec_from_file_location('hr_validation', CAPTURE_RUNNER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def complete(text):
    return (len(re.findall(r'^tracking_session_tests: 180 checks passed$', text, re.M)) == 1
            and 'tracking_session_tests profile=hr-1280x768 mode=lifecycle\n' in text)


def success_status(custom_kernels_only):
    return 'PASS_FILTERED_FORCED_SYNC' if custom_kernels_only else 'PASS_FORCED_SYNC'


def classify_lifecycle(record, text, control, custom_kernels_only):
    done = complete(text)
    clean = m.clean_sanitizer_log('racecheck', text)
    incomplete = m.incomplete(record, text)
    okay = (record['returncode'] == 0 and record.get('target', {}).get('returncode') == 0
            and done and clean and control and not incomplete)
    status = success_status(custom_kernels_only) if okay else 'INCOMPLETE' if incomplete else 'OPEN'
    return {'name': 'racecheck-lifecycle', 'status': status, 'target_completed': done,
            'zero_summary': m.summary_ok('racecheck', text), 'clean_sanitizer_log': clean,
            'positive_control_passed': control, 'native_schedule_acceptance': False,
            'full_kernel_coverage': not custom_kernels_only,
            'kernel_filter': CUSTOM_REGEX if custom_kernels_only else None,
            'execution': record}


def aggregate_status(results, custody_unchanged, error, custom_kernels_only):
    expected = [('baseline-lifecycle', 'PASS'), ('control-racecheck', 'PASS'),
                ('racecheck-lifecycle', success_status(custom_kernels_only))]
    actual = [(row.get('name'), row.get('status')) for row in results]
    return success_status(custom_kernels_only) if custody_unchanged and not error and actual == expected else 'OPEN'


def sanitizer_commands(args, sanitizer, build, command):
    options = [str(sanitizer), '--tool', 'racecheck', '--error-exitcode', '99',
               '--print-limit', '0', '--racecheck-num-workers', str(args.racecheck_num_workers),
               '--force-synchronization-limit', str(args.force_synchronization_limit)]
    control_filter = ['--kernel-name', f'regex={CONTROL_REGEX}'] if args.custom_kernels_only else []
    session_filter = ['--kernel-name', f'regex={CUSTOM_REGEX}'] if args.custom_kernels_only else []
    return (options, [*options, *control_filter, str(build / 'postprocess_sanitizer_control'), 'racecheck'],
            [*options, *session_filter, *command])


def sdk_paths(binary):
    listing = subprocess.check_output(['ldd', str(binary)], text=True)
    if 'not found' in listing:
        raise RuntimeError('lifecycle harness has unresolved dynamic libraries')
    libraries = {Path(match).resolve() for match in re.findall(r'=> (/\S+)', listing)}
    # Optical flow and sanitizer libraries are loaded dynamically, outside ldd.
    libraries.add(Path('/usr/lib/wsl/lib/libnvidia-opticalflow.so.1').resolve())
    libraries.update(path.resolve() for path in Path('/opt/cuda/compute-sanitizer').glob('*.so'))
    if not all(path.is_file() for path in libraries):
        raise RuntimeError('a required SDK or injection library is missing')
    return sorted(libraries)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--memory', default='8G')
    parser.add_argument('--force-synchronization-limit', type=int, default=8)
    parser.add_argument('--racecheck-num-workers', type=int, default=1,
                        help='CPU worker count; 0 uses tool automatic selection')
    parser.add_argument('--custom-kernels-only', action='store_true',
                        help='Scope lifecycle racecheck to saccade/CUB namespaces; successful captures are PASS_FILTERED_FORCED_SYNC')
    args = parser.parse_args(argv)
    if args.timeout < 1 or args.force_synchronization_limit < 1:
        parser.error('timeout and force-synchronization-limit must be positive')
    if not re.fullmatch(r'[1-9]\d*(?:[KMGT](?:i?B)?)?', args.memory):
        parser.error('memory must be a positive systemd byte size such as 8G')
    if args.racecheck_num_workers < 0:
        parser.error('racecheck-num-workers must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.source = REPO / 'build/hr-sanitizer-20261105-source'
    args.build = REPO / 'build/hr-sanitizer-20261105/cuda'
    args.memory_high = args.memory_max = args.memory
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError('output must be a fresh directory')
    output.mkdir(parents=True)
    capture = m.Capture(args, output)
    engine = Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine')
    sequence = Path('/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP')
    records = Path('/home/ray/developer/ai/saccade_v1-worktrees/hr-records-v2')
    harness = REPO / 'build/hr-sanitizer-20261105/harness'
    binary = harness / 'tracking_session_hr_tests'
    sanitizer = Path('/opt/cuda/bin/compute-sanitizer')
    command = [str(binary), str(engine), str(sequence.parent), str(records),
               'MOT17-10-SDP', '--profile', 'hr-1280x768', '--lifecycle']
    options, control_command, session_command = sanitizer_commands(args, sanitizer, args.build, command)
    passed_status = success_status(args.custom_kernels_only)
    results = []
    before = None
    libraries = []
    manifest = {
        'schema': 'hr-lifecycle-racecheck-forced-sync/1', 'started': m.now(),
        'status': 'RUNNING', 'formal_acceptance': False,
        'native_schedule_acceptance': False, 'forced_synchronization': True,
        'force_synchronization_limit': args.force_synchronization_limit,
        'racecheck_num_workers': args.racecheck_num_workers,
        'kernel_filter': CUSTOM_REGEX if args.custom_kernels_only else None,
        'positive_control_filter': CONTROL_REGEX if args.custom_kernels_only else None,
        'full_kernel_coverage': not args.custom_kernels_only,
        'custom_kernels_only': args.custom_kernels_only,
        'launch_limit': None, 'launch_skip': None,
        'hazard_caps_lowered': False, 'tools': ['racecheck'],
        'environment': capture.environment,
        'removed_environment': capture.removed_environment,
        'command': command, 'sanitizer_options': options,
        'control_command': control_command, 'session_command': session_command,
        'expected_checks': 180, 'source_revision': RUNTIME_REVISION,
        'resource_limits': {'timeout_seconds': args.timeout, 'memory_high': args.memory,
                            'memory_max': args.memory, 'memory_swap_max': 0},
        'scope': ('Clean a00ab3a runtime with separately compiled HR lifecycle harness. '
                  + ('Only saccade/CUB CUDA kernels; external library kernels are excluded. '
                     if args.custom_kernels_only else 'All CUDA kernels without a kernel filter. ')
                  + 'All 180 existing lifecycle checks under explicit '
                  'sanitizer stream synchronization, without launch limits. '
                  'OFA hardware itself is outside CUDA instrumentation. '
                  f'{passed_status} does not establish native-schedule acceptance.'),
    }
    m.save(output / 'manifest.json', manifest)

    def custody():
        files = [binary, harness / 'tracking_session_tests.cpp', harness / 'commands.json',
                 records / 'MOT17-10-SDP.scdr', records / 'MOT17-10-SDP.scmo',
                 Path(__file__), CAPTURE_RUNNER, capture.wrapper, sanitizer, *libraries]
        return {
            'runtime': m.source_snapshot(args.source, args.build, engine, sequence, 30),
            'harness_runners_and_libraries': {str(path): m.digest(path) for path in files},
        }

    lock_path = Path.home() / '.cache/saccade-v1/gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        print(f'Waiting for GPU lock: {lock_path}', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            libraries = sdk_paths(binary)
            before = custody()
            m.save(output / 'custody-before.json', before)
            if before['runtime']['revision'] != RUNTIME_REVISION:
                raise RuntimeError('unexpected clean runtime revision')
            if next(iter(before['runtime']['engine'].values())) != ENGINE_SHA256:
                raise RuntimeError('unexpected deployed engine')
            if f'CMAKE_HOME_DIRECTORY:INTERNAL={args.source}\n' not in (args.build / 'CMakeCache.txt').read_text():
                raise RuntimeError('build CMakeCache belongs to another source worktree')

            name = 'baseline-lifecycle'
            record = capture.run(name, command)
            text = (output / f'{name}.log').read_text(errors='replace')
            okay = (record['returncode'] == 0 and record.get('target', {}).get('returncode') == 0
                    and complete(text) and not m.incomplete(record, text))
            results.append({'name': name, 'status': 'PASS' if okay else 'FAIL',
                            'target_completed': complete(text), 'execution': record})
            m.save(output / 'results.json', results)
            if not okay:
                raise RuntimeError('HR lifecycle baseline failed')

            name = 'control-racecheck'
            record = capture.run(name, control_command)
            text = (output / f'{name}.log').read_text(errors='replace')
            control = (record['returncode'] == 99 and record.get('target', {}).get('returncode') == 99
                       and m.DIAGNOSTIC['racecheck'] in text and 'control=racecheck launch=' in text
                       and '========= COMPUTE-SANITIZER' in text
                       and not m.incomplete(record, text) and m.positive_summary_ok('racecheck', text)
                       and not m.INTERNAL.search(text.replace('Target application returned an error', '')))
            results.append({'name': name, 'status': 'PASS' if control else 'FAIL',
                            'expected_diagnostic': m.DIAGNOSTIC['racecheck'], 'execution': record})
            m.save(output / 'results.json', results)

            name = 'racecheck-lifecycle'
            record = capture.run(name, session_command)
            text = (output / f'{name}.log').read_text(errors='replace')
            result = classify_lifecycle(record, text, control, args.custom_kernels_only)
            results.append(result)
            m.save(output / 'results.json', results)
            print(f"{name}: {result['status']}; completed={result['target_completed']}", flush=True)
        except BaseException as error:
            manifest['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            capture.stop()
            after = None
            try:
                after = custody()
                m.save(output / 'custody-after.json', after)
            except Exception as error:
                manifest['custody_error'] = f'{type(error).__name__}: {error}'
            manifest['custody_unchanged'] = before is not None and before == after
            manifest['status'] = aggregate_status(results, manifest['custody_unchanged'],
                                                  manifest.get('error'), args.custom_kernels_only)
            manifest['finished'] = m.now()
            m.save(output / 'manifest.json', manifest)
    return 0 if manifest['status'] == passed_status else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        print(f'hr-lifecycle-racecheck: {error}', file=sys.stderr)
        sys.exit(2)
