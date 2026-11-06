#!/usr/bin/env python3
"""Freeze completed bounded-queue captures without promoting their scope.

Run only after all capture controllers have stopped:
  python3 -I build/hr-racecheck-20261106/control/archive_evidence.py --captures-finished
After REPORT.md has been written, refresh its artifact hash with:
  python3 -I build/hr-racecheck-20261106/control/archive_evidence.py --finalize-artifacts
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile


def repo_root():
    for path in Path(__file__).resolve().parents:
        if (path / 'scripts/run_hr_sanitizer_validation.py').is_file() and (path / 'CMakeLists.txt').is_file():
            return path
    raise RuntimeError('cannot discover repository ancestor')


ROOT = repo_root()
SOURCE = ROOT / 'build/hr-racecheck-20261106'
DESTINATION = ROOT / 'evidence/hr-sanitizer/2026-11-06-bounded-queue'
EXCLUDE = {'control', 'outputarchive', '__pycache__'}
ZERO = '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def elapsed(start, end):
    if not start or not end:
        return None
    try:
        return (dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return None


def tree_inventory(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if any(part in EXCLUDE for part in relative.parts):
            continue
        if path.is_symlink():
            result[str(relative)] = {'kind': 'symlink', 'target': os.readlink(path)}
        elif path.is_file():
            result[str(relative)] = {'kind': 'file', 'bytes': path.stat().st_size, 'sha256': digest(path)}
        elif path.is_dir():
            result[str(relative)] = {'kind': 'directory'}
    return result


def archive_directory(directory, destination):
    before = tree_inventory(directory)
    with tarfile.open(destination, 'w:gz', compresslevel=6, dereference=False) as archive:
        archive.add(directory, arcname=directory.name, recursive=False)
        for name in before:
            archive.add(directory / name, arcname=f'{directory.name}/{name}', recursive=False)
    verified = {}
    with tarfile.open(destination, 'r:gz') as archive:
        for member in archive.getmembers():
            if member.name == directory.name:
                continue
            relative = str(Path(member.name).relative_to(directory.name))
            expected = before[relative]
            if expected['kind'] == 'symlink':
                actual = {'kind': 'symlink', 'target': member.linkname}
            elif expected['kind'] == 'directory':
                actual = {'kind': 'directory'} if member.isdir() else {'kind': 'unexpected'}
            else:
                content = archive.extractfile(member)
                value = hashlib.sha256()
                count = 0
                for block in iter(lambda: content.read(1024 * 1024), b''):
                    count += len(block)
                    value.update(block)
                actual = {'kind': 'file', 'bytes': count, 'sha256': value.hexdigest()}
            if relative in verified or actual != expected:
                raise RuntimeError(f'archive member mismatch: {member.name}')
            verified[relative] = actual
    after = tree_inventory(directory)
    if verified != before or after != before:
        raise RuntimeError(f'archive inventory mismatch or source changed: {directory}')
    return {'archive': str(destination.relative_to(DESTINATION)), 'sha256': digest(destination),
            'bytes': destination.stat().st_size, 'verified_members': len(verified),
            'source_unchanged': True, 'members': before}


def telemetry_summary(path):
    peak, first, last = 0, None, None
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            first = first or sample.get('time')
            last = sample.get('time') or last
            peak = max(peak, integer(sample.get('memory.peak')),
                       integer(sample.get('systemd', {}).get('MemoryPeak')))
    return {'memory_peak_bytes': peak, 'first_sample': first, 'last_sample': last,
            'observed_interval_seconds': elapsed(first, last)}


def execution_summary(record, fallback_target=None, telemetry=None):
    record = record or {}
    target = record.get('target') or fallback_target or {}
    telemetry = telemetry or {}
    peaks = [integer(record.get('sampled_memory_peak_bytes')), integer(record.get('peak_bytes')),
             integer(target.get('memory.peak')), integer(telemetry.get('memory_peak_bytes'))]
    return {'returncode': record.get('returncode'), 'started': record.get('started'),
            'finished': record.get('finished'),
            'elapsed_seconds_including_lock_or_service_wait': elapsed(record.get('started'), record.get('finished')),
            'target_started': target.get('started'), 'target_finished': target.get('finished'),
            'target_elapsed_seconds': elapsed(target.get('started'), target.get('finished')),
            'memory_peak_bytes': max(peaks),
            'periodic_or_recorded_peak_bytes': record.get('sampled_memory_peak_bytes'),
            'final_target_peak_bytes': target.get('memory.peak'),
            'memory_events': record.get('sampled_memory_events'),
            'target_memory_events': target.get('memory.events'),
            'sampled_systemd_results': record.get('sampled_systemd_results'),
            'outer_timeout': record.get('outer_timeout', False), 'telemetry': telemetry}


def command_options(command):
    result = {'dump_kernel_launches': '--dump-kernel-launches' in command}
    for option in ('--force-synchronization-limit', '--racecheck-num-workers', '--kernel-name',
                   '--launch-count', '--launch-skip'):
        result[option[2:].replace('-', '_')] = command[command.index(option) + 1] if option in command else None
    return result


def execution_records(directory):
    patterns = ('*.command.json', '*.execution.json', 'command.json', 'execution.json',
                'results.json', 'target-started*.json', 'target-final*.json', '*.target.json',
                '*.telemetry.jsonl', 'telemetry.jsonl')
    return sorted({path.name for pattern in patterns for path in directory.glob(pattern)})


def interrupted_preflight(interruption_reason, records):
    return bool(interruption_reason) and not records


def ready_to_archive(manifest, interruption_reason, records):
    if interrupted_preflight(interruption_reason, records):
        return True
    return bool(manifest and (manifest.get('finished') or
                manifest.get('status') in ('FINISHED', 'OPEN', 'INCOMPLETE')))


def retention_category(manifest, interruption_reason, records):
    if interrupted_preflight(interruption_reason, records) or not manifest:
        return 'ABORTED_OR_UNSTARTED_PREFLIGHT'
    if manifest.get('error') or interruption_reason:
        return 'INTERRUPTED_ATTEMPT'
    return 'COMPLETED_CAPTURE_OR_PROBE'


def summarize(directory, archive):
    manifest = read_json(directory / 'manifest.json', {})
    interruption = read_json(directory / 'interruption-reason.json')
    is_probe = manifest.get('schema', '').startswith('racecheck-host-memory-probe') or directory.name.startswith('probe-')
    row = {'name': directory.name, 'archive': archive, 'kind': 'diagnostic_probe' if is_probe else 'session_or_lifecycle',
           'original_manifest_status': manifest.get('status'), 'original_manifest_scope': manifest.get('scope'),
           'started': manifest.get('started'), 'finished': manifest.get('finished'),
           'elapsed_seconds': elapsed(manifest.get('started'), manifest.get('finished')),
           'native_gate_closed': False, 'native_schedule_acceptance': False,
           'source_native_schedule_acceptance': manifest.get('native_schedule_acceptance'),
           'formal_acceptance': False, 'interruption_reason': interruption,
           'manifest_error': manifest.get('error'), 'custody_unchanged': manifest.get('custody_unchanged'),
           'results': []}
    for key in ('kernel_filter', 'full_kernel_coverage', 'force_synchronization_limit', 'forced_synchronization',
                'racecheck_num_workers', 'modes', 'frames', 'expected_checks', 'source_revision'):
        row[key] = manifest.get(key)
    results = read_json(directory / 'results.json', [])
    if isinstance(results, dict):
        results = [results]
    names = {result.get('name') for result in results}
    # An interrupted Capture.run can leave a command/log without an execution record.
    names.update(path.name.removesuffix('.command.json') for path in directory.glob('*.command.json'))
    by_name = {result['name']: result for result in results if result.get('name')}
    if is_probe:
        names.add('probe')
    for name in sorted(names):
        original = by_name.get(name, {})
        record_path = directory / ('execution.json' if name == 'probe' else f'{name}.execution.json')
        record = read_json(record_path, original.get('execution', {}))
        target_path = directory / ('target-final.json' if name == 'probe' else f'{name}.target.json')
        target = read_json(target_path, {})
        log_path = directory / ('run.log' if name == 'probe' else f'{name}.log')
        text = log_path.read_text(errors='replace') if log_path.exists() else ''
        command_path = directory / ('command.json' if name == 'probe' else f'{name}.command.json')
        command_data = read_json(command_path, {})
        command = command_data.get('target', [])
        if not isinstance(command, list):
            command = command_data.get('command', [])
        options = command_options(command)
        telemetry_path = directory / ('telemetry.jsonl' if name == 'probe' else f'{name}.telemetry.jsonl')
        summaries = re.findall(r'^========= RACECHECK SUMMARY:.*$', text, re.M)
        result = {key: value for key, value in original.items() if key != 'execution'}
        result.update(name=name, original_result_present=bool(original),
                      execution=execution_summary(record, target, telemetry_summary(telemetry_path)), command_options=options,
                      literal_racecheck_summaries=summaries,
                      literal_unique_zero_summary=summaries == [ZERO],
                      diagnostic_inventory=bool(options['dump_kernel_launches']),
                      native_gate_closed=False)
        if not original:
            result['status'] = 'DIAGNOSTIC_PROBE_NOT_ACCEPTANCE' if is_probe else 'UNFINISHED_OR_UNCLASSIFIED'
        if is_probe:
            result['constructor_complete_marker'] = bool(re.search(r'^constructor_probe complete repeats=\d+$', text, re.M))
            result['frame_coverage_claimed'] = False
            result['acceptance'] = 'DIAGNOSTIC_ONLY_NOT_SANITIZER_ACCEPTANCE'
        row['results'].append(result)
    row['retention_category'] = retention_category(manifest, interruption, execution_records(directory))
    return row


DISCOVERY = '''def discover_repo_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts/run_hr_sanitizer_validation.py").is_file() and (candidate / "CMakeLists.txt").is_file():
            return candidate
    raise RuntimeError("cannot discover repository ancestor for evidence rerun")


'''


def portable_rerun_text(text):
    replacements = []
    assignments = [
        (r'^ROOT = Path\(__file__\)\.resolve\(\)\.parents\[3\][ \t]*$',
         DISCOVERY + 'ROOT = discover_repo_root()',
         'replace fixed parent-depth ROOT with repository ancestor discovery'),
        (r"^REPO = Path\((['\"])/home/ray/developer/ai/saccade_v1\1\)[ \t]*$",
         DISCOVERY + 'REPO = discover_repo_root()',
         'replace fixed absolute REPO with repository ancestor discovery'),
    ]
    for pattern, replacement, description in assignments:
        text, count = re.subn(pattern, lambda _match: replacement, text, flags=re.M)
        if count:
            replacements.append(description)
    return text, replacements


def copy_controls(destination):
    controls = destination / 'control'
    controls.mkdir()
    derivations = []
    sources = [path for path in sorted((SOURCE / 'control').rglob('*'))
               if path.is_file() and '__pycache__' not in path.parts]
    sources.append(ROOT / 'scripts/run_hr_sanitizer_validation.py')
    for source in sources:
        relative = source.relative_to(SOURCE / 'control') if source.is_relative_to(SOURCE / 'control') else Path(source.name)
        captured = relative.with_name(relative.stem + '.captured.py') if relative.suffix == '.py' else relative
        copied = controls / captured
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
        if digest(source) != digest(copied):
            raise RuntimeError(f'control copy mismatch: {source}')
        if source.suffix == '.py':
            text, replacements = portable_rerun_text(source.read_text())
            rerun = controls / relative
            rerun.write_text(text)
            compile(text, str(rerun), 'exec')
            derivations.append({'original': str(source), 'captured': str(captured), 'captured_sha256': digest(copied),
                                'rerun': str(relative), 'rerun_sha256': digest(rerun),
                                'changes': replacements or ['byte-equivalent runnable copy'],
                                'authority': 'Rerun convenience only; captured source remains exact evidence.'})
    # These existing diagnostic helpers produced the constructor probe captures.
    prior = ROOT / 'evidence/hr-sanitizer-memory/2026-11-05/control'
    for name in ('run_probe.py', 'constructor_probe.cpp'):
        source = prior / name
        copied = controls / ('memory_' + source.stem + '.captured' + source.suffix)
        shutil.copy2(source, copied)
        if digest(source) != digest(copied):
            raise RuntimeError('prior diagnostic helper copy mismatch')
    save(controls / 'RERUN_DERIVATIONS.json', derivations)


def prior_contract(destination):
    prior = ROOT / 'evidence/hr-sanitizer/2026-11-05-a00ab3a'
    entries = read_json(prior / 'ARTIFACTS.json')['artifacts']
    binary = next(row for row in entries if row['path'] == 'raw/binaries.tar.gz')
    source = prior / binary['path']
    if digest(source) != binary['sha256'] or source.stat().st_size != binary['bytes']:
        raise RuntimeError('prior frozen binary archive does not match its published artifact manifest')
    save(destination / 'analysis/PRIOR_BINARY_CONTRACT.json', {
        'runtime_revision': 'a00ab3a6abd7cc2c1f823fd3fbac66a345f3ce48',
        'frozen_binary_archive': '../2026-11-05-a00ab3a/raw/binaries.tar.gz',
        'sha256': binary['sha256'], 'bytes': binary['bytes'],
        'prior_artifacts_manifest': '../2026-11-05-a00ab3a/ARTIFACTS.json',
        'prior_artifacts_sha256': digest(prior / 'ARTIFACTS.json'),
        'prior_archive_verified': True,
        'engine_sha256': '8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876',
        'scope': 'Reference to preserved clean runtime and separately compiled HR harness archive. Current capture custody files retain exact executable and input identities.'})


def artifact_manifest(destination, include_report):
    entries = []
    for path in sorted(destination.rglob('*')):
        if not path.is_file() or path.name == 'ARTIFACTS.json' or '__pycache__' in path.parts:
            continue
        if path.relative_to(destination) == Path('REPORT.md') and not include_report:
            continue
        entries.append({'path': str(path.relative_to(destination)), 'sha256': digest(path), 'bytes': path.stat().st_size})
    save(destination / 'ARTIFACTS.json', {'schema': 'hr-bounded-queue-artifacts/1',
         'report_included': include_report and (destination / 'REPORT.md').is_file(), 'artifacts': entries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures-finished', action='store_true', help='Explicitly confirm capture controllers have stopped before freezing')
    parser.add_argument('--finalize-artifacts', action='store_true', help='Refresh only ARTIFACTS.json after REPORT.md is written')
    args = parser.parse_args()
    if args.finalize_artifacts:
        if args.captures_finished or not DESTINATION.is_dir():
            parser.error('finalization requires an existing destination and no capture flag')
        artifact_manifest(DESTINATION, include_report=True)
        print(f'Finalized {DESTINATION / "ARTIFACTS.json"}')
        return
    if not args.captures_finished:
        parser.error('use --captures-finished only after all capture controllers have stopped')
    if DESTINATION.exists():
        parser.error('evidence destination must be fresh; existing evidence is never overwritten')
    directories = [path for path in sorted(SOURCE.iterdir()) if path.is_dir() and path.name not in EXCLUDE]
    for directory in directories:
        manifest = read_json(directory / 'manifest.json', {})
        interruption = read_json(directory / 'interruption-reason.json')
        if not ready_to_archive(manifest, interruption, execution_records(directory)):
            raise RuntimeError(f'capture has no terminal manifest: {directory.name}')
    (DESTINATION / 'raw').mkdir(parents=True)
    (DESTINATION / 'analysis').mkdir()
    archives, summaries = [], []
    for directory in directories:
        archive = archive_directory(directory, DESTINATION / 'raw' / f'{directory.name}.tar.gz')
        archives.append({'capture': directory.name, **archive})
        summaries.append(summarize(directory, archive['archive']))
    copy_controls(DESTINATION)
    prior_contract(DESTINATION)
    save(DESTINATION / 'analysis/ARCHIVE_VERIFICATION.json', {'status': 'BYTE_VERIFIED', 'captures': archives})
    save(DESTINATION / 'analysis/ABORTED_OR_UNSTARTED.json', [row for row in summaries
         if row['retention_category'] != 'COMPLETED_CAPTURE_OR_PROBE'])
    save(DESTINATION / 'analysis/SUMMARY.json', {
        'schema': 'hr-bounded-queue-summary/1', 'created': dt.datetime.now(dt.timezone.utc).isoformat(),
        'native_gate_closed': False, 'native_schedule_acceptance': False, 'formal_acceptance': False,
        'scope': 'Original scoped statuses are retained. Forced synchronization and filtered runs never close the native unfiltered gate. Constructor probes, including launch dumps, are diagnostic only.',
        'captures': summaries})
    artifact_manifest(DESTINATION, include_report=False)
    print(f'Archived and byte-verified {len(archives)} capture directories at {DESTINATION}')


if __name__ == '__main__':
    main()
