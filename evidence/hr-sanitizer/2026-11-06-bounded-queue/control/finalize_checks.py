#!/usr/bin/env python3
"""Read-only acceptance/custody checks, run after the capture controllers exit."""
import importlib.util
import json
from pathlib import Path
import re
import tarfile

def discover_repo_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts/run_hr_sanitizer_validation.py").is_file() and (candidate / "CMakeLists.txt").is_file():
            return candidate
    raise RuntimeError("cannot discover repository ancestor for evidence rerun")


ROOT = discover_repo_root()
BASE = ROOT / 'build/hr-racecheck-20261106'
spec = importlib.util.spec_from_file_location('hr_validation', ROOT / 'scripts/run_hr_sanitizer_validation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CASES = {
    'custom-sync8-workers8-four-modes': 'PASS_FILTERED_FORCED_SYNC',
    'lifecycle-custom-sync8-workers8': 'PASS_FILTERED_FORCED_SYNC',
    'full-sync1-workers8-ordinary-serial': 'PASS_FORCED_SYNC',
}


def read(path):
    return json.loads(path.read_text())


def main():
    records = []
    reference_dir = BASE / 'custom-sync8-workers8-four-modes'
    reference = {suffix: (reference_dir / ('baseline-graph-pipelined.' + suffix)).read_bytes()
                 for suffix in ('txt', 'bin', 'scmo')}
    assert len(reference['bin']) == 30 * 12368
    literal_files = []
    for name, status in CASES.items():
        directory = BASE / name
        manifest = read(directory / 'manifest.json')
        assert manifest['status'] == status and manifest['finished']
        assert manifest['custody_unchanged'] and not manifest['native_schedule_acceptance']
        assert read(directory / 'custody-before.json') == read(directory / 'custody-after.json')
        results = read(directory / 'results.json')
        assert len(results) == (9 if name.startswith('custom-') else 3)
        for row in results:
            execution = row['execution']
            expected_rc = 99 if row['name'].startswith('control-') else 0
            assert execution['returncode'] == expected_rc
            assert execution['target']['returncode'] == expected_rc
            text = (directory / (row['name'] + '.log')).read_text()
            assert not m.incomplete(execution, text)
            if row['name'].startswith('racecheck-'):
                assert row['status'] == status and row['target_completed']
                assert m.clean_sanitizer_log('racecheck', text)
                assert row.get('native_schedule_acceptance') is False
                if name.startswith('lifecycle-'):
                    assert len(re.findall(r'^tracking_session_tests: 180 checks passed$', text, re.M)) == 1
                else:
                    assert row['byte_parity'] and 'track_sequence frames=30 ' in text
            if row['name'].startswith('baseline-') or row['name'].startswith('racecheck-'):
                if not name.startswith('lifecycle-'):
                    for suffix, expected in reference.items():
                        path = directory / (row['name'] + '.' + suffix)
                        assert path.read_bytes() == expected, path
                        literal_files.append({'file': str(path.relative_to(ROOT)),
                                              'bytes': path.stat().st_size, 'sha256': m.digest(path)})
        records.append({'capture': name, 'status': status, 'custody_unchanged': True})
    full = BASE / 'full-sync1-workers8-ordinary-serial'
    command = read(full / 'racecheck-ordinary-serial.command.json')['target']
    assert '--kernel-name' not in command and '--launch-count' not in command and '--launch-skip' not in command
    assert '--serial' in command and '--no-cuda-graph' in command
    assert command[command.index('--force-synchronization-limit') + 1] == '1'
    assert command[command.index('--racecheck-num-workers') + 1] == '8'
    for name in ('full-sync8-workers8-graph', 'full-sync1-workers8-graph-24g'):
        directory = BASE / name
        manifest = read(directory / 'manifest.json')
        assert manifest['status'] == 'OPEN' and manifest['custody_unchanged']
        result = read(directory / 'results.json')[-1]
        assert result['status'] == 'INCOMPLETE' and not result['target_completed'] and not result['byte_parity']
    before = read(BASE / 'control/libraries-before.json')
    after = {path: m.digest(path) for path in before}
    assert before == after
    m.save(BASE / 'control/libraries-after.json', after)
    binaries = []
    paths = [ROOT / 'build/hr-sanitizer-20261105/cuda/track_sequence',
             ROOT / 'build/hr-sanitizer-20261105/cuda/postprocess_sanitizer_control',
             ROOT / 'build/hr-sanitizer-20261105/harness/tracking_session_hr_tests']
    with tarfile.open(ROOT / 'evidence/hr-sanitizer/2026-11-05-a00ab3a/raw/binaries.tar.gz') as archive:
        for path in paths:
            assert archive.extractfile(path.name).read() == path.read_bytes()
            binaries.append({'file': str(path.relative_to(ROOT)), 'sha256': m.digest(path),
                             'byte_identical_to_prior_archive': True})
    for test in ('runner', 'custom-runner', 'lifecycle-runner', 'archive', 'watch-inputs'):
        report = read(BASE / ('control/' + test + '-tests.json'))
        assert report.get('failed', 0) == 0
    result = {'schema': 'hr-racecheck-final-verification/1', 'time': m.now(), 'status': 'PASS',
              'captures': records, 'literal_output_files_compared': len(literal_files),
              'comparison': 'literal byte equality to graph baseline, independent of stored parity booleans',
              'outputs': literal_files, 'binaries': binaries, 'sdk_library_hashes_unchanged': len(after),
              'native_schedule_acceptance': False, 'formal_acceptance': False}
    m.save(BASE / 'control/FINAL_VERIFICATION.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'literal_output_files_compared',
                                                 'sdk_library_hashes_unchanged', 'captures')}, indent=2))


if __name__ == '__main__':
    main()
