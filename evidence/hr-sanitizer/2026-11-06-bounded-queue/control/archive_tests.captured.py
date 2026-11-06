#!/usr/bin/env python3
"""Pure readiness/scope checks; does not run the bundler or write captures."""
import hashlib
import importlib.util
import json
from pathlib import Path

source = Path(__file__).with_name('archive_evidence.py')
spec = importlib.util.spec_from_file_location('archive_evidence', source)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)
reason = {'stage': 'Waiting for GPU lock; no GPU baseline/control/session started',
          'reason': 'Interrupted before preflight completed'}
checks = []


def check(name, condition):
    checks.append({'name': name, 'passed': bool(condition)})


preflight = {'started': '2026-11-06T00:00:00+00:00'}
check('explicitly interrupted preflight accepted', archive.ready_to_archive(preflight, reason, []))
check('preflight stays outside completed category',
      archive.retention_category(preflight, reason, []) == 'ABORTED_OR_UNSTARTED_PREFLIGHT')
check('manifestless interrupted preflight accepted', archive.ready_to_archive({}, reason, []))
check('active unfinished manifest rejected',
      not archive.ready_to_archive({'status': 'RUNNING', **preflight}, None, ['racecheck.command.json']))
check('unfinished preflight without reason rejected', not archive.ready_to_archive(preflight, None, []))
check('empty preflight without reason rejected', not archive.ready_to_archive({}, None, []))
for record in ('racecheck.command.json', 'racecheck.execution.json', 'results.json',
               'target-started.json', 'target-final.json', 'racecheck.target.json', 'telemetry.jsonl'):
    check(f'interruption reason cannot override execution record {record}',
          not archive.ready_to_archive(preflight, reason, [record]))
check('completed capture accepted', archive.ready_to_archive({'finished': '2026-11-06T00:00:01+00:00'}, None,
                                                            ['racecheck.execution.json']))
check('completed interrupted capture stays interrupted',
      archive.retention_category({'finished': '2026-11-06T00:00:01+00:00', 'error': 'KeyboardInterrupt'},
                                 reason, ['racecheck.command.json']) == 'INTERRUPTED_ATTEMPT')

fixture = ('from pathlib import Path\n'
           'ROOT = Path(__file__).resolve().parents[3]\n'
           'quoted_root = "ROOT = Path(__file__).resolve().parents[3]"\n'
           'REPO = Path(\'/home/ray/developer/ai/saccade_v1\')\n'
           'quoted_repo = "REPO = Path(\'/home/ray/developer/ai/saccade_v1\')"\n')
original_fixture = fixture
derived, changes = archive.portable_rerun_text(fixture)
check('actual ROOT assignment transformed', '\nROOT = discover_repo_root()\n' in derived)
check('actual REPO assignment transformed', '\nREPO = discover_repo_root()\n' in derived)
check('quoted ROOT pattern remains literal', 'quoted_root = "ROOT = Path(__file__).resolve().parents[3]"' in derived)
check('quoted REPO pattern remains literal', 'quoted_repo = "REPO = Path(\'/home/ray/developer/ai/saccade_v1\')"' in derived)
check('only two assignment derivations recorded', len(changes) == 2)
check('original captured text remains unchanged', fixture == original_fixture)
compile(derived, '<assignment-and-quoted-pattern-fixture>', 'exec')
check('assignment and quoted-pattern derivative compiles', True)
self_text = source.read_text()
derived_self, self_changes = archive.portable_rerun_text(self_text)
check('archiver own pattern strings remain byte equivalent', derived_self == self_text and not self_changes)
compile(derived_self, '<derived-archive-evidence>', 'exec')
check('archiver own derivative compiles', True)
controls = sorted(source.parent.rglob('*.py')) + [archive.ROOT / 'scripts/run_hr_sanitizer_validation.py']
for control in controls:
    if '__pycache__' in control.parts:
        continue
    captured = control.read_text()
    derivative, _changes = archive.portable_rerun_text(captured)
    compile(derivative, str(control) + ':derived', 'exec')
    check(f'derived control compiles {control.name}', True)
    check(f'captured control unchanged {control.name}', control.read_text() == captured)
report = {'schema': 'hr-evidence-archive-pure-tests/1', 'gpu_work': False, 'archive_action_run': False,
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'checks': checks,
          'passed': sum(row['passed'] for row in checks), 'failed': sum(not row['passed'] for row in checks)}
Path(__file__).with_name('archive-tests.json').write_text(json.dumps(report, indent=2) + '\n')
print(f"{report['passed']} pure archive checks passed; {report['failed']} failed; no archive/GPU action")
raise SystemExit(bool(report['failed']))
