#!/usr/bin/env python3
"""CPU-only scope, flag and fail-closed tests for the lifecycle evidence runner."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lifecycle_capture', HERE / 'capture_lifecycle.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
COMPLETED = ('tracking_session_tests profile=hr-1280x768 mode=lifecycle\n'
             'tracking_session_tests: 180 checks passed\n')
ZERO = ('========= COMPUTE-SANITIZER\n'
        '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n')
POSITIVE = ('========= COMPUTE-SANITIZER\ncontrol=racecheck launch=1\n'
            '========= Error: Race reported between Write access and Read access\n'
            '========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n')
RECORD = {'returncode': 0, 'target': {'returncode': 0}, 'sampled_memory_events': {}}
checks = []


def check(name, predicate):
    assert predicate, name
    checks.append({'name': name, 'passed': True})


def classify(name, record=None, text=None, control=True, custom=False, expected='OPEN'):
    row = runner.classify_lifecycle(copy.deepcopy(RECORD) if record is None else record,
                                    COMPLETED + ZERO if text is None else text, control, custom)
    check(name, row['status'] == expected and not row['native_schedule_acceptance'])
    return row


full = classify('unfiltered success retains forced-sync tag', expected='PASS_FORCED_SYNC')
filtered = classify('filtered success retains both scope tags', custom=True,
                    expected='PASS_FILTERED_FORCED_SYNC')
check('filtered row denies full coverage', not filtered['full_kernel_coverage'])
check('unfiltered row declares full kernel selection', full['full_kernel_coverage'])
classify('zero summary does not replace completion', text=ZERO)
classify('wrong check count fails', text=(COMPLETED + ZERO).replace('180 checks', '179 checks'))
classify('duplicate completion fails', text=COMPLETED + COMPLETED + ZERO)
classify('wrong HR identity fails', text=(COMPLETED + ZERO).replace('hr-1280x768', 't01'))
classify('failed positive control fails', control=False)
classify('missing target exit record fails', record={'returncode': 0})
classify('target nonzero fails', record={**RECORD, 'target': {'returncode': 99}})
classify('service nonzero fails', record={**RECORD, 'returncode': 1})
classify('timeout fails incomplete', record={**RECORD, 'returncode': 124}, expected='INCOMPLETE')
classify('OOM event fails incomplete', record={**RECORD, 'sampled_memory_events': {'oom_kill': '1'}},
         expected='INCOMPLETE')
classify('OOM service result fails incomplete', record={**RECORD, 'sampled_systemd_results': ['oom-kill']},
         expected='INCOMPLETE')
classify('duplicate summary fails', text=COMPLETED + ZERO + ZERO)
classify('unknown sanitizer diagnostic fails', text=COMPLETED + ZERO + '========= Unknown record type\n')
classify('untracked launch fails', text=COMPLETED + ZERO + "========= didn't track launch\n")
classify('nonzero race summary fails', text=COMPLETED + POSITIVE)

rows = [{'name': 'baseline-lifecycle', 'status': 'PASS'},
        {'name': 'control-racecheck', 'status': 'PASS'}, full]
check('complete full aggregate remains forced-sync',
      runner.aggregate_status(rows, True, None, False) == 'PASS_FORCED_SYNC')
check('custody failure forces OPEN', runner.aggregate_status(rows, False, None, False) == 'OPEN')
check('exception forces OPEN', runner.aggregate_status(rows, True, 'interrupted', False) == 'OPEN')
check('missing lifecycle forces OPEN', runner.aggregate_status(rows[:-1], True, None, False) == 'OPEN')
check('filtered row cannot grant full scope',
      runner.aggregate_status([*rows[:-1], filtered], True, None, False) == 'OPEN')
check('unfiltered row cannot masquerade as filtered capture',
      runner.aggregate_status(rows, True, None, True) == 'OPEN')

default = runner.parse_args(['--output', '/tmp/not-created-lifecycle-test'])
check('default flags remain unfiltered one worker sync8',
      not default.custom_kernels_only and default.racecheck_num_workers == 1
      and default.force_synchronization_limit == 8)
auto = runner.parse_args(['--output', '/tmp/not-created-lifecycle-test', '--racecheck-num-workers', '0'])
check('automatic worker count is accepted', auto.racecheck_num_workers == 0)
for flag, value in [('--racecheck-num-workers', '-1'), ('--force-synchronization-limit', '0'),
                    ('--timeout', '0'), ('--memory', '0G'), ('--kernel-name', '.*')]:
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            runner.parse_args(['--output', '/tmp/not-created-lifecycle-test', flag, value])
        except SystemExit as error:
            check(f'reject invalid or arbitrary flag {flag}={value}', error.code == 2)
        else:
            raise AssertionError(f'accepted {flag}={value}')


def mocked_capture(name, custom=False, workers=1, scenario=None, expected='OPEN'):
    captured_commands = {}
    digest_counts = {}

    class FakeCapture:
        def __init__(self, args, output):
            self.output = output
            self.wrapper = output / 'target-wrapper.py'
            self.wrapper.write_text('# CPU-only mock wrapper\n')
            self.environment = {'PATH': '/usr/bin:/bin'}
            self.removed_environment = {}

        def run(self, stage, command):
            captured_commands[stage] = list(map(str, command))
            record = copy.deepcopy(RECORD)
            if stage == 'baseline-lifecycle':
                text = COMPLETED
            elif stage == 'control-racecheck':
                text = POSITIVE if scenario != 'bad-control' else ZERO
                record['returncode'] = record['target']['returncode'] = 99
            else:
                if scenario == 'raise-session':
                    raise RuntimeError('CPU mock interruption')
                text = COMPLETED + ZERO
                if scenario == 'missing-completion':
                    text = ZERO
                elif scenario == 'missing-final':
                    record.pop('target')
                elif scenario == 'oom':
                    record['sampled_memory_events']['oom'] = '1'
                elif scenario == 'unknown':
                    text += '========= Unknown record type\n'
            (self.output / f'{stage}.log').write_text(text)
            return record

        def stop(self):
            pass

    def fake_digest(path):
        key = str(path)
        digest_counts[key] = digest_counts.get(key, 0) + 1
        if scenario == 'changed-wrapper' and Path(path).name == 'target-wrapper.py' and digest_counts[key] > 1:
            return 'b' * 64
        return 'a' * 64

    runtime = {'revision': runner.RUNTIME_REVISION, 'engine': {'engine': runner.ENGINE_SHA256}}
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / 'capture'
        argv = ['--output', str(output), '--racecheck-num-workers', str(workers)]
        if custom:
            argv.append('--custom-kernels-only')
        with (patch.object(runner.m, 'Capture', FakeCapture),
              patch.object(runner, 'sdk_paths', return_value=[]),
              patch.object(runner.m, 'source_snapshot', return_value=runtime),
              patch.object(runner.m, 'digest', side_effect=fake_digest),
              patch.object(runner.fcntl, 'flock'),
              contextlib.redirect_stdout(io.StringIO())):
            try:
                result = runner.main(argv)
            except RuntimeError:
                check(name + ': exception path expected', scenario == 'raise-session')
                result = 2
        manifest = json.loads((output / 'manifest.json').read_text())
        check(name + ': expected aggregate scope', manifest['status'] == expected)
        check(name + ': never native acceptance', manifest['native_schedule_acceptance'] is False)
        check(name + ': explicit kernel coverage', manifest['full_kernel_coverage'] == (not custom))
        check(name + ': exit matches capture success', (result == 0) == expected.startswith('PASS_'))
        for stage in ('control-racecheck', 'racecheck-lifecycle'):
            command = captured_commands[stage]
            check(name + f': {stage} workers match',
                  command[command.index('--racecheck-num-workers') + 1] == str(workers))
            check(name + f': {stage} sync matches',
                  command[command.index('--force-synchronization-limit') + 1] == '8')
            check(name + f': {stage} explicit filter policy', ('--kernel-name' in command) == custom)
            if custom:
                regex = runner.CONTROL_REGEX if stage == 'control-racecheck' else runner.CUSTOM_REGEX
                check(name + f': {stage} exact regex',
                      command[command.index('--kernel-name') + 1] == f'regex={regex}')
        if scenario != 'raise-session':
            results = json.loads((output / 'results.json').read_text())
            check(name + ': three required stages retained', len(results) == 3)
        custody = json.loads((output / 'custody-before.json').read_text())
        kept = custody['harness_runners_and_libraries']
        check(name + ': shared runner in custody', str(runner.CAPTURE_RUNNER) in kept)
        check(name + ': local runner in custody', str(HERE / 'capture_lifecycle.py') in kept)
        check(name + ': target wrapper in custody', str(output / 'target-wrapper.py') in kept)
        if scenario == 'changed-wrapper':
            check(name + ': changed wrapper detected', manifest['custody_unchanged'] is False)


mocked_capture('full successful capture', workers=8, expected='PASS_FORCED_SYNC')
mocked_capture('filtered successful capture', custom=True, workers=0, expected='PASS_FILTERED_FORCED_SYNC')
for scenario in ('bad-control', 'missing-completion', 'missing-final', 'oom', 'unknown',
                 'changed-wrapper', 'raise-session'):
    mocked_capture(scenario, custom=True, scenario=scenario)

result = {'schema': 'hr-lifecycle-runner-cpu-tests/1', 'finished': runner.m.now(),
          'gpu_executed': False, 'checks': checks, 'passed': len(checks), 'failed': 0}
(HERE / 'lifecycle-runner-tests.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'lifecycle runner CPU tests: {len(checks)}/{len(checks)} passed; no GPU execution')
