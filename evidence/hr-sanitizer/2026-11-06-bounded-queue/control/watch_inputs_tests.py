#!/usr/bin/env python3
"""CPU temporary-directory checks for the read-only inotify progress observer."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / 'watch_inputs.py'
spec = importlib.util.spec_from_file_location('watch_inputs', SCRIPT)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)
checks = []


def check(name, condition):
    assert condition, name
    checks.append({'name': name, 'passed': True})


def event_bytes(name=b'000001.jpg', mask=watch.IN_OPEN, wd=1):
    padded = name + b'\0'
    padded += b'\0' * ((-len(padded)) % 4)
    return watch.HEADER.pack(wd, mask, 0, len(padded)) + padded


one = event_bytes()
events = watch.decode_events(one + event_bytes(b'000002.JPEG'))
check('decode two padded open events', len(events) == 2)
check('select six-digit JPG', watch.selected(events[0]))
check('select six-digit uppercase JPEG', watch.selected(events[1]))
for name in (b'00001.jpg', b'0000001.jpg', b'000001.png', b'other.jpg', b'000001.jpg.bak', b''):
    check('ignore invalid JPEG name ' + repr(name), not watch.selected(watch.decode_events(event_bytes(name))[0]))
check('non-open mask ignored', not watch.selected(watch.decode_events(event_bytes(mask=0x100))[0]))
overflow = watch.decode_events(watch.HEADER.pack(-1, watch.IN_Q_OVERFLOW, 0, 0))[0]
check('queue overflow decoded explicitly', overflow['mask'] == watch.IN_Q_OVERFLOW and overflow['watch_descriptor'] == -1)
check('queue overflow not input progress', not watch.selected(overflow))
for label, data in (
    ('partial header', one[:8]), ('partial filename', one[:-1]),
    ('missing NUL', watch.HEADER.pack(1, watch.IN_OPEN, 0, 3) + b'abc'),
    ('invalid slash', event_bytes(b'sub/000001.jpg')),
    ('invalid descriptor', watch.HEADER.pack(-2, watch.IN_OPEN, 0, 0)),
):
    try:
        watch.decode_events(data)
    except ValueError:
        check('reject malformed ' + label, True)
    else:
        raise AssertionError('accepted malformed ' + label)

identity = watch.process_identity(os.getpid())
check('read own Linux starttime', identity and identity['starttime_ticks'] > 0)
check('same live PID retained', watch.process_end_reason(os.getpid(), identity) is None)
with patch.object(watch, 'process_identity', return_value=None):
    check('absent PID terminates', watch.process_end_reason(os.getpid(), identity) == 'process_disappeared')
with patch.object(watch, 'process_identity', return_value={'state': 'Z', 'starttime_ticks': identity['starttime_ticks']}):
    check('zombie PID terminates', watch.process_end_reason(os.getpid(), identity) == 'process_exited')
with patch.object(watch, 'process_identity', return_value={'state': 'S', 'starttime_ticks': identity['starttime_ticks'] + 1}):
    check('PID reuse terminates', watch.process_end_reason(os.getpid(), identity) == 'pid_reused')


def rows(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError):
        return []


def await_kind(path, kind, process):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        found = rows(path)
        if any(row['kind'] == kind for row in found):
            return found
        if process.poll() is not None:
            raise AssertionError(f'observer exited {process.returncode}: {found}')
        time.sleep(0.01)
    raise AssertionError(f'observer did not emit {kind}')


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    inputs = root / 'inputs'
    inputs.mkdir()
    image = inputs / '000001.jpg'
    image.write_bytes(b'CPU fixture, not a real JPEG decoder input')
    invalid = inputs / 'ignored.txt'
    invalid.write_text('fixture')
    target = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
    output = root / 'access.jsonl'
    observer = subprocess.Popen([sys.executable, str(SCRIPT), '--directory', str(inputs),
                                 '--output', str(output), '--pid', str(target.pid), '--timeout', '5'])
    try:
        await_kind(output, 'watch_ready', observer)
        with image.open('rb') as stream:
            stream.read(1)
        with invalid.open('rb') as stream:
            stream.read(1)
        await_kind(output, 'input_open', observer)
        target.terminate()
        target.wait(timeout=2)
        check('observer exits after watched process disappears', observer.wait(timeout=3) == 0)
        observed = rows(output)
        opens = [row for row in observed if row['kind'] == 'input_open']
        check('real inotify open event delivered', any(row['filename'] == '000001.jpg' for row in opens))
        check('nonmatching input open omitted', all(row['filename'] != 'ignored.txt' for row in opens))
        check('all records timestamped', all('utc' in row and 'monotonic_seconds' in row for row in observed))
        check('start and terminal retained', observed[0]['kind'] == 'start' and observed[-1]['kind'] == 'terminal')
        check('explicitly denies PID and frame attribution',
              observed[0]['pid_attribution'] is False and observed[0]['frame_completion'] is False)
        check('exit marks undrained post-exit events', observed[-1]['queued_events_not_drained_after_pid_exit'])
        original_output = output.read_bytes()
        duplicate = subprocess.run([sys.executable, str(SCRIPT), '--directory', str(inputs),
                                    '--output', str(output), '--pid', str(os.getpid()), '--timeout', '0.1'],
                                   capture_output=True)
        check('fresh output enforced', duplicate.returncode != 0 and output.read_bytes() == original_output)
    finally:
        if target.poll() is None:
            target.kill()
            target.wait()
        if observer.poll() is None:
            observer.kill()
            observer.wait()

    output = root / 'signal.jsonl'
    observer = subprocess.Popen([sys.executable, str(SCRIPT), '--directory', str(inputs),
                                 '--output', str(output), '--pid', str(os.getpid()), '--timeout', '5'])
    try:
        await_kind(output, 'watch_ready', observer)
        observer.send_signal(signal.SIGTERM)
        check('SIGTERM exits cleanly', observer.wait(timeout=3) == 0)
        check('SIGTERM terminal retained', rows(output)[-1]['reason'] == 'signal')
    finally:
        if observer.poll() is None:
            observer.kill()
            observer.wait()

    output = root / 'timeout.jsonl'
    timed = subprocess.run([sys.executable, str(SCRIPT), '--directory', str(inputs),
                            '--output', str(output), '--pid', str(os.getpid()), '--timeout', '0.05'],
                           timeout=3, capture_output=True)
    check('bounded timeout is terminal', timed.returncode == 0 and rows(output)[-1]['reason'] == 'timeout')

result = {'schema': 'inotify-input-observer-cpu-tests/1', 'finished_utc': watch.dt.datetime.now(watch.dt.timezone.utc).isoformat(),
          'gpu_executed': False, 'real_dataset_observed': False, 'checks': checks,
          'passed': len(checks), 'failed': 0}
(HERE / 'watch-inputs-tests.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'inotify observer CPU tests: {len(checks)}/{len(checks)} passed; temporary directories only')
