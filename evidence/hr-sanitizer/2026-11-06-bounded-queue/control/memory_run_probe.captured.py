#!/usr/bin/env python3
"""Capture an isolated diagnostic command's cgroup and per-process host memory.

Explicit --env assignments are recorded verbatim. Inherited environment is limited
 to HOME/USER/LOGNAME/PATH/TMPDIR/LD_LIBRARY_PATH/CUDA_VISIBLE_DEVICES. This runner
reports observations, not sanitizer acceptance or workload correctness.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    with Path(path).open('w') as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write('\n'); out.flush(); os.fsync(out.fileno())


def read_text(path):
    try:
        return Path(path).read_text(errors='replace')
    except (OSError, ProcessLookupError):
        return None


def scalar_fields(text):
    fields = {}
    for line in (text or '').splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        fields[key] = value.strip()
    return fields


def byte_fields(fields):
    out = {}
    for key, value in fields.items():
        match = re.fullmatch(r'(\d+) kB', value)
        if match:
            out[key + '_bytes'] = int(match[1]) * 1024
        elif value.isdigit():
            out[key] = int(value)
    return out


def cgroup_sample(group):
    row = {'time': now(), 'control_group': group}
    if not group:
        return row
    base = Path('/sys/fs/cgroup') / group.lstrip('/')
    for name in ('memory.current', 'memory.peak', 'memory.swap.current', 'memory.swap.peak',
                 'memory.events', 'memory.events.local', 'memory.stat', 'cgroup.events'):
        text = read_text(base / name)
        if text is None:
            continue
        text = text.strip()
        if text.isdigit():
            row[name] = int(text)
        else:
            row[name] = {k: int(v) for line in text.splitlines()
                         if len(parts := line.split()) == 2 for k, v in [parts] if v.isdigit()}
    return row


def pids_in_cgroup(group):
    if not group:
        return []
    base = Path('/sys/fs/cgroup') / group.lstrip('/')
    files = [base / 'cgroup.procs']
    try:
        files += list(base.rglob('cgroup.procs'))
    except OSError:
        pass
    return sorted({int(word) for p in files for word in (read_text(p) or '').split() if word.isdigit()})


def process_sample(pid):
    base = Path('/proc') / str(pid)
    status = read_text(base / 'status')
    if status is None:
        return None
    fields = scalar_fields(status)
    selected = {key: fields[key] for key in (
        'Name', 'State', 'Pid', 'PPid', 'VmPeak', 'VmSize', 'VmRSS', 'RssAnon',
        'RssFile', 'RssShmem', 'VmData', 'VmStk', 'VmExe', 'VmLib', 'VmPTE',
        'VmSwap', 'Threads') if key in fields}
    smaps = scalar_fields(read_text(base / 'smaps_rollup'))
    try:
        command = (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
    except OSError:
        command = ''
    return {'pid': pid, 'comm': (read_text(base / 'comm') or '').strip(),
            'cmdline': command, 'status': selected, 'status_numeric': byte_fields(selected),
            'smaps_rollup': smaps, 'smaps_numeric': byte_fields(smaps)}


def target_wrapper(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    group = next((line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                  if line.startswith('0::')), '')
    started = now()
    record = {'started': started, 'wrapper_pid': os.getpid()}
    try:
        target = subprocess.Popen(spec['command'], env=spec['environment'])
        record['command_pid'] = target.pid
        save(spec['started_record'], record)
        record['returncode'] = target.wait()
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record.update(cgroup_sample(group))
        record['finished'] = now()
        save(spec['target_record'], record)
    status = record['returncode']
    return status if status >= 0 else 128 - status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--memory', default='8G')
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument('--working-directory', type=Path, default=Path.cwd())
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.command or args.timeout < 1:
        parser.error('provide a command after -- and a positive timeout')
    if not re.fullmatch(r'[1-9]\d*(?:[KMGT](?:i?B)?)?', args.memory):
        parser.error('memory must be a positive systemd byte size such as 8G')
    output = args.output.resolve()
    if output.exists():
        parser.error('output must be a fresh directory')
    args.working_directory = args.working_directory.resolve()
    if not args.working_directory.is_dir():
        parser.error('working directory does not exist')
    inherited = ('HOME', 'USER', 'LOGNAME', 'PATH', 'TMPDIR', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES')
    base_environment = {key: os.environ[key] for key in inherited if key in os.environ}
    base_environment.setdefault('PATH', '/usr/bin:/bin:/opt/cuda/bin')
    environment = dict(base_environment)
    explicit = {}
    for item in args.env:
        key, sep, value = item.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            parser.error('--env requires NAME=VALUE')
        if key in explicit:
            parser.error('duplicate explicit environment key: ' + key)
        explicit[key] = value
        environment[key] = value
    output.mkdir(parents=True)
    (output / 'maps').mkdir()
    wrapper = output / 'target-wrapper.py'
    wrapper.write_bytes(Path(__file__).read_bytes())
    unit = 'saccade-memory-' + uuid.uuid4().hex[:12] + '.service'
    timeout_command = ['/usr/bin/timeout', '--kill-after=10s', str(args.timeout), *args.command]
    spec = {'command': timeout_command, 'environment': environment,
            'target_record': str(output / 'target-final.json'),
            'started_record': str(output / 'target-started.json')}
    save(output / 'target-spec.json', spec)
    # Diagnostic preloads apply to the command and its children, not the Python
    # wrapper. This keeps observer allocations separate from injected records.
    launch = ['systemd-run', '--user', '--wait', '--pipe', '--collect', '--unit=' + unit,
              '--working-directory=' + str(args.working_directory), '--property=MemoryAccounting=yes',
              '--property=MemoryHigh=' + args.memory, '--property=MemoryMax=' + args.memory,
              '--property=MemorySwapMax=0', '--property=KillMode=control-group', '--property=OOMPolicy=kill',
              '--property=RuntimeMaxSec=' + str(args.timeout + 30), '/usr/bin/env', '-i',
              *[key + '=' + value for key, value in base_environment.items()],
              sys.executable, str(wrapper), '--_target', str(output / 'target-spec.json')]
    save(output / 'command.json', {'target': args.command, 'timeout_command': timeout_command,
                                  'service_launch': launch, 'target_environment': environment,
                                  'wrapper_environment': base_environment})
    manifest = {'schema': 'racecheck-host-memory-probe/1', 'started': now(), 'unit': unit,
                'status': 'WAITING', 'memory_high': args.memory, 'memory_max': args.memory,
                'memory_swap_max': 0, 'timeout_seconds': args.timeout, 'sample_interval_seconds': 0.5,
                'maps_interval_seconds': 5, 'maps_rss_growth_bytes': 256 * 1024 * 1024,
                'runner_sha256': hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                'scope': 'Host cgroup/process observation only; no sanitizer PASS classification.'}
    save(output / 'manifest.json', manifest)
    lock_path = Path.home() / '.cache/saccade-v1/gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    active = False
    record = {'started': now(), 'unit': unit, 'process_peaks': {}}
    maps_state = {}
    peak = 0
    event_max = {}
    systemd_results = set()
    group = ''
    returncode = 2

    def stop():
        if active:
            try:
                subprocess.run(['systemctl', '--user', 'stop', unit], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                pass

    try:
        with lock_path.open('a') as lock:
            print('Waiting for GPU lock:', lock_path, flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest['status'] = 'RUNNING'; manifest['service_started'] = now()
            save(output / 'manifest.json', manifest)
            print('Starting:', unit, flush=True)
            with (output / 'run.log').open('w') as log, (output / 'telemetry.jsonl').open('w') as telemetry:
                process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
                active = True
                deadline = time.monotonic() + args.timeout + 60
                while process.poll() is None:
                    shown = subprocess.run(['systemctl', '--user', 'show', unit,
                        '--property=ControlGroup', '--property=MemoryCurrent', '--property=MemoryPeak',
                        '--property=MemoryHigh', '--property=MemoryMax', '--property=MemorySwapMax',
                        '--property=Result', '--property=ExecMainStatus', '--property=ActiveState', '--property=SubState'],
                        capture_output=True, text=True, timeout=5, check=False)
                    props = dict(line.split('=', 1) for line in shown.stdout.splitlines() if '=' in line)
                    group = props.get('ControlGroup') or group
                    if props.get('Result'):
                        systemd_results.add(props['Result'])
                    sample = cgroup_sample(group)
                    sample['systemd'] = props
                    samples = []
                    tick = time.monotonic()
                    for pid in pids_in_cgroup(group):
                        row = process_sample(pid)
                        if row is None:
                            continue
                        rss = row['status_numeric'].get('VmRSS_bytes', 0)
                        prior = record['process_peaks'].setdefault(str(pid), {'comm': row['comm'], 'rss_bytes': 0, 'pss_bytes': 0})
                        prior['rss_bytes'] = max(prior['rss_bytes'], rss)
                        prior['pss_bytes'] = max(prior['pss_bytes'], row['smaps_numeric'].get('Pss_bytes', 0))
                        last = maps_state.get(pid)
                        if last is None or tick - last[0] >= 5 or rss - last[1] >= 256 * 1024 * 1024:
                            maps = read_text(Path('/proc') / str(pid) / 'maps')
                            if maps is not None:
                                count = 0 if last is None else last[2] + 1
                                filename = f'pid-{pid}-{count:05d}.maps'
                                with (output / 'maps' / filename).open('w') as out:
                                    out.write(maps); out.flush(); os.fsync(out.fileno())
                                row['maps_snapshot'] = 'maps/' + filename
                                maps_state[pid] = (tick, rss, count)
                        samples.append(row)
                    sample['processes'] = samples
                    telemetry.write(json.dumps(sample, sort_keys=True) + '\n'); telemetry.flush(); os.fsync(telemetry.fileno())
                    peak = max(peak, sample.get('memory.peak', 0), int(props.get('MemoryPeak', '0')) if props.get('MemoryPeak', '').isdigit() else 0)
                    for key, value in sample.get('memory.events', {}).items():
                        event_max[key] = max(event_max.get(key, 0), value)
                    if time.monotonic() > deadline:
                        record['outer_timeout'] = True
                        stop(); process.wait(timeout=20)
                        break
                    time.sleep(0.5)
                returncode = process.wait(timeout=20)
                record['returncode'] = returncode
    except BaseException as error:
        record['interruption'] = f'{type(error).__name__}: {error}'
        returncode = 130 if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 2
        print(record['interruption'], file=sys.stderr, flush=True)
    finally:
        stop()
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        record.update(finished=now(), sampled_memory_peak_bytes=peak, sampled_memory_events=event_max,
                      sampled_systemd_results=sorted(systemd_results))
        if process is not None:
            record['service_runner_returncode'] = process.returncode
        final = output / 'target-final.json'
        if final.exists():
            record['target'] = json.loads(final.read_text())
            peak = max(peak, record['target'].get('memory.peak', 0))
            record['sampled_memory_peak_bytes'] = peak
        save(output / 'execution.json', record)
        final_events = record.get('target', {}).get('memory.events', {})
        service_log = read_text(output / 'run.log') or ''
        record['service_reported_oom'] = 'Finished with result: oom-kill' in service_log
        resource = any(events.get(key, 0) for events in (event_max, final_events)
                       for key in ('oom', 'oom_kill', 'oom_group_kill')) or record['service_reported_oom']
        save(output / 'execution.json', record)
        timedout = record.get('outer_timeout') or record.get('target', {}).get('returncode') in (124, 137, -9)
        manifest.update(finished=now(), status='INTERRUPTED' if record.get('interruption') else 'OOM' if resource or 'oom-kill' in systemd_results else 'TIMEOUT' if timedout else 'FINISHED',
                        returncode=returncode, peak_bytes=peak)
        save(output / 'manifest.json', manifest)
        print(f"{manifest['status']}: service rc={returncode}, peak={peak} bytes", flush=True)
    return returncode


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--_target':
        sys.exit(target_wrapper(sys.argv[2]))
    def interrupted(signum, _frame):
        raise InterruptedError('signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    sys.exit(main())
