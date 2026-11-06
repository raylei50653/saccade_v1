#!/usr/bin/env python3
"""Observe JPEG opens with inotify; this is not PID attribution or frame completion."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import json
import os
from pathlib import Path
import re
import select
import signal
import struct
import time


IN_OPEN = 0x00000020
IN_UNMOUNT = 0x00002000
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
INVALIDATED = IN_UNMOUNT | IN_IGNORED | IN_DELETE_SELF | IN_MOVE_SELF
HEADER = struct.Struct('=iIII')
JPEG = re.compile(r'[0-9]{6}\.jpe?g', re.I)


def decode_events(data):
    """Decode complete Linux inotify records; reject malformed/truncated buffers."""
    events = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < HEADER.size:
            raise ValueError('truncated inotify event header')
        wd, mask, cookie, length = HEADER.unpack_from(data, offset)
        offset += HEADER.size
        if length > len(data) - offset:
            raise ValueError('truncated inotify filename')
        raw = data[offset:offset + length]
        offset += length
        if length and (not raw.endswith(b'\0') or b'/' in raw):
            raise ValueError('malformed inotify filename')
        if wd < 0 and not mask & IN_Q_OVERFLOW:
            raise ValueError('negative watch descriptor without queue overflow')
        events.append({'watch_descriptor': wd, 'mask': mask, 'cookie': cookie,
                       'filename': os.fsdecode(raw.split(b'\0', 1)[0])})
    return events


def selected(event):
    return bool(event['mask'] & IN_OPEN and JPEG.fullmatch(event['filename']))


def process_identity(pid):
    """Return state and Linux starttime; None means absent or unreadable."""
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = text[text.rindex(')') + 2:].split()
        return {'state': fields[0], 'starttime_ticks': int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def process_end_reason(pid, original):
    current = process_identity(pid)
    if current is None:
        return 'process_disappeared'
    if current['state'] in ('Z', 'X', 'x'):
        return 'process_exited'
    if original is None or current['starttime_ticks'] != original['starttime_ticks']:
        return 'pid_reused'
    return None


class Watch:
    def __init__(self, directory):
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_init1.restype = ctypes.c_int
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        libc.inotify_add_watch.restype = ctypes.c_int
        self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        mask = IN_OPEN | IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT
        self.wd = libc.inotify_add_watch(self.fd, os.fsencode(directory), mask)
        if self.wd < 0:
            code = ctypes.get_errno()
            self.close()
            raise OSError(code, os.strerror(code), str(directory))

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSONL file')
    parser.add_argument('--pid', type=int, required=True, help='Stop when this PID exits; inotify does not attribute opens to it')
    parser.add_argument('--timeout', type=float, default=3700)
    args = parser.parse_args(argv)
    if args.pid < 1 or args.timeout <= 0 or not args.timeout < float('inf'):
        parser.error('pid and finite timeout must be positive')
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir():
        parser.error('directory must name an existing directory')
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stopped = []
    previous = {}
    watch = None
    original = process_identity(args.pid)
    started = time.monotonic()
    counts = {'input_opens': 0, 'ignored_events': 0, 'queue_overflows': 0}
    reason = 'observer_error'
    exitcode = 1
    with output.open('x') as stream:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, lambda number, _frame: stopped.append(number))
        def emit(kind, **fields):
            row = {'kind': kind, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'monotonic_seconds': time.monotonic(), **fields}
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush()

        emit('start', directory=str(directory), watched_pid=args.pid, original_process=original,
             timeout_seconds=args.timeout, pid_attribution=False, frame_completion=False,
             scope='Filesystem access observation only. Opens may come from any process and may repeat. '
                   'No target PID attribution, CUDA completion, packet completion or sanitizer acceptance.',
             exit_policy='Check PID existence, state and original starttime before draining each event batch.')
        try:
            watch = Watch(directory)
            emit('watch_ready', watch_descriptor=watch.wd, mask=IN_OPEN)
            while True:
                if stopped:
                    reason, exitcode = 'signal', 0
                    break
                reason = process_end_reason(args.pid, original)
                if reason:
                    exitcode = 0
                    break
                remaining = args.timeout - (time.monotonic() - started)
                if remaining <= 0:
                    reason, exitcode = 'timeout', 0
                    break
                ready, _, _ = select.select([watch.fd], [], [], min(remaining, 0.25))
                if not ready:
                    continue
                # Stop before reading opens from a parent's post-exit input hashing.
                reason = process_end_reason(args.pid, original)
                if reason:
                    exitcode = 0
                    break
                try:
                    data = os.read(watch.fd, 1024 * 1024)
                except BlockingIOError:
                    continue
                for event in decode_events(data):
                    if event['mask'] & IN_Q_OVERFLOW:
                        counts['queue_overflows'] += 1
                        emit('queue_overflow', **event, observation_gap=True)
                    elif event['mask'] & INVALIDATED:
                        emit('watch_invalidated', **event)
                        raise OSError(errno.ESTALE, 'input directory watch invalidated')
                    elif selected(event):
                        counts['input_opens'] += 1
                        emit('input_open', **event)
                    else:
                        counts['ignored_events'] += 1
        except (OSError, ValueError) as error:
            reason = 'observer_error'
            emit('error', error=f'{type(error).__name__}: {error}')
        finally:
            if watch is not None:
                watch.close()
            emit('terminal', reason=reason, exitcode=exitcode, **counts,
                 elapsed_seconds=time.monotonic() - started,
                 signal=stopped[-1] if stopped else None,
                 final_process=process_identity(args.pid), pid_attribution=False,
                 frame_completion=False, queued_events_not_drained_after_pid_exit=True,
                 observation_gaps=counts['queue_overflows'] > 0)
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return exitcode


if __name__ == '__main__':
    raise SystemExit(main())
