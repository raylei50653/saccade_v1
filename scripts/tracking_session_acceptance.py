#!/usr/bin/env python3
"""Summarize the fixed TRACKING §20.8 score, overlap and throughput gates."""
import csv
from decimal import Decimal
import importlib.util
from pathlib import Path
import re
import sqlite3
import sys

sys.dont_write_bytecode = True

METRICS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW')


def metrics(detail):
    with Path(detail).open(newline='') as f:
        row = next(r for r in csv.DictReader(f) if r['seq'] == 'COMBINED')
    return {k: Decimal(row[k + '___AUC' if k in METRICS[:3] else k]) for k in METRICS}


def score(args):
    group, actual, expected, output = args
    values = metrics(actual)
    if expected != '-':
        reference = metrics(expected)
        for key in METRICS:
            if values[key] != reference[key]:
                raise RuntimeError(f'{group}: first differing score {key}: {values[key]} != {reference[key]}')
    path = Path(output)
    with path.open('a', newline='') as f:
        writer = csv.writer(f)
        if path.stat().st_size == 0:
            writer.writerow(('set', *METRICS, 'reference_equal'))
        writer.writerow((group, *(values[k] * (1 if k == 'IDSW' else 100) for k in METRICS), expected != '-'))
    print(group, ' '.join(f'{k}={values[k] * (1 if k == "IDSW" else 100)}' for k in METRICS),
          'reference equal' if expected != '-' else 'record only')


TIMING = re.compile(r'^track_sequence frames=(\d+) mode=(pipelined|serial) wall_ms=([\d.e+-]+) ms_per_frame=([\d.e+-]+)$', re.M)


def timing(path, mode):
    matches = TIMING.findall(path.read_text())
    if len(matches) != 1 or matches[0][1] != mode:
        raise RuntimeError(f'{path}: missing unique {mode} timing')
    frames, _, ms, _ = matches[0]
    return int(frames), Decimal(ms)


def throughput(args):
    cap = Path(args[0])
    sequences = [f'MOT17-{s}-SDP' for s in ('02', '04', '05', '09', '10', '11', '13')]
    total = {'before': [0, Decimal(0)], 'after': [0, Decimal(0)]}
    rows, raw = [], []
    for sequence in sequences:
        sums = {}
        for arm in total:
            n, ms = 0, Decimal(0)
            for repeat in (1, 2, 3):
                frames, wall = timing(cap / f'timing-{arm}-{sequence}-{repeat}.log', 'pipelined')
                raw.append((sequence, arm, repeat, frames, wall, wall / frames))
                n += frames
                ms += wall
            sums[arm] = (n, ms)
            total[arm][0] += n
            total[arm][1] += ms
        if sums['before'][0] != sums['after'][0]:
            raise RuntimeError(f'{sequence}: before/after frame counts differ')
        before = sums['before'][1] / sums['before'][0]
        after = sums['after'][1] / sums['after'][0]
        rows.append((sequence, sums['before'][0], before, after, 100 * (after / before - 1)))
    before = total['before'][1] / total['before'][0]
    after = total['after'][1] / total['after'][0]
    change = 100 * (after / before - 1)
    rows.append(('Total', total['before'][0], before, after, change))
    with (cap / 'throughput-runs.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(('sequence', 'arm', 'repeat', 'frames', 'wall_ms', 'ms_per_frame')); w.writerows(raw)
    with (cap / 'throughput.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(('sequence', 'frames_per_arm', 'before_ms_per_frame', 'after_ms_per_frame', 'change_percent')); w.writerows(rows)
    with (cap / 'serial-latency.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(('sequence', 'frames', 'wall_ms', 'ms_per_frame'))
        for sequence in sequences:
            frames, wall = timing(cap / f'timing-serial-{sequence}.log', 'serial')
            w.writerow((sequence, frames, wall, wall / frames))
    for row in rows:
        print(','.join(map(str, row)))
    if change > 3:
        raise RuntimeError(f'total throughput regression {change}% exceeds 3%')
    print('throughput: PASS')


def overlap(args):
    database, output = args
    db = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    spec = importlib.util.spec_from_file_location('placement', Path(__file__).with_name('check_stream_placement.py'))
    placement = importlib.util.module_from_spec(spec); spec.loader.exec_module(placement)
    kernels = db.execute(
        'SELECT k.start,k.end,k.streamId,d.value,r.globalTid,r.start,k.graphId FROM CUPTI_ACTIVITY_KIND_KERNEL k '
        'JOIN StringIds d ON d.id=k.demangledName '
        'LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId ORDER BY k.start').fetchall()
    gray = [k for k in kernels if ('F05', 'D') in placement.rule(k[3])]
    inputs = [k for k in kernels if re.match(placement.U + r'flow_input\(', k[3])]
    cells = [k for k in kernels if re.match(placement.U + r'flow_cells\(', k[3])]
    ranges = db.execute(
        'SELECT n.globalTid,n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId '
        'WHERE coalesce(n.text,s.value)=? AND n.end IS NOT NULL', (placement.ENQUEUE,)).fetchall()
    d_graphs = {k[6] for k in kernels if k[6] is not None
                and any(side == 'D' for _, side in placement.rule(k[3]))}
    if len(gray) != len(inputs) or len(gray) != len(cells) or len(gray) < 2:
        raise RuntimeError('cannot match F05/input/cells to frames')
    if len({k[2] for k in inputs + cells}) != 1 or len({k[2] for k in gray}) != 1 or gray[0][2] == inputs[0][2]:
        raise RuntimeError('F05 and F06 are not on distinct D/M streams')
    rows = []
    for frame in range(1, len(gray)):
        host_start = gray[frame][5]
        host_end = gray[frame + 1][5] if frame + 1 < len(gray) else float('inf')
        engine = [k for k in kernels if k[5] is not None and host_start <= k[5] < host_end
                  and ((k[6] in d_graphs and not placement.rule(k[3])) or
                       (k[6] is None and k[2] == gray[frame][2]
                        and any(t == k[4] and a <= k[5] <= b for t, a, b in ranges)))]
        # A1/A3 establish the M dependencies: OFA starts after the input copy
        # and completes before cells can read its output. If this whole device
        # interval is contained in the same-frame F03 interval, OFA execution
        # necessarily overlaps F03. Mere intersection of the bounds cannot pass.
        # These are execution bounds, not an OFA latency measurement.
        begin, end = inputs[frame][1], cells[frame][0]
        if engine:
            f03_begin, f03_end = min(k[0] for k in engine), max(k[1] for k in engine)
            rows.append((frame, begin, end, f03_begin, f03_end,
                         f03_begin <= begin < end <= f03_end))
    with Path(output).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('frame', 'ofa_earliest_start_ns', 'ofa_latest_end_ns', 'f03_start_ns', 'f03_end_ns', 'ofa_bounds_contained_in_f03'))
        w.writerows(rows)
    contained = sum(row[-1] for row in rows)
    if not contained:
        raise RuntimeError('nsys does not establish OFA overlap: no whole OFA device interval lies inside same-frame F03')
    print(f'OFA/F03 overlap: PASS by device-bound containment in {contained}/{len(rows)} frames; '
          f'D={gray[0][2]} M={inputs[0][2]}; A1/A3 establish OFA ordering between the M kernels')


OFA_API_ERROR = 'Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument" on CUDA API call to cuArrayGetPlane.'


def memcheck_ofa(args):
    """Prove NVOF's own cuArrayGetPlane errors are the whole of a memcheck log (§20.8 step 6). They stay open, not passed."""
    log, expected = args
    reports, summaries = [], []
    for line in Path(log).read_text(errors='replace').splitlines():
        if not line.startswith('========='):
            continue
        text = line[len('========='):].strip()
        if not text or text == 'COMPUTE-SANITIZER' or text.startswith('Saved host backtrace'):
            continue
        if text == OFA_API_ERROR:
            reports.append([])
        elif text.startswith('Host Frame:') and reports:
            reports[-1].append(text)
        elif text.startswith('ERROR SUMMARY:'):
            summaries.append(text)
        else:
            raise RuntimeError(f'memcheck report outside the OFA exception: {text}')
    match = re.fullmatch(r'ERROR SUMMARY: (\d+) errors?', summaries[0]) if len(summaries) == 1 else None
    if not match:
        raise RuntimeError(f'expected one complete error summary, found {summaries}')
    if int(match[1]) != len(reports):
        raise RuntimeError(f'{match[1]} errors counted, {len(reports)} printed')
    if len(reports) != int(expected):
        raise RuntimeError(f'{len(reports)} NVOF errors, expected {expected}')
    if not all(any('libnvidia-opticalflow.so.1' in frame for frame in frames) for frames in reports):
        raise RuntimeError('cuArrayGetPlane error whose backtrace does not pass through the NVOF library')
    print(f'memcheck stays open, not passed: {len(reports)} NVOF-internal cuArrayGetPlane errors as expected; no other API or device error')


if __name__ == '__main__':
    try:
        {'score': score, 'throughput': throughput, 'overlap': overlap,
         'memcheck-ofa': memcheck_ofa}[sys.argv[1]](sys.argv[2:])
    except (RuntimeError, sqlite3.Error, KeyError, ValueError, StopIteration) as error:
        print(f'acceptance: {error}', file=sys.stderr)
        sys.exit(1)
