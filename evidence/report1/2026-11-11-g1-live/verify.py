#!/usr/bin/env python3
"""Read-only receipt, full payload, and raw-statistics verification; no GPU calls."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / 'summary.json').read_text())
checksums = (ROOT / 'SHA256SUMS').read_text().splitlines()
for line in checksums:
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
packets = list((ROOT / 'raw').glob('*.bin.gz'))
assert len(packets) == summary['sessions'] == 8
reference = gzip.decompress(packets[0].read_bytes())
for path in packets:
    data = gzip.decompress(path.read_bytes())
    assert data == reference
    assert len(data) == summary['packet_bytes_each']
    assert hashlib.sha256(data).hexdigest() == summary['packet_sha256']
for style in ('single', 'queued'):
    samples = sorted(float(row['enqueue_to_receive_us']) / 1000
        for path in (ROOT / 'raw').glob('*-' + style + '.delivery.csv')
        for row in csv.DictReader(path.open()) if row['phase'] == 'measured')
    values = summary['single_latency_ms' if style == 'single' else 'queued_delivery_latency_ms']
    assert len(samples) == values['n'] == 480
    for field, quantile in [('p50', .5), ('p95', .95), ('p99', .99)]:
        assert samples[math.ceil(len(samples) * quantile) - 1] == values[field]
    assert statistics.fmean(samples) == values['mean']
    assert statistics.median(samples) == values['median']
    assert samples[0] == values['min'] and samples[-1] == values['max']
for kind in ('source', 'inputs', 'binaries'):
    assert json.loads((ROOT / (kind + '-before.json')).read_text()) == json.loads(
        (ROOT / (kind + '-after.json')).read_text())
identities = list((ROOT / 'raw').glob('*.identity.txt'))
assert len(identities) == 8
for path in identities:
    text = path.read_text()
    assert 'source_commit=' + summary['contract']['base_commit'] + '\n' in text
    assert 'source_dirty=false\n' in text and 'source_dirty=true\n' not in text
    assert 'motion_profile=v1-motion/2\n' in text and 'result_slots=4\n' in text
    assert 'kalman.nsa_enabled=0\n' in text and 'rows.measurement_box=0\n' in text
print(f'PASS: {len(checksums)} checksums, 8 full gzip packets, 480+480 samples and statistics, '
      'unchanged source/input/binary custody, and 8 clean runtime identities')
