#!/usr/bin/env python3
"""Read-only verification of full offline process E2E receipts; no GPU calls."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT/'summary.json').read_text())
checksums = (ROOT/'SHA256SUMS').read_text().splitlines()
for line in checksums:
    expected,name=line.split('  ',1)
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
for filename,info in summary['outputs'].items():
    values=[]
    for ordinal in range(1,4):
        path=ROOT/'raw'/f'run{ordinal}'/filename
        data=gzip.decompress(path.with_name(path.name+'.gz').read_bytes()) if filename in ('packets.bin','motion.scmo') else path.read_bytes()
        assert len(data)==info['bytes_each']
        assert hashlib.sha256(data).hexdigest()==info['sha256']
        values.append(data)
    assert values[0]==values[1]==values[2]
for kind in ('source','inputs','binaries'):
    assert json.loads((ROOT/(kind+'-before.json')).read_text())==json.loads((ROOT/(kind+'-after.json')).read_text())
runs=json.loads((ROOT/'runs.json').read_text())
assert len(runs)==3 and all(r['returncode']==0 and r['app_frames']==654 for r in runs)
for run in runs:
    assert (run['perf_counter_after_ns']-run['perf_counter_before_ns'])/1e6==run['wall_ms']
    text=(ROOT/f'run{run["ordinal"]}.log').read_text()
    assert 'source_dirty=false\n' in text and 'source_dirty=true\n' not in text
    assert 'kalman.nsa_enabled=0\n' in text and 'rows.measurement_box=0\n' in text
values=[r['wall_ms'] for r in runs]
assert statistics.fmean(values)==summary['process_wall_ms']['mean']
assert statistics.median(values)==summary['process_wall_ms']['median']
assert min(values)==summary['process_wall_ms']['min'] and max(values)==summary['process_wall_ms']['max']
assert statistics.fmean([v/654 for v in values])==summary['amortized_process_ms_per_frame']['mean']
assert 1962*1000/sum(values)==summary['aggregate_process_throughput_fps']
print(f'PASS: {len(checksums)} checksums; 3 complete MOT/packet/motion/config payloads; '
      '654-frame process timing receipts and statistics; source/input/binary custody')
