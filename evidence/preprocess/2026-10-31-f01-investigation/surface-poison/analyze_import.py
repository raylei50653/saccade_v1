#!/usr/bin/env python3
"""Offline checks of imported scratch diagnostics; never executes the harness."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'
REFERENCE = ROOT.parent / 'baseline.rgb'


def analyze_log(path):
    text = path.read_text()
    repeat = None
    phase_counts = {}
    attributes = []
    queries = []
    for line in text.splitlines():
        match = re.match(r'phase=decode repeat=(\d+)', line)
        if match:
            repeat = int(match[1])
            phase_counts[str(repeat)] = Counter()
        kind = None
        if line.startswith('========= Uninitialized __global__ memory read'):
            kind = 'uninitialized_global_read'
        elif line.startswith('========= Host API memory access error'):
            kind = 'host_api_memory_access'
        if repeat is not None and kind:
            phase_counts[str(repeat)][kind] += 1
        if line.startswith('surface_attr '):
            attributes.append(line)
        if line.startswith(('surface_query ', 'surface_poisoned ')):
            queries.append(line)
    exit_path = path.with_suffix('.exit')
    return {
        'file': str(path.relative_to(ROOT)),
        'recorded_exit': int(exit_path.read_text()) if exit_path.exists() else None,
        'summary_errors': [int(v) for v in re.findall(r'ERROR SUMMARY: (\d+) errors?\s*$', text, re.M)],
        'printed_errors_by_repeat': phase_counts,
        'unprinted_errors': [int(v) for v in re.findall(r'ERROR SUMMARY: (\d+) errors were not printed', text)],
        'attributes': attributes,
        'surface_actions': queries,
        'identity': [line for line in text.splitlines() if line.startswith('nvjpeg=')],
        'results': [line for line in text.splitlines() if line.startswith(('result=', 'repeat='))],
    }


def analyze_dump(path):
    data = path.read_bytes()
    ref = REFERENCE.read_bytes()
    assert len(data) == 16384 and len(ref) == 2835
    poison = 0x5a if '5a' in str(path.relative_to(RAW)) else 0xa5
    valid_offsets = set()
    rgb_matches = 0
    alpha_zero = 0
    for y in range(27):
        for x in range(35):
            offset = y * 512 + 4 * x
            pixel = y * 35 + x
            valid_offsets.update(range(offset, offset + 4))
            rgb_matches += data[offset:offset+3] == bytes(ref[c*945+pixel] for c in range(3))
            alpha_zero += data[offset+3] == 0
    outside = [v for i, v in enumerate(data) if i not in valid_offsets]
    remaining_poison = outside.count(poison)
    trailing_padding = sum(data[y*512+256:(y+1)*512].count(poison) for y in range(32))
    return {'file': str(path.relative_to(ROOT)), 'bytes': len(data), 'poison': poison,
            'rgb_pixels_matching_reference': rgb_matches, 'alpha_zero_pixels': alpha_zero,
            'outside_valid_rgba_bytes': len(outside), 'outside_bytes_equal_poison': remaining_poison,
            'outside_bytes_different_from_poison': len(outside)-remaining_poison,
            'trailing_256_bytes_per_row_equal_poison': trailing_padding}


def analyze_trace():
    path = RAW / 'nsys-rgb.sqlite'
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute('SELECT name FROM sqlite_master WHERE type="table"')}
        copies = [dict(row) for row in conn.execute('SELECT c.start,c.end,c.streamId,c.correlationId,c.bytes,k.label AS copy_kind FROM CUPTI_ACTIVITY_KIND_MEMCPY AS c JOIN ENUM_CUDA_MEMCPY_OPER AS k ON k.id=c.copyKind ORDER BY c.start')]
        kernels = [dict(row) for row in conn.execute('SELECT k.start,k.end,k.streamId,k.correlationId,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s ON s.id=k.demangledName ORDER BY k.start')]
        metadata = {row[0]: row[1] for row in conn.execute('SELECT name,value FROM META_DATA_CAPTURE')}
    return {'file': str(path.relative_to(ROOT)), 'memcopies': copies, 'kernels': kernels,
            'memset_activity_table_present': 'CUPTI_ACTIVITY_KIND_MEMSET' in tables,
            'capture_metadata': metadata,
            'fixture_jpeg_bytes': (ROOT.parents[3] / 'fixtures/preprocess/baseline.jpg').stat().st_size}


def main():
    manifest = json.loads((ROOT / 'IMPORT.json').read_text())
    for entry in manifest['files']:
        path = RAW / entry['path']
        assert path.stat().st_size == entry['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    logs = [analyze_log(path) for path in sorted((RAW/'capture').glob('*.log'))]
    dumps = [analyze_dump(path) for path in sorted((RAW/'dump').rglob('*.bin'))]
    assert len(dumps) == 6
    assert all(d['rgb_pixels_matching_reference'] == 945 and d['alpha_zero_pixels'] == 945 for d in dumps)
    assert all(d['trailing_256_bytes_per_row_equal_poison'] == 8192 for d in dumps)
    selected = {'attr-a5-initcheck.log': 945,
                'poison-y-preallocate-initcheck.log': 54,
                'poison-rgb-no-preallocate-surface-initcheck.log': 945}
    for name, count in selected.items():
        record = next(item for item in logs if Path(item['file']).name == name)
        assert record['summary_errors'] == [count] and not record['unprinted_errors']
        assert sum(record['printed_errors_by_repeat']['0'].values()) == count
        assert not record['printed_errors_by_repeat']['1'] and not record['printed_errors_by_repeat']['2']
    outputs = []
    for path in sorted(RAW.glob('*.bin')):
        data = path.read_bytes()
        outputs.append({'file': str(path.relative_to(ROOT)), 'bytes': len(data),
                        'matches_planar_rgb_reference': data == REFERENCE.read_bytes(),
                        'sha256': hashlib.sha256(data).hexdigest()})
    output = {'schema': 'f01-surface-poison-import-analysis/1',
              'scope': 'Offline verification of imported artifacts; missing original exits/commands are not inferred.',
              'logs': logs, 'surface_dumps': dumps, 'outputs': outputs, 'nsys': analyze_trace()}
    (ROOT/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'import_files_verified': len(manifest['files']), 'logs': len(logs),
                      'surface_dumps': dumps, 'trace': output['nsys']},indent=2))


if __name__ == '__main__':
    main()
