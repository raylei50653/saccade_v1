#!/usr/bin/env python3
"""Offline re-derive v4 results from raw logs/outputs/identity records. No GPU."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RECORD = re.compile(
    r'Uninitialized __global__ memory read of size (\d+) bytes\n'
    r'=+\s+at .*?rgba2rgb<\(nvjpegOutputFormat_t\)3,.*?\+0x170\n'
    r'=+\s+by thread \((\d+),(\d+),(\d+)\) in block \((\d+),(\d+),(\d+)\)\n'
    r'=+\s+Address (0x[0-9a-fA-F]+)')
MODES = [('none-yes', 'none', 'yes', 2835), ('rgb-yes', 'rgb', 'yes', 945),
         ('alpha-yes', 'alpha', 'yes', 2835), ('caller-yes', 'caller', 'yes', 2835),
         ('rgb-no', 'rgb', 'no', 2835)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def check_archives():
    result = []
    for name in ['capture-v4', 'attempts/v2-overflow', 'attempts/v3-classifier']:
        root = HERE / name
        manifest = load(root / 'ARCHIVE.json')
        for item in manifest['files']:
            encoded = (root / item['archive_relative']).read_bytes()
            assert sha(encoded) == item['archive_sha256'], item['archive_relative']
            data = gzip.decompress(encoded) if item['compression'] else encoded
            assert sha(data) == item['original_sha256'], item['source_relative']
            assert len(data) == item['original_bytes'], item['source_relative']
        result.append({'directory': name, 'verified_files': len(manifest['files'])})
    return result


def verify_final(root):
    origin = Path(load(root / 'ARCHIVE.json')['source'])
    prepared = load(root / 'prepared-identity.json')
    pins = prepared['files']
    for relative in ['annotation.cu', 'nvtx_mark.hpp', 'capture.py',
                     *['inputs/' + p.name for p in sorted((root / 'inputs').iterdir())]]:
        assert sha((root / relative).read_bytes()) == pins[str(origin / relative)], relative
    starts = list(root.glob('gpu-start-*.json'))
    finals = list(root.glob('gpu-result-*.json'))
    assert len(starts) == len(finals) == 2
    for p in starts:
        d = load(p)
        assert d['files'] == pins
        assert d['executing_runner']['sha256'] == sha((root / 'capture.py').read_bytes())
    for p in finals:
        d = load(p)
        assert d['failure'] is None and d['identity_unchanged'] is True
        assert d['files_after'] == pins
    assert (root / 'capture/compile.exit').read_text().strip() == '0'
    for letter in 'AB':
        assert (root / f'capture/djpeg-{letter}.exit').read_text().strip() == '0'
        ppm = (root / f'inputs/{letter}.ppm').read_bytes()
        m = re.match(rb'P6\s+35\s+27\s+255\n', ppm)
        assert m and len(ppm)-m.end() == 2835
        packed = ppm[m.end():]
        assert packed[::3]+packed[1::3]+packed[2::3] == (root / f'inputs/{letter}.rgb').read_bytes()
    assert (root / 'inputs/A.jpg').read_bytes() != (root / 'inputs/B.jpg').read_bytes()
    assert (root / 'inputs/A.rgb').read_bytes() != (root / 'inputs/B.rgb').read_bytes()
    expected_ids = {}
    facts = []
    for label, mode, nvtx, errors in MODES:
        case = root / label
        tool = gzip.decompress((case / 'tool.log.gz').read_bytes()).decode()
        stdout = (root / f'capture/{label}.stdout').read_text()
        stderr = (root / f'capture/{label}.stderr').read_text()
        command = load(root / f'capture/{label}.command.json')
        argv = command['argv']
        assert argv[argv.index('--nvtx')+1] == nvtx
        assert argv[argv.index('--print-limit')+1] == '0'
        assert argv[argv.index('--show-backtrace')+1] == 'yes'
        assert command['timeout_seconds'] == 60 and command['mem_available_kib'] >= 4194304
        assert int((root / f'capture/{label}.exit').read_text()) == 99
        assert f'diagnostic mode={mode} dimensions=35x27' in stdout
        assert 'result=complete warmup_fail_preserved=1' in stdout
        assert 'result=failed' not in stderr
        assert all(int(x) == 0 for x in re.findall(r'(?:cuda|driver|nvjpeg)_status=(-?\d+)', stderr))
        assert 'skipped' not in tool and 'overflow' not in tool and 'Target application returned an error' not in tool
        summary = re.findall(r'^========= ERROR SUMMARY: (\d+) errors$', tool, re.M)
        assert summary == [str(errors)], (label, summary)
        records = RECORD.findall(tool)
        assert len(records) == errors == len(re.findall(r'^========= Uninitialized ', tool, re.M))
        found = re.findall(r'^surface point=(\w+) repeat=(\d+) source=(0x[0-9a-f]+) base=(0x[0-9a-f]+) bytes=(\d+) pitch=(\d+) id=(\d+)$', stdout, re.M)
        assert len(found) == 5
        assert [x[0] for x in found] == ['discovered', 'before_annotation', 'after_decode', 'before_annotation', 'after_decode']
        identity = [x[2:] for x in found]
        assert len(set(identity)) == 1
        source, base, size, pitch, buffer_id = identity[0]
        source, base, size, pitch, buffer_id = int(source,16),int(base,16),int(size),int(pitch),int(buffer_id)
        assert buffer_id > 0 and pitch == 512 and size == 16384
        assert base <= source and source-base+26*pitch+4*35 <= size
        expected_ids[label] = buffer_id
        # Every reported batch covers the complete frozen frame exactly once.
        # RGB-marked repeats have no records and are not silently reconstructed.
        for start in range(0, len(records), 945):
            pixels = set()
            for size_read, tx, ty, tz, bx, by, bz, address in records[start:start+945]:
                tx,ty,tz,bx,by,bz = map(int,(tx,ty,tz,bx,by,bz))
                assert size_read == '3' and tz == bz == 0 and tx < 32 and ty < 8
                x,y = 32*bx+tx,8*by+ty
                assert 0 <= x < 35 and 0 <= y < 27
                assert int(address,16) == source+y*pitch+4*x
                assert (x,y) not in pixels
                pixels.add((x,y))
            assert len(pixels) == 945
        checkpoints = re.findall(r'^tool_checkpoint repeat=(\d+) complete_records=(\d+) uninitialized_headers=(\d+) bytes=(\d+) ', stdout, re.M)
        totals = [945,945,945] if errors == 945 else [945,1890,2835]
        assert [(int(r),int(c),int(h)) for r,c,h,_ in checkpoints] == [(i,n,n) for i,n in enumerate(totals)]
        annotations = re.findall(r'^annotation repeat=(\d+) mode=(\w+) ranges=(\d+) bytes=(\d+) changes_device_bytes=0 acknowledged=unknown$', stdout, re.M)
        count, marked = {'none':(0,0),'rgb':(945,2835),'alpha':(945,945),'caller':(1,2835)}[mode]
        assert annotations == [(str(i),mode,str(count),str(marked)) for i in (1,2)]
        assert stderr.count('api=annotation_diagnostic_sync cuda_status=0') == 2
        outputs = re.findall(r'^output repeat=(\d+) input=([AB]) rgb_exact=(\d+) differences=(\d+)$', stdout, re.M)
        assert outputs == [(str(i),letter,'1','0') for i,letter in enumerate('ABA')]
        for i, letter in enumerate('ABA'):
            assert (case / f'repeat-{i}.rgb').read_bytes() == (root / f'inputs/{letter}.rgb').read_bytes()
        maps_paths = set()
        for line in (case / 'loaded.maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields)==6 and fields[5].startswith('/') and any(s in fields[5] for s in ('libnvjpeg','libcuda.so')):
                maps_paths.add(fields[5])
        hashes = load(case / 'analysis.json')['loaded_modules']
        assert maps_paths == set(hashes)
        assert any('libnvjpeg' in p for p in hashes) and any('/drivers/' in p for p in hashes)
        assert all(pins[p] == h for p,h in hashes.items())
        facts.append({'label': label, 'actual_exit': 99, 'total_errors': errors,
                      'warmup_errors':945,'postwarmup_errors': errors-945,
                      'full_coordinate_batches': len(records)//945,
                      'all_three_RGB_outputs_exact': True,
                      'allocation_generation_stable': True})
    return {'controls': facts, 'buffer_ids_per_process': expected_ids,
            'source_sha256': sha((root/'annotation.cu').read_bytes()),
            'binary_sha256_recorded_before_after': pins[str(origin/'annotation')],
            'classification': 'data-preserving diagnostic intervention; original FAIL retained',
            'gpu_execution': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, default=HERE / 'capture-v4')
    args = parser.parse_args()
    archives = check_archives()
    final = verify_final(args.capture)
    # Failed attempts remain failed, including the original classifier decision.
    old = HERE / 'attempts/v2-overflow'
    overflow = gzip.decompress((old/'none-yes/tool.log.gz').read_bytes()).decode()
    assert 'Error buffer overflow' in overflow and '689 errors were skipped' in overflow
    assert 'result=failed' in (old/'capture/none-yes.stderr').read_text()
    rejected = load(HERE/'attempts/v3-classifier/none-yes/analysis.json')
    assert rejected['application_complete'] and not rejected['informative_diagnostic']
    assert not rejected['mapped_library_identity_matches_pins']
    print(json.dumps({'archive_checks':archives, 'raw_rederived':final,
                      'failed_attempts_preserved':True}, indent=2))


if __name__ == '__main__':
    main()
