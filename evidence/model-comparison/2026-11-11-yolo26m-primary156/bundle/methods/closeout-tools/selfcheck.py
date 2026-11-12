"""Bounded CPU-only positive/negative checks; never package live study outputs."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import tempfile
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rejected(action):
    try:
        action()
    except (RuntimeError, FileNotFoundError):
        return True
    raise RuntimeError('negative control was incorrectly accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'selfcheck.json')
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(HERE)
    assert not args.output.exists(), 'refuse overwrite selfcheck receipt'
    c = module('closeout')
    report = module('render_report')
    checks = {}
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    checks['source_AST'] = True
    with tempfile.TemporaryDirectory(prefix='selfcheck-', dir=HERE) as temp:
        folder = Path(temp)
        original = folder / 'original.bin'
        original.write_bytes(bytes(range(256)) * 4097)
        expected = hashlib.sha256(original.read_bytes()).hexdigest()
        rec = c.archive(original, folder / 'packet.gz', expected)
        assert rec['status'] == 'PASS' and rec['verified_uncompressed_sha256'] == expected
        assert original.exists() and c.sha(original) == expected
        checks['gzip_full_readback_SHA_bytes_and_source_retention'] = True
        checks['wrong_capture_SHA_rejected'] = rejected(lambda: c.archive(original, folder / 'bad.gz', '0' * 64))
        bundle = folder / 'bundle'
        (bundle / 'nested').mkdir(parents=True)
        (bundle / 'nested/SHA256SUMS').write_text('nested payload must also be covered\n')
        (bundle / 'normal.txt').write_text('payload\n')
        assert c.write_manifest(bundle)['files_verified'] == 2
        (bundle / 'nested/SHA256SUMS').write_text('tampered\n')
        checks['nested_SHA256SUMS_is_covered_and_tamper_rejected'] = rejected(lambda: c.verify_manifest(bundle))
        c.write_manifest(bundle)
        (bundle / 'unlisted.txt').write_text('unlisted\n')
        checks['unlisted_payload_rejected'] = rejected(lambda: c.verify_manifest(bundle))
        csvpath = folder / 'rows.csv'
        fields = ['sequence', 'arm', 'HOTA', 'GT_Dets']
        rows = [{'sequence': 'a', 'arm': 'main', 'HOTA': 71.23123456789013, 'GT_Dets': 100},
                {'sequence': 'a', 'arm': 'finetuned', 'HOTA': 70.1, 'GT_Dets': 100}]
        def write(rows_to_write):
            with csvpath.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows_to_write)
        write(rows)
        c.check_csv(csvpath, rows, fields)
        checks['CSV_complete_JSON_values_match'] = True
        write(rows[:1])
        checks['partial_CSV_rejected'] = rejected(lambda: c.check_csv(csvpath, rows, fields))
        write([rows[0], rows[0]])
        checks['duplicate_CSV_rows_rejected'] = rejected(lambda: c.check_csv(csvpath, rows, fields))
        write([{**r, 'HOTA': r['HOTA'] + 1} for r in rows])
        checks['CSV_value_tamper_rejected'] = rejected(lambda: c.check_csv(csvpath, rows, fields))
        checks['partial_report_rejected'] = rejected(lambda: report.render({'result': 'RUNNING'}, {'result': 'PASS'}, {}, 0))
        # A single same-filesystem rename publishes the already verified payload.
        c.write_manifest(bundle)
        destination = folder / 'published'
        bundle.rename(destination)
        assert (destination / 'SHA256SUMS').exists() and not bundle.exists()
        assert c.verify_manifest(destination)['result'] == 'PASS'
        checks['single_directory_rename_publishes_complete_verified_bundle'] = True
    payload = {'result': 'PASS', 'scope': 'bounded synthetic CPU helper checks only; not full156 packaging or final authority acceptance',
               'checks': checks, 'GPU_used': False, 'full_packager_executed': False,
               'source_sha256': {p.name: c.sha(p) for p in HERE.glob('*.py')}}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'result': 'PASS', 'checks': len(checks), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
