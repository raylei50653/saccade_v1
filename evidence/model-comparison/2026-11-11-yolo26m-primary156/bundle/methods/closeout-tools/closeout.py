#!/usr/bin/env python3
"""Gate and package completed primary156 evidence. No GPU or scoring is invoked.

Run without --execute to check final prerequisites only. A missing/partial/FAIL
authority exits nonzero before creating an output directory. Publication stages
the entire bundle privately and makes REPORT.md visible only after verification.
"""
from pathlib import Path
import argparse
import csv
import datetime
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent
EVIDENCE = REPO / 'evidence/model-comparison/2026-11-11-yolo26m-primary156'
BUNDLE = EVIDENCE / 'bundle'
PRIOR = REPO / 'build/yolo26m-trt-comparison-20261111'
ARMS = ['official', 'finetuned', 'main']
EXPECTED = {'DanceTrack/all': (65, 67304), 'MOT15/moving': (5, 3185), 'MOT15/other': (6, 2315),
            'MOT17/all': (7, 5316), 'MOT20/all': (4, 8931), 'PP22/moving': (9, 4004), 'PP22/static': (60, 44448)}
TEXT_SUFFIXES = {'.py', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.cmake', '.json', '.csv', '.log', '.txt',
                 '.tsv', '.ini', '.config', '.md', '.status', '.in'}
SKIP_DIRS = {'__pycache__', '.git', 'prepared', 'cells', 'preflight', 'staging'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def check_bound(item):
    path = Path(item['path'])
    require(path.is_file() and sha(path) == item['sha256'], 'bound source changed: ' + str(path))
    if 'bytes' in item:
        require(path.stat().st_size == item['bytes'], 'bound source byte count differs: ' + str(path))


def check_csv(path, expected_rows, fields):
    """Require complete CSV population and every value from final JSON objects."""
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, 'CSV fields differ: ' + str(path))
        actual = list(reader)
    require(len(actual) == len(expected_rows), 'CSV row count differs: ' + str(path))
    def canonical(row, expected=False):
        values = []
        for field in fields:
            value = row[field]
            if field in ('sequence', 'benchmark', 'primary_group', 'group', 'arm'):
                values.append(value)
            else:
                value = float(value)
                require(math.isfinite(value), 'CSV nonfinite value: ' + str(path))
                values.append(value)
        return tuple(values)
    # Sorting also preserves duplicate multiplicity; no row can silently replace another.
    require(sorted(canonical(r) for r in actual) == sorted(canonical(r, True) for r in expected_rows),
            'CSV values differ from final JSON: ' + str(path))


def gate(args):
    selection = load(ROOT / 'selection-runtime.json')
    by = {s['id']: s for s in selection['sequences']}
    groups = selection['primary_groups']
    require(len(by) == len(selection['sequences']) == 156 and sum(s['frames'] for s in by.values()) == 135503, 'selection is not complete156')
    require(set(groups) == set(EXPECTED), 'primary group names differ')
    flat = [name for names in groups.values() for name in names]
    require(len(flat) == len(set(flat)) == 156 and set(flat) == set(by), 'primary groups are not disjoint complete156')
    for name, (count, frames) in EXPECTED.items():
        require(len(groups[name]) == count and sum(by[n]['frames'] for n in groups[name]) == frames, 'group coverage differs: ' + name)
    require(not any(s['benchmark'] == 'SportsMOT' for s in by.values()), 'SportsMOT entered primary selection')
    complete = load(EVIDENCE / 'capture-complete.json')
    require(complete['complete'] is True and complete['cells'] == 468 and complete['frames'] == 406509 and complete['failures'] == [], 'capture-complete not final468 PASS')
    require(not list((ROOT / 'cells').glob('*/*/FAILURE.json')) and not (EVIDENCE / 'failures.json').exists(), 'retained capture FAILURE requires explicit resolution')
    before, after = [load(EVIDENCE / ('capture-binaries-' + k + '.json')) for k in ('before', 'after')]
    require(before == after, 'capture binaries before/after differ')
    for path, item in before.items():
        check_bound({'path': path, **item})
    require(load(ROOT / 'preflight-tools/receipt.json')['status'] == 'PASS', 'initial input preflight not PASS')
    inputs = load(args.inputs_after)
    require(inputs['status'] == 'PASS' and inputs['sequence_count'] == 156 and inputs['jpeg_frames'] == 135503 and
            inputs['GT_seqinfo_models_binary_scope_and_images_unchanged'] is True and
            inputs['inputs_before_sha256'] == sha(ROOT / 'inputs-before.json') and
            inputs['checker_sha256'] == sha(ROOT / 'preflight-tools/verify_inputs.py'), 'full final inputs-after not PASS/bound')
    capture = load(args.capture_audit)
    require(capture['status'] == 'PASS' and capture['mode'] == 'final' and capture['expected_cells'] == 468 and
            capture['completed_cells_verified'] == 468 and capture['pending_cells'] == 0 and
            capture['imported_cells_verified'] == 21 and capture['fresh_cells_verified'] == 447 and
            capture['errors'] == [] and capture['pending'] == [] and capture['SportsMOT_cells'] == 0 and
            capture['all_completed_validation_reports_PASS'] is True and capture['raw_packet_rehash_performed'] is True and
            capture['checker_sha256'] == sha(ROOT / 'preflight-tools/audit_capture.py'), 'final capture audit not complete/bound')
    require(capture['by_arm'] == {a: {'sequences': 156, 'frames': 135503} for a in ARMS}, 'final audit per-arm coverage differs')
    check_bound(capture['source_guards'])
    guards = load(capture['source_guards']['path'])
    for item in guards['files']:
        check_bound(item)
    check_bound(guards['core_before'])
    check_bound(guards['initial_review'])
    for item in capture['final_authorities'].values():
        check_bound(item)
    require(any(Path(item['path']).resolve() == args.inputs_after.resolve() and item['sha256'] == sha(args.inputs_after)
                for item in capture['final_authorities'].values()), 'capture audit is not bound to selected inputs-after')
    index = load(EVIDENCE / 'capture-index.json')
    require(set(index) == {a + '/' + n for a in ARMS for n in by}, 'capture index is not complete468')
    imports = 0
    for key, record in index.items():
        arm, name = key.split('/', 1)
        require(record == load(ROOT / 'cells' / arm / name / 'capture.json') and record['status'] == 'PASS' and record['frames'] == by[name]['frames'], 'capture receipt differs: ' + key)
        require(bool(record['imported']) == (by[name]['benchmark'] == 'MOT17'), 'import/fresh assignment differs: ' + key)
        imports += bool(record['imported'])
        check_bound(record['validation'])
        require(load(record['validation']['path'])['status'] == 'PASS', 'cell validation not PASS: ' + key)
    require(imports == 21, 'imported count differs')
    tools = ROOT / 'scoring-tools'
    freeze = load(tools / 'freeze.json')
    require(sha(tools / 'selection-frozen.json') == sha(ROOT / 'selection-runtime.json') == freeze['selection_sha256'] and
            sha(tools / 'policy-frozen.json') == freeze['policy_sha256'], 'scorer selection/policy changed')
    for path, digest in freeze['core_source_sha256'].items():
        require(sha(path) == digest, 'frozen scorer source changed: ' + path)
    vendor = tools / 'frozen-TrackEval'
    require(sha(vendor / 'SOURCE.json') == freeze['vendor_source_json_sha256'], 'frozen vendor manifest changed')
    manifest = load(vendor / 'SOURCE.json')
    for relative, digest in manifest['unchanged_files_sha256'].items():
        require(sha(vendor / relative) == sha(REPO / 'third_party/TrackEval' / relative) == digest, 'frozen algorithm changed: ' + relative)
    for relative, item in manifest['modified_initializers'].items():
        require(sha(vendor / relative) == item['vendored_sha256'], 'frozen initializer changed: ' + relative)
    scoring = ROOT / 'scoring'
    progress = load(scoring / 'progress.json')
    require(progress['status'] == 'PASS' and progress['sequence_count'] == 156 and progress['frames_per_arm'] == 135503 and
            set(progress['complete_groups']) == set(EXPECTED), 'scoring is incomplete')
    summary = load(scoring / 'summary.json')
    require(summary['result'] == 'PASS' and summary['schema'] == 'primary156-threeway-quality/1' and
            summary['sequence_count'] == 156 and summary['frames_per_arm'] == 135503 and summary['arms'] == ARMS and
            set(summary['primary_groups']) == set(EXPECTED) and len(summary['datasets']) == 5 and
            set(summary['per_sequence']) == set(by) and summary['MOT17_imported_results_match_previous_native_TRT'] is True and
            summary['formal_acceptance'] is False, 'full summary incomplete or wrong scope')
    for group, event in summary['group_events'].items():
        check_bound(event)
        config_path = Path(event['path']).parent / 'config.json'
        require(sha(config_path) == event['config_sha256'], 'group config changed: ' + group)
        for path, digest in load(config_path)['input_hashes'].items():
            require(sha(path) == digest, 'frozen GT/seqinfo/scoring-MOT input changed: ' + path)
    audit = load(args.scoring_audit)
    require(audit['result'] == 'PASS' and audit['primary_groups_checked'] == 7 and audit['dataset_groups_checked'] == 5 and
            audit['sequences_checked'] == 156 and audit['capture_validation_receipts_checked'] == 468 and
            audit['frames_per_arm'] == 135503 and audit['arms'] == ARMS and audit['formal_acceptance'] is False and
            len(audit['context_comparisons']) == 13 and len(audit['per_sequence_deltas']) == 156, 'independent final scoring audit incomplete')
    for key, path in {'selection': tools / 'selection-frozen.json', 'freeze': tools / 'freeze.json',
                      'summary': scoring / 'summary.json', 'raw_components': scoring / 'raw-components.json'}.items():
        require(audit['input_hashes'][key] == sha(path), 'independent audit binding differs: ' + key)
    fields = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'CLR_FP', 'CLR_FN', 'GT_Dets', 'GT_IDs', 'Dets', 'IDs', 'Frames']
    for filename, entries in [('primary-groups.csv', summary['primary_groups']), ('datasets.csv', summary['datasets'])]:
        rows = [{'group': group, 'arm': a, 'sequences': r['sequence_count'], **r['metrics'][a]} for group, r in entries.items() for a in ARMS]
        check_csv(scoring / filename, rows, ['group', 'arm', 'sequences', *fields])
    rows = [{'sequence': name, 'benchmark': r['benchmark'], 'primary_group': r['primary_group'], 'arm': a, **r['metrics'][a]}
            for name, r in summary['per_sequence'].items() for a in ARMS]
    check_csv(scoring / 'per-sequence.csv', rows, ['sequence', 'benchmark', 'primary_group', 'arm', *fields])
    delta_fields = ['sequence', 'benchmark', 'primary_group', 'frames', 'GT_Dets', *['delta_' + k + '_pp' for k in ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA')]]
    rows = [{**{k: r[k] for k in delta_fields if k in r}, **{'delta_' + k + '_pp': v for k, v in r['delta_pp'].items()}}
            for r in audit['per_sequence_deltas']]
    check_csv(args.scoring_audit.with_suffix('.per-sequence.csv'), rows, delta_fields)
    reviewer = load(ROOT / 'validation-tools/scorer-source-review.json')
    require(sha(ROOT / 'validation-tools/audit_primary.py') == reviewer['final_audit_script']['sha256'], 'independent audit source changed since source review')
    containers = [*summary['primary_groups'].values(), *summary['datasets'].values(), summary['pooled_all156']]
    containers += list(summary['per_sequence'].values())
    for row in containers:
        require(set(row['metrics']) == set(ARMS), 'metric arms differ')
        require(len({tuple(row['metrics'][a][k] for k in ('GT_Dets', 'GT_IDs', 'Frames')) for a in ARMS}) == 1, 'GT support differs across arms')
        for metric in row['metrics'].values():
            require(all(math.isfinite(float(v)) for v in metric.values()), 'nonfinite reported metric')
            require(abs(metric['MOTA'] - 100 * (1 - (metric['CLR_FP'] + metric['CLR_FN'] + metric['IDSW']) / max(1, metric['GT_Dets']))) < 2e-10, 'MOTA formula differs')
    delivery = load(PRIOR / 'export-tools/delivery.json')
    require(delivery['screen_failures_preserved'] is True, 'prior numerical FAIL boundary lost')
    for item in delivery['models']:
        if item['arm'] in ('official', 'finetuned'):
            require(item['trt_vs_pt_diagnostic_status'] == 'FAIL' and item['trt_vs_pt_max_raw_rel_l2'] > .01, 'primary FP16 screen relabeled')
            require(all(v['engine_sha256'] == item['engine_sha256'] for k, v in index.items() if k.startswith(item['arm'] + '/')), 'engine differs from prior export: ' + item['arm'])
    require(all(v['engine_sha256'] == delivery['main_frozen_engine']['sha256'] for k, v in index.items() if k.startswith('main/')), 'main engine differs from prior export')
    training = load(HERE / 'training-provenance.json')
    for item in training['authorities'].values():
        check_bound(item)
    require(training['finetuned']['checkpoint_sha256'] == '08be5abae55d26b762e262948ded7d3deb11295aa0509ee04ad3582a2c9774cd' and
            training['main']['checkpoint_sha256'] == '53bccffe9bdbf37bdcfc81d03e95ccbd2d142d67a557ea015b76d356e4fa5f7d', 'report training provenance does not match selected weights')
    authorities = [EVIDENCE / 'capture-complete.json', EVIDENCE / 'capture-index.json', args.inputs_after,
                   args.capture_audit, args.scoring_audit, scoring / 'summary.json', scoring / 'raw-components.json',
                   tools / 'freeze.json', PRIOR / 'export-tools/delivery.json', HERE / 'training-provenance.json',
                   scoring / 'primary-groups.csv', scoring / 'datasets.csv', scoring / 'per-sequence.csv',
                   args.scoring_audit.with_suffix('.per-sequence.csv')]
    return {'selection': selection, 'summary': summary, 'audit': audit, 'delivery': delivery, 'index': index,
            'authority_sha256': {str(p.resolve()): sha(p) for p in authorities}}


def copy_file(source, target):
    source, target = Path(source), Path(target)
    require(source.is_file() and not source.is_symlink(), 'refuse missing/symlink metadata: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), 'duplicate destination: ' + str(target))
    digest = sha(source)
    shutil.copy2(source, target)
    require(sha(target) == digest == sha(source), 'copy hash mismatch: ' + str(source))


def copy_text_tree(source, target, skip=SKIP_DIRS):
    for path in sorted(Path(source).rglob('*')):
        rel = path.relative_to(source)
        if any(part in skip or part.startswith('staging-') or part.startswith('selfcheck-') for part in rel.parts):
            continue
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name in ('LICENSE', 'CMakeLists.txt')):
            copy_file(path, Path(target) / rel)


def archive(source, destination, expected):
    source = Path(source)
    state = source.stat()
    require(not source.is_symlink(), 'refuse symlink capture output: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = hashlib.sha256()
    size = 0
    with source.open('rb') as incoming, destination.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=6, mtime=0) as outgoing:
            for block in iter(lambda: incoming.read(8 << 20), b''):
                original.update(block)
                size += len(block)
                outgoing.write(block)
    require(original.hexdigest() == expected, 'original output SHA differs from final capture: ' + str(source))
    restored = hashlib.sha256()
    restored_size = 0
    with gzip.open(destination, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            restored.update(block)
            restored_size += len(block)
    require(restored.hexdigest() == expected and restored_size == size == state.st_size, 'gzip uncompressed SHA/size mismatch: ' + str(source))
    require(source.is_file() and source.stat().st_size == state.st_size and source.stat().st_mtime_ns == state.st_mtime_ns,
            'source capture changed during compression: ' + str(source))
    return {'source': str(source), 'source_sha256': expected, 'uncompressed_bytes': size,
            'gzip_bytes': destination.stat().st_size, 'gzip_sha256': sha(destination),
            'verified_uncompressed_sha256': restored.hexdigest(), 'verified_uncompressed_bytes': restored_size,
            'source_preserved': True, 'status': 'PASS'}


def all_files(folder):
    result = []
    for path in folder.rglob('*'):
        require(not path.is_symlink(), 'symlink in evidence bundle: ' + str(path))
        if path.is_file() and path != folder / 'SHA256SUMS':
            result.append(path)
    return sorted(result)


def write_manifest(folder):
    entries = [(sha(p), p.relative_to(folder).as_posix()) for p in all_files(folder)]
    (folder / 'SHA256SUMS').write_text(''.join(d + '  ' + n + '\n' for d, n in entries))
    return verify_manifest(folder)


def verify_manifest(folder):
    listed = {}
    for line in (folder / 'SHA256SUMS').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        p = Path(relative)
        require(not p.is_absolute() and '..' not in p.parts and relative not in listed, 'invalid manifest path: ' + relative)
        require(len(digest) == 64 and sha(folder / relative) == digest, 'manifest SHA mismatch: ' + relative)
        listed[relative] = digest
    require(set(listed) == {p.relative_to(folder).as_posix() for p in all_files(folder)}, 'manifest file-set differs')
    return {'result': 'PASS', 'files_verified': len(listed), 'manifest_sha256': sha(folder / 'SHA256SUMS'),
            'manifest_excludes_only_itself': True, 'all_listed_hashes_match': True, 'complete_file_set_matches': True}


def execute(args, final):
    require(not BUNDLE.exists(), 'refuse to overwrite a published closeout bundle')
    receipt = HERE / 'closeout-receipt.json'
    require(not receipt.exists(), 'refuse to overwrite closeout receipt')
    require(HERE.stat().st_dev == EVIDENCE.stat().st_dev, 'staging and final bundle must share a filesystem for atomic directory rename')
    stage = Path(tempfile.mkdtemp(prefix='staging-', dir=HERE))
    save(stage / 'CLOSEOUT-GATES.json', {'result': 'PASS', 'completed_utc': now(), 'capture_cells': 468,
         'frames_total': 406509, 'imported_cells': 21, 'fresh_cells': 447, 'authority_sha256': final['authority_sha256'],
         'GPU_work': False, 'formal_acceptance': False, 'source_outputs_preserved': True})
    # Snapshot existing PLAN/provenance; all live study authority files stay in place.
    for path in sorted(EVIDENCE.iterdir()):
        require(path.is_file() and path.suffix in TEXT_SUFFIXES, 'unexpected pre-existing evidence payload: ' + str(path))
        copy_file(path, stage / path.name)
    copy_file(ROOT / 'inputs-before.json', stage / 'inputs-before.json')
    copy_file(ROOT / 'selection-runtime.json', stage / 'selection-runtime.json')
    copy_file(HERE / 'training-provenance.json', stage / 'training-provenance.json')
    for name in ('capture-driver.log', 'scoring-driver.log', 'gpu-telemetry.csv', 'progress.json'):
        copy_file(ROOT / name, stage / 'logs' / name)
    copy_file(ROOT / 'run_capture.py', stage / 'methods/run_capture.py')
    for name in ('preflight-tools', 'validation-tools', 'scoring-tools'):
        copy_text_tree(ROOT / name, stage / 'methods' / name)
    copy_text_tree(HERE, stage / 'methods/closeout-tools', SKIP_DIRS | {stage.name})
    copy_text_tree(ROOT / 'scoring', stage / 'metrics')
    for source, name in [(args.inputs_after, 'inputs-after.json'), (args.capture_audit, 'final-capture-audit.json'),
                         (args.scoring_audit, 'scoring-audit.json')]:
        copy_file(source, stage / 'audits' / name)
    copy_file(args.scoring_audit.with_suffix('.per-sequence.csv'), stage / 'audits/scoring-audit.per-sequence.csv')
    # Prior artifact identity/diagnostics and research source only; never copy an engine or weight.
    copy_file(PRIOR / 'export-tools/delivery.json', stage / 'prior-TRT-characterization/export-delivery.json')
    for name in ('build.json', 'build-attempt-01.json', 'harness-review.json'):
        copy_file(PRIOR / 'pipeline-tools' / name, stage / 'prior-TRT-characterization' / name)
    for name in ('model_trt.cpp', 'yolo_producer.cu', 'yolo_producer_cuda.hpp', 'tracking_session.cpp', 'build_adapter.py', 'preflight.py'):
        path = PRIOR / 'pipeline-tools' / name
        if path.exists():
            copy_file(path, stage / 'methods/prior-native-adapter' / name)
    copy_file(PRIOR / 'pipeline-tools/preflight/receipt.json', stage / 'prior-TRT-characterization/adapter-preflight.json')
    archives = {}
    for index, (key, record) in enumerate(sorted(final['index'].items()), 1):
        folder = ROOT / 'cells' / key
        dest = stage / 'outputs' / key
        for name, field in [('mot.txt', 'mot_sha256'), ('packets.bin', 'packets_sha256'), ('motion.scmo', 'motion_sha256')]:
            target = dest / (name + '.gz')
            archives[target.relative_to(stage).as_posix()] = archive(folder / name, target, record[field])
        for name in ('capture.json', 'resolved.config', 'track.log', 'validation.log', 'command.json', 'exit.json'):
            source = folder / name
            if source.exists():
                copy_file(source, dest / name)
        require(sha(folder / 'resolved.config') == record['resolved_config_sha256'], 'cell resolved config changed: ' + key)
        copy_file(record['validation']['path'], stage / 'audits/cells' / (key + '.json'))
        if index % 12 == 0 or index == 468:
            print(json.dumps({'phase': 'gzip-verified', 'cells': index, 'of': 468}), flush=True)
    require(len(archives) == 1404, 'incomplete archive count')
    save(stage / 'OUTPUT-ARCHIVES.json', {'result': 'PASS', 'archives': archives, 'archive_count': len(archives),
         'source_files_preserved': True, 'every_uncompressed_sha256_matches_capture': True})
    # Recheck all final authority bindings before exposing any finished report.
    for path, digest in final['authority_sha256'].items():
        require(sha(path) == digest, 'final authority changed during packaging: ' + path)
    spec = importlib.util.spec_from_file_location('primary156_report', HERE / 'render_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.render(final['summary'], final['audit'], final['delivery'], len(archives))
    (stage / 'REPORT.md').write_text(report)
    forbidden = {'.engine', '.onnx', '.pt', '.pth', '.jpg', '.jpeg', '.png', '.bmp', '.o', '.so', '.a'}
    require(not any(p.suffix.lower() in forbidden for p in stage.rglob('*') if p.is_file()), 'forbidden model/image/binary in payload')
    for relative in re.findall(r'\]\(([^)]+)\)', report):
        require((stage / relative).is_file() or relative == 'SHA256SUMS', 'report link target missing: ' + relative)
    staged_review = write_manifest(stage)
    # One rename exposes the complete verified directory, including report and manifest.
    # It never moves/deletes original capture authority files in the study evidence root.
    require(not BUNDLE.exists(), 'final bundle appeared during packaging; staging preserved')
    os.rename(stage, BUNDLE)
    review = verify_manifest(BUNDLE)
    require(review == staged_review, 'published bundle differs from verified stage')
    save(receipt, {**review, 'completed_utc': now(), 'evidence': str(BUNDLE), 'study_evidence_root': str(EVIDENCE),
                  'atomic_directory_publish': True, 'staging_before_rename': str(stage),
                  'gzip_archives_verified': len(archives), 'source_outputs_preserved': True,
                  'authority_sha256': final['authority_sha256'], 'formal_acceptance': False})
    print(json.dumps({'result': 'PASS', 'receipt': str(receipt), **review}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='package only after every final gate passes')
    parser.add_argument('--inputs-after', type=Path, default=ROOT / 'preflight-tools/inputs-after.json')
    parser.add_argument('--capture-audit', type=Path, default=ROOT / 'preflight-tools/final-capture-audit.json')
    parser.add_argument('--scoring-audit', type=Path, default=ROOT / 'validation-tools/scoring-audit.json')
    parser.add_argument('--verify-bundle', type=Path, help='read-only full SHA256SUMS/file-set verification')
    args = parser.parse_args()
    if args.verify_bundle:
        require(not args.execute, '--verify-bundle cannot be combined with --execute')
        print(json.dumps(verify_manifest(args.verify_bundle.resolve()), indent=2))
        return
    args.inputs_after = args.inputs_after.resolve()
    args.capture_audit = args.capture_audit.resolve()
    args.scoring_audit = args.scoring_audit.resolve()
    final = gate(args)
    print(json.dumps({'result': 'PASS_FINAL_GATES_ONLY', 'execute': args.execute,
                      'capture_cells': 468, 'sequence_count': 156, 'frames_per_arm': 135503}), flush=True)
    if args.execute:
        execute(args, final)


if __name__ == '__main__':
    main()
