"""Independent live/final coverage and receipt audit for the primary156 campaign.

--live reads completed receipts, resolved configs and small validation reports.
It never rehashes JPEG, GT or raw packets. --final requires all cells plus the
completed full input rehash, and hashes every retained output file once.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote
import argparse
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT.parents[1] / 'evidence/model-comparison/2026-11-11-yolo26m-primary156'
PRIOR = ROOT.parent / 'yolo26m-trt-comparison-20261111'
ARMS = ('main', 'official', 'finetuned')
PACKET_BYTES = 12368
BASELINE = PRIOR / 'quality/main/MOT17-10-SDP/resolved.config'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    assert not path.exists(), f'refuse overwrite {path}'
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def config(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            require(key not in result, f'duplicate config key {key}')
            result[key] = value
    return result


def tracker_output(d):
    return {k: v for k, v in d.items() if k.startswith(('tracker.', 'output.'))}


def source_guards():
    path = HERE / 'final-audit-source-guards.json'
    core_before = read(EVIDENCE / 'capture-binaries-before.json')
    review = read(ROOT / 'validation-tools/runner-review.json')
    require(sha(ROOT / 'run_capture.py') == review['runner_sha256'], 'runner differs from independent initial review')
    require(sha(ROOT / 'validation-tools/validate_cell.py') == review['validator_script_sha256'],
            'validator script differs from independent initial review')
    require(sha(ROOT / 'validation-tools/validate_packets') == review['validator_sha256'],
            'packet validator differs from independent initial review')
    require(sha(ROOT / 'selection-runtime.json') == review['selection_sha256'], 'selection differs from review')
    imported_baseline = read(ROOT / 'cells/main/MOT17-10-SDP/capture.json')
    require(sha(BASELINE) == imported_baseline['resolved_config_sha256'], 'baseline differs from validated prior/imported config')
    selected = [ROOT / 'run_capture.py', ROOT / 'validation-tools/validate_cell.py',
                ROOT / 'validation-tools/validate_packets', BASELINE, ROOT / 'scoring-tools/worker.py',
                HERE / 'verify_inputs.py', HERE / 'check_MOT17_import_inputs.py',
                ROOT / 'selection-runtime.json', ROOT / 'inputs-before.json']
    if path.exists():
        guards = read(path)
        require(sha(guards['initial_review']['path']) == guards['initial_review']['sha256'], 'initial review receipt changed')
        require(sha(guards['core_before']['path']) == guards['core_before']['sha256'], 'capture before authority changed')
        for item in guards['files']:
            require(sha(item['path']) == item['sha256'], f'frozen source/helper changed: {item["path"]}')
        return guards
    guards = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'snapshot_created_during_capture_after_completed_cells': len(list((ROOT / 'cells').glob('*/*/capture.json'))),
              'scope': 'runner and validators also match existing initial review; this additional helper snapshot does not retroactively relabel its creation time',
              'files': [bind(p) for p in selected], 'core_before': bind(EVIDENCE / 'capture-binaries-before.json'),
              'initial_review': bind(ROOT / 'validation-tools/runner-review.json')}
    write(path, guards)
    return guards


def audit_cell(arm, sequence, before, baseline, final):
    seqid = sequence['id']
    folder = ROOT / 'cells' / arm / seqid
    receipt_path = folder / 'capture.json'
    rec = read(receipt_path)
    require(rec['status'] == 'PASS', 'capture status not PASS')
    require(rec['frames'] == sequence['frames'], 'capture frame count differs')
    require(rec['engine_sha256'] == before['models'][arm]['sha256'], 'capture engine differs')
    imported = sequence['benchmark'] == 'MOT17'
    require(rec['imported'] is imported, 'import/fresh selection differs')
    for filename in ('mot.txt', 'packets.bin', 'motion.scmo', 'resolved.config'):
        require((folder / filename).is_file(), f'missing retained {filename}')
    require((folder / 'packets.bin').stat().st_size == sequence['frames'] * PACKET_BYTES,
            'raw packet size/frame count differs')
    expected_validation = ROOT / 'validation-tools/cells' / arm / (seqid + '.json')
    require(Path(rec['validation']['path']) == expected_validation, 'validation path differs')
    require(sha(expected_validation) == rec['validation']['sha256'], 'validation report hash differs')
    val = read(expected_validation)
    require((val['status'], val['arm'], val['sequence'], val['frames'], val['width'], val['height'], val['epoch']) ==
            ('PASS', arm, seqid, sequence['frames'], sequence['width'], sequence['height'], 1), 'validation identity differs')
    for flag in ('tracker_output_matches_frozen_baseline', 'source_and_execution_config_correct'):
        require(val[flag] is True, f'{flag} not true')
    for flag in ('canonical_all_frames', 'all_status_ok', 'all_unique_UID_ID_class0_finite_boxes'):
        require(val['packets'][flag] is True, f'packet {flag} not true')
    require(val['packets']['bytes_raw'] == sequence['frames'] * PACKET_BYTES, 'validated packet byte count differs')
    require(val['packets']['sha256_raw'] == rec['packets_sha256'], 'packet hash differs between capture/validation')
    require(val['packets']['validator_sha256'] == sha(ROOT / 'validation-tools/validate_packets'), 'packet validator identity differs')
    require(val['motion']['frames'] == sequence['frames'] and val['motion']['epoch'] == 1, 'motion count/epoch differs')
    require(val['motion']['all_floats_finite'] and val['motion']['all_frame_references_contiguous'], 'motion validity failed')
    require(val['motion']['sha256_of_file'] == rec['motion_sha256'], 'motion hash differs capture/validation')
    require(val['config_sha256'] == rec['resolved_config_sha256'], 'config hash differs capture/validation')
    identity = val['build_identity']
    require(identity['build.engine_sha256'] == before['models'][arm]['sha256'], 'validated engine differs')
    require(identity['build.executable_sha256'] == before['track_binary']['sha256'], 'validated binary differs')
    require(identity['input.seqinfo_sha256'] == sequence['sequence_info']['sha256'], 'validated seqinfo differs')
    d = config(folder / 'resolved.config')
    require(tracker_output(d) == baseline, 'resolved tracker/output settings differ')
    require(d['tracker.kalman.nsa_enabled'] == 'false' and d['tracker.rows.measurement_box'] == 'false', 'default options differ')
    require((d['implementation.backend'], d['implementation.input'], d['implementation.model_profile'],
             d['implementation.motion_profile']) == ('cuda', 'jpeg', 'hr-1280x768', 'v1-motion/2'), 'native/profile config differs')
    require((d['execution.mode'], d['execution.cuda_graph'], d['execution.candidate_slots'],
             d['execution.motion_slots'], d['execution.result_slots']) == ('pipelined', 'true', '2', '2', '4'), 'execution config differs')
    require((int(d['source.width']), int(d['source.height']), int(d['source.epoch'])) ==
            (sequence['width'], sequence['height'], 1), 'source dimensions/epoch differ')
    frozen_config_hash = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    require(val['frozen_tracker_output_sha256'] == frozen_config_hash, 'validator baseline identity differs')
    # Small output hashes are checked in live mode too; raw packet rehash waits.
    for filename, key in (('mot.txt', 'mot_sha256'), ('motion.scmo', 'motion_sha256'),
                          ('resolved.config', 'resolved_config_sha256')):
        require(sha(folder / filename) == rec[key], f'{filename} changed after capture')
    if final:
        require(sha(folder / 'packets.bin') == rec['packets_sha256'], 'raw packet bytes changed after validation')
    if imported:
        source = rec['source_capture']
        prior_source = read(EVIDENCE.parent / '2026-11-11-yolo26m-trt-performance/quality-capture.json')
        require(source == prior_source[arm + '/' + seqid], 'import source receipt differs from prior quality authority')
        require(source['native_CPU_exact'] is True and source['frames'] == sequence['frames'], 'import lacks prior native/CPU exact proof')
        for key in ('mot_sha256', 'packets_sha256', 'motion_sha256'):
            require(rec[key] == source[key], f'import differs from original {key}')
    else:
        require(rec['exit_code'] == 0 and rec['binary_sha256'] == before['track_binary']['sha256'], 'fresh exit/binary differs')
        expected = ['flock', '/home/ray/.cache/saccade-v1/gpu.lock', before['track_binary']['path'],
                    before['models'][arm]['path'], sequence['inference_path'], str(folder / 'mot.txt'),
                    '--profile', 'hr-1280x768', '--set', 'tracker.kalman.nsa_enabled=false', '--set',
                    'tracker.rows.measurement_box=false', '--packets', str(folder / 'packets.bin'),
                    '--motion', str(folder / 'motion.scmo'), '--resolved-config', str(folder / 'resolved.config')]
        require(rec['command'] == expected, 'fresh command differs from fixed protocol')
        require(read(folder / 'command.json')['argv'] == expected, 'fresh command sidecar differs')
        require(read(folder / 'exit.json')['exit_code'] == 0, 'fresh exit sidecar differs')
    return {'arm': arm, 'sequence': seqid, 'benchmark': sequence['benchmark'], 'frames': sequence['frames'],
            'imported': imported, 'capture': bind(receipt_path), 'validation': bind(expected_validation),
            'mot_sha256': rec['mot_sha256'], 'packets_sha256': rec['packets_sha256'],
            'packet_bytes': sequence['frames'] * PACKET_BYTES, 'motion_sha256': rec['motion_sha256'],
            'resolved_config_sha256': rec['resolved_config_sha256']}, rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--final', action='store_true')
    parser.add_argument('--inputs-after', type=Path, default=HERE / 'inputs-after.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    snapshot = source_guards()
    before = read(ROOT / 'inputs-before.json')
    selection = read(ROOT / 'selection-runtime.json')
    require(sha(ROOT / 'inputs-before.json') == read(HERE / 'receipt.json')['inputs_before']['sha256'], 'before-input authority changed')
    require(sha(ROOT / 'selection-runtime.json') == before['runtime_selection']['sha256'], 'runtime selection changed')
    require(sha(before['policy']['path']) == before['policy']['sha256'], 'scope policy changed')
    require(sha(before['source_primary_selection']['path']) == before['source_primary_selection']['sha256'], 'primary selection changed')
    sequences = {s['id']: s for s in selection['sequences']}
    require(len(sequences) == 156 and sum(s['frames'] for s in sequences.values()) == 135503, 'population differs')
    require(all(s['benchmark'] in ('MOT15','MOT17','MOT20','DanceTrack','PP22') for s in sequences.values()), 'unexpected/Sports benchmark')
    require(len(selection['excluded']['SportsMOT']) == 45, 'Sports exclusion changed')
    baseline = tracker_output(config(BASELINE))
    if args.final:
        early_complete = read(EVIDENCE / 'capture-complete.json')
        require(early_complete['complete'] is True and early_complete['cells'] == 468 and
                early_complete['frames'] == 406509 and not early_complete['failures'],
                'capture is not complete; final packet/input audit must wait')
        require(args.inputs_after.is_file() and read(args.inputs_after)['status'] == 'PASS',
                'full after-campaign JPEG/input guard is not complete')
    completed = []
    raw_receipts = {}
    pending = []
    errors = []
    expected_pairs = {(a, n) for a in ARMS for n in sequences}
    # Freeze the published-receipt set once so concurrent publication cannot
    # create an inconsistent live snapshot while arms are inspected in turn.
    published_pairs = {(p.parent.parent.name, p.parent.name)
                       for p in (ROOT / 'cells').glob('*/*/capture.json')}
    runner_order = [(a, s['id']) for s in sorted(sequences.values(),
                    key=lambda s: (s['benchmark'] != 'MOT17', s['benchmark'], s['id'])) for a in ARMS]
    position = {pair: i for i, pair in enumerate(runner_order)}
    if published_pairs:
        last = max(position[p] for p in published_pairs if p in position)
        earlier_missing = set(runner_order[:last + 1]) - published_pairs
        errors.extend(f'earlier cell missing before already completed later cells: {pair}' for pair in sorted(earlier_missing))
    actual_dirs = {(a.name, p.name) for a in (ROOT / 'cells').iterdir() if a.is_dir()
                   for p in a.iterdir() if p.is_dir()}
    for pair in sorted(actual_dirs - expected_pairs):
        errors.append(f'unexpected cell directory: {pair}')
    failures = list((ROOT / 'cells').glob('*/*/FAILURE.json'))
    errors.extend(f'capture failure retained: {p}' for p in failures)
    for arm in ARMS:
        for seqid, sequence in sequences.items():
            folder = ROOT / 'cells' / arm / seqid
            if (arm, seqid) not in published_pairs:
                pending.append({'arm': arm, 'sequence': seqid, 'directory_exists': folder.exists()})
                continue
            try:
                row, rec = audit_cell(arm, sequence, before, baseline, args.final)
                completed.append(row)
                raw_receipts[arm + '/' + seqid] = rec
            except Exception as error:
                errors.append(f'{arm}/{seqid}: {type(error).__name__}: {error}')
    by_arm = {a: {'sequences': sum(r['arm'] == a for r in completed),
                  'frames': sum(r['frames'] for r in completed if r['arm'] == a)} for a in ARMS}
    imports = sum(r['imported'] for r in completed)
    fresh = len(completed) - imports
    final_authorities = {}
    if args.final:
        require(not pending, f'final audit has {len(pending)} missing/incomplete cells')
        require(not errors, 'final completed-cell audit errors: ' + repr(errors[:5]))
        require(len(completed) == 468 and imports == 21 and fresh == 447, 'final cell/origin counts differ')
        require(all(x == {'sequences': 156, 'frames': 135503} for x in by_arm.values()), 'final per-arm coverage differs')
        core_before = read(EVIDENCE / 'capture-binaries-before.json')
        core_after = read(EVIDENCE / 'capture-binaries-after.json')
        require(core_before == core_after, 'capture core before/after differ')
        for path, info in core_before.items():
            require(Path(path).stat().st_size == info['bytes'] and sha(path) == info['sha256'], f'capture binary/source changed: {path}')
        require(read(EVIDENCE / 'capture-index.json') == raw_receipts, 'aggregate capture index differs from 468 independent receipts')
        capture_complete = read(EVIDENCE / 'capture-complete.json')
        require(capture_complete['complete'] is True and capture_complete['cells'] == 468 and
                capture_complete['frames'] == 406509 and capture_complete['failures'] == [], 'runner final completion differs')
        import_guard = read(HERE / 'import-MOT17-input-parity.json')
        require(import_guard['status'] == 'PASS' and import_guard['reused_cells'] == 21 and import_guard['jpeg_frames'] == 5316, 'import input parity not complete')
        after = read(args.inputs_after)
        require(after['status'] == 'PASS' and after['sequence_count'] == 156 and after['jpeg_frames'] == 135503 and
                after['GT_seqinfo_models_binary_scope_and_images_unchanged'] is True and
                after['inputs_before_sha256'] == sha(ROOT / 'inputs-before.json') and
                after['checker_sha256'] == sha(HERE / 'verify_inputs.py'), 'full after-campaign input guard incomplete')
        final_authorities = {str(p): bind(p) for p in (EVIDENCE / 'capture-binaries-before.json',
                             EVIDENCE / 'capture-binaries-after.json', EVIDENCE / 'capture-index.json',
                             EVIDENCE / 'capture-complete.json', HERE / 'import-MOT17-input-parity.json', args.inputs_after)}
    result = {'status': 'FAIL' if errors else ('PASS' if args.final else 'PASS_FOR_COMPLETED_CELLS_ONLY'),
              'mode': 'final' if args.final else 'live', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'expected_cells': 468, 'completed_cells_verified': len(completed), 'pending_cells': len(pending),
              'by_arm': by_arm, 'imported_cells_verified': imports, 'fresh_cells_verified': fresh,
              'raw_packet_bytes_retained': sum(r['packet_bytes'] for r in completed),
              'all_completed_validation_reports_PASS': not errors, 'SportsMOT_cells': 0,
              'errors': errors, 'pending': pending, 'cells': completed,
              'source_guards': bind(HERE / 'final-audit-source-guards.json'),
              'final_authorities': final_authorities, 'checker_sha256': sha(Path(__file__)),
              'JPEG_or_GT_hashing_performed_by_this_checker': False,
              'raw_packet_rehash_performed': args.final, 'GPU_used': False,
              'scope': 'coverage, retained-output and input/identity integrity only; HOTA quality comes from separate official TrackEval events',
              'formal_session_sanitizer_acceptance': False, 'wall_seconds': time.monotonic()-started}
    output = args.output or (HERE / ('final-capture-audit.json' if args.final else
                                     'live-capture-audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json'))
    write(output, result)
    print(json.dumps({k: result[k] for k in ('status','mode','completed_cells_verified','pending_cells',
                                            'by_arm','imported_cells_verified','fresh_cells_verified','errors','wall_seconds')}))
    require(not errors, 'completed-cell audit failures retained in ' + str(output))


if __name__ == '__main__':
    main()
