"""Compare preflight JPEG inventories with prior TRT quality input manifests.

Reads metadata, manifests and TSVs only. No JPEG bytes are read again.
"""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = Path('/home/ray/developer/ai/saccade_v1/evidence/model-comparison/2026-11-11-yolo26m-trt-performance')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    destination = ROOT / 'preflight-tools/import-MOT17-input-parity.json'
    assert not destination.exists(), 'refuse overwrite import parity receipt'
    before_path = ROOT / 'inputs-before.json'
    runtime_path = ROOT / 'selection-runtime.json'
    prior_before_path = PRIOR / 'inputs-before.json'
    prior_after_path = PRIOR / 'inputs-after.json'
    binaries_path = PRIOR / 'quality-binaries-after.json'
    current = json.loads(before_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    previous_before = json.loads(prior_before_path.read_text())
    previous = json.loads(prior_after_path.read_text())
    previous_binaries = json.loads(binaries_path.read_text())
    assert sha(runtime_path) == current['runtime_selection']['sha256']
    old = {str(Path(path).resolve(strict=True)): value for path, value in previous.items()}
    old_before = {str(Path(path).resolve(strict=True)): value for path, value in previous_before.items()}
    rows = []
    frames = jpeg_bytes = metadata_count = 0
    for seq in runtime['sequences']:
        if seq['benchmark'] != 'MOT17':
            continue
        seqid = seq['id']
        inventory = seq['jpeg_inventory']
        assert sha(inventory['path']) == inventory['sha256']
        count = byte_count = 0
        with Path(inventory['path']).open(newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                count += 1
                assert int(row['frame']) == count
                key = row['resolved_jpeg']
                literal = {'bytes': int(row['bytes']), 'sha256': row['sha256']}
                assert old[key] == literal, f'JPEG inventory differs from prior quality: {key}'
                assert old_before[key] == old[key], f'prior JPEG before/after not equal: {key}'
                byte_count += literal['bytes']
        assert count == seq['frames'] == inventory['frames']
        metadata = {}
        for role in ('ground_truth', 'sequence_info'):
            current_entry = current['sequences'][seqid][role]
            key = current_entry['resolved_path']
            literal = {'bytes': current_entry['bytes'], 'sha256': current_entry['sha256']}
            assert old[key] == literal and old_before[key] == literal
            metadata[role] = current_entry
            metadata_count += 1
        rows.append({'sequence': seqid, 'frames': count, 'jpeg_bytes': byte_count,
                     'jpeg_inventory': inventory, 'jpeg_sizes_hashes_equal_prior_before_after': True,
                     'GT_seqinfo_equal_prior_before_after': True, **metadata})
        frames += count
        jpeg_bytes += byte_count
    assert len(rows) == 7 and frames == 5316 and metadata_count == 14
    model_checks = {}
    for arm, model in current['models'].items():
        assert previous_binaries[model['path']] == model['sha256']
        # Prior JPEG inputs manifest binds main engine; raw6 engines are bound
        # by the dedicated prior quality-binaries-after authority instead.
        if model['resolved_path'] in old:
            assert old[model['resolved_path']] == {'bytes': model['bytes'], 'sha256': model['sha256']}
        model_checks[arm] = {'path': model['path'], 'sha256': model['sha256'], 'same_prior_quality': True,
                            'prior_authority': str(binaries_path),
                            'also_bound_by_prior_JPEG_inputs_manifest': model['resolved_path'] in old}
    binary = current['track_binary']
    assert previous_binaries[binary['path']] == binary['sha256']
    report = {'status': 'PASS', 'at': datetime.now(timezone.utc).isoformat(),
              'sequences': 7, 'jpeg_frames': frames, 'jpeg_bytes': jpeg_bytes, 'reused_cells': 21,
              'GT_seqinfo_pairs': 7, 'metadata_files': metadata_count,
              'all_JPEG_sizes_SHA256_equal_prior_quality': True,
              'GT_and_seqinfo_equal_prior_quality': True,
              'prior_inputs_before_after_equal_for_all_imported_inputs': True,
              'all_three_engines_same_prior_quality': True, 'track_binary_same_prior_quality': True,
              'JPEG_bytes_read_in_this_check': False, 'GPU_used': False,
              'scope': 'Receipt comparison of newly hashed before-campaign inventories against prior TRT quality before/after manifests; no new inference or JPEG rehash',
              'authorities': {'new_inputs_before': binding(before_path), 'runtime_selection': binding(runtime_path),
                              'prior_inputs_before': binding(prior_before_path), 'prior_inputs_after': binding(prior_after_path),
                              'prior_quality_binaries_after': binding(binaries_path), 'checker': binding(Path(__file__))},
              'models': model_checks, 'track_binary': binary, 'per_sequence': rows}
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in ('status','sequences','jpeg_frames','reused_cells',
                                          'GT_and_seqinfo_equal_prior_quality','all_three_engines_same_prior_quality',
                                          'track_binary_same_prior_quality','JPEG_bytes_read_in_this_check')}))


if __name__ == '__main__':
    main()
