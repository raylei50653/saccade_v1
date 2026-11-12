"""Freeze exact primary156 JPEG/GT/seqinfo/model inputs before native campaign.

New six-digit views link directly to physical eight-digit DanceTrack images.
The datasets, prior views, frozen scope and previous evidence are read only.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import configparser
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/ray/developer/ai/saccade_research/experiments/common/primary-selection-20261111.json')
TRT_ROOT = Path('/home/ray/developer/ai/saccade_v1/build/yolo26m-trt-comparison-20261111')
ENGINES = {
    'official': (TRT_ROOT / 'artifacts/official/model.engine', 'fad51c68686166c659386814b8f14f90dd4531afd12c2739a20031fbe6e6376e'),
    'finetuned': (TRT_ROOT / 'artifacts/finetuned/model.engine', '9720adc960119ae3a55fbfaf4f80a4dbc6b495ad47fd09e95fbfd97398382e0c'),
    'main': (Path('/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine'), '8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876'),
}
BINARY = TRT_ROOT / 'pipeline-tools/track_sequence'
ALLOWED = {'MOT15', 'MOT17', 'MOT20', 'DanceTrack', 'PP22'}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def binding(path, expected=None):
    path = Path(path)
    got = sha(path)
    if expected is not None and got != expected:
        raise RuntimeError(f'bound bytes changed: {path}: {got} != {expected}')
    return {'path': str(path), 'resolved_path': str(path.resolve(strict=True)), 'bytes': path.stat().st_size,
            'sha256': got, 'mtime_ns': path.stat().st_mtime_ns}


def hash_frame(item):
    fid, inference, physical = item
    if inference.resolve(strict=True) != physical:
        raise RuntimeError(f'JPEG alias target differs: {inference}')
    before = physical.stat()
    h = hashlib.sha256()
    prefix = None
    with physical.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            if prefix is None:
                prefix = block[:2]
            h.update(block)
    after = physical.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'JPEG changed during hashing: {physical}')
    if before.st_size == 0 or prefix != b'\xff\xd8':
        raise RuntimeError(f'not a nonempty JPEG: {physical}')
    return (fid, str(inference), str(physical), before.st_size, h.hexdigest(), before.st_mtime_ns)


def main():
    start = time.monotonic()
    started = now()
    assert not (ROOT / 'selection-runtime.json').exists(), 'refuse overwrite completed preflight'
    source_binding = binding(SOURCE)
    frozen = json.loads(SOURCE.read_text())
    policy_binding = binding(frozen['policy']['path'], frozen['policy']['sha256'])
    policy = json.loads(Path(frozen['policy']['path']).read_text())
    assert policy['default_scope'] == 'primary' and set(policy['primary_benchmarks']) == ALLOWED
    assert policy['excluded_from_default']['SportsMOT']['default_replay'] is False
    assert policy['excluded_from_default']['SportsMOT']['default_scoring'] is False
    assert frozen['sequence_count'] == len(frozen['sequences']) == 156
    assert len({s['id'] for s in frozen['sequences']}) == 156
    assert sum(s['frames'] for s in frozen['sequences']) == 135503
    assert all(s['benchmark'] in ALLOWED for s in frozen['sequences'])
    assert len(frozen['excluded']['SportsMOT']) == 45
    group_ids = [n for g in frozen['primary_groups'].values() for n in g]
    assert len(group_ids) == len(set(group_ids)) == 156
    assert set(group_ids) == {s['id'] for s in frozen['sequences']}
    history_binding = binding(frozen['source_selection']['path'], frozen['source_selection']['sha256'])
    models = {name: binding(path, expected) for name, (path, expected) in ENGINES.items()}
    binary = binding(BINARY)
    runtime = copy.deepcopy(frozen)
    runtime['schema'] = 'saccade-yolo26m-primary156-runtime-selection/1'
    runtime['source_primary_selection'] = source_binding
    runtime['runtime_created_utc'] = started
    runtime['inference_reads_gt'] = False
    inventories = HERE / 'jpeg-inventories'
    inventories.mkdir(exist_ok=False)
    metadata = {}
    jpeg_index = {}
    views = {}
    total_bytes = total_frames = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, seq in enumerate(runtime['sequences'], start=1):
            seqid = seq['id']
            assert '/' not in seqid and '..' not in seqid
            gt = binding(seq['ground_truth']['path'], seq['ground_truth']['sha256'])
            info = binding(seq['sequence_info']['path'], seq['sequence_info']['sha256'])
            parser = configparser.ConfigParser()
            parser.read(seq['sequence_info']['path'])
            c = parser['Sequence']
            assert (int(c['seqLength']), int(c['imWidth']), int(c['imHeight'])) == (seq['frames'], seq['width'], seq['height'])
            assert abs(float(c['frameRate']) - float(seq['fps'])) < 1e-9
            physical_base = Path(seq.get('path') or Path(seq['sequence_info']['path']).parent)
            recording_base = Path(seq.get('recording_path') or physical_base)
            image_dir = c['imDir']
            extension = c['imExt']
            source_images = physical_base / image_dir
            # Prefer the literal dataset paths over historical six-digit views.
            if (source_images / f'{1:06d}{extension}').is_file():
                inference_base = physical_base
                source_digits = 6
            elif (source_images / f'{1:08d}{extension}').is_file():
                inference_base = ROOT / 'image-views' / seqid
                (inference_base / 'img1').mkdir(parents=True, exist_ok=False)
                assert image_dir == 'img1', 'new view requires standard img1 header'
                shutil.copyfile(seq['sequence_info']['path'], inference_base / 'seqinfo.ini')
                source_digits = 8
                views[seqid] = {'path': str(inference_base), 'literal_dataset_path': str(physical_base),
                                'historical_recording_path_not_modified': str(recording_base),
                                'seqinfo': binding(inference_base / 'seqinfo.ini', info['sha256']),
                                'source_digits': 8, 'inference_digits': 6, 'links': seq['frames']}
            else:
                # Explicit fallback only if the recorded path is itself the data source.
                candidate = recording_base / image_dir / f'{1:06d}{extension}'
                if not candidate.is_file():
                    raise FileNotFoundError(f'no full JPEG sequence for {seqid}: {source_images}')
                inference_base = recording_base
                source_images = recording_base / image_dir
                source_digits = 6
            frame_items = []
            for fid in range(1, seq['frames'] + 1):
                source = source_images / f'{fid:0{source_digits}d}{extension}'
                physical = source.resolve(strict=True)
                if not physical.is_file():
                    raise FileNotFoundError(physical)
                inference = inference_base / image_dir / f'{fid:06d}{extension}'
                if source_digits == 8:
                    inference.symlink_to(physical)
                frame_items.append((fid, inference, physical))
            inventory = inventories / f'{seqid}.tsv'
            seq_bytes = 0
            with inventory.open('x', newline='') as target:
                csv_writer = csv.writer(target, delimiter='\t', lineterminator='\n')
                csv_writer.writerow(('frame', 'inference_jpeg', 'resolved_jpeg', 'bytes', 'sha256', 'mtime_ns'))
                for row in pool.map(hash_frame, frame_items):
                    csv_writer.writerow(row)
                    seq_bytes += row[3]
            ib = binding(inventory)
            seq['inference_path'] = str(inference_base)
            seq['inference_sequence_info'] = binding(inference_base / 'seqinfo.ini', info['sha256'])
            seq['jpeg_inventory'] = {**ib, 'frames': seq['frames'], 'jpeg_bytes': seq_bytes,
                                     'source_digits': source_digits, 'native_inference_digits': 6}
            seq['inference_path_is_new_view'] = source_digits == 8
            metadata[seqid] = {'ground_truth': gt, 'sequence_info': info,
                               'inference_sequence_info': seq['inference_sequence_info']}
            jpeg_index[seqid] = seq['jpeg_inventory']
            total_frames += seq['frames']
            total_bytes += seq_bytes
            progress = {'status': 'RUNNING', 'at': now(), 'complete_sequences': index, 'total_sequences': 156,
                        'frames_hashed': total_frames, 'total_frames': 135503, 'jpeg_bytes_hashed': total_bytes,
                        'last_sequence': seqid, 'elapsed_seconds': time.monotonic()-start}
            write(HERE / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    assert total_frames == 135503
    # Recheck small fixed authorities at the completed before-campaign boundary.
    assert sha(SOURCE) == source_binding['sha256'] and sha(Path(frozen['policy']['path'])) == policy_binding['sha256']
    for seqid, files in metadata.items():
        for item in files.values():
            assert sha(Path(item['path'])) == item['sha256']
    for item in [*models.values(), binary]:
        assert sha(Path(item['path'])) == item['sha256']
    runtime['input_preflight'] = {'status': 'PASS', 'jpeg_frames': total_frames, 'jpeg_bytes': total_bytes,
                                 'new_views': len(views), 'hash_workers': 4}
    write(ROOT / 'selection-runtime.json', runtime)
    before = {'status': 'PASS', 'started_utc': started, 'completed_utc': now(),
              'source_primary_selection': source_binding, 'policy': policy_binding,
              'historical_selection': history_binding, 'runtime_selection': binding(ROOT / 'selection-runtime.json'),
              'sequences': metadata, 'jpeg_inventories': jpeg_index, 'new_image_views': views,
              'models': models, 'track_binary': binary, 'preflight_source': binding(Path(__file__)),
              'sequence_count': 156, 'jpeg_frames': total_frames, 'jpeg_bytes': total_bytes,
              'excluded_SportsMOT_sequences': 45, 'all_expected_frames_available': True,
              'GT_and_seqinfo_match_frozen_selection': True, 'hash_workers': 4,
              'group_counts': {k: len(v) for k, v in frozen['primary_groups'].items()},
              'benchmark_counts': dict(Counter(s['benchmark'] for s in runtime['sequences'])),
              'inference_reads_gt': False, 'datasets_and_prior_views_modified': False,
              'wall_seconds': time.monotonic()-start}
    write(ROOT / 'inputs-before.json', before)
    receipt = {k: before[k] for k in ('status','sequence_count','jpeg_frames','jpeg_bytes','excluded_SportsMOT_sequences',
                                     'all_expected_frames_available','GT_and_seqinfo_match_frozen_selection',
                                     'hash_workers','group_counts','benchmark_counts','wall_seconds')}
    receipt.update({'runtime_selection': binding(ROOT / 'selection-runtime.json'),
                    'inputs_before': binding(ROOT / 'inputs-before.json'), 'new_image_views': len(views)})
    write(HERE / 'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(HERE / 'failure.json', {'status': 'FAIL', 'at': now(), 'error': type(error).__name__ + ': ' + str(error)})
        raise
