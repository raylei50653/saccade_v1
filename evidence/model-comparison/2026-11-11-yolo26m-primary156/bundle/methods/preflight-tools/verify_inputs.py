"""After-campaign read-only rehash against the exact before-campaign inventory."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def frame_check(row):
    inference, resolved = Path(row['inference_jpeg']), Path(row['resolved_jpeg'])
    assert inference.resolve(strict=True) == resolved, f'JPEG symlink target changed: {inference}'
    assert resolved.stat().st_size == int(row['bytes']), f'JPEG size changed: {resolved}'
    assert sha(resolved) == row['sha256'], f'JPEG bytes changed: {resolved}'
    return 1


def check(binding):
    assert sha(binding['path']) == binding['sha256'], f'bound file changed: {binding["path"]}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'inputs-after.json')
    args = parser.parse_args()
    assert not args.output.exists(), 'refuse overwrite after-campaign receipt'
    started = time.monotonic()
    authority = json.loads((HERE / 'receipt.json').read_text())
    check(authority['inputs_before'])
    before = json.loads((ROOT / 'inputs-before.json').read_text())
    for key in ('source_primary_selection', 'policy', 'historical_selection', 'runtime_selection', 'track_binary', 'preflight_source'):
        check(before[key])
    for item in before['models'].values():
        check(item)
    for seqid, metadata in before['sequences'].items():
        for item in metadata.values():
            check(item)
    frames = 0
    complete = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for seqid, inventory in before['jpeg_inventories'].items():
            check(inventory)
            with Path(inventory['path']).open(newline='') as stream:
                count = sum(pool.map(frame_check, csv.DictReader(stream, delimiter='\t')))
            assert count == inventory['frames'], f'inventory frame count changed: {seqid}'
            frames += count
            complete.append(seqid)
            print(json.dumps({'status': 'RUNNING', 'complete_sequences': len(complete), 'frames_verified': frames,
                              'last_sequence': seqid, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    assert len(complete) == 156 and frames == 135503
    result = {'status': 'PASS', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'sequence_count': 156, 'jpeg_frames': frames, 'hash_workers': 4,
              'GT_seqinfo_models_binary_scope_and_images_unchanged': True,
              'inputs_before_sha256': sha(ROOT / 'inputs-before.json'),
              'checker_sha256': sha(Path(__file__)), 'wall_seconds': time.monotonic()-started}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
