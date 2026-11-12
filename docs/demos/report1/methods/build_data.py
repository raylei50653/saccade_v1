#!/usr/bin/env python3
"""Export immutable measured observations and a source-only playback video.

No model inference, GT access, tracking rerun, or historical-evidence mutation.
Run from any cwd. Requires the existing shared JPEG sequence and ffmpeg/ffprobe.
"""
from __future__ import annotations

import argparse
import collections
import configparser
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
DEMO = Path(__file__).resolve().parents[1]
ASSETS = DEMO / 'assets'
LIVE = ROOT / 'evidence/report1/2026-11-11-g1-live'
OFFLINE = ROOT / 'evidence/report1/2026-11-11-g1-offline-e2e'
PACKET_HEADER = struct.Struct('<4sHHIIQqqqIIII')
PACKET_ROW = struct.Struct('<QQII6f')
MOTION_HEADER = struct.Struct('<4sI5iQQ')
MOTION_ROW = struct.Struct('<qqIii4f')
PACKET_BYTES, HEADER_BYTES, ROW_BYTES, CAPACITY = 12368, 80, 48, 256


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def packets(data: bytes, expected_count: int) -> tuple[list[dict], dict]:
    assert PACKET_HEADER.size == 64 and PACKET_ROW.size == ROW_BYTES
    assert len(data) == expected_count * PACKET_BYTES
    frames: list[dict] = []
    stats = collections.Counter()
    uids: set[int] = set()
    output_ids: set[int] = set()
    uid_outputs: dict[int, set[int]] = collections.defaultdict(set)
    for i in range(expected_count):
        offset = i * PACKET_BYTES
        (magic, version, header_size, packet_size, capacity, epoch, frame,
         committed, fail_frame, status, count, frame_error, fail_code) = PACKET_HEADER.unpack_from(data, offset)
        assert (magic, version, header_size, packet_size, capacity) == (b'SCRP', 1, HEADER_BYTES, PACKET_BYTES, CAPACITY)
        assert (epoch, frame, committed, fail_frame, status, frame_error, fail_code) == (1, i, i, -1, 0, 0, 0)
        assert count <= CAPACITY
        assert data[offset + 64:offset + HEADER_BYTES] == bytes(16)
        used_end = offset + HEADER_BYTES + count * ROW_BYTES
        assert data[used_end:offset + PACKET_BYTES] == bytes(offset + PACKET_BYTES - used_end)
        rows = []
        for j in range(count):
            row_offset = offset + HEADER_BYTES + j * ROW_BYTES
            uid, oid, cls, flags, x1, y1, x2, y2, observed, association = PACKET_ROW.unpack_from(data, row_offset)
            values = (x1, y1, x2, y2, observed, association)
            assert uid > 0 and oid > 0 and cls == 0 and flags in (1, 2)
            assert all(math.isfinite(value) for value in values)
            assert x2 > x1 and y2 > y1 and 0 <= observed <= 1
            if flags == 2:
                assert observed == 0 and association == 0
            # The canonical writer normalizes signed zero before serializing.
            for k, value in enumerate(values):
                if value == 0:
                    assert data[row_offset + 24 + k * 4:row_offset + 28 + k * 4] == bytes(4)
            rows.append([oid, uid, x1, y1, x2, y2, observed, association, flags])
            stats['rows'] += 1
            stats[f'flags_{flags}'] += 1
            stats['boxes_partly_outside_frame'] += int(x1 < 0 or y1 < 0 or x2 > 1920 or y2 > 1080)
            uids.add(uid)
            output_ids.add(oid)
            uid_outputs[uid].add(oid)
        frames.append({'i': i, 'rows': rows})
    return frames, {
        **dict(stats), 'unique_uid_count': len(uids), 'unique_output_id_count': len(output_ids),
        'min_rows_per_frame': min(len(f['rows']) for f in frames),
        'max_rows_per_frame': max(len(f['rows']) for f in frames),
        'uid_output_id_changes': {str(k): sorted(v) for k, v in uid_outputs.items() if len(v) > 1},
        'canonical_headers_reserved_unused_float_encoding_validated': True,
    }


def motions(data: bytes, frames: list[dict]) -> dict:
    assert MOTION_HEADER.size == 44 and MOTION_ROW.size == 44
    magic, version, original_w, original_h, scale, width, height, epoch, count = MOTION_HEADER.unpack_from(data)
    assert (magic, version, original_w, original_h, scale, width, height, epoch, count) == (
        b'SCMO', 1, 1920, 1080, 1, 1920, 1080, 1, len(frames))
    assert len(data) == MOTION_HEADER.size + count * MOTION_ROW.size
    statuses = collections.Counter()
    for i, frame in enumerate(frames):
        f, reference, status, px, py, pcr, confidence, tx, ty = MOTION_ROW.unpack_from(data, 44 + i * 44)
        assert f == i and status in (0, 1, 2, 3)
        assert all(math.isfinite(x) for x in (pcr, confidence, tx, ty))
        # This exact G1 run has an uninitialized first frame and valid later frames.
        assert (reference, status) == ((-1, 1) if i == 0 else (i - 1, 2))
        assert px == py == pcr == 0 and confidence == (0 if i == 0 else 1)
        if status != 2:
            assert tx == ty == 0
        frame['motion'] = {'tx': tx, 'ty': ty, 'status': status, 'ref': reference}
        statuses[status] += 1
    return {'status_counts': dict(statuses), 'frame_bytes': 44, 'header_bytes': 44,
            'translation_units': 'original-image pixels, reference frame to current image content; not camera pose or velocity',
            'g1_compatibility_fields': 'peak_x, peak_y and pcr always zero; confidence=1 only for valid status'}


def probe_video(path: Path) -> dict:
    cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
           '-show_entries', 'stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,color_range,color_space,color_transfer,color_primaries:format=duration,size',
           '-of', 'json', str(path)]
    result = json.loads(subprocess.check_output(cmd, text=True))
    stream, = result['streams']
    assert stream['codec_name'] == 'h264'
    assert (stream['width'], stream['height'], stream['pix_fmt']) == (1920, 1080, 'yuv420p')
    assert stream['avg_frame_rate'] == stream['r_frame_rate'] == '30/1'
    assert int(stream['nb_read_frames']) == int(stream['nb_frames']) == 654
    assert abs(float(result['format']['duration']) - 21.8) < 0.00001
    assert (stream['color_space'], stream['color_transfer'], stream['color_primaries']) == ('bt709', 'iec61966-2-1', 'bt709')
    assert stream['color_range'] == 'tv'
    # Validate that faststart actually placed metadata before media payload.
    box_types = []
    with path.open('rb') as handle:
        while header := handle.read(8):
            assert len(header) == 8
            size, kind = struct.unpack('>I4s', header)
            consumed = 8
            if size == 1:
                size, = struct.unpack('>Q', handle.read(8))
                consumed = 16
            if size == 0:
                size = path.stat().st_size - handle.tell() + consumed
            assert size >= consumed
            box_types.append(kind.decode('ascii'))
            handle.seek(size - consumed, 1)
    assert box_types.index('moov') < box_types.index('mdat')
    result['mp4_box_order'] = box_types
    result['faststart_verified'] = True
    result['command'] = cmd
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-video', action='store_true', help='Reuse an existing video only after provenance hash and ffprobe checks.')
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    live = json.loads((LIVE / 'summary.json').read_text())
    offline = json.loads((OFFLINE / 'summary.json').read_text())
    assert live['status'] == 'PASS_BOUNDED_MEASUREMENT_AND_PACKET_PARITY'
    assert offline['status'] == 'PASS_BOUNDED_PROCESS_COMPLETION_AND_OUTPUT_PARITY'
    assert live['contract']['base_commit'] == offline['contract']['runtime_base_commit']
    sequence = Path(offline['contract']['sequence'])
    config = configparser.ConfigParser()
    config.read(sequence / 'seqinfo.ini')
    info = config['Sequence']
    frame_count, fps = info.getint('seqLength'), info.getint('frameRate')
    assert (frame_count, fps, info.getint('imWidth'), info.getint('imHeight')) == (654, 30, 1920, 1080)
    inputs_manifest_path = OFFLINE / 'inputs-before.json'
    manifest = json.loads(inputs_manifest_path.read_text())
    jpeg_bytes = 0
    for i in range(1, frame_count + 1):
        path = sequence / info['imDir'] / f'{i:06}{info["imExt"]}'
        expected = manifest[str(path)]
        assert path.stat().st_size == expected['bytes'] and file_digest(path) == expected['sha256']
        jpeg_bytes += expected['bytes']
    assert file_digest(sequence / 'seqinfo.ini') == manifest[str(sequence / 'seqinfo.ini')]['sha256']
    print(f'Validated current sequence against archived manifest: {frame_count} JPEGs, {jpeg_bytes} bytes', flush=True)
    packets_path = OFFLINE / 'raw/run1/packets.bin.gz'
    motion_path = OFFLINE / 'raw/run1/motion.scmo.gz'
    packet_bytes, motion_bytes = gzip.decompress(packets_path.read_bytes()), gzip.decompress(motion_path.read_bytes())
    assert digest(packet_bytes) == offline['outputs']['packets.bin']['sha256']
    assert digest(motion_bytes) == offline['outputs']['motion.scmo']['sha256']
    frames, packet_stats = packets(packet_bytes, frame_count)
    motion_stats = motions(motion_bytes, frames)
    metrics = {
        'singleMeanMs': live['single_latency_ms']['mean'],
        'p95Ms': live['single_latency_ms']['p95'],
        'p99Ms': live['single_latency_ms']['p99'],
        'queuedMeanMs': live['queued_delivery_latency_ms']['mean'],
        'queuedP95Ms': live['queued_delivery_latency_ms']['p95'],
        'throughputFps': live['queued_throughput_fps'],
        'offlineTotalSeconds': offline['process_wall_ms']['mean'] / 1000,
    }
    data = {
        'meta': {
            'sequence': info['name'], 'width': 1920, 'height': 1080,
            'fps': fps, 'frameCount': frame_count, 'durationSeconds': frame_count / fps,
            'baseline': live['contract']['base_commit'], 'engineHash': live['contract']['engine_sha256'],
            'sourceVideo': 'assets/source.mp4', 'poster': 'assets/poster.jpg',
            'packetSha256': digest(packet_bytes), 'motionSha256': digest(motion_bytes),
            'replayOnly': True, 'driverVersion': live['driver_version'],
            'inputType': 'JPEG sequence converted to source-only H.264 for playback',
            'rowFields': ['output_id', 'uid', 'x1', 'y1', 'x2', 'y2', 'observed_score', 'association_score', 'flags'],
            'flags': {'1': 'observed', '2': 'predicted'},
            'frameMapping': 'frames[i] maps to source JPEG %06d.jpg at i+1; media time i/30 seconds',
            'outputIdMeaning': 'tracker output identity; may be reassigned by Bridge; not GT identity or final MOT renumbered ID',
            'boxMeaning': 'Kalman state xyxy in original-image pixels; measurement_box=false; no EOF interpolation',
            'trailsPolicy': 'derived visualization only; break across missing observations or identity changes',
            'metricScope': 'independent recorded benchmarks, not the replay rendering/playback frame time',
            'singleSamples': 480, 'queuedSamples': 480, 'offlineSequenceFrames': 654, 'offlineFreshProcesses': 3,
            'nsaEnabled': False, 'measurementBox': False, 'graphEnabled': True,
        },
        'frames': frames, 'metrics': metrics,
    }
    compact = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (ASSETS / 'data.json').write_text(compact + '\n')
    (ASSETS / 'data.js').write_text('window.SACCADE_DEMO=' + compact + ';\n')
    poster_source = sequence / info['imDir'] / '000221.jpg'
    shutil.copyfile(poster_source, ASSETS / 'poster.jpg')
    video = ASSETS / 'source.mp4'
    encode_command = ['ffmpeg', '-hide_banner', '-nostdin', '-n', '-framerate', '30', '-start_number', '1',
        '-i', str(sequence / info['imDir'] / '%06d.jpg'), '-frames:v', '654', '-an',
        '-vf', 'scale=in_range=pc:out_range=tv:in_color_matrix=bt601:out_color_matrix=bt709,setsar=1,setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-threads', '4', '-pix_fmt', 'yuv420p',
        '-fps_mode', 'passthrough', '-color_range', 'tv', '-colorspace', 'bt709',
        '-color_primaries', 'bt709', '-color_trc', 'iec61966-2-1', '-movflags', '+faststart', str(video)]
    started = time.monotonic()
    if video.exists():
        assert args.reuse_video, 'Video exists; use --reuse-video after checking the previous provenance.'
        previous = json.loads((ASSETS / 'provenance.json').read_text())
        assert previous['source_inputs']['archived_manifest_sha256'] == file_digest(inputs_manifest_path)
        assert previous['artifacts']['source.mp4']['sha256'] == file_digest(video)
        encode_seconds = previous['encode']['wall_seconds']
    else:
        with (ASSETS / 'source-encode.log').open('wb') as log:
            subprocess.run(encode_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        encode_seconds = time.monotonic() - started
    probe = probe_video(video)
    assert video.stat().st_size < 30 * 1024 * 1024
    # Input hashing is a custody check and is not a model/quality experiment.
    for i in range(1, frame_count + 1):
        path = sequence / info['imDir'] / f'{i:06}{info["imExt"]}'
        assert path.stat().st_size == manifest[str(path)]['bytes']
        assert file_digest(path) == manifest[str(path)]['sha256']
    provenance = {
        'schema': 'saccade-report1-replay-assets/1', 'baseline': live['contract']['base_commit'],
        'purpose': 'source-only playback video and exact observations for an offline demonstration',
        'new_inference_or_scoring': False, 'runtime_or_historical_evidence_modified': False,
        'generator': {'path': str(Path(__file__).resolve().relative_to(ROOT)), 'sha256': file_digest(Path(__file__))},
        'source_inputs': {
            'sequence': str(sequence), 'frame_count': frame_count, 'fps': fps,
            'jpeg_bytes': jpeg_bytes, 'image_hashes_verified_against_archived_manifest': True,
            'image_hashes_checked_before_and_after_encode': True,
            'archived_manifest': str(inputs_manifest_path.relative_to(ROOT)),
            'archived_manifest_sha256': file_digest(inputs_manifest_path),
            'seqinfo_sha256': file_digest(sequence / 'seqinfo.ini'),
            'poster_source': str(poster_source), 'poster_source_sha256': file_digest(poster_source),
            'packets_gzip': str(packets_path.relative_to(ROOT)), 'packets_gzip_sha256': file_digest(packets_path),
            'packet_raw_sha256': digest(packet_bytes),
            'motion_gzip': str(motion_path.relative_to(ROOT)), 'motion_gzip_sha256': file_digest(motion_path),
            'motion_raw_sha256': digest(motion_bytes),
            'live_summary': str((LIVE / 'summary.json').relative_to(ROOT)), 'live_summary_sha256': file_digest(LIVE / 'summary.json'),
            'offline_summary': str((OFFLINE / 'summary.json').relative_to(ROOT)), 'offline_summary_sha256': file_digest(OFFLINE / 'summary.json'),
        },
        'packet_validation': packet_stats, 'motion_validation': motion_stats,
        'encode': {'argv': encode_command, 'wall_seconds': encode_seconds, 'uses_gpu': False,
                   'ffmpeg_version': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
                   'ffprobe_version': subprocess.check_output(['ffprobe', '-version'], text=True).splitlines()[0],
                   'color_policy': 'JPEG full-range BT.601 YCbCr to limited-range BT.709 YCbCr; sRGB transfer metadata; browser playback asset, not model input',
                   'lossy_video': True, 'source_jpegs_unchanged': True},
        'ffprobe': probe,
        'artifacts': {name: {'bytes': (ASSETS / name).stat().st_size, 'sha256': file_digest(ASSETS / name)}
                      for name in ('data.json', 'data.js', 'poster.jpg', 'source.mp4', 'source-encode.log')},
        'limitations': [
            'prerecorded result playback, not a live-inference recording',
            'source playback is30FPS and separate from benchmark throughput',
            'IDs are tracker output IDs, not GT identities',
            'state-estimate boxes, no offline MOT interpolation',
            'visual clipping or trails do not modify the recorded observations',
            'independent benchmark metrics do not measure this player or video encoding',
        ],
    }
    dump(ASSETS / 'provenance.json', provenance)
    print(json.dumps({'status': 'PASS', 'frames': len(frames), 'rows': packet_stats['rows'],
                      'video_bytes': video.stat().st_size, 'duration_seconds': frame_count / fps,
                      'data_json_bytes': (ASSETS / 'data.json').stat().st_size}, indent=2), flush=True)


if __name__ == '__main__':
    main()
