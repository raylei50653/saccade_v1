#!/usr/bin/env python3
"""Render MOT17-02 ResultPacket observations as a source-rate replay video.

Run from the project with:
  uv run --with pillow python docs/demos/report1/methods/render_video_02.py

Only outputs in docs/demos/report1/assets are created. No inference or GT reads.
"""
import collections
import colorsys
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[4]
ASSETS = ROOT / 'docs/demos/report1/assets'
PACKETS = ROOT / 'evidence/report1/2026-11-11-mot17-02-demo/raw/run1/packets.bin.gz'
INPUT_MANIFEST = ROOT / 'evidence/report1/2026-11-11-mot17-02-demo/inputs-before.json'
SEQUENCE = Path('/home/ray/developer/ai/dataset/MOT17/train/MOT17-02-SDP')
FRAME_COUNT = 600
FPS = 30
WIDTH, HEIGHT = 1920, 1080
HEADER_BYTES, ROW_BYTES, PACKET_BYTES = 80, 48, 12368
HEADER = struct.Struct('<4sHHIIQqqqIIII')
ROW = struct.Struct('<QQII6f')
VIDEO = ASSETS / 'tracking-demo-02.mp4'
POSTER = ASSETS / 'tracking-demo-poster-02.jpg'
RECEIPT = ASSETS / 'video-render-02.json'


def sha(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def load_packets():
    payload = gzip.decompress(PACKETS.read_bytes())
    assert len(payload) == FRAME_COUNT * PACKET_BYTES
    result = []
    for frame in range(FRAME_COUNT):
        packet = payload[frame * PACKET_BYTES:(frame + 1) * PACKET_BYTES]
        magic, version, header_bytes, packet_bytes, capacity, epoch, index, committed, fail_frame, status, count, errors, fail_code = HEADER.unpack_from(packet)
        assert (magic, version, header_bytes, packet_bytes, capacity) == (b'SCRP', 1, HEADER_BYTES, PACKET_BYTES, 256)
        assert (epoch, index, committed, fail_frame, status, errors, fail_code) == (1, frame, frame, -1, 0, 0, 0)
        assert 0 <= count <= capacity
        assert packet[64:80] == bytes(16)
        assert packet[HEADER_BYTES + count * ROW_BYTES:] == bytes((capacity - count) * ROW_BYTES)
        rows = []
        for i in range(count):
            values = ROW.unpack_from(packet, HEADER_BYTES + i * ROW_BYTES)
            uid, output_id, class_id, flags, x1, y1, x2, y2, score, association_score = values
            assert uid > 0 and output_id > 0 and class_id == 0 and flags == 1
            assert all(math.isfinite(v) for v in (x1, y1, x2, y2, score, association_score))
            assert x1 < x2 and y1 < y2
            rows.append({'id': output_id, 'uid': uid, 'xyxy': (x1, y1, x2, y2)})
        result.append(rows)
    return payload, result


def id_color(output_id):
    # Stable output-ID colors, without implying correctness or ground truth.
    hue = (output_id * 0.618033988749895 + 0.36) % 1
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(hue, .65, 1))


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    if any(p.exists() for p in (VIDEO, POSTER, RECEIPT)):
        raise SystemExit('refusing to overwrite an existing video deliverable')
    baseline = '3cbaa518f0dc37d691404e95ed85c0109c81130d'
    payload, frames = load_packets()
    original_packet_sha = sha(PACKETS)
    original_manifest_sha = sha(INPUT_MANIFEST)
    source_manifest = json.loads(INPUT_MANIFEST.read_text())
    source_inputs = []
    for number in range(1, FRAME_COUNT + 1):
        path = SEQUENCE / 'img1' / f'{number:06}.jpg'
        expected = source_manifest[str(path)]
        actual = sha(path)
        assert path.stat().st_size == expected['bytes'] and actual == expected['sha256']
        source_inputs.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': actual})
    font_info = subprocess.check_output(['fc-match', '-f', '%{file}\n%{index}\n', 'Noto Sans CJK TC'], text=True).splitlines()
    font_file, font_index = font_info[0], int(font_info[1])
    font = lambda size: ImageFont.truetype(font_file, size=size, index=font_index)
    title_font, subtitle_font, footer_font, id_font, frame_font = font(29), font(18), font(21), font(20), font(24)
    ffmpeg_command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{WIDTH}x{HEIGHT}',
        '-framerate', str(FPS), '-i', 'pipe:0', '-an',
        '-vf', 'scale=in_range=pc:out_range=tv:out_color_matrix=bt709,setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-color_primaries', 'bt709', '-color_trc', 'iec61966-2-1', '-colorspace', 'bt709',
        '-color_range', 'tv', '-metadata', 'title=Saccade v1 - MOT17-02-SDP prerecorded tracking observations',
        '-metadata', 'comment=Actual packet observations replayed at source 30 FPS; playback is not inference speed.',
        str(VIDEO)]
    log_dir = ROOT / 'build/demo-report1-video-02'
    log_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_log = log_dir / 'ffmpeg.log'
    started = time.perf_counter()
    trails = {}
    row_count = 0
    with ffmpeg_log.open('wb') as error:
        encoder = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=error)
        try:
            for index, rows in enumerate(frames):
                source_path = SEQUENCE / 'img1' / f'{index + 1:06}.jpg'
                image = Image.open(source_path).convert('RGB')
                assert image.size == (WIDTH, HEIGHT)
                # Preserve the full image with a small vertical fit; title/footer never hide source pixels.
                image_width, image_height = 1706, 960
                left, top = (WIDTH - image_width) // 2, 76
                sx, sy = image_width / WIDTH, image_height / HEIGHT
                canvas = Image.new('RGB', (WIDTH, HEIGHT), '#091827')
                canvas.paste(image.resize((image_width, image_height), Image.Resampling.LANCZOS), (left, top))
                overlay = Image.new('RGBA', canvas.size)
                draw = ImageDraw.Draw(overlay)
                visible_identities = {(r['id'], r['uid']) for r in rows}
                trails = {k: v for k, v in trails.items() if k in visible_identities}
                for row in rows:
                    output_id = row['id']
                    identity = (output_id, row['uid'])
                    x1, y1, x2, y2 = row['xyxy']
                    anchor = (left + min(WIDTH, max(0, (x1 + x2) / 2)) * sx,
                              top + min(HEIGHT, max(0, y2)) * sy)
                    previous = trails.get(identity)
                    if previous is None or previous['last'] != index - 1:
                        previous = {'last': index, 'points': collections.deque(maxlen=24)}
                        trails[identity] = previous
                    previous['last'] = index
                    previous['points'].append(anchor)
                    color = id_color(output_id)
                    if len(previous['points']) > 1:
                        draw.line(list(previous['points']), fill=color + (170,), width=3)
                for row in rows:
                    x1, y1, x2, y2 = row['xyxy']
                    x1, x2 = (left + min(WIDTH, max(0, x)) * sx for x in (x1, x2))
                    y1, y2 = (top + min(HEIGHT, max(0, y)) * sy for y in (y1, y2))
                    if x1 >= x2 or y1 >= y2:
                        continue
                    color = id_color(row['id'])
                    draw.rectangle((x1, y1, x2, y2), outline=color + (240,), width=3)
                    label = f"ID {row['id']}"
                    label_width = draw.textbbox((0, 0), label, font=id_font)[2] + 12
                    label_left = min(max(left, x1), left + image_width - label_width)
                    label_top = min(max(top, y1 - 28), top + image_height - 29)
                    draw.rounded_rectangle((label_left, label_top, label_left + label_width, label_top + 27), radius=3, fill=color + (245,))
                    draw.text((label_left + 6, label_top + 1), label, font=id_font, fill=(5, 18, 28, 255))
                canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(canvas)
                draw.text((30, 6), 'Saccade v1｜MOT17-02-SDP 追蹤回放', font=title_font, fill='#EAF4FA')
                draw.text((31, 46), 'HR 1280×768 FP16 · G1 NVOFA · CUDA Graph', font=subtitle_font, fill='#9DB6C8')
                counter = f'FRAME {index + 1:04} / {FRAME_COUNT}   觀測輸出 {len(rows):02}'
                counter_width = draw.textbbox((0, 0), counter, font=frame_font)[2]
                draw.text((WIDTH - counter_width - 31, 18), counter, font=frame_font, fill='#72E3CA')
                draw.line((30, 69, WIDTH - 30, 69), fill='#21445B', width=1)
                footer = '預先計算的觀測結果｜原速 30 FPS｜播放速度不代表推論速度'
                footer_width = draw.textbbox((0, 0), footer, font=footer_font)[2]
                draw.text(((WIDTH - footer_width) / 2, 1043), footer, font=footer_font, fill='#AAC2D1')
                if index == 220:
                    canvas.save(POSTER, quality=92, subsampling=0)
                encoder.stdin.write(canvas.tobytes())
                row_count += len(rows)
                if (index + 1) % 100 == 0:
                    print(f'rendered {index + 1}/{FRAME_COUNT}', flush=True)
            encoder.stdin.close()
            code = encoder.wait(timeout=120)
            if code:
                raise RuntimeError(f'ffmpeg exit {code}: {ffmpeg_log.read_text()}')
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise
    probe_command = ['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,color_range,color_space,color_transfer,color_primaries:format=duration,size',
        '-of', 'json', str(VIDEO)]
    probe = json.loads(subprocess.check_output(probe_command, text=True))
    stream, = probe['streams']
    assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
    assert stream['width'] == WIDTH and stream['height'] == HEIGHT
    assert int(stream['nb_read_frames']) == FRAME_COUNT and stream['avg_frame_rate'] == '30/1'
    assert abs(float(probe['format']['duration']) - FRAME_COUNT / FPS) < .0001
    assert sha(PACKETS) == original_packet_sha and sha(INPUT_MANIFEST) == original_manifest_sha
    assert (stream['color_space'], stream['color_primaries'], stream['color_transfer'], stream['color_range']) == ('bt709', 'bt709', 'iec61966-2-1', 'tv')
    for item in source_inputs:
        path = SEQUENCE / 'img1' / item['file']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    decode_command = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(VIDEO), '-an', '-f', 'null', '-']
    decode = subprocess.run(decode_command, capture_output=True, text=True, timeout=120)
    assert decode.returncode == 0, decode.stderr
    receipt = {
        'schema': 'saccade-demo-video-render/1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runtime_baseline': baseline,
        'source_sequence': str(SEQUENCE), 'source_fps': FPS, 'source_frames': FRAME_COUNT,
        'output_frames': FRAME_COUNT, 'duration_seconds': FRAME_COUNT / FPS,
        'display_mode': 'prerecorded observed ResultPacket rows; not live inference or GT',
        'box_semantics': 'Kalman state boxes; measurement_box=false; render clipping only',
        'id_semantics': 'output_id, may change via Bridge; trails contiguous same output_id and UID only',
        'trail_length_frames': 24,
        'source_packet_gzip': str(PACKETS.relative_to(ROOT)),
        'source_packet_gzip_sha256': original_packet_sha,
        'source_packet_payload_sha256': hashlib.sha256(payload).hexdigest(),
        'source_manifest': str(INPUT_MANIFEST.relative_to(ROOT)),
        'source_manifest_sha256': original_manifest_sha,
        'source_jpegs': source_inputs,
        'rows_drawn': row_count,
        'render_script': str(Path(__file__).resolve().relative_to(ROOT)),
        'render_script_sha256': sha(Path(__file__).resolve()),
        'render_command': ['uv', 'run', '--with', 'pillow', 'python', 'docs/demos/report1/methods/render_video_02.py'],
        'ffmpeg_command': ffmpeg_command,
        'ffprobe_command': probe_command, 'ffprobe_result': probe,
        'decode_command': decode_command, 'full_decode': {'returncode': decode.returncode, 'stderr': decode.stderr},
        'source_jpeg_hashes_before_after_equal': True,
        'poster_source_frame_one_based': 221,
        'pillow_version': pillow_version,
        'ffmpeg_version': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'font': {'path': font_file, 'collection_index': font_index, 'sha256': sha(Path(font_file))},
        'render_wall_seconds': time.perf_counter() - started,
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (VIDEO, POSTER)},
        'source_evidence_unchanged': True,
        'no_new_inference_training_quality_scoring_or_gt': True,
        'limitations': ['video colors may change slightly with JPEG decode/resize/H.264 encoding',
                        'rendered observations do not prove identity correctness',
                        'source is prerecorded; playback rate is not inference speed',
                        'original per-frame video; interactive display interpolation does not apply to this MP4']}
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'video': str(VIDEO.relative_to(ROOT)), 'poster': str(POSTER.relative_to(ROOT)),
                      'bytes': VIDEO.stat().st_size, 'frames': FRAME_COUNT, 'duration': FRAME_COUNT / FPS,
                      'render_seconds': receipt['render_wall_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
