"""Validate one complete two-sequence detector capture and replay fixed G1 records."""
from pathlib import Path
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time

ROOT = Path('/home/ray/developer/ai/saccade_v1')
RUN = ROOT / 'build/yolo26m-threeway-20261111'
TOOLS = RUN / 'replay-tools'
RECORDS = Path('/home/ray/developer/ai/saccade_v1-worktrees/hr-records-v2')
EXE = ROOT / 'build/cpu/track_offline'
HEADER = struct.Struct('<4sIIQQ')
FRAME = struct.Struct('<QQ12I')
ROW = struct.Struct('<5f3I')
FRAME_BYTES = 9664

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_scdr(path, expected):
    payload = path.read_bytes()
    assert HEADER.unpack_from(payload) == (b'SCDR', 1, FRAME_BYTES, 1, expected)
    assert len(payload) == 28 + expected * FRAME_BYTES
    totals = dict(kept=0, suppressed=0, class_dropped=0, degenerate=0, low_score=0, topk_dropped=0)
    for frame in range(expected):
        offset = 28 + frame * FRAME_BYTES
        h = FRAME.unpack_from(payload, offset)
        epoch, fid, schema, count, capacity, flags, done, kept, suppressed, low, topk, classes, degenerate, reserved = h
        assert (epoch, fid, schema, capacity, flags, done, reserved) == (1, frame, 1, 300, 0, 0x34504f46, 0)
        assert count <= 300 and count == kept + suppressed
        assert low + topk + classes + degenerate + count == 20160
        ids = set()
        counts = {1: 0, 2: 0}
        previous_score = math.inf
        previous_id = -1
        for i in range(count):
            x1, y1, x2, y2, score, cls, cid, flag = ROW.unpack_from(payload, offset + 64 + 32 * i)
            assert all(math.isfinite(v) for v in (x1, y1, x2, y2, score))
            assert x2 > x1 and y2 > y1 and .001 <= score <= 1 and cls == 0 and flag in counts
            assert cid < 20160 and cid not in ids
            assert score < previous_score or (score == previous_score and cid > previous_id)
            previous_score, previous_id = score, cid
            ids.add(cid);counts[flag] += 1
        assert (counts[1], counts[2]) == (kept, suppressed)
        assert payload[offset+64+count*32:offset+FRAME_BYTES] == bytes((300-count)*32)
        for name, value in zip(totals, (kept, suppressed, classes, degenerate, low, topk)):
            totals[name] += value
    return {'frames': expected, 'all_frame_status_ok': True, 'all_header_and_row_invariants_pass': True, **totals}

def main():
    arm = sys.argv[1]
    assert arm in ('official', 'finetuned')
    assert not any(key.startswith('SACCADE_CFG_') for key in os.environ)
    receipt_path = RUN / arm / 'replay-receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        'arm': arm, 'scope': 'fixed main-derived G1 motion; current CPU tracker defaults; not full native session acceptance',
        'track_offline_sha256': sha(EXE), 'sequences': {}}
    for seq, frames in [('MOT17-10-SDP', 654), ('MOT17-02-SDP', 600)]:
        if len(sys.argv) > 2 and seq != sys.argv[2]:
            continue
        capture = RUN / arm / seq
        inference = json.loads((capture / 'inference.json').read_text())
        assert inference['status'] == 'PASS' and inference['frames'] == frames
        det = capture / 'detections.scdr'
        assert sha(det) == inference['scdr_sha256']
        alias = RUN / arm / (seq + '.scdr')
        assert not alias.exists() and not alias.is_symlink()
        alias.symlink_to(Path(seq) / 'detections.scdr')
        motion = RECORDS / (seq + '.scmo')
        validation = validate_scdr(det, frames)
        output = RUN / arm / 'mot'
        output.mkdir(exist_ok=True, parents=True)
        mot = output / (seq + '.txt')
        packets = output / (seq + '.packets.bin')
        config = output / (seq + '.config')
        cmd = [str(EXE), str(det), str(motion), str(mot), '--resolved-config', str(config), '--packets', str(packets)]
        before = {'detection_sha256': sha(det), 'motion_sha256': sha(motion), 'binary_sha256': sha(EXE)}
        start = time.monotonic()
        result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (output / (seq + '.replay.log')).write_text(result.stdout)
        assert result.returncode == 0, result.stdout
        packet_cmd = [str(TOOLS / 'validate_packets'), str(packets), str(frames)]
        checked = subprocess.run(packet_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (output / (seq + '.packet-validation.log')).write_text(checked.stdout)
        assert checked.returncode == 0, checked.stdout
        assert before == {'detection_sha256': sha(det), 'motion_sha256': sha(motion), 'binary_sha256': sha(EXE)}
        rec = {**before, 'input_validation': validation, 'command': cmd, 'exit': result.returncode,
               'elapsed_s': time.monotonic()-start, 'mot_sha256': sha(mot), 'packets_sha256': sha(packets),
               'resolved_config_sha256': sha(config), 'packet_validation_command': packet_cmd,
               'packet_validation_exit': checked.returncode, 'packet_validation': checked.stdout.strip()}
        receipt['sequences'][seq] = rec
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        print(arm, seq, checked.stdout.strip(), flush=True)
    assert receipt['track_offline_sha256'] == sha(EXE)

if __name__ == '__main__':
    main()
