"""Process each completed capture once; keep serialized receipts for each arm."""
from pathlib import Path
import json
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent.parent
SEQUENCES = ['MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-11-SDP',
             'MOT17-13-SDP', 'MOT17-10-SDP', 'MOT17-02-SDP']
START = time.monotonic()
while True:
    total = 0
    for arm in ['official', 'finetuned']:
        receipt = RUN / arm / 'replay-receipt.json'
        completed = json.loads(receipt.read_text())['sequences'] if receipt.exists() else {}
        for seq in SEQUENCES:
            if seq in completed:
                total += 1
                continue
            capture = RUN / arm / seq / 'inference.json'
            if not capture.exists():
                continue
            try:
                result = json.loads(capture.read_text())
            except json.JSONDecodeError:
                continue
            if result.get('status') != 'PASS':
                continue
            subprocess.run([sys.executable, str(RUN / 'replay-tools/replay_arm.py'), arm, seq], check=True)
            total += 1
    if total == 14:
        print('ALL14_REPLAYS_COMPLETE', flush=True)
        break
    if time.monotonic() - START > 3600:
        raise TimeoutError('capture wait exceeded 1hour')
    time.sleep(2)
