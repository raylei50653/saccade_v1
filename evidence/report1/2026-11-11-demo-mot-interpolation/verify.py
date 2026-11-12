#!/usr/bin/env python3
"""Verify frozen receipt and replay F09/module/MOT parity without input edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    checks = 0
    for line in (EVIDENCE / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert sha(EVIDENCE / name) == expected, name
        checks += 1
    frozen = json.loads((EVIDENCE / 'parity.json').read_text())
    assert frozen['status'] == 'PASS'
    with tempfile.TemporaryDirectory(prefix='saccade-mot-gap-parity-') as directory:
        output = Path(directory) / 'replay.json'
        subprocess.run([sys.executable, '-B', str(ROOT / frozen['checker']['path']),
                        '--output', str(output)], check=True)
        replay = json.loads(output.read_text())
    for key in set(frozen) - {'created_utc', 'node'}:
        assert replay[key] == frozen[key], key
    print(f'PASS: {checks} receipt checksums; independent full replay matches frozen numeric evidence')


if __name__ == '__main__':
    main()
