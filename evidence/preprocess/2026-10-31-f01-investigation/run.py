#!/usr/bin/env python3
"""Sequential, bounded SDK investigation. Every actual child status is retained."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CAP = ROOT / "capture"
SAN = "/opt/cuda/bin/compute-sanitizer"

def run(name, command):
    if (CAP / (name + ".command")).exists():
        raise RuntimeError(f"refusing to overwrite {name}")
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available < 4194304:
        raise RuntimeError("need 4 GiB MemAvailable")
    (CAP / (name + ".command")).write_text(shlex.join(command) + "\n")
    (CAP / (name + ".memory")).write_text(f"time={datetime.datetime.now().astimezone().isoformat()}\navailable_kib={available}\n")
    with (CAP / (name + ".log")).open("w") as stream:
        proc = subprocess.run(["/usr/bin/time", "-v", "-o", str(CAP / (name + ".time")), "timeout", "--signal=TERM", "--kill-after=10s", "60s", *command], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    (CAP / (name + ".exit")).write_text(str(proc.returncode) + "\n")
    output = (CAP / (name + ".log")).read_text()
    interesting = [s for s in output.splitlines() if 'ERROR SUMMARY' in s or s.startswith('result=') or s.startswith('nvjpeg=')]
    print(name, "exit=" + str(proc.returncode), *interesting, sep="\n", flush=True)
    return proc.returncode

def experiment(name, *, backend="hw", fmt="rgb", flags="default", preallocate="preallocate", fill="uninit", repeats=2, old=False, tools=("normal", "memcheck", "initcheck"), print_limit=5):
    prefix = ["env", "LD_LIBRARY_PATH=" + str(ROOT / "old-nvjpeg")] if old else []
    for tool in tools:
        case = name + "-" + tool
        program = [str(ROOT / "repro"), backend, fmt, flags, preallocate, fill, str(repeats), "fixtures/preprocess/baseline.jpg", str(ROOT / "baseline.rgb"), str(ROOT / (case + ".bin"))]
        sanitizer = [] if tool == "normal" else [SAN, "--tool", tool, "--error-exitcode", "99", "--print-limit", str(print_limit)]
        run(case, prefix + sanitizer + program)

if __name__ == '__main__':
    phase = sys.argv[1]
    if phase == 'identity':
        run('identity', ['bash', '-c', 'date -Is; git rev-parse HEAD; git status --porcelain; uname -srmo; nvidia-smi; /opt/cuda/bin/nvcc --version; /opt/cuda/bin/compute-sanitizer --version; ldd "$1/repro"; env LD_LIBRARY_PATH="$1/old-nvjpeg" ldd "$1/repro"; sha256sum "$1/repro" "$1/repro.cu" "$1/old-nvjpeg/libnvjpeg.so.13" /opt/cuda/lib64/libnvjpeg.so.13 /opt/cuda/lib64/libcudart.so.13', '_', str(ROOT)])
        run('djpeg', ['bash', '-c', 'djpeg -dct int -nosmooth -rgb fixtures/preprocess/baseline.jpg > "$1/baseline.ppm"', '_', str(ROOT)])
        magic, dims, maxval, rgb = (ROOT / 'baseline.ppm').read_bytes().split(b'\n', 3)
        assert (magic, dims, maxval) == (b'P6', b'35 27', b'255')
        assert len(rgb) == 35 * 27 * 3
        (ROOT / 'baseline.rgb').write_bytes(rgb[::3] + rgb[1::3] + rgb[2::3])
    elif phase == 'baseline':
        experiment('new-rgb')
        experiment('old-rgb', old=True)
        experiment('new-rgb-no-pipeline', flags='no-pipeline')
        experiment('new-rgb-no-preallocate', preallocate='no-preallocate')
        experiment('new-rgb-output-zero', fill='zero')
    elif phase == 'formats':
        for fmt in ('rgbi', 'yuv', 'y', 'nv12'):
            experiment('new-' + fmt, fmt=fmt)
    elif phase == 'full-pixels':
        experiment('new-rgb-once-full', repeats=1, tools=('initcheck',), print_limit=0)
    elif phase == 'reader-controls':
        experiment('skip-uninit', backend='skip', repeats=1, tools=('memcheck','initcheck'))
        experiment('skip-zero', backend='skip', fill='zero', repeats=1, tools=('initcheck',))
        experiment('hybrid-rgb', backend='hybrid')
    elif phase == 'hashes':
        records = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.bin'))}
        (ROOT / 'output-hashes.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(records, indent=2))
    else:
        raise ValueError(phase)
