#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=$PWD/build-preprocess-initcheck-control
run() {
  local name=$1 status=0
  shift
  printf '%q ' "$@" > "$out/$name.command"
  printf '\n' >> "$out/$name.command"
  /usr/bin/time -v -o "$out/$name.time" "$@" > "$out/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$out/$name.exit"
  printf '%s exit=%s\n' "$name" "$status"
}
run build2 c++ -std=c++20 -O2 -Wall -Wextra -Werror -I/opt/cuda/include \
  "$out/repro.cpp" -L/opt/cuda/lib64 -Wl,-rpath,/opt/cuda/lib64 -lcudart -lnvjpeg -o "$out/repro"
[[ $(cat "$out/build2.exit") == 0 ]]
run versions2 bash -c 'nvidia-smi; /opt/cuda/bin/nvcc --version; /opt/cuda/bin/compute-sanitizer --version; ldd "$1"; sha256sum "$1" /opt/cuda/lib64/libnvjpeg.so /opt/cuda/lib64/libcudart.so; git rev-parse HEAD; git status --porcelain' _ "$out/repro"
run djpeg2 bash -c 'djpeg -dct int -nosmooth -rgb "$1" > "$2"' _ fixtures/preprocess/baseline.jpg "$out/baseline.ppm"
python3 - "$out" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
magic,wh,maxval,data=(root/'baseline.ppm').read_bytes().split(b'\n',3)
assert (magic,wh,maxval)==(b'P6',b'35 27',b'255')
assert len(data)==35*27*3
(root/'baseline.rgb').write_bytes(data[::3]+data[1::3]+data[2::3])
PY
for mode in v2-dirty v2-zero v2-a5 hybrid; do
  run "$mode-normal" "$out/repro" "$mode" fixtures/preprocess/baseline.jpg "$out/baseline.rgb" "$out/$mode-normal.rgb"
  for tool in memcheck initcheck; do
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    ((available >= 4194304))
    run "$mode-$tool" timeout --signal=TERM --kill-after=15s 120s \
      /opt/cuda/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 --print-limit 10 \
      "$out/repro" "$mode" fixtures/preprocess/baseline.jpg "$out/baseline.rgb" "$out/$mode-$tool.rgb"
  done
done
sha256sum "$out/repro.cpp" "$out/control.sh" "$out/baseline.rgb" "$out"/*-*.rgb > "$out/output2.sha256"
