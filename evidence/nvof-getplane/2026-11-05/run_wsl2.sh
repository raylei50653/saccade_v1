#!/usr/bin/env bash
# Builds repro_getplane on WSL2/Linux and records every configuration in capture/.
# Run under the GPU lock: flock ~/.cache/saccade-v1/gpu.lock bash run_wsl2.sh BUILD_DIR
set -u
here=$(cd "$(dirname "$0")" && pwd)
build=$1
san=/opt/cuda/bin/compute-sanitizer
cap=$here/capture
rm -rf "$cap"; mkdir -p "$cap" "$build"
g++ -std=c++17 -O2 -Wall -Wextra -I"$here/../../../third_party/nvof" -I/opt/cuda/include "$here/repro_getplane.cpp" \
  -L/usr/lib/wsl/lib -Wl,-rpath,/usr/lib/wsl/lib -lcuda -ldl -o "$build/repro_getplane" > "$cap/build.log" 2>&1 || { cat "$cap/build.log"; exit 1; }
cd "$build" || exit 1

case_run() {
  local name=$1 status=0
  shift
  timeout -k 10 300 "$@" > "$cap/$name.log" 2>&1 || status=$?
  printf 'exit=%s\n' "$status" >> "$cap/$name.log"
  printf '%-24s exit=%-3s getplane=%-5s other_reports=%-3s %s\n' "$name" "$status" \
    "$(grep -c 'on CUDA API call to cuArrayGetPlane' "$cap/$name.log")" \
    "$(grep -E '^=========' "$cap/$name.log" | grep -Evc 'cuArrayGetPlane|Saved host backtrace|Host Frame|COMPUTE-SANITIZER|ERROR SUMMARY|^========= *$')" \
    "$(grep -o 'ERROR SUMMARY: .*' "$cap/$name.log" | head -1)"
}
memcheck=("$san" --tool memcheck --print-limit 0 --error-exitcode 99)
case_run plain-execute-2 ./repro_getplane 2
case_run explicit-init-only "${memcheck[@]}" --report-api-errors explicit ./repro_getplane 0
case_run explicit-execute-1 "${memcheck[@]}" --report-api-errors explicit ./repro_getplane 1
case_run explicit-execute-2 "${memcheck[@]}" --report-api-errors explicit ./repro_getplane 2
case_run explicit-execute-30 "${memcheck[@]}" --report-api-errors explicit ./repro_getplane 30
case_run extended-execute-2 "${memcheck[@]}" --report-api-errors extended ./repro_getplane 2
case_run no-report-execute-2 "${memcheck[@]}" --report-api-errors no ./repro_getplane 2

{
  echo '== date'; date -Is
  echo '== os'; uname -srmo
  echo '== nvidia-smi'; nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv
  nvidia-smi | sed -n '3p'
  echo '== compute-sanitizer'; "$san" --version | sed -n '1p;3p'
  echo '== compiler'; g++ --version | head -1
  echo '== cuda headers'; grep -m1 'define CUDA_VERSION' /opt/cuda/include/cuda.h
  echo '== libraries'
  sha256sum /usr/lib/wsl/lib/libnvidia-opticalflow.so.1 /usr/lib/wsl/lib/libnvcuvid.so.1 /usr/lib/wsl/lib/libcuda.so.1
  echo '== ldd'; ldd ./repro_getplane | grep -E 'cuda|nvidia|nvcuvid' | sed 's/ (0x.*//'
  echo '== repro'; (cd "$here" && sha256sum repro_getplane.cpp)
  (cd "$here/../../../third_party/nvof" && sha256sum nvOpticalFlowCommon.h nvOpticalFlowCuda.h)
} > "$cap/identity.log" 2>&1
