#!/usr/bin/env bash
# Read-only audit of the original YOLO26m T3->T1 Mamba-FPN head: fixed source
# blobs and locators from the legacy repository's Git objects, local artifact
# identity, checkpoint containers and the sibling head ONNX. No Python, no
# legacy program execution, no checkpoint import; tools are native C++ written
# for this audit. Usage from the v1 repository root:
#   bash evidence/s01/2026-10-29/audit/capture.sh build-NEW-OUTPUT
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=${1:?usage: capture.sh build-NEW-OUTPUT}
[[ $out == build-* && ! -e $out ]] || { echo "output must be a new build-* path" >&2; exit 2; }
legacy=${LEGACY_ROOT:-/home/ray/developer/ai/saccade}
rev=e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2   # TRAINING_WORKFLOW source snapshot
train_rev=09eb1747                             # first commit with the T1 args keys, 2026-06-14
audit=evidence/s01/2026-10-29/audit
mkdir -p "$out/capture" "$out/work"
cap=$out/capture work=$out/work
run() {
  local name=$1 code=0
  shift
  printf '%q ' "$@" > "$cap/$name.command"; printf '\n' >> "$cap/$name.command"
  "$@" > "$cap/$name.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$code"
  return "$code"
}
src_files=(
  src/saccade/perception/temporal_yolo/mamba_head.py
  src/tracking/mamba_scan.cu
  include/tracking/mamba_scan.cuh
  include/tracking/mamba_scan_plugin.hpp
  src/tracking/mamba_scan_plugin.cpp
  scripts/model/export_mamba_head_onnx.py
  scripts/train/temporal_yolo/train_mamba_gt.py
  scripts/train/temporal_yolo/run_v14replica_yolo26m.sh
  src/saccade/perception/temporal_yolo/mamba_gated_detector.py
  configs/presets/mamba_whole_graph_m.yaml
  LICENSE
)
run environment bash -c 'date -Is; uname -srmo; git --version; c++ --version | head -1; protoc --version; pkg-config --modversion protobuf; unzip -v | head -1; curl --version | head -1'
run legacy-identity bash -c 'git -C "$1" rev-parse HEAD; git -C "$1" status --porcelain=v1 --untracked-files=no; git -C "$1" remote get-url origin; git -C "$1" log -1 --format="%H %cI %s" "$2"; git -C "$1" log -1 --format="%H %cI %s" "$3"' _ "$legacy" "$rev" "$train_rev"
run legacy-repository-metadata gh api repos/raylei50653/saccade --jq '{visibility, private, license: .license.spdx_id, default_branch}'
run source-blobs bash -c 'l=$1; r=$2; shift 2; for f; do printf "%s\t%s\t%s\t" "$f" "$(git -C "$l" rev-parse "$r:$f")" "$(git -C "$l" show "$r:$f" | wc -c)"; git -C "$l" show "$r:$f" | sha256sum | cut -d" " -f1; done' _ "$legacy" "$rev" "${src_files[@]}"
# Exact line locators for every contract clause cited in docs/SCAN_CONTRACT.md.
run source-locators bash -c '
  show() { git -C "$1" show "$2:$3" | grep -n -E "$4" | sed "s|^|$3:|"; }
  l=$1 r=$2 h=src/saccade/perception/temporal_yolo/mamba_head.py k=src/tracking/mamba_scan.cu
  show "$l" "$r" "$h" "^def _selective_scan|^def _cross_scan_mamba|^class MambaBlock|delta = F.softplus|deltaA = |deltaB_u = |h = torch.zeros|h = deltaA|y_t = |ys = ys \+ u \* D|d_inner = d_model \* expand|self.in_proj = |self.conv1d = |padding=d_conv - 1|c_rank = |self.x_proj = |self.dt_proj = |A = torch.arange|self.A_log = |self.D = |x_t = self.conv1d|x_t = F.silu|dt = self.dt_proj|B_ssm = |C_ssm = |A = -torch.exp|y = scan\(|y = y \* F.silu|y = self.out_proj|torch.flip|scans_batched|seq = block|\.mean\(dim=0\)|if C.shape\[-1\] < N|a_per_channel = 1 if|elif self.use_cross_scan|x_up = self.upsample|x_cls = torch.cat|x_reg = torch.cat|nn.PixelShuffle|_validate_cuda_scan_state_count\(n|n <= 0 or n > 32"
  show "$l" "$r" "$k" "softplus_f32|if \(x > 20.0f\)|if \(x < -20.0f\)|return logf|valid_scan_state_count|float h = 0.0f|for \(int t = 0; t < L; t\+\+\)|deltaA = expf|deltaB_u = delta_btd|h = deltaA \* h|acc = C_btn|__shfl_xor_sync\(mask, acc|acc \+= D_d|A\[a_per_channel|\(b \* L \+ t\) \* D_dim|\(b \* L \+ t\) \* N"
  show "$l" "$r" src/tracking/mamba_scan_plugin.cpp "nbInputs < 5|params_.N = a_|a_per_channel = \(a_dims|has_D = \(nbInputs|need_C_broadcast = |kLINEAR|kFLOAT \|\| dtype|outputTypes\[0\]"
  show "$l" "$r" include/tracking/mamba_scan_plugin.hpp "domain|type   =|kSelectiveScan"
  show "$l" "$r" scripts/model/export_mamba_head_onnx.py "saccade::SelectiveScan|domain_s|opset_version|input_names|output_names|mamba_ckpt: str =|in_channels: tuple|output: str ="
' _ "$legacy" "$rev"
run training-era-scan-functions bash "$audit/compare-scan-functions.sh" "$legacy" "$train_rev" "$rev"
run training-era-kernel-diff git -C "$legacy" diff "$train_rev" "$rev" -- src/tracking/mamba_scan.cu include/tracking/mamba_scan.cuh
run scan-source-history git -C "$legacy" log --format='%h %cI %s' "$rev" -- src/tracking/mamba_scan.cu src/saccade/perception/temporal_yolo/mamba_head.py src/tracking/mamba_scan_plugin.cpp scripts/model/export_mamba_head_onnx.py
artifacts=(
  runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt
  runs/mamba_gt_yolo26m_v14replica_t3_t1_gpu_decode/best.ckpt
  runs/mamba_gt_yolo26m_v14replica_t3_t1_gpu_decode/best_recall.ckpt
  models/yolo/mamba_head_26m.onnx
  models/yolo/mamba_head_26m.engine
)
run artifact-identity bash -c 'cd "$1"; shift; for f; do stat -c "%s bytes  mtime=%y  %n" "$f"; sha256sum "$f"; done; git status --porcelain=v1 --ignored -- "$@"' _ "$legacy" "${artifacts[@]}"
run compile-ckpt-inspector c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -I include \
  "$audit/torch_ckpt_inspect.cpp" src/file_io.cpp src/sha256.cpp -o "$work/torch_ckpt_inspect"
declare -A ckpts=([t1]=${artifacts[0]} [gpu_best]=${artifacts[1]} [gpu_recall]=${artifacts[2]})
for name in t1 gpu_best gpu_recall; do
  run "unzip-$name" unzip -q "$legacy/${ckpts[$name]}" -d "$work/$name"
  run "archive-$name" bash -c 'ls "$1"' _ "$work/$name"
  dir=$work/$name/$(ls "$work/$name")
  run "ckpt-$name-top" "$work/torch_ckpt_inspect" "$dir" '*'
  for key in epoch selection mamba_args args student; do
    run "ckpt-$name-$key" "$work/torch_ckpt_inspect" "$dir" "$key"
  done
done
run download-onnx-schema curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/onnx/onnx/d3f6b795aedb48eaecc881bf5e8f5dd6efbe25b3/onnx/onnx.proto -o "$work/onnx.proto"
printf '%s  %s\n' 5943f7cb1d0359edfbc7871cb8ec3095ca942c04dea28ee15c926d51e1185efe "$work/onnx.proto" > "$work/onnx.proto.sha256"
run onnx-schema-hash sha256sum -c "$work/onnx.proto.sha256"
run generate-onnx-schema protoc --cpp_out="$work" -I "$work" "$work/onnx.proto"
# pkg-config supplies native SDK flags; intentional word splitting.
run compile-onnx-dump bash -c 'c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -I include -I "$2" "$1/onnx_graph_dump.cpp" src/file_io.cpp src/sha256.cpp "$2/onnx.pb.cc" $(pkg-config --cflags --libs protobuf) -o "$2/onnx_graph_dump"' _ "$audit" "$work"
run onnx-dump "$work/onnx_graph_dump" "$legacy/models/yolo/mamba_head_26m.onnx"
run onnx-checkpoint-match bash "$audit/match-initializers.sh" "$cap/onnx-dump.log" \
  "$cap/ckpt-t1-student.log" "$cap/ckpt-gpu_best-student.log" "$cap/ckpt-gpu_recall-student.log"
run tool-hashes sha256sum "$audit/torch_ckpt_inspect.cpp" "$audit/onnx_graph_dump.cpp" "$audit/compare-scan-functions.sh" \
  "$audit/match-initializers.sh" "$work/torch_ckpt_inspect" "$work/onnx_graph_dump"
