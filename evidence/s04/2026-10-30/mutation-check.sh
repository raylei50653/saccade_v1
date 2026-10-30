#!/usr/bin/env bash
# S04 sensitivity checks. Mutate one clause of a private source copy, rebuild
# incrementally, and require the unchanged targeted test to reject the mutant.
# Usage: mutation-check.sh SOURCE_ROOT OUTPUT_DIR BUNDLE BUNDLE_SHA256
#   engine:GROUP  rebuild the head engine from the bundle with the mutant, then
#                 head_trt_tests --group GROUP on it (network mutants)
#   suite:GROUP   head_trt_tests --group GROUP on the baseline engine (harness)
#   cpu:TARGET    head_reference_tests or head_weights_tests (reference, converter)
set -euo pipefail
src=$(cd "${1:?source root}" && pwd)
out=${2:?output directory}
bundle=${3:?bundle}
bundle_sha=${4:?bundle sha256}
mkdir -p "$out"
out=$(cd "$out" && pwd)
[[ ! -e "$out/source" ]] || { echo 'mutation source already exists' >&2; exit 2; }
mkdir "$out/source"
tar -C "$src" --exclude='./.git' --exclude='./build-*' -cf - . | tar -C "$out/source" -xf -
private=$out/source
build=$out/build
trt=${SACCADE_TENSORRT_ROOT:-/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk}
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
targets=(head_trt head_trt_tests head_reference_tests head_weights_tests)
cmake -S "$private" -B "$build" -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_TENSORRT=ON \
  "-DSACCADE_TENSORRT_ROOT=$trt" "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 \
  > "$out/configure.log" 2>&1
cmake --build "$build" --parallel 8 --target "${targets[@]}" > "$out/baseline-build.log" 2>&1
timeout 600 "$build/head_trt" build "$bundle" "$bundle_sha" "$out/baseline.engine" "$out/baseline-build-report.txt" > "$out/baseline-engine.log" 2>&1
engine_sha=$(sha256sum "$out/baseline.engine" | cut -d' ' -f1)
timeout 900 "$build/head_trt_tests" "$out/baseline.engine" "$engine_sha" "$bundle" "$bundle_sha" > "$out/baseline.log" 2>&1
timeout 900 "$build/head_reference_tests" > "$out/baseline-reference.log" 2>&1
timeout 900 "$build/head_weights_tests" > "$out/baseline-weights.log" 2>&1
# Format: name|relative path|sed expression|kind:group
mutations=(
  'pixel_shuffle_dcr|src/head_trt_network.cpp|s/shape({1, kHeadDModel, kHeadReduction, kHeadReduction, small, small})/shape({1, kHeadReduction, kHeadReduction, kHeadDModel, small, small})/; s/permutation({0, 1, 4, 2, 5, 3})/permutation({0, 3, 4, 1, 5, 2})/|engine:numeric'
  'conv1d_anticausal|src/head_trt_network.cpp|s/shape({0, 0, 0, 0}), shape({4, kHeadDInner, 1, L}), "mamba.conv1d.causal"/shape({0, 0, 0, kHeadConvKernel - 1}), shape({4, kHeadDInner, 1, L}), "mamba.conv1d.causal"/|engine:numeric'
  'concat_reversed|src/head_trt_network.cpp|s/ITensor\* parts\[\] = {a, shuffled};/ITensor* parts[] = {shuffled, a};/|engine:numeric'
  'xz_halves_swapped|src/head_trt_network.cpp|s/shape({0, 0, 0}), shape({4, L, kHeadDInner}), "mamba.x_half"/shape({0, 0, kHeadDInner}), shape({4, L, kHeadDInner}), "mamba.x_half"/; s/shape({0, 0, kHeadDInner}), shape({4, L, kHeadDInner}), "mamba.z_half"/shape({0, 0, 0}), shape({4, L, kHeadDInner}), "mamba.z_half"/|engine:numeric'
  'dt_b_swapped|src/head_trt_network.cpp|s/shape({0, 0, 0}), shape({4, L, kHeadDState}), "mamba.dt"/shape({0, 0, kHeadDState}), shape({4, L, kHeadDState}), "mamba.dt"/; s/shape({0, 0, kHeadDState}), shape({4, L, kHeadDState}), "mamba.B"/shape({0, 0, 0}), shape({4, L, kHeadDState}), "mamba.B"/|engine:numeric'
  'delta_bias_negated|src/head_trt_network.cpp|s/\*dt_bias, ElementWiseOperation::kSUM), "mamba.delta"/*dt_bias, ElementWiseOperation::kSUB), "mamba.delta"/|engine:numeric'
  'residual_subtracted|src/head_trt_network.cpp|s/\*tokens, ElementWiseOperation::kSUM), "mamba.residual"/*tokens, ElementWiseOperation::kSUB), "mamba.residual"/|engine:numeric'
  'a_not_negated|src/head_trt_network.cpp|s/UnaryOperation::kNEG), "mamba.A"/UnaryOperation::kABS), "mamba.A"/|engine:numeric'
  'gate_without_silu|src/head_trt_network.cpp|s/\*silu(z, "mamba.z")/*z/|engine:numeric'
  'output_names_swapped|src/head_trt_network.cpp|s/outputs\[k\]->setName(kHeadOutputNames\[k\]);/outputs[k]->setName(kHeadOutputNames[k == 1 ? 2 : k == 2 ? 1 : k]);/|engine:numeric'
  'level_weights_swapped|src/head_trt_network.cpp|s/return w_.at(prefix_ + key);/return w_.at((prefix_ == "p4." ? std::string("p5.") : prefix_ == "p5." ? std::string("p4.") : prefix_) + key);/|engine:numeric'
  'tf32_enabled|src/head_trt_network.cpp|s/config_->clearFlag(BuilderFlag::kTF32);/config_->setFlag(BuilderFlag::kTF32);/|engine:numeric'
  'tactic_guard_ignores_winograd|src/head_trt_network.cpp|s/{"wngd", "winograd", "fft", /{"fft", /|suite:io_contract'
  'output_check_skipped|src/scan_trt_frame.cpp|s/^            if (check_outputs)$/            if (false)/|suite:atomicity'
  'input_check_skipped|src/scan_trt_frame.cpp|s/if (check_inputs) scan_frame_check_finite(/if (false) scan_frame_check_finite(/|suite:atomicity'
  'gate_skips_last_status|src/scan_trt_frame.cpp|s/list.count = static_cast<unsigned>(slot.statuses.size());/list.count = static_cast<unsigned>(slot.statuses.size()) - 1;/|suite:atomicity'
  'publish_first_output_only|src/scan_trt_frame.cpp|s/scan_frame_publish(ctl, slot.payload, payload_count,/scan_frame_publish(ctl, slot.payload, outputs.front().count,/|suite:numeric'
  'missing_consumed_wait|src/scan_trt_frame.cpp|s/if (k >= S) wait(slot.consumed, d, k - S, "consumed", s);/\/\/ no consumed wait/|suite:ownership'
  'spec_output_shape_unchecked|src/scan_trt_frame.cpp|s/require(same_shape(engine->getTensorShape(t.name.c_str()), t.shape), "engine output shape differs from the spec: " + t.name);/(void)0;/|suite:io_contract'
  'ref_pixel_shuffle_dcr|src/head_reference.cpp|s/const auto in_c = dcr ? (i \* r + j) \* C + c : c \* r \* r + i \* r + j;/const auto in_c = !dcr ? (i * r + j) * C + c : c * r * r + i * r + j;/|cpu:head_reference_tests'
  'ref_conv1d_offset|src/head_reference.cpp|s/const auto s = anticausal ? t + k : t + k - (K - 1);/const auto s = anticausal ? t + k : t + k - (K - 2);/|cpu:head_reference_tests'
  'ref_scan_without_d|src/head_reference.cpp|s/^            if (d) {$/            if (false) {/|cpu:head_reference_tests'
  'ref_sigma_zero_for_dot_products|src/head_reference.cpp|s/return std::sqrt(var + 2 \* terms \* kUnit \* kUnit \* mag \* mag);/return 0 * std::sqrt(var + 2 * terms * kUnit * kUnit * mag * mag);/|cpu:head_reference_tests'
  'converter_ignores_strides|src/head_weights.cpp|s/if (!ref.contiguous()) reject(/if (false) reject(/|cpu:head_weights_tests'
  'converter_level_swap|src/head_weights.cpp|s/n = std::to_string(i);/n = std::to_string(i == 0 ? 0 : 3 - i);/|cpu:head_weights_tests'
  'bundle_skips_tensor_sha|src/head_weights.cpp|s/if (sha256_hex(data.substr(offset, size)) != sha) fail(/if (false) fail(/|cpu:head_weights_tests'
)
status=0
printf 'mutation\tcheck\texit\tresult\n' > "$out/results.tsv"
for entry in "${mutations[@]}"; do
  IFS='|' read -r name file expr check <<< "$entry"
  sed "$expr" "$src/$file" > "$private/$file"
  if cmp -s "$src/$file" "$private/$file"; then
    echo "$name: mutation did not apply" >&2
    printf '%s\t%s\t-\tNOT_APPLIED\n' "$name" "$check" >> "$out/results.tsv"
    status=1
    continue
  fi
  diff -u "$src/$file" "$private/$file" > "$out/$name.diff" || true
  if ! cmake --build "$build" --parallel 8 --target "${targets[@]}" > "$out/$name.build.log" 2>&1; then
    echo "$name: mutant did not compile (not a detection)" >&2
    printf '%s\t%s\tbuild\tINVALID\n' "$name" "$check" >> "$out/results.tsv"
    cp "$src/$file" "$private/$file"
    status=1
    continue
  fi
  kind=${check%%:*} group=${check#*:}
  case $kind in
    engine) cmd=(timeout 900 bash -c '"$1" build "$2" "$3" "$4" "$5" && "$6" "$4" "$(sha256sum "$4" | cut -d" " -f1)" "$2" "$3" --group "$7"' _
                 "$build/head_trt" "$bundle" "$bundle_sha" "$out/$name.engine" "$out/$name.build-report.txt" "$build/head_trt_tests" "$group") ;;
    suite) cmd=(timeout 900 "$build/head_trt_tests" "$out/baseline.engine" "$engine_sha" "$bundle" "$bundle_sha" --group "$group") ;;
    cpu) cmd=(timeout 900 "$build/$group") ;;
  esac
  printf '%q ' "${cmd[@]}" > "$out/$name.command"; printf '\n' >> "$out/$name.command"
  code=0
  "${cmd[@]}" > "$out/$name.log" 2>&1 || code=$?
  cp "$src/$file" "$private/$file"
  rm -f "$out/$name.engine"
  if [[ $code == 0 ]]; then
    echo "$name [$check]: INSENSITIVE"
    printf '%s\t%s\t%s\tINSENSITIVE\n' "$name" "$check" "$code" >> "$out/results.tsv"
    status=1
  else
    echo "$name [$check]: detected (exit $code)"
    printf '%s\t%s\t%s\tDETECTED\n' "$name" "$check" "$code" >> "$out/results.tsv"
  fi
done
exit "$status"
