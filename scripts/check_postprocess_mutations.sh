#!/usr/bin/env bash
# F04 negative controls. Mutate only a disposable copied CUDA source tree.
# Usage: bash scripts/check_postprocess_mutations.sh BASELINE_BUILD build-NEW-STAGE [--check-only]
# --check-only verifies exact mutation matches/diffs without compiling or using GPU.
set -euo pipefail

usage() {
  echo 'usage: check_postprocess_mutations.sh BASELINE_BUILD build-NEW-STAGE [--check-only]' >&2
  exit 2
}
[[ $# -ge 2 && $# -le 3 ]] || usage
baseline_arg=$1
stage_name=$2
check_only=false
if [[ $# == 3 ]]; then [[ $3 == --check-only ]] || usage; check_only=true; fi
repo=$(git rev-parse --show-toplevel)
cd "$repo"
[[ $stage_name == build-* && $stage_name != */* && ! -e $stage_name && ! -L $stage_name ]] || {
  echo 'stage must be a new build-* directory at the repository root' >&2
  exit 2
}
[[ -d $baseline_arg ]] || { echo 'baseline build directory is missing' >&2; exit 2; }
baseline=$(cd "$baseline_arg" && pwd)
[[ -f $baseline/CMakeCache.txt ]] || { echo 'baseline CMakeCache.txt is missing' >&2; exit 2; }
if [[ $check_only == false && ! -x $baseline/postprocess_cuda_tests ]]; then
  echo 'baseline postprocess_cuda_tests executable is missing' >&2
  exit 2
fi
for dependency in awk bash cmake cp diff find git rg sha256sum sort timeout touch; do
  command -v "$dependency" >/dev/null || { echo "missing dependency: $dependency" >&2; exit 2; }
done
timeout_seconds=${SACCADE_POSTPROCESS_MUTATION_TIMEOUT:-300}
[[ $timeout_seconds =~ ^[1-9][0-9]*$ ]] || usage
cache_value() { awk -v key="$1" 'index($0,key ":")==1 {sub(/^[^=]*=/, ""); print; exit}' "$baseline/CMakeCache.txt"; }
nvcc=$(cache_value CMAKE_CUDA_COMPILER)
cxx=$(cache_value CMAKE_CXX_COMPILER)
[[ -n $nvcc && -n $cxx ]] || { echo 'baseline compiler identities are missing' >&2; exit 2; }

mkdir "$stage_name"
stage=$(cd "$stage_name" && pwd)
cap=$stage/capture
source_tree=$stage/source
mkdir "$cap" "$source_tree"
finished=false
finish() {
  local status=$?
  trap - EXIT
  printf 'exit=%s\ncomplete=%s\ncheck_only=%s\nfinished=%s\n' \
    "$status" "$finished" "$check_only" "$(date -Is)" > "$cap/status.txt"
  exit "$status"
}
trap finish EXIT

# Only the small build inputs are copied: no historical evidence or build tree.
cp -a -- CMakeLists.txt cmake include src cuda tests tools fixtures "$source_tree/"
cp -a -- cuda/postprocess.cu "$cap/baseline-postprocess.cu"
git rev-parse HEAD > "$cap/source-commit.txt"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status.txt"
printf 'baseline_build=%s\nsource_tree=%s\narchitecture=120\ncheck_only=%s\nstarted=%s\n' \
  "$baseline" "$source_tree" "$check_only" "$(date -Is)" > "$cap/identity.txt"
rg --files CMakeLists.txt cmake include src cuda tests tools fixtures | sort | while IFS= read -r path; do
  sha256sum -- "$path"
done > "$cap/original-source.sha256"
sha256sum docs/POSTPROCESS_POLICY.md >> "$cap/original-source.sha256"
sha256sum scripts/check_postprocess_mutations.sh "$baseline/CMakeCache.txt" > "$cap/control-inputs.sha256"
if [[ -x $baseline/postprocess_cuda_tests ]]; then
  sha256sum "$baseline/postprocess_cuda_tests" > "$cap/baseline-binary.sha256"
fi

guard_source() {
  sha256sum --check --status "$cap/original-source.sha256" || {
    echo 'original source changed during mutation checks; results are incomplete' >&2
    return 2
  }
}
run() {
  local name=$1 status=0
  shift
  guard_source
  printf '%q ' "$@" > "$cap/$name.command"
  printf '\n' >> "$cap/$name.command"
  "$@" > "$cap/$name.stdout" 2> "$cap/$name.stderr" || status=$?
  printf '%s\n' "$status" > "$cap/$name.exit"
  guard_source
  return "$status"
}

# Literal replacement rather than a regexp-based patch. Every occurrence count
# is checked; a changed production source cannot silently weaken a mutation.
replace_exact() {
  local needle=$1 replacement=$2 expected=${3:-1}
  local temporary=$source_tree/cuda/postprocess.cu.next
  awk -v needle="$needle" -v replacement="$replacement" -v expected="$expected" \
      -v hits="$cap/$variant.matches" '
    {
      line=$0; output=""
      while ((position=index(line,needle)) != 0) {
        output=output substr(line,1,position-1) replacement
        line=substr(line,position+length(needle)); count++
      }
      print output line
    }
    END {
      printf "expected=%d actual=%d needle=%s\n",expected,count,needle >> hits
      if (count != expected) {
        printf "mutation literal match count: expected %d, got %d: %s\n",expected,count,needle > "/dev/stderr"
        exit 2
      }
    }
  ' "$source_tree/cuda/postprocess.cu" > "$temporary"
  mv -- "$temporary" "$source_tree/cuda/postprocess.cu"
}
mutate() {
  case "$variant" in
    iou-greater-or-equal)
      replace_exact 'if (overlap > kPostprocessNmsIou)' 'if (overlap >= kPostprocessNmsIou)'
      diagnostic='nms-iou-exact-half: header differs' ;;
    suppressed-can-suppress)
      replace_exact 'if (result[j].flags != kDetectionKept) continue;' 'if (result[j].flags == 0) continue;'
      diagnostic='nms-greedy-chain: header differs' ;;
    class-logit-argmax)
      replace_exact 'float best_score = -1.0f;' 'float best_score = -1.0f; float best_logit = -FLT_MAX;'
      replace_exact 'if (score > best_score) { best_score = score; best_class = c; }' \
        'if (logit > best_logit) { best_logit = logit; best_score = score; best_class = c; }'
      diagnostic='class-rounded-sigmoid-saturation: header differs' ;;
    reverse-candidate-id-ties)
      replace_exact '(rival == score && other < id)' '(rival == score && other > id)'
      diagnostic='level-first-last-layout: row 0 bits differ' ;;
    wrong-p3-stride)
      replace_exact 'local = id, side = 80, stride = 8;' 'local = id, side = 80, stride = 16;'
      diagnostic='single-asymmetric-map-no-clip: row 0 bits differ' ;;
    swap-left-top)
      replace_exact 'const float raw[4]{__fsub_rn(ax, reg[0]), __fsub_rn(ay, reg[1]),' \
        'const float raw[4]{__fsub_rn(ax, reg[1]), __fsub_rn(ay, reg[0]),'
      diagnostic='single-asymmetric-map-no-clip: row 0 bits differ' ;;
    swap-xy-mapping)
      replace_exact 'const float sx = static_cast<float>(request.mapping.scale_x);' \
        'const float sx = static_cast<float>(request.mapping.scale_y);'
      replace_exact 'const float sy = static_cast<float>(request.mapping.scale_y);' \
        'const float sy = static_cast<float>(request.mapping.scale_x);'
      diagnostic='single-asymmetric-map-no-clip: row 0 bits differ' ;;
    kept-only-pool)
      replace_exact 'else header.count = count;' 'else header.count = header.kept;'
      diagnostic='nms-equal-score-id-suppressed-pool: header differs' ;;
    ignore-producer-flags)
      replace_exact 'if (producer->flags != 0) flags |= kPostprocessUpstream;' \
        'if (false && producer->flags != 0) flags |= kPostprocessUpstream;'
      diagnostic='producer-upstream-failure: header differs' ;;
    no-row-reset)
      replace_exact 'for (unsigned i = 0; i < sizeof(DetectionCandidates); ++i) bytes[i] = 0;' \
        'for (unsigned i = 0; i < sizeof(DetectionCandidatesHeader); ++i) bytes[i] = 0;'
      diagnostic='empty: row 0 bits differ' ;;
    *) echo "unknown mutation: $variant" >&2; return 2 ;;
  esac
}

baseline_args=(--no-held-stream)
f03_dumps=$(cache_value SACCADE_POSTPROCESS_F03_DUMPS)
if [[ -n $f03_dumps ]]; then baseline_args+=(--f03-dumps "$f03_dumps"); fi
if [[ $check_only == false ]]; then
  # Reject an already failing baseline before generating negative-control claims.
  run environment nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
  run compiler-cuda "$nvcc" --version
  run compiler-cxx "$cxx" --version
  run baseline-existing timeout "$timeout_seconds" "$baseline/postprocess_cuda_tests" "${baseline_args[@]}"
  run configure cmake -S "$source_tree" -B "$stage/build" -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CXX_COMPILER=$cxx" "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_POSTPROCESS=ON -DSACCADE_ENABLE_TENSORRT=OFF \
    -DSACCADE_ENABLE_MODEL_TRT=OFF -DSACCADE_ENABLE_PREPROCESS=OFF
  run build-baseline cmake --build "$stage/build" --target postprocess_cuda_tests --parallel 2
  sha256sum "$stage/build/postprocess_cuda_tests" "$stage/build/libsaccade_postprocess_cuda.a" > "$cap/staged-baseline-binary.sha256"
  run baseline-staged timeout "$timeout_seconds" "$stage/build/postprocess_cuda_tests" "${baseline_args[@]}"
fi

variants=(iou-greater-or-equal suppressed-can-suppress class-logit-argmax reverse-candidate-id-ties
          wrong-p3-stride swap-left-top swap-xy-mapping kept-only-pool ignore-producer-flags no-row-reset)
printf 'variant\tresult\texit\texpected_diagnostic\n' > "$cap/results.tsv"
for variant in "${variants[@]}"; do
  guard_source
  cp -a -- "$cap/baseline-postprocess.cu" "$source_tree/cuda/postprocess.cu"
  mutate
  diff_status=0
  diff -u --label baseline/cuda/postprocess.cu --label "$variant/cuda/postprocess.cu" \
    "$cap/baseline-postprocess.cu" "$source_tree/cuda/postprocess.cu" > "$cap/$variant.diff" || diff_status=$?
  [[ $diff_status == 1 ]] || { echo "mutation has no valid source diff: $variant" >&2; exit 2; }
  sha256sum "$source_tree/cuda/postprocess.cu" > "$cap/$variant.source.sha256"
  cp -a -- "$source_tree/cuda/postprocess.cu" "$cap/$variant.cu"
  printf '%s\n' "$diagnostic" > "$cap/$variant.expected-diagnostic"
  if [[ $check_only == true ]]; then
    printf '%s\tMATCHES_ONLY_GPU_UNEXECUTED\t-\t%s\n' "$variant" "$diagnostic" >> "$cap/results.tsv"
    continue
  fi
  # A compile failure is never evidence that a behavioral mutation was caught.
  run "$variant-build" cmake --build "$stage/build" --target postprocess_cuda_tests --parallel 2
  sha256sum "$stage/build/postprocess_cuda_tests" "$stage/build/libsaccade_postprocess_cuda.a" > "$cap/$variant.binary.sha256"
  status=0
  run "$variant-run" timeout "$timeout_seconds" "$stage/build/postprocess_cuda_tests" --no-held-stream || status=$?
  # Only the program's ordinary test-failure exit plus its exact intended
  # diagnostic counts. Crashes, timeouts and unrelated failures are failures.
  if [[ $status != 1 ]] || ! rg --fixed-strings --line-regexp --quiet -- "$diagnostic" "$cap/$variant-run.stderr"; then
    printf '%s\tNOT_DETECTED_AS_EXPECTED\t%s\t%s\n' "$variant" "$status" "$diagnostic" >> "$cap/results.tsv"
    echo "mutation did not produce its expected test failure: $variant (exit=$status)" >&2
    exit 1
  fi
  printf '%s\tDETECTED\t%s\t%s\n' "$variant" "$status" "$diagnostic" >> "$cap/results.tsv"
  printf 'mutation=%s DETECTED\n' "$variant"
done

cp -a -- "$cap/baseline-postprocess.cu" "$source_tree/cuda/postprocess.cu"
touch "$source_tree/cuda/postprocess.cu"
guard_source
if [[ $check_only == false ]]; then
  run restore-build cmake --build "$stage/build" --target postprocess_cuda_tests --parallel 2
  run restore-baseline timeout "$timeout_seconds" "$stage/build/postprocess_cuda_tests" "${baseline_args[@]}"
  sha256sum "$stage/build/postprocess_cuda_tests" > "$cap/restored-binary.sha256"
fi
find "$cap" -type f ! -name SHA256SUMS ! -name status.txt -print0 | sort -z | \
  while IFS= read -r -d '' path; do sha256sum -- "$path"; done > "$cap/SHA256SUMS"
finished=true
if [[ $check_only == true ]]; then
  printf 'mutation_matches=%s GPU_UNEXECUTED capture=%s\n' "${#variants[@]}" "$cap"
else
  printf 'mutations_detected=%s capture=%s\n' "${#variants[@]}" "$cap"
fi
