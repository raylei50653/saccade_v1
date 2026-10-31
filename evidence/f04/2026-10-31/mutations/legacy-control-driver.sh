#!/usr/bin/env bash
# Ignored orchestration only: preserve frozen production scripts/source.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: run.sh BASELINE_BUILD build-NEW-MUTATIONS' >&2; exit 2; }
repo=$(git rev-parse --show-toplevel)
cd "$repo"
baseline=$(cd "$1" && pwd)
new_stage=$2
bash scripts/check_postprocess_mutations.sh "$baseline" "$new_stage"
stage=$(cd "$new_stage" && pwd)
cap=$stage/capture
source_tree=$stage/source
build=$stage/build
baseline_test=$cap/baseline-postprocess_cuda_tests.cpp
cp -a "$source_tree/tests/postprocess_cuda_tests.cpp" "$baseline_test"
cp -a "$0" "$cap/legacy-control-driver.sh"
dumps=$(awk 'index($0,"SACCADE_POSTPROCESS_F03_DUMPS:")==1 {sub(/^[^=]*=/, ""); print; exit}' "$baseline/CMakeCache.txt")
[[ -n $dumps ]] || { echo 'missing pinned replay directory in baseline cache' >&2; exit 2; }
guard_source() { sha256sum --check --status "$cap/original-source.sha256"; }
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
needle='for (unsigned kind = 0; kind < 2; ++kind) for (unsigned l = 0; l < 3; ++l) {'
replacement='for (unsigned l = 0; l < 3; ++l) for (unsigned kind = 0; kind < 2; ++kind) {'
awk -v needle="$needle" -v replacement="$replacement" '
    { line=$0; result=""; while((at=index(line,needle))!=0) {
        result=result substr(line,1,at-1) replacement
        line=substr(line,at+length(needle)); count++
      }; print result line }
    END { if(count!=1) {printf "legacy parser expected 1 literal match, got %d\n",count > "/dev/stderr"; exit 2}}
' "$baseline_test" > "$source_tree/tests/postprocess_cuda_tests.cpp.next"
mv "$source_tree/tests/postprocess_cuda_tests.cpp.next" "$source_tree/tests/postprocess_cuda_tests.cpp"
cp -a "$source_tree/tests/postprocess_cuda_tests.cpp" "$cap/legacy-parser-postprocess_cuda_tests.cpp"
status=0
diff -u --label baseline/tests/postprocess_cuda_tests.cpp --label legacy-parser/tests/postprocess_cuda_tests.cpp \
    "$baseline_test" "$source_tree/tests/postprocess_cuda_tests.cpp" > "$cap/legacy-parser.diff" || status=$?
[[ $status == 1 ]] || { echo 'legacy parser mutation did not produce a valid diff' >&2; exit 2; }
sha256sum "$source_tree/tests/postprocess_cuda_tests.cpp" "$source_tree/cuda/postprocess.cu" > "$cap/legacy-parser.source.sha256"
printf '%s\n' 'F03 replay disagrees with independent historical kept count: input 0' > "$cap/legacy-parser.expected-diagnostic"
run legacy-parser-build cmake --build "$build" --target postprocess_cuda_tests --parallel 2
sha256sum "$build/postprocess_cuda_tests" "$build/libsaccade_postprocess_cuda.a" > "$cap/legacy-parser.binary.sha256"
status=0
run legacy-parser-run timeout 300 "$build/postprocess_cuda_tests" --no-held-stream --f03-dumps "$dumps" || status=$?
if [[ $status != 1 ]] || ! rg --fixed-strings --line-regexp --quiet -- \
    'F03 replay disagrees with independent historical kept count: input 0' "$cap/legacy-parser-run.stderr"; then
    printf 'legacy-parser\tNOT_DETECTED_AS_EXPECTED\t%s\n' "$status" > "$cap/legacy-parser.result.tsv"
    echo "legacy parser control did not produce expected diagnostic (exit=$status)" >&2
    exit 1
fi
printf 'legacy-parser\tDETECTED\t1\tF03 replay disagrees with independent historical kept count: input 0\n' > "$cap/legacy-parser.result.tsv"
cp -a "$baseline_test" "$source_tree/tests/postprocess_cuda_tests.cpp"
touch "$source_tree/tests/postprocess_cuda_tests.cpp"
run legacy-parser-restore-build cmake --build "$build" --target postprocess_cuda_tests --parallel 2
run legacy-parser-restored-baseline timeout 300 "$build/postprocess_cuda_tests" --no-held-stream --f03-dumps "$dumps"
sha256sum "$build/postprocess_cuda_tests" "$build/libsaccade_postprocess_cuda.a" > "$cap/legacy-parser-restored-binary.sha256"
printf 'exit=0\nlegacy_parser_detected=true\nrestored_baseline=PASS\nfinished=%s\n' "$(date -Is)" > "$cap/legacy-parser-status.txt"
find "$cap" -type f ! -name SHA256SUMS -print0 | sort -z | \
    while IFS= read -r -d '' path; do sha256sum "$path"; done > "$cap/SHA256SUMS"
guard_source
printf 'mutations=10/10 legacy_parser=DETECTED restored_baseline=PASS capture=%s\n' "$cap"
