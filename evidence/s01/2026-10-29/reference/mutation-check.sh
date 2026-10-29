#!/usr/bin/env bash
# Fixture sensitivity check: each mutation changes one contract clause in a
# private copy of src/selective_scan.cpp; the unmodified tests must then fail.
# Usage from the repository root: bash mutation-check.sh SOURCE_ROOT OUTPUT_DIR
set -euo pipefail
src=${1:?source root}; out=${2:?output dir}
mkdir -p "$out"
mutations=(
  'softplus_threshold_inclusive|s/return x > 20.0 ? x/return x >= 20.0 ? x/'
  'drop_d_skip|s/if (s.has_d) acc +=/if (false) acc +=/'
  'state_not_reset_per_channel|s/std::fill(h.begin(), h.end(), 0.0);//'
  'per_channel_a_reads_row_0|s/(s.a_rows == 1 ? 0 : d \* N)/0/'
  'full_c_reads_state_0|s/(s.c_width == 1 ? 0 : n)/0/'
  'decay_uses_previous_delta_order|s/h\[n\] = in_range(decay \* h\[n\] + drive/h[n] = in_range(h[n] + decay * drive/'
  'direction_2_flips_h|s/flip_w = direction == 1 || direction == 2/flip_w = direction == 1 || direction == 3/'
  'merge_sums_without_mean|s/out\[(i \* channels + c) \* L + p\] = sum \/ 4.0;/out[(i * channels + c) * L + p] = sum;/'
)
status=0
for entry in "${mutations[@]}"; do
  name=${entry%%|*}; expr=${entry#*|}
  sed "$expr" "$src/src/selective_scan.cpp" > "$out/$name.cpp"
  if cmp -s "$out/$name.cpp" "$src/src/selective_scan.cpp"; then echo "$name: mutation did not apply"; status=1; continue; fi
  c++ -std=c++20 -O2 -ffp-contract=off -I "$src/include" "$out/$name.cpp" "$src/tests/selective_scan_tests.cpp" \
      "$src/src/file_io.cpp" -o "$out/$name.bin"
  if "$out/$name.bin" "$src/fixtures/selective-scan" > "$out/$name.log" 2>&1; then
    echo "$name: tests PASSED on mutated reference (insensitive)"; status=1
  else
    echo "$name: detected ($(tail -1 "$out/$name.log"))"
  fi
done
exit "$status"
