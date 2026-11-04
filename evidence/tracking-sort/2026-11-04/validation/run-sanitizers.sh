#!/usr/bin/env bash
set -euo pipefail
cd /home/ray/developer/ai/saccade_v1
sanitizer=/opt/cuda/bin/compute-sanitizer
out=build/tracking-sort-evidence
for tool in memcheck initcheck racecheck synccheck; do
    args=(--tool "$tool" --error-exitcode 99)
    [[ $tool != initcheck ]] || args+=(--check-api-memory-access no)
    "$sanitizer" "${args[@]}" build/tracking-sort/tracking_cuda_tests > "$out/$tool.log" 2>&1
    if [[ $tool == racecheck ]]; then
        rg -Fq 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' "$out/$tool.log"
        diagnostic='Error: Race reported between'
    else
        rg -Fq 'ERROR SUMMARY: 0 errors' "$out/$tool.log"
        case $tool in
            memcheck) diagnostic='Invalid __global__ write';;
            initcheck) diagnostic='Uninitialized __global__ memory read';;
            synccheck) diagnostic='Barrier error detected';;
        esac
    fi
    rg -q 'checks passed' "$out/$tool.log"
    status=0
    "$sanitizer" --tool "$tool" --error-exitcode 99 build/tracking-sort/postprocess_sanitizer_control "$tool" \
        > "$out/control-$tool.log" 2>&1 || status=$?
    [[ $status == 99 ]]
    rg -Fq "$diagnostic" "$out/control-$tool.log"
done
printf 'PASS: four sanitizer tools and matching positive controls\n' > "$out/sanitizers-status.txt"
