#!/usr/bin/env bash
set -euo pipefail
task_dir=${1:-build-scan-design-replay}
[[ ! -e $task_dir ]] || { echo 'use a fresh output directory' >&2; exit 2; }
mkdir -p "$task_dir"
task_dir=$(CDPATH= cd -- "$task_dir" && pwd)
task_legacy=/home/ray/developer/ai/saccade
task_rev=e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2
capture() {
    local name=$1
    shift
    printf '%q ' "$@" > "$task_dir/$name.command"
    printf '\n' >> "$task_dir/$name.command"
    local code=0
    "$@" > "$task_dir/$name.log" 2>&1 || code=$?
    printf '%s\n' "$code" > "$task_dir/$name.exit"
    return "$code"
}
capture environment bash -c 'date -Is; uname -srmo; git --version; sed --version | head -1; sha256sum --version | head -1'
capture identity git -C "$task_legacy" rev-parse "$task_rev" "$task_rev:docs/reference/math_model.md" "$task_rev:docs/reference/yolo_model_architecture.svg" "$task_rev:LICENSE"
capture source-metadata bash -o pipefail -c 'for path in docs/reference/math_model.md docs/reference/yolo_model_architecture.svg LICENSE; do printf "%s\t" "$path"; git -C "$1" cat-file -s "$2:$path"; git -C "$1" show "$2:$path" | sha256sum; done' _ "$task_legacy" "$task_rev"
capture architecture-excerpt bash -o pipefail -c 'git -C "$1" show "$2:docs/reference/math_model.md" | nl -ba | sed -n "369,468p"' _ "$task_legacy" "$task_rev"
capture scan-fields-in-architecture bash -o pipefail -c 'git -C "$1" show "$2:docs/reference/math_model.md" | nl -ba | sed -n "369,468p" | rg -n -i "mamba|state|scan|delta|softplus|flatten|stride|reg_max|cls|gate|initial|final|reset|bias|direction"' _ "$task_legacy" "$task_rev"
capture license-declaration bash -o pipefail -c 'git -C "$1" show "$2:LICENSE" | sed -n "1,12p"' _ "$task_legacy" "$task_rev"
capture svg-structure bash -o pipefail -c 'git -C "$1" show "$2:docs/reference/yolo_model_architecture.svg" | rg -n -i "desc|mamba|scan|state|softplus|P3|P4|P5|cls|reg|gate|delta"' _ "$task_legacy" "$task_rev"
