#!/usr/bin/env bash
# Fixed acceptance for docs/TRACKING.md §22.5. A fresh build/NAME is required.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[[ $# == 1 && $1 =~ ^build/[A-Za-z0-9][A-Za-z0-9._-]*$ && ! -e $1 && ! -L $1 ]] || {
  echo 'usage: run_nsa_v2_acceptance.sh build/NEW-OUTPUT' >&2; exit 2;
}
[[ -z $(git status --porcelain) ]] || { echo 'clean committed source required' >&2; exit 2; }
# Hold the shared lock over the entire sequential run, including all GPU tools.
exec flock /home/ray/.cache/saccade-v1/gpu.lock timeout --kill-after=30 28800 \
  python3 -I scripts/nsa_v2_acceptance.py "$1"
