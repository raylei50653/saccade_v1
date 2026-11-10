#!/usr/bin/env bash
# Install the CPU-only evaluation environment independently of training/runtime.
set -euo pipefail
[[ $# == 0 ]] || { echo 'usage: setup_eval.sh' >&2; exit 2; }
# shellcheck source-path=SCRIPTDIR
# shellcheck source=eval_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"
command -v uv >/dev/null || { echo 'setup_eval.sh requires uv' >&2; exit 2; }
venv=$saccade_eval_root/venv
if [[ ! -x $venv/bin/python ]]; then
  uv venv --python "${SACCADE_EVAL_BOOTSTRAP_PYTHON:-3.12}" "$venv"
fi
uv pip install --python "$venv/bin/python" -r "$saccade_eval_repo/scripts/requirements-eval.lock.txt"
"$venv/bin/python" -I -c 'import numpy, scipy; print("NumPy", numpy.__version__, "SciPy", scipy.__version__)'
printf 'evaluation Python: %s\n' "$venv/bin/python"
