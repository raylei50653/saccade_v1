#!/bin/sh
# Read-only upstream audit. Downloads source as text; never executes it.
set -eu
out=${1:-build-scan-source-replay/mambapsa}
mkdir -p "$out"
rev=2940f1634cfce1f63201e6411ffe9008951461f3
capture() {
  label=$1
  shift
  printf '%s\n' "$*" > "$out/$label.command"
  set +e
  "$@" > "$out/$label.log" 2>&1
  result=$?
  set -e
  printf '%s\n' "$result" > "$out/$label.exit"
  return "$result"
}
fetch() {
  label=$1
  url=$2
  dest=$3
  capture "$label" curl --fail --location --silent --show-error --max-time 45 "$url" -o "$out/$dest"
}
capture environment sh -c 'date -Is; uname -a; curl --version | head -1; jq --version; git --version'
fetch tree "https://api.github.com/repos/henrychan0719/MambaPSA/git/trees/$rev?recursive=1" tree.json
fetch commit "https://api.github.com/repos/henrychan0719/MambaPSA/commits/$rev" commit.json
fetch current-main 'https://api.github.com/repos/henrychan0719/MambaPSA/commits/main' current-main.json
fetch releases 'https://api.github.com/repos/henrychan0719/MambaPSA/releases' releases.json
fetch repository 'https://api.github.com/repos/henrychan0719/MambaPSA' repository.json
for name in README.md mamba_psa.py bivision_mamba.py dual_mamba.py requirements.txt yolo26n-mambapsa.yaml yolo26n-mamba-p3.yaml yolo26n-mamba-p4.yaml yolo26n-mamba.yaml yolo26n-dualmamba.yaml train.py val.py; do
  fetch "$name" "https://raw.githubusercontent.com/henrychan0719/MambaPSA/$rev/$name" "$name.txt"
done
fetch mambapsa-args "https://raw.githubusercontent.com/henrychan0719/MambaPSA/$rev/runs/detect/runs/voc_mambapsa/args.yaml" mambapsa-args.yaml.txt
fetch paper-html 'https://arxiv.org/html/2607.12681v1' paper.html
capture inventory jq -r '.tree[] | [.type,.path,.sha,(.size // "")] | @tsv' "$out/tree.json"
capture native-candidates sh -c 'jq -r '\''.tree[].path'\'' "$1/tree.json" | rg -i "onnx|safetensors|\\.(bin|engine|plan|cpp|cu|h|hpp)$|license|export" || test "$?" -eq 1' sh "$out"
capture checkpoint-inventory jq -r '.tree[] | select(.path | endswith(".pt")) | [.path,.sha,.size] | @tsv' "$out/tree.json"
(cd "$out"
  find . -maxdepth 1 -type f \( -name '*.json' -o -name '*.txt' -o -name 'paper.html' \) ! -name SIZES.txt -print | LC_ALL=C sort > source-files.list
  xargs sha256sum < source-files.list > SHA256SUMS
  xargs wc -c < source-files.list > SIZES.txt
)
