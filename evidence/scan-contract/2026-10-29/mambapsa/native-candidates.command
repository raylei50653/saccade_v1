sh -c jq -r '.tree[].path' "$1/tree.json" | rg -i "onnx|safetensors|\\.(bin|engine|plan|cpp|cu|h|hpp)$|license|export" || test "$?" -eq 1 sh evidence/scan-contract/2026-10-29/mambapsa
