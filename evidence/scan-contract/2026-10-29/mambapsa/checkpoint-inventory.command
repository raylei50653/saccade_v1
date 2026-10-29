jq -r .tree[] | select(.path | endswith(".pt")) | [.path,.sha,.size] | @tsv evidence/scan-contract/2026-10-29/mambapsa/tree.json
