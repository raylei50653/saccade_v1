jq -r .tree[] | [.type,.path,.sha,(.size // "")] | @tsv evidence/scan-contract/2026-10-29/mambapsa/tree.json
