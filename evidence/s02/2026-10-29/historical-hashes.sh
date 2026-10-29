#!/usr/bin/env bash
# Verify every earlier evidence bundle without editing it: the S01 round's
# checker (P01-A, both CUDA synthetic bundles, scan-contract) plus the S01
# bundle and the scan-contract MambaPSA sub-bundle.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
bash evidence/s01/2026-10-29/reference/historical-hashes.sh
printf '== %s\n' evidence/s01/2026-10-29/SHA256SUMS
sha256sum --quiet -c evidence/s01/2026-10-29/SHA256SUMS && echo "all entries OK"
printf '== %s\n' evidence/scan-contract/2026-10-29/mambapsa/SHA256SUMS
(cd evidence/scan-contract/2026-10-29/mambapsa && sha256sum --quiet -c SHA256SUMS) && echo "all entries OK"
