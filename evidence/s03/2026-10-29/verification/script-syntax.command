bash -c $'\nset -euo pipefail\nfor file in evidence/s03/2026-10-29/*.sh evidence/s03/2026-10-29/verification/*.sh; do\n  bash -n "$file"\n  echo "$file: ok"\ndone\n' 
