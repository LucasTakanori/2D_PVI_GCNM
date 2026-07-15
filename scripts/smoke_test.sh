#!/usr/bin/env bash
# End-to-end smoke test (tiny in-memory mesh, no HDF5 required).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

echo "=== GCNM-PVI smoke test ==="
echo "GCNM_ROOT=${GCNM_ROOT}"
echo "PVI_SOLVER_ROOT=${PVI_SOLVER_ROOT}"
echo "GCNM_PYTHON=${GCNM_PYTHON}"

"${GCNM_PYTHON}" -m gcnm_pvi.smoke_test "$@"
