#!/usr/bin/env bash
# Install GCNM-PVI Python dependencies into GCNM_PYTHON venv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCNM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GCNM_PYTHON="${GCNM_PYTHON:-${GCNM_ROOT}/.venv/bin/python}"

if [[ ! -x "${GCNM_PYTHON}" ]]; then
  echo "ERROR: set GCNM_PYTHON to an existing python3 binary" >&2
  exit 1
fi

"${GCNM_PYTHON}" -m ensurepip --upgrade 2>/dev/null || true
"${GCNM_PYTHON}" -m pip install --upgrade pip
"${GCNM_PYTHON}" -m pip install -r "${GCNM_ROOT}/requirements.txt"

echo "Done. Verify with: bash scripts/smoke_test.sh"
