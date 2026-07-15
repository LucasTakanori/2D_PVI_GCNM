#!/usr/bin/env bash
# Shared environment for all GCNM-PVI scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCNM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# cluster paths (override in your shell if needed)
if [[ -f "${GCNM_ROOT}/env/cluster.env" ]]; then
  # shellcheck source=/dev/null
  source "${GCNM_ROOT}/env/cluster.env"
fi

export GCNM_ROOT
export PVI_SOLVER_ROOT="${PVI_SOLVER_ROOT:-${GCNM_ROOT}/../Peripheral-Vascular-Impedance-Imaging/python_port/pvi_solver}"
export PYTHONPATH="${GCNM_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Python interpreter (override for a shared cluster environment if desired)
export GCNM_PYTHON="${GCNM_PYTHON:-${GCNM_ROOT}/.venv/bin/python}"

if [[ ! -x "${GCNM_PYTHON}" ]]; then
  echo "ERROR: GCNM_PYTHON not found: ${GCNM_PYTHON}" >&2
  echo "Set GCNM_PYTHON to a venv with requirements.txt installed." >&2
  exit 1
fi

cd "${GCNM_ROOT}"
