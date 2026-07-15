#!/usr/bin/env bash
# Export one US### ring mesh collection entry to Python PVI HDF5 files.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

RING_ID="${1:-${GCNM_RING_ID:-}}"
OUT_DIR="${2:-${GCNM_RING_OUT:-${GCNM_ROOT}/data/ring_meshes/${RING_ID}}}"
COLLECTION="${GCNM_RING_COLLECTION:-${PVI_SOLVER_ROOT}/../../data/mesh_collection_b045.mat}"
IMG_SIZE="${GCNM_IMG_SIZE:-40}"

if [[ -z "${RING_ID}" ]]; then
  echo "Usage: bash scripts/export_ring_mesh.sh US### [OUT_DIR]" >&2
  exit 1
fi

"${GCNM_PYTHON}" -m gcnm_pvi.ring_mesh_collection \
  --collection "${COLLECTION}" \
  --ring-id "${RING_ID}" \
  --out-dir "${OUT_DIR}" \
  --img-size "${IMG_SIZE}"
