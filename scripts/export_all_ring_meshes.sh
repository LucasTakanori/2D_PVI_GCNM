#!/usr/bin/env bash
# Export every ring size from both authoritative PVI MATLAB mesh collections.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

PVI_DATA_DIR="${PVI_MESH_DATA_DIR:-${PVI_SOLVER_ROOT}/../../data}"
OUT_ROOT="${GCNM_ALL_RING_OUT:-${GCNM_ROOT}/data/ring_meshes/collections}"
IMG_SIZE="${GCNM_IMG_SIZE:-40}"

for variant in b035 b045; do
  collection="${PVI_DATA_DIR}/mesh_collection_${variant}.mat"
  if [[ ! -f "${collection}" ]]; then
    echo "ERROR: missing collection: ${collection}" >&2
    exit 1
  fi
  mapfile -t ring_ids < <(
    "${GCNM_PYTHON}" -m gcnm_pvi.ring_mesh_collection \
      --collection "${collection}" \
      --ring-id US060 \
      --out-dir /tmp/unused-ring-export \
      --list
  )
  for ring_id in "${ring_ids[@]}"; do
    out_dir="${OUT_ROOT}/${variant}/${ring_id}"
    echo "Exporting ${variant}/${ring_id} -> ${out_dir}"
    "${GCNM_PYTHON}" -m gcnm_pvi.ring_mesh_collection \
      --collection "${collection}" \
      --ring-id "${ring_id}" \
      --out-dir "${out_dir}" \
      --img-size "${IMG_SIZE}"
  done
done

echo "Exported all ring collections to ${OUT_ROOT}"
