#!/usr/bin/env bash
# Resume the completed coordinate rollout using source HDF5 for Newton channels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs

REGISTRY="${REPO_ROOT}/data/registries/main_b045_v1.json"
SPLIT_MANIFEST="${REPO_ROOT}/data/manifests/pvi_subject_splits_mask05_v1.json"
TASK_MANIFEST="${REPO_ROOT}/data/manifests/coordinate_main_b045_bp_364_v1.tsv"
COORDINATE_ROOT="${REPO_ROOT}/gcnm_parquet/coordinate_direct_main_b045_v1"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/coordinate_main_b045_bp_v1"
PVI_ML_ROOT="$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml"
OBSOLETE_REFERENCE_ROOT="${REPO_ROOT}/gcnm_parquet/reference_newton_image_main_b045_v1"

for path in "${REGISTRY}" "${SPLIT_MANIFEST}" "${TASK_MANIFEST}" \
            "${COORDINATE_ROOT}" "${PVI_ML_ROOT}"; do
  [[ -e "${path}" ]] || { echo "required path missing: ${path}" >&2; exit 2; }
done
[[ ! -e "${OBSOLETE_REFERENCE_ROOT}" ]] || {
  echo "obsolete Newton Parquet root still exists: ${OBSOLETE_REFERENCE_ROOT}" >&2
  exit 2
}
[[ ! -e "${ARTIFACT_ROOT}" ]] || {
  echo "immutable BP artifact root already exists: ${ARTIFACT_ROOT}" >&2
  exit 2
}
[[ "$(find "${COORDINATE_ROOT}/ring_parts" -mindepth 2 -maxdepth 2 -name manifest.json | wc -l)" -eq 15 ]] || {
  echo "coordinate rollout does not contain 15 completed ring manifests" >&2
  exit 2
}
[[ "$(($(wc -l < "${TASK_MANIFEST}") - 1))" -eq 364 ]] || {
  echo "BP task manifest must contain exactly 364 runs" >&2
  exit 2
}
head -1 "${TASK_MANIFEST}" | grep -q $'\treference_registry\t' || {
  echo "BP manifest does not use the HDF5 registry contract" >&2
  exit 2
}

finalize_job="$(sbatch --parsable \
  --export="ALL,REPO_ROOT=${REPO_ROOT},COORDINATE_ROOT=${COORDINATE_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST}" \
  slurm/launch_finalize_coordinate_main_b045_parquet.sh)"
bp_job="$(sbatch --parsable --dependency="afterok:${finalize_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},PVI_ML_ROOT=${PVI_ML_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_train_coordinate_main_b045_bp_packed.sh)"
gif_job="$(sbatch --parsable --array=0-363%8 --dependency="afterok:${bp_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_coordinate_main_b045_bp_gifs.sh)"

echo "Parquet/HDF5 validation: ${finalize_job}"
echo "364 packed BP runs: ${bp_job}"
echo "364 prediction-aligned GIF tasks: ${gif_job}"
