#!/usr/bin/env bash
# Submit the CRS mirror of the completed 64-channel coordinate GCNM BP matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs data/manifests

SOURCE_TSV="${REPO_ROOT}/data/manifests/coordinate_main_b045_bp_364_v1.tsv"
SOURCE_JSON="${REPO_ROOT}/data/manifests/coordinate_main_b045_bp_364_v1.json"
TASK_MANIFEST="${REPO_ROOT}/data/manifests/coordinate_main_b045_crs_bp_364_v1.tsv"
TASK_JSON="${REPO_ROOT}/data/manifests/coordinate_main_b045_crs_bp_364_v1.json"
SPLIT_MANIFEST="${REPO_ROOT}/data/manifests/pvi_subject_splits_mask05_v1.json"
COORDINATE_ROOT="${REPO_ROOT}/gcnm_parquet/coordinate_direct_main_b045_v1"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/coordinate_main_b045_crs_bp_v1"
PVI_ML_ROOT="$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml"

for path in "${SOURCE_TSV}" "${SOURCE_JSON}" "${SPLIT_MANIFEST}" \
            "${COORDINATE_ROOT}/manifest.json" "${PVI_ML_ROOT}"; do
  [[ -e "${path}" ]] || { echo "required path missing: ${path}" >&2; exit 2; }
done
for path in "${TASK_MANIFEST}" "${TASK_JSON}" "${ARTIFACT_ROOT}"; do
  [[ ! -e "${path}" ]] || { echo "immutable CRS rollout path already exists: ${path}" >&2; exit 2; }
done

"${GCNM_PYTHON}" scripts/build_coordinate_main_b045_crs_manifest.py \
  --source-tsv "${SOURCE_TSV}" --source-json "${SOURCE_JSON}" \
  --output-tsv "${TASK_MANIFEST}" --output-json "${TASK_JSON}"
[[ "$(($(wc -l < "${TASK_MANIFEST}") - 1))" -eq 364 ]] || {
  echo "CRS task manifest does not contain exactly 364 runs" >&2; exit 2;
}

exports="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},PVI_ML_ROOT=${PVI_ML_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}"
bp_job="$(sbatch --parsable --export="${exports}" \
  slurm/launch_train_coordinate_main_b045_crs_bp_packed.sh)"
gif_job="$(sbatch --parsable --array=0-363%16 --dependency="afterok:${bp_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_coordinate_main_b045_bp_gifs_parallel.sh)"

echo "364 CRS BP experiments: ${bp_job}"
echo "prediction-aligned CRS GIFs: ${gif_job}"
