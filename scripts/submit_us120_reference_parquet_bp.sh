#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs

PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json}"
TASK_MANIFEST="${TASK_MANIFEST:-${REPO_ROOT}/data/manifests/us120_reference_parquet_bp_v1.tsv}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/pvi_bp_us120_reference_parquet_v1}"

for required in "${PVI_ML_ROOT}" "${SPLIT_MANIFEST}" "${TASK_MANIFEST}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 2; }
done
[[ ! -e gcnm_parquet/us120_pilot_reference_image_v1 ]] || { echo "immutable image Parquet root already exists" >&2; exit 2; }
[[ ! -e gcnm_parquet/us120_pilot_reference_bioz_v1 ]] || { echo "immutable BioZ Parquet root already exists" >&2; exit 2; }

if [[ "$(($(wc -l < "${TASK_MANIFEST}") - 1))" -ne 8 ]]; then
  echo "reference task manifest must contain exactly eight runs" >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "dry run: 2 CPU exports followed by 8 CRT GPU runs"
  exit 0
fi

export_job="$(sbatch --parsable --array=0-1%2 \
  --export="ALL,REPO_ROOT=${REPO_ROOT}" \
  slurm/launch_export_us120_reference_parquet.sh)"
export_id="${export_job%%;*}"
train_job="$(sbatch --parsable --array=0-7%4 --dependency="afterok:${export_id}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},PVI_ML_ROOT=${PVI_ML_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_train_us120_reference_parquet_bp.sh)"

echo "reference Parquet export: ${export_job} (image + BioZ, CPU only)"
echo "reference CRT BP matrix: ${train_job} (8 runs, after successful export)"
echo "artifact root: ${ARTIFACT_ROOT}"
