#!/usr/bin/env bash
# Prove CPU artifact export, then resume/skip the immutable 364-run matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"

TASK_MANIFEST="${REPO_ROOT}/data/manifests/coordinate_main_b045_bp_364_v1.tsv"
SPLIT_MANIFEST="${REPO_ROOT}/data/manifests/pvi_subject_splits_mask05_v1.json"
COORDINATE_ROOT="${REPO_ROOT}/gcnm_parquet/coordinate_direct_main_b045_v1"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/coordinate_main_b045_bp_v1"
PVI_ML_ROOT="$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml"

for path in "${TASK_MANIFEST}" "${SPLIT_MANIFEST}" \
            "${COORDINATE_ROOT}/manifest.json" "${ARTIFACT_ROOT}" "${PVI_ML_ROOT}"; do
  [[ -e "${path}" ]] || { echo "resume prerequisite missing: ${path}" >&2; exit 2; }
done
[[ "$(($(wc -l < "${TASK_MANIFEST}") - 1))" -eq 364 ]] || {
  echo "BP task manifest must contain 364 runs" >&2; exit 2;
}

exports="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},PVI_ML_ROOT=${PVI_ML_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}"
smoke_job="$(sbatch --parsable --export="${exports}" \
  slurm/launch_smoke_coordinate_main_b045_bp_postprocess.sh)"
bp_job="$(sbatch --parsable --dependency="afterok:${smoke_job}" \
  --export="${exports}" slurm/launch_train_coordinate_main_b045_bp_packed.sh)"
gif_job="$(sbatch --parsable --array=0-363%8 --dependency="afterok:${bp_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_coordinate_main_b045_bp_gifs.sh)"

echo "CPU artifact-export smoke: ${smoke_job}"
echo "resumable packed BP matrix: ${bp_job}"
echo "prediction-aligned GIFs: ${gif_job}"
