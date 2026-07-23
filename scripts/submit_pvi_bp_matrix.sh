#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs data/manifests
REGISTRY="${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}"
PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/pvi_bp_gcnm_v1}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?set SPLIT_MANIFEST to the frozen sample-ID manifest}"
COORDINATE_PARQUET_ROOT="${COORDINATE_PARQUET_ROOT:?set COORDINATE_PARQUET_ROOT}"
GLOBAL_VOLTAGE_SLOTS_PARQUET_ROOT="${GLOBAL_VOLTAGE_SLOTS_PARQUET_ROOT:?set GLOBAL_VOLTAGE_SLOTS_PARQUET_ROOT}"
TASK_MANIFEST="${TASK_MANIFEST:-${REPO_ROOT}/data/manifests/pvi_bp_mask05_matrix_v1.tsv}"
TASK_MANIFEST_JSON="${TASK_MANIFEST_JSON:-${REPO_ROOT}/data/manifests/pvi_bp_mask05_matrix_v1.json}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
if ! [[ "${MAX_CONCURRENT}" =~ ^[1-4]$ ]]; then
  echo "MAX_CONCURRENT must be an integer from 1 through 4" >&2
  exit 2
fi
for required_path in "${REGISTRY}" "${PVI_ML_ROOT}" "${SPLIT_MANIFEST}" \
                     "${COORDINATE_PARQUET_ROOT}" "${GLOBAL_VOLTAGE_SLOTS_PARQUET_ROOT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

"${GCNM_PYTHON:-python}" -m gcnm_pvi.preflight_pvi_bp \
  --registry "${REGISTRY}" \
  --coordinate-root "${COORDINATE_PARQUET_ROOT}" \
  --global-voltage-slots-root "${GLOBAL_VOLTAGE_SLOTS_PARQUET_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}"

"${GCNM_PYTHON:-python}" -m gcnm_pvi.bp_run_matrix \
  --registry "${REGISTRY}" \
  --coordinate-root "${COORDINATE_PARQUET_ROOT}" \
  --global-voltage-slots-root "${GLOBAL_VOLTAGE_SLOTS_PARQUET_ROOT}" \
  --tsv "${TASK_MANIFEST}" --json "${TASK_MANIFEST_JSON}"

TASK_COUNT="$(( $(wc -l < "${TASK_MANIFEST}") - 1 ))"
if [[ "${TASK_COUNT}" -ne 728 ]]; then
  echo "refusing to submit ${TASK_COUNT} tasks; mask05 matrix must contain 728" >&2
  exit 2
fi
ARRAY_SPEC="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry run: mask05 BP matrix contains ${TASK_COUNT} independent experiments (${ARRAY_SPEC})"
  echo "task manifest: ${TASK_MANIFEST}"
  echo "JSON manifest: ${TASK_MANIFEST_JSON}"
  echo "No SLURM jobs submitted."
  exit 0
fi
job="$(sbatch --parsable --array="${ARRAY_SPEC}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},PVI_ML_ROOT=${PVI_ML_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_train_pvi_bp.sh)"
echo "mask05 BP matrix: ${job} (${ARRAY_SPEC})"
echo "At four concurrent tasks: 4 GPUs, 64 CPUs, and 256 GB requested."
