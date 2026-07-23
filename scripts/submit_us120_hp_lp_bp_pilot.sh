#!/usr/bin/env bash
# Preflight and submit the eight CRT-only GCNM BP pilot runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs data/manifests

REGISTRY="${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json}"
PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
PARQUET_BASE="${PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet}"
COORDINATE_ROOT="${COORDINATE_ROOT:-${PARQUET_BASE}/us120_pilot_coordinate_hp_lp_v1}"
VESSEL_ROOT="${VESSEL_ROOT:-${PARQUET_BASE}/us120_pilot_global_voltage_slots_hp_lp_v1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/pvi_bp_us120_hp_lp_pilot_v1}"
TASK_MANIFEST="${TASK_MANIFEST:-${REPO_ROOT}/data/manifests/us120_hp_lp_bp_pilot_3ch_v1.tsv}"
TASK_MANIFEST_JSON="${TASK_MANIFEST_JSON:-${REPO_ROOT}/data/manifests/us120_hp_lp_bp_pilot_3ch_v1.json}"

for required in "${REGISTRY}" "${SPLIT_MANIFEST}" "${PVI_ML_ROOT}" \
                "${COORDINATE_ROOT}" "${VESSEL_ROOT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required pilot input is missing: ${required}" >&2
    exit 2
  fi
done

"${GCNM_PYTHON}" -m gcnm_pvi.preflight_pvi_bp \
  --registry "${REGISTRY}" --split-manifest "${SPLIT_MANIFEST}" \
  --coordinate-root "${COORDINATE_ROOT}" \
  --global-voltage-slots-root "${VESSEL_ROOT}" \
  --subject subject006 --subject subject010 --expected-subjects 2

"${GCNM_PYTHON}" -m gcnm_pvi.bp_run_matrix \
  --mode gcnm --registry "${REGISTRY}" \
  --coordinate-root "${COORDINATE_ROOT}" \
  --global-voltage-slots-root "${VESSEL_ROOT}" \
  --family coordinate --family global_voltage_slots \
  --architecture crt \
  --subject subject006 --subject subject010 --channel-mode 3ch \
  --expected-subjects 2 --expected-runs 8 \
  --tsv "${TASK_MANIFEST}" --json "${TASK_MANIFEST_JSON}"

TASK_COUNT="$(( $(wc -l < "${TASK_MANIFEST}") - 1 ))"
if [[ "${TASK_COUNT}" -ne 8 ]]; then
  echo "refusing to submit ${TASK_COUNT} tasks; expected 8 pilot experiments" >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "dry run: ${TASK_COUNT} pilot experiments, array 0-7%4"
  exit 0
fi

job="$(sbatch --parsable --array=0-7%4 \
  --export="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},PVI_ML_ROOT=${PVI_ML_ROOT},REGISTRY=${REGISTRY},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_train_us120_bp_pilot.sh)"
job_id="${job%%;*}"
summary_job="$(sbatch --parsable --dependency="afterok:${job_id}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_summarize_us120_bp_pilot.sh)"
echo "US120 3-channel CRT BP pilot: ${job} (8 tasks, at most 4 concurrent)"
echo "paired BP summary: ${summary_job}"
echo "The array ceiling is 4 GPUs, 64 CPUs, and 1000 GB."
