#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SCRATCH_BASE="/mmfs1/scratch/${USER}/gcnm_population_6ch"
BASE_CACHE_ROOT="${SCRATCH_BASE}/img2wave_main_within_seed42_exact_pvi_v3"
COMPLETION_ROOT="${SCRATCH_BASE}/img2wave_legacy_pd_missing_payload_v1"
CRT_CACHE_ROOT="${SCRATCH_BASE}/pd13_img2wave_legacy_disjoint_exact_pvi_view_v4"
SAMBA_CACHE_ROOT="${SCRATCH_BASE}/pd17_img2wave_legacy_disjoint_exact_pvi_view_v4"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts"
LOG_ROOT="${REPO_ROOT}/logs/gcnm_population_6ch_disjoint"
PVI_ML_ROOT="$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml"
mkdir -p "${LOG_ROOT}"

for target in \
  "${ARTIFACT_ROOT}/pd13-crt-gcnm6ch-img-to-waveform-exact-pvi" \
  "${ARTIFACT_ROOT}/pd17-samba-gcnm6ch-img-to-waveform-exact-pvi"; do
  if [[ -e "${target}" ]]; then
    checkpoint="${target}/main/checkpoints/dataset_lazy_checkpoints.pth"
    [[ -f "${checkpoint}" ]] || {
      echo "refusing incomplete PD artifact tree without resumable checkpoint: ${target}" >&2
      exit 2
    }
    echo "will resume PVI artifact tree: ${target}"
  fi
done

test -f "${BASE_CACHE_ROOT}/_SUCCESS"
test -f "${BASE_CACHE_ROOT}/manifest.json"

completion_dependency=()
if [[ -f "${COMPLETION_ROOT}/_SUCCESS" && -f "${COMPLETION_ROOT}/manifest.json" ]]; then
  completion_job="already complete"
else
  completion_job="$(sbatch --parsable \
    --output="${LOG_ROOT}/gcnm6-pd-payload_%j.out" \
    --error="${LOG_ROOT}/gcnm6-pd-payload_%j.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},SCRATCH_BASE=${SCRATCH_BASE},BASE_CACHE_ROOT=${BASE_CACHE_ROOT},COMPLETION_ROOT=${COMPLETION_ROOT}" \
    slurm/launch_build_population_6ch_legacy_completion.sh)"
  completion_dependency+=(--dependency="afterok:${completion_job}")
fi

views_ready=true
for root in "${CRT_CACHE_ROOT}" "${SAMBA_CACHE_ROOT}"; do
  if [[ ! -f "${root}/_SUCCESS" || ! -f "${root}/manifest.json" ]]; then
    views_ready=false
  fi
done

train_dependency=()
if [[ "${views_ready}" == true ]]; then
  view_job="already complete"
else
  view_job="$(sbatch --parsable --array=0-1%2 "${completion_dependency[@]}" \
    --output="${LOG_ROOT}/gcnm6-pd-view_%A_%a.out" \
    --error="${LOG_ROOT}/gcnm6-pd-view_%A_%a.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},SCRATCH_BASE=${SCRATCH_BASE},BASE_CACHE_ROOT=${BASE_CACHE_ROOT},COMPLETION_ROOT=${COMPLETION_ROOT},CRT_CACHE_ROOT=${CRT_CACHE_ROOT},SAMBA_CACHE_ROOT=${SAMBA_CACHE_ROOT}" \
    slurm/launch_build_population_6ch_disjoint_view.sh)"
  train_dependency+=(--dependency="afterok:${view_job}")
fi

train_job="$(sbatch --parsable --array=0-1%2 "${train_dependency[@]}" \
  --job-name="gcnm6-pd-img2wave" \
  --output="${LOG_ROOT}/gcnm6-pd-train_%A_%a.out" \
  --error="${LOG_ROOT}/gcnm6-pd-train_%A_%a.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CRT_CACHE_ROOT=${CRT_CACHE_ROOT},SAMBA_CACHE_ROOT=${SAMBA_CACHE_ROOT},PVI_ML_ROOT=${PVI_ML_ROOT},SPLIT_MODE=disjoint" \
  slurm/launch_train_population_6ch_img_waveform.sh)"

echo "legacy missing-payload build: ${completion_job}"
echo "pd13/pd17 disjoint index-view builds: ${view_job}"
echo "pd13 CRT and pd17 Samba six-channel training: ${train_job}"
echo "PD13 Parquet view: ${CRT_CACHE_ROOT}"
echo "PD17 Parquet view: ${SAMBA_CACHE_ROOT}"
echo "PVI artifacts: ${ARTIFACT_ROOT}"
echo "scheduler logs: ${LOG_ROOT}"
