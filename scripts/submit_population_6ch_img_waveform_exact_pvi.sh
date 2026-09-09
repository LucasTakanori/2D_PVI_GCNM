#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SCRATCH_BASE="/mmfs1/scratch/${USER}/gcnm_population_6ch"
SOURCE_CACHE_ROOT="${SCRATCH_BASE}/pw_main_within_seed42_v1"
CACHE_ROOT="${SCRATCH_BASE}/img2wave_main_within_seed42_exact_pvi_v3"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts"
LOG_ROOT="${REPO_ROOT}/logs/gcnm_population_6ch_exact_pvi"
PVI_ML_ROOT="$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml"
mkdir -p "${LOG_ROOT}"

for target in \
  "${ARTIFACT_ROOT}/crt-gcnm6ch-img-to-waveform-exact-pvi" \
  "${ARTIFACT_ROOT}/samba-gcnm6ch-img-to-waveform-exact-pvi"; do
  if [[ -e "${target}" ]]; then
    checkpoint="${target}/main/checkpoints/dataset_lazy_checkpoints.pth"
    [[ -f "${checkpoint}" ]] || {
      echo "refusing incomplete PVI artifact tree without resumable checkpoint: ${target}" >&2
      exit 2
    }
    echo "will resume PVI artifact tree: ${target}"
  fi
done

dependency=()
if [[ -f "${CACHE_ROOT}/_SUCCESS" && -f "${CACHE_ROOT}/manifest.json" ]]; then
  echo "using completed Parquet cache: ${CACHE_ROOT}"
  cache_job="already complete"
else
  cache_job="$(sbatch --parsable \
    --output="${LOG_ROOT}/gcnm6-exact-cache_%j.out" \
    --error="${LOG_ROOT}/gcnm6-exact-cache_%j.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},SOURCE_CACHE_ROOT=${SOURCE_CACHE_ROOT},CACHE_ROOT=${CACHE_ROOT}" \
    slurm/launch_build_population_6ch_exact_pvi_cache.sh)"
  dependency+=(--dependency="afterok:${cache_job}")
fi
train_job="$(sbatch --parsable --array=0-1%2 "${dependency[@]}" \
  --output="${LOG_ROOT}/gcnm6-exact-train_%A_%a.out" \
  --error="${LOG_ROOT}/gcnm6-exact-train_%A_%a.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CACHE_ROOT=${CACHE_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},PVI_ML_ROOT=${PVI_ML_ROOT}" \
  slurm/launch_train_population_6ch_img_waveform.sh)"

echo "exact PVI schedule/cache build: ${cache_job}"
echo "CRT and Samba image-to-waveform training (two GPUs concurrently): ${train_job}"
echo "Parquet cache: ${CACHE_ROOT}"
echo "PVI artifacts: ${ARTIFACT_ROOT}"
echo "scheduler logs: ${LOG_ROOT}"
