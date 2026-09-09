#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

SOURCE_CACHE_ROOT="/mmfs1/scratch/${USER}/gcnm_population_6ch/pw_main_within_seed42_v1"
CACHE_ROOT="/mmfs1/scratch/${USER}/gcnm_population_6ch/img2wave_main_within_seed42_stratified_v2"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/population_within_gcnm6ch_img2wave_stratified_v2"
PVI_ML_ROOT="$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml"

for target in \
  "${ARTIFACT_ROOT}/crt-gcnm6ch-img-to-waveform-stratified" \
  "${ARTIFACT_ROOT}/samba-gcnm6ch-img-to-waveform-stratified"; do
  if [[ -e "${target}" ]]; then
    echo "refusing to overwrite existing population experiment: ${target}" >&2
    exit 2
  fi
done

cache_job="$(sbatch --parsable \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SOURCE_CACHE_ROOT=${SOURCE_CACHE_ROOT},CACHE_ROOT=${CACHE_ROOT}" \
  slurm/launch_repack_population_6ch_stratified.sh)"
train_job="$(sbatch --parsable --array=0-1%2 --dependency="afterok:${cache_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CACHE_ROOT=${CACHE_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},PVI_ML_ROOT=${PVI_ML_ROOT}" \
  slurm/launch_train_population_6ch_img_waveform.sh)"

echo "pre-stratified six-channel cache: ${cache_job}"
echo "CRT and Samba image-to-waveform training (two GPUs concurrently): ${train_job}"
