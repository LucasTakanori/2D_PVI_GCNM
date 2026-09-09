#!/bin/bash
#SBATCH --job-name=gcnm6-strat-cache
#SBATCH --output=logs/gcnm6-strat-cache_%j.out
#SBATCH --error=logs/gcnm6-strat-cache_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=220GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/pw_main_within_seed42_v1}"
CACHE_ROOT="${CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/img2wave_main_within_seed42_stratified_v2}"

test -f "${SOURCE_CACHE_ROOT}/_SUCCESS"
test -f "${SOURCE_CACHE_ROOT}/manifest.json"

if [[ -f "${CACHE_ROOT}/_SUCCESS" && -f "${CACHE_ROOT}/manifest.json" ]]; then
  echo "validated pre-stratified cache already exists: ${CACHE_ROOT}"
  exit 0
fi

resume=()
if [[ -f "${CACHE_ROOT}/_INCOMPLETE" ]]; then
  resume+=(--resume)
fi

python -u scripts/repack_population_6ch_stratified.py \
  --source-root "${SOURCE_CACHE_ROOT}" \
  --output-root "${CACHE_ROOT}" \
  --batch-size 32 \
  --cluster-size 30 \
  --row-groups-per-file 64 \
  --workers 4 \
  --seed 42 \
  "${resume[@]}"
