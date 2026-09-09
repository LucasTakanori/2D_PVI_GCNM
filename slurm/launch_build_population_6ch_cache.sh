#!/bin/bash
#SBATCH --job-name=gcnm6-pw-cache
#SBATCH --output=logs/gcnm6-pw-cache_%j.out
#SBATCH --error=logs/gcnm6-pw-cache_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
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

PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/fundational_pvi}"
COORDINATE_ROOT="${COORDINATE_ROOT:-${REPO_ROOT}/gcnm_parquet/coordinate_direct_main_b045_v1}"
REGISTRY="${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/manifests/pw_population_within_mask05_seed42_v1.json}"
CACHE_ROOT="${CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/pw_main_within_seed42_v1}"

if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
  python -u scripts/build_population_within_split.py \
    --pvi-ml-root "${PVI_ML_ROOT}" \
    --coordinate-root "${COORDINATE_ROOT}" \
    --registry "${REGISTRY}" \
    --output "${SPLIT_MANIFEST}" \
    --seed 42
fi

if [[ -f "${CACHE_ROOT}/_SUCCESS" && -f "${CACHE_ROOT}/manifest.json" ]]; then
  echo "validated cache already exists: ${CACHE_ROOT}"
  exit 0
fi

resume=()
if [[ -f "${CACHE_ROOT}/_INCOMPLETE" ]]; then
  resume+=(--resume)
fi
python -u scripts/build_population_6ch_cache.py \
  --coordinate-root "${COORDINATE_ROOT}" \
  --registry "${REGISTRY}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output-root "${CACHE_ROOT}" \
  --workers 16 \
  --row-group-size 64 \
  "${resume[@]}"
