#!/bin/bash
#SBATCH --job-name=gcnm6-pw
#SBATCH --output=logs/gcnm6-pw_%A_%a.out
#SBATCH --error=logs/gcnm6-pw_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

case "${SLURM_ARRAY_TASK_ID}" in
  0) experiment=crt ;;
  1) experiment=samba ;;
  *) echo "invalid task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
CACHE_ROOT="${CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/img2wave_main_within_seed42_stratified_v2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/population_within_gcnm6ch_img2wave_stratified_v2}"

test -f "${CACHE_ROOT}/_SUCCESS"
test -f "${CACHE_ROOT}/manifest.json"
python -u -m gcnm_pvi.train_population_6ch_bp \
  --experiment "${experiment}" \
  --pvi-ml-root "${PVI_ML_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --seed 42 \
  --batch-size 32 \
  --num-workers 8 \
  --max-epochs 500
