#!/bin/bash
# Four CRS processes per NVL GPU: 16 dynamic workers over the 364-run matrix.
#SBATCH --job-name=coord91-crspack
#SBATCH --output=logs/coord91-crspack_%j.out
#SBATCH --error=logs/coord91-crspack_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:4

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m gcnm_pvi.bp_packed_runner \
  --manifest "${TASK_MANIFEST:?set TASK_MANIFEST}" \
  --artifact-root "${ARTIFACT_ROOT:?set ARTIFACT_ROOT}" \
  --launcher "${REPO_ROOT}/slurm/launch_train_coordinate_main_b045_crs_bp.sh" \
  --workers 16 --gpus 4 \
  --logs "${REPO_ROOT}/logs/coordinate_main_b045_crs_bp"
