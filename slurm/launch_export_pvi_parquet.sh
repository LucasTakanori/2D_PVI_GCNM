#!/bin/bash
#SBATCH --job-name=gcnm-pvi-export
#SBATCH --output=logs/gcnm-pvi-export_%j.out
#SBATCH --error=logs/gcnm-pvi-export_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# Thirty-two concurrent real sessions retain large element/raster buffers while
# shards are serialized.  Production exports need headroom for those buffers
# and Arrow compression temporaries.
#SBATCH --mem=500GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:2

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export GCNM_PHYSICS_WORKERS="${GCNM_PHYSICS_WORKERS:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
ARGS=(
  --registry "${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}"
  --family "${FAMILY:?set FAMILY}"
  --output-root "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
  --checkpoint-root "${CHECKPOINT_ROOT:-${REPO_ROOT}/models/mesh_representations/${FAMILY}}"
  --session-workers "${GCNM_EXPORT_SESSION_WORKERS:-32}"
  --gpu-count "${GCNM_EXPORT_GPU_COUNT:-2}"
)
if [[ "${FAMILY}" == "diffusion" ]]; then
  ARGS+=(--prior-root "${PRIOR_ROOT:-${REPO_ROOT}/data/mesh_training/diffusion}")
fi
if [[ -n "${SUBJECTS:-}" ]]; then read -r -a subject_args <<< "${SUBJECTS}"; ARGS+=(--subjects "${subject_args[@]}"); fi
python -u -m gcnm_pvi.export_pvi_parquet "${ARGS[@]}"
python -m gcnm_pvi.validate_pvi_parquet --root "${OUTPUT_ROOT}"
