#!/bin/bash
# Pack sixteen independent ring-model trainings onto four NVL GPUs (four per GPU).
#SBATCH --job-name=gcnm-beats50-pack
#SBATCH --output=logs/gcnm-beats50-pack_%j.out
#SBATCH --error=logs/gcnm-beats50-pack_%j.err
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
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"

python -u -m gcnm_pvi.mesh_packed_runner \
  --manifest "${MESH_TASK_MANIFEST:-${REPO_ROOT}/data/manifests/mesh_training_v3.json}" \
  --stage train --workers 16 --task-cpus 4 --gpus 4
