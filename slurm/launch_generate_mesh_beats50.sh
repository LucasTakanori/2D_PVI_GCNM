#!/bin/bash
# One CPU array task generates one ring's shared 1,000-beat training pack.
#SBATCH --job-name=gcnm-beats50-data
#SBATCH --output=logs/gcnm-beats50-data_%A_%a.out
#SBATCH --error=logs/gcnm-beats50-data_%A_%a.err
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
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US120 US125 US130)
RING="${RING:-${RINGS[${SLURM_ARRAY_TASK_ID:?array task id is required}]}}"
CONFIG="${REPO_ROOT}/configs/rings_b045/${RING}.yaml"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/mesh_training/beats1000x50_v1}"
DATASET_DIR="${DATASET_ROOT}/${RING}"

if [[ -e "${DATASET_DIR}" ]]; then
  echo "ERROR: immutable beat dataset already exists: ${DATASET_DIR}" >&2
  exit 1
fi

python -u -m gcnm_pvi.generate_multisubject_beat_dataset \
  --config "${CONFIG}" --out-dir "${DATASET_DIR}" \
  --train-beats 800 --validation-beats 100 --test-beats 100 \
  --frames 50 --sample-stride 1 \
  --train-samples-per-beat 50 --validation-samples-per-beat 50 \
  --train-nonlinear-fraction 0 --validation-nonlinear-fraction 0 \
  --jacobian-bank-size 8 --seed 20260716 \
  --finger-size-variation 0.08 --finger-rotation-deg 10 \
  --finger-position-mm 1.0 \
  --artery-size-variation 0.20 --artery-position-mm 1.0 \
  --artery-rotation-deg 15 --conductivity-variation 0.10 \
  --diffusion-variation 0.15 --waveform-shape-variation 0.12 \
  --duration-variation 0.08 --white-noise-rel 0.03 \
  --correlated-noise-rel 0.02 --contact-static-sd 0.15 \
  --contact-drift-sd 0.003 --channel-gain-sd 0.02 \
  --current-gain-sd 0.03 --artifact-probability 0 \
  --nonvascular-probability 0

python -m gcnm_pvi.validate_beat_dataset --root "${DATASET_DIR}"
