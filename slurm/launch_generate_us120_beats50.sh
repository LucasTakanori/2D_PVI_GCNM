#!/bin/bash
# Generate the controlled 1,000-beat US120 comparison dataset. This is a CPU
# finite-element task and intentionally requests no GPU.
#SBATCH --job-name=us120-beats50-data
#SBATCH --output=logs/us120-beats50-data_%j.out
#SBATCH --error=logs/us120-beats50-data_%j.err
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

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_default_beats1000x50_US120_v1}"

python -u -m gcnm_pvi.generate_multisubject_beat_dataset \
  --config "${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}" \
  --out-dir "${DATASET_DIR}" \
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
