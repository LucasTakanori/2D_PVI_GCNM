#!/bin/bash
# Record manifests for the four completed component-specific US120 trainings.
# No model training is performed here.
#SBATCH --job-name=us120-manifest-repair
#SBATCH --output=logs/us120-manifest-repair_%A_%a.out
#SBATCH --error=logs/us120-manifest-repair_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=01:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"

FAMILIES=(coordinate coordinate global_voltage_slots global_voltage_slots)
COMPONENTS=(hp lp hp lp)
TASK_ID="${SLURM_ARRAY_TASK_ID:?array 0-3 required}"
FAMILY="${FAMILIES[${TASK_ID}]}"
COMPONENT="${COMPONENTS[${TASK_ID}]}"
RING=US120
CONFIG="${REPO_ROOT}/configs/rings_b045/${RING}.yaml"
MODEL_DIR="${REPO_ROOT}/models/hp_lp_us120_v1/${FAMILY}/${COMPONENT}/${RING}"
python -u -m gcnm_pvi.mesh_run_manifest --record \
  --family "${FAMILY}" --ring "${RING}" --config "${CONFIG}" \
  --model-dir "${MODEL_DIR}" --output "${MODEL_DIR}/run_manifest.json"
