#!/bin/bash
# Refresh authoritative ring/checkpoint/config/mesh hashes after all training.
#SBATCH --job-name=gcnm-mesh-manifests
#SBATCH --output=logs/gcnm-mesh-manifests_%j.out
#SBATCH --error=logs/gcnm-mesh-manifests_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
python -m gcnm_pvi.mesh_packed_runner \
  --manifest data/manifests/mesh_training_v2.json \
  --stage train --workers 8 --task-cpus 8 --gpus 4 --preflight
RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US120 US125 US130)
for FAMILY in coordinate global_voltage_slots; do
  for RING in "${RINGS[@]}"; do
    MODEL_DIR="${REPO_ROOT}/models/mesh_representations/${FAMILY}/${RING}"
    python -m gcnm_pvi.mesh_run_manifest --record \
      --family "${FAMILY}" --ring "${RING}" \
      --config "${REPO_ROOT}/configs/rings_b045/${RING}.yaml" \
      --model-dir "${MODEL_DIR}" --output "${MODEL_DIR}/run_manifest.json"
  done
done
echo "refreshed 30 selected mesh run manifests"
