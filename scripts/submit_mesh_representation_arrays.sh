#!/usr/bin/env bash
# Generate the 15 synthetic beat packs and train the 30 ring-specific GCNMs.
# Parquet materialization and BP training are owned by pvi_gcnm_bp_pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs

"${GCNM_PYTHON}" -m gcnm_pvi.ring_configs
"${GCNM_PYTHON}" -m gcnm_pvi.mesh_registry --data-root "${PVI_DATA_ROOT}"
"${GCNM_PYTHON}" -m gcnm_pvi.mesh_run_manifest

DATASET_ROOT="${REPO_ROOT}/data/mesh_training/beats1000x50_v1"
CHECKPOINT_ROOT="${REPO_ROOT}/models/mesh_representations/beats1000x50_v1"
RESULT_ROOT="${REPO_ROOT}/data/mesh_training_results/beats1000x50_v1"

for path in "${DATASET_ROOT}" "${CHECKPOINT_ROOT}" "${RESULT_ROOT}"; do
  if [[ -e "${path}" ]]; then
    echo "ERROR: immutable GCNM output already exists: ${path}" >&2
    exit 1
  fi
done

# Four CPU tasks exactly fill the 64-CPU/1-TB ceiling.
data_job="$(sbatch --parsable --array=0-14%4 \
  --export="ALL,REPO_ROOT=${REPO_ROOT},DATASET_ROOT=${DATASET_ROOT}" \
  slurm/launch_generate_mesh_beats50.sh)"

# One four-GPU allocation runs sixteen model processes (four per GPU). Each
# model receives four explicit CPU physics lanes.
train_job="$(sbatch --parsable \
  --dependency="afterok:${data_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},MESH_TASK_MANIFEST=${REPO_ROOT}/data/manifests/mesh_training_v3.json" \
  slurm/launch_train_beats50_packed.sh)"

echo "15 ring beat datasets (array 0-14%4, CPU only): ${data_job}"
echo "30 coordinate/global GCNM trainings (16 processes on 4 GPUs): ${train_job}"
echo "Parquet export and BP training were not submitted; use pvi_gcnm_bp_pipeline."
