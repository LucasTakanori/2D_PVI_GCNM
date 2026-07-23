#!/bin/bash
# Four concurrent CPU/FEM shards: 20 train, 5 validation, 5 test anatomies.
#SBATCH --job-name=diff-cohort-gen
#SBATCH --output=logs/diff-cohort-gen_%j.out
#SBATCH --error=logs/diff-cohort-gen_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
PYTHON_ENV="$(dirname "$(dirname "${GCNM_PYTHON}")")"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-diff-cohort-${SLURM_JOB_ID}"

SHARD_PARENT="${SHARD_PARENT:-${REPO_ROOT}/data/differential_generalization_US120_v2_shards}"
MERGED_ROOT="${MERGED_ROOT:-${REPO_ROOT}/data/full_band_beats_US120_differential_generalization_v2}"
DIFFERENTIAL_ROOT="${DIFFERENTIAL_ROOT:-${REPO_ROOT}/data/differential_generalization_US120_clean_v2}"
if [[ -e "${SHARD_PARENT}" || -e "${MERGED_ROOT}" || -e "${DIFFERENTIAL_ROOT}" ]]; then
  echo "immutable shard or merged output already exists" >&2
  exit 2
fi
mkdir -p "${SHARD_PARENT}"

# Counts sum to 20/5/5. Every process owns its FEM objects. Sixteen BLAS threads
# per process keep four simultaneous generators within the 64-core allocation.
TRAIN_COUNTS=(5 5 5 5)
VALIDATION_COUNTS=(2 1 1 1)
TEST_COUNTS=(2 1 1 1)
SEEDS=(20260731 20260732 20260733 20260734)
pids=()
for shard in 0 1 2 3; do
  output="${SHARD_PARENT}/shard${shard}"
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=220G \
    --output="${REPO_ROOT}/logs/diff-cohort-gen_${SLURM_JOB_ID}_shard${shard}.out" \
    --error="${REPO_ROOT}/logs/diff-cohort-gen_${SLURM_JOB_ID}_shard${shard}.err" \
    /usr/bin/env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
        NUMEXPR_NUM_THREADS=16 \
        PYTHONPATH="${PYTHONPATH}" MPLCONFIGDIR="${MPLCONFIGDIR}/shard${shard}" \
    "${PYTHON_ENV}/bin/python" -u -m gcnm_pvi.generate_hp_lp_beat_dataset \
      --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
      --output-root "${output}" \
      --train-anatomies "${TRAIN_COUNTS[${shard}]}" \
      --validation-anatomies "${VALIDATION_COUNTS[${shard}]}" \
      --test-anatomies "${TEST_COUNTS[${shard}]}" \
      --train-mode linearized --validation-mode nonlinear --test-mode nonlinear \
      --linearization-reference anatomy --jacobian-bank-size 0 \
      --seed "${SEEDS[${shard}]}" \
      --maximum-linearized-component-nrmse 0.50 \
      --minimum-linearized-component-correlation 0.80 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more synthetic generator shards failed; merge was not attempted" >&2
  exit 3
fi

"${PYTHON_ENV}/bin/python" -u -m gcnm_pvi.merge_hp_lp_dataset_shards \
  --shard "${SHARD_PARENT}/shard0" \
  --shard "${SHARD_PARENT}/shard1" \
  --shard "${SHARD_PARENT}/shard2" \
  --shard "${SHARD_PARENT}/shard3" \
  --output-root "${MERGED_ROOT}"

"${PYTHON_ENV}/bin/python" -u -m gcnm_pvi.validate_hp_lp_beat_dataset \
  --root "${MERGED_ROOT}" --allow-unverified-exact \
  --output "${MERGED_ROOT}/validation.json"

"${PYTHON_ENV}/bin/python" -u -m gcnm_pvi.make_differential_cohort_pack \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --dataset-root "${MERGED_ROOT}" \
  --output-root "${DIFFERENTIAL_ROOT}" \
  --voltage-key V_clean_delta_reference

echo "[$(date --iso-8601=seconds)] differential generalization cohort complete"
