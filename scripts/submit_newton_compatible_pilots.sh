#!/usr/bin/env bash
# Submit four independent US120 pilots: two families x two subjects.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs

PILOT_ROOT="${GCNM_PILOT_ROOT:-${REPO_ROOT}/pilot_outputs/newton_compatible_v1}"
CHECKPOINT_ROOT="${REPO_ROOT}/models/mesh_representations/beats1000x50_v1"
declare -a jobs=()

for family in coordinate global_voltage_slots; do
  for subject in subject006 subject010; do
    output="${PILOT_ROOT}/${family}/${subject}"
    if [[ -e "${output}" ]]; then
      echo "ERROR: immutable pilot output exists: ${output}" >&2
      exit 1
    fi
  done
done

for family in coordinate global_voltage_slots; do
  for subject in subject006 subject010; do
    output="${PILOT_ROOT}/${family}/${subject}"
    job="$(sbatch --parsable \
      --export="ALL,REPO_ROOT=${REPO_ROOT},FAMILY=${family},SUBJECT=${subject},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},OUTPUT_ROOT=${output}" \
      slurm/launch_newton_compatible_pilot.sh)"
    jobs+=("${family}/${subject}:${job}")
  done
done

printf '%s\n' "${jobs[@]}"
