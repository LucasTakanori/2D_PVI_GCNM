#!/usr/bin/env bash
# Re-render and archive both Figure 26 three-beat datasets under the exact
# visualization contract of the original ten-beat audit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"

REFERENCE_ROOT="${REFERENCE_ROOT:-${REPO_ROOT}/exports/all_patients_64ch_six_channel_10beats_v1}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-24}"

source_names=(
  "figure26_dual_mesh_sensitivity_3beats_v1"
  "figure26_projected_fine_trained_3beats_v1"
)
output_names=(
  "figure26_dual_mesh_sensitivity_3beats_matched_to_10beat_v2"
  "figure26_projected_fine_trained_3beats_matched_to_10beat_v2"
)

mkdir -p logs
for index in "${!source_names[@]}"; do
  source_root="${REPO_ROOT}/exports/${source_names[index]}"
  output_root="${REPO_ROOT}/exports/${output_names[index]}"
  archive_path="${output_root}.zip"
  for path in "${output_root}" "${archive_path}" "${archive_path}.sha256"; do
    if [[ -e "${path}" ]]; then
      echo "refusing to overwrite existing output: ${path}" >&2
      exit 2
    fi
  done

  "${GCNM_PYTHON}" -u scripts/export_matched_three_beat_comparison.py prepare \
    --source-root "${source_root}" \
    --reference-root "${REFERENCE_ROOT}" \
    --output-root "${output_root}"

  array_job="$(
    sbatch --parsable \
      --array="0-90%${ARRAY_CONCURRENCY}" \
      --export="ALL,REPO_ROOT=${REPO_ROOT},MATCHED_OUTPUT_ROOT=${output_root}" \
      slurm/launch_export_matched_three_beat_subject.sh
  )"
  array_id="${array_job%%;*}"
  finalize_job="$(
    sbatch --parsable \
      --dependency="afterok:${array_id}" \
      --export="ALL,REPO_ROOT=${REPO_ROOT},MATCHED_OUTPUT_ROOT=${output_root}" \
      slurm/launch_finalize_matched_three_beat.sh
  )"
  finalize_id="${finalize_job%%;*}"
  archive_job="$(
    sbatch --parsable \
      --dependency="afterok:${finalize_id}" \
      --export="ALL,REPO_ROOT=${REPO_ROOT},FIGURE26_SOURCE_NAME=${output_names[index]}" \
      slurm/launch_archive_figure26_dual_mesh.sh
  )"

  printf '%s\n' \
    "source: ${source_names[index]}" \
    "matched output: ${output_names[index]}" \
    "subject array: ${array_job}" \
    "validation: ${finalize_job}" \
    "archive: ${archive_job}"
done
