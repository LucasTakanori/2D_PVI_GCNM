#!/bin/bash
# Archive the four completed CRS BP artifact trees after GIF generation.
#SBATCH --job-name=coord91-crs-tars
#SBATCH --output=logs/coord91-crs-tars_%j.out
#SBATCH --error=logs/coord91-crs-tars_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/coordinate_main_b045_crs_bp_v1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${REPO_ROOT}/exports/coordinate_main_b045_crs_bp_v1_archives}"
mkdir -p "${ARCHIVE_ROOT}"

targets=(
  gcnm-coordinate-3ch-crs-image-to-fiducials
  gcnm-coordinate-3ch-crs-image-to-waveform
  gcnm-newton_coordinate-6ch-crs-image-to-fiducials
  gcnm-newton_coordinate-6ch-crs-image-to-waveform
)

for target in "${targets[@]}"; do
  source_dir="${ARTIFACT_ROOT}/${target}"
  archive="${ARCHIVE_ROOT}/${target}.tar.gz"
  temporary="${archive}.partial.${SLURM_JOB_ID}"
  [[ -d "${source_dir}" ]] || {
    echo "missing CRS artifact directory: ${source_dir}" >&2
    exit 2
  }
  trap 'rm -f "${temporary}"' EXIT
  echo "[$(date --iso-8601=seconds)] archiving ${target}"
  tar -C "${ARTIFACT_ROOT}" \
    -I "pigz -p ${SLURM_CPUS_PER_TASK}" \
    -cf "${temporary}" "${target}"
  mv -f "${temporary}" "${archive}"
  trap - EXIT
  echo "[$(date --iso-8601=seconds)] completed ${archive}"
done

checksum_tmp="${ARCHIVE_ROOT}/SHA256SUMS.partial.${SLURM_JOB_ID}"
trap 'rm -f "${checksum_tmp}"' EXIT
(
  cd "${ARCHIVE_ROOT}"
  sha256sum \
    gcnm-coordinate-3ch-crs-image-to-fiducials.tar.gz \
    gcnm-coordinate-3ch-crs-image-to-waveform.tar.gz \
    gcnm-newton_coordinate-6ch-crs-image-to-fiducials.tar.gz \
    gcnm-newton_coordinate-6ch-crs-image-to-waveform.tar.gz
) > "${checksum_tmp}"
mv -f "${checksum_tmp}" "${ARCHIVE_ROOT}/SHA256SUMS"
trap - EXIT

du -h "${ARCHIVE_ROOT}"/*.tar.gz
echo "[$(date --iso-8601=seconds)] all CRS archives complete"
