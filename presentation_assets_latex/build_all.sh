#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TECTONIC_BIN="${TECTONIC:-}"

if [[ -z "${TECTONIC_BIN}" ]]; then
  if command -v tectonic >/dev/null 2>&1; then
    TECTONIC_BIN="$(command -v tectonic)"
  elif [[ -x /home/lsanc68/.local/bin/tectonic ]]; then
    TECTONIC_BIN=/home/lsanc68/.local/bin/tectonic
  else
    echo "Tectonic was not found. Set TECTONIC=/absolute/path/to/tectonic." >&2
    exit 1
  fi
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/tectonic-cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/tmp/tectonic-config}"

compile_group() {
  local group="$1"
  local source_dir="${ROOT}/${group}"
  local output_dir="${ROOT}/build/${group}"
  local source

  mkdir -p "${output_dir}"
  shopt -s nullglob
  for source in "${source_dir}"/*.tex; do
    echo "[${group}] $(basename "${source}")"
    (
      cd "${source_dir}"
      "${TECTONIC_BIN}" --keep-logs --outdir "${output_dir}" "$(basename "${source}")"
    )
  done
  shopt -u nullglob
}

compile_group tables
compile_group charts
compile_group galleries

echo "Rendered assets are in ${ROOT}/build"
