#!/usr/bin/env bash
# Fail if the staged Git repository contains private/generated project artifacts.
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: run this inside the clean publication Git clone" >&2
  exit 2
fi

PROHIBITED='(^|/)(data|logs|models|env|\.venv|venv|samples|outputs|checkpoints)(/|$)|\.(npy|npz|h5|hdf5|mat|eit|pt|pth|ckpt|out|err|log)$'
ALLOWLIST='^(\.gitignore|README\.md|requirements\.txt|cluster\.env\.example|ANATOMICAL_GCNM\.md|IMPLEMENTATION_REPORT\.md|PVI_GCNM_ADAPTATION_PLAN\.md|SUBJECT006_STATUS\.md|configs/[^/]+\.yaml|gcnm_pvi/[^/]+\.py|scripts/[^/]+\.sh|slurm/[^/]+\.sh|tests/test_[^/]+\.py|reports/gcnm_pvi_latex/(README\.md|main\.(tex|pdf)|make_[^/]+\.py|figures/[^/]+\.pdf))$'
allowlist_failure=0
while IFS= read -r -d '' file; do
  if [[ "${file}" =~ ${PROHIBITED} ]]; then
    echo "ERROR: prohibited data, model, environment, or log artifact is tracked: ${file}" >&2
    allowlist_failure=1
  fi
  if [[ ! "${file}" =~ ${ALLOWLIST} ]]; then
    echo "ERROR: tracked path is outside the publication allowlist: ${file}" >&2
    allowlist_failure=1
  fi
  mode=$(git ls-files -s -- "${file}" | awk '{print $1}')
  if [[ "${mode}" == "120000" ]]; then
    echo "ERROR: tracked symlink is not allowed: ${file}" >&2
    allowlist_failure=1
  fi
done < <(git ls-files -z)
if (( allowlist_failure )); then exit 1; fi

if git grep --cached -IEn \
  'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY' -- .; then
  echo "ERROR: possible credential or private key in tracked text" >&2
  exit 1
fi

if git grep --cached -IEn '/[m]mfs1|/[h]ome/[A-Za-z0-9._-]+|ece_[b]st_link' -- .; then
  echo "ERROR: private cluster path found in tracked text" >&2
  exit 1
fi

large=0
while IFS= read -r -d '' file; do
  size=$(git cat-file -s ":${file}")
  if (( size > 10000000 )); then
    echo "ERROR: tracked file exceeds 10 MB: ${file} (${size} bytes)" >&2
    large=1
  fi
done < <(git ls-files -z)
if (( large )); then exit 1; fi

echo "Publication audit passed: no private artifacts, credentials, or files over 10 MB."
