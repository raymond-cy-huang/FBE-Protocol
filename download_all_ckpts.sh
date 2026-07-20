#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info() {
  echo "[INFO] $*"
}

run_model_downloader() {
  local label="$1"
  local script="$2"

  if [[ ! -f "${script}" ]]; then
    echo "[ERROR] Missing ${label} weight downloader: ${script}" >&2
    return 1
  fi

  chmod +x "${script}"
  info "Running ${label} downloader: ${script#${ROOT_DIR}/}"
  (
    cd "$(dirname "${script}")"
    bash "./$(basename "${script}")"
  )
}

run_model_downloader \
  "BBoxMaskPose" \
  "${ROOT_DIR}/models/BBoxMaskPose/models/SAM/download_ckpts.sh"

run_model_downloader \
  "SAM" \
  "${ROOT_DIR}/models/sam/checkpoints/download_ckpts.sh"

run_model_downloader \
  "SAM2" \
  "${ROOT_DIR}/models/sam2/checkpoints/download_ckpts.sh"

info "All model weight download scripts finished."
