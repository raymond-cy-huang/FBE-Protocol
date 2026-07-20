#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${FBE_CONDA_ENV:-fbe-protocol}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
MODEL_WEIGHTS_SCRIPT="${ROOT_DIR}/download_all_models_weights.sh"
RETRIES="${FBE_SETUP_RETRIES:-3}"
OPENMMLAB_PACKAGES=(
  "mmengine"
  "mmcv==2.1.0"
  "mmdet==3.3.0"
  "mmpretrain==1.2.0"
)
BBOX_RUNTIME_PACKAGES=(
  "hydra-core>=1.3.2"
  "iopath>=0.1.10"
  "sparsemax>=0.1.9"
  "loguru>=0.7.0"
)
CONDA_EXE="${CONDA_EXE:-}"

warn() {
  echo "[WARN] $*" >&2
}

info() {
  echo "[INFO] $*"
}

find_conda() {
  if [[ -n "${CONDA_EXE}" && -x "${CONDA_EXE}" ]]; then
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
    return 0
  fi

  for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      CONDA_EXE="${candidate}"
      return 0
    fi
  done

  return 1
}

run_with_retries() {
  local label="$1"
  shift

  local attempt=1
  while true; do
    info "${label} (attempt ${attempt}/${RETRIES})"
    if "$@"; then
      return 0
    fi

    if [[ "${attempt}" -ge "${RETRIES}" ]]; then
      echo "[ERROR] ${label} failed after ${RETRIES} attempts." >&2
      return 1
    fi

    attempt=$((attempt + 1))
    sleep 2
  done
}

setup_conda_env() {
  if ! find_conda; then
    warn "conda was not found. Skipping conda env setup for '${ENV_NAME}'."
    warn "Install Miniconda/Anaconda, then rerun: bash setup.sh"
    return 0
  fi

  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] Missing environment file: ${ENV_FILE}" >&2
    return 1
  fi

  if "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    run_with_retries \
      "Updating conda env '${ENV_NAME}'" \
      "${CONDA_EXE}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
  else
    run_with_retries \
      "Creating conda env '${ENV_NAME}'" \
      "${CONDA_EXE}" env create -n "${ENV_NAME}" -f "${ENV_FILE}"
  fi
}

install_openmmlab_deps() {
  if ! find_conda; then
    warn "conda was not found. Skipping OpenMMLab dependency setup."
    return 0
  fi

  run_with_retries \
    "Installing OpenMMLab dependencies into '${ENV_NAME}'" \
    "${CONDA_EXE}" run -n "${ENV_NAME}" mim install "${OPENMMLAB_PACKAGES[@]}"
}

install_bbox_runtime_deps() {
  if ! find_conda; then
    warn "conda was not found. Skipping BBoxMaskPose runtime dependency setup."
    return 0
  fi

  run_with_retries \
    "Installing BBoxMaskPose runtime dependencies into '${ENV_NAME}'" \
    "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install "${BBOX_RUNTIME_PACKAGES[@]}"
}

download_model_weights() {
  if [[ ! -f "${MODEL_WEIGHTS_SCRIPT}" ]]; then
    echo "[ERROR] Missing model weights downloader: ${MODEL_WEIGHTS_SCRIPT}" >&2
    return 1
  fi

  chmod +x "${MODEL_WEIGHTS_SCRIPT}"
  run_with_retries "Downloading all model weights" bash "${MODEL_WEIGHTS_SCRIPT}"
}

setup_conda_env
install_openmmlab_deps
install_bbox_runtime_deps
download_model_weights

info "Setup complete."
info "Run conda activate ${ENV_NAME} to start using the environment."
