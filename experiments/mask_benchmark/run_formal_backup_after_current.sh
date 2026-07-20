#!/usr/bin/env bash
set -euo pipefail

FORMAL_DIR="/mnt/d/Ph.D/01_Experiments_GAN_Inv_Log/2026.03.08_All_Experiments_Start/exper_16_mask_bench_mark/formal"
REPO_ROOT="/home/rz/raymond_project/FBE-Protocol"
PYTHON="/home/rz/miniconda3/envs/fbe-protocol/bin/python"

if [[ -f "${FORMAL_DIR}/run.pid" ]]; then
  formal_pid="$(cat "${FORMAL_DIR}/run.pid")"
  while ps -p "${formal_pid}" >/dev/null 2>&1; do
    sleep 20
  done
fi

cd "${REPO_ROOT}"
"${PYTHON}" "${REPO_ROOT}/experiments/mask_benchmark/generate_formal_masks.py" \
  --input-root /mnt/d/Dataset/CelebAMask-combined/gt_image_backup_100 \
  --output-dir "${FORMAL_DIR}" \
  --limit 101
