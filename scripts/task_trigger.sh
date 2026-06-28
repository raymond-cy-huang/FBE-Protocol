#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_DIR="${REPO_ROOT}/scripts"
CONFIG_PATH="${REPO_ROOT}/configs/global_path.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

read_profile_keys() {
    local prefix="$1"
    "${PYTHON_BIN}" - "${CONFIG_PATH}" "${prefix}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
prefix = sys.argv[2]
try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML is required to read configs/global_path.yaml. Run `bash setup.sh`.")

if not path.exists():
    raise SystemExit(0)
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
if not isinstance(data, dict):
    raise SystemExit(0)
for key in sorted(k for k in data if k.startswith(prefix)):
    print(key)
PY
}

read_config_value() {
    local section="$1"
    local key="$2"
    local default_value="$3"
    "${PYTHON_BIN}" - "${CONFIG_PATH}" "${section}" "${key}" "${default_value}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
section = sys.argv[2]
key = sys.argv[3]
default = sys.argv[4]
try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML is required to read configs/global_path.yaml. Run `bash setup.sh`.")

if not path.exists():
    print(default)
    raise SystemExit(0)
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
section_data = data.get(section, {}) if isinstance(data, dict) else {}
value = section_data.get(key, default) if isinstance(section_data, dict) else default
print(value)
PY
}

resolve_repo_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${REPO_ROOT}/${path}"
    fi
}

choose_from_list() {
    local title="$1"
    shift
    local items=("$@")
    local choice

    echo "${title}" >&2
    for i in "${!items[@]}"; do
        printf '(%d). %s\n' "$((i + 1))" "$(basename "${items[$i]}")" >&2
    done

    while true; do
        read -r -p "Select number: " choice
        if [[ "${choice}" =~ ^[0-9]+$ ]] && ((choice >= 1 && choice <= ${#items[@]})); then
            printf '%s\n' "${items[$((choice - 1))]}"
            return 0
        fi
        echo "Invalid selection. Please choose 1-${#items[@]}." >&2
    done
}

read_variant_px_default() {
    local prompt="$1"
    local default_value="$2"
    local value

    while true; do
        read -r -p "${prompt} [${default_value}]: " value
        value="${value:-${default_value}}"
        if [[ "${value}" == "-1" || "${value}" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "${value}"
            return 0
        fi
        echo "Invalid value. Please enter -1 or a non-negative integer." >&2
    done
}

read_float_default() {
    local prompt="$1"
    local default_value="$2"
    local value

    while true; do
        read -r -p "${prompt} [${default_value}]: " value
        value="${value:-${default_value}}"
        if [[ "${value}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
            printf '%s\n' "${value}"
            return 0
        fi
        echo "Invalid value. Please enter a number." >&2
    done
}

read_choice_default() {
    local prompt="$1"
    local default_value="$2"
    local value

    while true; do
        read -r -p "${prompt} [${default_value}]: " value
        value="${value:-${default_value}}"
        if [[ "${value}" == "foreground" || "${value}" == "background" ]]; then
            printf '%s\n' "${value}"
            return 0
        fi
        echo "Invalid value. Please enter foreground or background." >&2
    done
}

main() {
    shopt -s nullglob

    local tasks=("${TASK_DIR}"/*.py)
    if ((${#tasks[@]} == 0)); then
        echo "[ERROR] No task scripts found in ${TASK_DIR}" >&2
        exit 1
    fi

    local task
    task="$(choose_from_list "Which task do you want to run?" "${tasks[@]}")"
    local task_name
    task_name="$(basename "${task}")"
    local task_stem
    task_stem="${task_name%.py}"

    echo
    cd "${REPO_ROOT}"

    local profile_prefix
    if [[ "${task_name}" == "fbe_background_swapping.py" ]]; then
        profile_prefix="fbe_background_swapping"
    elif [[ "${task_name}" == "run_copy_paste_composition.py" ]]; then
        profile_prefix="copy_paste_composition_path"
    else
        profile_prefix="fbe_extract_multi_masks_path"
    fi

    mapfile -t profiles < <(read_profile_keys "${profile_prefix}")
    if ((${#profiles[@]} == 0)); then
        profiles=("${profile_prefix}00")
    fi
    local profile
    profile="$(choose_from_list "Which path profile do you want to use?" "${profiles[@]}")"
    local image_dir
    image_dir="$(read_config_value "${profile}" input_dir images)"
    local output_dir
    output_dir="$(read_config_value "${profile}" output_dir results)"
    local output_layout
    output_layout="$(read_config_value "${profile}" output_layout task)"
    local output_mode
    output_mode="$(read_config_value "${profile}" output_mode full)"

    if [[ "${task_name}" == "run_copy_paste_composition.py" ]]; then
        local dataset_root
        dataset_root="$(read_config_value "${profile}" dataset_root "")"
        local resolved_output_dir
        resolved_output_dir="$(resolve_repo_path "${output_dir}")"

        echo "[INFO] Profile: ${profile}"
        echo "[INFO] Dataset: ${dataset_root}"
        echo "[INFO] Output : ${resolved_output_dir}"
        echo "[INFO] Running ${task_name}"
        exec "${PYTHON_BIN}" "${task}" --path-profile "${profile}"
    fi

    local variant_px_default
    variant_px_default="$(read_config_value "${profile}" variant_px 5)"
    local variant_px
    variant_px="$(read_variant_px_default "Mask variant radius in pixels (erode/dilate, -1 disables variants)" "${variant_px_default}")"

    if [[ "${task_name}" == "fbe_background_swapping.py" ]]; then
        local dataset_root
        dataset_root="$(read_config_value "${profile}" dataset_root "")"
        local resolved_output_dir
        resolved_output_dir="$(resolve_repo_path "${output_dir}")"
        local mask_region
        mask_region="$(read_choice_default "Mask region for IoU pairing (foreground/background)" "$(read_config_value "${profile}" mask_region background)")"
        local min_iou
        min_iou="$(read_float_default "Minimum IoU threshold (-1 disables pruning threshold)" "$(read_config_value "${profile}" min_iou -1)")"
        local max_pairs
        max_pairs="$(read_variant_px_default "Max pairs" "$(read_config_value "${profile}" max_pairs 20)")"
        local anchor_count
        anchor_count="$(read_variant_px_default "Anchor count (0 disables anchor mode)" "$(read_config_value "${profile}" anchor_count 10)")"
        local pair_mask_size
        pair_mask_size="$(read_variant_px_default "Pair mask size" "$(read_config_value "${profile}" pair_mask_size 128)")"

        echo "[INFO] Profile: ${profile}"
        if [[ -n "${dataset_root}" ]]; then
            echo "[INFO] Dataset: ${dataset_root}"
        else
            echo "[INFO] Images : $(resolve_repo_path "${image_dir}")"
            echo "[INFO] Masks  : $(resolve_repo_path "$(read_config_value "${profile}" mask_dir results/fbe_extract_multi_masks)")"
        fi
        local swap_output
        swap_output="${resolved_output_dir}"
        if [[ "${output_layout}" == "task" ]]; then
            swap_output="${swap_output}/${task_stem}"
        fi
        echo "[INFO] Output : ${swap_output}"
        echo "[INFO] IoU    : region=${mask_region}, min=${min_iou}, max_pairs=${max_pairs}, anchors=${anchor_count}"
        echo "[INFO] Variant radius: ${variant_px}"
        echo "[INFO] Running ${task_name}"
        exec "${PYTHON_BIN}" "${task}" \
            --path-profile "${profile}" \
            --mask-region "${mask_region}" \
            --min-iou "${min_iou}" \
            --max-pairs "${max_pairs}" \
            --anchor-count "${anchor_count}" \
            --pair-mask-size "${pair_mask_size}" \
            --variant-px "${variant_px}"
    fi

    if [[ "${task_name}" == "fbe_extract_multi_masks.py" ]]; then
        local multi_output
        multi_output="$(resolve_repo_path "${output_dir}")"
        if [[ "${output_layout}" == "task" ]]; then
            multi_output="${multi_output}/${task_stem}"
        fi
        echo "[INFO] Profile: ${profile}"
        echo "[INFO] Input  : $(resolve_repo_path "${image_dir}")"
        echo "[INFO] Output : ${multi_output}"
        echo "[INFO] Mode   : ${output_mode}"
        echo "[INFO] Variant radius: ${variant_px}"
        echo "[INFO] Running ${task_name}"
        exec "${PYTHON_BIN}" "${task}" --path-profile "${profile}" --variant-px "${variant_px}"
    fi

    local resolved_image_dir
    resolved_image_dir="$(resolve_repo_path "${image_dir}")"
    local resolved_output_dir
    resolved_output_dir="$(resolve_repo_path "${output_dir}")"
    if [[ "${output_layout}" == "task" ]]; then
        resolved_output_dir="${resolved_output_dir}/${task_stem}"
    fi
    local images=(
        "${resolved_image_dir}"/*.png
        "${resolved_image_dir}"/*.jpg
        "${resolved_image_dir}"/*.jpeg
        "${resolved_image_dir}"/*.bmp
        "${resolved_image_dir}"/*.webp
        "${resolved_image_dir}"/*.tif
        "${resolved_image_dir}"/*.tiff
    )
    if ((${#images[@]} == 0)); then
        echo "[ERROR] No images found in ${resolved_image_dir}" >&2
        exit 1
    fi

    local image
    image="$(choose_from_list "Which image do you want to process?" "${images[@]}")"

    echo
    echo "[INFO] Output: ${resolved_output_dir}"
    echo "[INFO] Profile: ${profile}"
    echo "[INFO] Mode  : ${output_mode}"
    echo "[INFO] Variant radius: ${variant_px}"
    echo "[INFO] Running ${task_name} on $(basename "${image}")"
    exec "${PYTHON_BIN}" "${task}" "${image}" --output-dir "${resolved_output_dir}" --output-layout "${output_layout}" --output-mode "${output_mode}" --variant-px "${variant_px}"
}

main "$@"
