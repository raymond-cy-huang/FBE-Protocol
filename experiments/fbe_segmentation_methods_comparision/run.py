#!/usr/bin/env python3
"""Compare BBOX, SAM, and SAM2 raw and normalized masks."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mask_benchmark.run import (
    IMAGE_EXTS,
    add_label,
    detect_person_box,
    draw_box,
    full_image_box,
    load_sam,
    load_sam2,
    norm_mask,
    overlay_mask,
    predict_sam_mask,
    read_image,
    resize_like,
    run_bbox_mask_pose,
    write_image,
)
from fbe_protocol.mask.providers import (
    detect_face_core,
    select_face_filtered_mask,
    select_fixed_inverted_mask,
)

METHOD_LABELS = {
    "bbox": "BBOX",
    "sam": "SAM",
    "sam2": "SAM2",
}

METHOD_COLORS = {
    "bbox": (0, 255, 255),
    "sam": (0, 0, 255),
    "sam2": (255, 0, 0),
}


def maybe_reexec_in_conda() -> None:
    env_name = os.environ.get("FBE_CONDA_ENV", "fbe-protocol")
    if os.environ.get("FBE_PROTOCOL_NO_CONDA_REEXEC") == "1":
        return
    if os.environ.get("CONDA_DEFAULT_ENV") == env_name:
        return

    for candidate in [
        Path.home() / f"miniconda3/envs/{env_name}/bin/python",
        Path.home() / f"anaconda3/envs/{env_name}/bin/python",
    ]:
        if candidate.exists():
            env = {
                **os.environ,
                "FBE_PROTOCOL_NO_CONDA_REEXEC": "1",
                "CONDA_DEFAULT_ENV": env_name,
            }
            os.execve(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]], env)


if Path(sys.argv[0]).resolve() == Path(__file__).resolve():
    maybe_reexec_in_conda()


def collect_images(input_dir: Path, recursive: bool, limit: int | None) -> list[Path]:
    if recursive:
        images = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        images = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be greater than 0.")
        images = images[:limit]
    return images


def binary(mask: np.ndarray) -> np.ndarray:
    return mask > 0


def pairwise_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a_bin = binary(a)
    b_bin = binary(b)
    intersection = int(np.logical_and(a_bin, b_bin).sum())
    union = int(np.logical_or(a_bin, b_bin).sum())
    a_area = int(a_bin.sum())
    b_area = int(b_bin.sum())
    total = int(a_bin.size)
    xor = int(np.logical_xor(a_bin, b_bin).sum())
    dice_den = a_area + b_area
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float((2 * intersection) / dice_den) if dice_den else 1.0,
        "agreement": float(1.0 - (xor / total)) if total else 1.0,
        "intersection_px": float(intersection),
        "union_px": float(union),
        "xor_px": float(xor),
        "method_a_area_px": float(a_area),
        "method_b_area_px": float(b_area),
        "method_a_area_ratio": float(a_area / total) if total else 0.0,
        "method_b_area_ratio": float(b_area / total) if total else 0.0,
        "abs_area_ratio_diff": float(abs(a_area - b_area) / total) if total else 0.0,
    }


def append_pairwise_rows(
    rows: list[dict[str, object]],
    image_path: Path,
    mask_type: str,
    masks: dict[str, np.ndarray],
) -> None:
    for method_a, method_b in combinations(METHOD_LABELS, 2):
        metrics = pairwise_metrics(masks[method_a], masks[method_b])
        rows.append(
            {
                "image": image_path.name,
                "mask_type": mask_type,
                "method_a": method_a,
                "method_b": method_b,
                **metrics,
            }
        )


def select_sam_like_mask(
    predictor,
    image_bgr: np.ndarray,
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    selection_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if selection_mode == "fixed_inverted":
        selected = select_fixed_inverted_mask(predictor, image_rgb, box)
    elif selection_mode == "face_filter":
        face_core, face_core_source = detect_face_core(image_bgr)
        selected = select_face_filtered_mask(predictor, image_rgb, box, face_core)
        selected = {
            **selected,
            "face_core_source": face_core_source,
            "face_core_xyxy": ",".join(str(v) for v in face_core),
        }
    else:
        raw_mask = predict_sam_mask(predictor, image_rgb, box)
        selected = {
            "mask": raw_mask,
            "selection_reason": "best_score",
            "inverted": False,
            "selection_mode": "best_score",
        }
    selected_mask = selected["mask"]
    raw_output_mask = selected.get("raw", selected_mask)
    metadata = {k: v for k, v in selected.items() if k not in {"mask", "raw", "raw_inverted"}}
    metadata["selection_mode"] = selection_mode
    metadata["raw_output_polarity"] = "original_candidate" if "raw" in selected else "selected_candidate"
    metadata["normalized_input_polarity"] = "selected_mask"
    return selected_mask, raw_output_mask, metadata


def make_method_panel(
    image_bgr: np.ndarray,
    prompt_overlay: np.ndarray,
    raw_mask: np.ndarray,
    normalized_mask: np.ndarray,
    method: str,
    prompt_label: str,
) -> np.ndarray:
    width = image_bgr.shape[1]
    height = image_bgr.shape[0]
    raw_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    normalized_bgr = cv2.cvtColor(normalized_mask, cv2.COLOR_GRAY2BGR)
    raw_overlay = overlay_mask(image_bgr, raw_mask, (0, 180, 255))
    normalized_overlay = overlay_mask(image_bgr, normalized_mask, (0, 255, 0))
    label = METHOD_LABELS[method]
    cells = [
        add_label(prompt_overlay, f"{label}: {prompt_label}"),
        add_label(raw_bgr, f"{label}: raw mask"),
        add_label(raw_overlay, f"{label}: raw overlay"),
        add_label(normalized_bgr, f"{label}: normalized mask"),
        add_label(normalized_overlay, f"{label}: normalized overlay"),
    ]
    return np.concatenate([resize_like(cell, (width, height)) for cell in cells], axis=1)


def write_boundary_comparison(
    path: Path,
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    title: str,
) -> None:
    out = image_bgr.copy()
    for method, color in METHOD_COLORS.items():
        contours, _ = cv2.findContours(
            np.where(masks[method] > 0, 255, 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, color, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out,
        f"{title}: BBOX=yellow SAM=red SAM2=blue",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    write_image(path, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "images")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/fbe_segmentation_methods_comparision")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-checkpoint", type=Path, default=REPO_ROOT / "models/sam/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=REPO_ROOT / "models/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt-mode", choices=("opencv_box", "full_box"), default="full_box")
    parser.add_argument(
        "--sam-selection-mode",
        choices=("fixed_inverted", "face_filter", "best_score"),
        default="fixed_inverted",
        help="Candidate-selection policy for SAM/SAM2 raw masks.",
    )
    parser.add_argument("--invert-sam-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    images = collect_images(args.input_dir, args.recursive, args.limit)
    if not images:
        raise ValueError(f"No images found: {args.input_dir}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Input : {args.input_dir}")
    print(f"[INFO] Output: {args.output_dir}")
    print(f"[INFO] Images: {len(images)}")
    print(f"[INFO] Device: {device}")

    sam_predictor = load_sam(args.sam_model_type, args.sam_checkpoint, device)
    sam2_predictor = load_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device)

    summary_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    raw_metric_rows: list[dict[str, object]] = []
    normalized_metric_rows: list[dict[str, object]] = []

    for index, image_path in enumerate(images, start=1):
        print(f"[INFO] ({index}/{len(images)}) Processing {image_path}")
        image_bgr = read_image(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_dir = args.output_dir / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        if args.prompt_mode == "full_box":
            box, box_source = full_image_box(image_bgr)
        else:
            box, box_source = detect_person_box(image_bgr)

        box_overlay = draw_box(image_bgr, box, box_source)
        write_image(image_dir / "input_prompt_box.png", box_overlay)

        raw_bbox_path = run_bbox_mask_pose(image_path, image_dir)
        raw_bbox = cv2.imread(str(raw_bbox_path), cv2.IMREAD_GRAYSCALE)
        if raw_bbox is None:
            raise ValueError(f"Cannot read BBOX raw mask: {raw_bbox_path}")

        norm_input_sam, raw_sam, sam_metadata = select_sam_like_mask(
            sam_predictor,
            image_bgr,
            image_rgb,
            box,
            args.sam_selection_mode,
        )
        norm_input_sam2, raw_sam2, sam2_metadata = select_sam_like_mask(
            sam2_predictor,
            image_bgr,
            image_rgb,
            box,
            args.sam_selection_mode,
        )
        if args.invert_sam_masks:
            raw_sam = cv2.bitwise_not(raw_sam)
            raw_sam2 = cv2.bitwise_not(raw_sam2)
            norm_input_sam = cv2.bitwise_not(norm_input_sam)
            norm_input_sam2 = cv2.bitwise_not(norm_input_sam2)
            sam_metadata["extra_invert_sam_masks"] = True
            sam2_metadata["extra_invert_sam_masks"] = True

        raw_masks = {
            "bbox": raw_bbox,
            "sam": raw_sam,
            "sam2": raw_sam2,
        }
        normalized_inputs = {
            "bbox": raw_bbox,
            "sam": norm_input_sam,
            "sam2": norm_input_sam2,
        }
        normalized_masks = {method: norm_mask(image_bgr, raw) for method, raw in normalized_inputs.items()}

        for method in METHOD_LABELS:
            write_image(image_dir / f"raw_{method}.png", raw_masks[method])
            write_image(image_dir / f"normalized_{method}.png", normalized_masks[method])

        panels = [
            make_method_panel(image_bgr, image_bgr, raw_masks["bbox"], normalized_masks["bbox"], "bbox", "input"),
            make_method_panel(image_bgr, box_overlay, raw_masks["sam"], normalized_masks["sam"], "sam", box_source),
            make_method_panel(image_bgr, box_overlay, raw_masks["sam2"], normalized_masks["sam2"], "sam2", box_source),
        ]
        write_image(image_dir / "comparison.png", np.concatenate(panels, axis=0))
        write_boundary_comparison(image_dir / "boundary_raw.png", image_bgr, raw_masks, "raw")
        write_boundary_comparison(image_dir / "boundary_normalized.png", image_bgr, normalized_masks, "normalized")

        append_pairwise_rows(raw_metric_rows, image_path, "raw", raw_masks)
        append_pairwise_rows(normalized_metric_rows, image_path, "normalized", normalized_masks)
        for method, metadata in [("sam", sam_metadata), ("sam2", sam2_metadata)]:
            selection_rows.append(
                {
                    "image": image_path.name,
                    "method": method,
                    "box_source": box_source,
                    "box_xyxy": ",".join(str(v) for v in box),
                    **metadata,
                }
            )

        summary_rows.append(
            {
                "image": image_path.name,
                "box_source": box_source,
                "box_xyxy": ",".join(str(v) for v in box),
                "output_dir": str(image_dir.relative_to(REPO_ROOT)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "sam_selection_metadata.csv", selection_rows)
    write_csv(args.output_dir / "pairwise_raw_metrics.csv", raw_metric_rows)
    write_csv(args.output_dir / "pairwise_normalized_metrics.csv", normalized_metric_rows)
    print(f"[DONE] Results -> {args.output_dir}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
