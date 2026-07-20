#!/usr/bin/env python3
"""Evaluate generated foreground masks against GT masks."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

METHODS = {
    "BBMP": ("bbmp_mask_dir", "M_bbox"),
    "SAM": ("sam_mask_dir", "M_sam"),
    "SAM2": ("sam2_mask_dir", "M_sam2"),
}


def natural_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    tail = stem.rsplit("_", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, stem)


def image_id_from_gt_image(path: Path) -> str:
    return path.stem


def parse_foreground_values(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(v.strip()) for v in value.split(",") if v.strip()}


def read_mask(path: Path, foreground_values: set[int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"cannot_read_file: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if foreground_values is not None:
        binary = np.isin(mask, list(foreground_values)).astype(np.uint8)
    else:
        binary = (mask > 0).astype(np.uint8)
    if binary.ndim != 2 or binary.size == 0:
        raise ValueError(f"invalid_mask: {path}")
    return binary


def read_pred_mask(path: Path, gt_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"cannot_read_file: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 127).astype(np.uint8)
    if binary.shape != gt_shape:
        binary = cv2.resize(binary, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = (binary > 0).astype(np.uint8)
    return binary


def safe_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def safe_dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    denom = (a > 0).sum() + (b > 0).sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter) / denom)


def boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=width)
    return (mask - eroded).astype(np.uint8)


def boundary_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    h, w = gt.shape
    width = max(1, int(round(0.02 * math.sqrt(h * h + w * w))))
    gt_band = boundary_band(gt, width)
    pred_band = boundary_band(pred, width)
    return safe_iou(gt_band, pred_band)


def contour_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    gt_contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, gt_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(out, pred_contours, -1, (0, 0, 255), 2)
    return out


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def make_comparison(image: np.ndarray, gt: np.ndarray, preds: dict[str, np.ndarray]) -> np.ndarray:
    panels = [
        add_label(image, "original"),
        add_label(mask_to_bgr(gt), "GT mask"),
        add_label(mask_to_bgr(preds["BBMP"]), "BBMP mask"),
        add_label(mask_to_bgr(preds["SAM"]), "SAM mask"),
        add_label(mask_to_bgr(preds["SAM2"]), "SAM2 mask"),
    ]
    return np.concatenate(panels, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate foreground mask segmentation metrics.")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--gt_mask_dir", type=Path, required=True)
    parser.add_argument("--bbmp_mask_dir", type=Path, required=True)
    parser.add_argument("--sam_mask_dir", type=Path, required=True)
    parser.add_argument("--sam2_mask_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--mask_ext", default=".png")
    parser.add_argument("--num_visualize", type=int, default=0)
    parser.add_argument("--num_valid", type=int, default=2000)
    parser.add_argument("--gt_foreground_values", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir = args.out_dir / "qualitative"
    gt_values = parse_foreground_values(args.gt_foreground_values)

    image_paths = sorted(args.image_dir.glob(f"*{args.image_ext}"), key=natural_key)
    per_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    valid_ids: list[str] = []

    for image_path in image_paths:
        if len(valid_ids) >= args.num_valid:
            break
        image_id = image_id_from_gt_image(image_path)
        gt_path = args.gt_mask_dir / f"{image_id}_mask{args.mask_ext}"
        pred_paths = {
            method: getattr(args, dir_attr) / f"{image_id}_{suffix}{args.mask_ext}"
            for method, (dir_attr, suffix) in METHODS.items()
        }

        missing = []
        if not gt_path.exists():
            missing.append(("GT", gt_path))
        for method, path in pred_paths.items():
            if not path.exists():
                missing.append((method, path))
        if missing:
            for method, path in missing:
                skipped_rows.append({"image_id": image_id, "method": method, "reason": "missing_file", "path": str(path)})
            continue

        try:
            gt = read_mask(gt_path, gt_values)
            preds = {method: read_pred_mask(path, gt.shape) for method, path in pred_paths.items()}
            if gt.sum() == 0:
                raise ValueError("invalid_gt_mask_empty")
        except Exception as exc:  # noqa: BLE001 - evaluation should continue.
            skipped_rows.append({"image_id": image_id, "method": "ALL", "reason": type(exc).__name__, "path": str(exc)})
            continue

        for method, pred in preds.items():
            per_rows.append(
                {
                    "image_id": image_id,
                    "method": method,
                    "iou": safe_iou(gt, pred),
                    "dice": safe_dice(gt, pred),
                    "biou": boundary_iou(gt, pred),
                }
            )
        valid_ids.append(image_id)

        if len(valid_ids) <= args.num_visualize:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                if image.shape[:2] != gt.shape:
                    image = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
                write_image(qualitative_dir / f"{image_id}_comparison.png", make_comparison(image, gt, preds))
                for method, pred in preds.items():
                    write_image(qualitative_dir / f"{image_id}_overlay_{method.lower()}.png", contour_overlay(image, gt, pred))

    per_path = args.out_dir / "per_image_mask_metrics.csv"
    with per_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "method", "iou", "dice", "biou"])
        writer.writeheader()
        writer.writerows(per_rows)

    skipped_path = args.out_dir / "skipped_samples.csv"
    with skipped_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "method", "reason", "path"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    summary_rows = []
    for method in METHODS:
        rows = [row for row in per_rows if row["method"] == method]
        skipped = [row for row in skipped_rows if row["method"] in (method, "ALL")]
        metrics = {}
        for metric in ["iou", "dice", "biou"]:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            metrics[f"mean_{metric}"] = float(values.mean()) if len(values) else float("nan")
            metrics[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) else float("nan")
        summary_rows.append({"method": method, **metrics, "num_valid": len(rows), "num_skipped": len(skipped)})

    summary_path = args.out_dir / "summary_mask_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method",
            "mean_iou",
            "std_iou",
            "mean_dice",
            "std_dice",
            "mean_biou",
            "std_biou",
            "num_valid",
            "num_skipped",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("| Method | IoU ↑ | Dice ↑ | BIoU ↑ |")
    print("|--------|------:|-------:|-------:|")
    for row in summary_rows:
        print(f"| {row['method']:<6} | {row['mean_iou']:.4f} | {row['mean_dice']:.4f} | {row['mean_biou']:.4f} |")
    print(f"[DONE] valid_image_groups={len(valid_ids)} skipped_records={len(skipped_rows)} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
