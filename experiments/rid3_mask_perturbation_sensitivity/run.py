#!/usr/bin/env python3
"""RID3 mask perturbation sensitivity measurement.

This script measures GT-vs-pSp background fidelity using existing mask variant
folders. It does not create, erode, dilate, or overwrite masks.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WINDOWS_DRIVE_PATTERN = re.compile(r"^([A-Za-z]):[\\/](.*)$")

MASK_SETTINGS = [
    ("original", "Original"),
    ("dilate_3x3", "Dilate 3x3"),
    ("dilate_5x5", "Dilate 5x5"),
    ("erode_3x3", "Erode 3x3"),
    ("erode_5x5", "Erode 5x5"),
]

METRICS = [
    "iou_bg",
    "lpips_bg",
    "rmse_bg",
    "mae_bg",
    "psnr_bg",
    "ssim_bg",
    "overlap_area_ratio",
]


@dataclass(frozen=True)
class Pair:
    image_id: str
    gender: str
    gt_image: Path
    comparison_image: Path


class LPIPSEvaluator:
    def __init__(self, device: str | None = None, net: str = "vgg", min_size: int = 64):
        import lpips

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.min_size = min_size
        self.model = lpips.LPIPS(net=net).eval().to(device)

    def __call__(self, reference: np.ndarray, prediction: np.ndarray) -> float:
        reference = self._ensure_min_size(reference)
        prediction = self._ensure_min_size(prediction)
        ref_tensor = self._to_tensor(reference).to(self.device)
        pred_tensor = self._to_tensor(prediction).to(self.device)
        with torch.no_grad():
            return float(self.model(ref_tensor, pred_tensor).item())

    def _ensure_min_size(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        target_height = max(height, self.min_size)
        target_width = max(width, self.min_size)
        if (height, width) == (target_height, target_width):
            return image

        top = (target_height - height) // 2
        bottom = target_height - height - top
        left = (target_width - width) // 2
        right = target_width - width - left
        fill = image.reshape(-1, image.shape[-1]).mean(axis=0).tolist()
        return cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=fill,
        )

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"LPIPS expects HxWx3 RGB image, got shape={image.shape}")
        image01 = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
        return torch.from_numpy(image01).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def normalize_path_for_platform(path: Path) -> Path:
    if os.name == "nt":
        return path
    path_text = str(path)
    match = WINDOWS_DRIVE_PATTERN.match(path_text)
    if not match:
        return path
    drive, tail = match.groups()
    return Path("/mnt") / drive.lower() / tail.replace("\\", "/")


def load_rgb_uint8(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape {image.shape}: {path}")
    return image.astype(np.uint8, copy=False)


def load_mask_bool(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape {mask.shape}: {path}")
    return mask > 0


def resize_mask_if_needed(mask: np.ndarray, image_shape: tuple[int, int], notes: list[str], label: str) -> np.ndarray:
    if mask.shape == image_shape:
        return mask
    notes.append(f"{label}_mask_resized_from_{mask.shape[1]}x{mask.shape[0]}_to_{image_shape[1]}x{image_shape[0]}")
    resized = cv2.resize(mask.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def resize_image_if_needed(image: np.ndarray, target_shape: tuple[int, int], notes: list[str], label: str) -> np.ndarray:
    if image.shape[:2] == target_shape:
        return image
    notes.append(f"{label}_image_resized_from_{image.shape[1]}x{image.shape[0]}_to_{target_shape[1]}x{target_shape[0]}")
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)


def masked_values(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_ref = reference[mask].astype(np.float32)
    selected_pred = prediction[mask].astype(np.float32)
    if selected_ref.size == 0:
        raise ValueError("Background overlap selects no pixels.")
    return selected_ref, selected_pred


def psnr_from_mse(mse: float, data_range: float = 255.0) -> float:
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def masked_crop_images(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("Background overlap selects no pixels.")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ref_crop = reference[y0:y1, x0:x1].copy()
    pred_crop = prediction[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]

    ref_fill = ref_crop[mask_crop].mean(axis=0)
    pred_fill = pred_crop[mask_crop].mean(axis=0)
    ref_crop[~mask_crop] = ref_fill
    pred_crop[~mask_crop] = pred_fill
    return ref_crop, pred_crop


def ssim_rgb(reference: np.ndarray, prediction: np.ndarray) -> float:
    if min(reference.shape[:2]) < 7:
        return float("nan")
    return float(
        structural_similarity(
            reference.astype(np.float32),
            prediction.astype(np.float32),
            channel_axis=-1,
            data_range=255.0,
        )
    )


def collect_pairs(
    gt_root: Path,
    comparison_root: Path,
    method: str,
    image_ext: str,
    comparison_image_ext: str,
) -> tuple[list[Pair], list[dict[str, object]]]:
    pairs: list[Pair] = []
    skipped: list[dict[str, object]] = []
    for gender in ("female", "male"):
        gt_dir = gt_root / f"gt_{gender}"
        comparison_dir = comparison_root / f"{method}_{gender}"
        if not gt_dir.exists():
            skipped.append(skip_row("", gender, "all", "missing_gt_image_dir", str(gt_dir)))
            continue
        if not comparison_dir.exists():
            skipped.append(skip_row("", gender, "all", f"missing_{method}_image_dir", str(comparison_dir)))
            continue

        for gt_image in sorted(gt_dir.glob(f"*{image_ext}")):
            prefix = gt_image.name.removesuffix(f"_{gender}_gt{image_ext}")
            if prefix == gt_image.name:
                skipped.append(skip_row(gt_image.stem, gender, "all", "unexpected_gt_image_name", str(gt_image)))
                continue
            image_id = f"{prefix}_{gender}"
            comparison_image = comparison_dir / f"{prefix}_{gender}_{method}{comparison_image_ext}"
            if not comparison_image.exists():
                skipped.append(skip_row(image_id, gender, "all", f"missing_{method}_image", str(comparison_image)))
                continue
            pairs.append(Pair(image_id=image_id, gender=gender, gt_image=gt_image, comparison_image=comparison_image))
    return pairs, skipped


def mask_path(mask_root: Path, role: str, gender: str, setting: str, image_id: str, mask_ext: str) -> Path:
    folder = f"{role}_{gender}_mask" if setting == "original" else f"{role}_{gender}_{setting}"
    role_suffix = role
    return mask_root / folder / f"{image_id}_{role_suffix}_mask{mask_ext}"


def skip_row(image_id: str, gender: str, mask_setting: str, reason: str, detail: str = "") -> dict[str, object]:
    return {
        "image_id": image_id,
        "gender": gender,
        "mask_setting": mask_setting,
        "reason": reason,
        "detail": detail,
    }


def evaluate_pair_setting(
    pair: Pair,
    setting: str,
    gt_mask_root: Path,
    comparison_mask_root: Path,
    method: str,
    mask_ext: str,
    lpips_evaluator: LPIPSEvaluator,
) -> dict[str, object]:
    notes: list[str] = []
    gt_image = load_rgb_uint8(pair.gt_image)
    comparison_image = load_rgb_uint8(pair.comparison_image)
    if gt_image.shape[:2] != comparison_image.shape[:2]:
        comparison_image = resize_image_if_needed(comparison_image, gt_image.shape[:2], notes, method)

    gt_mask_file = mask_path(gt_mask_root, "gt", pair.gender, setting, pair.image_id, mask_ext)
    comparison_mask_file = mask_path(comparison_mask_root, method, pair.gender, setting, pair.image_id, mask_ext)
    gt_fg = load_mask_bool(gt_mask_file)
    comparison_fg = load_mask_bool(comparison_mask_file)
    gt_fg = resize_mask_if_needed(gt_fg, gt_image.shape[:2], notes, "gt")
    comparison_fg = resize_mask_if_needed(comparison_fg, gt_image.shape[:2], notes, method)

    gt_bg = ~gt_fg
    comparison_bg = ~comparison_fg
    bg_overlap = gt_bg & comparison_bg
    bg_union = gt_bg | comparison_bg
    overlap_pixels = int(bg_overlap.sum())
    union_pixels = int(bg_union.sum())
    if overlap_pixels == 0:
        raise ValueError("background overlap area is zero")
    if union_pixels == 0:
        raise ValueError("background union area is zero")

    ref_values, pred_values = masked_values(gt_image, comparison_image, bg_overlap)
    diff = ref_values - pred_values
    abs_diff = np.abs(diff)
    sq_diff = diff**2
    mse = float(np.mean(sq_diff))
    ref_crop, pred_crop = masked_crop_images(gt_image, comparison_image, bg_overlap)

    return {
        "image_id": pair.image_id,
        "gender": pair.gender,
        "comparison_method": method,
        "mask_setting": setting,
        "gt_image": str(pair.gt_image),
        "comparison_image": str(pair.comparison_image),
        "gt_mask": str(gt_mask_file),
        "comparison_mask": str(comparison_mask_file),
        "iou_bg": float(overlap_pixels / union_pixels),
        "lpips_bg": lpips_evaluator(ref_crop, pred_crop),
        "rmse_bg": float(math.sqrt(mse)),
        "mae_bg": float(np.mean(abs_diff)),
        "psnr_bg": psnr_from_mse(mse),
        "ssim_bg": ssim_rgb(ref_crop, pred_crop),
        "overlap_area_ratio": float(overlap_pixels / bg_overlap.size),
        "overlap_pixels": overlap_pixels,
        "union_pixels": union_pixels,
        "notes": ";".join(notes),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = float(row[key])
        if math.isfinite(value):
            values.append(value)
    return values


def summarize(per_image_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    by_setting = {
        setting: [row for row in per_image_rows if row["mask_setting"] == setting]
        for setting, _ in MASK_SETTINGS
    }
    original_rows = by_setting["original"]
    original_means = {
        metric: mean(finite_values(original_rows, metric)) if finite_values(original_rows, metric) else float("nan")
        for metric in METRICS
    }

    for setting, label in MASK_SETTINGS:
        rows = by_setting[setting]
        summary: dict[str, object] = {
            "mask_setting": setting,
            "mask_setting_label": label,
            "count": len(rows),
        }
        for metric in METRICS:
            values = finite_values(rows, metric)
            metric_mean = mean(values) if values else float("nan")
            metric_std = pstdev(values) if len(values) > 1 else 0.0
            summary[f"mean_{metric}"] = metric_mean
            summary[f"std_{metric}"] = metric_std
            summary[f"delta_{metric}"] = "" if setting == "original" else metric_mean - original_means[metric]
        summary_rows.append(summary)
    return summary_rows


def format_float(value: object, digits: int = 4) -> str:
    if value == "":
        return "-"
    value_float = float(value)
    if math.isnan(value_float):
        return "nan"
    if math.isinf(value_float):
        return "inf"
    return f"{value_float:.{digits}f}"


def write_latex_table(path: Path, summary_rows: list[dict[str, object]]) -> None:
    headers = [
        "Mask Setting",
        "IoUbg $\\uparrow$",
        "$\\Delta$IoU",
        "LPIPSbg $\\downarrow$",
        "$\\Delta$LPIPS",
        "RMSEbg $\\downarrow$",
        "$\\Delta$RMSE",
        "MAEbg $\\downarrow$",
        "$\\Delta$MAE",
        "PSNRbg $\\uparrow$",
        "$\\Delta$PSNR",
        "SSIMbg $\\uparrow$",
        "$\\Delta$SSIM",
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Mask Robustness Analysis of the Background-Fidelity Evaluation Protocol}",
        "\\label{tab:mask_robustness_bg_fidelity}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrrrrrr}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        values = [
            str(row["mask_setting_label"]).replace("x", "$\\times$"),
            format_float(row["mean_iou_bg"]),
            format_float(row["delta_iou_bg"]),
            format_float(row["mean_lpips_bg"]),
            format_float(row["delta_lpips_bg"]),
            format_float(row["mean_rmse_bg"]),
            format_float(row["delta_rmse_bg"]),
            format_float(row["mean_mae_bg"]),
            format_float(row["delta_mae_bg"]),
            format_float(row["mean_psnr_bg"]),
            format_float(row["delta_psnr_bg"]),
            format_float(row["mean_ssim_bg"]),
            format_float(row["delta_ssim_bg"]),
        ]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_dir", type=Path, required=True, help="Normalized dataset root containing gt_female/gt_male.")
    parser.add_argument("--psp_dir", type=Path, required=True, help="Normalized dataset root containing method_female/method_male.")
    parser.add_argument("--gt_mask_dir", type=Path, required=True, help="Root containing existing GT mask variant folders.")
    parser.add_argument("--psp_mask_dir", type=Path, required=True, help="Root containing existing method mask variant folders.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--method", choices=("psp", "e4e", "pti"), default="psp")
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--comparison_image_ext", default=None)
    parser.add_argument("--mask_ext", default=".png")
    parser.add_argument("--device", default=None)
    parser.add_argument("--lpips_net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--lpips_min_size", type=int, default=64)
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_dir = normalize_path_for_platform(args.gt_dir)
    psp_dir = normalize_path_for_platform(args.psp_dir)
    gt_mask_dir = normalize_path_for_platform(args.gt_mask_dir)
    psp_mask_dir = normalize_path_for_platform(args.psp_mask_dir)
    out_dir = normalize_path_for_platform(args.out_dir)
    image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"
    comparison_image_ext = args.comparison_image_ext or image_ext
    comparison_image_ext = comparison_image_ext if comparison_image_ext.startswith(".") else f".{comparison_image_ext}"
    mask_ext = args.mask_ext if args.mask_ext.startswith(".") else f".{args.mask_ext}"

    pairs, skipped_rows = collect_pairs(gt_dir, psp_dir, args.method, image_ext, comparison_image_ext)
    lpips_evaluator = LPIPSEvaluator(device=args.device, net=args.lpips_net, min_size=args.lpips_min_size)

    per_image_path = out_dir / "per_image_mask_sensitivity.csv"
    summary_path = out_dir / "summary_mask_sensitivity.csv"
    skipped_path = out_dir / "skipped_samples.csv"
    table_path = out_dir / "table_mask_sensitivity.tex"

    if args.no_resume:
        for path in (per_image_path, summary_path, skipped_path, table_path):
            path.unlink(missing_ok=True)

    per_image_fields = [
        "image_id",
        "gender",
        "comparison_method",
        "mask_setting",
        "gt_image",
        "comparison_image",
        "gt_mask",
        "comparison_mask",
        "iou_bg",
        "lpips_bg",
        "rmse_bg",
        "mae_bg",
        "psnr_bg",
        "ssim_bg",
        "overlap_area_ratio",
        "overlap_pixels",
        "union_pixels",
        "notes",
    ]
    skipped_fields = ["image_id", "gender", "mask_setting", "reason", "detail"]

    per_image_rows: list[dict[str, object]] = [] if args.no_resume else read_existing_rows(per_image_path)
    processed = {(str(row["image_id"]), str(row["mask_setting"])) for row in per_image_rows}
    existing_skip_rows = [] if args.no_resume else read_existing_rows(skipped_path)
    skipped_processed = {
        (str(row["image_id"]), str(row["mask_setting"]))
        for row in existing_skip_rows
        if row.get("image_id") and row.get("mask_setting") not in {"", "all"}
    }
    evaluations = [
        (pair, setting)
        for pair in pairs
        for setting, _label in MASK_SETTINGS
        if (pair.image_id, setting) not in processed and (pair.image_id, setting) not in skipped_processed
    ]
    buffered_rows: list[dict[str, object]] = []
    buffered_skips: list[dict[str, object]] = list(skipped_rows)

    def flush() -> None:
        nonlocal buffered_rows, buffered_skips
        append_csv(per_image_path, buffered_rows, per_image_fields)
        append_csv(skipped_path, buffered_skips, skipped_fields)
        per_image_rows.extend(buffered_rows)
        summary_rows = summarize(per_image_rows)
        summary_fields = list(summary_rows[0].keys()) if summary_rows else ["mask_setting"]
        write_csv(summary_path, summary_rows, summary_fields)
        write_latex_table(table_path, summary_rows)
        buffered_rows = []
        buffered_skips = []

    total = len(evaluations)
    with tqdm(total=total, desc="RID3 mask sensitivity", unit="eval") as progress:
        for pair, setting in evaluations:
            try:
                buffered_rows.append(
                    evaluate_pair_setting(
                        pair,
                            setting,
                            gt_mask_dir,
                            psp_mask_dir,
                            args.method,
                            mask_ext,
                            lpips_evaluator,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - log and continue per experiment spec.
                buffered_skips.append(skip_row(pair.image_id, pair.gender, setting, type(exc).__name__, str(exc)))
            progress.update(1)
            if len(buffered_rows) + len(buffered_skips) >= args.flush_every:
                flush()

    flush()

    print(f"pairs: {len(pairs)}")
    print(f"per_image_rows: {len(per_image_rows)}")
    print(f"skipped_rows: {len(read_existing_rows(skipped_path))}")
    print(f"out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
