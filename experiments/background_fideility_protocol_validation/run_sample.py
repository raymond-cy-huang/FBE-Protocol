#!/usr/bin/env python3
"""Run Section 4.2 background fidelity protocol validation."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fbe_protocol.metrics.image_io import load_image, load_mask

DEFAULT_INPUT_ROOT = Path(
    r"D:\Ph.D\01_Experiments_GAN_Inv_Log\2026.03.08_All_Experiments_Start"
    r"\exper_05_psp_face_frontal_analysis\eval_bg_iou"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\Ph.D\01_Experiments_GAN_Inv_Log\2026.03.08_All_Experiments_Start"
    r"\FBE-Protocol_final_output\4.2_Background_Fidelity_Protocol_Validation"
)

WINDOWS_DRIVE_PATTERN = re.compile(r"^([A-Za-z]):[\\/](.*)$")

GROUPS = {
    "female": {
        "label": "psp_frontal_female",
        "gt_dir": "gt_female",
        "gt_mask_dir": "gt_female_mask",
        "pred_dir": "psp_frontal_female",
        "pred_mask_dir": "psp_frontal_female_mask",
    },
    "male": {
        "label": "psp_frontal_male",
        "gt_dir": "gt_male",
        "gt_mask_dir": "gt_male_mask",
        "pred_dir": "psp_frontal_male",
        "pred_mask_dir": "psp_frontal_male_mask",
    },
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
METRIC_KEYS = [
    "bg_iou",
    "valid_bg_pixels",
    "valid_bg_ratio",
    "union_bg_ratio",
    "bg_psnr_intersection",
    "bg_rmse_intersection",
    "bg_mae_intersection",
    "bg_ssim_intersection",
    "bg_lpips_intersection",
]


@dataclass(frozen=True)
class Pair:
    group: str
    sample_id: str
    gt_img: Path
    gt_mask: Path
    pred_img: Path
    pred_mask: Path


class LPIPSEvaluator:
    def __init__(self, net: str = "vgg", device: str | None = None, min_size: int = 64):
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


def find_one(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def collect_pairs(input_root: Path, group_key: str) -> list[Pair]:
    spec = GROUPS[group_key]
    gt_dir = input_root / spec["gt_dir"]
    gt_mask_dir = input_root / spec["gt_mask_dir"]
    pred_dir = input_root / spec["pred_dir"]
    pred_mask_dir = input_root / spec["pred_mask_dir"]

    pairs: list[Pair] = []
    for gt_img in sorted(path for path in gt_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS):
        sample_id = gt_img.name.split("_", 1)[0]
        gt_mask = find_one(gt_mask_dir, f"{sample_id}_{group_key}_gt_mask")
        pred_img = find_one(pred_dir, f"{sample_id}_{group_key}_psp_frontal")
        pred_mask = find_one(pred_mask_dir, f"{sample_id}_{group_key}_psp_frontal_mask")
        if gt_mask and pred_img and pred_mask:
            pairs.append(
                Pair(
                    group=str(spec["label"]),
                    sample_id=sample_id,
                    gt_img=gt_img,
                    gt_mask=gt_mask,
                    pred_img=pred_img,
                    pred_mask=pred_mask,
                )
            )
    return pairs


def sample_pairs(input_root: Path, sample_size: int, seed: int) -> list[Pair]:
    pairs = collect_pairs(input_root, "female") + collect_pairs(input_root, "male")
    if sample_size <= 0:
        return sorted(pairs, key=lambda pair: (pair.group, pair.sample_id))
    if len(pairs) < sample_size:
        raise ValueError(f"Found only {len(pairs)} matched pairs; need {sample_size}.")
    rng = random.Random(seed)
    return sorted(rng.sample(pairs, sample_size), key=lambda pair: (pair.group, pair.sample_id))


def masked_values(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != prediction.shape:
        raise ValueError(f"Image shape mismatch: {reference.shape} vs {prediction.shape}")
    if mask.shape[:2] != reference.shape[:2]:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {reference.shape[:2]}")
    selected_ref = reference[mask]
    selected_pred = prediction[mask]
    if selected_ref.size == 0:
        raise ValueError("Background intersection selects no pixels.")
    return selected_ref, selected_pred


def psnr_from_mse(mse: float, data_range: float = 255.0) -> float:
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def masked_crop_images(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("Background intersection selects no pixels.")
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


def skimage_ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
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


def normalize_path_for_platform(path: Path) -> Path:
    if os.name == "nt":
        return path
    path_text = str(path)
    match = WINDOWS_DRIVE_PATTERN.match(path_text)
    if not match:
        return path
    drive, tail = match.groups()
    return Path("/mnt") / drive.lower() / tail.replace("\\", "/")


def evaluate_pair(pair: Pair, lpips_evaluator: LPIPSEvaluator) -> dict[str, object]:
    gt = load_image(pair.gt_img) * 255.0
    pred = load_image(pair.pred_img) * 255.0
    gt_fg = load_mask(pair.gt_mask).astype(bool)
    pred_fg = load_mask(pair.pred_mask).astype(bool)

    if gt.shape != pred.shape:
        raise ValueError(f"{pair.sample_id}: image shape mismatch {gt.shape} vs {pred.shape}")
    if gt_fg.shape != gt.shape[:2] or pred_fg.shape != gt.shape[:2]:
        raise ValueError(f"{pair.sample_id}: mask shape mismatch")

    gt_bg = ~gt_fg
    pred_bg = ~pred_fg
    bg_intersection = gt_bg & pred_bg
    bg_union = gt_bg | pred_bg

    ref_values, pred_values = masked_values(gt, pred, bg_intersection)
    abs_diff = np.abs(ref_values - pred_values)
    sq_diff = (ref_values - pred_values) ** 2
    mse = float(np.mean(sq_diff))
    bg_ref_crop, bg_pred_crop = masked_crop_images(gt, pred, bg_intersection)

    return {
        "image_name": pair.gt_img.name,
        "bg_iou": float(bg_intersection.sum() / bg_union.sum()),
        "valid_bg_pixels": int(bg_intersection.sum()),
        "valid_bg_ratio": float(bg_intersection.mean()),
        "union_bg_ratio": float(bg_union.mean()),
        "bg_psnr_intersection": psnr_from_mse(mse),
        "bg_rmse_intersection": float(math.sqrt(mse)),
        "bg_mae_intersection": float(np.mean(abs_diff)),
        "bg_ssim_intersection": skimage_ssim(bg_ref_crop, bg_pred_crop),
        "bg_lpips_intersection": lpips_evaluator(bg_ref_crop, bg_pred_crop),
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
    values = []
    for row in rows:
        if row.get("status") == "failed" or row.get(key, "") == "":
            continue
        value = float(row[key])
        if math.isfinite(value):
            values.append(value)
    return values


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    failed_count = sum(1 for row in rows if row.get("status") == "failed")
    success_count = len(rows) - failed_count
    summary: dict[str, object] = {
        "split": "overall",
        "count": len(rows),
        "success_count": success_count,
        "failed_count": failed_count,
    }
    for metric in METRIC_KEYS:
        values = finite_values(rows, metric)
        summary[f"mean_{metric}"] = mean(values) if values else ""
        summary[f"std_{metric}"] = pstdev(values) if len(values) > 1 else 0.0
    return [summary]


def error_row(pair: Pair, exc: Exception) -> dict[str, object]:
    row: dict[str, object] = {
        "image_name": pair.gt_img.name,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }
    for metric in METRIC_KEYS:
        row[metric] = ""
    return row


def format_summary_value(row: dict[str, object], key: str) -> str:
    value = row.get(key, "")
    if value == "":
        return "nan"
    return f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-size", type=int, default=10, help="Total sampled images. Use 0 for all matched pairs.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--lpips-min-size", type=int, default=64, help="Minimum H/W passed to LPIPS after padding.")
    parser.add_argument("--device", default=None, help="LPIPS device. Defaults to cuda if available, otherwise cpu.")
    parser.add_argument("--flush-every", type=int, default=100, help="Write incremental CSV checkpoints every N images.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing result CSV and recompute the run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = normalize_path_for_platform(args.input_root)
    output_root = normalize_path_for_platform(args.output_root)
    run_name = "full" if args.sample_size <= 0 else f"sample_{args.sample_size}_seed{args.seed}"
    sample_dir = output_root / run_name
    result_csv = sample_dir / "bg_fideility_result.csv"
    summary_csv = sample_dir / "bg_fideility_summarization.csv"
    lpips_evaluator = LPIPSEvaluator(net=args.lpips_net, device=args.device, min_size=args.lpips_min_size)
    pairs = sample_pairs(input_root, args.sample_size, args.seed)
    pair_fields = [
        "image_name",
        "status",
        "error",
        "bg_iou",
        "valid_bg_pixels",
        "valid_bg_ratio",
        "union_bg_ratio",
        "bg_psnr_intersection",
        "bg_rmse_intersection",
        "bg_mae_intersection",
        "bg_ssim_intersection",
        "bg_lpips_intersection",
    ]

    pair_rows = [] if args.no_resume else read_existing_rows(result_csv)
    processed = {str(row["image_name"]) for row in pair_rows if row.get("image_name")}
    pending_pairs = [pair for pair in pairs if pair.gt_img.name not in processed]
    buffered_rows: list[dict[str, object]] = []

    def flush_buffer() -> None:
        nonlocal buffered_rows
        append_csv(result_csv, buffered_rows, pair_fields)
        pair_rows.extend(buffered_rows)
        buffered_rows = []
        write_csv(summary_csv, summarize(pair_rows), list(summarize(pair_rows)[0].keys()))

    if args.no_resume and result_csv.exists():
        result_csv.unlink()
    if args.no_resume and summary_csv.exists():
        summary_csv.unlink()

    with tqdm(total=len(pending_pairs), desc="Background fidelity", unit="image") as progress:
        for offset, pair in enumerate(pending_pairs, start=1):
            completed = len(processed) + offset
            remaining = len(pairs) - completed
            progress.set_postfix(
                current=f"{completed}/{len(pairs)}",
                remaining=remaining,
                image=pair.gt_img.name,
                refresh=True,
            )
            try:
                row = evaluate_pair(pair, lpips_evaluator)
                row["status"] = "passed"
                row["error"] = ""
            except Exception as exc:
                tqdm.write(f"[WARN] skipped {pair.gt_img.name}: {type(exc).__name__}: {exc}")
                row = error_row(pair, exc)
            buffered_rows.append(row)
            if len(buffered_rows) >= max(args.flush_every, 1):
                flush_buffer()
            progress.update(1)
    flush_buffer()
    summary_rows = summarize(pair_rows)
    summary_fields = list(summary_rows[0].keys()) if summary_rows else ["split", "count"]
    write_csv(summary_csv, summary_rows, summary_fields)

    print(f"input_root: {input_root}")
    print(f"output_root: {sample_dir}")
    print(f"sample_size_total: {args.sample_size}")
    print(f"total_pairs: {len(pairs)}")
    print(f"resumed_rows: {len(processed)}")
    print(f"lpips_net: {args.lpips_net}")
    print(f"lpips_min_size: {args.lpips_min_size}")
    for row in summary_rows:
        print(
            f"{row['split']}: n={row['count']}, "
            f"success={row['success_count']}, failed={row['failed_count']}, "
            f"bg_iou={format_summary_value(row, 'mean_bg_iou')}, "
            f"bg_psnr={format_summary_value(row, 'mean_bg_psnr_intersection')}, "
            f"bg_rmse={format_summary_value(row, 'mean_bg_rmse_intersection')}, "
            f"bg_mae={format_summary_value(row, 'mean_bg_mae_intersection')}, "
            f"bg_ssim={format_summary_value(row, 'mean_bg_ssim_intersection')}, "
            f"bg_lpips={format_summary_value(row, 'mean_bg_lpips_intersection')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
