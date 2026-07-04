#!/usr/bin/env python3
"""Run Section 4.5 semantic-editing background fidelity measurement."""

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

from fbe_protocol.metrics.image_io import load_image, load_mask

DEFAULT_INPUT_ROOT = Path(
    r"D:\Ph.D\01_Experiments_GAN_Inv_Log\2026.03.08_All_Experiments_Start"
    r"\FBE-Protocol_final_output\4.5_Semantic_Editing_Validation"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT

WINDOWS_DRIVE_PATTERN = re.compile(r"^([A-Za-z]):[\\/](.*)$")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
REFERENCE_EDIT_IDX = "002"

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

DATASETS = {
    "sefa": ("age", "gender", "smile"),
    "interface_gan": ("age", "gender", "smile"),
}


@dataclass(frozen=True)
class Pair:
    method: str
    attribute: str
    sample_id: str
    edit_idx: str
    reference_img: Path
    reference_mask: Path
    edited_img: Path
    edited_mask: Path


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


def normalize_path_for_platform(path: Path) -> Path:
    if os.name == "nt":
        return path
    path_text = str(path)
    match = WINDOWS_DRIVE_PATTERN.match(path_text)
    if not match:
        return path
    drive, tail = match.groups()
    return Path("/mnt") / drive.lower() / tail.replace("\\", "/")


def parse_edit_stem(stem: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(\d{3})_(\d{3})", stem)
    if not match:
        return None
    return match.group(1), match.group(2)


def collect_pairs(input_root: Path, include_reference_step: bool = False) -> tuple[list[Pair], list[dict[str, object]]]:
    pairs: list[Pair] = []
    inventory_rows: list[dict[str, object]] = []

    for method, attributes in DATASETS.items():
        method_root = input_root / method
        for attribute in attributes:
            image_dir = method_root / attribute
            mask_dir = method_root / f"{attribute}_mask"
            if not image_dir.exists():
                inventory_rows.append(inventory_row(method, attribute, "missing_image_dir", image_dir, None, None))
                continue
            if not mask_dir.exists():
                inventory_rows.append(inventory_row(method, attribute, "missing_mask_dir", mask_dir, None, None))
                continue

            edited_images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
            by_sample: dict[str, dict[str, Path]] = {}
            for image_path in edited_images:
                parsed = parse_edit_stem(image_path.stem)
                if parsed is None:
                    continue
                sample_id, edit_idx = parsed
                by_sample.setdefault(sample_id, {})[edit_idx] = image_path

            for sample_id, edits in sorted(by_sample.items()):
                ref_img = edits.get(REFERENCE_EDIT_IDX)
                ref_mask = mask_dir / f"{sample_id}_{REFERENCE_EDIT_IDX}_mask.png"
                for edit_idx, edited_img in sorted(edits.items()):
                    if edit_idx == REFERENCE_EDIT_IDX and not include_reference_step:
                        continue
                    edited_mask = mask_dir / f"{sample_id}_{edit_idx}_mask.png"
                    status = "paired"
                    if ref_img is None:
                        status = "missing_reference_image"
                    elif not ref_mask.exists():
                        status = "missing_reference_mask"
                    elif not edited_mask.exists():
                        status = "missing_edited_mask"

                    inventory_rows.append(
                        inventory_row(
                            method,
                            attribute,
                            status,
                            edited_img,
                            ref_img,
                            edited_mask,
                            sample_id=sample_id,
                            edit_idx=edit_idx,
                            reference_mask=ref_mask,
                        )
                    )
                    if status == "paired" and ref_img is not None:
                        pairs.append(
                            Pair(
                                method=method,
                                attribute=attribute,
                                sample_id=sample_id,
                                edit_idx=edit_idx,
                                reference_img=ref_img,
                                reference_mask=ref_mask,
                                edited_img=edited_img,
                                edited_mask=edited_mask,
                            )
                        )
    return pairs, inventory_rows


def inventory_row(
    method: str,
    attribute: str,
    status: str,
    edited_image: Path,
    reference_image: Path | None,
    edited_mask: Path | None,
    *,
    sample_id: str = "",
    edit_idx: str = "",
    reference_mask: Path | None = None,
) -> dict[str, object]:
    return {
        "method": method,
        "attribute": attribute,
        "sample_id": sample_id,
        "edit_idx": edit_idx,
        "status": status,
        "reference_image": "" if reference_image is None else str(reference_image),
        "edited_image": str(edited_image),
        "reference_mask": "" if reference_mask is None else str(reference_mask),
        "edited_mask": "" if edited_mask is None else str(edited_mask),
    }


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


def evaluate_pair(pair: Pair, lpips_evaluator: LPIPSEvaluator) -> dict[str, object]:
    reference = load_image(pair.reference_img) * 255.0
    prediction = load_image(pair.edited_img) * 255.0
    ref_fg = load_mask(pair.reference_mask).astype(bool)
    pred_fg = load_mask(pair.edited_mask).astype(bool)

    if reference.shape != prediction.shape:
        raise ValueError(f"{pair.sample_id}_{pair.edit_idx}: image shape mismatch {reference.shape} vs {prediction.shape}")
    if ref_fg.shape != reference.shape[:2] or pred_fg.shape != reference.shape[:2]:
        raise ValueError(f"{pair.sample_id}_{pair.edit_idx}: mask shape mismatch")

    ref_bg = ~ref_fg
    pred_bg = ~pred_fg
    bg_intersection = ref_bg & pred_bg
    bg_union = ref_bg | pred_bg

    ref_values, pred_values = masked_values(reference, prediction, bg_intersection)
    abs_diff = np.abs(ref_values - pred_values)
    sq_diff = (ref_values - pred_values) ** 2
    mse = float(np.mean(sq_diff))
    bg_ref_crop, bg_pred_crop = masked_crop_images(reference, prediction, bg_intersection)

    return {
        "method": pair.method,
        "attribute": pair.attribute,
        "sample_id": pair.sample_id,
        "edit_idx": pair.edit_idx,
        "reference_image_name": pair.reference_img.name,
        "edited_image_name": pair.edited_img.name,
        "reference_mask_name": pair.reference_mask.name,
        "edited_mask_name": pair.edited_mask.name,
        "comparison": "Xpp_vs_Xp",
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


def error_row(pair: Pair, exc: Exception) -> dict[str, object]:
    row: dict[str, object] = {
        "method": pair.method,
        "attribute": pair.attribute,
        "sample_id": pair.sample_id,
        "edit_idx": pair.edit_idx,
        "reference_image_name": pair.reference_img.name,
        "edited_image_name": pair.edited_img.name,
        "reference_mask_name": pair.reference_mask.name,
        "edited_mask_name": pair.edited_mask.name,
        "comparison": "Xpp_vs_Xp",
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }
    for metric in METRIC_KEYS:
        row[metric] = ""
    return row


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


def summarize_group(rows: list[dict[str, object]], split: str, method: str = "", attribute: str = "", edit_idx: str = "") -> dict[str, object]:
    failed_count = sum(1 for row in rows if row.get("status") == "failed")
    summary: dict[str, object] = {
        "split": split,
        "method": method,
        "attribute": attribute,
        "edit_idx": edit_idx,
        "count": len(rows),
        "success_count": len(rows) - failed_count,
        "failed_count": failed_count,
    }
    for metric in METRIC_KEYS:
        values = finite_values(rows, metric)
        summary[f"mean_{metric}"] = mean(values) if values else ""
        summary[f"std_{metric}"] = pstdev(values) if len(values) > 1 else 0.0
    return summary


def summarize(rows: list[dict[str, object]], inventory_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    all_rows = list(rows)
    inventory_failures = []
    for inv in inventory_rows:
        if inv["status"] in {"missing_reference_image", "missing_reference_mask", "missing_edited_mask", "missing_mask_dir"}:
            inventory_failures.append(
                {
                    "method": inv["method"],
                    "attribute": inv["attribute"],
                    "sample_id": inv["sample_id"],
                    "edit_idx": inv["edit_idx"],
                    "status": "failed",
                }
            )
    all_rows.extend(inventory_failures)

    summaries: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in all_rows}):
        method_rows = [row for row in all_rows if row["method"] == method]
        summaries.append(summarize_group(method_rows, "method", method=method))
        for attribute in sorted({str(row["attribute"]) for row in method_rows}):
            attr_rows = [row for row in method_rows if row["attribute"] == attribute]
            summaries.append(summarize_group(attr_rows, "method_attribute", method=method, attribute=attribute))
            for edit_idx in sorted({str(row.get("edit_idx", "")) for row in attr_rows if row.get("edit_idx", "")}):
                edit_rows = [row for row in attr_rows if row.get("edit_idx") == edit_idx]
                summaries.append(
                    summarize_group(edit_rows, "method_attribute_edit_idx", method=method, attribute=attribute, edit_idx=edit_idx)
                )
    summaries.append(summarize_group(all_rows, "overall"))
    return summaries


def make_pair_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (str(row["method"]), str(row["attribute"]), str(row["sample_id"]), str(row["edit_idx"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="bg_fidelity_measurement")
    parser.add_argument("--include-reference-step", action="store_true", help="Also evaluate *_002 against itself.")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--lpips-min-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = normalize_path_for_platform(args.input_root)
    output_root = normalize_path_for_platform(args.output_root) / args.run_name
    result_csv = output_root / "semantic_editing_bg_fidelity_result.csv"
    summary_csv = output_root / "semantic_editing_bg_fidelity_summarization.csv"
    inventory_csv = output_root / "semantic_editing_bg_fidelity_inventory.csv"

    pairs, inventory_rows = collect_pairs(input_root, include_reference_step=args.include_reference_step)
    inventory_fields = [
        "method",
        "attribute",
        "sample_id",
        "edit_idx",
        "status",
        "reference_image",
        "edited_image",
        "reference_mask",
        "edited_mask",
    ]
    write_csv(inventory_csv, inventory_rows, inventory_fields)

    if args.inventory_only:
        paired_count = sum(1 for row in inventory_rows if row["status"] == "paired")
        print(f"input_root: {input_root}")
        print(f"output_root: {output_root}")
        print(f"inventory_rows: {len(inventory_rows)}")
        print(f"paired_rows: {paired_count}")
        print(f"inventory_csv: {inventory_csv}")
        return 0

    if args.no_resume:
        for path in (result_csv, summary_csv):
            if path.exists():
                path.unlink()

    pair_fields = [
        "method",
        "attribute",
        "sample_id",
        "edit_idx",
        "reference_image_name",
        "edited_image_name",
        "reference_mask_name",
        "edited_mask_name",
        "comparison",
        "status",
        "error",
        *METRIC_KEYS,
    ]

    pair_rows = [] if args.no_resume else read_existing_rows(result_csv)
    processed = {make_pair_key(row) for row in pair_rows if row.get("method") and row.get("attribute")}
    pending_pairs = [pair for pair in pairs if (pair.method, pair.attribute, pair.sample_id, pair.edit_idx) not in processed]
    buffered_rows: list[dict[str, object]] = []
    lpips_evaluator = LPIPSEvaluator(net=args.lpips_net, device=args.device, min_size=args.lpips_min_size)

    def flush_buffer() -> None:
        nonlocal buffered_rows
        append_csv(result_csv, buffered_rows, pair_fields)
        pair_rows.extend(buffered_rows)
        buffered_rows = []
        summary_rows = summarize(pair_rows, inventory_rows)
        write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()))

    with tqdm(total=len(pending_pairs), desc="Semantic-editing BG fidelity", unit="pair") as progress:
        for offset, pair in enumerate(pending_pairs, start=1):
            completed = len(processed) + offset
            progress.set_postfix(
                current=f"{completed}/{len(pairs)}",
                pair=f"{pair.method}/{pair.attribute}/{pair.sample_id}_{pair.edit_idx}",
                refresh=True,
            )
            try:
                row = evaluate_pair(pair, lpips_evaluator)
                row["status"] = "passed"
                row["error"] = ""
            except Exception as exc:
                tqdm.write(
                    f"[WARN] skipped {pair.method}/{pair.attribute}/{pair.sample_id}_{pair.edit_idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                row = error_row(pair, exc)
            buffered_rows.append(row)
            if len(buffered_rows) >= max(args.flush_every, 1):
                flush_buffer()
            progress.update(1)
    flush_buffer()

    summary_rows = summarize(pair_rows, inventory_rows)
    write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()))

    print(f"input_root: {input_root}")
    print(f"output_root: {output_root}")
    print(f"total_pairs: {len(pairs)}")
    print(f"resumed_rows: {len(processed)}")
    print(f"pending_pairs: {len(pending_pairs)}")
    print(f"result_csv: {result_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"inventory_csv: {inventory_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
