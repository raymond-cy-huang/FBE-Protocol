#!/usr/bin/env python3
"""Measure pSp background-color face fidelity against GT images."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fbe_protocol.metrics.image_io import load_image

DEFAULT_PRED_ROOT = Path(
    r"D:\Ph.D\01_Experiments_GAN_Inv_Log\2026.03.08_All_Experiments_Start"
    r"\exper_06_psp_background_color_affect_face_fideility\bg_color_psp_inv"
)
DEFAULT_GT_ROOT = Path(
    r"D:\Ph.D\01_Experiments_GAN_Inv_Log\2026.03.08_All_Experiments_Start"
    r"\_normalized_dataset"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\Ph.D\01_Experiments_GAN_Inv_Log\2026.03.08_All_Experiments_Start"
    r"\exper_06_psp_background_color_affect_face_fideility\bg_color_psp_inv_metrics_full"
)

WINDOWS_DRIVE_PATTERN = re.compile(r"^([A-Za-z]):[\\/](.*)$")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

METHODS = [
    ("psp", "pSp", "psp_{gender}"),
    ("blur", "pSp_blur", "blur_{gender}_psp"),
    ("dark", "pSp_dark", "dark_{gender}_psp"),
    ("gray", "pSp_gray", "gray_{gender}_psp"),
    ("mean", "pSp_mean", "mean_{gender}_psp"),
    ("median", "pSp_med", "median_{gender}_psp"),
    ("mode", "pSp_mode", "mode_{gender}_psp"),
    ("random", "pSp_rand", "random_{gender}_psp"),
    ("white", "pSp_white", "white_{gender}_psp"),
    ("switch", "pSp_switch", "switch_{gender}_psp"),
]
GENDERS = ("female", "male")
METRIC_KEYS = ("ssim", "lpips", "psnr", "rmse", "mae", "id_similarity")


@dataclass(frozen=True)
class Pair:
    method: str
    table_method: str
    gender: str
    sample_id: str
    gt_image: Path
    pred_image: Path


class LPIPSEvaluator:
    def __init__(self, net: str = "vgg", device: str | None = None):
        import lpips

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = lpips.LPIPS(net=net).eval().to(device)

    def __call__(self, reference: np.ndarray, prediction: np.ndarray) -> float:
        if reference.shape != prediction.shape:
            raise ValueError(f"Image shape mismatch: {reference.shape} vs {prediction.shape}")
        ref_tensor = self._to_tensor(reference).to(self.device)
        pred_tensor = self._to_tensor(prediction).to(self.device)
        with torch.no_grad():
            return float(self.model(ref_tensor, pred_tensor).item())

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"LPIPS expects HxWx3 RGB image, got shape={image.shape}")
        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


class IDSimilarityEvaluator:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        pad_ratio: float = 0.25,
    ):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        providers = []
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.pad_ratio = float(pad_ratio)
        self.providers = providers
        self.app = FaceAnalysis(name=model_name, providers=providers)
        ctx_id = 0 if providers and providers[0] == "CUDAExecutionProvider" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def __call__(self, reference: np.ndarray, prediction: np.ndarray) -> float:
        ref_embedding = self._embedding(reference)
        pred_embedding = self._embedding(prediction)
        return cosine(ref_embedding, pred_embedding)

    def _embedding(self, image: np.ndarray) -> np.ndarray:
        image = pad_image(image, self.pad_ratio)
        faces = self.app.get(rgb01_to_bgr_u8(image))
        if not faces:
            raise RuntimeError("no face detected")
        face = max(faces, key=lambda item: bbox_area(item.bbox))
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = face.embedding
        return np.asarray(embedding, dtype=np.float32)


def normalize_path_for_platform(path: Path) -> Path:
    if os.name == "nt":
        return path
    path_text = str(path)
    match = WINDOWS_DRIVE_PATTERN.match(path_text)
    if not match:
        return path
    drive, tail = match.groups()
    return Path("/mnt") / drive.lower() / tail.replace("\\", "/")


def sample_id_from_name(path: Path) -> str:
    match = re.match(r"^(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse sample id from {path.name}")
    return match.group(1)


def image_files(directory: Path) -> Iterable[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def index_by_sample_id(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in image_files(directory):
        sample_id = sample_id_from_name(path)
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample id {sample_id} in {directory}")
        indexed[sample_id] = path
    return indexed


def pred_dir(pred_root: Path, method_template: str, gender: str) -> Path:
    return pred_root / method_template.format(gender=gender)


def collect_pairs(pred_root: Path, gt_root: Path, sample_per_group: int = 0) -> list[Pair]:
    pairs: list[Pair] = []
    for method, table_method, folder_template in METHODS:
        for gender in GENDERS:
            gt_dir = gt_root / f"gt_{gender}"
            prediction_dir = pred_dir(pred_root, folder_template, gender)
            gt_by_id = index_by_sample_id(gt_dir)
            pred_by_id = index_by_sample_id(prediction_dir)
            missing_pred = sorted(set(gt_by_id) - set(pred_by_id))
            extra_pred = sorted(set(pred_by_id) - set(gt_by_id))
            if missing_pred or extra_pred:
                raise ValueError(
                    f"{method}/{gender}: missing_pred={len(missing_pred)}, extra_pred={len(extra_pred)}"
                )
            sample_ids = sorted(gt_by_id)
            if sample_per_group > 0:
                sample_ids = sample_ids[:sample_per_group]
            pairs.extend(
                Pair(
                    method=method,
                    table_method=table_method,
                    gender=gender,
                    sample_id=sample_id,
                    gt_image=gt_by_id[sample_id],
                    pred_image=pred_by_id[sample_id],
                )
                for sample_id in sample_ids
            )
    return pairs


def rgb01_to_bgr_u8(image: np.ndarray) -> np.ndarray:
    rgb = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def pad_image(image: np.ndarray, pad_ratio: float) -> np.ndarray:
    pad = int(min(image.shape[:2]) * pad_ratio)
    if pad <= 0:
        return image
    return np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)


def bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))


def psnr(reference: np.ndarray, prediction: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((reference - prediction) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10((data_range * data_range) / mse))


def evaluate_pair(pair: Pair, lpips_evaluator: LPIPSEvaluator, id_evaluator: IDSimilarityEvaluator) -> dict[str, object]:
    reference = load_image(pair.gt_image)
    prediction = load_image(pair.pred_image)
    if reference.shape != prediction.shape:
        raise ValueError(f"Image shape mismatch: {reference.shape} vs {prediction.shape}")

    diff = reference.astype(np.float32) - prediction.astype(np.float32)
    row: dict[str, object] = {
        "method": pair.method,
        "table_method": pair.table_method,
        "gender": pair.gender,
        "sample_id": pair.sample_id,
        "gt_image": pair.gt_image.name,
        "pred_image": pair.pred_image.name,
        "status": "passed",
        "error": "",
        "ssim": float(
            structural_similarity(
                reference.astype(np.float32),
                prediction.astype(np.float32),
                channel_axis=-1,
                data_range=1.0,
            )
        ),
        "lpips": lpips_evaluator(reference, prediction),
        "psnr": psnr(reference, prediction),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
    }
    try:
        row["id_similarity"] = id_evaluator(reference, prediction)
        row["id_status"] = "passed"
        row["id_error"] = ""
    except Exception as exc:
        row["id_similarity"] = ""
        row["id_status"] = "failed"
        row["id_error"] = f"{type(exc).__name__}: {exc}"
    return row


def error_row(pair: Pair, exc: Exception) -> dict[str, object]:
    row: dict[str, object] = {
        "method": pair.method,
        "table_method": pair.table_method,
        "gender": pair.gender,
        "sample_id": pair.sample_id,
        "gt_image": pair.gt_image.name,
        "pred_image": pair.pred_image.name,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "id_status": "failed",
        "id_error": "",
    }
    for key in METRIC_KEYS:
        row[key] = ""
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


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def summarize(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_gender: list[dict[str, object]] = []
    merged: list[dict[str, object]] = []
    for method, table_method, _folder_template in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        for gender in GENDERS:
            gender_rows = [row for row in method_rows if row["gender"] == gender]
            by_gender.append(summary_row(gender_rows, table_method=table_method, gender=gender))
        merged.append(summary_row(method_rows, table_method=table_method, gender=""))
    return by_gender, merged


def summary_row(rows: list[dict[str, object]], table_method: str, gender: str) -> dict[str, object]:
    row: dict[str, object] = {
        "Method": table_method,
        "gender": gender,
        "N": len(rows),
        "success_count": sum(1 for item in rows if item.get("status") == "passed"),
        "failed_count": sum(1 for item in rows if item.get("status") == "failed"),
    }
    for key in ("ssim", "lpips", "psnr", "rmse", "mae", "id_similarity"):
        values = finite_values(rows, key)
        row[f"mean_{key}"] = mean(values) if values else ""
        row[f"std_{key}"] = pstdev(values) if len(values) > 1 else 0.0
        row[f"valid_{key}_count"] = len(values)
    row["failed_id_count"] = sum(1 for item in rows if item.get("id_status") == "failed")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="sample_check")
    parser.add_argument("--sample-per-group", type=int, default=2)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--device", default=None)
    parser.add_argument("--flush-every", type=int, default=100, help="Write result/summary checkpoints every N pairs.")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing result CSV instead of resuming it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_root = normalize_path_for_platform(args.pred_root)
    gt_root = normalize_path_for_platform(args.gt_root)
    output_root = normalize_path_for_platform(args.output_root) / args.run_name
    pairs = collect_pairs(pred_root, gt_root, sample_per_group=args.sample_per_group)
    result_csv = output_root / "bg_color_psp_face_fidelity_result.csv"
    summary_by_gender_csv = output_root / "summary_by_gender.csv"
    summary_merged_csv = output_root / "summary_merged.csv"
    timing_csv = output_root / "timing.csv"

    manifest_fields = ["method", "table_method", "gender", "sample_id", "gt_image", "pred_image"]
    manifest_rows = [
        {
            "method": pair.method,
            "table_method": pair.table_method,
            "gender": pair.gender,
            "sample_id": pair.sample_id,
            "gt_image": str(pair.gt_image),
            "pred_image": str(pair.pred_image),
        }
        for pair in pairs
    ]
    write_csv(output_root / "manifest.csv", manifest_rows, manifest_fields)

    start_time = time.perf_counter()
    lpips_start = time.perf_counter()
    lpips_evaluator = LPIPSEvaluator(net=args.lpips_net, device=args.device)
    lpips_init_seconds = time.perf_counter() - lpips_start
    id_start = time.perf_counter()
    id_evaluator = IDSimilarityEvaluator()
    id_init_seconds = time.perf_counter() - id_start

    fieldnames = [
        "method",
        "table_method",
        "gender",
        "sample_id",
        "gt_image",
        "pred_image",
        "status",
        "error",
        "ssim",
        "lpips",
        "psnr",
        "rmse",
        "mae",
        "id_similarity",
        "id_status",
        "id_error",
    ]

    if args.no_resume and result_csv.exists():
        result_csv.unlink()

    pair_rows: list[dict[str, object]] = [] if args.no_resume else read_csv_rows(result_csv)
    processed = {
        (str(row.get("method", "")), str(row.get("gender", "")), str(row.get("sample_id", "")))
        for row in pair_rows
        if row.get("method") and row.get("gender") and row.get("sample_id")
    }
    pending_pairs = [
        pair
        for pair in pairs
        if (pair.method, pair.gender, pair.sample_id) not in processed
    ]
    buffered_rows: list[dict[str, object]] = []

    def write_checkpoints() -> None:
        by_gender, merged = summarize(pair_rows)
        write_csv(summary_by_gender_csv, by_gender, list(by_gender[0].keys()))
        write_csv(summary_merged_csv, merged, list(merged[0].keys()))
        elapsed_seconds = time.perf_counter() - start_time
        completed = len(pair_rows)
        timing_rows = [
            {
                "pairs_total": len(pairs),
                "pairs_completed": completed,
                "pairs_pending": len(pairs) - completed,
                "sample_per_group": args.sample_per_group,
                "elapsed_seconds": elapsed_seconds,
                "lpips_init_seconds": lpips_init_seconds,
                "id_init_seconds": id_init_seconds,
                "seconds_per_completed_pair_including_init": elapsed_seconds / completed if completed else "",
                "estimated_seconds_for_total_including_init": elapsed_seconds / completed * len(pairs) if completed else "",
                "estimated_remaining_seconds": (elapsed_seconds / completed) * (len(pairs) - completed) if completed else "",
            }
        ]
        write_csv(timing_csv, timing_rows, list(timing_rows[0].keys()))

    def flush_buffer() -> None:
        nonlocal buffered_rows
        append_csv(result_csv, buffered_rows, fieldnames)
        pair_rows.extend(buffered_rows)
        buffered_rows = []
        write_checkpoints()

    with tqdm(
        total=len(pairs),
        initial=len(processed),
        desc="BG-color pSp fidelity",
        unit="pair",
    ) as progress:
        for pair in pending_pairs:
            progress.set_postfix(
                method=pair.method,
                gender=pair.gender,
                sample=pair.sample_id,
                refresh=True,
            )
            try:
                row = evaluate_pair(pair, lpips_evaluator, id_evaluator)
            except Exception as exc:
                row = error_row(pair, exc)
            buffered_rows.append(row)
            if len(buffered_rows) >= max(args.flush_every, 1):
                flush_buffer()
            progress.update(1)

    elapsed_seconds = time.perf_counter() - start_time
    flush_buffer()
    timing_rows = [
        {
            "pairs_total": len(pairs),
            "pairs_completed": len(pair_rows),
            "pairs_pending": len(pairs) - len(pair_rows),
            "sample_per_group": args.sample_per_group,
            "elapsed_seconds": elapsed_seconds,
            "lpips_init_seconds": lpips_init_seconds,
            "id_init_seconds": id_init_seconds,
            "seconds_per_completed_pair_including_init": elapsed_seconds / len(pair_rows) if pair_rows else "",
            "estimated_seconds_for_total_including_init": elapsed_seconds / len(pair_rows) * len(pairs) if pair_rows else "",
            "estimated_remaining_seconds": 0,
        }
    ]
    write_csv(timing_csv, timing_rows, list(timing_rows[0].keys()))

    print(f"pred_root: {pred_root}")
    print(f"gt_root: {gt_root}")
    print(f"output_root: {output_root}")
    print(f"pairs_total: {len(pairs)}")
    print(f"resumed_rows: {len(processed)}")
    print(f"completed_rows: {len(pair_rows)}")
    print(f"elapsed_seconds: {elapsed_seconds:.3f}")
    if pair_rows:
        print(f"seconds_per_completed_pair_including_init: {elapsed_seconds / len(pair_rows):.3f}")
        print(f"estimated_hours_for_total_including_init: {elapsed_seconds / len(pair_rows) * len(pairs) / 3600:.2f}")
    print(f"lpips_init_seconds: {lpips_init_seconds:.3f}")
    print(f"id_init_seconds: {id_init_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
