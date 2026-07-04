#!/usr/bin/env python3
"""Generate erode/dilate variants for normalized GT mask folders."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_DATASET_ROOT = (
    r"D:\Ph.D\01_Experiments_GAN_Inv_Log\2026.03.08_All_Experiments_Start\_normalized_dataset"
)
DEFAULT_GROUPS = (
    ("gt_male", "gt_male_mask"),
    ("gt_female", "gt_female_mask"),
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize_path(path: str) -> Path:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if os.name != "nt" and match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3x3/5x5 ellipse erode and dilate masks for normalized GT masks."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=None, help="Default: dataset root.")
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args()


def iter_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {path}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Failed to write mask: {path}")


def generate_group(
    input_dir: Path,
    output_root: Path,
    output_prefix: str,
    kernel_sizes: list[int],
    iterations: int,
) -> dict[str, int]:
    images = iter_images(input_dir)
    if not images:
        raise ValueError(f"No mask images found in: {input_dir}")

    outputs = {
        f"{output_prefix}_{operation}_{size}x{size}": output_root / f"{output_prefix}_{operation}_{size}x{size}"
        for size in kernel_sizes
        for operation in ("erode", "dilate")
    }
    for output_dir in outputs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    iterator = images
    if tqdm is not None:
        iterator = tqdm(images, desc=output_prefix, unit="mask")

    for image_path in iterator:
        source = binary_mask(image_path)
        for size in kernel_sizes:
            if size < 1:
                raise ValueError(f"Kernel size must be positive: {size}")
            kernel_size = size if size % 2 == 1 else size + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            eroded = cv2.erode(source, kernel, iterations=iterations)
            dilated = cv2.dilate(source, kernel, iterations=iterations)

            write_mask(outputs[f"{output_prefix}_erode_{size}x{size}"] / image_path.name, eroded)
            write_mask(outputs[f"{output_prefix}_dilate_{size}x{size}"] / image_path.name, dilated)

    return {name: len(iter_images(path)) for name, path in outputs.items()}


def main() -> None:
    args = parse_args()
    dataset_root = normalize_path(args.dataset_root)
    output_root = normalize_path(args.output_root) if args.output_root else dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Output root : {output_root}")
    print("[INFO] Kernel      : ellipse")
    print(f"[INFO] Sizes       : {', '.join(f'{size}x{size}' for size in args.kernel_sizes)}")
    print(f"[INFO] Iterations  : {args.iterations}")

    all_counts: dict[str, int] = {}
    for output_prefix, input_name in DEFAULT_GROUPS:
        input_dir = dataset_root / input_name
        if not input_dir.exists():
            raise FileNotFoundError(f"Input mask folder not found: {input_dir}")
        all_counts.update(
            generate_group(
                input_dir=input_dir,
                output_root=output_root,
                output_prefix=output_prefix,
                kernel_sizes=args.kernel_sizes,
                iterations=args.iterations,
            )
        )

    print("[INFO] Output counts:")
    for name in sorted(all_counts):
        print(f"  {name}: {all_counts[name]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
