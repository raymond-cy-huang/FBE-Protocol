#!/usr/bin/env python3
"""Build backup CelebAMask-HQ image/mask pairs from the original archive."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_NAME = "gt_celeba-m-hq-img_{idx}.jpg"
MASK_NAME = "gt_celeba-m-hq-img_{idx}_mask.png"


def find_part_masks(mask_root: Path, idx: int) -> list[Path]:
    prefix = f"{idx:05d}_"
    shard_dir = mask_root / str(idx // 2000)
    if shard_dir.is_dir():
        return sorted(shard_dir.glob(f"{prefix}*.png"))
    return sorted(mask_root.rglob(f"{prefix}*.png"))


def merge_part_masks(paths: list[Path]) -> np.ndarray:
    merged = None
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {path}")
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        merged = binary if merged is None else cv2.bitwise_or(merged, binary)
    if merged is None:
        raise ValueError("No part masks to merge")
    return merged


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build backup_100 image/mask pairs.")
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/c/Users/User/Downloads/archive (1)/CelebAMask-HQ"))
    parser.add_argument("--combined-root", type=Path, default=Path("/mnt/d/Dataset/CelebAMask-combined"))
    parser.add_argument("--count", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_image_dir = args.source_root / "CelebA-HQ-img"
    src_mask_dir = args.source_root / "CelebAMask-HQ-mask-anno"
    dst_root = args.combined_root / "backup_100"
    dst_image_dir = dst_root / "gt_image"
    dst_mask_dir = dst_root / "gt_mask"
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    existing = []
    for path in (args.combined_root / "gt_image").glob("gt_celeba-m-hq-img_*.jpg"):
        try:
            existing.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    if not existing:
        raise ValueError(f"No existing gt_image files found: {args.combined_root / 'gt_image'}")

    start = max(existing) + 1
    end = start + args.count
    rows = []
    for idx in range(start, end):
        src_image = src_image_dir / f"{idx}.jpg"
        if not src_image.exists():
            raise FileNotFoundError(f"Missing source image: {src_image}")
        part_masks = find_part_masks(src_mask_dir, idx)
        if not part_masks:
            raise FileNotFoundError(f"Missing source part masks for index {idx}")

        dst_image = dst_image_dir / IMAGE_NAME.format(idx=idx)
        dst_mask = dst_mask_dir / MASK_NAME.format(idx=idx)
        shutil.copy2(src_image, dst_image)
        merged = merge_part_masks(part_masks)
        write_image(dst_mask, merged)
        rows.append((idx, src_image, len(part_masks), dst_image, dst_mask, int((merged > 0).sum())))
        print(f"[DONE] {idx}: parts={len(part_masks)} foreground={rows[-1][-1]}")

    summary = dst_root / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8", newline="") as handle:
        handle.write("idx,source_image,part_mask_count,output_image,output_mask,foreground_pixels\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")
    print(f"[DONE] backup_100 -> {dst_root}")


if __name__ == "__main__":
    main()
