#!/usr/bin/env python3
"""Create a soft foreground mask from a hard BBoxMaskPose foreground mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fbe_protocol.mask.split_bg_fg import normalize_mask, write_image
else:
    from .split_bg_fg import normalize_mask, write_image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/soft_mask"


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            return mask[:, :, 3]
        return cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    return mask


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def create_soft_mask(
    mask: np.ndarray,
    shape: tuple[int, int] | None = None,
    kernel_size: int = 31,
    sigma: float = 10.0,
) -> np.ndarray:
    """Return a float32 foreground alpha matte in [0, 1].

    The input mask follows the hard BBoxMaskPose convention: foreground is
    non-zero and background is zero.
    """
    if shape is None:
        if mask.ndim == 3:
            shape = mask.shape[:2]
        else:
            shape = mask.shape

    hard_mask = normalize_mask(mask, shape)
    alpha = hard_mask.astype(np.float32) / 255.0
    kernel_size = _odd_kernel(kernel_size)
    alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), sigmaX=float(sigma))
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def soft_mask_to_u8(soft_mask: np.ndarray) -> np.ndarray:
    return np.clip(soft_mask.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask", type=Path, help="Hard foreground mask path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-layout", choices=("task", "direct"), default="task")
    parser.add_argument("--kernel-size", type=int, default=31)
    parser.add_argument("--sigma", type=float, default=10.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    mask = _read_mask(args.mask)
    soft_mask = create_soft_mask(mask, kernel_size=args.kernel_size, sigma=args.sigma)
    output_dir = args.output_dir / args.mask.stem if args.output_layout == "task" else args.output_dir
    output_path = output_dir / f"{args.mask.stem}_soft_mask.png"
    write_image(output_path, soft_mask_to_u8(soft_mask))
    return output_path


def main() -> None:
    output_path = run(parse_args())
    print(f"[DONE] soft mask saved to: {output_path}")


if __name__ == "__main__":
    main()
