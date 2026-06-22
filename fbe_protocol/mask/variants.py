from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskVariants:
    original: np.ndarray
    eroded: np.ndarray
    dilated: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "original": self.original,
            "eroded": self.eroded,
            "dilated": self.dilated,
        }


def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Mask variants require a 2D mask, got shape: {mask.shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def build_mask_variants(
    mask: np.ndarray,
    erode_iter: int = 5,
    dilate_iter: int = 5,
    kernel_size: int = 3,
) -> MaskVariants:
    """Build boundary sensitivity masks from the same original mask.

    A 3x3 kernel with N iterations approximates an N-pixel inward/outward
    boundary perturbation.
    """
    original = _normalize_binary_mask(mask)
    kernel_size = _odd_at_least(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    eroded = original.copy()
    if erode_iter > 0:
        eroded = cv2.erode(original, kernel, iterations=int(erode_iter))

    dilated = original.copy()
    if dilate_iter > 0:
        dilated = cv2.dilate(original, kernel, iterations=int(dilate_iter))

    return MaskVariants(
        original=original,
        eroded=_normalize_binary_mask(eroded),
        dilated=_normalize_binary_mask(dilated),
    )
