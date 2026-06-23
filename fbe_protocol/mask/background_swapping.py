from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BackgroundSwapResult:
    first_foreground_second_background: np.ndarray
    second_foreground_first_background: np.ndarray


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _mask_alpha(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {shape}")
    return (mask > 0).astype(np.float32)[:, :, None]


def compose_foreground_background(
    foreground_image: np.ndarray,
    background_image: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    foreground = _to_bgr(foreground_image)
    background = _to_bgr(background_image)
    if foreground.shape != background.shape:
        raise ValueError(f"Image shapes must match: {foreground.shape} != {background.shape}")

    alpha = _mask_alpha(foreground_mask, foreground.shape[:2])
    out = background.astype(np.float32) * (1.0 - alpha) + foreground.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def swap_backgrounds(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> BackgroundSwapResult:
    return BackgroundSwapResult(
        first_foreground_second_background=compose_foreground_background(
            first_image,
            second_image,
            first_mask,
        ),
        second_foreground_first_background=compose_foreground_background(
            second_image,
            first_image,
            second_mask,
        ),
    )
