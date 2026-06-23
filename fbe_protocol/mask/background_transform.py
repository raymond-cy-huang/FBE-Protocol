from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


SUPPORTED_BACKGROUND_MODES = ("white", "black", "mean", "median", "mode", "blur", "gray")


@dataclass(frozen=True)
class BackgroundTransformConfig:
    mode: str = "white"
    erode_kernel: int = 31
    blur_kernel: int = 31
    mode_bins: int = 32
    mode_sample_max: int = 300000

    @classmethod
    def from_env(cls) -> "BackgroundTransformConfig":
        mode = os.getenv("BMP_BG_MODE", "white").strip().lower()
        if mode not in SUPPORTED_BACKGROUND_MODES:
            mode = "white"
        return cls(
            mode=mode,
            erode_kernel=int(os.getenv("BMP_BG_ERODE_K", "31")),
            blur_kernel=int(os.getenv("BMP_BG_BLUR_K", "31")),
            mode_bins=int(os.getenv("BMP_BG_MODE_BINS", "32")),
            mode_sample_max=int(os.getenv("BMP_BG_MODE_SAMPLE_MAX", "300000")),
        )


@dataclass(frozen=True)
class BackgroundTransformResult:
    foreground: np.ndarray
    background: np.ndarray
    fill_rgb: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_u8_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image_f = image.astype(np.float32)
    if image_f.max() <= 1.0 + 1e-6:
        return np.clip(image_f * 255.0, 0, 255).astype(np.uint8)
    return np.clip(image_f, 0, 255).astype(np.uint8)


def _odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def _bg_stat_mask(mask_bool: np.ndarray, kernel_size: int) -> np.ndarray:
    k = _odd_at_least(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_u8 = mask_bool.astype(np.uint8) * 255
    eroded = cv2.erode(mask_u8, kernel, iterations=1) > 0
    if int(eroded.sum()) < 256:
        return mask_bool
    return eroded


def quantized_mode_color(
    bg_pixels: np.ndarray,
    bins: int = 32,
    sample_max: int = 300000,
) -> np.ndarray:
    if bg_pixels.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)

    if bg_pixels.shape[0] > sample_max:
        rng = np.random.default_rng(12345)
        idx = rng.choice(bg_pixels.shape[0], size=sample_max, replace=False)
        bg_pixels = bg_pixels[idx]

    bins = max(2, min(256, int(bins)))
    bin_size = 256 // bins
    if bin_size < 1:
        bin_size = 1
        bins = 256

    quantized = (bg_pixels // bin_size).astype(np.int32)
    code = quantized[:, 0] * (bins * bins) + quantized[:, 1] * bins + quantized[:, 2]
    uniq, counts = np.unique(code, return_counts=True)
    best_code = int(uniq[np.argmax(counts)])

    qb0 = best_code // (bins * bins)
    rem = best_code % (bins * bins)
    qb1 = rem // bins
    qb2 = rem % bins

    in_bin = (quantized[:, 0] == qb0) & (quantized[:, 1] == qb1) & (quantized[:, 2] == qb2)
    if not np.any(in_bin):
        center = np.array(
            [(qb0 + 0.5) * bin_size, (qb1 + 0.5) * bin_size, (qb2 + 0.5) * bin_size],
            dtype=np.float32,
        )
        return np.clip(center, 0, 255).astype(np.uint8)

    return np.clip(bg_pixels[in_bin].mean(axis=0), 0, 255).astype(np.uint8)


def apply_background_transform(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    config: BackgroundTransformConfig | None = None,
) -> BackgroundTransformResult:
    config = config or BackgroundTransformConfig()
    mode = config.mode if config.mode in SUPPORTED_BACKGROUND_MODES else "white"

    image_u8 = _to_u8_bgr(image)
    if foreground_mask.shape != image_u8.shape[:2]:
        raise ValueError(f"foreground mask shape {foreground_mask.shape} does not match image shape {image_u8.shape[:2]}")

    fore_mask = foreground_mask.astype(bool)
    back_mask = ~fore_mask
    h, w = image_u8.shape[:2]
    fg_img = image_u8.copy()
    fill_rgb: np.ndarray | None = None
    metadata: dict[str, Any] = {"mode": mode}

    if mode in ("white", "black", "mean", "median", "mode"):
        if mode == "white":
            fill_rgb = np.array([255, 255, 255], dtype=np.uint8)
        elif mode == "black":
            fill_rgb = np.array([0, 0, 0], dtype=np.uint8)
        else:
            stat_mask = _bg_stat_mask(back_mask, config.erode_kernel)
            bg_pixels = image_u8[stat_mask]
            if bg_pixels.size == 0:
                bg_pixels = image_u8[back_mask] if back_mask.any() else image_u8.reshape(-1, 3)

            if mode == "mean":
                fill_rgb = np.clip(bg_pixels.mean(axis=0), 0, 255).astype(np.uint8)
            elif mode == "median":
                fill_rgb = np.clip(np.median(bg_pixels, axis=0), 0, 255).astype(np.uint8)
            else:
                fill_rgb = quantized_mode_color(
                    bg_pixels,
                    bins=config.mode_bins,
                    sample_max=config.mode_sample_max,
                )
            metadata.update(
                {
                    "background_pixel_count": int(bg_pixels.shape[0]),
                    "mode_bins": int(config.mode_bins),
                    "erode_kernel": int(config.erode_kernel),
                }
            )

        fg_img[back_mask] = fill_rgb
        bg_img = np.tile(fill_rgb.reshape(1, 1, 3), (h, w, 1))
        bg_img[back_mask] = image_u8[back_mask]

    elif mode == "blur":
        blur_kernel = _odd_at_least(config.blur_kernel)
        blurred = cv2.GaussianBlur(image_u8, (blur_kernel, blur_kernel), 0)
        fg_img[back_mask] = blurred[back_mask]
        bg_img = np.full((h, w, 3), 255, dtype=np.uint8)
        bg_img[back_mask] = blurred[back_mask]
        metadata["blur_kernel"] = blur_kernel

    else:
        gray1 = cv2.cvtColor(image_u8, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
        fg_img[back_mask] = gray3[back_mask]
        bg_img = np.full((h, w, 3), 255, dtype=np.uint8)
        bg_img[back_mask] = gray3[back_mask]

    if fill_rgb is not None:
        metadata["fill_rgb"] = fill_rgb.tolist()

    return BackgroundTransformResult(
        foreground=fg_img,
        background=bg_img,
        fill_rgb=fill_rgb,
        metadata=metadata,
    )
