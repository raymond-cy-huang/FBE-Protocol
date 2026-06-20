from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import MaskConfig
from .result import MaskResult


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        if img.shape[2] == 4:
            return img[:, :, :3]
        return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _whiteness_map(bgr: np.ndarray) -> np.ndarray:
    b = bgr[:, :, 0].astype(np.uint16)
    g = bgr[:, :, 1].astype(np.uint16)
    r = bgr[:, :, 2].astype(np.uint16)
    return np.minimum(np.minimum(b, g), r).astype(np.uint8)


def _percentile_contrast(
    x: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.5,
    delta_range: float = 1e-6,
) -> np.ndarray:
    xf = x.astype(np.float32)
    lo, hi = np.percentile(xf, (p_low, p_high))
    if hi <= lo + delta_range:
        return x
    y = (xf - lo) * 255.0 / (hi - lo)
    return np.clip(y, 0, 255).astype(np.uint8)


def _odd_at_least(v: int, mn: int = 3) -> int:
    v = max(int(v), mn)
    return v if v % 2 == 1 else v + 1


def _auto_kernels(h: int, w: int) -> tuple[int, int, int]:
    m = min(h, w)
    k_open = _odd_at_least(int(m * 0.008), 3)
    k_close = _odd_at_least(int(m * 0.016), 5)
    k_fill = _odd_at_least(int(m * 0.020), 7)
    return k_open, k_close, k_fill


def _largest_connected_component(mask_bin: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 2:
        return mask_bin
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask_bin)
    out[labels == max_idx] = 255
    return out


def _refine_mask_input(mask: np.ndarray, config: MaskConfig) -> tuple[np.ndarray, dict[str, Any]]:
    # raw mask generation:
    # BBoxMaskPose is called by the CLI wrapper. The package API receives the
    # produced raw mask as `mask` and normalizes it below.
    bgr = _to_bgr(mask)
    h, w = bgr.shape[:2]

    # minimum-channel whiteness map:
    # Use min(B, G, R) to keep only pixels that are bright in every channel.
    m_white = _whiteness_map(bgr)

    # contrast normalization:
    # Stretch the whiteness map between configured percentiles when the
    # percentile range is wide enough. The default delta preserves the previous
    # hi > lo + 1e-6 behavior.
    p_low, p_high = config.contrast_percentiles
    m_contrast = _percentile_contrast(
        m_white,
        p_low,
        p_high,
        config.contrast_range_delta,
    )

    # median smoothing:
    # Median filtering removes small isolated artifacts before thresholding.
    m_smooth = cv2.medianBlur(m_contrast, _odd_at_least(config.median_kernel))

    # adaptive thresholding:
    # Prefer a fixed threshold when specified, otherwise use Otsu with the
    # existing fallback to tau_0 when Otsu is too low.
    if config.threshold is not None:
        threshold_used = float(config.threshold)
        threshold_mode = "fixed"
        otsu_t = threshold_used
        _, m_b = cv2.threshold(m_smooth, threshold_used, 255, cv2.THRESH_BINARY)
    elif config.use_otsu:
        otsu_t, m_b = cv2.threshold(m_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_used = float(otsu_t)
        threshold_mode = "Otsu"
        if otsu_t < config.tau_min:
            _, m_b = cv2.threshold(m_smooth, config.tau_0, 255, cv2.THRESH_BINARY)
            threshold_used = float(config.tau_0)
            threshold_mode = "fixed fallback"
    else:
        otsu_t = float(config.tau_0)
        threshold_used = float(config.tau_0)
        threshold_mode = "fixed"
        _, m_b = cv2.threshold(m_smooth, config.tau_0, 255, cv2.THRESH_BINARY)

    if config.invert:
        m_b = cv2.bitwise_not(m_b)

    # morphological cleanup:
    # Close small gaps, keep the largest foreground component, then fill small
    # holes only when the foreground area is within the configured valid range.
    _, auto_close, auto_fill = _auto_kernels(h, w)
    k_close = config.close_kernel if config.close_kernel is not None else auto_close
    k_fill = config.fill_kernel if config.fill_kernel is not None else auto_fill

    m_close = cv2.morphologyEx(m_b, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    m_main = _largest_connected_component(m_close) if config.keep_largest_component else m_close

    area_ratio = float(m_main.mean() / 255.0)
    fill_applied = config.fill_area_min < area_ratio < config.fill_area_max
    m_fill = m_main.copy()
    if fill_applied:
        m_fill = cv2.morphologyEx(m_fill, cv2.MORPH_CLOSE, np.ones((k_fill, k_fill), np.uint8))

    # binary mask output:
    # Normalize all foreground pixels to 255 and all background pixels to 0.
    m_fill = np.where(m_fill > 0, 255, 0).astype(np.uint8)
    metadata = {
        "height": h,
        "width": w,
        "whiteness_formula": "min(B,G,R)",
        "contrast_percentiles": f"{p_low},{p_high}",
        "median_kernel": _odd_at_least(config.median_kernel),
        "otsu_t": float(otsu_t),
        "tau_min": config.tau_min,
        "tau_0": config.tau_0,
        "threshold_used": threshold_used,
        "threshold_mode": threshold_mode,
        "invert": bool(config.invert),
        "connected_components": "largest 8-connected foreground component"
        if config.keep_largest_component
        else "all foreground components",
        "k_close": k_close,
        "k_fill": k_fill,
        "area_ratio": area_ratio,
        "fill_applied": fill_applied,
    }
    return m_fill, metadata


def _refine_boundary(
    mask_255: np.ndarray,
    open_k: int = 0,
    close_k: int = 0,
    erode_iter: int = 0,
    dilate_iter: int = 0,
) -> np.ndarray:
    m = np.where(mask_255 > 0, 255, 0).astype(np.uint8)

    # boundary refinement:
    # Apply the configured open, close, erode, and dilate sequence. Defaults
    # match the previous implementation.
    if open_k > 0:
        k = np.ones((_odd_at_least(open_k), _odd_at_least(open_k)), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = np.ones((_odd_at_least(close_k), _odd_at_least(close_k)), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if erode_iter > 0:
        m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=int(erode_iter))
    if dilate_iter > 0:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=int(dilate_iter))

    return np.where(m > 0, 255, 0).astype(np.uint8)


def _extract_from_mask(
    mask_input: np.ndarray,
    image: np.ndarray | None,
    config: MaskConfig,
) -> MaskResult:
    mask, metadata = _refine_mask_input(mask_input, config)
    mask = _refine_boundary(
        mask,
        open_k=config.boundary_open_kernel,
        close_k=config.boundary_close_kernel,
        erode_iter=config.boundary_erode_iter,
        dilate_iter=config.boundary_dilate_iter,
    )
    metadata.update(
        {
            "boundary_open_k": _odd_at_least(config.boundary_open_kernel)
            if config.boundary_open_kernel > 0
            else 0,
            "boundary_close_k": _odd_at_least(config.boundary_close_kernel)
            if config.boundary_close_kernel > 0
            else 0,
            "boundary_erode_kernel": "3x3 square",
            "boundary_erode_iter": int(config.boundary_erode_iter),
            "boundary_dilate_kernel": "3x3 square",
            "boundary_dilate_iter": int(config.boundary_dilate_iter),
            "output_dtype": "uint8",
            "output_values": "{0,255}",
            "foreground_value": 255,
            "background_value": 0,
        }
    )

    foreground = None
    background = None
    result_warnings: list[str] = []
    if image is not None:
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"image shape {image.shape[:2]} does not match mask shape {mask.shape[:2]}")
        foreground = image.copy()
        background = image.copy()
        foreground[mask == 0] = 0
        background[mask == 255] = 0

    return MaskResult(
        mask=mask,
        foreground=foreground,
        background=background,
        metadata=metadata,
        warnings=result_warnings,
    )


def extract(
    image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    config: MaskConfig | None = None,
) -> MaskResult:
    if mask is None:
        if image is not None:
            raise NotImplementedError(
                "Image-to-mask generation is not implemented in the package API. "
                "Use mask= or the CLI BBoxMaskPose wrapper."
            )
        raise ValueError("mask or image is required")

    return _extract_from_mask(mask, image, config or MaskConfig())
