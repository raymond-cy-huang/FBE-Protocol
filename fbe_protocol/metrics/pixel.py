"""Pixel-level metrics."""

from __future__ import annotations

import numpy as np

from .base import PairwiseMetric
from .registry import register_metric


def _validate_pair(reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != prediction.shape:
        raise ValueError(f"Image shape mismatch: {reference.shape} vs {prediction.shape}")
    return reference.astype(np.float32), prediction.astype(np.float32)


def _masked_values(diff: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return diff.reshape(-1)
    if mask.ndim == 2:
        mask = mask[:, :, None]
    if mask.shape[:2] != diff.shape[:2]:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs image shape {diff.shape}")
    mask_bool = mask.astype(bool)
    if mask_bool.shape[-1] == 1 and diff.ndim == 3:
        mask_bool = np.repeat(mask_bool, diff.shape[-1], axis=-1)
    values = diff[mask_bool]
    if values.size == 0:
        raise ValueError("Mask selects no pixels.")
    return values


class MAE(PairwiseMetric):
    name = "mae"
    higher_is_better = False

    def __call__(self, reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
        reference, prediction = _validate_pair(reference, prediction)
        values = _masked_values(np.abs(reference - prediction), mask)
        return float(np.mean(values))


class RMSE(PairwiseMetric):
    name = "rmse"
    higher_is_better = False

    def __call__(self, reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
        reference, prediction = _validate_pair(reference, prediction)
        values = _masked_values((reference - prediction) ** 2, mask)
        return float(np.sqrt(np.mean(values)))


class PSNR(PairwiseMetric):
    name = "psnr"
    higher_is_better = True

    def __init__(self, data_range: float = 1.0):
        self.data_range = float(data_range)

    def __call__(self, reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
        reference, prediction = _validate_pair(reference, prediction)
        values = _masked_values((reference - prediction) ** 2, mask)
        mse = float(np.mean(values))
        if mse == 0.0:
            return float("inf")
        return float(10.0 * np.log10((self.data_range**2) / mse))


register_metric("mae", MAE)
register_metric("rmse", RMSE)
register_metric("psnr", PSNR)
