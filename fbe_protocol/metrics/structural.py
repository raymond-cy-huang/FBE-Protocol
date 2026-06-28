"""Structural similarity metrics."""

from __future__ import annotations

import numpy as np

from .base import PairwiseMetric
from .registry import register_metric


class SSIM(PairwiseMetric):
    name = "ssim"
    higher_is_better = True

    def __init__(self, data_range: float = 1.0, channel_axis: int = -1):
        self.data_range = float(data_range)
        self.channel_axis = channel_axis

    def __call__(self, reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
        if reference.shape != prediction.shape:
            raise ValueError(f"Image shape mismatch: {reference.shape} vs {prediction.shape}")
        if mask is not None:
            raise NotImplementedError("Masked SSIM is not implemented; pass unmasked images or crop before calling.")

        from skimage.metrics import structural_similarity

        score = structural_similarity(
            reference.astype(np.float32),
            prediction.astype(np.float32),
            channel_axis=self.channel_axis,
            data_range=self.data_range,
        )
        return float(score)


register_metric("ssim", SSIM)
