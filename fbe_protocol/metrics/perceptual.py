"""Deep perceptual metrics."""

from __future__ import annotations

import numpy as np

from .base import PairwiseMetric
from .registry import register_metric


class LPIPS(PairwiseMetric):
    name = "lpips"
    higher_is_better = False

    def __init__(self, net: str = "vgg", device: str | None = None):
        import lpips
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch = torch
        self.model = lpips.LPIPS(net=net).eval().to(self.device)

    def _to_tensor(self, image: np.ndarray):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"LPIPS expects HxWx3 RGB image, got shape={image.shape}")
        tensor = self.torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    def __call__(self, reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
        if reference.shape != prediction.shape:
            raise ValueError(f"Image shape mismatch: {reference.shape} vs {prediction.shape}")
        if mask is not None:
            raise NotImplementedError("Masked LPIPS is not implemented; pass unmasked images or crop before calling.")

        ref_tensor = self._to_tensor(reference).to(self.device)
        pred_tensor = self._to_tensor(prediction).to(self.device)
        with self.torch.no_grad():
            distance = self.model(ref_tensor, pred_tensor)
        return float(distance.item())


register_metric("lpips", LPIPS)
