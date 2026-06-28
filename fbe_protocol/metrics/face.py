"""Face identity similarity metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import PairwiseMetric
from .registry import register_metric


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))


def _pad_image(image: np.ndarray, pad_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    pad = int(min(height, width) * pad_ratio)
    if pad <= 0:
        return image
    return np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)


class IDSimilarity(PairwiseMetric):
    name = "id_similarity"
    higher_is_better = True

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        providers: Sequence[str] | None = None,
        pad_ratio: float = 0.25,
    ):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        if providers is None:
            available = set(ort.get_available_providers())
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        self.pad_ratio = float(pad_ratio)
        self.providers = list(providers)
        self.app = FaceAnalysis(name=model_name, providers=self.providers)
        ctx_id = 0 if self.providers and self.providers[0] == "CUDAExecutionProvider" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    @staticmethod
    def _rgb01_to_bgr_u8(image: np.ndarray) -> np.ndarray:
        import cv2

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
        rgb = (image * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _pick_largest_face(faces):
        def area(face) -> float:
            x1, y1, x2, y2 = face.bbox.astype(np.float32)
            return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

        return max(faces, key=area)

    def _embedding(self, image: np.ndarray) -> np.ndarray:
        image = _pad_image(image, self.pad_ratio)
        faces = self.app.get(self._rgb01_to_bgr_u8(image))
        if not faces:
            raise RuntimeError("ID similarity failed: no face detected.")

        face = self._pick_largest_face(faces)
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = face.embedding
        return np.asarray(embedding, dtype=np.float32)

    def __call__(self, reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
        if mask is not None:
            raise NotImplementedError("Masked ID similarity is not implemented.")
        return _cosine(self._embedding(reference), self._embedding(prediction))


register_metric("id_similarity", IDSimilarity, aliases=("id", "identity", "identity_similarity"))
