"""Image loading helpers for metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]
ImageInput = Union[PathLike, np.ndarray]


def to_rgb01(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWxC image, got shape={image.shape}")

    image = image[:, :, :3]
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0

    image = image.astype(np.float32)
    if image.size and image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def load_image(path: PathLike) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    return to_rgb01(image)


def load_mask(path: PathLike, threshold: float = 0.5) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    mask = mask.astype(np.float32) / 255.0
    return (mask > threshold).astype(np.float32)


def resolve_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return to_rgb01(image)
    return load_image(image)


def resolve_mask(mask: ImageInput | None) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.float32)
        if mask.size and mask.max() > 1.0:
            mask = mask / 255.0
        return (mask > 0.5).astype(np.float32)
    return load_mask(mask)
