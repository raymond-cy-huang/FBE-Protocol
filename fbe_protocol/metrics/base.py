"""Base interfaces for FBE metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PairwiseMetric(ABC):
    """Metric interface for comparing two images."""

    name: str
    higher_is_better: bool | None = None

    @abstractmethod
    def __call__(self, reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Return the metric value for two RGB float images in [0, 1]."""


MetricConfig = dict[str, Any]
