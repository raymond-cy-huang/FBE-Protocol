from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MaskConfig:
    # Contrast normalization hyperparameters.
    contrast_percentiles: Tuple[float, float] = (1.0, 99.5)
    contrast_range_delta: float = 1e-6

    # Median smoothing hyperparameters.
    median_kernel: int = 5

    # Adaptive thresholding hyperparameters.
    tau_min: int = 80
    tau_0: int = 160
    use_otsu: bool = True
    threshold: Optional[float] = None
    invert: bool = False

    # Morphological cleanup hyperparameters.
    auto_kernel_scale: bool = True
    close_kernel: Optional[int] = None
    fill_kernel: Optional[int] = None
    keep_largest_component: bool = True
    fill_area_min: float = 0.05
    fill_area_max: float = 0.95

    # Boundary refinement hyperparameters.
    boundary_open_kernel: int = 0
    boundary_close_kernel: int = 0
    boundary_erode_iter: int = 1
    boundary_dilate_iter: int = 1
