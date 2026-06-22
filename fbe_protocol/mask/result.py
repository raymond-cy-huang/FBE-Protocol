from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MaskResult:
    mask: np.ndarray
    foreground: np.ndarray | None = None
    background: np.ndarray | None = None
    mask_variants: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
