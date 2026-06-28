"""Dependency checks for metric construction."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

from .registry import normalize_metric_name


@dataclass(frozen=True)
class MetricDependencyReport:
    metric: str
    available: bool
    missing_modules: tuple[str, ...]


METRIC_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "mae": ("numpy",),
    "rmse": ("numpy",),
    "psnr": ("numpy",),
    "ssim": ("numpy", "skimage"),
    "lpips": ("numpy", "torch", "lpips"),
    "id_similarity": ("numpy", "cv2", "onnxruntime", "insightface"),
}


class MetricsEnvChecker:
    """Checks whether requested metric dependencies exist in the active env."""

    def dependencies_for(self, metric_name: str) -> tuple[str, ...]:
        key = normalize_metric_name(metric_name)
        if key not in METRIC_DEPENDENCIES:
            raise KeyError(f"Unknown metric dependencies for '{metric_name}'.")
        return METRIC_DEPENDENCIES[key]

    def check(self, metric_name: str) -> MetricDependencyReport:
        key = normalize_metric_name(metric_name)
        dependencies = self.dependencies_for(key)
        missing = tuple(module for module in dependencies if find_spec(module) is None)
        return MetricDependencyReport(metric=key, available=not missing, missing_modules=missing)

    def check_many(self, metric_names: list[str] | tuple[str, ...]) -> list[MetricDependencyReport]:
        return [self.check(name) for name in metric_names]

    def require(self, metric_names: list[str] | tuple[str, ...]) -> None:
        reports = self.check_many(metric_names)
        missing = [report for report in reports if not report.available]
        if not missing:
            return

        details = "; ".join(
            f"{report.metric}: missing {', '.join(report.missing_modules)}" for report in missing
        )
        raise RuntimeError(f"Metric dependencies are not available in this environment: {details}")
