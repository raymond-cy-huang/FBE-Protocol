"""Flexible task builder for pairwise metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .base import MetricConfig, PairwiseMetric
from .image_io import ImageInput, resolve_image, resolve_mask
from .metrics_env_checker import MetricsEnvChecker
from .registry import get_metric, normalize_metric_name


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kwargs: MetricConfig = field(default_factory=dict)


class MetricsTasker:
    """Runs a configured collection of pairwise metrics."""

    def __init__(self, metrics: Iterable[PairwiseMetric]):
        self.metrics = list(metrics)

    def evaluate_pair(self, reference: ImageInput, prediction: ImageInput, mask: ImageInput | None = None) -> dict[str, float]:
        ref_image = resolve_image(reference)
        pred_image = resolve_image(prediction)
        mask_image = resolve_mask(mask)

        if ref_image.shape != pred_image.shape:
            raise ValueError(f"Image shape mismatch: {ref_image.shape} vs {pred_image.shape}")

        return {metric.name: float(metric(ref_image, pred_image, mask_image)) for metric in self.metrics}

    def evaluate_pairs(
        self,
        pairs: Iterable[tuple[ImageInput, ImageInput]],
        mask: ImageInput | None = None,
    ) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for reference, prediction in pairs:
            row: dict[str, float | str] = {
                "reference": str(reference),
                "prediction": str(prediction),
            }
            row.update(self.evaluate_pair(reference, prediction, mask))
            rows.append(row)
        return rows

    def evaluate_folders(self, reference_dir: str | Path, prediction_dir: str | Path) -> list[dict[str, float | str]]:
        reference_dir = Path(reference_dir)
        prediction_dir = Path(prediction_dir)
        pairs = []
        for reference_path in sorted(path for path in reference_dir.iterdir() if path.is_file()):
            prediction_path = prediction_dir / reference_path.name
            if prediction_path.exists():
                pairs.append((reference_path, prediction_path))
        return self.evaluate_pairs(pairs)


class MetricsTaskerBuilder:
    """Builder for dynamically composing requested metrics."""

    def __init__(self, env_checker: MetricsEnvChecker | None = None, check_environment: bool = True):
        self.env_checker = env_checker or MetricsEnvChecker()
        self.check_environment = check_environment
        self._specs: list[MetricSpec] = []

    def add(self, name: str, **kwargs) -> "MetricsTaskerBuilder":
        self._specs.append(MetricSpec(name=normalize_metric_name(name), kwargs=dict(kwargs)))
        return self

    def add_many(self, metrics: Iterable[str | MetricSpec | tuple[str, MetricConfig]]) -> "MetricsTaskerBuilder":
        for metric in metrics:
            if isinstance(metric, MetricSpec):
                self._specs.append(metric)
            elif isinstance(metric, tuple):
                name, kwargs = metric
                self.add(name, **kwargs)
            else:
                self.add(metric)
        return self

    def build(self) -> MetricsTasker:
        names = [spec.name for spec in self._specs]
        if self.check_environment:
            self.env_checker.require(names)
        return MetricsTasker(get_metric(spec.name, **spec.kwargs) for spec in self._specs)


def build_metrics_tasker(
    metrics: Iterable[str | MetricSpec | tuple[str, MetricConfig]],
    check_environment: bool = True,
) -> MetricsTasker:
    return MetricsTaskerBuilder(check_environment=check_environment).add_many(metrics).build()
