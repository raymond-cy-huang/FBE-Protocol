"""FBE metric implementations and dynamic metric task builder."""

from __future__ import annotations

from . import face, perceptual, pixel, structural
from .base import PairwiseMetric
from .metrics_env_checker import MetricDependencyReport, MetricsEnvChecker
from .metrics_tasker import MetricSpec, MetricsTasker, MetricsTaskerBuilder, build_metrics_tasker
from .registry import get_metric, is_registered, list_metrics, register_metric

__all__ = [
    "MetricDependencyReport",
    "MetricSpec",
    "MetricsEnvChecker",
    "MetricsTasker",
    "MetricsTaskerBuilder",
    "PairwiseMetric",
    "build_metrics_tasker",
    "get_metric",
    "is_registered",
    "list_metrics",
    "register_metric",
]
