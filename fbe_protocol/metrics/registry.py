"""Metric registry with lazy construction."""

from __future__ import annotations

from collections.abc import Callable

from .base import PairwiseMetric

MetricFactory = Callable[..., PairwiseMetric]

_METRIC_REGISTRY: dict[str, MetricFactory] = {}
_ALIASES: dict[str, str] = {
    "id": "id_similarity",
    "identity": "id_similarity",
    "identity_similarity": "id_similarity",
}


def normalize_metric_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return _ALIASES.get(key, key)


def register_metric(name: str, metric_cls: MetricFactory, aliases: tuple[str, ...] = ()) -> None:
    key = normalize_metric_name(name)
    _METRIC_REGISTRY[key] = metric_cls
    for alias in aliases:
        _ALIASES[normalize_metric_name(alias)] = key


def get_metric(name: str, **kwargs) -> PairwiseMetric:
    key = normalize_metric_name(name)
    if key not in _METRIC_REGISTRY:
        raise KeyError(f"Metric '{name}' is not registered. Available metrics: {list_metrics()}")
    return _METRIC_REGISTRY[key](**kwargs)


def list_metrics() -> list[str]:
    return sorted(_METRIC_REGISTRY)


def is_registered(name: str) -> bool:
    return normalize_metric_name(name) in _METRIC_REGISTRY
