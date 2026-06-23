from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

MaskRegion = Literal["foreground", "background"]


@dataclass(frozen=True)
class ScoredPair:
    first_index: int
    second_index: int
    score: float


def mask_to_bool(mask: np.ndarray, region: MaskRegion = "foreground") -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_bool = mask > 0
    if region == "background":
        return ~mask_bool
    if region != "foreground":
        raise ValueError(f"Unsupported mask region: {region}")
    return mask_bool


def mask_iou(
    first: np.ndarray,
    second: np.ndarray,
    region: MaskRegion = "foreground",
) -> float:
    first_bool = mask_to_bool(first, region=region)
    second_bool = mask_to_bool(second, region=region)
    if first_bool.shape != second_bool.shape:
        raise ValueError(f"Mask shapes must match: {first_bool.shape} != {second_bool.shape}")

    intersection = np.logical_and(first_bool, second_bool).sum()
    union = np.logical_or(first_bool, second_bool).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def pairwise_mask_iou_matrix(
    masks: Sequence[np.ndarray],
    region: MaskRegion = "foreground",
) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=np.float32)

    masks_bool = [mask_to_bool(mask, region=region).reshape(-1) for mask in masks]
    first_shape = masks_bool[0].shape
    if any(mask.shape != first_shape for mask in masks_bool):
        raise ValueError("All masks must have the same flattened shape.")

    matrix = np.stack(masks_bool).astype(np.float32)
    areas = matrix.sum(axis=1)
    intersections = matrix @ matrix.T
    unions = areas[:, None] + areas[None, :] - intersections
    return np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)


def select_non_overlapping_pairs(
    pairs: Sequence[ScoredPair],
    max_pairs: int | None = None,
) -> list[ScoredPair]:
    selected: list[ScoredPair] = []
    used_indices: set[int] = set()

    for pair in sorted(pairs, key=lambda item: item.score, reverse=True):
        if pair.first_index in used_indices or pair.second_index in used_indices:
            continue
        selected.append(pair)
        used_indices.update((pair.first_index, pair.second_index))
        if max_pairs is not None and len(selected) >= max_pairs:
            break

    return selected


def find_iou_candidate_pairs(
    masks: Sequence[np.ndarray],
    region: MaskRegion = "foreground",
    min_iou: float = 0.8,
    max_pairs: int | None = None,
) -> list[ScoredPair]:
    """Find non-overlapping mask pairs using area-bound pruning.

    The area ratio min(area_a, area_b) / max(area_a, area_b) is an upper
    bound on IoU, so pairs below min_iou can be skipped without exact scoring.
    """
    if min_iou < 0:
        scores = pairwise_mask_iou_matrix(masks, region=region)
        pair_i, pair_j = np.triu_indices(len(masks), k=1)
        all_pairs = [
            ScoredPair(int(i), int(j), float(scores[i, j]))
            for i, j in zip(pair_i, pair_j)
        ]
        return select_non_overlapping_pairs(all_pairs, max_pairs=max_pairs)

    if min_iou > 1:
        raise ValueError(f"min_iou must be <= 1, got: {min_iou}")
    if len(masks) < 2:
        return []

    masks_bool = [mask_to_bool(mask, region=region).reshape(-1) for mask in masks]
    first_shape = masks_bool[0].shape
    if any(mask.shape != first_shape for mask in masks_bool):
        raise ValueError("All masks must have the same flattened shape.")

    areas = np.array([int(mask.sum()) for mask in masks_bool], dtype=np.int64)
    order = np.argsort(areas)
    sorted_areas = areas[order]

    candidates: list[ScoredPair] = []
    for sorted_pos, first_original_index in enumerate(order):
        first_area = sorted_areas[sorted_pos]
        if first_area == 0:
            continue

        max_area = first_area / min_iou if min_iou > 0 else sorted_areas[-1]
        upper = int(np.searchsorted(sorted_areas, max_area, side="right"))

        first_mask = masks_bool[int(first_original_index)]
        for other_pos in range(sorted_pos + 1, upper):
            second_original_index = int(order[other_pos])
            second_area = sorted_areas[other_pos]
            if second_area == 0:
                continue

            area_bound = first_area / second_area
            if area_bound < min_iou:
                continue

            second_mask = masks_bool[second_original_index]
            intersection = np.logical_and(first_mask, second_mask).sum()
            union = first_area + second_area - intersection
            score = float(intersection / union) if union > 0 else 0.0
            if score >= min_iou:
                candidates.append(
                    ScoredPair(
                        first_index=int(first_original_index),
                        second_index=second_original_index,
                        score=score,
                    )
                )

    return select_non_overlapping_pairs(candidates, max_pairs=max_pairs)
