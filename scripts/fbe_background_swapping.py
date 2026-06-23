#!/usr/bin/env python3
"""Swap backgrounds for mask-IoU matched image pairs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from fbe_protocol.mask import (
    MaskRegion,
    ScoredPair,
    build_mask_variants,
    find_iou_candidate_pairs,
    mask_to_bool,
    swap_backgrounds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TASK_NAME = Path(__file__).stem
CONFIG_PATH = REPO_ROOT / "configs/global_path.yaml"
DEFAULTS = {
    "image_dir": REPO_ROOT / "images",
    "mask_dir": REPO_ROOT / "results/fbe_extract_multi_masks",
    "dataset_root": None,
    "output_dir": REPO_ROOT / "results",
    "output_layout": "task",
    "mask_region": "background",
    "min_iou": 0.8,
    "anchor_count": 0,
    "max_pairs": 20,
    "pair_mask_size": 128,
    "variant_px": -1,
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read configs/global_path.yaml. Run `bash setup.sh`.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    if not args.path_profile:
        return args
    config = read_config(CONFIG_PATH)
    section = config.get(args.path_profile, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config profile must be a YAML object: {args.path_profile}")

    def set_if_present(attr: str, key: str, caster=lambda value: value) -> None:
        value = section.get(key)
        if value is not None and getattr(args, attr) == DEFAULTS[attr]:
            setattr(args, attr, caster(value))

    set_if_present("dataset_root", "dataset_root", Path)
    set_if_present("image_dir", "image_dir", Path)
    set_if_present("mask_dir", "mask_dir", Path)
    set_if_present("output_dir", "output_dir", Path)
    set_if_present("output_layout", "output_layout", str)
    set_if_present("mask_region", "mask_region", str)
    set_if_present("min_iou", "min_iou", float)
    set_if_present("anchor_count", "anchor_count", int)
    set_if_present("max_pairs", "max_pairs", int)
    set_if_present("pair_mask_size", "pair_mask_size", int)
    set_if_present("variant_px", "variant_px", int)
    return args


def iter_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def normalized_dataset_mask_path(dataset_root: Path, image_path: Path) -> Path:
    if "_female_" in image_path.name:
        mask_dir = dataset_root / "gt_female_mask"
    elif "_male_" in image_path.name:
        mask_dir = dataset_root / "gt_male_mask"
    else:
        raise ValueError(f"Cannot infer mask directory from image name: {image_path.name}")
    return mask_dir / image_path.name.replace("_gt.jpg", "_gt_mask.png")


def find_mask(mask_dir: Path, image_path: Path) -> Path:
    candidates = [
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / image_path.stem / f"{image_path.stem}_mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find mask for {image_path.name} under {mask_dir}")


def resize_mask_for_pairing(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return mask
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def resize_to_first(first: np.ndarray, second: np.ndarray, interpolation: int) -> np.ndarray:
    h, w = first.shape[:2]
    if second.shape[:2] == (h, w):
        return second
    return cv2.resize(second, (w, h), interpolation=interpolation)


def make_preview(first: np.ndarray, second: np.ndarray, first_out: np.ndarray, second_out: np.ndarray) -> np.ndarray:
    panels = [first, second, first_out, second_out]
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels)
    preview = np.full((height, width, 3), 245, dtype=np.uint8)
    x = 0
    for panel in panels:
        preview[: panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1]
    return preview


def variant_masks(mask: np.ndarray, variant_px: int) -> dict[str, np.ndarray]:
    if variant_px < 0:
        return {"original": mask}
    variants = build_mask_variants(mask, erode_iter=variant_px, dilate_iter=variant_px).as_dict()
    return variants


def find_anchor_candidate_pairs(
    masks: list[np.ndarray],
    region: MaskRegion,
    min_iou: float,
    anchor_count: int,
) -> list[ScoredPair]:
    pairing_masks = [mask_to_bool(mask, region=region).reshape(-1) for mask in masks]
    areas = np.array([int(mask.sum()) for mask in pairing_masks], dtype=np.int64)
    pairs: list[ScoredPair] = []

    for anchor_index in range(min(anchor_count, len(pairing_masks))):
        anchor = pairing_masks[anchor_index]
        anchor_area = areas[anchor_index]
        best_index = -1
        best_score = -1.0

        for candidate_index, candidate in enumerate(pairing_masks):
            if candidate_index == anchor_index:
                continue
            candidate_area = areas[candidate_index]
            max_area = max(anchor_area, candidate_area)
            if max_area == 0:
                continue
            area_bound = min(anchor_area, candidate_area) / max_area
            if min_iou >= 0 and area_bound < min_iou:
                continue

            intersection = np.logical_and(anchor, candidate).sum()
            union = anchor_area + candidate_area - intersection
            score = float(intersection / union) if union > 0 else 0.0
            if min_iou >= 0 and score < min_iou:
                continue
            if score > best_score:
                best_index = candidate_index
                best_score = score

        if best_index >= 0:
            pairs.append(ScoredPair(anchor_index, best_index, best_score))

    return pairs


def run(args: argparse.Namespace) -> None:
    dataset_root = resolve_repo_path(args.dataset_root) if args.dataset_root is not None else None
    image_dir = dataset_root / "gt_mix_gender" if dataset_root is not None else resolve_repo_path(args.image_dir)
    mask_dir = resolve_repo_path(args.mask_dir)
    output_root = resolve_repo_path(args.output_dir)
    if args.output_layout == "task":
        output_root = output_root / TASK_NAME

    image_paths = iter_images(image_dir)
    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least two images in: {image_dir}")

    if dataset_root is not None:
        mask_paths = [normalized_dataset_mask_path(dataset_root, image_path) for image_path in image_paths]
    else:
        mask_paths = [find_mask(mask_dir, image_path) for image_path in image_paths]
    missing_masks = [path for path in mask_paths if not path.exists()]
    if missing_masks:
        raise FileNotFoundError(f"Missing {len(missing_masks)} masks, first missing: {missing_masks[0]}")

    masks = [read_mask(path) for path in mask_paths]
    pairing_masks = [resize_mask_for_pairing(mask, args.pair_mask_size) for mask in masks]
    if args.anchor_count > 0:
        pairs = find_anchor_candidate_pairs(
            pairing_masks,
            region=args.mask_region,
            min_iou=args.min_iou,
            anchor_count=args.anchor_count,
        )
    else:
        max_pairs = None if args.max_pairs <= 0 else args.max_pairs
        pairs = find_iou_candidate_pairs(
            pairing_masks,
            region=args.mask_region,
            min_iou=args.min_iou,
            max_pairs=max_pairs,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for pair_index, pair in enumerate(pairs, start=1):
        first_image_path = image_paths[pair.first_index]
        second_image_path = image_paths[pair.second_index]
        first_mask_path = mask_paths[pair.first_index]
        second_mask_path = mask_paths[pair.second_index]

        first_image = read_image(first_image_path)
        second_image = resize_to_first(first_image, read_image(second_image_path), cv2.INTER_AREA)
        first_mask = read_mask(first_mask_path)
        second_mask = resize_to_first(first_mask, read_mask(second_mask_path), cv2.INTER_NEAREST)

        pair_dir = output_root / f"pair_{pair_index:06d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "pair_index": pair_index,
            "first_image": first_image_path.name,
            "second_image": second_image_path.name,
            "first_mask": str(first_mask_path),
            "second_mask": str(second_mask_path),
            "mask_region": args.mask_region,
            "mask_iou": f"{pair.score:.8f}",
            "variant_px": args.variant_px,
        }

        first_variants = variant_masks(first_mask, args.variant_px)
        second_variants = variant_masks(second_mask, args.variant_px)
        for variant_name, first_variant_mask in first_variants.items():
            second_variant_mask = second_variants[variant_name]
            result = swap_backgrounds(first_image, second_image, first_variant_mask, second_variant_mask)

            suffix = "" if variant_name == "original" else f"_{variant_name}"
            first_out_name = f"A_fg_B_bg{suffix}.png"
            second_out_name = f"B_fg_A_bg{suffix}.png"
            write_image(pair_dir / first_out_name, result.first_foreground_second_background)
            write_image(pair_dir / second_out_name, result.second_foreground_first_background)
            record[f"a_fg_b_bg{suffix}"] = str(pair_dir / first_out_name)
            record[f"b_fg_a_bg{suffix}"] = str(pair_dir / second_out_name)

            if variant_name == "original":
                preview = make_preview(
                    first_image,
                    second_image,
                    result.first_foreground_second_background,
                    result.second_foreground_first_background,
                )
                write_image(pair_dir / "preview.png", preview)
                record["preview"] = str(pair_dir / "preview.png")

        records.append(record)
        print(
            f"[OK] pair_{pair_index:06d} IoU={pair.score:.6f}: "
            f"{first_image_path.name} <-> {second_image_path.name}"
        )

    metadata_path = output_root / "pairs.csv"
    if records:
        fieldnames = sorted({key for record in records for key in record})
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    print(f"[DONE] pairs={len(records)} output={output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-profile", default=None)
    parser.add_argument("--image-dir", type=Path, default=DEFAULTS["image_dir"])
    parser.add_argument("--mask-dir", type=Path, default=DEFAULTS["mask_dir"])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULTS["dataset_root"], help="Normalized dataset root containing gt_mix_gender and gt_*_mask folders.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULTS["output_dir"])
    parser.add_argument("--output-layout", choices=("task", "direct"), default=DEFAULTS["output_layout"])
    parser.add_argument("--mask-region", choices=("foreground", "background"), default=DEFAULTS["mask_region"])
    parser.add_argument("--min-iou", type=float, default=DEFAULTS["min_iou"])
    parser.add_argument("--max-pairs", type=int, default=DEFAULTS["max_pairs"], help="Maximum pairs to output. Use <=0 for no limit.")
    parser.add_argument("--anchor-count", type=int, default=DEFAULTS["anchor_count"], help="Use the first N images as A anchors and search B over the full pool.")
    parser.add_argument("--pair-mask-size", type=int, default=DEFAULTS["pair_mask_size"])
    parser.add_argument(
        "--variant-px",
        type=int,
        default=DEFAULTS["variant_px"],
        help="Pixel radius for eroded/dilated mask variants. Use -1 to disable variant outputs.",
    )
    return apply_profile(parser.parse_args())


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
