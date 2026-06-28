#!/usr/bin/env python3
"""Generate the Copy-Paste composition baseline for inversion results."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/global_path.yaml"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULTS = {
    "dataset_root": None,
    "output_dir": None,
    "methods": ["psp", "e4e", "pti"],
    "genders": ["female", "male"],
    "threshold": 127,
}


@dataclass(frozen=True)
class SamplePaths:
    method: str
    gender: str
    key: str
    gt: Path
    inversion: Path
    mask: Path
    output: Path


def _windows_drive_from_mnt(path: Path) -> Path:
    text = path.as_posix()
    parts = text.split("/")
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        return Path(f"{drive}:/" + "/".join(parts[3:]))
    return path


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        return REPO_ROOT / path
    if path.exists():
        return path
    windows_path = _windows_drive_from_mnt(path)
    if windows_path != path and windows_path.exists():
        return windows_path
    return path


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
        if key not in section or section[key] is None:
            return
        if getattr(args, attr) == DEFAULTS[attr]:
            setattr(args, attr, caster(section[key]))

    set_if_present("dataset_root", "dataset_root", Path)
    set_if_present("output_dir", "output_dir", Path)
    set_if_present("methods", "methods", lambda value: list(value) if isinstance(value, list) else str(value).split(","))
    set_if_present("genders", "genders", lambda value: list(value) if isinstance(value, list) else str(value).split(","))
    set_if_present("threshold", "threshold", int)
    return args


def _iter_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _index_by_base(path: Path, suffix: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for image_path in _iter_images(path):
        stem = image_path.stem
        if stem.endswith(suffix):
            indexed[stem[: -len(suffix)]] = image_path
    return indexed


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def read_mask(path: Path, shape: tuple[int, int], threshold: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    if mask.shape != shape:
        height, width = shape
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > threshold).astype(np.float32)[:, :, None]


def copy_paste_compose(gt: np.ndarray, inversion: np.ndarray, mask_fg: np.ndarray) -> np.ndarray:
    if gt.shape[:2] != inversion.shape[:2]:
        height, width = gt.shape[:2]
        inversion = cv2.resize(inversion, (width, height), interpolation=cv2.INTER_AREA)
    output = mask_fg * inversion.astype(np.float32) + (1.0 - mask_fg) * gt.astype(np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def progress_iter(items: list[SamplePaths]):
    if tqdm is not None:
        yield from tqdm(items, desc="Copy-Paste", unit="image")
        return

    total = len(items)
    for index, item in enumerate(items, start=1):
        if index == 1 or index % 100 == 0 or index == total:
            print(f"[INFO] Copy-Paste: {index}/{total}")
        yield item


def progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def build_samples(dataset_root: Path, output_root: Path, methods: list[str], genders: list[str]) -> tuple[list[SamplePaths], int]:
    samples: list[SamplePaths] = []
    skipped = 0

    for method in methods:
        method = method.strip()
        if not method:
            continue
        for gender in genders:
            gender = gender.strip()
            if not gender:
                continue

            gt_dir = dataset_root / f"gt_{gender}"
            inv_dir = dataset_root / f"{method}_{gender}"
            mask_dir = dataset_root / f"gt_{gender}_mask"
            out_dir = output_root / f"{method}_{gender}_copy_paste"

            gt_images = _index_by_base(gt_dir, "_gt")
            inv_images = _index_by_base(inv_dir, f"_{method}")
            masks = _index_by_base(mask_dir, "_gt_mask")
            all_keys = sorted(set(gt_images) | set(inv_images) | set(masks))

            missing_count = 0
            for key in all_keys:
                missing = []
                if key not in gt_images:
                    missing.append("GT")
                if key not in inv_images:
                    missing.append("inversion")
                if key not in masks:
                    missing.append("mask")
                if missing:
                    skipped += 1
                    missing_count += 1
                    print(f"[WARN] {method}_{gender}: skip {key}; missing {', '.join(missing)}")
                    continue

                samples.append(
                    SamplePaths(
                        method=method,
                        gender=gender,
                        key=key,
                        gt=gt_images[key],
                        inversion=inv_images[key],
                        mask=masks[key],
                        output=out_dir / f"{key}_{method}_cp.png",
                    )
                )

            print(
                f"[INFO] {method}_{gender}: gt={len(gt_images)} inversion={len(inv_images)} "
                f"mask={len(masks)} matched={len(all_keys) - missing_count} skipped={missing_count}"
            )

    return samples, skipped


def run(args: argparse.Namespace) -> Path:
    if args.dataset_root is None:
        raise ValueError("Provide --dataset-root or a --path-profile with dataset_root.")
    if args.output_dir is None:
        raise ValueError("Provide --output-dir or a --path-profile with output_dir.")

    dataset_root = resolve_path(args.dataset_root)
    output_root = resolve_path(args.output_dir)
    methods = [method.strip() for method in args.methods if method.strip()]
    genders = [gender.strip() for gender in args.genders if gender.strip()]

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    samples, preflight_skipped = build_samples(dataset_root, output_root, methods, genders)
    total_images = len(samples) + preflight_skipped
    processing_skipped = 0
    successes = 0
    records: list[dict[str, str]] = []

    for sample in progress_iter(samples):
        try:
            gt = read_image(sample.gt)
            inversion = read_image(sample.inversion)
            mask_fg = read_mask(sample.mask, gt.shape[:2], args.threshold)
            output = copy_paste_compose(gt=gt, inversion=inversion, mask_fg=mask_fg)
            write_image(sample.output, output)
        except Exception as exc:
            processing_skipped += 1
            progress_write(f"[WARN] {sample.method}_{sample.gender}: skip {sample.key}; {exc}")
            continue

        successes += 1
        records.append(
            {
                "method": sample.method,
                "gender": sample.gender,
                "key": sample.key,
                "gt": str(sample.gt),
                "inversion": str(sample.inversion),
                "mask": str(sample.mask),
                "output": str(sample.output),
            }
        )

    manifest_path = output_root / "copy_paste_manifest.csv"
    if records:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", "gender", "key", "gt", "inversion", "mask", "output"])
            writer.writeheader()
            writer.writerows(records)

    print("\nCopy-Paste")
    print("Foreground from inversion output (X') combined with original GT background using a hard binary mask.")
    print("Used as a diagnostic upper-reference baseline for background preservation.")
    print(f"total images: {total_images}")
    print(f"successful images: {successes}")
    print(f"skipped images: {preflight_skipped + processing_skipped}")
    print(f"output: {output_root}")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-profile", default=None, help="Profile name in configs/global_path.yaml.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULTS["dataset_root"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULTS["output_dir"])
    parser.add_argument("--methods", nargs="+", default=DEFAULTS["methods"])
    parser.add_argument("--genders", nargs="+", choices=("female", "male"), default=DEFAULTS["genders"])
    parser.add_argument("--threshold", type=int, default=DEFAULTS["threshold"], help="Foreground mask threshold.")
    return apply_profile(parser.parse_args())


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
