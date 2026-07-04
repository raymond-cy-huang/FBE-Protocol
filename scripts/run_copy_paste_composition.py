#!/usr/bin/env python3
"""Generate the Copy-Paste Composition diagnostic baseline.

This baseline performs a hard pixel-wise transfer from an inversion image to
the GT image using the inversion foreground mask:

    output[p] = inversion[p] if inv_mask[p] > 0 else gt[p]

No mask refinement, blending, inpainting, color correction, resizing, or
post-processing is applied.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/global_path.yaml"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULTS = {
    "dataset_root": None,
    "gt_dir": None,
    "inv_dir": None,
    "mask_dir": None,
    "output_dir": None,
    "methods": ["psp", "e4e", "pti"],
    "genders": ["female", "male"],
    "num_visualizations": 8,
    "seed": 15,
}


@dataclass(frozen=True)
class SamplePaths:
    method: str
    gender: str
    sample_id: str
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


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
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
    set_if_present("gt_dir", "gt_dir", Path)
    set_if_present("inv_dir", "inv_dir", Path)
    set_if_present("mask_dir", "mask_dir", Path)
    set_if_present("output_dir", "output_dir", Path)
    set_if_present("methods", "methods", lambda value: list(value) if isinstance(value, list) else str(value).split(","))
    set_if_present("genders", "genders", lambda value: list(value) if isinstance(value, list) else str(value).split(","))
    set_if_present("num_visualizations", "num_visualizations", int)
    set_if_present("seed", "seed", int)
    return args


def iter_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def index_by_suffix(path: Path, suffix: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for image_path in iter_images(path):
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


def read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8)


def copy_paste_compose(gt: np.ndarray, inversion: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if gt.shape[:2] != inversion.shape[:2] or gt.shape[:2] != mask.shape:
        raise ValueError(f"size mismatch: gt={gt.shape[:2]} inversion={inversion.shape[:2]} mask={mask.shape}")
    mask3 = mask[:, :, None]
    return (inversion * mask3 + gt * (1 - mask3)).astype(np.uint8)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_group_samples(
    gt_dir: Path,
    inv_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    method: str,
    gender: str,
) -> tuple[list[SamplePaths], list[dict[str, str]]]:
    gt_images = index_by_suffix(gt_dir, f"_{gender}_gt")
    inv_images = index_by_suffix(inv_dir, f"_{gender}_{method}")
    masks = index_by_suffix(mask_dir, f"_{gender}_{method}_mask")
    all_ids = sorted(set(gt_images) | set(inv_images) | set(masks))

    samples: list[SamplePaths] = []
    missing_rows: list[dict[str, str]] = []
    for sample_id in all_ids:
        missing = []
        if sample_id not in gt_images:
            missing.append("gt")
        if sample_id not in inv_images:
            missing.append("inversion")
        if sample_id not in masks:
            missing.append("mask")
        if missing:
            missing_rows.append(
                {
                    "method": method,
                    "gender": gender,
                    "sample_id": sample_id,
                    "missing": "|".join(missing),
                    "gt_dir": str(gt_dir),
                    "inv_dir": str(inv_dir),
                    "mask_dir": str(mask_dir),
                }
            )
            continue

        samples.append(
            SamplePaths(
                method=method,
                gender=gender,
                sample_id=sample_id,
                gt=gt_images[sample_id],
                inversion=inv_images[sample_id],
                mask=masks[sample_id],
                output=output_dir / f"cp_{method}_{gender}" / f"{sample_id}.png",
            )
        )

    print(
        f"[INFO] cp_{method}_{gender}: gt={len(gt_images)} inversion={len(inv_images)} "
        f"mask={len(masks)} matched={len(samples)} missing={len(missing_rows)}"
    )
    return samples, missing_rows


def build_dataset_samples(
    dataset_root: Path,
    output_dir: Path,
    methods: list[str],
    genders: list[str],
) -> tuple[list[SamplePaths], list[dict[str, str]]]:
    samples: list[SamplePaths] = []
    missing_rows: list[dict[str, str]] = []
    for method in methods:
        for gender in genders:
            group_samples, group_missing = build_group_samples(
                gt_dir=dataset_root / f"gt_{gender}",
                inv_dir=dataset_root / f"{method}_{gender}",
                mask_dir=dataset_root / f"{method}_{gender}_mask",
                output_dir=output_dir,
                method=method,
                gender=gender,
            )
            samples.extend(group_samples)
            missing_rows.extend(group_missing)
    return samples, missing_rows


def build_visualization(gt: np.ndarray, inversion: np.ndarray, mask: np.ndarray, result: np.ndarray) -> np.ndarray:
    mask_bgr = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return np.concatenate([gt, inversion, mask_bgr, result], axis=1)


def run(args: argparse.Namespace) -> Path:
    output_dir = resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("Provide --output_dir/--output-dir or a --path-profile with output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [method.strip() for method in args.methods if method.strip()]
    genders = [gender.strip() for gender in args.genders if gender.strip()]

    dataset_root = resolve_path(args.dataset_root)
    gt_dir = resolve_path(args.gt_dir)
    inv_dir = resolve_path(args.inv_dir)
    mask_dir = resolve_path(args.mask_dir)

    if dataset_root is not None:
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
        samples, missing_rows = build_dataset_samples(dataset_root, output_dir, methods, genders)
    else:
        if gt_dir is None or inv_dir is None or mask_dir is None:
            raise ValueError("Provide either --dataset_root or all of --gt_dir, --inv_dir, and --mask_dir.")
        method = args.method or inv_dir.name.split("_", 1)[0]
        gender = args.gender or inv_dir.name.rsplit("_", 1)[-1]
        samples, missing_rows = build_group_samples(gt_dir, inv_dir, mask_dir, output_dir, method, gender)

    random.seed(args.seed)
    visualization_ids = {sample.sample_id for sample in random.sample(samples, min(args.num_visualizations, len(samples)))}
    manifest_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []

    for index, sample in enumerate(samples, start=1):
        if index == 1 or index % 100 == 0 or index == len(samples):
            print(f"[INFO] Copy-Paste: {index}/{len(samples)}")
        try:
            gt = read_image(sample.gt)
            inversion = read_image(sample.inversion)
            mask = read_binary_mask(sample.mask)
            result = copy_paste_compose(gt, inversion, mask)
            write_image(sample.output, result)
            if sample.sample_id in visualization_ids:
                viz = build_visualization(gt, inversion, mask, result)
                write_image(output_dir / "visualizations" / f"{sample.method}_{sample.gender}_{sample.sample_id}.png", viz)
        except Exception as exc:
            skipped_rows.append(
                {
                    "method": sample.method,
                    "gender": sample.gender,
                    "sample_id": sample.sample_id,
                    "reason": str(exc),
                    "gt": str(sample.gt),
                    "inversion": str(sample.inversion),
                    "mask": str(sample.mask),
                }
            )
            print(f"[WARN] skip {sample.method}_{sample.gender}/{sample.sample_id}: {exc}")
            continue

        manifest_rows.append(
            {
                "method": sample.method,
                "gender": sample.gender,
                "sample_id": sample.sample_id,
                "gt": str(sample.gt),
                "inversion": str(sample.inversion),
                "mask": str(sample.mask),
                "output": str(sample.output),
            }
        )

    log_dir = output_dir / "logs"
    write_csv(log_dir / "copy_paste_manifest.csv", manifest_rows, ["method", "gender", "sample_id", "gt", "inversion", "mask", "output"])
    write_csv(log_dir / "copy_paste_missing.csv", missing_rows, ["method", "gender", "sample_id", "missing", "gt_dir", "inv_dir", "mask_dir"])
    write_csv(log_dir / "copy_paste_skipped.csv", skipped_rows, ["method", "gender", "sample_id", "reason", "gt", "inversion", "mask"])

    print("\nCopy-Paste Composition")
    print("Pixel-wise hard-mask transfer from inversion foreground to the original GT image.")
    print(f"successful images: {len(manifest_rows)}")
    print(f"missing-file skips: {len(missing_rows)}")
    print(f"processing skips: {len(skipped_rows)}")
    print(f"output: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-profile", default=None, help="Profile name in configs/global_path.yaml.")
    parser.add_argument("--dataset-root", "--dataset_root", dest="dataset_root", type=Path, default=DEFAULTS["dataset_root"])
    parser.add_argument("--gt-dir", "--gt_dir", dest="gt_dir", type=Path, default=DEFAULTS["gt_dir"])
    parser.add_argument("--inv-dir", "--inv_dir", dest="inv_dir", type=Path, default=DEFAULTS["inv_dir"])
    parser.add_argument("--mask-dir", "--mask_dir", dest="mask_dir", type=Path, default=DEFAULTS["mask_dir"])
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=DEFAULTS["output_dir"])
    parser.add_argument("--methods", nargs="+", default=DEFAULTS["methods"])
    parser.add_argument("--genders", nargs="+", choices=("female", "male"), default=DEFAULTS["genders"])
    parser.add_argument("--method", choices=("psp", "e4e", "pti"), default=None, help="Method name for single-group CLI mode.")
    parser.add_argument("--gender", choices=("female", "male"), default=None, help="Gender name for single-group CLI mode.")
    parser.add_argument("--num-visualizations", "--num_visualizations", dest="num_visualizations", type=int, default=DEFAULTS["num_visualizations"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    return apply_profile(parser.parse_args())


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
