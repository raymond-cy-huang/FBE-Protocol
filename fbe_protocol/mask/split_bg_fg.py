#!/usr/bin/env python3
"""Split an image into foreground and background images with a binary mask."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fbe_protocol.mask.config import MaskConfig
    from fbe_protocol.mask.extract import extract
else:
    from .config import MaskConfig
    from .extract import extract


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/global_path.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/split_bg_fg"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULTS = {
    "image": None,
    "mask": None,
    "dataset_root": None,
    "genders": ["female", "male"],
    "output_dir": DEFAULT_OUTPUT_DIR,
    "output_layout": "task",
    "output_mode": "full",
    "background_fill": "black",
    "refine_mask": False,
}


@dataclass(frozen=True)
class SplitForegroundBackgroundResult:
    foreground: np.ndarray
    background: np.ndarray
    mask: np.ndarray


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def normalize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != shape:
        height, width = shape
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def split_foreground_background(
    image: np.ndarray,
    mask: np.ndarray,
    refine_mask: bool = False,
) -> SplitForegroundBackgroundResult:
    image_bgr = _to_bgr(image)

    if refine_mask:
        config = MaskConfig(
            boundary_open_kernel=0,
            boundary_close_kernel=0,
            boundary_erode_iter=0,
            boundary_dilate_iter=0,
            variant_erode_iter=-1,
            variant_dilate_iter=-1,
        )
        result = extract(image=image_bgr, mask=mask, config=config)
        if result.foreground is None or result.background is None:
            raise RuntimeError("Mask extraction did not produce foreground/background outputs.")
        return SplitForegroundBackgroundResult(
            foreground=result.foreground,
            background=result.background,
            mask=result.mask,
        )

    mask_255 = normalize_mask(mask, image_bgr.shape[:2])
    foreground = np.zeros_like(image_bgr)
    background = np.zeros_like(image_bgr)
    foreground[mask_255 == 255] = image_bgr[mask_255 == 255]
    background[mask_255 == 0] = image_bgr[mask_255 == 0]
    return SplitForegroundBackgroundResult(
        foreground=foreground,
        background=background,
        mask=mask_255,
    )


def make_preview(image: np.ndarray, result: SplitForegroundBackgroundResult) -> np.ndarray:
    image_bgr = _to_bgr(image)
    mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    panels = [image_bgr, mask_bgr, result.foreground, result.background]
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels)
    preview = np.full((height, width, 3), 245, dtype=np.uint8)
    x = 0
    for panel in panels:
        preview[: panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1]
    return preview


def make_background_output(result: SplitForegroundBackgroundResult, fill: str = "black") -> np.ndarray:
    fill = fill.strip().lower()
    if fill == "transparent":
        alpha = np.where(result.mask == 255, 0, 255).astype(np.uint8)
        return np.dstack([result.background, alpha])
    if fill == "white":
        out = result.background.copy()
        out[result.mask == 255] = 255
        return out
    return result.background


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULTS["image"], help="Input image path.")
    parser.add_argument("--path-profile", default=None, help="Profile name in configs/global_path.yaml.")
    parser.add_argument("--mask", type=Path, default=DEFAULTS["mask"], help="Binary foreground mask path. Foreground is non-zero.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULTS["dataset_root"], help="Normalized dataset root for batch mode.")
    parser.add_argument("--genders", nargs="+", choices=("female", "male"), default=DEFAULTS["genders"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULTS["output_dir"])
    parser.add_argument("--output-layout", choices=("task", "direct"), default=DEFAULTS["output_layout"])
    parser.add_argument("--output-mode", choices=("full", "bg_only"), default=DEFAULTS["output_mode"])
    parser.add_argument("--background-fill", choices=("black", "white", "transparent"), default=DEFAULTS["background_fill"])
    parser.add_argument("--refine-mask", action="store_true", default=DEFAULTS["refine_mask"], help="Refine the input mask through fbe_protocol.mask.extract before splitting.")
    return apply_profile(parser.parse_args())


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
        if key not in section or section[key] is None:
            return
        if getattr(args, attr) == DEFAULTS[attr]:
            setattr(args, attr, caster(section[key]))

    def parse_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    set_if_present("image", "image", Path)
    set_if_present("mask", "mask", Path)
    set_if_present("dataset_root", "dataset_root", Path)
    set_if_present("genders", "genders", lambda value: list(value) if isinstance(value, list) else str(value).split(","))
    set_if_present("output_dir", "output_dir", Path)
    set_if_present("output_layout", "output_layout", str)
    set_if_present("output_mode", "output_mode", str)
    set_if_present("background_fill", "background_fill", str)
    set_if_present("refine_mask", "refine_mask", parse_bool)
    return args


def _iter_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _index_by_base(path: Path, suffix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for image_path in _iter_images(path):
        stem = image_path.stem
        if stem.endswith(suffix):
            out[stem[: -len(suffix)]] = image_path
    return out


def run_dataset_batch(args: argparse.Namespace) -> Path:
    dataset_root = resolve_repo_path(args.dataset_root)
    output_root = resolve_repo_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for gender in args.genders:
        image_dir = dataset_root / f"gt_{gender}"
        mask_dir = dataset_root / f"gt_{gender}_mask"
        out_dir = output_root / f"gt_{gender}_bg"
        out_dir.mkdir(parents=True, exist_ok=True)

        images = _index_by_base(image_dir, "_gt")
        masks = _index_by_base(mask_dir, "_gt_mask")
        keys = sorted(set(images) & set(masks))
        print(f"[INFO] gt_{gender}: images={len(images)} masks={len(masks)} matched={len(keys)}")

        for index, key in enumerate(keys, start=1):
            image = _read_image(images[key])
            mask = _read_image(masks[key], cv2.IMREAD_GRAYSCALE)
            result = split_foreground_background(image, mask, refine_mask=args.refine_mask)
            out_path = out_dir / f"{key}_gt_bg.png"
            write_image(out_path, make_background_output(result, args.background_fill))
            records.append(
                {
                    "gender": gender,
                    "key": key,
                    "image": str(images[key]),
                    "mask": str(masks[key]),
                    "output": str(out_path),
                }
            )
            if index % 100 == 0 or index == len(keys):
                print(f"[OK] gt_{gender}: {index}/{len(keys)}")

    metadata_path = output_root / "split_bg_manifest.csv"
    if records:
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["gender", "key", "image", "mask", "output"])
            writer.writeheader()
            writer.writerows(records)
    print(f"[DONE] batch backgrounds={len(records)} root={output_root}")
    return output_root


def run(args: argparse.Namespace) -> Path:
    if args.dataset_root is not None:
        return run_dataset_batch(args)
    if args.image is None or args.mask is None:
        raise ValueError("Provide image + --mask, or use --dataset-root for batch mode.")

    args.output_dir = resolve_repo_path(args.output_dir)
    image = _read_image(resolve_repo_path(args.image))
    mask = _read_image(resolve_repo_path(args.mask), cv2.IMREAD_GRAYSCALE)
    result = split_foreground_background(image, mask, refine_mask=args.refine_mask)

    output_dir = args.output_dir / args.image.stem if args.output_layout == "task" else args.output_dir
    write_image(output_dir / f"{args.image.stem}_background.png", make_background_output(result, args.background_fill))
    if args.output_mode == "full":
        write_image(output_dir / f"{args.image.stem}_mask.png", result.mask)
        write_image(output_dir / f"{args.image.stem}_foreground.png", result.foreground)
        write_image(output_dir / f"{args.image.stem}_preview.png", make_preview(image, result))
    return output_dir


def main() -> None:
    output_dir = run(parse_args())
    print(f"[DONE] split foreground/background outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
