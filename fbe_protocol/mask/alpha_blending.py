#!/usr/bin/env python3
"""Alpha blend foreground/background images using a soft foreground mask."""

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
    from fbe_protocol.mask.soft_mask import create_soft_mask, soft_mask_to_u8
    from fbe_protocol.mask.split_bg_fg import split_foreground_background, write_image
else:
    from .soft_mask import create_soft_mask, soft_mask_to_u8
    from .split_bg_fg import split_foreground_background, write_image


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/global_path.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/alpha_blending"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULTS = {
    "dataset_root": None,
    "methods": ["e4e", "psp", "pti"],
    "genders": ["female", "male"],
    "image": None,
    "foreground": None,
    "background": None,
    "mask": None,
    "soft_mask": None,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "output_layout": "task",
    "output_mode": "full",
    "kernel_size": 31,
    "sigma": 10.0,
    "save_soft_mask": False,
}


@dataclass(frozen=True)
class AlphaBlendingResult:
    image: np.ndarray
    soft_mask: np.ndarray
    foreground: np.ndarray
    background: np.ndarray


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


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            return mask[:, :, 3]
        return cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    return mask


def load_soft_mask(
    mask: np.ndarray | None,
    soft_mask: np.ndarray | None,
    shape: tuple[int, int],
    kernel_size: int = 31,
    sigma: float = 10.0,
) -> np.ndarray:
    if soft_mask is not None:
        alpha = soft_mask.astype(np.float32)
        if alpha.max(initial=0.0) > 1.0:
            alpha /= 255.0
        if alpha.shape != shape:
            height, width = shape
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    if mask is None:
        raise ValueError("Either mask or soft_mask is required.")
    return create_soft_mask(mask, shape=shape, kernel_size=kernel_size, sigma=sigma)


def alpha_blend_images(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray | None = None,
    soft_mask: np.ndarray | None = None,
    kernel_size: int = 31,
    sigma: float = 10.0,
) -> AlphaBlendingResult:
    foreground_bgr = _to_bgr(foreground)
    background_bgr = _to_bgr(background)
    if foreground_bgr.shape[:2] != background_bgr.shape[:2]:
        height, width = foreground_bgr.shape[:2]
        background_bgr = cv2.resize(background_bgr, (width, height), interpolation=cv2.INTER_AREA)

    alpha = load_soft_mask(
        mask=mask,
        soft_mask=soft_mask,
        shape=foreground_bgr.shape[:2],
        kernel_size=kernel_size,
        sigma=sigma,
    )
    alpha_3c = alpha[:, :, None]
    blended = foreground_bgr.astype(np.float32) * alpha_3c + background_bgr.astype(np.float32) * (1.0 - alpha_3c)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return AlphaBlendingResult(
        image=blended,
        soft_mask=alpha,
        foreground=foreground_bgr,
        background=background_bgr,
    )


def make_preview(result: AlphaBlendingResult) -> np.ndarray:
    soft_mask_bgr = cv2.cvtColor(soft_mask_to_u8(result.soft_mask), cv2.COLOR_GRAY2BGR)
    panels = [result.foreground, result.background, soft_mask_bgr, result.image]
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels)
    preview = np.full((height, width, 3), 245, dtype=np.uint8)
    x = 0
    for panel in panels:
        preview[: panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1]
    return preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULTS["dataset_root"], help="Normalized dataset root for batch mode.")
    parser.add_argument("--methods", nargs="+", default=DEFAULTS["methods"], help="Inversion methods to run in batch mode.")
    parser.add_argument("--genders", nargs="+", choices=("female", "male"), default=DEFAULTS["genders"], help="Genders to run in batch mode.")
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULTS["image"], help="Original image for soft mask + original fg/bg mode.")
    parser.add_argument("--path-profile", default=None, help="Profile name in configs/global_path.yaml.")
    parser.add_argument("--foreground", "--fg", type=Path, default=DEFAULTS["foreground"], help="Specified foreground image.")
    parser.add_argument("--background", "--bg", type=Path, default=DEFAULTS["background"], help="Specified background image.")
    parser.add_argument("--mask", type=Path, default=DEFAULTS["mask"], help="Hard foreground mask. It will be converted to a soft mask first.")
    parser.add_argument("--soft-mask", type=Path, default=DEFAULTS["soft_mask"], help="Precomputed soft foreground mask.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULTS["output_dir"])
    parser.add_argument("--output-layout", choices=("task", "direct"), default=DEFAULTS["output_layout"])
    parser.add_argument(
        "--output-mode",
        choices=("full", "blend_only"),
        default=DEFAULTS["output_mode"],
        help="full writes preview/optional soft mask; blend_only writes only the alpha blended result.",
    )
    parser.add_argument("--kernel-size", type=int, default=DEFAULTS["kernel_size"])
    parser.add_argument("--sigma", type=float, default=DEFAULTS["sigma"])
    parser.add_argument("--save-soft-mask", action="store_true", default=DEFAULTS["save_soft_mask"])
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
    set_if_present("dataset_root", "dataset_root", Path)
    set_if_present("methods", "methods", lambda value: list(value) if isinstance(value, list) else str(value).split(","))
    set_if_present("genders", "genders", lambda value: list(value) if isinstance(value, list) else str(value).split(","))
    set_if_present("foreground", "foreground", Path)
    set_if_present("background", "background", Path)
    set_if_present("mask", "mask", Path)
    set_if_present("soft_mask", "soft_mask", Path)
    set_if_present("output_dir", "output_dir", Path)
    set_if_present("output_layout", "output_layout", str)
    set_if_present("output_mode", "output_mode", str)
    set_if_present("kernel_size", "kernel_size", int)
    set_if_present("sigma", "sigma", float)
    set_if_present("save_soft_mask", "save_soft_mask", parse_bool)
    return args


def _iter_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _index_by_base(path: Path, suffix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for image_path in _iter_images(path):
        stem = image_path.stem
        if not stem.endswith(suffix):
            continue
        out[stem[: -len(suffix)]] = image_path
    return out


def _write_alpha_blend(path: Path, result: AlphaBlendingResult) -> None:
    write_image(path, result.image)


def run_dataset_batch(args: argparse.Namespace) -> Path:
    dataset_root = resolve_repo_path(args.dataset_root)
    output_root = resolve_repo_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for method in args.methods:
        method = method.strip()
        if not method:
            continue
        for gender in args.genders:
            gender = gender.strip()
            fg_dir = dataset_root / f"{method}_{gender}"
            bg_dir = dataset_root / f"gt_{gender}"
            soft_mask_dir = dataset_root / f"gt_{gender}_mask_soft"
            hard_mask_dir = dataset_root / f"gt_{gender}_mask"
            out_dir = output_root / f"{method}_{gender}_alpha_blend"
            out_dir.mkdir(parents=True, exist_ok=True)

            foregrounds = _index_by_base(fg_dir, f"_{method}")
            backgrounds = _index_by_base(bg_dir, "_gt")
            soft_masks = _index_by_base(soft_mask_dir, "_gt_mask_soft") if soft_mask_dir.exists() else {}
            hard_masks = _index_by_base(hard_mask_dir, "_gt_mask") if hard_mask_dir.exists() else {}
            keys = sorted(set(foregrounds) & set(backgrounds) & (set(soft_masks) | set(hard_masks)))

            print(
                f"[INFO] {method}_{gender}: fg={len(foregrounds)} bg={len(backgrounds)} "
                f"soft_mask={len(soft_masks)} hard_mask={len(hard_masks)} matched={len(keys)}"
            )

            missing = sorted(set(foregrounds) - set(backgrounds))
            if missing:
                print(f"[WARN] {method}_{gender}: missing GT backgrounds for {len(missing)} foregrounds; first={missing[0]}")

            for index, key in enumerate(keys, start=1):
                foreground = _read_image(foregrounds[key])
                background = _read_image(backgrounds[key])
                soft_mask = _read_mask(soft_masks[key]) if key in soft_masks else None
                hard_mask = _read_mask(hard_masks[key]) if key not in soft_masks and key in hard_masks else None
                result = alpha_blend_images(
                    foreground=foreground,
                    background=background,
                    mask=hard_mask,
                    soft_mask=soft_mask,
                    kernel_size=args.kernel_size,
                    sigma=args.sigma,
                )
                out_path = out_dir / f"{key}_{method}_alpha_blend.png"
                _write_alpha_blend(out_path, result)
                records.append(
                    {
                        "method": method,
                        "gender": gender,
                        "key": key,
                        "foreground": str(foregrounds[key]),
                        "background": str(backgrounds[key]),
                        "mask": str(soft_masks.get(key, hard_masks.get(key, ""))),
                        "output": str(out_path),
                    }
                )
                if index % 100 == 0 or index == len(keys):
                    print(f"[OK] {method}_{gender}: {index}/{len(keys)}")

    metadata_path = output_root / "alpha_blending_manifest.csv"
    if records:
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["method", "gender", "key", "foreground", "background", "mask", "output"],
            )
            writer.writeheader()
            writer.writerows(records)
    print(f"[DONE] batch outputs={len(records)} root={output_root}")
    return output_root


def _output_dir(args: argparse.Namespace, stem: str) -> Path:
    return args.output_dir / stem if args.output_layout == "task" else args.output_dir


def _load_mask_args(args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None]:
    if args.mask is None and args.soft_mask is None:
        raise ValueError("Provide --mask or --soft-mask.")
    if args.mask is not None and args.soft_mask is not None:
        raise ValueError("Provide only one of --mask or --soft-mask.")
    mask = _read_mask(resolve_repo_path(args.mask)) if args.mask is not None else None
    soft_mask = _read_mask(resolve_repo_path(args.soft_mask)) if args.soft_mask is not None else None
    return mask, soft_mask


def _inputs_from_args(args: argparse.Namespace) -> tuple[str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    mask, soft_mask = _load_mask_args(args)

    has_explicit_pair = args.foreground is not None or args.background is not None
    if has_explicit_pair:
        if args.foreground is None or args.background is None:
            raise ValueError("--foreground and --background must be provided together.")
        foreground = _read_image(resolve_repo_path(args.foreground))
        background = _read_image(resolve_repo_path(args.background))
        return args.foreground.stem, foreground, background, mask, soft_mask

    if args.image is None:
        raise ValueError("Provide either image + mask/soft-mask or foreground + background + mask/soft-mask.")

    image = _read_image(resolve_repo_path(args.image))
    split_mask = mask if mask is not None else soft_mask_to_u8(load_soft_mask(None, soft_mask, _to_bgr(image).shape[:2]))
    split = split_foreground_background(image, split_mask)
    return args.image.stem, split.foreground, split.background, mask, soft_mask


def run(args: argparse.Namespace) -> Path:
    if args.dataset_root is not None:
        return run_dataset_batch(args)

    stem, foreground, background, mask, soft_mask = _inputs_from_args(args)
    result = alpha_blend_images(
        foreground=foreground,
        background=background,
        mask=mask,
        soft_mask=soft_mask,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
    )

    args.output_dir = resolve_repo_path(args.output_dir)
    output_dir = _output_dir(args, stem)
    write_image(output_dir / f"{stem}_alpha_blended.png", result.image)
    if args.output_mode == "full":
        write_image(output_dir / f"{stem}_preview.png", make_preview(result))
    if args.output_mode == "full" and args.save_soft_mask:
        write_image(output_dir / f"{stem}_soft_mask.png", soft_mask_to_u8(result.soft_mask))
    return output_dir


def main() -> None:
    output_dir = run(parse_args())
    print(f"[DONE] alpha blending outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
