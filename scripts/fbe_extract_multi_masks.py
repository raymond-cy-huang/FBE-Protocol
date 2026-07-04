#!/usr/bin/env python3
"""Extract masks for all images from configured input/output folders."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fbe_extract_mask import IMAGE_EXTS, REPO_ROOT, extract_image, iter_images

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit("tqdm is required for progress display. Run `bash setup.sh`.") from exc

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to read configs/global_path.yaml. Run `bash setup.sh`.") from exc

CONFIG_PATH = REPO_ROOT / "configs/global_path.yaml"
CONFIG_PREFIX = "fbe_extract_multi_masks_path"
TASK_NAME = Path(__file__).stem


def read_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def profile_keys(config: dict[str, object]) -> list[str]:
    return sorted(k for k in config if k.startswith(CONFIG_PREFIX))


def resolve_repo_path(value: object, default: Path) -> Path:
    path = Path(str(value)) if value else default
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    config = read_config(CONFIG_PATH)
    parser = argparse.ArgumentParser(description="Extract binary foreground masks for an image folder.")
    parser.add_argument("--path-profile", default=f"{CONFIG_PREFIX}00")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-layout", choices=("task", "direct"), default=None)
    parser.add_argument("--output-mode", choices=("full", "mask_only"), default=None)
    parser.add_argument("--bbox-dir", type=Path, default=REPO_ROOT / "models/BBoxMaskPose")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--open", type=int, default=0)
    parser.add_argument("--close", type=int, default=0)
    parser.add_argument("--erode", type=int, default=1)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true", help="Skip images whose output mask already exists.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record per-image errors and continue.")
    parser.add_argument("--error-csv", type=Path, default=None, help="CSV path for per-image extraction errors.")
    parser.add_argument("--recursive", action="store_true", help="Recursively collect images below input-dir.")
    parser.add_argument(
        "--preserve-relative",
        action="store_true",
        help="Preserve each image's relative parent folder below output-dir.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N collected images.")
    parser.add_argument(
        "--variant-px",
        type=int,
        default=5,
        help="Pixel radius for eroded/dilated mask variants. Use -1 to disable variant outputs.",
    )
    args = parser.parse_args()

    section = config.get(args.path_profile, {})
    if not isinstance(section, dict):
        available = ", ".join(profile_keys(config)) or "<none>"
        raise ValueError(f"Config profile must be a YAML object: {args.path_profile}. Available: {available}")

    args.input_dir = args.input_dir or resolve_repo_path(section.get("input_dir"), REPO_ROOT / "images")
    args.output_dir = args.output_dir or resolve_repo_path(section.get("output_dir"), REPO_ROOT / "results")
    args.output_layout = args.output_layout or str(section.get("output_layout", "task"))
    args.output_mode = args.output_mode or str(section.get("output_mode", "full"))
    if args.output_layout not in {"task", "direct"}:
        raise ValueError(f"output_layout must be task or direct: {args.output_layout}")
    if args.output_mode not in {"full", "mask_only"}:
        raise ValueError(f"output_mode must be full or mask_only: {args.output_mode}")
    return args


def output_mask_path(image_path: Path, task_output_dir: Path, output_layout: str) -> Path:
    image_result_dir = task_output_dir / image_path.stem if output_layout == "task" else task_output_dir
    return image_result_dir / f"{image_path.stem}_mask.png"


def collect_images(input_dir: Path, recursive: bool, limit: int | None) -> list[Path]:
    if recursive and input_dir.is_dir():
        images = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        images = iter_images(input_dir)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be greater than 0.")
        images = images[:limit]
    return images


def image_output_dir(image_path: Path, input_dir: Path, task_output_dir: Path, args: argparse.Namespace) -> Path:
    if not args.preserve_relative or not input_dir.is_dir():
        return task_output_dir
    relative_parent = image_path.parent.relative_to(input_dir)
    return task_output_dir / relative_parent


def append_error(path: Path, image_path: Path, exc: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "status", "error"])
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "image_name": image_path.name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def main() -> None:
    args = parse_args()
    task_output_dir = args.output_dir / TASK_NAME if args.output_layout == "task" else args.output_dir
    images = collect_images(args.input_dir, args.recursive, args.limit)
    if not images:
        raise ValueError(f"No images found in: {args.input_dir}")

    print(f"[INFO] Input : {args.input_dir}")
    print(f"[INFO] Output: {task_output_dir}")
    print(f"[INFO] Mode  : {args.output_mode}")
    print(f"[INFO] Recursive: {args.recursive}")
    print(f"[INFO] Preserve relative folders: {args.preserve_relative}")
    print(f"[INFO] Total : {len(images)} images")
    error_csv = args.error_csv or (task_output_dir / "extraction_errors.csv")
    for image_path in tqdm(images, desc="Extracting masks", unit="image"):
        current_output_dir = image_output_dir(image_path, args.input_dir, task_output_dir, args)
        if args.skip_existing and output_mask_path(image_path, current_output_dir, args.output_layout).exists():
            tqdm.write(f"[SKIP] Existing mask -> {image_path.relative_to(args.input_dir) if args.input_dir.is_dir() else image_path.name}")
            continue
        tqdm.write(f"[INFO] Processing {image_path.relative_to(args.input_dir) if args.input_dir.is_dir() else image_path.name}")
        try:
            extract_image(image_path, current_output_dir, args)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            tqdm.write(f"[WARN] Failed {image_path.name}: {type(exc).__name__}: {exc}")
            append_error(error_csv, image_path, exc)


if __name__ == "__main__":
    main()
