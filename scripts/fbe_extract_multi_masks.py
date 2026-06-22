#!/usr/bin/env python3
"""Extract masks for all images from configured input/output folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from fbe_extract_mask import REPO_ROOT, extract_image, iter_images

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


def main() -> None:
    args = parse_args()
    task_output_dir = args.output_dir / TASK_NAME if args.output_layout == "task" else args.output_dir
    images = iter_images(args.input_dir)
    if not images:
        raise ValueError(f"No images found in: {args.input_dir}")

    print(f"[INFO] Input : {args.input_dir}")
    print(f"[INFO] Output: {task_output_dir}")
    print(f"[INFO] Mode  : {args.output_mode}")
    print(f"[INFO] Total : {len(images)} images")
    for image_path in tqdm(images, desc="Extracting masks", unit="image"):
        tqdm.write(f"[INFO] Processing {image_path.name}")
        extract_image(image_path, task_output_dir, args)


if __name__ == "__main__":
    main()
