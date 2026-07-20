#!/usr/bin/env python3
"""Generate formal bbox/sam/sam2 mask-only outputs."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fbe_protocol.mask import BBoxMaskProvider, Sam2MaskProvider, SamMaskProvider

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def maybe_reexec_in_conda() -> None:
    env_name = os.environ.get("FBE_CONDA_ENV", "fbe-protocol")
    if os.environ.get("FBE_PROTOCOL_NO_CONDA_REEXEC") == "1":
        return
    if os.environ.get("CONDA_DEFAULT_ENV") == env_name:
        return

    env_python = Path.home() / f"miniconda3/envs/{env_name}/bin/python"
    if not env_python.exists():
        env_python = Path.home() / f"anaconda3/envs/{env_name}/bin/python"
    if not env_python.exists():
        return

    env = {
        **os.environ,
        "FBE_PROTOCOL_NO_CONDA_REEXEC": "1",
        "CONDA_DEFAULT_ENV": env_name,
    }
    os.execve(str(env_python), [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


maybe_reexec_in_conda()


def natural_image_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def collect_images(input_root: Path, limit: int) -> list[Path]:
    for candidate in [input_root / "image", input_root / "gt_image", input_root]:
        if candidate.is_dir():
            input_dir = candidate
            break
    else:
        raise FileNotFoundError(f"Input path not found: {input_root}")

    images = sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_image_key,
    )
    return images[:limit]


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def image_id(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[-1]


def write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mask-only formal outputs for bbox, sam, and sam2.")
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/d/Dataset/CelebAMask-combined"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/Ph.D/01_Experiments_GAN_Inv_Log/2026.03.08_All_Experiments_Start/exper_16_mask_bench_mark/formal"),
    )
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-checkpoint", type=Path, default=REPO_ROOT / "models/sam/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=REPO_ROOT / "models/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--selection-mode", choices=("fixed_inverted", "face_filter"), default="face_filter")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true", help="Regenerate masks even if all outputs already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = collect_images(args.input_root, args.limit)
    if not images:
        raise ValueError(f"No images found under: {args.input_root}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ["bbox", "sam", "sam2", "_work"]:
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)

    providers = {
        "bbox": BBoxMaskProvider(),
        "sam": SamMaskProvider(
            model_type=args.sam_model_type,
            checkpoint=args.sam_checkpoint,
            device=device,
            selection_mode=args.selection_mode,
        ),
        "sam2": Sam2MaskProvider(
            model_cfg=args.sam2_model_cfg,
            checkpoint=args.sam2_checkpoint,
            device=device,
            selection_mode=args.selection_mode,
        ),
    }

    rows = []
    failed_rows = []
    summary_fields = ["image", "status", "failed_methods", "bbox", "sam", "sam2"]
    failed_fields = ["image", "image_id", "method", "error_type", "error"]

    def flush_progress() -> None:
        if rows:
            with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(rows)
        if failed_rows:
            with (args.output_dir / "failed.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=failed_fields)
                writer.writeheader()
                writer.writerows(failed_rows)

    for index, image_path in enumerate(images, start=1):
        stem = image_path.stem
        outputs = {
            "bbox": args.output_dir / "bbox" / f"{stem}_M_bbox.png",
            "sam": args.output_dir / "sam" / f"{stem}_M_sam.png",
            "sam2": args.output_dir / "sam2" / f"{stem}_M_sam2.png",
        }
        if not args.no_resume and all(path.exists() for path in outputs.values()):
            print(f"[SKIP] [{index}/{len(images)}] {image_path.name}")
            rows.append(
                {
                    "image": image_path.name,
                    "status": "skipped",
                    "failed_methods": "",
                    **{k: str(v) for k, v in outputs.items()},
                }
            )
            continue
        if not args.no_resume and any(path.exists() for path in outputs.values()):
            for path in outputs.values():
                if path.exists():
                    path.unlink()

        print(f"[INFO] [{index}/{len(images)}] {image_path.name}")
        image_bgr = read_image(image_path)
        work_dir = args.output_dir / "_work" / stem
        pending_dir = args.output_dir / "_pending" / stem
        work_dir.mkdir(parents=True, exist_ok=True)
        if pending_dir.exists():
            shutil.rmtree(pending_dir)
        pending_dir.mkdir(parents=True, exist_ok=True)

        failed_methods = []
        pending_outputs = {}
        for key, provider in providers.items():
            try:
                result = provider.generate(image_path, image_bgr, work_dir)
                pending_outputs[key] = pending_dir / outputs[key].name
                write_image(pending_outputs[key], result.mask)
            except Exception as exc:  # noqa: BLE001 - batch generation should keep going per image.
                failed_methods.append(key)
                failed_rows.append(
                    {
                        "image": image_path.name,
                        "image_id": image_id(image_path),
                        "method": key,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"[ERROR] [{index}/{len(images)}] {image_path.name} {key}: {type(exc).__name__}: {exc}")

        status = "failed" if failed_methods else "done"
        if failed_methods:
            for path in [*outputs.values(), *pending_outputs.values()]:
                if path.exists():
                    path.unlink()
            if pending_dir.exists():
                shutil.rmtree(pending_dir)
        else:
            for key, pending_path in pending_outputs.items():
                if pending_path == outputs[key]:
                    continue
                outputs[key].parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pending_path), str(outputs[key]))
            if pending_dir.exists():
                shutil.rmtree(pending_dir)

        rows.append(
            {
                "image": image_path.name,
                "status": status,
                "failed_methods": ",".join(failed_methods),
                **{k: str(v) for k, v in outputs.items()},
            }
        )

        if index % 25 == 0 or index == len(images):
            flush_progress()
        else:
            flush_progress()

    flush_progress()
    print(f"[DONE] Results -> {args.output_dir}")


if __name__ == "__main__":
    main()
