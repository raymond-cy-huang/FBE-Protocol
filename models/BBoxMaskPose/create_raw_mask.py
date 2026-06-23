#!/usr/bin/env python3
"""Create a BBoxMaskPose mask for one image without a shell temp folder."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def max_iter_file(directory: Path, pattern: str) -> Path:
    selected: Path | None = None
    index = 1
    while True:
        candidate = directory / pattern.format(iter=index)
        if not candidate.exists():
            break
        selected = candidate
        index += 1
    if selected is None:
        raise FileNotFoundError(f"No iteration output found in: {directory}")
    return selected


def list_debug_files(directory: Path) -> str:
    if not directory.exists():
        return "  <missing directory>"
    files = sorted(p.name for p in directory.iterdir() if p.is_file())
    return "\n".join(f"  {name}" for name in files) if files else "  <no files>"


def run_bmp_demo(repo_dir: Path, python_bin: Path, image_path: Path) -> None:
    project_root = repo_dir.parent.parent
    env = {
        **os.environ,
        "BMP_BG_MODE": "black",
        "PYTHONPATH": f"{project_root}:{repo_dir}:{os.environ.get('PYTHONPATH', '')}",
    }
    command = [
        str(python_bin),
        str(repo_dir / "demo/bmp_demo.py"),
        str(repo_dir / "configs/bmp_D3.yaml"),
        str(image_path),
    ]
    subprocess.run(command, cwd=str(repo_dir), env=env, check=True)


def clear_demo_outputs(repo_dir: Path) -> None:
    shutil.rmtree(repo_dir / "demo/outputs", ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a BBoxMaskPose mask for one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("BMP_OUTPUT_DIR", "raw_mask")))
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--keep-outputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    env_name = os.environ.get("BMP_CONDA_ENV", "fbe-protocol")
    python_bin = args.python or Path(os.environ.get("BMP_PYTHON", Path.home() / f"miniconda3/envs/{env_name}/bin/python"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    print(f"[INFO] Processing image -> {image_path}")

    clear_demo_outputs(repo_dir)
    raw_output_dir = repo_dir / "demo/outputs" / stem
    run_bmp_demo(repo_dir, python_bin, image_path)
    try:
        raw_mask = max_iter_file(raw_output_dir, f"{stem}_iter{{iter}}_Mask_white_on_black.jpg")
    except FileNotFoundError as exc:
        print(f"[ERROR] raw mask not found: {exc}", file=sys.stderr)
        print(f"[DEBUG] Existing files under {raw_output_dir}:", file=sys.stderr)
        print(list_debug_files(raw_output_dir), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"[INFO] Using highest iter raw mask -> {raw_mask}")

    output_mask = output_dir / f"{stem}_mask.jpg"
    shutil.move(str(raw_mask), str(output_mask))
    print(f"[DONE] final mask -> {output_mask}")

    if not args.keep_outputs:
        clear_demo_outputs(repo_dir)


if __name__ == "__main__":
    main()
