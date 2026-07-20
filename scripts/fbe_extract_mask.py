#!/usr/bin/env python3
"""Extract binary foreground masks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
CONDA_ENV_NAME = os.environ.get("FBE_CONDA_ENV", "fbe-protocol")


def maybe_reexec_in_conda() -> None:
    if os.environ.get("FBE_PROTOCOL_NO_CONDA_REEXEC") == "1":
        return
    if os.environ.get("CONDA_DEFAULT_ENV") == CONDA_ENV_NAME:
        return

    env_python_candidates = [
        Path.home() / f"miniconda3/envs/{CONDA_ENV_NAME}/bin/python",
        Path.home() / f"anaconda3/envs/{CONDA_ENV_NAME}/bin/python",
    ]
    env_python = next((p for p in env_python_candidates if p.exists()), None)
    if env_python is not None:
        env = {
            **os.environ,
            "FBE_PROTOCOL_NO_CONDA_REEXEC": "1",
            "CONDA_DEFAULT_ENV": CONDA_ENV_NAME,
        }
        os.execve(str(env_python), [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)

    conda_exe = os.environ.get("CONDA_EXE")
    candidates = [
        Path(conda_exe) if conda_exe else None,
        Path.home() / "miniconda3/bin/conda",
        Path.home() / "anaconda3/bin/conda",
    ]
    conda_path = next((p for p in candidates if p is not None and p.exists()), None)
    if conda_path is None:
        return

    env = {**os.environ, "FBE_PROTOCOL_NO_CONDA_REEXEC": "1"}
    cmd = [str(conda_path), "run", "-n", CONDA_ENV_NAME, "python", str(Path(__file__).resolve()), *sys.argv[1:]]
    completed = subprocess.run(cmd, env=env)
    raise SystemExit(completed.returncode)


if Path(sys.argv[0]).resolve() == Path(__file__).resolve():
    maybe_reexec_in_conda()

import cv2
import numpy as np

from fbe_protocol.mask import MaskConfig, extract

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

SCRIPT_NAME = Path(__file__).stem
DEFAULT_RESULT_DIR = REPO_ROOT / "results" / SCRIPT_NAME
DEFAULT_MASK_MODEL_CONFIG = REPO_ROOT / "configs/mask_model.yaml"


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _imread_any(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def run_bbox_mask_pose(image_path: Path, image_result_dir: Path, bbox_dir: Path) -> Path:
    image_result_dir.mkdir(parents=True, exist_ok=True)
    script = bbox_dir / "create_raw_mask.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing BBoxMaskPose mask creator: {script}")

    command = [
        sys.executable,
        str(script),
        str(image_path.resolve()),
        "--output-dir",
        str(image_result_dir.resolve()),
        "--python",
        sys.executable,
    ]
    try:
        subprocess.run(
            command,
            cwd=str(bbox_dir),
            env=dict(os.environ),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "BBoxMaskPose failed while creating the mask. "
            "If the error mentions missing mmcv/mmdet/mmpretrain, run `bash setup.sh` "
            "or install them manually with "
            "`conda run -n fbe-protocol mim install mmengine 'mmcv==2.1.0' "
            "'mmdet==3.3.0' 'mmpretrain==1.2.0'`."
        ) from exc

    mask_path = image_result_dir / f"{image_path.stem}_mask.jpg"
    if not mask_path.exists():
        raise FileNotFoundError(f"Expected mask was not produced: {mask_path}")
    return mask_path


def read_mask_model_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"active_model": "bbox_mask_pose", "models": {}}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read configs/mask_model.yaml. Run `bash setup.sh`.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Mask model config must be a YAML object: {path}")
    return data


def resolve_repo_path(value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def selected_mask_model(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    config = read_mask_model_config(args.mask_model_config)
    model_name = args.mask_model or str(config.get("active_model", "bbox_mask_pose"))
    models = config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"`models` must be a YAML object in {args.mask_model_config}")

    section = models.get(model_name, {})
    if not isinstance(section, dict):
        available = ", ".join(sorted(str(k) for k in models)) or "<none>"
        raise ValueError(f"Unknown mask model `{model_name}`. Available: {available}")
    return model_name, section


def _torch_device(section: dict[str, object]) -> str:
    if section.get("device"):
        return str(section["device"])

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_rgb_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _select_largest_mask(masks: list[dict[str, object]], backend: str) -> np.ndarray:
    if not masks:
        raise RuntimeError(f"{backend} did not produce any masks.")
    largest = max(masks, key=lambda item: int(item.get("area", 0)))
    segmentation = largest.get("segmentation")
    if segmentation is None:
        raise RuntimeError(f"{backend} produced a mask without `segmentation`.")
    return np.asarray(segmentation).astype(np.uint8) * 255


def _write_generated_mask(mask: np.ndarray, image_path: Path, image_result_dir: Path, suffix: str) -> Path:
    image_result_dir.mkdir(parents=True, exist_ok=True)
    mask_path = image_result_dir / f"{image_path.stem}_{suffix}.png"
    write_image(mask_path, mask)
    return mask_path


def run_sam(image_path: Path, image_result_dir: Path, section: dict[str, object]) -> Path:
    root = resolve_repo_path(section.get("root", "models/sam"))
    checkpoint = resolve_repo_path(section.get("checkpoint", "models/sam/checkpoints/sam_vit_b_01ec64.pth"))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint}. Run `bash setup.sh`.")

    sys.path.insert(0, str(root))
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    model_type = str(section.get("model_type", "vit_b"))
    generator_args = section.get("generator_args", {})
    if not isinstance(generator_args, dict):
        raise ValueError("SAM generator_args must be a YAML object.")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=_torch_device(section))
    mask_generator = SamAutomaticMaskGenerator(sam, **generator_args)
    mask = _select_largest_mask(mask_generator.generate(_read_rgb_image(image_path)), "SAM")
    return _write_generated_mask(mask, image_path, image_result_dir, "sam_raw_mask")


def run_sam2(image_path: Path, image_result_dir: Path, section: dict[str, object]) -> Path:
    root = resolve_repo_path(section.get("root", "models/sam2"))
    checkpoint = resolve_repo_path(section.get("checkpoint", "models/sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {checkpoint}. Run `bash setup.sh`.")

    sys.path.insert(0, str(root))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    generator_args = section.get("generator_args", {})
    if not isinstance(generator_args, dict):
        raise ValueError("SAM2 generator_args must be a YAML object.")

    model = build_sam2(
        str(section.get("model_cfg", "configs/sam2.1/sam2.1_hiera_t.yaml")),
        str(checkpoint),
        device=_torch_device(section),
    )
    mask_generator = SAM2AutomaticMaskGenerator(model, **generator_args)
    mask = _select_largest_mask(mask_generator.generate(_read_rgb_image(image_path)), "SAM2")
    return _write_generated_mask(mask, image_path, image_result_dir, "sam2_raw_mask")


def run_raw_mask_generator(image_path: Path, image_result_dir: Path, args: argparse.Namespace) -> Path:
    model_name, section = selected_mask_model(args)
    model_type = str(section.get("type", model_name))
    if model_type == "bbox_mask_pose":
        bbox_dir = resolve_repo_path(section.get("root", args.bbox_dir))
        if args.bbox_dir != REPO_ROOT / "models/BBoxMaskPose":
            bbox_dir = args.bbox_dir
        return run_bbox_mask_pose(image_path, image_result_dir, bbox_dir)
    if model_type == "sam":
        return run_sam(image_path, image_result_dir, section)
    if model_type == "sam2":
        return run_sam2(image_path, image_result_dir, section)
    raise ValueError(f"Unsupported mask model type: {model_type}")


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Failed to write image: {path}")


def make_combine(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3]
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if image_bgr.shape[:2] != mask_bgr.shape[:2]:
        mask_bgr = cv2.resize(mask_bgr, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([image_bgr, mask_bgr], axis=1)


def build_config(args: argparse.Namespace) -> MaskConfig:
    return MaskConfig(
        invert=args.invert,
        boundary_open_kernel=args.open,
        boundary_close_kernel=args.close,
        boundary_erode_iter=args.erode,
        boundary_dilate_iter=args.dilate,
        variant_erode_iter=args.variant_px,
        variant_dilate_iter=args.variant_px,
    )


def extract_from_mask_file(
    mask_path: Path,
    image_result_dir: Path,
    args: argparse.Namespace,
    image_path: Path,
) -> None:
    mask_input = _imread_any(mask_path)
    image = _imread_any(image_path)

    result = extract(
        image=image,
        mask=mask_input,
        config=build_config(args),
    )

    stem = image_path.stem
    output_path = image_result_dir / f"{stem}_mask.png"
    combine_path = image_result_dir / f"{stem}_combine.png"
    write_image(output_path, result.mask)
    if args.output_mode == "full":
        write_image(combine_path, make_combine(image, result.mask))
        for variant_name, variant_mask in result.mask_variants.items():
            if variant_name == "original":
                continue
            write_image(image_result_dir / f"{stem}_mask_{variant_name}.png", variant_mask)

    print(f"[OK] {image_path.stem} -> {output_path} unique={np.unique(result.mask).tolist()}")


def extract_image(image_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    output_layout = getattr(args, "output_layout", "task")
    image_result_dir = output_dir / image_path.stem if output_layout == "task" else output_dir
    generated_mask = run_raw_mask_generator(image_path, image_result_dir, args)
    extract_from_mask_file(generated_mask, image_result_dir, args, image_path=image_path)
    generated_mask.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract binary foreground masks."
    )
    parser.add_argument("input", nargs="?", type=Path, default=REPO_ROOT / "images", help="Input image file.")
    parser.add_argument("--mask", type=Path, default=None, help="Skip BBoxMaskPose and extract from this mask file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-layout", choices=("task", "direct"), default="task")
    parser.add_argument("--output-mode", choices=("full", "mask_only"), default="full")
    parser.add_argument("--bbox-dir", type=Path, default=REPO_ROOT / "models/BBoxMaskPose")
    parser.add_argument("--mask-model-config", type=Path, default=DEFAULT_MASK_MODEL_CONFIG)
    parser.add_argument("--mask-model", choices=("bbox_mask_pose", "sam", "sam2"), default=None)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mask is not None:
        if not args.input.is_file():
            raise ValueError("--mask requires input to be an image file so the combine image can be created.")
        image_result_dir = args.output_dir / args.input.stem if args.output_layout == "task" else args.output_dir
        extract_from_mask_file(args.mask, image_result_dir, args, image_path=args.input)
        return

    if not args.input.is_file():
        raise ValueError(f"Input must be an image file for fbe_extract_mask.py: {args.input}")
    extract_image(args.input, args.output_dir, args)


if __name__ == "__main__":
    main()
