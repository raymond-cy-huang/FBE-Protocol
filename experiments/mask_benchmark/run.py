#!/usr/bin/env python3
"""Generate visual mask comparisons for BBoxMaskPose, SAM, and SAM2."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fbe_protocol.mask import MaskConfig, extract

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


if Path(sys.argv[0]).resolve() == Path(__file__).resolve():
    maybe_reexec_in_conda()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def iter_images(input_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        images = images[:limit]
    return images


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, int(round(x0))))
    y0 = max(0, min(height - 1, int(round(y0))))
    x1 = max(x0 + 1, min(width, int(round(x1))))
    y1 = max(y0 + 1, min(height, int(round(y1))))
    return x0, y0, x1, y1


def expand_box(
    box_xywh: tuple[int, int, int, int],
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    up_ratio: float,
    down_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = box_xywh
    cx = x + w / 2
    new_w = w * scale_x
    x0 = cx - new_w / 2
    x1 = cx + new_w / 2
    y0 = y - h * up_ratio
    y1 = y + h * down_ratio
    return clamp_box((x0, y0, x1, y1), width, height)


def full_image_box(image_bgr: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    height, width = image_bgr.shape[:2]
    return (0, 0, width, height), "full_image_box"


def detect_person_box(image_bgr: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    height, width = image_bgr.shape[:2]

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(
        image_bgr,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
    )
    if len(boxes) > 0:
        idx = int(np.argmax(np.asarray(weights).reshape(-1)))
        x, y, w, h = [int(v) for v in boxes[idx]]
        return clamp_box((x, y, x + w, y + h), width, height), "opencv_hog_person"

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_dir = Path(cv2.data.haarcascades)
    for cascade_name, label, params in [
        ("haarcascade_fullbody.xml", "opencv_haar_fullbody", (1.05, 3)),
        ("haarcascade_upperbody.xml", "opencv_haar_upperbody", (1.05, 3)),
    ]:
        cascade = cv2.CascadeClassifier(str(cascade_dir / cascade_name))
        detections = cascade.detectMultiScale(gray, scaleFactor=params[0], minNeighbors=params[1])
        if len(detections) > 0:
            x, y, w, h = max(detections, key=lambda b: int(b[2]) * int(b[3]))
            return expand_box((int(x), int(y), int(w), int(h)), width, height, 1.25, 1.20, 0.05, 1.05), label

    face_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        return expand_box((int(x), int(y), int(w), int(h)), width, height, 3.6, 5.8, 1.0, 5.0), "opencv_haar_face_expanded"

    margin_x = int(width * 0.08)
    return (margin_x, 0, width - margin_x, height), "opencv_fallback_center"


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    mask_bool = mask > 0
    out = image_bgr.copy()
    color_layer = np.zeros_like(out)
    color_layer[:, :] = color
    out[mask_bool] = cv2.addWeighted(out, 0.45, color_layer, 0.55, 0)[mask_bool]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def draw_box(image_bgr: np.ndarray, box: tuple[int, int, int, int], source: str) -> np.ndarray:
    out = image_bgr.copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 3)
    cv2.putText(out, source, (x0, max(22, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
    return out


def add_label(image_bgr: np.ndarray, label: str) -> np.ndarray:
    out = image_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_like(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if image.shape[1] > size[0] else cv2.INTER_LINEAR)


def make_method_panel(
    image_bgr: np.ndarray,
    prompt_overlay: np.ndarray,
    raw_mask: np.ndarray,
    norm_mask: np.ndarray,
    label: str,
    prompt_label: str,
) -> np.ndarray:
    width = image_bgr.shape[1]
    height = image_bgr.shape[0]
    raw_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    norm_bgr = cv2.cvtColor(norm_mask, cv2.COLOR_GRAY2BGR)
    overlay = overlay_mask(image_bgr, norm_mask, (0, 255, 0))
    cells = [
        add_label(prompt_overlay, f"{label}: {prompt_label}"),
        add_label(overlay, f"{label}: overlay"),
        add_label(raw_bgr, f"{label}: raw mask"),
        add_label(norm_bgr, f"{label}: norm mask"),
    ]
    return np.concatenate([resize_like(cell, (width, height)) for cell in cells], axis=1)


def run_bbox_mask_pose(image_path: Path, output_dir: Path) -> Path:
    script = REPO_ROOT / "models/BBoxMaskPose/create_raw_mask.py"
    command = [
        sys.executable,
        str(script),
        str(image_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--python",
        sys.executable,
    ]
    subprocess.run(command, cwd=str(REPO_ROOT / "models/BBoxMaskPose"), check=True)
    raw_path = output_dir / f"{image_path.stem}_mask.jpg"
    if not raw_path.exists():
        raise FileNotFoundError(f"BBoxMaskPose did not produce: {raw_path}")
    return raw_path


def load_sam(model_type: str, checkpoint: Path, device: str):
    sys.path.insert(0, str(REPO_ROOT / "models/sam"))
    from segment_anything import SamPredictor, sam_model_registry

    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    return SamPredictor(model)


def load_sam2(model_cfg: str, checkpoint: Path, device: str):
    sys.path.insert(0, str(REPO_ROOT / "models/sam2"))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(model_cfg, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


def predict_sam_mask(predictor, image_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.asarray(box, dtype=np.float32)[None, :],
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return (masks[best].astype(np.uint8) * 255)


def norm_mask(image_bgr: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    result = extract(
        image=image_bgr,
        mask=raw_mask,
        config=MaskConfig(variant_erode_iter=-1, variant_dilate_iter=-1),
    )
    return result.mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BBoxMaskPose, SAM, and SAM2 person masks.")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "images")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/mask_benchmark")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-checkpoint", type=Path, default=REPO_ROOT / "models/sam/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=REPO_ROOT / "models/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--prompt-mode",
        choices=("opencv_box", "full_box"),
        default="opencv_box",
        help="Prompt strategy for SAM/SAM2.",
    )
    parser.add_argument(
        "--opencv-box-only",
        action="store_true",
        help="Only write OpenCV prompt box overlays and summary.csv.",
    )
    parser.add_argument(
        "--invert-sam-masks",
        action="store_true",
        help="Invert SAM/SAM2 raw masks before normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = iter_images(args.input_dir, args.limit)
    if not images:
        raise ValueError(f"No images found: {args.input_dir}")

    if args.opencv_box_only:
        sam_predictor = None
        sam2_predictor = None
    else:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam_predictor = load_sam(args.sam_model_type, args.sam_checkpoint, device)
        sam2_predictor = load_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device)

    rows = []
    for image_path in images:
        print(f"[INFO] Processing {image_path}")
        image_bgr = read_image(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_dir = args.output_dir / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        if args.prompt_mode == "full_box":
            box, box_source = full_image_box(image_bgr)
        else:
            box, box_source = detect_person_box(image_bgr)
        box_overlay = draw_box(image_bgr, box, box_source)
        box_filename = "input_full_box.png" if args.prompt_mode == "full_box" else "input_opencv_box.png"
        write_image(image_dir / box_filename, box_overlay)

        if args.opencv_box_only:
            rows.append(
                {
                    "image": image_path.name,
                    "box_source": box_source,
                    "box_xyxy": ",".join(str(v) for v in box),
                    "output_dir": str(image_dir.relative_to(REPO_ROOT)),
                }
            )
            continue

        raw_bbmp_path = run_bbox_mask_pose(image_path, image_dir)
        raw_bbmp = cv2.imread(str(raw_bbmp_path), cv2.IMREAD_GRAYSCALE)
        if raw_bbmp is None:
            raise ValueError(f"Cannot read BBoxMaskPose raw mask: {raw_bbmp_path}")
        raw_sam = predict_sam_mask(sam_predictor, image_rgb, box)
        raw_sam2 = predict_sam_mask(sam2_predictor, image_rgb, box)
        if args.invert_sam_masks:
            raw_sam = cv2.bitwise_not(raw_sam)
            raw_sam2 = cv2.bitwise_not(raw_sam2)

        masks = {
            "bbmp": (raw_bbmp, norm_mask(image_bgr, raw_bbmp)),
            "sam": (raw_sam, norm_mask(image_bgr, raw_sam)),
            "sam2": (raw_sam2, norm_mask(image_bgr, raw_sam2)),
        }

        for name, (raw, norm) in masks.items():
            write_image(image_dir / f"raw_{name}.png", raw)
            write_image(image_dir / f"Mask_{name}.png", norm)

        panels = []
        for label, key, prompt_overlay, prompt_label in [
            ("BBoxMaskPose", "bbmp", image_bgr, "input"),
            ("SAM", "sam", box_overlay, f"input + {box_source}"),
            ("SAM2", "sam2", box_overlay, f"input + {box_source}"),
        ]:
            raw, norm = masks[key]
            panels.append(make_method_panel(image_bgr, prompt_overlay, raw, norm, label, prompt_label))
        comparison = np.concatenate(panels, axis=0)
        write_image(image_dir / "comparison.png", comparison)

        edge_compare = image_bgr.copy()
        for key, color in [("bbmp", (0, 255, 255)), ("sam", (0, 0, 255)), ("sam2", (255, 0, 0))]:
            contours, _ = cv2.findContours(masks[key][1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edge_compare, contours, -1, color, 2)
        cv2.putText(edge_compare, "BBMP=yellow SAM=red SAM2=blue", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        write_image(image_dir / "boundary_comparison.png", edge_compare)

        rows.append(
            {
                "image": image_path.name,
                "box_source": box_source,
                "box_xyxy": ",".join(str(v) for v in box),
                "output_dir": str(image_dir.relative_to(REPO_ROOT)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "box_source", "box_xyxy", "output_dir"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] Results -> {args.output_dir}")


if __name__ == "__main__":
    main()
