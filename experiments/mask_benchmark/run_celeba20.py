#!/usr/bin/env python3
"""Run BBoxMaskPose, SAM, and SAM2 masks on the first 20 CelebAMask images."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mask_benchmark.run import (
    add_label,
    overlay_mask,
    read_image,
    write_image,
)
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


def collect_images(input_dir: Path, limit: int) -> list[Path]:
    images = sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_image_key,
    )
    return images[:limit]


def selected_area_ratio(selection: dict[str, object], image_area: int) -> float:
    if "area_ratio" in selection:
        return float(selection["area_ratio"])
    if "raw_inverted_area" in selection:
        return float(selection["raw_inverted_area"]) / image_area
    if "area" in selection:
        return float(selection["area"]) / image_area
    return -1.0


def selected_area(selection: dict[str, object]) -> object:
    if "area" in selection:
        return selection["area"]
    if "raw_inverted_area" in selection:
        return selection["raw_inverted_area"]
    return ""


def make_comparison(
    image_bgr: np.ndarray,
    m_bbox: np.ndarray,
    m_sam: np.ndarray,
    m_sam2: np.ndarray,
) -> np.ndarray:
    panels = [
        add_label(image_bgr, "input"),
        add_label(overlay_mask(image_bgr, m_bbox, (0, 255, 255)), "M_bbox overlay"),
        add_label(overlay_mask(image_bgr, m_sam, (0, 0, 255)), "M_sam overlay"),
        add_label(overlay_mask(image_bgr, m_sam2, (255, 0, 0)), "M_sam2 overlay"),
        add_label(cv2.cvtColor(m_bbox, cv2.COLOR_GRAY2BGR), "M_bbox"),
        add_label(cv2.cvtColor(m_sam, cv2.COLOR_GRAY2BGR), "M_sam"),
        add_label(cv2.cvtColor(m_sam2, cv2.COLOR_GRAY2BGR), "M_sam2"),
    ]
    h, w = image_bgr.shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    row1 = np.concatenate(panels[:4], axis=1)
    row2 = np.concatenate([add_label(blank, ""), *panels[4:]], axis=1)
    return np.concatenate([row1, row2], axis=0)


def make_boundary_comparison(image_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    out = image_bgr.copy()
    colors = {
        "M_bbox": (0, 255, 255),
        "M_sam": (0, 0, 255),
        "M_sam2": (255, 0, 0),
    }
    for name, mask in masks.items():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colors[name], 2)
    cv2.putText(
        out,
        "M_bbox=yellow M_sam=red M_sam2=blue",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M_bbox, M_sam, and M_sam2 on CelebAMask images.")
    parser.add_argument("--input-dir", type=Path, default=Path("/mnt/d/Dataset/CelebAMask-combined/image"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/Ph.D/01_Experiments_GAN_Inv_Log/2026.03.08_All_Experiments_Start/exper_16_mask_bench_mark"),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-checkpoint", type=Path, default=REPO_ROOT / "models/sam/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=REPO_ROOT / "models/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--prefix-output-names",
        action="store_true",
        help="Prefix generated output filenames with the input image stem.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("fixed_inverted", "face_filter"),
        default="fixed_inverted",
        help="How to select SAM/SAM2 masks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images = collect_images(args.input_dir, args.limit)
    if not images:
        raise ValueError(f"No images found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    for index, image_path in enumerate(images, start=1):
        print(f"[INFO] [{index}/{len(images)}] Processing {image_path}")
        image_bgr = read_image(image_path)
        image_dir = args.output_dir / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        def out_name(name: str) -> str:
            if args.prefix_output_names:
                return f"{image_path.stem}_{name}"
            return name

        write_image(image_dir / out_name("input.png"), image_bgr)

        bbox_result = providers["bbox"].generate(image_path, image_bgr, image_dir)
        sam_result = providers["sam"].generate(image_path, image_bgr, image_dir)
        sam2_result = providers["sam2"].generate(image_path, image_bgr, image_dir)
        sam = sam_result.metadata
        sam2 = sam2_result.metadata
        box = sam["box_xyxy"]
        face_core = sam["face_core_xyxy"]

        write_image(image_dir / out_name("raw_bbox.png"), bbox_result.raw_mask)
        write_image(image_dir / out_name("raw_sam_selected.png"), sam_result.raw_mask)
        write_image(image_dir / out_name("raw_sam2_selected.png"), sam2_result.raw_mask)
        write_image(image_dir / out_name("M_bbox.png"), bbox_result.mask)
        write_image(image_dir / out_name("M_sam.png"), sam_result.mask)
        write_image(image_dir / out_name("M_sam2.png"), sam2_result.mask)
        write_image(image_dir / out_name("overlay_bbox.png"), overlay_mask(image_bgr, bbox_result.mask, (0, 255, 255)))
        write_image(image_dir / out_name("overlay_sam.png"), overlay_mask(image_bgr, sam_result.mask, (0, 0, 255)))
        write_image(image_dir / out_name("overlay_sam2.png"), overlay_mask(image_bgr, sam2_result.mask, (255, 0, 0)))
        write_image(image_dir / out_name("comparison.png"), make_comparison(image_bgr, bbox_result.mask, sam_result.mask, sam2_result.mask))
        write_image(
            image_dir / out_name("boundary_comparison.png"),
            make_boundary_comparison(
                image_bgr,
                {"M_bbox": bbox_result.mask, "M_sam": sam_result.mask, "M_sam2": sam2_result.mask},
            ),
        )

        rows.append(
            {
                "image": image_path.name,
                "box_source": sam["box_source"],
                "box_xyxy": ",".join(str(v) for v in box),
                "selection_mode": args.selection_mode,
                "face_core_source": sam["face_core_source"],
                "face_core_xyxy": ",".join(str(v) for v in face_core),
                "sam_best_idx": sam["best_idx"],
                "sam_best_score": f"{sam['best_score']:.6f}",
                "sam_candidate_count": sam["candidate_count"],
                "sam_selected_inverted": sam.get("inverted", True),
                "sam_selected_area": selected_area(sam),
                "sam_selected_area_ratio": f"{selected_area_ratio(sam, image_bgr.shape[0] * image_bgr.shape[1]):.6f}",
                "sam_face_coverage": f"{sam.get('face_coverage', -1):.6f}",
                "sam_selection_reason": sam.get("selection_reason", "fixed_inverted"),
                "sam_valid_variant_count": sam.get("valid_variant_count", ""),
                "sam2_best_idx": sam2["best_idx"],
                "sam2_best_score": f"{sam2['best_score']:.6f}",
                "sam2_candidate_count": sam2["candidate_count"],
                "sam2_selected_inverted": sam2.get("inverted", True),
                "sam2_selected_area": selected_area(sam2),
                "sam2_selected_area_ratio": f"{selected_area_ratio(sam2, image_bgr.shape[0] * image_bgr.shape[1]):.6f}",
                "sam2_face_coverage": f"{sam2.get('face_coverage', -1):.6f}",
                "sam2_selection_reason": sam2.get("selection_reason", "fixed_inverted"),
                "sam2_valid_variant_count": sam2.get("valid_variant_count", ""),
                "output_dir": str(image_dir),
            }
        )

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Results -> {args.output_dir}")


if __name__ == "__main__":
    main()
