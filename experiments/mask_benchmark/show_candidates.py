#!/usr/bin/env python3
"""Show SAM and SAM2 candidate masks for quick visual inspection."""

from __future__ import annotations

import argparse
import csv
import math
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
    detect_person_box,
    draw_box,
    iter_images,
    overlay_mask,
    read_image,
    write_image,
)


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


def load_sam(model_type: str, checkpoint: Path, device: str):
    sys.path.insert(0, str(REPO_ROOT / "models/sam"))
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    return SamPredictor(model), SamAutomaticMaskGenerator(model, points_per_side=16)


def load_sam2(model_cfg: str, checkpoint: Path, device: str):
    sys.path.insert(0, str(REPO_ROOT / "models/sam2"))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(model_cfg, str(checkpoint), device=device)
    return SAM2ImagePredictor(model), SAM2AutomaticMaskGenerator(model, points_per_side=16)


def prompt_candidates(predictor, image_rgb: np.ndarray, box: tuple[int, int, int, int]) -> list[dict[str, object]]:
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.asarray(box, dtype=np.float32)[None, :],
        multimask_output=True,
    )
    candidates = []
    for idx, (mask, score) in enumerate(zip(masks, scores)):
        mask_u8 = mask.astype(np.uint8) * 255
        candidates.append(
            {
                "idx": idx,
                "mask": mask_u8,
                "score": float(score),
                "area": int(mask_u8.sum() // 255),
            }
        )
    return candidates


def invert_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    inverted = []
    for cand in candidates:
        next_cand = dict(cand)
        mask = np.asarray(next_cand["mask"], dtype=np.uint8)
        next_cand["mask"] = cv2.bitwise_not(mask)
        next_cand["area"] = int(next_cand["mask"].sum() // 255)
        inverted.append(next_cand)
    return inverted


def automatic_candidates(generator, image_rgb: np.ndarray, max_candidates: int) -> list[dict[str, object]]:
    annotations = generator.generate(image_rgb)
    annotations = sorted(
        annotations,
        key=lambda item: (float(item.get("predicted_iou", 0.0)), float(item.get("stability_score", 0.0))),
        reverse=True,
    )
    if max_candidates > 0:
        annotations = annotations[:max_candidates]
    candidates = []
    for idx, ann in enumerate(annotations):
        mask_u8 = np.asarray(ann["segmentation"]).astype(np.uint8) * 255
        candidates.append(
            {
                "idx": idx,
                "mask": mask_u8,
                "score": float(ann.get("predicted_iou", 0.0)),
                "stability": float(ann.get("stability_score", 0.0)),
                "area": int(ann.get("area", mask_u8.sum() // 255)),
                "bbox": ann.get("bbox", []),
            }
        )
    return candidates


def make_gallery(
    image_bgr: np.ndarray,
    candidates: list[dict[str, object]],
    title: str,
    cols: int = 4,
) -> np.ndarray:
    if not candidates:
        blank = np.zeros_like(image_bgr)
        return add_label(blank, f"{title}: no candidates")

    h, w = image_bgr.shape[:2]
    thumb_w = min(256, w)
    thumb_h = max(1, int(h * thumb_w / w))
    rows = math.ceil(len(candidates) / cols)
    cells = []
    for cand in candidates:
        mask = cand["mask"]
        panel = overlay_mask(image_bgr, mask, (0, 255, 0))
        label = f"{title} #{cand['idx']} score={cand['score']:.3f} area={cand['area']}"
        if "stability" in cand:
            label += f" stab={cand['stability']:.3f}"
        panel = add_label(panel, label)
        cells.append(cv2.resize(panel, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))

    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    while len(cells) < rows * cols:
        cells.append(blank.copy())
    row_imgs = [np.concatenate(cells[i * cols : (i + 1) * cols], axis=1) for i in range(rows)]
    return np.concatenate(row_imgs, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show SAM/SAM2 candidate masks.")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "images")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/mask_benchmark_candidates")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-auto-candidates", type=int, default=24, help="Use 0 to write every automatic candidate.")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-checkpoint", type=Path, default=REPO_ROOT / "models/sam/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=REPO_ROOT / "models/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--invert-masks", action="store_true", help="Invert SAM/SAM2 masks before drawing galleries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam_predictor, sam_generator = load_sam(args.sam_model_type, args.sam_checkpoint, device)
    sam2_predictor, sam2_generator = load_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device)

    rows = []
    for image_path in iter_images(args.input_dir, args.limit):
        print(f"[INFO] Candidates for {image_path}")
        image_bgr = read_image(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_dir = args.output_dir / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        box, box_source = detect_person_box(image_bgr)
        write_image(image_dir / "input_opencv_box.png", draw_box(image_bgr, box, box_source))

        sam_prompt = prompt_candidates(sam_predictor, image_rgb, box)
        sam2_prompt = prompt_candidates(sam2_predictor, image_rgb, box)
        sam_auto = automatic_candidates(sam_generator, image_rgb, args.max_auto_candidates)
        sam2_auto = automatic_candidates(sam2_generator, image_rgb, args.max_auto_candidates)
        if args.invert_masks:
            sam_prompt = invert_candidates(sam_prompt)
            sam2_prompt = invert_candidates(sam2_prompt)
            sam_auto = invert_candidates(sam_auto)
            sam2_auto = invert_candidates(sam2_auto)

        suffix = "_inverted" if args.invert_masks else ""
        write_image(image_dir / f"sam_prompt_candidates{suffix}.png", make_gallery(image_bgr, sam_prompt, "SAM prompt"))
        write_image(image_dir / f"sam2_prompt_candidates{suffix}.png", make_gallery(image_bgr, sam2_prompt, "SAM2 prompt"))
        write_image(image_dir / f"sam_auto_candidates{suffix}.png", make_gallery(image_bgr, sam_auto, "SAM auto"))
        write_image(image_dir / f"sam2_auto_candidates{suffix}.png", make_gallery(image_bgr, sam2_auto, "SAM2 auto"))

        rows.append(
            {
                "image": image_path.name,
                "box_source": box_source,
                "box_xyxy": ",".join(str(v) for v in box),
                "sam_prompt_count": len(sam_prompt),
                "sam2_prompt_count": len(sam2_prompt),
                "sam_auto_count_written": len(sam_auto),
                "sam2_auto_count_written": len(sam2_auto),
                "output_dir": str(image_dir.relative_to(REPO_ROOT)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "box_source",
                "box_xyxy",
                "sam_prompt_count",
                "sam2_prompt_count",
                "sam_auto_count_written",
                "sam2_auto_count_written",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] Candidate galleries -> {args.output_dir}")


if __name__ == "__main__":
    main()
