from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .config import MaskConfig
from .extract import extract

REPO_ROOT = Path(__file__).resolve().parents[2]
SelectionMode = Literal["fixed_inverted", "face_filter"]


@dataclass
class GeneratedMask:
    name: str
    raw_mask: np.ndarray
    mask: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


def full_image_box(image_bgr: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    height, width = image_bgr.shape[:2]
    return (0, 0, width, height), "full_image_box"


def refine_raw_mask(image_bgr: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    result = extract(
        image=image_bgr,
        mask=raw_mask,
        config=MaskConfig(variant_erode_iter=-1, variant_dilate_iter=-1),
    )
    return result.mask


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


def run_bbox_mask_pose(image_path: Path, output_dir: Path, python: str | None = None) -> Path:
    script = REPO_ROOT / "models/BBoxMaskPose/create_raw_mask.py"
    python = python or sys.executable
    command = [
        python,
        str(script),
        str(image_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--python",
        python,
    ]
    subprocess.run(command, cwd=str(REPO_ROOT / "models/BBoxMaskPose"), check=True)
    raw_path = output_dir / f"{image_path.stem}_mask.jpg"
    if not raw_path.exists():
        raise FileNotFoundError(f"BBoxMaskPose did not produce: {raw_path}")
    return raw_path


def detect_face_core(image_bgr: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        x0 = int(w * 0.30)
        x1 = int(w * 0.70)
        y0 = int(h * 0.18)
        y1 = int(h * 0.58)
        return (x0, y0, x1, y1), "fallback_center_face_core"

    x, y, fw, fh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    x0 = int(x + fw * 0.20)
    x1 = int(x + fw * 0.80)
    y0 = int(y + fh * 0.22)
    y1 = int(y + fh * 0.78)
    return (x0, y0, x1, y1), "opencv_face_core"


def box_coverage(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float((region > 0).mean())


def select_fixed_inverted_mask(
    predictor,
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
) -> dict[str, object]:
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.asarray(box, dtype=np.float32)[None, :],
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    raw = masks[best_idx].astype(np.uint8) * 255
    raw_inverted = cv2.bitwise_not(raw)
    return {
        "best_idx": best_idx,
        "best_score": float(scores[best_idx]),
        "mask": raw_inverted,
        "raw": raw,
        "raw_inverted": raw_inverted,
        "raw_area": int(raw.sum() // 255),
        "raw_inverted_area": int(raw_inverted.sum() // 255),
        "candidate_count": int(len(scores)),
        "inverted": True,
        "selection_reason": "fixed_inverted",
    }


def select_face_filtered_mask(
    predictor,
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    face_core: tuple[int, int, int, int],
) -> dict[str, object]:
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.asarray(box, dtype=np.float32)[None, :],
        multimask_output=True,
    )
    h, w = image_rgb.shape[:2]
    image_area = h * w
    candidates = []
    for idx, (mask_bool, score) in enumerate(zip(masks, scores)):
        raw = mask_bool.astype(np.uint8) * 255
        for inverted, candidate_mask in [(False, raw), (True, cv2.bitwise_not(raw))]:
            area = int(candidate_mask.sum() // 255)
            area_ratio = area / image_area
            face_coverage = box_coverage(candidate_mask, face_core)
            valid = face_coverage >= 0.55 and 0.03 <= area_ratio <= 0.92
            candidates.append(
                {
                    "idx": int(idx),
                    "score": float(score),
                    "mask": candidate_mask,
                    "area": area,
                    "area_ratio": area_ratio,
                    "face_coverage": face_coverage,
                    "inverted": inverted,
                    "valid": valid,
                }
            )

    valid_candidates = [c for c in candidates if c["valid"]]
    if valid_candidates:
        selected = max(valid_candidates, key=lambda c: (c["score"], c["face_coverage"], -abs(c["area_ratio"] - 0.45)))
        reason = "face_filtered"
    else:
        selected = max(candidates, key=lambda c: (c["face_coverage"], c["score"], -abs(c["area_ratio"] - 0.45)))
        reason = "fallback_best_face_coverage"

    return {
        "best_idx": selected["idx"],
        "best_score": selected["score"],
        "mask": selected["mask"],
        "area": selected["area"],
        "area_ratio": selected["area_ratio"],
        "face_coverage": selected["face_coverage"],
        "inverted": bool(selected["inverted"]),
        "selection_reason": reason,
        "candidate_count": int(len(masks)),
        "variant_count": int(len(candidates)),
        "valid_variant_count": int(len(valid_candidates)),
    }


class BBoxMaskProvider:
    name = "bbox"
    output_name = "M_bbox"

    def generate(self, image_path: Path, image_bgr: np.ndarray, output_dir: Path) -> GeneratedMask:
        raw_path = run_bbox_mask_pose(image_path, output_dir)
        raw_mask = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise ValueError(f"Cannot read BBoxMaskPose raw mask: {raw_path}")
        return GeneratedMask(
            name=self.output_name,
            raw_mask=raw_mask,
            mask=refine_raw_mask(image_bgr, raw_mask),
            metadata={"raw_mask_path": str(raw_path)},
        )


class SamLikeMaskProvider:
    output_name = "M_sam"

    def __init__(self, predictor, selection_mode: SelectionMode = "face_filter") -> None:
        self.predictor = predictor
        self.selection_mode = selection_mode

    def generate(self, image_path: Path, image_bgr: np.ndarray, output_dir: Path) -> GeneratedMask:
        del image_path, output_dir
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        box, box_source = full_image_box(image_bgr)
        face_core, face_core_source = detect_face_core(image_bgr)
        if self.selection_mode == "face_filter":
            selected = select_face_filtered_mask(self.predictor, image_rgb, box, face_core)
        else:
            selected = select_fixed_inverted_mask(self.predictor, image_rgb, box)

        raw_mask = selected["mask"]
        return GeneratedMask(
            name=self.output_name,
            raw_mask=raw_mask,
            mask=refine_raw_mask(image_bgr, raw_mask),
            metadata={
                **{k: v for k, v in selected.items() if k != "mask"},
                "box_source": box_source,
                "box_xyxy": box,
                "face_core_source": face_core_source,
                "face_core_xyxy": face_core,
                "selection_mode": self.selection_mode,
            },
        )


class SamMaskProvider(SamLikeMaskProvider):
    name = "sam"
    output_name = "M_sam"

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: Path = REPO_ROOT / "models/sam/checkpoints/sam_vit_b_01ec64.pth",
        device: str = "cpu",
        selection_mode: SelectionMode = "face_filter",
    ) -> None:
        super().__init__(load_sam(model_type, checkpoint, device), selection_mode=selection_mode)


class Sam2MaskProvider(SamLikeMaskProvider):
    name = "sam2"
    output_name = "M_sam2"

    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint: Path = REPO_ROOT / "models/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        device: str = "cpu",
        selection_mode: SelectionMode = "face_filter",
    ) -> None:
        super().__init__(load_sam2(model_cfg, checkpoint, device), selection_mode=selection_mode)
