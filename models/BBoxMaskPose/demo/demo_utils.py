"""
Utilities for the BMP demo:
- Visualization of detections, masks, and poses
- Mask and bounding-box processing
- Pose non-maximum suppression (NMS)
- Animated GIF creation of demo iterations
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from pycocotools import mask as Mask
from sam2.distinctipy import get_colors
from tqdm import tqdm

FBE_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(FBE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(FBE_REPO_ROOT))

from fbe_protocol.mask.background_transform import BackgroundTransformConfig, apply_background_transform

### Visualization hyperparameters
MIN_CONTOUR_AREA: int = 50
BBOX_WEIGHT: float = 0.9
MASK_WEIGHT: float = 0.6
BACK_MASK_WEIGHT: float = 0.6
POSE_WEIGHT: float = 0.8


"""
posevis is our custom visualization library for pose estimation. For compatibility, we also provide a lite version that has fewer features but still reproduces visualization from the paper.
"""
try:
    from posevis import pose_visualization
except ImportError:
    from posevis_lite import pose_visualization


class DotDict(dict):
    """Dictionary with attribute access and nested dict wrapping."""

    def __getattr__(self, name: str) -> any:
        if name in self:
            val = self[name]
            if isinstance(val, dict):
                val = DotDict(val)
                self[name] = val
            return val
        raise AttributeError("No attribute named {!r}".format(name))

    def __setattr__(self, name: str, value: any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self:
            del self[name]
        else:
            raise AttributeError("No attribute named {!r}".format(name))


def filter_instances(instances: InstanceData, indices):
    """
    Return a new InstanceData containing only the entries of 'instances' at the given indices.
    """
    if instances is None:
        return None
    data = {}
    # Attributes to filter
    for attr in [
        "bboxes",
        "bbox_scores",
        "keypoints",
        "keypoint_scores",
        "scores",
        "pred_masks",
        "refined_masks",
        "sam_scores",
        "sam_kpts",
    ]:
        if hasattr(instances, attr):
            arr = getattr(instances, attr)
            data[attr] = arr[indices] if arr is not None else None
    return InstanceData(**data)


def concat_instances(instances1: InstanceData, instances2: InstanceData):
    """
    Concatenate two InstanceData objects along the first axis, preserving order.
    If instances1 or instances2 is None, returns the other.
    """
    if instances1 is None:
        return instances2
    if instances2 is None:
        return instances1
    data = {}
    for attr in [
        "bboxes",
        "bbox_scores",
        "keypoints",
        "keypoint_scores",
        "scores",
        "pred_masks",
        "refined_masks",
        "sam_scores",
        "sam_kpts",
    ]:
        arr1 = getattr(instances1, attr, None)
        arr2 = getattr(instances2, attr, None)
        if arr1 is None and arr2 is None:
            continue
        if arr1 is None:
            data[attr] = arr2
        elif arr2 is None:
            data[attr] = arr1
        else:
            data[attr] = np.concatenate([arr1, arr2], axis=0)
    return InstanceData(**data)


def _visualize_predictions(
    img: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    masks: List[Optional[List[np.ndarray]]],
    poses: List[Optional[np.ndarray]],
    vis_type: str = "mask",
    mask_is_binary: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render bounding boxes, segmentation masks, and poses on the input image.

    Args:
        img (np.ndarray): BGR image of shape (H, W, 3).
        bboxes (np.ndarray): Array of bounding boxes [x, y, w, h].
        scores (np.ndarray): Confidence scores for each bbox.
        masks (List[Optional[List[np.ndarray]]]): Polygon masks per instance.
        poses (List[Optional[np.ndarray]]): Keypoint arrays per instance.
        vis_type (str): Flags for visualization types separated by '+'.
        mask_is_binary (bool): Whether input masks are binary arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The visualized image and color map.
    """
    vis_types = vis_type.split("+")

    # Exclude white, black, and green colors from the palette as they are not distinctive
    colors = (np.array(get_colors(len(bboxes), exclude_colors=[(0, 1, 0), (0, 0, 0), (1, 1, 1)], rng=0)) * 255).astype(
        int
    )

    if mask_is_binary:
        poly_masks: List[Optional[List[np.ndarray]]] = []
        for binary_mask in masks:
            if binary_mask is not None:
                contours, _ = cv2.findContours(
                    (binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                polys = [cnt.flatten() for cnt in contours if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA]
            else:
                polys = None
            poly_masks.append(polys)
        masks = poly_masks  # type: ignore

    if "inv-mask" in vis_types:
        stencil = np.zeros_like(img)

    for bbox, score, mask_poly, pose, color in zip(bboxes, scores, masks, poses, colors):
        bbox = _update_bbox_by_mask(list(map(int, bbox)), mask_poly, img.shape)
        color_list = color.tolist()
        img_copy = img.copy()

        if "bbox" in vis_types:
            x, y, w, h = bbox
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color_list, 2)
            img = cv2.addWeighted(img, 1 - BBOX_WEIGHT, img_copy, BBOX_WEIGHT, 0)

        if mask_poly is not None and "mask" in vis_types:
            for seg in mask_poly:
                seg_pts = np.array(seg).reshape(-1, 1, 2).astype(int)
                cv2.fillPoly(img_copy, [seg_pts], color_list)
            img = cv2.addWeighted(img, 1 - MASK_WEIGHT, img_copy, MASK_WEIGHT, 0)

        if mask_poly is not None and "mask-out" in vis_types:
            for seg in mask_poly:
                seg_pts = np.array(seg).reshape(-1, 1, 2).astype(int)
                cv2.fillPoly(img, [seg_pts], (0, 0, 0))

        if mask_poly is not None and "inv-mask" in vis_types:
            for seg in mask_poly:
                seg = np.array(seg).reshape(-1, 1, 2).astype(int)
                if cv2.contourArea(seg) < MIN_CONTOUR_AREA:
                    continue
                cv2.fillPoly(stencil, [seg], (255, 255, 255))

        if pose is not None and "pose" in vis_types:
            vis_img = pose_visualization(
                img.copy(),
                pose.reshape(-1, 3),
                width_multiplier=8,
                differ_individuals=True,
                color=color_list,
                keep_image_size=True,
            )
            img = cv2.addWeighted(img, 1 - POSE_WEIGHT, vis_img, POSE_WEIGHT, 0)

    if "inv-mask" in vis_types:
        img = cv2.addWeighted(img, 1 - BACK_MASK_WEIGHT, cv2.bitwise_and(img, stencil), BACK_MASK_WEIGHT, 0)

    return img, colors


from pathlib import Path
from typing import Any, Optional
import os
import numpy as np

def visualize_itteration(
    img: np.ndarray,
    detections: Any,
    iteration_idx: int,
    output_root: Path,
    img_name: str,
    with_text: bool = False,
) -> Optional[np.ndarray]:
    """
    Signature-compatible visualize_itteration.

    Supported BG modes (single mode only):
      - white | black | mean | median

    Control (no call-site change):
      1) Env var: BMP_BG_MODE in {"white","black","mean","median"}
      2) detections.bg_mode (or detections["bg_mode"]) if present

    Output files (saved under output_root/img_name/iter_XXX/):
      - X_<mode>.png        : foreground kept, background filled with selected color (candidate inversion input)
      - BGkeep_<mode>.png   : background kept, foreground filled with selected color
      - panel_<mode>.png    : [original | X_<mode> | BGkeep_<mode>]

    Returns:
      - panel image (np.uint8) or None if required masks are unavailable.
    """

    # ----------------------------
    # Helpers
    # ----------------------------
    def _ensure_u8_rgb(im: np.ndarray) -> np.ndarray:
        if im is None:
            raise ValueError("img is None")
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        if im.ndim != 3 or im.shape[2] != 3:
            raise ValueError(f"img must be HxWx3, got {im.shape}")
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)
        return im

    def _ensure_bool(m: np.ndarray) -> Optional[np.ndarray]:
        if m is None:
            return None
        if m.dtype == np.bool_:
            return m
        return m > 0

    def _get_from_detections(det: Any, name: str):
        if hasattr(det, name):
            return getattr(det, name)
        if isinstance(det, dict) and name in det:
            return det[name]
        return None

    def _mask_erode(mask_bool: np.ndarray, ksize: int = 15, iterations: int = 1) -> np.ndarray:
        if mask_bool is None:
            return None
        if ksize <= 1 or iterations <= 0:
            return mask_bool
        try:
            import cv2
            kernel = np.ones((ksize, ksize), np.uint8)
            m = (mask_bool.astype(np.uint8) * 255)
            m = cv2.erode(m, kernel, iterations=iterations)
            return m > 0
        except Exception:
            return mask_bool

    def _make_base(H: int, W: int, rgb_u8: np.ndarray) -> np.ndarray:
        return np.tile(rgb_u8.reshape(1, 1, 3), (H, W, 1))

    def _save_img(path: Path, arr_u8: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
            cv2.imwrite(str(path), arr_u8[:, :, ::-1])  # RGB->BGR
        except Exception:
            from PIL import Image
            Image.fromarray(arr_u8).save(str(path))

    def _normalize_mask_shape(m: Optional[np.ndarray], H: int, W: int) -> Optional[np.ndarray]:
        if m is None:
            return None
        if m.shape == (H, W):
            return m
        # common cases: (1,H,W) or (H,W,1)
        if m.ndim == 3 and m.shape[0] == 1 and m.shape[1:] == (H, W):
            return m[0]
        if m.ndim == 3 and m.shape[:2] == (H, W) and m.shape[2] == 1:
            return m[:, :, 0]
        return None

    # ----------------------------
    # Inputs
    # ----------------------------
    img_u8 = _ensure_u8_rgb(img)
    H, W = img_u8.shape[:2]

    # Determine mode
    mode = os.getenv("BMP_BG_MODE", "").strip().lower()
    if not mode:
        dm = _get_from_detections(detections, "bg_mode")
        if isinstance(dm, str) and dm.strip():
            mode = dm.strip().lower()
    if mode not in {"white", "black", "mean", "median"}:
        mode = "white"

    # Obtain masks
    fore_mask = _get_from_detections(detections, "fore_mask")
    back_mask = _get_from_detections(detections, "back_mask")

    # Fallback candidates (keep harmless)
    if fore_mask is None:
        fore_mask = _get_from_detections(detections, "person_mask")
    if fore_mask is None:
        fore_mask = _get_from_detections(detections, "mask")

    fore = _ensure_bool(fore_mask)
    back = _ensure_bool(back_mask)

    fore = _normalize_mask_shape(fore, H, W)
    back = _normalize_mask_shape(back, H, W)

    # Infer missing masks
    if back is None and fore is not None:
        back = ~fore
    if fore is None and back is not None:
        fore = ~back

    if fore is None or back is None:
        # Do not crash pipeline
        return None

    # ----------------------------
    # Compute background mean / median on eroded background
    # ----------------------------
    back_er = _mask_erode(back, ksize=15, iterations=1)
    if int(back_er.sum()) < 128:
        back_er = back

    bg_pixels = img_u8[back_er]
    if bg_pixels.size == 0:
        bg_pixels = img_u8.reshape(-1, 3)

    mean_rgb = bg_pixels.mean(axis=0)
    median_rgb = np.median(bg_pixels, axis=0)

    if mode == "white":
        code_rgb = np.array([255, 255, 255], dtype=np.uint8)
    elif mode == "black":
        code_rgb = np.array([0, 0, 0], dtype=np.uint8)
    elif mode == "mean":
        code_rgb = np.clip(mean_rgb, 0, 255).astype(np.uint8)
    elif mode == "median":
        code_rgb = np.clip(median_rgb, 0, 255).astype(np.uint8)
    else:
        code_rgb = np.array([255, 255, 255], dtype=np.uint8)

    base = _make_base(H, W, code_rgb)

    # Foreground-kept image (this is your X_<mode>)
    X_mode = base.copy()
    X_mode[fore] = img_u8[fore]

    # Background-kept image (visualization)
    BGkeep = base.copy()
    BGkeep[back] = img_u8[back]

    # Panel: [original | X_mode | BGkeep]
    panel = np.concatenate([img_u8, X_mode, BGkeep], axis=1)

    # ----------------------------
    # Save outputs
    # ----------------------------
    output_root = Path(output_root)
    out_dir = output_root / img_name / f"iter_{iteration_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_img(out_dir / f"X_{mode}.png", X_mode)
    _save_img(out_dir / f"BGkeep_{mode}.png", BGkeep)
    _save_img(out_dir / f"panel_{mode}.png", panel)

    return panel


# ============================================================================
# White & Black & Mean & Median & Blur, Gray
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import cv2


def visualize_itteration(
    img: np.ndarray,
    detections: Any,
    iteration_idx: int,
    output_root: Path,
    img_name: str,
    with_text: bool = False,
) -> Optional[np.ndarray]:

    bboxes = getattr(detections, "bboxes", None)
    scores = getattr(detections, "scores", None)
    pred_masks = getattr(detections, "pred_masks", None)         # MaskPose_(in)
    refined_masks = getattr(detections, "refined_masks", None)   # SAM refined masks
    keypoints = getattr(detections, "keypoints", None)

    H, W = img.shape[:2]

    # ------------------------------------------------------------
    # 0) Ensure uint8 for stable cv2.imwrite colors
    # ------------------------------------------------------------
    if img.dtype != np.uint8:
        img_f = img.astype(np.float32)
        if img_f.max() <= 1.0 + 1e-6:
            img_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(img_f, 0, 255).astype(np.uint8)
    else:
        img_u8 = img

    # ------------------------------------------------------------
    # 1) Build the foreground mask from model outputs.
    # ------------------------------------------------------------
    fore_mask = _build_foreground_mask(detections, H, W)

    back_mask = ~fore_mask

    # ------------------------------------------------------------
    # 2) Compose background-transformed outputs.
    # ------------------------------------------------------------
    bg_config = BackgroundTransformConfig.from_env()
    transform = apply_background_transform(img_u8, fore_mask, bg_config)
    mode = transform.metadata["mode"]
    fg_img = transform.foreground
    bg_img = transform.background

    if transform.fill_rgb is not None and mode not in ("white", "black"):
        try:
            base = f"{img_name}_iter{iteration_idx + 1}"
            with open(os.path.join(output_root, f"{base}_bg_{mode}_rgb.txt"), "w") as f:
                f.write(
                    f"mode={mode}\n"
                    f"fill_rgb={transform.fill_rgb.tolist()}\n"
                    f"N={int(transform.metadata.get('background_pixel_count', 0))}\n"
                    f"bins={int(transform.metadata.get('mode_bins', bg_config.mode_bins))}\n"
                )
        except Exception:
            pass

    # ------------------------------------------------------------
    # 2.5) Save
    # ------------------------------------------------------------
    os.makedirs(output_root, exist_ok=True)
    base = f"{img_name}_iter{iteration_idx + 1}"

    foreground_black = img_u8.copy()
    foreground_black[back_mask] = np.array([0, 0, 0], dtype=np.uint8)
    foreground_white = img_u8.copy()
    foreground_white[back_mask] = np.array([255, 255, 255], dtype=np.uint8)

    background_black = np.zeros_like(img_u8)
    background_black[back_mask] = img_u8[back_mask]
    background_white = np.full_like(img_u8, 255)
    background_white[back_mask] = img_u8[back_mask]

    _write_binary_mask_variants(Path(output_root), base, fore_mask)

    # Preserve the original white baseline names; suffix other modes.
    if mode == "white":
        fg_path = os.path.join(output_root, f"{base}_MaskPoseIn_Foreground.jpg")
        bg_path = os.path.join(output_root, f"{base}_MaskPoseIn_Background.jpg")
    else:
        fg_path = os.path.join(output_root, f"{base}_MaskPoseIn_Foreground_{mode}.jpg")
        bg_path = os.path.join(output_root, f"{base}_MaskPoseIn_Background_{mode}.jpg")

    cv2.imwrite(fg_path, fg_img)
    cv2.imwrite(bg_path, bg_img)
    cv2.imwrite(os.path.join(output_root, f"{base}_Foreground_black.jpg"), foreground_black)
    cv2.imwrite(os.path.join(output_root, f"{base}_Foreground_white.jpg"), foreground_white)
    cv2.imwrite(os.path.join(output_root, f"{base}_Background_black.jpg"), background_black)
    cv2.imwrite(os.path.join(output_root, f"{base}_Background_white.jpg"), background_white)

    # ------------------------------------------------------------
    # 3) Return masked_out for the next iteration.
    # ------------------------------------------------------------
    masked_out: Optional[np.ndarray] = None
    if refined_masks is not None:
        masked_out, _ = _visualize_predictions(
            img_u8.copy(),
            bboxes,
            scores,
            refined_masks,
            keypoints,
            vis_type="mask-out",
            mask_is_binary=True,
        )

    return masked_out

def create_GIF(
    img_path: Path,
    output_root: Path,
    bmp_x: int = 2,
) -> None:
    """
    Compile iteration images into an animated GIF using ffmpeg.

    Args:
        img_path (Path): Path to a sample iteration image.
        output_root (Path): Directory to save the GIF.
        bmp_x (int): Number of BMP iterations.
        duration_per_frame (int): Frame display duration in ms.

    Raises:
        RuntimeError: If ffmpeg is not available or images are missing.
    """
    display_dur = 1.5  # seconds
    fade_dur = 1.0
    fps = 10
    scale_width = 300  # Resize width for GIF, height will be auto-scaled to maintain aspect ratio

    # Check if ffmpeg is installed. If not, raise warning and return
    if shutil.which("ffmpeg") is None:
        print_log("FFMpeg is not installed. GIF creation will be skipped.", logger="current", level=logging.WARNING)
        return
    print_log("Creating GIF with FFmpeg...", logger="current")

    dirname, filename = os.path.split(img_path)
    img_name_wo_ext, _ = os.path.splitext(filename)

    gif_image_names = [
        "Detector_(out)",
        "MaskPose_(in)",
        "MaskPose_(out)",
        "prompting_kpts",
        "SAM_Masks",
        "Mask-Out",
    ]

    # Create black image of the same size as the last image
    last_img_path = os.path.join(dirname, "{}_iter1_{}".format(img_name_wo_ext, gif_image_names[0]) + ".jpg")
    last_img = cv2.imread(last_img_path)
    if last_img is None:
        print_log("Could not read image {}.".format(last_img_path), logger="current", level=logging.ERROR)
        return
    black_img = np.zeros_like(last_img)
    cv2.imwrite(os.path.join(dirname, "black_image.jpg"), black_img)

    gif_images = []
    for iter in range(bmp_x):
        iter_img_path = os.path.join(dirname, "{}_iter{}_".format(img_name_wo_ext, iter + 1))
        for img_name in gif_image_names:

            if iter + 1 == bmp_x and img_name == "Mask-Out":
                # Skip the last iteration's Mask-Out image
                continue

            img_file = "{}{}.jpg".format(iter_img_path, img_name)
            if not os.path.exists(img_file):
                print_log("{} does not exist, skipping.".format(img_file), logger="current", level=logging.WARNING)
                continue
            gif_images.append(img_file)

    if len(gif_images) == 0:
        print_log("No images found for GIF creation.", logger="current", level=logging.WARNING)
        return

    # Add 'before' and 'after' images
    after1_img = os.path.join(dirname, "{}_iter{}_Final_Poses.jpg".format(img_name_wo_ext, bmp_x))
    after2_img = os.path.join(dirname, "{}_iter{}_SAM_Masks.jpg".format(img_name_wo_ext, bmp_x))
    # gif_images.append(os.path.join(dirname, "black_image.jpg"))  # Add black image at the end
    gif_images.append(after1_img)
    gif_images.append(after2_img)
    gif_images.append(os.path.join(dirname, "black_image.jpg"))  # Add black image at the end

    # Create a GIF from the images
    gif_output_path = os.path.join(output_root, "{}_bmp_{}x.gif".format(img_name_wo_ext, bmp_x))

    # 0. Make sure images exist and are divisible by 2
    for img in gif_images:
        if not os.path.exists(img):
            print_log("Image {} does not exist, skipping GIF creation.".format(img), logger="current", level=logging.WARNING)
            return
        # Check if image dimensions are divisible by 2
        img_data = cv2.imread(img)
        if img_data.shape[1] % 2 != 0 or img_data.shape[0] % 2 != 0:
            print_log(
                "Image {} dimensions are not divisible by 2, resizing.".format(img),
                logger="current",
                level=logging.WARNING,
            )
            resized_img = cv2.resize(img_data, (img_data.shape[1] // 2 * 2, img_data.shape[0] // 2 * 2))
            cv2.imwrite(img, resized_img)

    # 1. inputs
    in_args = []
    for p in gif_images:
        in_args += ["-loop", "1", "-t", str(display_dur), "-i", p]

    # 2. build xfade chain
    n = len(gif_images)
    parts = []
    for i in range(1, n):
        # left label: first is input [0:v], then [v1], [v2], …
        left = "[{}:v]".format(i - 1) if i == 1 else "[v{}]".format(i - 1)
        right = "[{}:v]".format(i)
        out = "[v{}]".format(i)
        offset = (i - 1) * (display_dur + fade_dur) + display_dur
        parts.append(
            "{}{}xfade=transition=fade:".format(left, right)
            + "duration={}:offset={:.3f}{}".format(fade_dur, offset, out)
        )
    filter_complex = ";".join(parts)

    # 3. make MP4 slideshow
    mp4 = "slideshow.mp4"
    cmd1 = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-v",
        "quiet",
        "-hide_banner",
        "-y",
        *in_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v{}]".format(n - 1),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        mp4,
    ]
    subprocess.run(cmd1, check=True)

    # 4. palette
    palette = "palette.png"
    vf = "fps={}".format(fps)
    if scale_width:
        vf += ",scale={}: -1:flags=lanczos".format(scale_width)

    # 5. generate palette
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-v",
            "quiet",
            "-hide_banner",
            "-y",
            "-i",
            mp4,
            "-vf",
            vf + ",palettegen",
            palette,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # 6. build final GIF
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-v",
            "quiet",
            "-hide_banner",
            "-y",
            "-i",
            mp4,
            "-i",
            palette,
            "-lavfi",
            vf + "[x];[x][1:v]paletteuse",
            gif_output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Clean up temporary files
    os.remove(mp4)
    os.remove(palette)
    os.remove(os.path.join(dirname, "black_image.jpg"))

    print_log(f"GIF saved as '{gif_output_path}'", logger="current")


def _update_bbox_by_mask(
    bbox: List[int], mask_poly: Optional[List[List[int]]], image_shape: Tuple[int, int, int]
) -> List[int]:
    """
    Adjust bounding box to tightly fit mask polygon.

    Args:
        bbox (List[int]): Original [x, y, w, h].
        mask_poly (Optional[List[List[int]]]): Polygon coordinates.
        image_shape (Tuple[int,int,int]): Image shape (H, W, C).

    Returns:
        List[int]: Updated [x, y, w, h] bounding box.
    """
    if mask_poly is None or len(mask_poly) == 0:
        return bbox

    mask_rle = Mask.frPyObjects(mask_poly, image_shape[0], image_shape[1])
    mask_rle = Mask.merge(mask_rle)
    bbox_segm_xywh = Mask.toBbox(mask_rle)
    bbox_segm_xyxy = np.array(
        [
            bbox_segm_xywh[0],
            bbox_segm_xywh[1],
            bbox_segm_xywh[0] + bbox_segm_xywh[2],
            bbox_segm_xywh[1] + bbox_segm_xywh[3],
        ]
    )

    bbox = bbox_segm_xywh

    return bbox.astype(int).tolist()


def pose_nms(config: Any, image_kpts: np.ndarray, image_bboxes: np.ndarray, num_valid_kpts: np.ndarray) -> np.ndarray:
    """
    Perform OKS-based non-maximum suppression on detected poses.

    Args:
        config (Any): Configuration with confidence_thr and oks_thr.
        image_kpts (np.ndarray): Detected keypoints of shape (N, K, 3).
        image_bboxes (np.ndarray): Corresponding bboxes (N,4).
        num_valid_kpts (np.ndarray): Count of valid keypoints per instance.

    Returns:
        np.ndarray: Indices of kept instances.
    """
    # Sort image kpts by average score - lowest first
    # scores = image_kpts[:, :, 2].mean(axis=1)
    # sort_idx = np.argsort(scores)
    # image_kpts = image_kpts[sort_idx, :, :]

    # Compute OKS between all pairs of poses
    oks_matrix = np.zeros((image_kpts.shape[0], image_kpts.shape[0]))
    for i in range(image_kpts.shape[0]):
        for j in range(image_kpts.shape[0]):
            gt_bbox_xywh = image_bboxes[i].copy()
            gt_bbox_xyxy = gt_bbox_xywh.copy()
            gt_bbox_xyxy[2:] += gt_bbox_xyxy[:2]
            gt = {
                "keypoints": image_kpts[i].copy(),
                "bbox": gt_bbox_xyxy,
                "area": gt_bbox_xywh[2] * gt_bbox_xywh[3],
            }
            dt = {"keypoints": image_kpts[j].copy(), "bbox": gt_bbox_xyxy}
            gt["keypoints"][:, 2] = (gt["keypoints"][:, 2] > config.confidence_thr) * 2
            oks = compute_oks(gt, dt)
            if oks > 1:
                breakpoint()
            oks_matrix[i, j] = oks

    np.fill_diagonal(oks_matrix, -1)
    is_subset = oks_matrix > config.oks_thr

    remove_instances = []
    while is_subset.any():
        # Find the pair with the highest OKS
        i, j = np.unravel_index(np.argmax(oks_matrix), oks_matrix.shape)

        # Keep the one with the highest number of keypoints
        if num_valid_kpts[i] > num_valid_kpts[j]:
            remove_idx = j
        else:
            remove_idx = i

        # Remove the column from is_subset
        oks_matrix[:, remove_idx] = 0
        oks_matrix[remove_idx, j] = 0
        remove_instances.append(remove_idx)
        is_subset = oks_matrix > config.oks_thr

    keep_instances = np.setdiff1d(np.arange(image_kpts.shape[0]), remove_instances)

    return keep_instances


def compute_oks(gt: Dict[str, Any], dt: Dict[str, Any], use_area: bool = True, per_kpt: bool = False) -> float:
    """
    Compute Object Keypoint Similarity (OKS) between ground-truth and detected poses.

    Args:
        gt (Dict): Ground-truth keypoints and bbox info.
        dt (Dict): Detected keypoints and bbox info.
        use_area (bool): Whether to normalize by GT area.
        per_kpt (bool): Whether to return per-keypoint OKS array.

    Returns:
        float: OKS score or mean OKS.
    """
    sigmas = (
        np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
        / 10.0
    )
    vars = (sigmas * 2) ** 2
    k = len(sigmas)
    visibility_condition = lambda x: x > 0
    g = np.array(gt["keypoints"]).reshape(k, 3)
    xg = g[:, 0]
    yg = g[:, 1]
    vg = g[:, 2]
    k1 = np.count_nonzero(visibility_condition(vg))
    bb = gt["bbox"]
    x0 = bb[0] - bb[2]
    x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]
    y1 = bb[1] + bb[3] * 2

    d = np.array(dt["keypoints"]).reshape((k, 3))
    xd = d[:, 0]
    yd = d[:, 1]

    if k1 > 0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg

    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
        dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)

    if use_area:
        e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
    else:
        tmparea = gt["bbox"][3] * gt["bbox"][2] * 0.53
        e = (dx**2 + dy**2) / vars / (tmparea + np.spacing(1)) / 2

    if per_kpt:
        oks = np.exp(-e)
        if k1 > 0:
            oks[~visibility_condition(vg)] = 0

    else:
        if k1 > 0:
            e = e[visibility_condition(vg)]
        oks = np.sum(np.exp(-e)) / e.shape[0]

    return oks


def _build_foreground_mask(detections: Any, height: int, width: int) -> np.ndarray:
    bboxes = getattr(detections, "bboxes", None)
    pred_masks = getattr(detections, "pred_masks", None)
    fore_mask = np.zeros((height, width), dtype=bool)

    if pred_masks is None:
        return fore_mask

    if hasattr(pred_masks, "detach"):
        masks_np = pred_masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(pred_masks)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    if masks_np.ndim != 3 or masks_np.size == 0:
        return fore_mask

    maxv = float(np.max(masks_np))
    threshold = 0.5 if maxv <= 1.5 else 127.0
    raw_mask = (masks_np > threshold).any(axis=0)

    bbox_mask = np.zeros((height, width), dtype=bool)
    if bboxes is not None:
        if hasattr(bboxes, "detach"):
            boxes = bboxes.detach().cpu().numpy()
        else:
            boxes = np.asarray(bboxes)

        if boxes.ndim == 1 and boxes.size >= 4:
            boxes = boxes[None, :]
        if boxes.ndim == 2 and boxes.shape[1] >= 4:
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                x1 = max(0, min(width, int(x1)))
                x2 = max(0, min(width, int(x2)))
                y1 = max(0, min(height, int(y1)))
                y2 = max(0, min(height, int(y2)))
                if x2 > x1 and y2 > y1:
                    bbox_mask[y1:y2, x1:x2] = True

    if bbox_mask.any():
        score_raw = int((raw_mask & bbox_mask).sum())
        score_inv = int((~raw_mask & bbox_mask).sum())
        return raw_mask if score_raw >= score_inv else ~raw_mask
    return ~raw_mask if float(raw_mask.mean()) > 0.60 else raw_mask


def _write_binary_mask_variants(output_root: Path, base: str, fore_mask: np.ndarray) -> None:
    mask_white_on_black = np.zeros((*fore_mask.shape, 3), dtype=np.uint8)
    mask_white_on_black[fore_mask] = 255

    mask_black_on_white = np.full((*fore_mask.shape, 3), 255, dtype=np.uint8)
    mask_black_on_white[fore_mask] = 0

    cv2.imwrite(str(Path(output_root) / f"{base}_Mask_white_on_black.jpg"), mask_white_on_black)
    cv2.imwrite(str(Path(output_root) / f"{base}_Mask_black_on_white.jpg"), mask_black_on_white)
