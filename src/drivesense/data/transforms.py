"""Image preprocessing, augmentation transforms, and 3D-to-2D box projection.

Provides:
- ``DriveSenseTransform``: composable PIL-based transform for SFT training data.
- ``resize_with_aspect_ratio``: letterbox resize to a fixed resolution.
- ``load_and_preprocess_image``: load from path + resize to model input size.
- ``normalize_bbox_to_1000``: convert pixel coords to Qwen-VL's [0, 1000] grounding range.
- ``get_2d_bbox_from_3d``: project a nuScenes 3D annotation to a 2D camera bbox.

All spatial transforms update bounding box coordinates consistently.
Phase 1a implementation.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import ImageEnhance, ImageFilter

# nuScenes devkit and pyquaternion — available on HPC/data envs, not macOS dev.
try:
    from nuscenes.utils.geometry_utils import view_points  # type: ignore[import]
    from pyquaternion import Quaternion  # type: ignore[import]
    _NUSCENES_AVAILABLE = True
except ImportError:
    view_points = None  # type: ignore[assignment]
    Quaternion = None  # type: ignore[assignment, misc]
    _NUSCENES_AVAILABLE = False


class DriveSenseTransform:
    """Composable PIL-based image transform that jointly updates bounding boxes.

    In ``train`` mode applies random horizontal flip and color jitter before
    letterbox-resizing to the target resolution. In ``val``/``test`` mode only
    resizes.  All spatial operations keep box coordinates consistent.

    Args:
        config: Data config dict from configs/data.yaml.
        mode: ``"train"`` (augmentation + resize) or ``"val"``/``"test"`` (resize only).
    """

    def __init__(self, config: dict, mode: str = "train") -> None:
        self.mode = mode
        preproc = config.get("preprocessing", {})
        target = preproc.get("target_resolution", [672, 448])
        self.target_w: int = int(target[0])
        self.target_h: int = int(target[1])
        # Augmentation hyper-parameters (only used in train mode).
        self._flip_prob: float = 0.5
        self._brightness_range: tuple[float, float] = (0.8, 1.2)
        self._contrast_range: tuple[float, float] = (0.8, 1.2)
        self._blur_prob: float = 0.15
        self._blur_radius: float = 1.0

    def __call__(
        self, image: PILImage.Image, boxes: list[list[float]]
    ) -> tuple[PILImage.Image, list[list[float]]]:
        """Apply transforms to image and corresponding bounding boxes.

        Args:
            image: Input PIL Image (dashcam frame) with boxes in pixel coordinates.
            boxes: List of ``[x1, y1, x2, y2]`` boxes in pixel coordinates.

        Returns:
            ``(transformed_image, transformed_boxes)`` — PIL Image resized to
            ``(target_w, target_h)`` and boxes updated to match.
        """
        w, h = image.size

        if self.mode == "train":
            # Random horizontal flip.
            if random.random() < self._flip_prob:
                image = image.transpose(PILImage.FLIP_LEFT_RIGHT)
                boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]

            # Random brightness + contrast jitter.
            if random.random() < 0.5:
                factor = random.uniform(*self._brightness_range)
                image = ImageEnhance.Brightness(image).enhance(factor)
            if random.random() < 0.5:
                factor = random.uniform(*self._contrast_range)
                image = ImageEnhance.Contrast(image).enhance(factor)

            # Random Gaussian blur.
            if random.random() < self._blur_prob:
                image = image.filter(ImageFilter.GaussianBlur(self._blur_radius))

        # Letterbox resize — updates boxes with scale + offset.
        image, boxes = _apply_letterbox(image, boxes, self.target_w, self.target_h)
        return image, boxes


def resize_with_aspect_ratio(
    image: PILImage.Image,
    target_width: int,
    target_height: int,
) -> PILImage.Image:
    """Resize image to target dimensions using letterboxing to preserve aspect ratio.

    The image is scaled uniformly to fit within the target bounds, then padded
    with black pixels to reach exactly ``(target_width, target_height)``.

    Args:
        image: Input PIL Image.
        target_width: Target width in pixels.
        target_height: Target height in pixels.

    Returns:
        Resized and padded PIL Image of exactly ``(target_width, target_height)``.
    """
    result, _ = _apply_letterbox(image, [], target_width, target_height)
    return result


def load_and_preprocess_image(
    image_path: Path,
    target_size: tuple[int, int] = (672, 448),
) -> PILImage.Image:
    """Load an image from disk and resize to the model input resolution.

    Converts to RGB and applies letterbox resizing so the aspect ratio is
    preserved.  No normalization is applied here — that is handled by the
    model's image processor at training time.

    Args:
        image_path: Path to the source image file.
        target_size: ``(width, height)`` target resolution. Defaults to ``(672, 448)``
            matching the model's dashcam-optimised resolution from configs/model.yaml.

    Returns:
        Preprocessed PIL Image ready for the model processor.
    """
    image = PILImage.open(Path(image_path)).convert("RGB")
    return resize_with_aspect_ratio(image, target_size[0], target_size[1])


def normalize_bbox_to_1000(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[int]:
    """Convert pixel-coordinate bounding box to Qwen-VL's [0, 1000] grounding range.

    Qwen3-VL uses normalised integer coordinates in the range [0, 1000] for
    its grounding output format — both in prompts and predicted text.

    Args:
        bbox: ``[x1, y1, x2, y2]`` in pixel coordinates.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        ``[x1, y1, x2, y2]`` with each value rounded to the nearest integer in
        the range [0, 1000].
    """
    x1, y1, x2, y2 = bbox
    return [
        round(1000 * x1 / image_width),
        round(1000 * y1 / image_height),
        round(1000 * x2 / image_width),
        round(1000 * y2 / image_height),
    ]


def get_2d_bbox_from_3d(
    nusc: object,
    annotation_token: str,
    camera_token: str,
) -> list[int] | None:
    """Project a nuScenes 3D annotation box to a 2D camera bounding box.

    Transforms the 3D box from global coordinates into the camera frame using
    ego-vehicle pose and camera calibration, projects all 8 corners to the image
    plane, and returns the enclosing axis-aligned rectangle in Qwen-VL's [0, 1000]
    normalised coordinate space.

    Args:
        nusc: Initialised ``NuScenes`` SDK object.
        annotation_token: Token identifying the ``sample_annotation`` record.
        camera_token: ``sample_data`` token for the target camera (e.g. CAM_FRONT).

    Returns:
        ``[x1, y1, x2, y2]`` normalised to [0, 1000], or ``None`` if all box
        corners lie behind the camera plane (z <= 0).

    Raises:
        ImportError: If nuScenes devkit / pyquaternion is not installed.
    """
    if not _NUSCENES_AVAILABLE:
        raise ImportError(
            "nuScenes devkit + pyquaternion required. Install with: pip install nuscenes-devkit"
        )

    cam_data = nusc.get("sample_data", camera_token)  # type: ignore[attr-defined]
    ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])  # type: ignore[attr-defined]
    cal = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])  # type: ignore[attr-defined]

    # Retrieve box in global frame and transform → ego frame → camera frame.
    box = nusc.get_box(annotation_token)  # type: ignore[attr-defined]
    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)
    box.translate(-np.array(cal["translation"]))
    box.rotate(Quaternion(cal["rotation"]).inverse)

    # 8 corners in camera frame; shape (3, 8).
    corners: np.ndarray = box.corners()
    valid_mask = corners[2, :] > 0  # keep only corners in front of camera
    if not np.any(valid_mask):
        return None

    intrinsic = np.array(cal["camera_intrinsic"])
    pts = view_points(corners[:, valid_mask], intrinsic, normalize=True)  # (3, M)

    img_w: int = cam_data["width"]
    img_h: int = cam_data["height"]
    x1 = float(np.clip(np.min(pts[0]), 0, img_w))
    y1 = float(np.clip(np.min(pts[1]), 0, img_h))
    x2 = float(np.clip(np.max(pts[0]), 0, img_w))
    y2 = float(np.clip(np.max(pts[1]), 0, img_h))

    return normalize_bbox_to_1000([x1, y1, x2, y2], img_w, img_h)


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _apply_letterbox(
    image: PILImage.Image,
    boxes: list[list[float]],
    target_w: int,
    target_h: int,
) -> tuple[PILImage.Image, list[list[float]]]:
    """Letterbox-resize image and adjust box coordinates.

    Args:
        image: Input PIL Image.
        boxes: Boxes in ``[x1, y1, x2, y2]`` pixel coordinates.
        target_w: Target canvas width.
        target_h: Target canvas height.

    Returns:
        ``(resized_image, updated_boxes)`` where updated_boxes reflect the
        scaled + padded coordinate space.
    """
    orig_w, orig_h = image.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    resized = image.resize((new_w, new_h), PILImage.LANCZOS)
    canvas = PILImage.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, (pad_x, pad_y))

    updated: list[list[float]] = [
        [
            x1 * scale + pad_x,
            y1 * scale + pad_y,
            x2 * scale + pad_x,
            y2 * scale + pad_y,
        ]
        for x1, y1, x2, y2 in boxes
    ]
    return canvas, updated
