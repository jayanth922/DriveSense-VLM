"""Bounding box overlay and hazard detection result visualization.

Provides utilities to draw predicted and ground truth bounding boxes on
dashcam frames, annotate them with hazard class / severity labels, and
create side-by-side comparison grids for evaluation notebooks and W&B logging.

Implementation target: Phase 1a (basic overlays) / Phase 4b (eval grids)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

# Pillow ImageDraw — in core deps; always available
try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
except ImportError:
    PILImage = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

# W&B optional
try:
    import wandb  # type: ignore[import]
except ImportError:
    wandb = None  # type: ignore[assignment]

# Severity colour map: 1 (low) → green, 5 (critical) → red
SEVERITY_COLORS = {
    1: (0, 200, 0),
    2: (150, 220, 0),
    3: (255, 165, 0),
    4: (255, 80, 0),
    5: (220, 0, 0),
}


def draw_detection(
    image: Image.Image,
    prediction: dict,
    ground_truth: dict | None = None,
    show_gt: bool = True,
) -> Image.Image:
    """Draw predicted (and optionally GT) bounding boxes on a dashcam frame.

    Args:
        image: Input PIL Image (dashcam frame).
        prediction: Prediction dict with 'bbox_2d', 'hazard_class', 'severity'.
        ground_truth: Optional GT dict in the same format. Drawn in white/dashed.
        show_gt: Whether to draw the ground truth box alongside the prediction.

    Returns:
        New PIL Image with annotation overlays.
    """
    raise NotImplementedError("Phase 1a: use ImageDraw to overlay boxes and labels")


def create_eval_grid(
    images: list[Image.Image],
    predictions: list[dict],
    ground_truths: list[dict],
    cols: int = 4,
) -> Image.Image:
    """Create a grid of annotated frames for evaluation visualization.

    Args:
        images: List of PIL Images.
        predictions: List of prediction dicts.
        ground_truths: List of GT dicts.
        cols: Number of columns in the grid layout.

    Returns:
        Single PIL Image with all frames arranged in a grid.
    """
    raise NotImplementedError("Phase 4b: arrange annotated frames into a grid image")


def log_eval_images_to_wandb(
    images: list[Image.Image],
    predictions: list[dict],
    ground_truths: list[dict],
    step: int | None = None,
    max_images: int = 16,
) -> None:
    """Log annotated evaluation images to Weights & Biases.

    Args:
        images: List of PIL Images.
        predictions: List of prediction dicts.
        ground_truths: List of GT dicts.
        step: W&B global step for x-axis alignment.
        max_images: Maximum number of images to log (to avoid storage bloat).
    """
    raise NotImplementedError(
        "Phase 4b: create annotated images and call wandb.log({'eval/images': ...})"
    )


def save_detection_image(
    image: Image.Image,
    prediction: dict,
    output_path: Path,
    ground_truth: dict | None = None,
) -> Path:
    """Save an annotated detection image to disk.

    Args:
        image: Input PIL Image.
        prediction: Prediction dict.
        output_path: Path to save the annotated image.
        ground_truth: Optional GT dict.

    Returns:
        Path to the saved image file.
    """
    raise NotImplementedError("Phase 1a: call draw_detection and save to output_path")
