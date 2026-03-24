"""Bounding box overlay and hazard detection result visualization utilities.

Provides:
- ``draw_bbox_on_image``: draw labelled boxes on a frame (Phase 1a data inspection).
- ``draw_detection``: overlay prediction (+ optional GT) on a dashcam frame.
- ``save_detection_image``: annotate and save a single detection to disk.
- ``create_rarity_distribution_plot``: bar chart of rarity score distribution.
- ``create_sample_grid``: NxM grid of top-scoring frames for data inspection.
- ``create_cooccurrence_heatmap``: 6×6 signal co-occurrence heatmap (Phase 1a-spark).
- ``create_scene_richness_bar_chart``: per-scene mean rarity scores (Phase 1a-spark).
- ``create_signal_prevalence_pie_chart``: signal prevalence pie chart (Phase 1a-spark).
- ``create_eval_grid``: evaluation grid — Phase 4b stub.
- ``log_eval_images_to_wandb``: W&B eval logging — Phase 4b stub.

Phase 1a: draw_bbox_on_image, draw_detection, save_detection_image,
          create_rarity_distribution_plot, create_sample_grid.
Phase 1a-spark: create_cooccurrence_heatmap, create_scene_richness_bar_chart,
                create_signal_prevalence_pie_chart.
Phase 4b: create_eval_grid, log_eval_images_to_wandb.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Pillow is a core dep — always available.
try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PILImage = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

# matplotlib is optional — used only for distribution plots and grids.
try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    _MPL_AVAILABLE = False

# W&B is optional.
try:
    import wandb  # type: ignore[import]
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

# Severity → RGB colour (1=safe green, 5=critical red).
SEVERITY_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 200, 0),
    2: (150, 220, 0),
    3: (255, 165, 0),
    4: (255, 80, 0),
    5: (220, 0, 0),
}
_DEFAULT_COLOR = (0, 180, 255)  # cyan for unlabelled boxes

_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "red": (220, 0, 0), "green": (0, 200, 0),
    "blue": (0, 120, 220), "yellow": (255, 220, 0),
    "cyan": (0, 200, 220), "orange": (255, 140, 0),
}


def draw_bbox_on_image(
    image: Image.Image,
    bboxes: list[dict],
    normalized: bool = True,
) -> Image.Image:
    """Draw labelled bounding boxes on an image and return an annotated copy.

    Args:
        image: Input PIL Image.
        bboxes: List of box dicts, each with keys:
            ``"bbox"`` ([x1, y1, x2, y2]),
            ``"label"`` (str, shown above the box),
            ``"color"`` (optional str name or RGB tuple; defaults to cyan).
        normalized: If ``True``, bbox coordinates are in Qwen-VL's [0, 1000]
            range and will be scaled to pixel space before drawing.

    Returns:
        New PIL Image with box overlays (original is not modified).
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)
    img_w, img_h = image.size

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, AttributeError):
        font = ImageFont.load_default()

    for item in bboxes:
        x1, y1, x2, y2 = item["bbox"]
        if normalized:
            x1 = x1 * img_w / 1000
            y1 = y1 * img_h / 1000
            x2 = x2 * img_w / 1000
            y2 = y2 * img_h / 1000

        raw_color = item.get("color", _DEFAULT_COLOR)
        color: tuple[int, int, int]
        if isinstance(raw_color, str):
            color = _COLOR_MAP.get(raw_color.lower(), _DEFAULT_COLOR)
        else:
            color = tuple(raw_color)  # type: ignore[assignment]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = item.get("label", "")
        if label:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            label_y = max(0, y1 - th - 2)
            draw.rectangle([x1, label_y, x1 + tw + 4, label_y + th + 2], fill=color)
            draw.text((x1 + 2, label_y + 1), label, fill=(255, 255, 255), font=font)

    return out


def draw_detection(
    image: Image.Image,
    prediction: dict,
    ground_truth: dict | None = None,
    show_gt: bool = True,
) -> Image.Image:
    """Overlay predicted (and optionally GT) detection boxes on a dashcam frame.

    Prediction box is drawn in a severity-mapped colour; GT box is drawn in white.
    Both use Qwen-VL's [0, 1000] normalised coordinate convention.

    Args:
        image: Input PIL Image (dashcam frame).
        prediction: Dict with keys ``"bbox_2d"``, ``"hazard_class"``, ``"severity"`` (1-5).
        ground_truth: Optional GT dict in the same format; drawn in white if provided.
        show_gt: Whether to draw the ground truth box.

    Returns:
        New PIL Image with annotation overlays.
    """
    severity = int(prediction.get("severity", 3))
    color = SEVERITY_COLORS.get(severity, _DEFAULT_COLOR)
    hazard_label = f"{prediction.get('hazard_class', 'hazard')} sev={severity}"

    bboxes: list[dict] = [
        {"bbox": prediction["bbox_2d"], "label": hazard_label, "color": color}
    ]
    if show_gt and ground_truth:
        bboxes.append(
            {"bbox": ground_truth["bbox_2d"], "label": "GT", "color": (255, 255, 255)}
        )
    return draw_bbox_on_image(image, bboxes, normalized=True)


def save_detection_image(
    image: Image.Image,
    prediction: dict,
    output_path: Path,
    ground_truth: dict | None = None,
) -> Path:
    """Annotate a frame with detection results and save it to disk.

    Args:
        image: Input PIL Image.
        prediction: Prediction dict with ``"bbox_2d"``, ``"hazard_class"``, ``"severity"``.
        output_path: Destination path for the annotated image (PNG or JPEG).
        ground_truth: Optional GT dict drawn alongside the prediction.

    Returns:
        Path to the saved image file.
    """
    annotated = draw_detection(image, prediction, ground_truth)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path


def create_rarity_distribution_plot(
    distribution: dict[int, int], output_path: Path
) -> Path:
    """Create and save a bar chart of the rarity score distribution.

    Args:
        distribution: Dict mapping rarity score (0-6) to frame count.
        output_path: Path to save the PNG chart.

    Returns:
        Path to the saved chart file.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib required for plots. Run: pip install matplotlib")

    scores = sorted(distribution.keys())
    counts = [distribution[s] for s in scores]
    total = max(sum(counts), 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(scores, counts, color="#2196F3", edgecolor="white", width=0.7)
    ax.set_xlabel("Rarity Score (0–6)", fontsize=12)
    ax.set_ylabel("Frame Count", fontsize=12)
    ax.set_title("nuScenes Rarity Score Distribution", fontsize=14)
    ax.set_xticks(scores)

    for bar, count in zip(bars, counts, strict=False):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{count}\n({100 * count / total:.1f}%)",
                ha="center", va="bottom", fontsize=9,
            )

    ax.axvline(x=3 - 0.5, color="red", linestyle="--", linewidth=1.2,
               label="min_score threshold (3)")
    ax.legend(fontsize=9)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved rarity distribution plot to %s", output_path)
    return output_path


def create_sample_grid(
    images_with_metadata: list[dict],
    output_path: Path,
    grid_size: tuple[int, int] = (3, 3),
) -> Path:
    """Create a grid of sample images with rarity score overlays for data inspection.

    Args:
        images_with_metadata: List of dicts, each with keys ``"image_path"`` (Path),
            ``"score"`` (int), and ``"signals"`` (dict of active signals).
        output_path: Path to save the grid PNG.
        grid_size: ``(rows, cols)`` grid layout. Defaults to 3×3.

    Returns:
        Path to the saved grid image.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib required for grids. Run: pip install matplotlib")

    rows, cols = grid_size
    n = min(len(images_with_metadata), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    # Flatten axes regardless of grid shape (handles 1×1, 1×N, M×N uniformly).
    axes_flat: list = list(np.array(axes).flat)

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= n:
            continue
        meta = images_with_metadata[i]
        try:
            img = PILImage.open(Path(meta["image_path"]))
            ax.imshow(img)
        except (OSError, FileNotFoundError):
            ax.set_facecolor("#333333")

        active = [k for k, v in meta.get("signals", {}).items() if v.get("active")]
        title = f"Score: {meta['score']}  [{', '.join(active[:3])}]"
        ax.set_title(title, fontsize=7, pad=2)

    fig.suptitle("Top-Scoring Rare Hazard Frames", fontsize=13, y=1.01)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sample grid to %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Phase 1a-spark: Spark analytics visualizations
# ------------------------------------------------------------------

def create_cooccurrence_heatmap(
    cooccurrence_data: list[dict],
    output_path: Path,
) -> Path:
    """Create and save a 6×6 signal co-occurrence heatmap.

    Args:
        cooccurrence_data: List of dicts with keys ``"signal_a"``, ``"signal_b"``,
            ``"cooccurrence"`` (as returned by ``SparkAnalytics.signal_cooccurrence()``
            collected to Python).
        output_path: Destination PNG path.

    Returns:
        Path to the saved heatmap image.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib required. Run: pip install matplotlib")

    signals = [
        "sig_proximity", "sig_occlusion", "sig_density",
        "sig_weather", "sig_vru", "sig_cyclist",
    ]
    labels = [s.replace("sig_", "") for s in signals]
    matrix = np.zeros((6, 6), dtype=float)
    lookup = {(r["signal_a"], r["signal_b"]): r["cooccurrence"] for r in cooccurrence_data}
    for i, sa in enumerate(signals):
        for j, sb in enumerate(signals):
            matrix[i, j] = lookup.get((sa, sb), 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Signal Co-occurrence (frame count)", fontsize=12)

    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Frame Count")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved co-occurrence heatmap to %s", output_path)
    return output_path


def create_scene_richness_bar_chart(
    scene_stats: list[dict],
    output_path: Path,
    top_n: int = 20,
) -> Path:
    """Create a horizontal bar chart of mean rarity score per scene.

    Args:
        scene_stats: List of dicts with keys ``"scene_token"``,
            ``"mean_rarity_score"``, ``"total_frames"`` (as returned by
            ``SparkAnalytics.per_scene_stats()`` collected to Python).
        output_path: Destination PNG path.
        top_n: Show only the top N scenes by mean score.

    Returns:
        Path to the saved chart.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib required. Run: pip install matplotlib")

    sorted_scenes = sorted(scene_stats, key=lambda r: r["mean_rarity_score"], reverse=True)
    top = sorted_scenes[:top_n]

    labels = [r["scene_token"][:12] for r in top]
    scores = [r["mean_rarity_score"] for r in top]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.4)))
    bars = ax.barh(range(len(top)), scores, color="#4CAF50", edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Rarity Score", fontsize=11)
    ax.set_title(f"Scene Richness — Top {top_n} Scenes", fontsize=13)

    for bar, score in zip(bars, scores, strict=False):
        ax.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}", va="center", fontsize=8,
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scene richness chart to %s", output_path)
    return output_path


def create_signal_prevalence_pie_chart(
    signal_counts: dict[str, int],
    output_path: Path,
) -> Path:
    """Create a pie chart of active-signal prevalence across all frames.

    Args:
        signal_counts: Dict mapping signal name to frame count where active,
            e.g. ``{"proximity": 120, "occlusion": 80, ...}``.
        output_path: Destination PNG path.

    Returns:
        Path to the saved chart.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib required. Run: pip install matplotlib")

    labels = list(signal_counts.keys())
    sizes = [max(v, 0) for v in signal_counts.values()]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0", "#00BCD4"]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors[: len(labels)],
        startangle=140,
        pctdistance=0.75,
    )
    for t in texts:
        t.set_fontsize(9)
    for at in autotexts:
        at.set_fontsize(8)

    ax.set_title("Rarity Signal Prevalence (% of scored frames)", fontsize=12)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved signal prevalence pie chart to %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Phase 4b stubs
# ------------------------------------------------------------------

def create_eval_grid(
    images: list[Image.Image],
    predictions: list[dict],
    ground_truths: list[dict],
    cols: int = 4,
) -> Image.Image:
    """Create a grid of annotated evaluation frames. (Phase 4b)

    Args:
        images: List of PIL Images.
        predictions: List of prediction dicts.
        ground_truths: List of GT dicts.
        cols: Number of columns in the grid layout.

    Returns:
        Single PIL Image with all frames in a grid.
    """
    raise NotImplementedError("Phase 4b: arrange draw_detection frames into a grid image")


def log_eval_images_to_wandb(
    images: list[Image.Image],
    predictions: list[dict],
    ground_truths: list[dict],
    step: int | None = None,
    max_images: int = 16,
) -> None:
    """Log annotated evaluation images to Weights & Biases. (Phase 4b)

    Args:
        images: List of PIL Images.
        predictions: List of prediction dicts.
        ground_truths: List of GT dicts.
        step: W&B global step for x-axis alignment.
        max_images: Maximum number of images to log.
    """
    raise NotImplementedError(
        "Phase 4b: annotate images with draw_detection and call wandb.log()"
    )
