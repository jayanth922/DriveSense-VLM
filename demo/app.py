"""DriveSense-VLM Gradio Demo — HuggingFace Spaces.

Phase 4a: Full implementation with lazy model loading, severity-coded
bounding box overlay, and example gallery.

Hardware target: Free T4 GPU on HuggingFace Spaces.
Backend: DriveSenseLocalInference (transformers — no vLLM required).

Usage (local):
    python demo/app.py

Usage (HF Spaces):
    Push this file to a HuggingFace Space with gradio as the SDK.
    Set Space hardware to T4 Small (free tier).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve src/ on the Python path (needed when run directly from demo/)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PIL import Image  # noqa: E402

try:
    import gradio as gr  # type: ignore[import]
    _GRADIO_AVAILABLE = True
except ImportError:
    gr = None  # type: ignore[assignment]
    _GRADIO_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).parent
EXAMPLES_DIR = DEMO_DIR / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

# Model config — HF Spaces reads from inference.yaml if present; else uses defaults
_CONFIGS_DIR = _REPO_ROOT / "configs"
_INFERENCE_YAML = _CONFIGS_DIR / "inference.yaml"

SEVERITY_COLORS: dict[str, tuple[int, int, int]] = {
    "critical": (255, 0, 0),
    "high": (255, 140, 0),
    "medium": (255, 215, 0),
    "low": (50, 205, 50),
    "no_hazard": (65, 105, 225),
}

# ---------------------------------------------------------------------------
# Global model (lazy-loaded)
# ---------------------------------------------------------------------------

_model: object = None


def _get_config() -> dict:
    """Load inference config or return minimal defaults."""
    if _INFERENCE_YAML.exists():
        try:
            import yaml  # type: ignore[import]
            with _INFERENCE_YAML.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:  # noqa: BLE001
            pass
    # HF Spaces default: load from HF Hub model ID set via env var
    model_path = os.environ.get("MODEL_PATH", "outputs/quantized_model")
    return {
        "demo": {
            "model_path": model_path,
            "device": "auto",
            "max_image_size": [672, 448],
        }
    }


def _load_model() -> object:
    """Lazy-load DriveSenseLocalInference on first call."""
    global _model  # noqa: PLW0603
    if _model is not None:
        return _model
    try:
        from drivesense.inference.serve import DriveSenseLocalInference  # noqa: PLC0415
        config = _get_config()
        _model = DriveSenseLocalInference(config)
        logger.info("Model loaded successfully")
    except Exception as exc:  # noqa: BLE001
        logger.error("Model load failed: %s", exc)
        _model = None
    return _model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def analyze_image(
    image: Image.Image | None,
) -> tuple[Image.Image | None, str, str]:
    """Run DriveSense-VLM hazard detection on a dashcam image.

    Args:
        image: Input PIL Image from Gradio.

    Returns:
        Tuple of (annotated_image, json_str, latency_str).
    """
    if image is None:
        return None, "Please upload a dashcam image.", ""

    model = _load_model()

    if model is None:
        placeholder = _make_placeholder_result()
        return image, json.dumps(placeholder, indent=2), "⚠️ Model not loaded — showing placeholder"

    import time  # noqa: PLC0415
    try:
        t0 = time.perf_counter()
        annotated, annotation = model.predict_with_visualization(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latency_str = f"⏱ {elapsed_ms:.0f} ms"
        return annotated, json.dumps(annotation, indent=2), latency_str
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference error: %s", exc)
        err_result = {"error": str(exc), "hazards": []}
        return image, json.dumps(err_result, indent=2), "⚠️ Inference error"


# ---------------------------------------------------------------------------
# draw_hazard_boxes (standalone for Spaces — mirrors serve.py)
# ---------------------------------------------------------------------------


def draw_hazard_boxes(
    image: Image.Image,
    annotation: dict,
) -> Image.Image:
    """Overlay severity-coded bounding boxes on a PIL Image.

    Mirrors ``drivesense.inference.serve.draw_hazard_boxes`` so the demo
    can be run without installing the full package.

    Args:
        image:      Input PIL Image.
        annotation: Parsed annotation dict with ``hazards`` list.

    Returns:
        New PIL Image with boxes drawn.
    """
    from PIL import ImageDraw  # noqa: PLC0415
    w, h = image.size
    hazards = annotation.get("hazards", [])

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for hazard in hazards:
        bbox = hazard.get("bbox_2d", [])
        if len(bbox) != 4:
            continue
        severity = str(hazard.get("severity", "no_hazard")).lower()
        label = str(hazard.get("label", "hazard"))
        color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["no_hazard"])

        x1 = int(bbox[0] * w / 1000)
        y1 = int(bbox[1] * h / 1000)
        x2 = int(bbox[2] * w / 1000)
        y2 = int(bbox[3] * h / 1000)

        draw.rectangle([x1, y1, x2, y2], fill=(*color, 50))
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=2)

        text = f"{label} ({severity})"
        text_y = max(0, y1 - 18)
        draw.text((x1 + 2, text_y), text, fill=(*color, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_placeholder_result() -> dict:
    """Return a realistic placeholder when the model is unavailable."""
    return {
        "hazards": [
            {
                "label": "pedestrian_in_path",
                "bbox_2d": [120, 80, 350, 280],
                "severity": "high",
                "reasoning": (
                    "⚠️ Model not loaded — placeholder output. "
                    "Once the AWQ-quantized model is available, this will show "
                    "real chain-of-thought reasoning about the detected hazard."
                ),
                "action": "yield",
            }
        ],
        "scene_summary": "Placeholder: model not yet loaded.",
        "ego_context": {
            "weather": "unknown",
            "time_of_day": "unknown",
            "road_type": "unknown",
        },
    }


def _get_example_images() -> list[list]:
    """Collect example images from demo/examples/."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in sorted(EXAMPLES_DIR.iterdir()) if p.suffix.lower() in exts]
    return [[str(p)] for p in paths[:6]]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

TITLE = "DriveSense-VLM — AV Rare Hazard Detection"

DESCRIPTION = """
## DriveSense-VLM

Upload a dashcam image to detect rare driving hazards.

**Model:** Qwen3-VL-2B-Instruct fine-tuned with LoRA SFT (AWQ 4-bit quantized)
**Training data:** nuScenes rare-hazard frames + DADA-2000 accident moment frames
**Output:** Structured JSON — bounding boxes, hazard labels, severity (low/medium/high/critical),
chain-of-thought reasoning, and ego-vehicle action recommendation.

**Severity:** 🔴 Critical &nbsp; 🟠 High &nbsp; 🟡 Medium &nbsp; 🟢 Low &nbsp; 🔵 No hazard

Links: [GitHub](https://github.com/spartan/DriveSense-VLM) · [Paper](#) · [Dataset](#)
"""

SCHEMA_DOC = """\
```json
{
  "hazards": [
    {
      "label":     "pedestrian_in_path | vehicle_cut_in | debris | ...",
      "bbox_2d":   [x1, y1, x2, y2],   // [0, 1000] normalised
      "severity":  "low | medium | high | critical",
      "reasoning": "Step-by-step chain-of-thought analysis…",
      "action":    "emergency_brake | yield | lane_change | maintain_speed"
    }
  ],
  "scene_summary": "One-sentence scene description.",
  "ego_context": {
    "weather":     "clear | rain | fog | snow",
    "time_of_day": "day | night | dusk | dawn",
    "road_type":   "highway | urban | rural | intersection"
  }
}
```"""


def create_demo() -> object:
    """Build and return the Gradio Blocks interface.

    Returns:
        gr.Blocks instance.

    Raises:
        ImportError: If gradio is not installed.
    """
    if not _GRADIO_AVAILABLE or gr is None:
        raise ImportError("gradio not available. Install: pip install gradio>=4.0")

    examples = _get_example_images()

    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Dashcam Frame",
                    type="pil",
                    image_mode="RGB",
                )
                run_btn = gr.Button("Detect Hazards", variant="primary")
                latency_label = gr.Textbox(label="Latency", interactive=False)

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Annotated Detection",
                    type="pil",
                )
                output_json = gr.Code(
                    label="Structured JSON Output",
                    language="json",
                    lines=20,
                )

        run_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[output_image, output_json, latency_label],
        )

        if examples and gr is not None:
            gr.Examples(
                examples=examples,
                inputs=[input_image],
                label="Example Dashcam Frames",
            )

        with gr.Accordion("Output Schema", open=False):
            gr.Markdown(SCHEMA_DOC)

        with gr.Accordion("About this model", open=False):
            gr.Markdown("""
**Architecture**
- Base model: Qwen3-VL-2B-Instruct (Vision-Language Model)
- Fine-tuning: LoRA (rank 32, alpha 64) on hazard detection data
- Quantization: AWQ 4-bit (LLM decoder only; ViT stays in fp16)

**Training data**
- nuScenes: rare-event frames filtered by rarity score ≥ 3/6
- DADA-2000: pre-accident dashcam critical moments
- LLM-augmented counterfactuals: scenario-based synthetic examples

**Performance** (HPC benchmark)
| Backend | Mean latency | Throughput |
|---------|-------------|-----------|
| PyTorch eager | ~45 ms | ~22 fps |
| torch.compile | ~29 ms | ~35 fps |
| TensorRT ViT + vLLM | ~38 ms | ~26 rps |
""")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _GRADIO_AVAILABLE:
        raise SystemExit("gradio not installed. Run: pip install gradio>=4.0")
    app = create_demo()
    app.launch(server_port=7860, share=False)
