"""DriveSense-VLM Gradio Demo — HuggingFace Spaces (standalone).

Standalone Gradio app for HF Spaces deployment. Does NOT depend on the
repo's src/ package — all inference logic is inlined here.

Hardware target: Free T4 GPU on HuggingFace Spaces.

Usage (local):
    MODEL_PATH=./model python demo/app.py

Usage (HF Spaces):
    Push this file to a HuggingFace Space with gradio as the SDK.
    Set Space hardware to T4 Small (free tier).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw

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

MODEL_PATH = os.environ.get("MODEL_PATH", "./model")
PROCESSOR_PATH = os.environ.get("PROCESSOR_PATH", MODEL_PATH)

PROMPT = (
    "Analyze this dashcam image for safety hazards. Return JSON with hazards array "
    "containing bbox_2d (normalized 0-1000), label, severity (low/medium/high/critical), "
    "reasoning, and action for each hazard. Include scene_summary and ego_context "
    "(weather, time_of_day, road_type)."
)

SEVERITY_COLORS: dict[str, tuple[int, int, int]] = {
    "critical": (255, 0, 0),
    "high": (255, 140, 0),
    "medium": (255, 215, 0),
    "low": (50, 205, 50),
    "no_hazard": (65, 105, 225),
}

# ---------------------------------------------------------------------------
# Global model (lazy-loaded on first inference call)
# ---------------------------------------------------------------------------

_model: object = None
_processor: object = None


def _load_model() -> object:
    """Lazy-load model and processor on first call. Returns model or None."""
    global _model, _processor  # noqa: PLW0603
    if _model is not None:
        return _model
    try:
        import torch  # type: ignore[import]
        from transformers import (  # type: ignore[import]
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        logger.info("Loading model from %s …", MODEL_PATH)
        _processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        _model.eval()  # type: ignore[union-attr]
        logger.info("Model loaded")
    except Exception as exc:  # noqa: BLE001
        logger.error("Model load failed: %s", exc)
        _model = None
        _processor = None
    return _model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict:
    """Extract first JSON object from model output; strip ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3].rstrip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"hazards": [], "scene_summary": text, "ego_context": {}}


def draw_hazard_boxes(image: Image.Image, annotation: dict) -> Image.Image:
    """Overlay severity-coded bounding boxes on a PIL Image.

    Args:
        image:      Input PIL Image.
        annotation: Parsed annotation dict with ``hazards`` list.

    Returns:
        New PIL Image with boxes drawn.
    """
    w, h = image.size
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for hazard in annotation.get("hazards", []):
        bbox = hazard.get("bbox_2d", [])
        if len(bbox) != 4:
            continue
        sev = str(hazard.get("severity", "no_hazard")).lower()
        label = str(hazard.get("label", "hazard"))
        color = SEVERITY_COLORS.get(sev, SEVERITY_COLORS["no_hazard"])

        x1 = int(bbox[0] * w / 1000)
        y1 = int(bbox[1] * h / 1000)
        x2 = int(bbox[2] * w / 1000)
        y2 = int(bbox[3] * h / 1000)

        draw.rectangle([x1, y1, x2, y2], fill=(*color, 50))
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=2)
        text = f"{label} ({sev})"
        draw.text((x1 + 2, max(0, y1 - 18)), text, fill=(*color, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


def analyze_image(
    image: Image.Image | None,
    max_tokens: int = 300,
) -> tuple[Image.Image | None, str, str]:
    """Run hazard detection on a dashcam image.

    Args:
        image:      Input PIL Image from Gradio.
        max_tokens: Maximum new tokens for generation.

    Returns:
        Tuple of (annotated_image, json_str, status_str).
    """
    if image is None:
        return None, "Please upload a dashcam image.", "—"

    model = _load_model()

    if model is None or _processor is None:
        placeholder = _make_placeholder_result()
        return image, json.dumps(placeholder, indent=2), "⚠️ Model not loaded"

    try:
        import torch  # type: ignore[import]

        image = image.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}]
        text = _processor.apply_chat_template(  # type: ignore[union-attr]
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _processor(  # type: ignore[union-attr]
            text=[text], images=[image], return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(  # type: ignore[union-attr]
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        ms = (time.perf_counter() - t0) * 1000

        raw = _processor.decode(  # type: ignore[union-attr]
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        annotation = _parse_json(raw)
        annotated = draw_hazard_boxes(image, annotation)
        n = len(annotation.get("hazards", []))
        return annotated, json.dumps(annotation, indent=2), f"✓ {n} hazard(s) | {ms:.0f} ms"

    except Exception as exc:  # noqa: BLE001
        logger.error("Inference error: %s", exc)
        return image, json.dumps({"error": str(exc), "hazards": []}, indent=2), "⚠️ Error"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_placeholder_result() -> dict:
    """Return a placeholder when the model is unavailable."""
    return {
        "hazards": [{
            "label": "pedestrian_in_path",
            "bbox_2d": [120, 80, 350, 280],
            "severity": "high",
            "reasoning": "⚠️ Model not loaded — placeholder output.",
            "action": "yield",
        }],
        "scene_summary": "Model not loaded.",
        "ego_context": {"weather": "unknown", "time_of_day": "unknown", "road_type": "unknown"},
    }


def _get_example_images() -> list[list]:
    """Collect up to 6 example images from demo/examples/."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in sorted(EXAMPLES_DIR.iterdir()) if p.suffix.lower() in exts]
    return [[str(p)] for p in paths[:6]]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

TITLE = "DriveSense-VLM: Autonomous Vehicle Hazard Detection"

DESCRIPTION = (
    "SFT-optimized Qwen2.5-VL-3B for rare hazard detection. NF4 4-bit quantized (2.4 GB).\n\n"
    "**Severity:** 🔴 Critical &nbsp; 🟠 High &nbsp; 🟡 Medium &nbsp; 🟢 Low"
)


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
            with gr.Column():
                input_image = gr.Image(label="Dashcam Frame", type="pil", image_mode="RGB")
                max_tok = gr.Slider(50, 500, value=300, step=10, label="Max tokens")
                run_btn = gr.Button("Detect Hazards", variant="primary")
                status_lbl = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                output_image = gr.Image(label="Annotated Detection", type="pil")
                output_json = gr.Code(
                    label="Structured JSON Output", language="json", lines=20
                )

        run_btn.click(
            fn=analyze_image,
            inputs=[input_image, max_tok],
            outputs=[output_image, output_json, status_lbl],
        )

        if examples and gr is not None:
            gr.Examples(examples=examples, inputs=[input_image], label="Example Dashcam Frames")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _GRADIO_AVAILABLE:
        raise SystemExit("gradio not installed. Run: pip install gradio>=4.0")
    app = create_demo()
    app.launch(server_port=7860, share=False)
