"""DriveSense-VLM — HuggingFace Spaces app.

Standalone Gradio app that loads the published model from the HuggingFace Hub
(``jayanth7111/DriveSense-VLM``) and serves an interactive hazard-detection demo.

Deploy by pushing this directory (app.py + requirements.txt + README.md) to a
HuggingFace Space with the Gradio SDK. Set the Space hardware to T4 Small.
"""

from __future__ import annotations

import html
import json
import logging
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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

APP_DIR = Path(__file__).parent
EXAMPLES_DIR = APP_DIR / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

# Published model + processor live in one HF Hub repo.
MODEL_REPO = os.environ.get("MODEL_REPO", "jayanth7111/DriveSense-VLM")
PROCESSOR_REPO = os.environ.get("PROCESSOR_REPO", MODEL_REPO)

# 200 tokens is enough for full structured output and faster than 300.
DEFAULT_MAX_TOKENS = 200

PROMPT = (
    "Analyze this dashcam image for safety hazards. Return JSON with hazards array "
    "containing bbox_2d (normalized 0-1000), label, severity (low/medium/high/critical), "
    "reasoning, and action for each hazard. Include scene_summary and ego_context "
    "(weather, time_of_day, road_type)."
)

# Legacy RGB palette — kept as a fallback for the box renderer.
SEVERITY_COLORS: dict[str, tuple[int, int, int]] = {
    "critical": (255, 0, 0),
    "high": (255, 140, 0),
    "medium": (255, 215, 0),
    "low": (50, 205, 50),
    "no_hazard": (65, 105, 225),
}

# Modern UI design palette (hex) — used for box overlays, cards and badges.
SEVERITY_HEX: dict[str, str] = {
    "critical": "#DC2626",
    "high": "#EA580C",
    "medium": "#CA8A04",
    "low": "#16A34A",
    "no_hazard": "#2563EB",
}

# ---------------------------------------------------------------------------
# Global model (lazy-loaded on first inference call)
# ---------------------------------------------------------------------------

_model: object = None
_processor: object = None


def _load_model() -> object:
    """Lazy-load model and processor from the HF Hub. Returns model or None."""
    global _model, _processor  # noqa: PLW0603
    if _model is not None:
        return _model
    try:
        import torch  # type: ignore[import]
        from transformers import (  # type: ignore[import]
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        logger.info("Loading model from HF Hub: %s …", MODEL_REPO)
        _processor = AutoProcessor.from_pretrained(PROCESSOR_REPO)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_REPO,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        _model.eval()  # type: ignore[union-attr]
        logger.info("Model loaded from %s", MODEL_REPO)
    except Exception as exc:  # noqa: BLE001
        logger.error("Model load failed: %s", exc)
        _model = None
        _processor = None
    return _model


# ---------------------------------------------------------------------------
# Formatting / colour helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert ``#RRGGBB`` to an ``(R, G, B)`` tuple."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _severity_hex(severity: str) -> str:
    """Return the hex colour for a severity, defaulting to ``no_hazard``."""
    return SEVERITY_HEX.get(str(severity).lower(), SEVERITY_HEX["no_hazard"])


def _severity_rgb(severity: str) -> tuple[int, int, int]:
    """Return the RGB colour for a severity (hex palette → RGB)."""
    return _hex_to_rgb(_severity_hex(severity))


def _format_latency(ms: float) -> str:
    """Format a millisecond latency, switching to seconds above 1000 ms."""
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f} ms"


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


def _get_font(size: int = 16) -> object:
    """Return a TrueType font at the requested size, falling back to default."""
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return None


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: object) -> tuple[int, int]:
    """Measure rendered text size, compatible across Pillow versions."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except Exception:  # noqa: BLE001
        return len(text) * 8, 14


def draw_hazard_boxes(image: Image.Image, annotation: dict) -> Image.Image:
    """Overlay severity-coded bounding boxes on a full-brightness image copy.

    Args:
        image:      Input PIL Image.
        annotation: Parsed annotation dict with a ``hazards`` list.

    Returns:
        New PIL Image (same size as input) with boxes drawn.
    """
    base = image.convert("RGB").copy()
    draw = ImageDraw.Draw(base)
    font = _get_font(16)
    w, h = base.size

    for hazard in annotation.get("hazards", []):
        bbox = hazard.get("bbox_2d", [])
        if len(bbox) != 4:
            continue
        sev = str(hazard.get("severity", "no_hazard")).lower()
        label = str(hazard.get("label", "hazard"))
        rgb = _severity_rgb(sev)

        x1 = int(bbox[0] * w / 1000)
        y1 = int(bbox[1] * h / 1000)
        x2 = int(bbox[2] * w / 1000)
        y2 = int(bbox[3] * h / 1000)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=4)

        text = f"{label} · {sev}"
        tw, th = _text_size(draw, text, font)
        ty = y1 - th - 8 if y1 - th - 8 >= 0 else y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 6], fill=rgb)
        draw.text((x1 + 5, ty + 3), text, fill=(255, 255, 255), font=font)

    return base


def _run_model(image: Image.Image, max_tokens: int) -> tuple[dict, float]:
    """Run inference and return ``(annotation, latency_ms)``."""
    model = _load_model()
    if model is None or _processor is None:
        return _make_placeholder_result(), 0.0

    import torch  # type: ignore[import]

    image = image.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ]}]
    text = _processor.apply_chat_template(  # type: ignore[union-attr]
        messages, tokenize=False, add_generation_prompt=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = _processor(  # type: ignore[union-attr]
        text=[text], images=[image], return_tensors="pt"
    ).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(  # type: ignore[union-attr]
            **inputs, max_new_tokens=max_tokens, do_sample=False
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000

    raw = _processor.decode(  # type: ignore[union-attr]
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return _parse_json(raw), ms


# ---------------------------------------------------------------------------
# HTML rendering for the summary panel and hazard cards
# ---------------------------------------------------------------------------


def _badge(text: str, bg: str = "#475569") -> str:
    """Return an inline HTML badge span."""
    return (
        f'<span style="display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;'
        f'border-radius:9999px;background:{bg};color:#fff;font-size:12px;'
        f'font-weight:600;">{html.escape(text)}</span>'
    )


def _summary_html(annotation: dict, latency_ms: float) -> str:
    """Build the summary panel: hazard count, latency, scene summary, ego badges."""
    hazards = annotation.get("hazards", [])
    n = len(hazards)
    count_color = "#16A34A" if n == 0 else "#DC2626"
    scene = annotation.get("scene_summary", "") or "—"
    ego = annotation.get("ego_context", {}) or {}

    ego_badges = "".join(
        _badge(f"{k.replace('_', ' ')}: {ego[k]}")
        for k in ("weather", "time_of_day", "road_type")
        if ego.get(k)
    ) or '<span style="color:#94a3b8;font-size:13px;">No scene context</span>'

    return f"""
<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;
            padding:14px 16px;border:1px solid #e2e8f0;border-radius:12px;
            background:#f8fafc;">
  <div style="text-align:center;min-width:90px;">
    <div style="font-size:38px;font-weight:800;line-height:1;color:{count_color};">{n}</div>
    <div style="font-size:12px;color:#64748b;text-transform:uppercase;
                letter-spacing:.5px;">hazard{'s' if n != 1 else ''}</div>
  </div>
  <div style="border-left:1px solid #e2e8f0;padding-left:16px;flex:1;min-width:200px;">
    <div style="font-size:13px;color:#334155;margin-bottom:6px;">
      ⏱ <b>{html.escape(_format_latency(latency_ms))}</b> inference</div>
    <div style="font-size:13px;color:#475569;margin-bottom:8px;">
      {html.escape(scene)}</div>
    <div>{ego_badges}</div>
  </div>
</div>
""".strip()


def _hazard_card(hazard: dict) -> str:
    """Build one hazard card with severity colour coding."""
    sev = str(hazard.get("severity", "no_hazard")).lower()
    color = _severity_hex(sev)
    label = str(hazard.get("label", "hazard"))
    reasoning = str(hazard.get("reasoning", "") or "—")
    action = str(hazard.get("action", "") or "—")

    return f"""
<div style="border:1px solid #e2e8f0;border-left:5px solid {color};
            border-radius:10px;padding:12px 14px;margin-bottom:10px;background:#fff;">
  <div style="display:flex;align-items:center;justify-content:space-between;
              margin-bottom:6px;">
    <span style="font-size:15px;font-weight:700;color:{color};">
      {html.escape(label)}</span>
    {_badge(sev.upper(), color)}
  </div>
  <div style="font-size:13px;color:#334155;margin-bottom:6px;">
    <b>Why:</b> {html.escape(reasoning)}</div>
  <div style="font-size:13px;color:#334155;">
    <b>Action:</b> {html.escape(action)}</div>
</div>
""".strip()


def _hazards_html(annotation: dict) -> str:
    """Render all hazard cards, or a clear empty state."""
    hazards = annotation.get("hazards", [])
    if not hazards:
        return (
            '<div style="padding:18px;text-align:center;border:1px dashed #86efac;'
            'border-radius:12px;background:#f0fdf4;color:#15803d;font-weight:600;">'
            "✓ No hazards detected — clear scene</div>"
        )
    return "".join(_hazard_card(h) for h in hazards)


def _info_html(message: str) -> str:
    """Render a neutral informational panel."""
    return (
        '<div style="padding:18px;text-align:center;border:1px dashed #cbd5e1;'
        'border-radius:12px;background:#f8fafc;color:#64748b;">'
        f"{html.escape(message)}</div>"
    )


# ---------------------------------------------------------------------------
# Inference entry points
# ---------------------------------------------------------------------------


def analyze_image(
    image: Image.Image | None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[Image.Image | None, str, str]:
    """Run hazard detection (back-compat 3-tuple API)."""
    if image is None:
        return None, "Please upload a dashcam image.", "—"
    try:
        annotation, ms = _run_model(image, max_tokens)
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference error: %s", exc)
        return image, json.dumps({"error": str(exc), "hazards": []}, indent=2), "⚠️ Error"

    annotated = draw_hazard_boxes(image, annotation)
    n = len(annotation.get("hazards", []))
    return annotated, json.dumps(annotation, indent=2), f"✓ {n} hazard(s) | {_format_latency(ms)}"


def analyze(
    image: Image.Image | None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[Image.Image | None, str, str, str]:
    """Run hazard detection for the rich Gradio UI.

    Returns:
        Tuple of (annotated_image, summary_html, hazards_html, json_str).
    """
    if image is None:
        return None, _info_html("Upload a dashcam image to begin."), "", "{}"
    try:
        annotation, ms = _run_model(image, max_tokens)
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference error: %s", exc)
        return (
            image,
            _info_html(f"Inference error: {exc}"),
            "",
            json.dumps({"error": str(exc), "hazards": []}, indent=2),
        )

    annotated = draw_hazard_boxes(image, annotation)
    return (
        annotated,
        _summary_html(annotation, ms),
        _hazards_html(annotation),
        json.dumps(annotation, indent=2),
    )


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
    """Collect up to 6 example images from the examples/ directory."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in sorted(EXAMPLES_DIR.iterdir()) if p.suffix.lower() in exts]
    return [[str(p)] for p in paths[:6]]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

TITLE = "DriveSense-VLM: Autonomous Vehicle Hazard Detection"

DESCRIPTION = (
    "Upload a dashcam frame and DriveSense-VLM detects rare, safety-critical road "
    "hazards — drawing bounding boxes and explaining the risk and recommended ego action. "
    "It returns structured JSON: per-hazard box, label, severity, reasoning and action, "
    "plus a scene summary and ego context.\n\n"
    "> ⏱ Inference runs on T4 GPU (~20–40s per image). "
    "Model: **Qwen2.5-VL-3B**, NF4 quantized.\n\n"
    "**Severity:** 🔴 Critical &nbsp; 🟠 High &nbsp; 🟡 Medium &nbsp; 🟢 Low"
)


def create_demo() -> object:
    """Build and return the Gradio Blocks interface.

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
                input_image = gr.Image(label="Dashcam Frame", type="pil", image_mode="RGB")
                max_tok = gr.Slider(
                    50, 500, value=DEFAULT_MAX_TOKENS, step=10, label="Max tokens"
                )
                run_btn = gr.Button("Detect Hazards", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Annotated Detection", type="pil")
                summary_panel = gr.HTML(value=_info_html("Upload a dashcam image to begin."))
                hazards_panel = gr.HTML()
                with gr.Accordion("Raw JSON output", open=False):
                    output_json = gr.Code(label="", language="json", lines=18)

        run_btn.click(
            fn=analyze,
            inputs=[input_image, max_tok],
            outputs=[output_image, summary_panel, hazards_panel, output_json],
        )

        if examples and gr is not None:
            gr.Examples(examples=examples, inputs=[input_image], label="Example Dashcam Frames")

    return demo


if __name__ == "__main__":
    if not _GRADIO_AVAILABLE:
        raise SystemExit("gradio not installed. Run: pip install gradio>=4.0")
    create_demo().launch()
