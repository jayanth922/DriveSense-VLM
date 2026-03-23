"""DriveSense-VLM Gradio Demo — HuggingFace Spaces.

Phase 4a stub: defines the Gradio interface layout with placeholder outputs.
The full implementation (model loading, inference, bbox overlay) is completed
in Phase 4a after the quantized model is available.

This file is the entry point for the HuggingFace Space.
Hardware target: Free T4 GPU on HuggingFace Spaces.

Usage (local):
    python demo/app.py

Usage (HF Spaces):
    Push this file to a HuggingFace Space with gradio as the SDK.
    Set Space hardware to T4 Small (free tier).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import gradio as gr
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).parent
EXAMPLES_DIR = DEMO_DIR / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

PLACEHOLDER_RESULT = {
    "bbox_2d": [120, 80, 350, 280],
    "hazard_class": "pedestrian_in_path",
    "severity": 4,
    "reasoning": (
        "[Phase 4a — Model not yet loaded] "
        "This placeholder shows the expected output format. "
        "After Phase 3b (AWQ quantization), the model will provide "
        "real chain-of-thought reasoning about the detected hazard."
    ),
    "action": "emergency_brake",
}


def predict(image: Image.Image | None) -> tuple[Image.Image | None, str]:
    """Run DriveSense-VLM hazard detection on a dashcam image.

    Args:
        image: Input PIL Image from Gradio image component.

    Returns:
        Tuple of (annotated_image, json_result_string).
        In Phase 4a stub, returns the input image unchanged + placeholder JSON.
    """
    if image is None:
        return None, "Please upload a dashcam image."

    # Phase 4a TODO: Load quantized model and run inference
    # from drivesense.inference.serve import DriveSenseServer
    # server = DriveSenseServer("configs/inference.yaml")
    # result = server.predict(image)

    # Phase 4a TODO: Overlay bounding box on image
    # from drivesense.utils.visualization import draw_detection
    # annotated = draw_detection(image, result)

    result_json = json.dumps(PLACEHOLDER_RESULT, indent=2)
    return image, result_json


# ── Gradio Interface ────────────────────────────────────────────────────────────

DESCRIPTION = """
## DriveSense-VLM — AV Rare Hazard Detection

Upload a dashcam image to detect rare driving hazards.

**Model:** Qwen3-VL-2B-Instruct + LoRA SFT (AWQ 4-bit)
**Training data:** nuScenes rare-hazard frames + DADA-2000 accident frames
**Output:** Structured JSON with bounding box, hazard class, severity (1–5),
chain-of-thought reasoning, and ego-vehicle action recommendation.

> **Note:** Phase 4a stub — model inference not yet active. Placeholder output shown.
"""

with gr.Blocks(title="DriveSense-VLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# DriveSense-VLM")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Dashcam Frame",
                type="pil",
                image_mode="RGB",
            )
            run_btn = gr.Button("Detect Hazards", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Annotated Detection",
                type="pil",
            )
            output_json = gr.Code(
                label="Structured JSON Output",
                language="json",
                lines=15,
            )

    run_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[output_image, output_json],
    )

    with gr.Accordion("Output Schema", open=False):
        gr.Markdown("""
```json
{
  "bbox_2d": [x1, y1, x2, y2],
  "hazard_class": "pedestrian_in_path | vehicle_cut_in | debris | ...",
  "severity": 1,
  "reasoning": "Step-by-step analysis of the detected hazard...",
  "action": "emergency_brake | yield | lane_change | maintain_speed"
}
```
        """)


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
