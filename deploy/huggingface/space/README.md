---
title: DriveSense-VLM Hazard Detection
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# DriveSense-VLM hazard demo

This Space runs the DriveSense-VLM research model on one dashcam image at a time. The model is
Qwen2.5-VL-3B fine-tuned with LoRA on rare-hazard nuScenes frames.

Upload a dashcam frame and the model returns:

- bounding boxes colored by severity;
- structured JSON with label, severity, explanation, and action;
- a scene summary and basic context fields.

## How it works

The application loads the model from
[`jayanth7111/DriveSense-VLM`](https://huggingface.co/jayanth7111/DriveSense-VLM)
in NF4 format and runs on a single T4 GPU. Response time depends on output length and can take
several seconds per image.

Severity colours: 🔴 Critical &nbsp; 🟠 High &nbsp; 🟡 Medium &nbsp; 🟢 Low

## Notes

- This is a research demo, not a driving or safety system.
- Override the model source by setting the `MODEL_REPO` environment variable.
