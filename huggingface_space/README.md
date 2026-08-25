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

# DriveSense-VLM — Autonomous Vehicle Hazard Detection

Interactive demo for **DriveSense-VLM**, an SFT-optimized
[Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) fine-tuned with
LoRA for **rare, safety-critical hazard detection** in dashcam footage.

Upload a dashcam frame and the model returns:

- **Bounding boxes** around each hazard, colour-coded by severity
- **Structured JSON**: `bbox_2d`, `label`, `severity`, `reasoning`, `action`
- A **scene summary** and **ego context** (weather, time of day, road type)

## How it works

The model is loaded from the Hub repo
[`jayanth7111/DriveSense-VLM`](https://huggingface.co/jayanth7111/DriveSense-VLM)
(NF4 4-bit quantized) and runs on a single T4 GPU. Expect **~20–40 s per image**.

Severity colours: 🔴 Critical &nbsp; 🟠 High &nbsp; 🟡 Medium &nbsp; 🟢 Low

## Notes

- Outputs are advisory and intended for research/evaluation — not for direct
  vehicle control.
- Override the model source by setting the `MODEL_REPO` environment variable.
