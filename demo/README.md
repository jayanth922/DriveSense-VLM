---
title: DriveSense-VLM
emoji: 🚗
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Rare driving hazard detection with Qwen3-VL-2B
---

# DriveSense-VLM — AV Rare Hazard Detection

Upload a dashcam image to detect rare driving hazards.

**Model:** Qwen3-VL-2B-Instruct + LoRA SFT (AWQ 4-bit quantized)

**Output:** Structured JSON with bounding boxes, hazard labels,
severity (low/medium/high/critical), chain-of-thought reasoning,
and ego-vehicle action recommendation.

## Severity Colour Key

| Colour | Severity |
|--------|----------|
| 🔴 Red | Critical |
| 🟠 Orange | High |
| 🟡 Yellow | Medium |
| 🟢 Green | Low |
| 🔵 Blue | No hazard |

## Output Schema

```json
{
  "hazards": [
    {
      "label":     "pedestrian_in_path",
      "bbox_2d":   [x1, y1, x2, y2],
      "severity":  "high",
      "reasoning": "Step-by-step analysis…",
      "action":    "yield"
    }
  ],
  "scene_summary": "One-sentence scene description.",
  "ego_context": {
    "weather":     "clear",
    "time_of_day": "day",
    "road_type":   "urban"
  }
}
```

## Hardware

Runs on HuggingFace Spaces free T4 GPU using AWQ 4-bit quantized inference.
