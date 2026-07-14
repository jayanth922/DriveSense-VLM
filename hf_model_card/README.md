---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-3B-Instruct
tags:
  - autonomous-driving
  - hazard-detection
  - vision-language-model
  - lora
  - bitsandbytes
  - nf4
datasets:
  - nuScenes
pipeline_tag: image-text-to-text
---

# DriveSense-VLM

**SFT-optimized vision-language model for autonomous-vehicle rare hazard detection.**

DriveSense-VLM is a LoRA-fine-tuned [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
that takes a single dashcam frame and returns structured JSON describing safety-critical
hazards: bounding box, hazard label, severity, chain-of-thought reasoning, and the
recommended ego-vehicle action.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jayanth922/DriveSense-VLM/blob/main/notebooks/05_demo.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-DriveSense--VLM-181717?logo=github)](https://github.com/jayanth922/DriveSense-VLM)

---

## Model details

| | |
|---|---|
| **Base model**       | Qwen/Qwen2.5-VL-3B-Instruct |
| **Adapter**          | LoRA (rank 32, alpha 64), merged into base weights |
| **Quantization**     | bitsandbytes NF4 (4-bit), double-quant, bfloat16 compute |
| **Vision encoder**   | Qwen2.5-VL ViT in fp16 (kept full-precision for grounding accuracy) |
| **Output schema**    | JSON: `hazards[]{bbox_2d, label, severity, reasoning, action}`, `scene_summary`, `ego_context` |
| **Image resolution** | 672 × 448 (16h × 24w = 384 patches at 28×28 patch size) |

---

## Training

| | |
|---|---|
| **Dataset**       | 688 rare-hazard nuScenes v1.0-trainval frames (549/72/67), 3D-GT-projected boxes |
| **Epochs**        | 8 |
| **Train loss**    | 0.75 (train ≈ val, no overfitting) |
| **LoRA targets**  | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` |
| **Hardware**      | Single A100 |

---

## Evaluation

### Detection quality

Level-1 grounding, v2 test set (n = 67; measured, reported exactly as observed):

| Metric | Value |
|---|---|
| Output parse rate          | 76% |
| Detection F1 @ IoU 0.5     | 1.4% |
| Detection Recall @ IoU 0.5 | 1.0% |
| Mean best-pair IoU         | 0.12 |
| Frame detect-rate @ IoU 0.1 / 0.3 / 0.5 | 33% / 20% / 5% |

These numbers are low and reported honestly — the small, narrow training set (688 frames, ~8
nuScenes logs, mostly daytime) is the dominant limiter. See Limitations.

### Demo quantization

The T4 Spaces demo loads the model with bitsandbytes NF4 (4-bit, double-quant, bf16 compute) to
fit the free-tier 16 GB GPU. This is a memory-fit measure for the demo; no separately benchmarked
compression/latency numbers are claimed here.

---

## Quick start

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import torch

REPO = "jayanth922/DriveSense-VLM"

processor = AutoProcessor.from_pretrained(REPO)
model = AutoModelForImageTextToText.from_pretrained(
    REPO,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

PROMPT = (
    "Analyze this dashcam image for safety hazards. Return JSON with hazards array "
    "containing bbox_2d (normalized 0-1000), label, severity (low/medium/high/critical), "
    "reasoning, and action for each hazard. Include scene_summary and ego_context "
    "(weather, time_of_day, road_type)."
)

image = Image.open("dashcam.jpg").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text":  PROMPT},
]}]
text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)

print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Intended use

- **Portfolio / research demonstration** of VLM fine-tuning, quantization, and grounding for
  the autonomous-driving domain.
- **Educational** reference implementation of a structured-output VLM pipeline.

**Not intended for**: deployment in any safety-critical or production autonomous-driving system.

---

## Limitations

- **Low grounding accuracy** — Detection F1 @ IoU 0.5 is 1.4% and mean best-pair IoU is 0.12;
  the model localizes something near a hazard on ~1/3 of frames but rarely reaches IoU 0.5.
- **Small, narrow training set** — 688 frames from ~8 nuScenes logs, predominantly daytime, two
  cities; expect degraded performance on dashcams that differ in mounting, FoV, or conditions.
- **Dense-frame parse failures (~24%)** — on the densest multi-hazard frames the output exceeds
  the generation token budget and ends mid-JSON (16/67 test frames, ~1,164 tokens at the 1024
  cap). Root cause is undertraining on a small dataset; hazard-object repetition may contribute.
- **No temporal context** — single-frame inference; hazards needing motion cues are weaker.
- **Quantization noise** — the NF4 demo introduces a small accuracy delta vs. bf16.

---

## Files

| File | Purpose |
|---|---|
| `*.safetensors`            | NF4-quantized merged model weights |
| `config.json`              | Model architecture + quantization config |
| `quant_config.json`        | bitsandbytes quantization metadata |
| `tokenizer*`, `*.json`     | Processor / tokenizer / chat template |
| `examples/*.jpg`           | Sample dashcam frames for the Gradio demo |
| `README.md`                | This model card |

---

## Links

- **GitHub repo**: <https://github.com/jayanth922/DriveSense-VLM>
- **Colab demo**: [`notebooks/05_demo.ipynb`](https://colab.research.google.com/github/jayanth922/DriveSense-VLM/blob/main/notebooks/05_demo.ipynb)
- **Base model**: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- **Dataset**: [nuScenes](https://www.nuscenes.org/) (v1.0-trainval)

## License

Apache-2.0. Inherits the [Qwen2.5-VL license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)
for the base weights.
