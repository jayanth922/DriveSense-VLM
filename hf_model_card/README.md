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
| **Dataset**       | 9,158 rare-hazard nuScenes v1.0-trainval frames (7,228/889/1,041), 3D-GT-projected boxes |
| **Epochs**        | 5 |
| **Train loss**    | 0.40 (eval loss 0.66 — mild overfitting) |
| **LoRA targets**  | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` |
| **Hardware**      | Single A100 |

---

## Evaluation

### Detection quality

Level-1 grounding, v3 test set (n = 1041; measured, reported exactly as observed):

| Metric | Value |
|---|---|
| Output parse rate | 98.7% |
| Detection Precision @ IoU 0.5 | 40% |
| Detection Recall @ IoU 0.5 | 24% |
| Detection F1 @ IoU 0.5 | 30% |
| Mean IoU of matched boxes | 0.67 |
| Mean best-pair IoU | 0.51 |
| Frame detect-rate @ IoU 0.1 / 0.3 / 0.5 | 82% / 75% / 66% |
| Classification accuracy (matched) | 94% |
| Severity within +/-1 / Spearman rho | 98.6% / 0.40 |

The model is **high-precision, well-localized, and conservative**: when it predicts a box it is
usually right (40% precision) and tight (mean IoU 0.67 on matched boxes, above the 0.55 target),
and it names the hazard class correctly 94% of the time. The weak axis is **recall** -- it misses
much of the rare long tail. See Limitations.

> Note: an earlier version of this card reported ~1.4% F1. That was a coordinate-convention bug --
> inference ran the image processor at a different resolution than training, so Qwen2.5-VL fell
> back to its native absolute-pixel box convention and predicted boxes drifted out of the 0-1000
> space the labels use, collapsing every IoU to ~0. The bug is fixed; the numbers above are the
> corrected v3 results.

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

- **Low recall on rare hazards** -- Recall @ IoU 0.5 is 24%. The model is conservative:
  precision (40%) and localization (mean IoU 0.67 on matched boxes) are strong, but it misses much
  of the long tail (e.g. `unusual_object`, 24 instances, is never detected). Undertraining on a
  small, rare-by-construction set is the dominant limiter.
- **Narrow training distribution** -- v3 uses 7,228 train frames from nuScenes v1.0-trainval,
  predominantly daytime; expect degraded performance on dashcams that differ in mounting, FoV, or
  conditions (night / heavy weather are out of distribution).
- **Mild overfitting** -- v3 eval loss (0.66) is roughly double train loss (0.40); fewer epochs or
  earlier stopping would likely lift recall.
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
