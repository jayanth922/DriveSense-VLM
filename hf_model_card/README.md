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
| **Dataset**       | 2,754 nuScenes examples (rarity-filtered + LLM counterfactual augmentation) |
| **Epochs**        | 5 |
| **Eval loss**     | 0.312 |
| **LoRA targets**  | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` |
| **Hardware**      | Google Colab Pro A100 |

---

## Evaluation

### Detection quality

| Metric | Value |
|---|---|
| Parse rate (valid JSON)   | 99.1% |
| Mean IoU                  | 0.550 |
| Severity classification   | 82.9% accuracy |
| F1 (hazard detection)     | 0.107 |

### Optimization

| Metric | Value |
|---|---|
| Compression ratio       | 3.1× (vs. fp16 base) |
| VRAM reduction          | 68% |
| `torch.compile` speedup | 1.48× over eager |

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

- **Low recall (6.1%)** — the model is conservative and frequently misses hazards present in
  the scene; suitable for ranking / triage, not as a sole detector.
- **Label fragmentation** — semantically similar hazards (e.g. `pedestrian_in_path`,
  `pedestrian_crossing`) are treated as distinct classes by the F1 calculator, depressing
  the score.
- **Limited geographic / sensor diversity** — trained on three nuScenes blobs only; expect
  degraded performance on dashcams that differ substantially in mounting, FoV, or weather.
- **No temporal context** — single-frame inference. Hazards that require motion cues (e.g.
  cut-ins, pedestrian intent) are weaker.
- **Quantization noise** — NF4 reduces VRAM but introduces a small accuracy delta vs. fp16.

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
- **Datasets**: [nuScenes](https://www.nuscenes.org/), DADA-2000

## License

Apache-2.0. Inherits the [Qwen2.5-VL license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)
for the base weights.
