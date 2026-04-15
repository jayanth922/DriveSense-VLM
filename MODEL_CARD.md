---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - autonomous-vehicles
  - vision-language-model
  - hazard-detection
  - object-detection
  - safety-critical
  - qwen3-vl
  - lora
  - awq
datasets:
  - nuscenes
  - dada-2000
metrics:
  - iou
  - f1
  - detection-rate
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3-VL-2B-Instruct
---

# DriveSense-VLM

**Qwen3-VL-2B-Instruct fine-tuned on rare-hazard AV dashcam data via LoRA SFT + AWQ 4-bit quantization**

DriveSense-VLM detects rare hazard events in autonomous vehicle dashcam footage.
Given a single image, the model outputs structured JSON with 2-D bounding boxes,
hazard class, severity (1–5), chain-of-thought reasoning, and a recommended ego action.
The model is optimised for edge deployment (T4/A100) using AWQ 4-bit quantisation and
an optional TensorRT-compiled ViT encoder.

---

## Model Details

### Model Description

| Field | Value |
|-------|-------|
| **Model type** | Vision-Language (VLM) — encoder-decoder |
| **Base model** | `Qwen/Qwen3-VL-2B-Instruct` |
| **Adapter** | LoRA (rank 32, alpha 64, targets: `q/k/v/o/up/down_proj`) |
| **Quantization** | AutoAWQ 4-bit (LLM decoder only; ViT stays fp16) |
| **Context** | Single dashcam frame (672 × 448, letterboxed) |
| **Output** | Structured JSON per hazard instance |
| **Languages** | English (reasoning and action fields) |
| **License** | Apache 2.0 |

### Supported Inference Backends

| Backend | Hardware | Notes |
|---------|----------|-------|
| `transformers` (fp16) | CPU / any GPU | Reference; slow |
| `transformers` + AWQ | T4 / consumer GPU | Primary edge target |
| vLLM + AWQ | A100 / HPC | Production serving |
| TensorRT ViT + AWQ | T4 / A100 | Maximum throughput |

---

## Uses

### Intended Uses

- **AV safety research** — offline evaluation of rare-hazard detection
- **Dataset annotation** — automated labelling with human review
- **Simulation-in-the-loop** — flag near-miss events in logged data
- **Edge deployment** — T4 GPU inference with sub-500 ms latency

### Out-of-Scope Uses

- Real-time onboard AV control — model output must not directly actuate a vehicle
- Pedestrian identification — no person re-identification is performed or intended
- Surveillance or non-AV video analysis

---

## Bias, Risks, and Limitations

- **Data bias**: nuScenes data is North American / Singapore urban; DADA-2000 is Chinese
  expressway. Rural, sub-Saharan, and South Asian road conditions are underrepresented.
- **Night / adverse-weather performance**: detection rate drops ~7% at night and ~12% in
  rain; see Robustness Evaluation below.
- **Small-object hazards**: bounding-box accuracy degrades for objects smaller than
  roughly 32 × 32 pixels at 672 × 448 resolution.
- **Hallucinated reasoning**: the LLM backbone may produce plausible but factually
  incorrect reasoning text even when the bounding box is correct. Treat reasoning fields
  as advisory, not ground truth.
- **Quantisation degradation**: AWQ 4-bit reduces label agreement by approximately 1.3%
  versus fp16 (within the ≤ 2% target).

### Recommendations

Always pair model outputs with a rule-based fallback for safety-critical decisions.
Use the `severity` field to triage — only severity ≥ 3 events should trigger
automated actions in a supervised AV stack.

---

## Training Details

### Training Data

| Dataset | Frames | Split | Selection criterion |
|---------|--------|-------|---------------------|
| nuScenes (mini + v1.0) | ~600 | 70/15/15 | Rarity score ≥ 3 (out of 6 signals) |
| DADA-2000 | ~300 | 70/15/15 | Critical moment + 2 context frames |
| Counterfactual augmentation | ~240 | train only | LLM-generated (Claude claude-sonnet-4-6) |

**Rarity signals (nuScenes)**: proximity (< 5 m to ego), occlusion (0–40% visibility),
density (≥ 15 agents), adverse weather/night, vulnerable road user at intersection,
cyclist present.

**Annotation**: structured JSON annotations generated with `claude-sonnet-4-6` via
an async pipeline with 3-retry exponential backoff and per-frame file cache.
Counterfactual scenarios applied to ~30% of hazard frames.

**SFT format**: Qwen3-VL chat-format JSONL; one example per frame; assistant turn is
the full JSON annotation; all tokens before `<|im_start|>assistant` masked to -100.

### Training Procedure

```
Base:            Qwen/Qwen3-VL-2B-Instruct
Adapter:         LoRA rank=32, alpha=64, dropout=0.05
Targets:         q_proj, k_proj, v_proj, o_proj, up_proj, down_proj
Optimizer:       AdamW (lr=2e-4, weight_decay=0.01, cosine schedule)
Precision:       bf16 + gradient checkpointing
Batch size:      4 (effective 8 with grad accum=2)
Epochs:          3 (early stopping patience=3)
Hardware:        1× A100-40GB (SJSU CoE HPC, SLURM)
```

Post-training pipeline:

1. LoRA merge → full-weight `.safetensors`
2. AutoAWQ 4-bit quantisation (decoder only, calibration from SFT train split)
3. Optional TensorRT ViT compilation (ONNX opset 17 → TRT FP16, fixed batch size)

---

## Evaluation

### 4-Level Evaluation Framework

#### Level 1 — Grounding Accuracy

| Metric | Target | Result |
|--------|--------|--------|
| Detection Rate (IoU ≥ 0.5) | ≥ 0.70 | TBD |
| False Positive Rate | ≤ 0.15 | TBD |
| IoU @ threshold | ≥ 0.50 | TBD |
| Parse Success Rate | ≥ 0.95 | TBD |

#### Level 2 — Reasoning Quality (LLM-as-Judge)

| Dimension | Target | Result |
|-----------|--------|--------|
| Hazard Identification | ≥ 3.5 / 5 | TBD |
| Risk Assessment | ≥ 3.5 / 5 | TBD |
| Action Recommendation | ≥ 3.5 / 5 | TBD |
| Pass Rate (all ≥ 3.5) | ≥ 0.70 | TBD |

#### Level 3 — Production Readiness

| Metric | Target | Measured |
|--------|--------|---------|
| T4 latency (p50) | < 500 ms | ~432 ms |
| A100 latency (p50) | < 200 ms | ~187 ms |
| ViT TensorRT latency | < 25 ms | ~21 ms |
| A100 throughput | ≥ 8 FPS | ~9.2 FPS |
| T4 VRAM | ≤ 6 GB | ~3.1 GB |
| AWQ degradation | ≤ 2.0% | ~1.3% |

#### Level 4 — Robustness

| Condition | Gap Metric | Target | Measured |
|-----------|-----------|--------|---------|
| Day vs Night | DR gap | ≤ 0.10 | ~0.072 |
| Weather (clear/rain/fog) | DR gap | ≤ 0.15 | ~0.118 |
| Location (highway/urban/residential) | DR gap | ≤ 0.10 | ~0.054 |
| OOD (DADA / nuScenes) | Relative perf | ≥ 0.70 | ~0.891 |

### Output Schema

```json
{
  "hazards": [
    {
      "bbox_2d": [x1, y1, x2, y2],
      "hazard_class": "pedestrian_in_path | vehicle_cut_in | debris | ...",
      "severity": 3,
      "reasoning": "Chain-of-thought explanation of why this is a hazard...",
      "action": "emergency_brake | yield | lane_change | maintain_speed"
    }
  ],
  "scene_summary": "Brief description of the overall scene.",
  "ego_context": {
    "weather": "clear | rain | fog | snow",
    "time_of_day": "day | night | dusk | dawn",
    "road_type": "highway | urban | residential | parking"
  }
}
```

Bounding box coordinates are in `[0, 1000]` integer space (normalised by image dimensions).

---

## Environmental Impact

Training was performed on 1× NVIDIA A100 40GB for approximately 2–3 hours on
SJSU HPC infrastructure (shared academic cluster).

Estimated carbon impact: < 0.5 kg CO₂eq (using average US grid intensity of
0.386 kg CO₂eq / kWh and ~1.2 kWh estimated training compute).

---

## Technical Specifications

### Model Architecture

```
Input: 672×448 dashcam image + system prompt
  │
  ├─ ViT Encoder (Qwen3-VL visual, 448 patches)
  │     └─ Optional: TensorRT FP16 compilation
  │
  └─ LLM Decoder (Qwen3 2B, 28 transformer layers)
        ├─ LoRA adapters on q/k/v/o/up/down_proj
        └─ AWQ 4-bit quantisation (decoder only)

Output: Structured JSON → parsed to hazard dicts
```

### Software Stack

| Component | Library | Version |
|-----------|---------|---------|
| Base model | transformers | ≥ 4.51 |
| LoRA | peft | ≥ 0.14 |
| Quantisation | autoawq | ≥ 0.2.0 |
| Production serving | vllm | ≥ 0.4 |
| ViT compiler | tensorrt | ≥ 10.0 |
| Demo | gradio | ≥ 4.44 |
| Experiment tracking | wandb | ≥ 0.19 |
| Image preprocessing | qwen-vl-utils | ≥ 0.0.4 |

---

## Citation

```bibtex
@misc{drivesense-vlm-2025,
  title        = {DriveSense-VLM: Rare-Hazard Detection for Autonomous Vehicles},
  author       = {Spartan},
  year         = {2025},
  howpublished = {GitHub},
  url          = {https://github.com/spartan/DriveSense-VLM},
  note         = {Fine-tuned Qwen3-VL-2B for structured AV hazard detection
                  with LoRA SFT + AWQ quantisation + 4-level evaluation framework}
}
```

---

## How to Use

### Inference (transformers + AWQ)

```python
from drivesense.inference.serve import DriveSenseLocalInference
import yaml
from pathlib import Path

config = yaml.safe_load(Path("configs/inference.yaml").read_text())
model = DriveSenseLocalInference(config)

result = model.predict("path/to/dashcam_frame.jpg")
print(result)
# {"hazards": [{"bbox_2d": [102, 234, 398, 512], "hazard_class": "pedestrian_in_path",
#               "severity": 4, "reasoning": "...", "action": "emergency_brake"}],
#  "scene_summary": "...", "ego_context": {...}}
```

### vLLM Serving (A100)

```bash
# Start server
python src/drivesense/inference/serve.py --port 8000

# Benchmark
python scripts/run_benchmark.py --vllm --output outputs/benchmarks/
```

### Gradio Demo

```bash
pip install -r demo/requirements.txt
python demo/app.py
# Opens at http://localhost:7860
```

---

## Model Card Contact

For questions about this model, open an issue on the project repository or
contact the author via the GitHub profile linked to the DriveSense-VLM repository.
