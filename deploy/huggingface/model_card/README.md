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

DriveSense-VLM is Qwen2.5-VL-3B-Instruct fine-tuned with LoRA for structured hazard analysis in
dashcam frames. It returns 2D boxes, one of seven hazard labels, severity, a short explanation, and
a recommended driving action.

[GitHub repository](https://github.com/jayanth922/DriveSense-VLM) |
[Colab demo](https://colab.research.google.com/github/jayanth922/DriveSense-VLM/blob/main/notebooks/05_demo.ipynb)

This is a research model for offline evaluation. Do not use it for vehicle control or other
safety-critical decisions.

## Model details

| Field | Value |
|---|---|
| Base model | Qwen/Qwen2.5-VL-3B-Instruct |
| Fine-tuning | LoRA, rank 32, alpha 64 |
| LoRA targets | q/k/v/o and up/down projections |
| Selected version | v3 |
| Training split | 7,228 train / 889 validation / 1,041 test |
| Training hardware | one A100 |
| Demo format | bitsandbytes NF4 on a T4 |

## Data and supervision

The selected v3 model uses nuScenes v1.0-trainval CAM_FRONT frames. Its 2D boxes are projected from
nuScenes 3D annotations. Claude supplies severity, explanation, and action text for those boxes; it
does not localize them.

A later v4 candidate added 1,442 targeted records with Claude-emitted boxes. That candidate was
blocked by the regression gate. A controlled follow-up showed that GT-projected boxes produced
better localization than foundation-model-emitted boxes. Details are in the
[box-provenance report](https://github.com/jayanth922/DriveSense-VLM/blob/main/docs/TASK3_DECONFOUND.md).

## Evaluation

v3 and v4 were evaluated on the same 1,041-frame test set.

### Grounding at IoU >= 0.5

| Version | Train frames | Precision | Recall | F1 | Mean IoU | Class accuracy | Parse rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3, selected | 7,228 | 0.40 | 0.24 | 0.30 | 0.67 | 0.94 | 98.7% |
| v4, blocked | 8,670 | 0.37 | 0.19 | 0.25 | 0.656 | 0.946 | 97.4% |

The model is better at localizing hazards it finds than at finding all hazards. Recall is the main
limitation.

### Detection by condition

| Bucket | v3 | v4 |
|---|---:|---:|
| Overall | 28.0% | 23.0% |
| Tiny boxes | 22.8% | 17.2% |
| Small boxes | 46.4% | 42.9% |
| Medium boxes | 52.6% | 52.6% |
| Rain | 12.5% | 7.4% |
| Night and tiny | 12.7% | 10.7% |
| Clear and medium | 69.0% | 69.0% |

v4 became worse on all three protected weak buckets, so it was not promoted.

### Reasoning quality

Claude Sonnet 5 judged 1,027 v3 outputs on a 1-5 scale.

| Dimension | Score |
|---|---:|
| Correctness | 3.03 |
| Completeness | 2.66 |
| Action relevance | 3.80 |
| Overall | 3.16 |
| Pass rate, all dimensions >= 3.5 | 26% |

## T4 inference measurements

| Configuration | Decode tok/s | TTFT | End-to-end p50 | Weight memory | Output comparison |
|---|---:|---:|---:|---:|---|
| fp16 | 17.0 | 770 ms | 11.64 s | about 6.0 GB | reference |
| fp16 + prompt lookup | 20.4 | 781 ms | 9.79 s | about 6.0 GB | exact match 1.00 |
| NF4 | 12.6 | 820 ms | about 15.9 s | 2.63 GB | character similarity 0.36 |

Prompt lookup improved median latency by 16% without changing output. NF4 reduced memory by about
2.3x but made decoding slower and changed the generated text. At batch size 4, fp16 aggregate
throughput increased from 14.9 to 33.7 tokens/s.

See the full
[inference study](https://github.com/jayanth922/DriveSense-VLM/blob/main/docs/INFERENCE_OPTIMIZATION.md)
and its
[benchmark script](https://github.com/jayanth922/DriveSense-VLM/blob/main/scripts/inference_benchmark.py).

## Usage

    from PIL import Image
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    repo = "jayanth922/DriveSense-VLM"
    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(
        repo,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    image = Image.open("dashcam.jpg").convert("RGB")
    prompt = (
        "Analyze this dashcam frame. Return JSON containing hazards with bbox_2d "
        "in normalized 0-1000 coordinates, label, severity, reasoning, and action."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    prompt_tokens = inputs["input_ids"].shape[1]
    print(processor.decode(output[0][prompt_tokens:], skip_special_tokens=True))

## Limitations

- Recall is 24% at IoU 0.5.
- Tiny, distant, rainy, and nighttime hazards perform worst.
- Training data is predominantly daytime nuScenes data.
- Inference uses a single frame and has no motion context.
- NF4 changes output relative to fp16.
- Measured T4 latency is several seconds per image, not real time.
- Reasoning scores come from an LLM judge rather than a human safety review.

## Files

| File | Purpose |
|---|---|
| safetensors files | model or adapter weights |
| config and tokenizer files | model, processor, and chat configuration |
| quant_config.json | bitsandbytes quantization settings |
| examples | sample images used by the demo |

## License

Apache 2.0. Base weights remain subject to the
[Qwen2.5-VL license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE).
