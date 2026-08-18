#!/usr/bin/env python3
r"""DriveSense-VLM inference optimization benchmark (Colab-ready, free T4).

Bottleneck-driven study: decode is memory-bandwidth-bound, so we measure the effect of
(a) quantization (move fewer bytes) and (b) prompt-lookup speculative decoding (take fewer
steps), each with a parse-rate quality gate. Prints a Pareto table + roofline utilization.

Usage (Colab):
  pip install -q "transformers>=4.46,<5" accelerate bitsandbytes qwen-vl-utils pillow
  python inference_benchmark.py \
     --model Qwen/Qwen2.5-VL-3B-Instruct \
     --adapter jayanth7111/DriveSense-VLM \        # optional; omit for base model
     --images img1.jpg img2.jpg img3.jpg \
     --runs 3 --max-new-tokens 512 \
     --configs fp16 fp16+lookup nf4 nf4+lookup int8

Notes:
- Processor pinned to training resolution (min_pixels 256 / max_pixels 768 -> 200704/602112)
  so boxes stay in the 0-1000 space (no coordinate drift).
- Prompt-lookup uses HF generate(prompt_lookup_num_tokens=N); it is greedy-equivalent, so
  output is identical to the plain greedy baseline -> a free latency win on structured JSON.
- T4 HBM bandwidth ~320 GB/s is the memory roofline used for the utilization sentence.
"""
import argparse, json, time, statistics as st
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

SYS = ("You are DriveSense, an autonomous vehicle hazard detection system. Analyze the "
       "dashcam image and identify all safety-critical hazards. Output a structured JSON "
       "response with bounding boxes (normalized 0-1000), hazard classification, severity, "
       "reasoning, and recommended action.")
USER = ("Analyze this dashcam image for safety hazards. Identify all hazards with bounding "
        "boxes, classify each, assess severity, explain your reasoning, and recommend an "
        "action. Respond with JSON only.")
T4_HBM_GBs = 320.0  # memory roofline for the utilization estimate


def build_inputs(processor, image_path):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": [{"type": "image", "image": image_path},
                                         {"type": "text", "text": USER}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img = Image.open(image_path).convert("RGB")
    return processor(text=[text], images=[img], return_tensors="pt")


def load_model(model_id, adapter, mode):
    kw = dict(torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    if mode == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    elif mode == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    m = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m


def parse_ok(text):
    import re
    mt = re.search(r"\{.*\}", text, re.DOTALL)
    if not mt:
        return False
    try:
        d = json.loads(mt.group(0))
        return isinstance(d.get("hazards"), list)
    except Exception:
        return False


@torch.no_grad()
def bench_config(cfg, model_id, adapter, processor, images, max_new, runs, weight_gb):
    quant = cfg.split("+")[0]
    lookup = "lookup" in cfg
    compile_ = "compile" in cfg
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = load_model(model_id, adapter, quant)
    load_vram = torch.cuda.max_memory_allocated() / 1e9
    if compile_:
        model.forward = torch.compile(model.forward, mode="reduce-overhead")
    gen_kw = dict(max_new_tokens=max_new, do_sample=False)
    if lookup:
        gen_kw["prompt_lookup_num_tokens"] = 10

    prefills, tokps, e2es, oks = [], [], [], 0
    # warmup
    ins = build_inputs(processor, images[0]).to("cuda")
    _ = model.generate(**ins, **gen_kw)
    torch.cuda.reset_peak_memory_stats()
    for r in range(runs):
        for img in images:
            ins = build_inputs(processor, img).to("cuda")
            n_in = ins["input_ids"].shape[1]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            first = model.generate(**{**ins}, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            out = model.generate(**ins, **gen_kw)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            gen = out.shape[1] - n_in
            prefills.append((t1 - t0) * 1000)
            if gen > 1 and (t2 - t1) > 0:
                tokps.append((gen - 1) / (t2 - t1))
            e2es.append(t2 - t0)
            txt = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
            oks += int(parse_ok(txt))
    peak = torch.cuda.max_memory_allocated() / 1e9
    n = runs * len(images)
    med_tokps = st.median(tokps) if tokps else 0.0
    row = dict(config=cfg, load_vram=round(load_vram, 2), peak_vram=round(peak, 2),
               prefill_ms=round(st.median(prefills), 1),
               decode_tokps=round(med_tokps, 1),
               e2e_s=round(st.median(e2es), 2),
               parse_rate=round(oks / n, 3),
               roofline_pct=round(100 * (med_tokps * weight_gb) / T4_HBM_GBs, 1) if med_tokps else 0.0)
    del model; torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--configs", nargs="+",
                    default=["fp16", "fp16+lookup", "nf4", "nf4+lookup", "int8"])
    a = ap.parse_args()

    # weight bytes streamed per decode step (approx; for the roofline utilization estimate)
    gb = {"fp16": 6.0, "nf4": 2.2, "int8": 3.5}
    processor = AutoProcessor.from_pretrained(
        a.model, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    print(f"GPU: {torch.cuda.get_device_name(0)} | model={a.model} | adapter={a.adapter}")
    rows = []
    for cfg in a.configs:
        try:
            w = gb.get(cfg.split("+")[0], 6.0)
            r = bench_config(cfg, a.model, a.adapter, processor, a.images,
                             a.max_new_tokens, a.runs, w)
            rows.append(r); print("done:", r, flush=True)
        except Exception as e:  # keep going; one bad config shouldn't kill the sweep
            print(f"CONFIG {cfg} FAILED: {type(e).__name__}: {e}", flush=True)

    hdr = ["config", "load_vram", "peak_vram", "prefill_ms", "decode_tokps", "e2e_s",
           "parse_rate", "roofline_pct"]
    print("\n=== INFERENCE PARETO (T4) ===")
    print(" | ".join(f"{h:>12}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{str(r[h]):>12}" for h in hdr))
    Path("inference_benchmark_results.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote inference_benchmark_results.json")
    if rows:
        base = next((r for r in rows if r["config"] == "fp16"), rows[0])
        print(f"\nRoofline: best decode reaches ~{max(r['roofline_pct'] for r in rows)}% of the "
              f"T4 HBM ceiling (~{T4_HBM_GBs:.0f} GB/s) — decode is memory-bandwidth-bound, "
              f"which is exactly what quantization + prompt-lookup target.")
        print(f"Baseline fp16 e2e={base['e2e_s']}s; check parse_rate stays within ~1pt of "
              f"baseline for every config (quality gate).")


if __name__ == "__main__":
    main()
