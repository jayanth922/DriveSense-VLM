#!/usr/bin/env python3
"""DriveSense-VLM inference benchmark v2 — industry-shaped, single-GPU (Colab T4).

Bottleneck-driven: decode is memory-bandwidth-bound, so we measure the effect of
(a) quantization and (b) prompt-lookup speculative decoding, and we report the metrics a
serving team actually uses:
  * TTFT (time-to-first-token), TPOT/ITL (per-token latency, median + p95)
  * end-to-end p50 / p95 / p99 latency
  * decode tokens/s + roofline utilization (% of the GPU HBM ceiling)
  * throughput vs batch size (tokens/s and requests/s) — the serving metric
  * an output-EQUIVALENCE quality gate vs the FP16 baseline (works without task labels:
    prompt-lookup is greedy-equivalent -> exact match; quantization drift is bounded)
  * optional vLLM comparison (--vllm) as an optimized-runtime reference

Usage (Colab T4):
  pip install "transformers>=4.46,<5" accelerate bitsandbytes qwen-vl-utils pillow
  python inference_benchmark.py --model Qwen/Qwen2.5-VL-3B-Instruct \
      --images a.jpg b.jpg c.jpg --runs 3 --max-new-tokens 256 \
      --configs fp16 fp16+lookup nf4 nf4+lookup int8 --batches 1 2 4
"""
import argparse, json, time, statistics as st, difflib
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
T4_HBM_GBs = 320.0
WEIGHT_GB = {"fp16": 6.0, "nf4": 2.2, "int8": 3.5}


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
    return xs[k]


def build_inputs(processor, image_paths):
    """Batched inputs for one or more images (padding handles the batch)."""
    texts, imgs = [], []
    for ip in image_paths:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": [{"type": "image", "image": ip},
                                             {"type": "text", "text": USER}]}]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        imgs.append(Image.open(ip).convert("RGB"))
    return processor(text=texts, images=imgs, padding=True, return_tensors="pt")


def load_model(model_id, adapter, quant):
    kw = dict(dtype=torch.float16, device_map="cuda")
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    elif quant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    m = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m


@torch.no_grad()
def latency_quality(cfg, model, processor, images, max_new, runs, baseline_out):
    lookup = "lookup" in cfg
    gen_kw = dict(max_new_tokens=max_new, do_sample=False)
    if lookup:
        gen_kw["prompt_lookup_num_tokens"] = 10
    ttft, tpot, e2e, tokps = [], [], [], []
    outputs = {}
    ins0 = build_inputs(processor, [images[0]]).to("cuda")
    _ = model.generate(**ins0, **gen_kw)                       # warmup
    torch.cuda.reset_peak_memory_stats()
    for _ in range(runs):
        for img in images:
            ins = build_inputs(processor, [img]).to("cuda")
            n_in = ins["input_ids"].shape[1]
            torch.cuda.synchronize(); a = time.perf_counter()
            _ = model.generate(**ins, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize(); b = time.perf_counter()          # TTFT
            out = model.generate(**ins, **gen_kw)
            torch.cuda.synchronize(); c = time.perf_counter()          # full
            gen = out.shape[1] - n_in
            e2e.append(c - b); ttft.append((b - a))
            dec = max(c - b - (b - a), 1e-6)                            # decode-only time
            if gen > 1:
                tpot.append(dec / (gen - 1)); tokps.append((gen - 1) / dec)
            outputs[img] = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    peak = torch.cuda.max_memory_allocated() / 1e9
    # output-equivalence vs FP16 baseline
    if baseline_out is None:
        exact, sim = None, None
    else:
        ex, sm = [], []
        for img in images:
            a, b = baseline_out.get(img, ""), outputs.get(img, "")
            ex.append(1.0 if a == b else 0.0)
            sm.append(difflib.SequenceMatcher(None, a, b).ratio())
        exact, sim = round(sum(ex) / len(ex), 3), round(sum(sm) / len(sm), 3)
    med_tokps = st.median(tokps) if tokps else 0.0
    return outputs, dict(
        peak_vram=round(peak, 2),
        ttft_ms=round(1000 * st.median(ttft), 1),
        tpot_ms=round(1000 * st.median(tpot), 1) if tpot else 0.0,
        tpot_p95_ms=round(1000 * pct(tpot, 95), 1) if tpot else 0.0,
        decode_tokps=round(med_tokps, 1),
        e2e_p50=round(pct(e2e, 50), 2), e2e_p95=round(pct(e2e, 95), 2), e2e_p99=round(pct(e2e, 99), 2),
        roofline_pct=round(100 * med_tokps * WEIGHT_GB.get(cfg.split("+")[0], 6.0) / T4_HBM_GBs, 1),
        exact_match_vs_fp16=exact, char_sim_vs_fp16=sim)


@torch.no_grad()
def throughput(model, processor, images, batches, max_new=64):
    out = {}
    for B in batches:
        try:
            batch_imgs = (images * ((B // len(images)) + 1))[:B]
            ins = build_inputs(processor, batch_imgs).to("cuda")
            n_in = ins["input_ids"].shape[1]
            _ = model.generate(**ins, max_new_tokens=4, do_sample=False)   # warmup
            torch.cuda.synchronize(); t0 = time.perf_counter()
            o = model.generate(**ins, max_new_tokens=max_new, do_sample=False)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            new = (o.shape[1] - n_in) * B
            out[B] = dict(tok_s=round(new / dt, 1), req_s=round(B / dt, 3))
        except Exception as e:
            out[B] = f"OOM/err: {type(e).__name__}"
            torch.cuda.empty_cache(); break
    return out


def run_hf(args, processor):
    order = args.configs[:]
    if "fp16" in order:                      # ensure baseline first
        order.remove("fp16"); order = ["fp16"] + order
    rows, baseline_out = [], None
    for cfg in order:
        quant = cfg.split("+")[0]
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            model = load_model(args.model, args.adapter, quant)
            load_vram = torch.cuda.max_memory_allocated() / 1e9
            outs, m = latency_quality(cfg, model, processor, args.images, args.max_new_tokens,
                                      args.runs, baseline_out)
            m = {"config": cfg, "load_vram": round(load_vram, 2), **m}
            if args.batches:
                m["throughput"] = throughput(model, processor, args.images, args.batches)
            if cfg == "fp16":
                baseline_out = outs
            rows.append(m); print("done:", json.dumps(m), flush=True)
            del model; torch.cuda.empty_cache()
        except Exception as e:
            print(f"CONFIG {cfg} FAILED: {type(e).__name__}: {e}", flush=True)
    return rows


def run_vllm(args):
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        print(f"[vllm] not available ({e}); skipping optimized-runtime reference.", flush=True)
        return None
    try:
        t0 = time.perf_counter()
        llm = LLM(model=args.model, dtype="float16", max_model_len=4096,
                  limit_mm_per_prompt={"image": 1}, enforce_eager=False)
        sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
        prompts = [{"prompt": f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n"
                    f"<|vision_start|><|image_pad|><|vision_end|>{USER}<|im_end|>\n<|im_start|>assistant\n",
                    "multi_modal_data": {"image": Image.open(i).convert('RGB')}} for i in args.images]
        _ = llm.generate(prompts, sp)                       # warmup
        s = time.perf_counter(); res = llm.generate(prompts, sp); dt = time.perf_counter() - s
        tok = sum(len(o.outputs[0].token_ids) for o in res)
        return {"engine": "vllm", "n": len(prompts), "wall_s": round(dt, 2),
                "throughput_tok_s": round(tok / dt, 1), "load_s": round(s - t0, 1)}
    except Exception as e:
        print(f"[vllm] run failed: {type(e).__name__}: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--configs", nargs="+",
                    default=["fp16", "fp16+lookup", "nf4", "nf4+lookup", "int8"])
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--out", default="inference_benchmark_results.json")
    a = ap.parse_args()

    processor = AutoProcessor.from_pretrained(
        a.model, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    print(f"GPU: {torch.cuda.get_device_name(0)} | model={a.model} | adapter={a.adapter}", flush=True)

    rows = run_hf(a, processor)
    vllm = run_vllm(a) if a.vllm else None

    # summary table
    cols = ["config", "load_vram", "peak_vram", "ttft_ms", "tpot_ms", "decode_tokps",
            "e2e_p50", "e2e_p95", "roofline_pct", "exact_match_vs_fp16", "char_sim_vs_fp16"]
    print("\n=== LATENCY / QUALITY (T4, batch 1) ===")
    print(" | ".join(f"{c:>13}" for c in cols))
    for r in rows:
        print(" | ".join(f"{str(r.get(c,'-')):>13}" for c in cols))
    print("\n=== THROUGHPUT (tokens/s @ batch) ===")
    for r in rows:
        tp = r.get("throughput", {})
        cells = ", ".join(f"b{B}={v['tok_s'] if isinstance(v,dict) else v}" for B, v in tp.items())
        print(f"  {r['config']:>12}: {cells}")
    if vllm:
        print("\n=== vLLM reference ===\n ", json.dumps(vllm))
    best_rl = max((r.get("roofline_pct", 0) for r in rows), default=0)
    print(f"\nRoofline: best decode reaches ~{best_rl}% of the T4 HBM ceiling (~{T4_HBM_GBs:.0f} GB/s) "
          f"-> decode is memory-bandwidth-bound (quantization + prompt-lookup target exactly this).")
    print("Quality gate: prompt-lookup should show exact_match_vs_fp16 ~1.0 (greedy-equivalent); "
          "quantization char_sim_vs_fp16 should stay high (bounded drift).")
    Path(a.out).write_text(json.dumps({"hf": rows, "vllm": vllm}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
