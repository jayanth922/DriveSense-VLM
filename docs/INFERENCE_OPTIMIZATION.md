# Inference optimization study

[Back to the project README](../README.md)

This study measures the Qwen2.5-VL-3B DriveSense inference path on one NVIDIA T4. It compares
precision, speculative decoding, and batch size while checking whether the generated output stays
equivalent to the fp16 baseline.

## Bottleneck

Each request has two main phases:

- **Prefill:** encode the image and prompt once.
- **Decode:** generate the structured JSON one token at a time.

The output is long enough that decode dominates end-to-end latency. At batch size 1, each decode
step streams roughly 6 GB of fp16 weights from T4 HBM and performs relatively little arithmetic.
The measured baseline uses 31.8% of the T4's 320 GB/s memory bandwidth, so this workload is
memory-bandwidth-bound rather than compute-bound.

This diagnosis suggests three useful tests:

1. reduce the number of decode steps with prompt-lookup speculative decoding;
2. reduce weight memory with quantization;
3. amortize weight reads across requests with batching.

## Benchmark method

[scripts/inference_benchmark.py](../scripts/inference_benchmark.py) records:

- time to first token (TTFT);
- time per output token (TPOT);
- end-to-end p50, p95, and p99 latency;
- decode tokens per second;
- peak GPU memory;
- estimated HBM roofline utilization;
- aggregate throughput by batch size;
- exact match and character similarity against fp16 output.

The processor uses the same image limits as training: min_pixels=200704 and max_pixels=602112. The
benchmark uses greedy decoding so prompt lookup can be compared exactly with the baseline.

Example:

    python scripts/inference_benchmark.py \
      --model Qwen/Qwen2.5-VL-3B-Instruct \
      --adapter /path/to/lora_adapter \
      --images frame1.jpg frame2.jpg frame3.jpg \
      --runs 3 \
      --max-new-tokens 256 \
      --configs fp16 fp16+lookup nf4 nf4+lookup int8 \
      --batches 1 2 4 \
      --out inference_benchmark_results.json

## Measured results

The main run used one T4 with 16 GB VRAM and 320 GB/s HBM bandwidth. Batch size is 1 unless noted.

| Configuration | Decode tok/s | TTFT | TPOT | End-to-end p50 | Weight memory | HBM utilization | Output vs fp16 |
|---|---:|---:|---:|---:|---:|---:|---|
| fp16 | 17.0 | 770 ms | 59.0 ms | 11.64 s | about 6.0 GB | 31.8% | reference |
| fp16 + prompt lookup | 20.4 | 781 ms | 49.1 ms | 9.79 s | about 6.0 GB | 38.2% | exact match 1.00 |
| NF4 | 12.6 | 770 ms | 79.4 ms | about 15.9 s | 2.63 GB | 10.4% | character similarity 0.359 |
| NF4 + prompt lookup | 15.4 | 770 ms | 64.9 ms | about 13.0 s | 4.42 GB peak allocated | not reported | character similarity 0.338 |
| INT8 | 4.6 | about 1,080 ms | 217 ms | 53.53 s | about 3.5 GB | 5.0% | character similarity 0.289 |

### Batch throughput

Aggregate decode throughput:

| Batch size | 1 | 2 | 4 |
|---|---:|---:|---:|
| fp16 | 14.9 | 23.8 | 33.7 |
| NF4 | 11.1 | 18.2 | 29.0 |

The 17.0-token/s fp16 figure is decode-only throughput. The 14.9-token/s batch-size-1 figure
includes prefill and should be compared with the batch-size-4 result. Using the same definition,
batching improves aggregate fp16 throughput by about 2.3x.

## Verification run

A second run used one benchmark harness on a Kaggle T4 with the base Qwen2.5-VL-3B model. It
checked the weight constants, TTFT behavior, and the distinction between decode-only and
end-to-end throughput. It did not repeat the LoRA output-quality comparison.

| Configuration | Decode tok/s | TTFT | HBM utilization |
|---|---:|---:|---:|
| fp16 | 17.8 | 770 ms | 33.5% |
| NF4 | 13.3 | 820 ms | 11.0% |
| INT8 | 5.0 | 1,084 ms | 5.5% |

In the same run, fp16 measured 17.8 decode-only tokens/s and 14.8 end-to-end tokens/s. End-to-end
throughput increased 2.43x from batch size 1 to batch size 4.

The verification also corrected three earlier reporting issues:

- the two fp16 batch-size-1 numbers use different definitions rather than representing run-to-run
  variance;
- the NF4 weight constant is 2.63 GB, not 2.2 GB;
- TTFT is not identical across precision modes and increases with quantization.

## Findings

### Prompt lookup improves latency without changing output

The prompt and output share schema keys, labels, brackets, and other repeated token sequences.
Prompt lookup drafts these sequences and verifies them against the model. Incorrect drafts are
discarded.

This raised decode throughput from 17.0 to 20.4 tokens/s and reduced median end-to-end latency from
11.64 to 9.79 seconds. Output matched fp16 exactly.

### NF4 saves memory but is slower at batch size 1

NF4 reduced weight memory from about 6.0 GB to 2.63 GB. It did not improve latency. The cost of
dequantizing weights on every step outweighed the memory-bandwidth saving in this low-batch
workload, reducing decode throughput to 12.6 tokens/s.

NF4 output also diverged from fp16 with character similarity 0.359. For a model that emits numeric
coordinates, that difference can move a box and must be followed by the full grounding and
robustness evaluation.

### INT8 is not suitable for this T4 path

The bitsandbytes INT8 configuration delivered 4.6 tokens/s and 53.53-second median latency. It was
both slower and less similar to the fp16 output than NF4, so it is not recommended here.

### Batching is the best throughput option

At batch size 4, fp16 aggregate throughput reached 33.7 tokens/s without an out-of-memory error.
This is useful for offline evaluation and labeling, where total throughput matters more than one
request's latency.

## Recommendation

- Use **fp16 with prompt lookup** for the lowest single-request latency measured here.
- Use **fp16 at batch size 4** for offline throughput.
- Use **NF4 only when memory capacity is the main constraint**, followed by Level 1 and Level 4
  quality evaluation.
- Do not use the tested INT8 path on a T4.

The measured p50 remains 9.79 seconds even with prompt lookup. This is not a real-time inference
path and is not presented as one.

## Measurement boundaries

- vLLM did not run successfully in the T4 environment, so no vLLM row is reported.
- Similarity values compare optimized output with fp16 output, not with ground truth.
- The 4.42 GB NF4-plus-lookup value is peak allocated VRAM, not weight size.
- torch.compile is not included in the measured table.
- TensorRT was investigated separately at the vision-encoder level; see
  [TENSORRT_RUNBOOK.md](TENSORRT_RUNBOOK.md).
