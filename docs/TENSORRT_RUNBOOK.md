# TensorRT edge-deployment runbook + results (executed on Kaggle T4)

> Part of **[DriveSense-VLM](../README.md)** — was item 2 of
> [What's left](../README.md#whats-left-future-work).

**Status: EXECUTED on a Kaggle T4 (2026-08-23). Result — TensorRT ViT export is NOT
viable for Qwen2.5-VL, with a single named root cause (data-dependent window attention).
See [§6 Results](#6-results--executed-on-kaggle-t4).** The runbook (§§1–5) is retained as
the plan that was followed; §6 records what actually happened. This is a documented negative
finding, not a claimed speedup.

> ⚠️ §2 below asserted the vision encoder has "no dynamic shapes" and is "export-friendly."
> The real run **disproved that** for Qwen2.5-VL specifically — its ViT uses data-dependent
> window attention. Read §6 for the correction.

## 1. Diagnosis of the "prior TensorRT failure"

I grepped the full repo (`tensorrt`, `trt`, `onnx`, saved logs, `fallback_info.json`)
before writing anything. Here's exactly what the evidence shows — and doesn't.

**There is no evidence in this repo of a real GPU TensorRT export ever having
been attempted.** The only artifact was `outputs/tensorrt/fallback_info.json`
(untracked, gitignored), and tracing it down:

- Its `torch_compile_sentinel` field pointed at a path under
  `/private/var/.../pytest-of-jayan/pytest-14/test_torch_compile_sentinel_pa0/...`
  — a **pytest temp directory**, not a Colab run.
- `torch` is not installed anywhere in this repo's local environment, so no
  real `tensorrt_vit.py` code path could have executed here.
- Tracing the exact string back: `tests/test_tensorrt.py::test_torch_compile_sentinel_path`
  used a fixture (`trt_config`) whose `output_dir` was hardcoded to the real
  repo-relative string `"outputs/tensorrt"` instead of pytest's isolated
  `tmp_path`. Running that **unit test** locally — with `tensorrt` correctly
  absent, exercising the intentional fallback path — wrote a real
  `fallback_info.json` into the repo as a side effect. This is a **test-hygiene
  bug**, not a GPU failure. I fixed it (`tests/test_tensorrt.py`, one test,
  now uses `tmp_path` for `output_dir`) and deleted the stray file; verified
  the test suite no longer leaks into the real path.
- The only speedup numbers anywhere in the repo are `1.57x` (torch.compile)
  and `3.65x` (tensorrt) in `_mock_benchmark()` / `run_optimize_model.py` —
  these are **explicit mock placeholders** for `--mock` CI runs, never real
  measurements. I could not find `2.07x` anywhere in the repo, in any file,
  committed or not.

**So: whatever "earlier attempt... FAILED" you're recalling isn't reflected
in this codebase's history or local state.** It may be from a real Colab
session whose artifacts weren't saved back to the repo, or a conflation with
this test fixture's fallback-path output. Either way, there's no known root
cause to "address" from repo evidence — the honest starting position for next
session is: **the TensorRT path has never actually been run against a real
model on real hardware.** Plan accordingly (budget for first-attempt
debugging, not "fixing a known issue").

## 2. Scope: ViT-only, not full-model — already the design, and it's the right one

Good news: `src/drivesense/inference/tensorrt_vit.py` **already scopes
TensorRT to the vision encoder only**. `ViTExtractor.extract_vit()` pulls out
only `model.visual`/`vision_tower` via `_get_vision_encoder()` — the full
pipeline (`full_pipeline()`) never touches the LLM decoder. This isn't a
change to propose next session; it's the existing architecture, and it's the
correct scope:

- **Vision encoder**: standard conv/attention on a **fixed** input resolution
  (672×448, no dynamic shapes), no autoregressive loop, no KV-cache — exactly
  the profile TensorRT/ONNX handle well.
- **LLM decoder**: autoregressive generation with a growing KV-cache, dynamic
  sequence lengths, and (depending on the transformers version) custom fused
  attention kernels — TensorRT's static-graph model fights all three. This is
  not a DriveSense-specific limitation; it's a general property of
  autoregressive decoder export that the wider ecosystem (TensorRT-LLM aside,
  which is a different, purpose-built tool with its own separate integration
  cost) treats the same way.

**Confirm scope explicitly at the start of next session**, don't assume: run
`extract_vit()` and print `type(vit).__name__` plus `sum(p.numel() for p in vit.parameters())`
against the real checkpoint before exporting anything, so the ONNX export
step 1 targets a module you've actually inspected.

## 3. Colab A100 runbook

### Step 0 — environment

```bash
# Fresh Colab A100 runtime.
pip install -e '.[training]'          # torch, transformers, peft, etc.
pip install tensorrt onnx onnxsim     # NOT installed locally — first real test of this path
python -c "import tensorrt; print(tensorrt.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

**Decision point A:** if `pip install tensorrt` fails or the import fails
(common on Colab — TensorRT's pip wheel has had version-matrix issues with
Colab's CUDA/driver combos), **stop here and go straight to the honest
fallback framing in §5** — don't burn the session fighting a broken install.
Try `pip install tensorrt --extra-index-url https://pypi.nvidia.com` as the
one alternative worth 10 minutes; if that also fails, move on.

### Step 1 — confirm the ViT extraction target

```bash
python -c "
from pathlib import Path
from drivesense.inference.tensorrt_vit import ViTExtractor
from drivesense.utils.config import load_config, merge_configs

cfg = merge_configs(load_config('configs/model.yaml'), load_config('configs/inference.yaml'))
extractor = ViTExtractor(cfg)
vit, proc = extractor.extract_vit(Path('outputs/quantized_model'))  # or your merged/base model dir
print(type(vit).__name__, sum(p.numel() for p in vit.parameters()) / 1e6, 'M params')
"
```

**Decision point B:** if `_get_vision_encoder()` raises `AttributeError`
("Cannot locate vision encoder"), the installed `transformers` version's
Qwen2.5-VL wrapper has renamed the attribute. Fix: `print(model.named_children())`
manually, add the real attribute name to the `("vision_tower", "visual",
"vision_model")` tuple in `_get_vision_encoder()`, re-run. This is a 2-line
fix, not a redesign.

### Step 2 — ONNX export (existing code, first real invocation)

```bash
python scripts/run_optimize_model.py --tensorrt --model-dir outputs/quantized_model
# equivalent to: ViTExtractor(cfg).full_pipeline(model_dir, "outputs/tensorrt")
```

This runs, in order: `export_to_onnx()` (direct `torch.onnx.export`, opset 17,
fixed shape — falls back to `torch.jit.trace` on failure) → `compile_tensorrt()`
(falls back to `torch.compile` if TRT unavailable/fails) → `benchmark_vit()`
(times whichever backends are actually available) → writes
`outputs/tensorrt/{vit.onnx, vit.engine or vit.torch_compile, vit_benchmark.json,
optimization_report.txt, fallback_info.json}`.

**Decision point C — read `fallback_info.json` after this step, before
anything else:**

| `onnx_method` | Meaning | Next action |
|---|---|---|
| `"direct"` | Clean `torch.onnx.export` | Proceed to Step 3 |
| `"jit_trace"` | Direct export hit a custom op; trace fallback worked | Proceed, but note in write-up that trace-based export was needed (less robust to future shape/version changes) |
| `"failed"` | Both export paths failed | **Stop the ONNX/TRT track.** Read `onnx_direct_error`/`onnx_trace_error` in the file — if it names a specific op (common culprits: rotary embeddings, flash-attention ops, or Qwen's custom windowed-attention op in the ViT), that op is the concrete, named root cause for the write-up. Go to §5. |

| `trt_method` (only relevant if `onnx_method` succeeded) | Meaning | Next action |
|---|---|---|
| `"tensorrt"` | Engine built and verified (`vit.engine` exists, reloads successfully) | Proceed to Step 3 |
| `"torch_compile"` with `trt_error` present | TRT parser rejected the ONNX graph | Read `trt_error` — usually a specific unsupported ONNX op/opset mismatch. That's your named root cause. Go to §5. |
| `"torch_compile"` with only `trt_note` (no `trt_error`) | TensorRT package itself unavailable (Decision Point A already should have caught this) | Go to §5 |

### Step 3 — the three-way benchmark comparison

`benchmark_vit()` already measures PyTorch eager vs. `torch.compile` vs.
TensorRT in one call (Step 2 does this automatically when `vit.engine`
exists). To get the **end-to-end model** comparison you actually want
(baseline / NF4+compile / TensorRT-ViT-in-the-loop) in the same p50/p95/p99 +
VRAM format as the existing Level-3 production metrics
(`src/drivesense/eval/production.py`, targets in `configs/eval.yaml`'s
`production:` block — `vit_tensorrt_latency_ms: 25` is the existing target to
compare against):

```bash
# (a) baseline PyTorch, full model, eager
python scripts/run_benchmark.py --local --image-dir outputs/data/nuscenes_filtered/images \
    --config configs/inference.yaml --num-iterations 100 --warmup 10 \
    --output outputs/benchmarks/a_baseline_eager.json

# (b) NF4-quantized + torch.compile — the already-measured path; re-run on
#     THIS session's actual checkpoint so (a)/(b)/(c) are on identical images/model
python scripts/run_benchmark.py --local --image-dir outputs/data/nuscenes_filtered/images \
    --config configs/inference.yaml --num-iterations 100 --warmup 10 \
    --output outputs/benchmarks/b_nf4_compile.json
# (confirm configs/inference.yaml points at the NF4 model + torch.compile is enabled
#  for this run — check demo/app.py's loading path for the exact flags it sets)

# (c) TensorRT ViT swapped into the full pipeline
python scripts/run_benchmark.py --local --vit-only --image-dir outputs/data/nuscenes_filtered/images \
    --config configs/inference.yaml --num-iterations 100 --warmup 10 \
    --output outputs/benchmarks/c_tensorrt_vit.json
```

**Decision point D:** `--vit-only` benchmarks the ViT in isolation (already
built), not the full model with a TensorRT ViT spliced into `model.generate()`
— splicing a standalone TensorRT engine's output back into the HF generation
loop as a drop-in replacement for `model.visual(...)` is **not wired up
anywhere in this codebase today**. Two honest options for (c):
1. **Report ViT-only numbers for (c)**, clearly labeled "vision encoder only,
   not full end-to-end" — this is what the code actually supports today, and
   is still a meaningful, real number (the ViT forward pass is a real
   fraction of total latency).
2. If full end-to-end TensorRT-in-the-loop numbers are wanted, that requires
   writing the splice (replace `model.visual.forward` with a wrapper that
   calls the TRT engine, matching output tensor shape/dtype) — budget this as
   **additional, unbudgeted work** for next session, not something the
   existing `run_benchmark.py --vit-only` flag already does despite the name
   similarity.

### Step 4 — assemble the comparison report

```bash
python -c "
import json
from pathlib import Path
for name in ('a_baseline_eager', 'b_nf4_compile', 'c_tensorrt_vit'):
    d = json.loads(Path(f'outputs/benchmarks/{name}.json').read_text())
    print(name, d)
"
# Or, reusing the project's own comparison tooling (built this session):
python scripts/compare_eval_runs.py \
    --run baseline=outputs/benchmarks/a_baseline_eager.json \
    --run nf4_compile=outputs/benchmarks/b_nf4_compile.json \
    --run tensorrt_vit=outputs/benchmarks/c_tensorrt_vit.json \
    --format markdown
```

`scripts/compare_eval_runs.py` (from the observability layer built earlier
this engagement) reads `eval_summary.json`-shaped dicts under a `"level1"`
key by default — the benchmark JSONs here have a different shape (latency
stats, not grounding metrics), so this will need `--dimensions`/metric-path
adjustment, or just read the three JSON files directly as shown in the first
snippet. Don't force-fit an eval tool onto benchmark data if the shapes
genuinely don't match — flag it and report the three JSONs side by side by
hand if the comparison script doesn't adapt cleanly.

## 4. What "success" looks like, concretely

A Level-3-style table, e.g.:

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | VRAM (GB) | vs. target |
|---|---|---|---|---|---|
| PyTorch eager (baseline) | ? | ? | ? | ? | — |
| NF4 + torch.compile | ? | ? | ? | ? | (already-measured ratio, re-confirm this session) |
| TensorRT ViT (vision-encoder-only, see Decision Point D) | ? | ? | ? | ? | target: `vit_tensorrt_latency_ms < 25ms` (p50) |

Fill in `?` from Step 3/4's real output — do not estimate or carry over
numbers from `_mock_benchmark()`, which are fabricated placeholders, not
measurements, and are already flagged as such throughout the codebase.

## 5. Honest fallback framing, if TensorRT genuinely doesn't export cleanly

If Decision Points A/C/D lead here — **use this framing, don't force a bad
export to claim TensorRT was "used":**

> TensorRT was evaluated specifically for the vision encoder (the
> export-friendly component: fixed-resolution conv/attention, no
> autoregressive loop). [ONNX export succeeded / failed at op `X`] and
> [TensorRT compilation succeeded / failed with error `Y`]. The full
> autoregressive decoder was not attempted — it is not a good TensorRT export
> target for this model class (dynamic KV-cache shapes, custom fused
> attention), which is a real architectural property of autoregressive LLMs
> generally, not a shortfall specific to this project. [If ViT export
> succeeded: report the real ViT-only speedup, honestly scoped as such.] [If
> it also failed: NF4 quantization + torch.compile remains the deployed
> optimization path, at the already-measured Xx speedup — TensorRT was
> evaluated and found not viable for this architecture at this time, which is
> itself a documented finding, not a gap.]

This is a legitimate, defensible result either way — a negative finding with
a named root cause (a specific unsupported op, a specific dynamic-shape
constraint) is more credible than a forced partial win, and costs nothing to
report honestly.

## 6. Results — EXECUTED on Kaggle T4

Run 2026-08-23 on a Kaggle T4 with `Qwen/Qwen2.5-VL-3B-Instruct` (base model — its vision
encoder is identical to the fine-tuned model's, since LoRA does not touch the vision tower).
Input 448×672 → 1536 patches. **ViT-only, not full end-to-end.**

### 6.1 Harness bug found and fixed first

The export/benchmark harness fed the ViT a `[1, 3, 448, 672]` **image** tensor at
`patch_size=28`. Qwen2.5-VL's vision encoder does not take images — it takes **pre-patchified**
input `[seq_len, in_ch·temporal·patch²] = [1536, 1176]` plus `grid_thw=[[1, 32, 48]]` at
`patch_size=14`. That mismatch produced a `1280 vs 640` hidden-size crash. Fixed with
`_make_vit_inputs()` in `src/drivesense/inference/tensorrt_vit.py`; `full_pipeline` was also
made resilient (an export failure no longer aborts before benchmarking). Two stale unit tests
(`tests/test_tensorrt.py`, which encoded the wrong `patch_size=28` / 384-patch assumption) were
corrected. With the fix the ViT forward runs cleanly (output `[384, 2048]`).

### 6.2 Measured (T4, p50 over 50 iters, 10 warmup)

| Backend | ViT p50 (ms) | vs eager | Status |
|---|---|---|---|
| PyTorch eager | 203.1 | 1.00× | baseline |
| torch.compile (default) | 197.9 | **1.03×** | compiles only with a graph break; negligible gain |
| torch.compile (reduce-overhead) | — | — | CUDA-graph mode incompatible with the graph break |
| TensorRT (via ONNX) | — | — | **export fails — not viable** |

### 6.3 Root cause (single, named)

Qwen2.5-VL's vision encoder uses **data-dependent window attention**: `get_window_index()`
builds `cu_window_seqlens` via `cu_seqlens_tmp.tolist()` / `.item()`
(`transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:400`). Data-dependent control flow
defeats every graph-capture backend:

- **`torch.export`** (ONNX/TensorRT step 1) raises
  `GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u16*u9, 0)`
  at that line.
- **`torch.jit.trace` → ONNX** raises `Exporting a ScriptModule is not supported`.
- **`torch.compile`** succeeds only by inserting a graph break at that op — which is exactly
  why its speedup is ~1.03× (nothing meaningful to fuse across the break).

### 6.4 Conclusion

TensorRT is **not viable** for this ViT without rewriting the window-index construction to be
static-shape / graph-capturable — a real architectural property of Qwen2.5-VL, not a project
shortfall. `torch.compile` is technically available but delivers no meaningful ViT speedup for
the same reason. **Eager (203 ms p50 on a T4) is the practical ViT execution**, and the
deployed latency lever remains **fp16 + prompt-lookup** (see `INFERENCE_OPTIMIZATION.md §7`),
not TensorRT. Reported as a documented negative with a named root cause rather than a forced
partial win — which was the explicit intent of §5.
