# DriveSense-VLM

> SFT-optimized vision-language model for autonomous-vehicle rare hazard detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Model: Qwen2.5-VL-3B](https://img.shields.io/badge/model-Qwen2.5--VL--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

DriveSense-VLM fine-tunes [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
with LoRA SFT to detect and explain rare, safety-critical hazards in autonomous-driving dashcam
frames. Given a single frame, it returns structured JSON: a bounding box per hazard, a 7-class
hazard label, a severity rating, one sentence of reasoning, and a recommended ego-vehicle action.

Training boxes come from nuScenes 3D ground truth projected into the 2D camera frame; a
foundation model (Claude) only writes the severity/reasoning/action text for each real box — it
does not localize. That held for the base v2/v3 training set, but not for one experimental
addition (the v4 targeted-mining turn), which had the model draw boxes directly instead. Task 3,
below, measures how much that provenance switch actually cost.

---

## What's here

An end-to-end pipeline: mine rare frames from nuScenes, auto-label them with a foundation model
behind a validation gate, fine-tune a small VLM to detect and reason about them, then evaluate
across four levels (grounding, reasoning, production readiness, robustness) with a regression
gate that blocks a candidate from replacing the current model. Two full turns of that loop have
run (v2→v3, v3→v4); a controlled ablation (Task 3) followed up on a question the second turn
left open. Inference is separately profiled on a T4 to find the actual bottleneck rather than
guessing at one, and a small MLOps layer (a metrics registry, a regression gate, CI) ties the
whole thing together.

Every detection number below traces to [`results/metrics_registry.json`](results/metrics_registry.json);
every inference number traces to [`INFERENCE_OPTIMIZATION.md` §7](docs/INFERENCE_OPTIMIZATION.md),
reproducible with [`scripts/inference_benchmark.py`](scripts/inference_benchmark.py); Task 3's
numbers trace to the committed eval output at
[`results/task3_deconfound/`](results/task3_deconfound/) (full write-up in
[`TASK3_DECONFOUND.md`](docs/TASK3_DECONFOUND.md)) and are reproducible with
[`deconfound/RUNBOOK.md`](deconfound/RUNBOOK.md).

---

## Task 3 — does box provenance actually matter (latest work)

The v3→v4 flywheel turn added 1,442 targeted rain/night frames and, at the same time, switched
those frames' boxes from GT-projected to foundation-model-emitted — two changes at once, so the
resulting regression on rain/night/tiny buckets couldn't be attributed to either one cleanly (see
[`FLYWHEEL_V4_FINDINGS.md` § Label-provenance
confound](docs/FLYWHEEL_V4_FINDINGS.md#label-provenance-confound-in-the-v4-experiment)). Task 3 isolates
box provenance directly: two LoRA arms sharing an identical base/val/test set and an identical set
of targeted frame ids, differing only in whether the targeted boxes are FM-emitted or
GT-projected. Full write-up, per-condition breakdown, and limitations are in
[`TASK3_DECONFOUND.md`](docs/TASK3_DECONFOUND.md); the headline is below.

The original v3/v4 per-frame training data did not survive, so this is a faithful reconstruction
of the experiment design at reduced scale (base 2,652 frames, targeted 1,162, test 402 of the
fixed 1,041), not a byte-for-byte replay of the published v3/v4 rows — read the FM-vs-GT delta as
the finding, not the absolute recall numbers.

| metric | FM | GT |
|---|---|---|
| Recall (detection rate) | 0.101 | 0.167 |
| Precision | 0.176 | 0.330 |
| F1 | 0.128 | 0.222 |
| False-positive rate (lower is better) | 0.488 | 0.169 |
| Mean best-pair IoU | 0.244 | 0.458 |
| Frame detect @ IoU 0.5 | 0.266 | 0.566 |
| No-hazard accuracy | 0.512 | 0.831 |
| Mean IoU (matched) | 0.632 | 0.641 |
| Label accuracy (matched) | 0.962 | 0.954 |

GT-projected boxes win on every axis except matched-class label accuracy, where the two arms are
within a point of each other — the gap is in whether and where a box gets drawn, not in what it's
called once drawn. The effect is largest exactly where it matters most: the FM arm detects
nothing at all at night (0.000 vs GT's 0.151) or in rain (0.000 vs GT's 0.050, and every FM box
it does draw in rain is wrong). Box-supervision provenance was a real, previously-confounded
driver of the v4 regression, not a minor detail. Limitations — reduced scale, a small rain bucket
(104 frames), single seed — are stated in full in [`TASK3_DECONFOUND.md`](docs/TASK3_DECONFOUND.md).

The raw eval output backing this table is committed at
[`results/task3_deconfound/`](results/task3_deconfound/) (`results_fm/`, `results_gt/`,
`deconfound_result.json`) — reproduce with [`deconfound/RUNBOOK.md`](deconfound/RUNBOOK.md), which
walks phases 0–7 (nuScenes reconstruction, GT-describe, FM-label, arm assembly, training,
evaluation) with a cost gate and a $0 mock dry-run before any real spend.

---

## Results (v2 → v3 → v4, fixed test set)

All numbers in this section are measured on the fixed 1,041-frame test set (nuScenes
v1.0-trainval, predominantly daytime).

### Detection (L1 grounding, IoU ≥ 0.5)

| version | train ex. | Precision | Recall | F1 | mean IoU | class acc | parse |
|---|---|---|---|---|---|---|---|
| v3 (naive scale-up) | 7,228 | 0.40 | 0.24 | 0.30 | 0.67 | 0.94 | 98.7% |
| v4 (targeted + adverse) | 8,670 | 0.37 | 0.19 | 0.25 | 0.656 | 0.95 | 97.4% |

Precision and localization are strong (mean IoU 0.67, above the 0.55 target) and the hazard class
is named correctly 94–95% of the time; the weak axis is recall on the rare long tail, especially
tiny and distant boxes. An earlier version of this card reported ~1.4% F1 — that was a
coordinate-convention bug (inference ran the image processor at a different resolution than
training, so predicted boxes fell outside the labels' 0–1000 space and every IoU collapsed to
~0). It's fixed; these are the corrected numbers.

### Reasoning (L2, LLM-as-judge, Claude Sonnet 5, n = 1,027, v3)

| Dimension | Mean |
|---|---|
| Correctness | 3.03 |
| Completeness | 2.66 |
| Action relevance | 3.80 |
| Overall | 3.16 / 5 |
| Pass rate (all dims ≥ 3.5) | 26% |

Recommended driving actions are the strongest dimension; completeness is weakest, consistent with
the low detection recall — the model under-reports hazards more than it misjudges them.

### Robustness (L4, detection@0.5 by bucket)

| bucket | v3 | v4 |
|---|---|---|
| tiny box (78% of hazards) | 22.8% | 17.2% |
| small / medium box | 46.4% / 52.6% | 42.9% / 52.6% |
| rain | 12.5% | 7.4% |
| night + tiny | 12.7% | 10.7% |
| clear + medium | 69% | 69% |

v3 scaled data naively (2.7k→7.2k train examples) and generalization suffered (eval_loss
0.31→0.66). v4 then added 1,442 targeted rain/night frames through the flywheel, and those exact
buckets regressed further. The regression gate blocked the v4 candidate and v3 stayed the
production model. Task 3 above is the follow-up that separates how much of that regression was
the targeting versus the box-provenance shift riding along with it. See
[`DEBUGGING_POSTMORTEM.md`](docs/DEBUGGING_POSTMORTEM.md), [`FLYWHEEL.md`](docs/FLYWHEEL.md),
[`FLYWHEEL_V4_FINDINGS.md`](docs/FLYWHEEL_V4_FINDINGS.md), and the generated
[`results/mlops_report.md`](results/mlops_report.md).

### Inference (single T4, 16 GB, 320 GB/s HBM)

Decode is memory-bandwidth-bound — the fp16 baseline converts 31.8% of HBM bandwidth into useful
decode work — so each optimization targets that bottleneck specifically:

| config | decode tok/s | TTFT | e2e p50 | weights | HBM roofline | vs fp16 output |
|---|---|---|---|---|---|---|
| fp16 (baseline) | 17.0 | 770 ms | 11.64 s | ~6.0 GB | 31.8% | reference |
| fp16 + prompt-lookup | 20.4 (+20%) | 781 ms | 9.79 s (−16%) | ~6.0 GB | 38.2% | exact_match 1.00 |
| NF4 (4-bit) | 12.6 (slower) | 820 ms | ~15.9 s | 2.63 GB | 10.4% | char_sim 0.36 |
| INT8 | 4.6 | ~1080 ms | 53.53 s | ~3.5 GB | 5.0% | char_sim 0.29 |

Throughput (fp16 aggregate decode): 14.9 → 33.7 tok/s at batch 4 (~2.3×). Prompt-lookup
speculative decoding is a free latency win — the structured-JSON output repeats prompt tokens
verbatim, so an n-gram drafter lands often, and verification keeps the output bit-identical to
fp16. NF4 is a memory lever, not a speedup: it shrinks weights ~2.3× to fit a 16 GB card, but
decode gets slower and the output drifts (char_sim 0.36), so it ships only behind an L1/L4
re-eval, never on a VRAM number alone. INT8 is worse on every axis and isn't recommended here.
End-to-end latency is seconds per image (11.6 s fp16 / 9.8 s with prompt-lookup) — this is an
autoregressive VLM benchmarked as a compression/throughput story, not a real-time claim.

Full diagnosis (roofline, quality gate, percentiles) is in
[`INFERENCE_OPTIMIZATION.md`](docs/INFERENCE_OPTIMIZATION.md).

### Limitations

Recall on the rare long tail is the weak axis — 24% (v3) at IoU 0.5, with some classes
(`unusual_object`, 24 instances) never detected — while precision (40%) and box tightness (mean
IoU 0.67 on matches) are strong; more training data, naive or targeted, did not move this (see
Task 3 and the v4 finding above for why). The training distribution is narrow: nuScenes
v1.0-trainval is predominantly daytime, and targeted adverse-weather mining did not close the
gap. v3 shows mild overfitting (eval loss 0.66 vs train loss 0.40); v4 trained clean (3 epochs,
eval_loss 0.694) but still regressed the mined buckets. Parse failures are rare (98.7% v3 / 97.4%
v4). This is research / offline-evaluation work — not for real-time or safety-critical control.

This is a portfolio project optimized for lifecycle rigor over a headline accuracy number: it
surfaces and diagnoses real failures (the coordinate-convention bug; naive and targeted data
scaling both failing to lift recall; the box-provenance confound and its follow-up) rather than
reporting a single cherry-picked result.

---

## Pipeline stages

Data curation projects nuScenes 3D ground-truth boxes into the 2D camera frame (near-plane
frustum clipped) and scores each frame across 6 composite rarity signals for mining; a validation
gate blocks the label set from training on any sign of collapse (repeated boxes, oversized boxes,
schema violations) before it reaches the trainer. Training is LoRA SFT on Qwen2.5-VL-3B-Instruct
(rank 32, alpha 64, targets `q/k/v/o/up/down_proj`) with prefix-masked labels, bf16, on a single
A100. Evaluation runs a 4-level framework — grounding (IoU + Hungarian matching), reasoning
(LLM-as-judge), production readiness, and stratified robustness — and includes an all-zero-IoU
abort that refuses to report boxless or garbage predictions rather than silently passing them
through. A regression gate (`scripts/run_regression_gate.py`) compares a new candidate against
the current production model on the weak buckets specifically (rain, night+tiny, tiny) and blocks
promotion on any regression, wired into CI. `src/drivesense/monitoring/drift.py` is a
population-stability check for production traffic — a scaffold ready to plug into a serving
pipeline, not a deployed monitor (see `docs/OBSERVABILITY.md`).

---

## Quick start (local dev, CPU)

```bash
git clone https://github.com/jayanth922/DriveSense-VLM.git
cd DriveSense-VLM
pip install pyyaml pillow numpy scipy tqdm      # core, CPU-safe
python -m pytest tests/ -v                       # test suite, no GPU or downloads
```

A full training + eval run needs a GPU (Colab or RunPod) — see the runbooks under `notebooks/`
or `deconfound/RUNBOOK.md` for the Task 3 ablation specifically.

---

## Repo map

| Path | What it is |
|---|---|
| `README.md` | This file — status, results, what's left |
| `docs/TASK3_DECONFOUND.md` | Task 3 box-provenance ablation: full write-up, per-condition table, limitations |
| `deconfound/` | Task 3 pipeline — reconstruction, arm assembly, training config, comparison, `RUNBOOK.md` |
| `docs/FLYWHEEL.md` | The mine → label → gate → train → eval → gate loop, stage by stage |
| `docs/FLYWHEEL_V4_FINDINGS.md` | The v3→v4 turn in full: what was mined, what regressed, why |
| `docs/DEBUGGING_POSTMORTEM.md` | Failures diagnosed: the coordinate bug, naive scaling, targeted scaling |
| `docs/INFERENCE_OPTIMIZATION.md` | Bottleneck-driven inference study; §7 is the measured T4 results |
| `docs/MODEL_CARD.md` / `hf_model_card/` | Model cards (repo-facing and HuggingFace-facing) |
| `results/metrics_registry.json` | Source of truth for v2/v3/v4 metrics and the gate policy |
| `results/mlops_report.md` | Generated v2→v3→v4 comparison and gate verdict |
| `mlops_report.py` | Builds the report; `--gate` exits non-zero on regression (used in CI) |
| `scripts/inference_benchmark.py` | Reproduces §7 (batching, percentiles, equivalence gate) |
| `scripts/v4/` | The v4 flywheel turn's pipeline (mine → label → build → finalize), reused by Task 3 |
| `scripts/` | Pipeline CLIs: filter, annotate, train, evaluate, mine, gate, ship |
| `src/drivesense/` | Library: `data/`, `training/`, `eval/`, `inference/`, `monitoring/` |
| `src/drivesense/data/spark_pipeline.py` | Distributed PySpark rarity-scoring + analytics ETL |
| `docs/` | Deep dives: observability, closed-loop mining, TensorRT runbook |
| `configs/*.yaml` | All hyperparameters and paths — never hardcoded in source |
| `tests/` | CPU-only, mock-backed test suite — no GPU, downloads, or API keys |
| `huggingface_space/` | Gradio app deployed to HuggingFace Spaces (T4, NF4) |
| `notebooks/` | Colab execution notebooks (data → training → optimization → eval) |
| `.github/workflows/ci.yml` | Tests, mock pipeline smoke, and the regression gate |

---

## What's left

Three items from earlier open threads are done: a T4 re-run settled the three §7 measurement
caveats (see `INFERENCE_OPTIMIZATION.md` §7), the TensorRT ViT runbook executed on a Kaggle T4
with a negative result (`torch.export` fails on Qwen2.5-VL's data-dependent window attention, so
TensorRT isn't viable for this ViT; the deployed latency lever stays fp16 + prompt-lookup, see
`docs/TENSORRT_RUNBOOK.md` §6), and Task 3's box-provenance A/B ran end to end on an H100 — see
the result table above and [`TASK3_DECONFOUND.md`](docs/TASK3_DECONFOUND.md).

Future work, none blocking:

- Scale Task 3 back up — base 2,652 → the original 7,228 target, and test 402 → the full fixed
  1,041 frames — to check the FM-vs-GT delta holds at full scale, not just the reduced
  reconstruction it was measured on.
- Multi-seed Task 3 re-runs for confidence intervals; the current result is a single run per arm.
- An optional v4b ablation: retrain dropping the 216 `no_hazard` negatives introduced in v4, to
  test whether they were separately suppressing recall alongside the box-provenance effect Task 3
  measured.

---

## Tech stack

| Component | Technology | Notes |
|---|---|---|
| Base model | Qwen2.5-VL-3B-Instruct | Apache 2.0 |
| Fine-tuning | LoRA via PEFT | rank 32, alpha 64 |
| Training | HuggingFace Transformers | LoRA SFT, prefix masking, bf16 |
| Demo quantization | bitsandbytes NF4 (4-bit) | HF Spaces T4 demo |
| Data | nuScenes v1.0-trainval | rare-hazard filtered, GT-projected boxes (v2/v3/base); see Task 3 for the FM-emitted exception |
| Distributed ETL | PySpark | 6-signal rarity scoring + analytics, explicit schemas |
| Annotation | Anthropic Claude | describe-only (severity/reasoning/action); FM-emitted `bbox_2d` only for v4's targeted addition and Task 3's FM arm |
| Tracking | Weights & Biases | training metrics |
| Lint / format | Ruff + Black | line-length 100 |
| Testing | pytest | CPU-only, mock-backed |

---

## Testing

```bash
python -m pytest tests/ -v
```

The suite (587 tests, 583 pass / 4 skip without a GPU) is CPU-only and mock-backed — no GPU,
model downloads, or API keys required.

---

## Acknowledgments

- Qwen Team (Alibaba) for Qwen2.5-VL-3B-Instruct (Apache 2.0)
- nuScenes / Motional for the nuScenes autonomous-driving dataset
- HuggingFace for Transformers, PEFT, and Spaces
- Anthropic for the Claude API used in describe-only annotation and LLM-as-judge evaluation