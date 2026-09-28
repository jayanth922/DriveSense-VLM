# Qwen2.5-VL vision-encoder export investigation

[Back to the project README](../README.md)

This document records an attempted ONNX and TensorRT export of the Qwen2.5-VL vision encoder. The
test ran on a Kaggle T4 on 2026-08-23. Export failed because the encoder contains data-dependent
window-attention logic that cannot be captured as a static graph.

No TensorRT speedup is claimed.

## Result

Input was 448 x 672, producing 1,536 pre-merge vision patches. Measurements are ViT-only, not
end-to-end generation.

| Backend | ViT p50 | Relative to eager | Result |
|---|---:|---:|---|
| PyTorch eager | 203.1 ms | 1.00x | baseline |
| torch.compile, default | 197.9 ms | 1.03x | graph break; negligible gain |
| torch.compile, reduce-overhead | not available | not available | CUDA graph incompatible with the graph break |
| TensorRT through ONNX | not available | not available | export failed |

The practical implementation remains eager ViT execution. End-to-end latency work uses fp16 prompt
lookup instead; see [INFERENCE_OPTIMIZATION.md](INFERENCE_OPTIMIZATION.md).

## Why export failed

Qwen2.5-VL builds attention windows from runtime sequence data. In
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py, get_window_index converts cumulative
sequence lengths with tolist()/item(). That data-dependent control flow breaks static graph
capture.

Observed failures:

- torch.export stopped with GuardOnDataDependentSymNode while evaluating a runtime expression in
  the window-index path;
- torch.jit.trace followed by ONNX export reported that exporting the resulting ScriptModule was
  unsupported;
- torch.compile inserted a graph break at the same operation, leaving little work to fuse.

A working TensorRT path would require rewriting the window-index construction for fixed shapes or
using a backend with native support for this Qwen2.5-VL operation.

## Harness correction made before the test

The first harness version passed an image tensor shaped [1, 3, 448, 672] directly to the vision
encoder. Qwen2.5-VL's visual module expects pre-patchified input shaped
[sequence_length, channels x temporal x patch_size squared], plus grid_thw.

For this run the correct inputs were:

- patch size 14;
- flattened patch width 1,176;
- patch tensor [1,536, 1,176];
- grid_thw [[1, 32, 48]].

The incorrect patch-size-28 assumptions were fixed in
src/drivesense/inference/tensorrt_vit.py and tests/test_tensorrt.py. The corrected eager forward
returned [384, 2,048].

## Scope

The repository intentionally targets only the vision encoder:

- src/drivesense/inference/tensorrt_vit.py extracts model.visual, vision_tower, or vision_model;
- the autoregressive language decoder is not exported;
- run_benchmark.py --vit-only measures the encoder in isolation.

The decoder has changing sequence lengths and a growing KV cache. Exporting the full multimodal
generate loop would require TensorRT-LLM or a separate serving integration, not a standard ONNX
export.

## Reproduction

Run this on a CUDA environment with the project training dependencies:

    python -m pip install tensorrt onnx onnxsim
    python -c "import torch, tensorrt; print(torch.__version__, tensorrt.__version__)"
    python scripts/run_optimize_model.py \
      --tensorrt \
      --model-dir outputs/merged_model

The pipeline attempts:

1. direct ONNX export with opset 17;
2. a trace-based ONNX fallback;
3. TensorRT engine compilation when ONNX succeeds;
4. torch.compile fallback;
5. eager, compiled, and TensorRT benchmarking for available backends.

Outputs are written under outputs/tensorrt/, including fallback_info.json and
optimization_report.txt.

## Reading fallback_info.json

| Field | Meaning |
|---|---|
| onnx_method=direct | direct ONNX export succeeded |
| onnx_method=jit_trace | direct export failed but trace export succeeded |
| onnx_method=failed | neither ONNX path worked |
| trt_method=tensorrt | a TensorRT engine was built and reloaded |
| trt_method=torch_compile with trt_error | TensorRT parsing or compilation failed |
| trt_method=torch_compile with trt_note | TensorRT was unavailable |

Do not treat a torch.compile fallback as a TensorRT result.

## Historical artifact correction

Before the GPU run, outputs/tensorrt/fallback_info.json appeared to show an earlier failure. It was
actually written by tests/test_tensorrt.py because a fixture pointed at the real output directory
instead of pytest's temporary directory. The test now uses tmp_path, and the leaked file was
removed. Mock speedup values in run_optimize_model.py are CI placeholders, not measurements.

## If the export is revisited

The next useful experiment is not another unchanged ONNX attempt. It should first replace or
specialize get_window_index so its output is static for one fixed grid. Only then should the ONNX
and TensorRT pipeline be rerun. Report ViT-only numbers separately from full-model latency unless a
TensorRT encoder is actually integrated into model.generate().
