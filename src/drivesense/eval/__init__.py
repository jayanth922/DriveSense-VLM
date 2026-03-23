"""4-level evaluation framework for DriveSense-VLM.

Submodules:
    grounding:   Level 1 — Bounding box IoU and hazard detection rate (Phase 4b).
    reasoning:   Level 2 — LLM-as-judge reasoning and classification quality (Phase 4b).
    production:  Level 3 — Latency, throughput, and VRAM benchmarks (Phase 4b).
    robustness:  Level 4 — Stratified performance across driving conditions (Phase 4b).
"""

from __future__ import annotations
