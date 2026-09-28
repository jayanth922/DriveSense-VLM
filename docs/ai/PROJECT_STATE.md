# Project state

## Project objective

DriveSense-VLM is a Qwen2.5-VL-3B rare-hazard detection system for autonomous-driving
images, with a closed data flywheel, GT/F foundation-model box-provenance experiments,
evaluation gates, monitoring, and deployment/optimization studies.

## Current milestone

The repository structure and user-facing documentation have been revised. The edits are complete
and verified locally but remain uncommitted pending user review.

## Current architecture and invariants

- Importable code lives under `src/drivesense/`; operational CLIs live under `scripts/`.
- Reproducible studies live under `experiments/`; deployment packaging lives under
  `deploy/`.
- Tests are CPU/mock safe by default; heavyweight optional dependencies skip cleanly.
- Detection metrics are sourced from `results/metrics_registry.json`; measured inference
  results are documented in `docs/INFERENCE_OPTIMIZATION.md`.
- Generated data, models, logs, and local run outputs remain ignored.

## Completed or verified work

- Moved the Task 3 de-confound study to `experiments/task3_deconfound/`.
- Moved Hugging Face Space/model-card assets to `deploy/huggingface/`.
- Moved the MLOps report CLI to `scripts/mlops_report.py` and made defaults repo-root
  relative.
- Moved the Java compatibility fixture to `tests/fixtures/java_patch/`.
- Updated documentation, notebooks, CI, configs, tests, and deployment scripts to the new
  paths.
- Removed only cache/build junk; retained local ignored outputs and environments.
- Rewrote the README, model cards, design notes, experiment reports, and deployment notes in
  direct language while preserving measured results and limitations.

## Active problem

No implementation blocker. The uncommitted refactor needs user review before any commit.

## Relevant files

- `README.md`
- `scripts/mlops_report.py`
- `scripts/deploy_to_space.py`
- `scripts/upload_to_hf.py`
- `experiments/task3_deconfound/`
- `deploy/huggingface/`
- `tests/test_spark_pipeline.py`

## Verification commands and latest results

- `.venv/bin/python scripts/run_sanity_check.py` — 60/60 checks passed.
- `.venv/bin/python -m pytest -q tests` — 561 passed, 4 skipped.
- `python3 scripts/mlops_report.py` — generated the report and returned the expected v4
  `BLOCK` verdict.
- Moved experiment and Hugging Face CLIs passed `--help` smoke checks.
- Local Markdown link audit — zero missing relative links.

## Known blockers or risks

- The refactor is on branch `fix/docs-discrepancies`, not `main`.
- No commit or push has been requested for this structural change.

## Next bounded task

Review the final diff/status, then commit only if explicitly requested.
