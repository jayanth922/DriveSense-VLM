# Deployment assets

This directory contains files packaged for external model hosting. Training and evaluation code
remains under `src/` and `scripts/`.

- [`huggingface/space/`](huggingface/space/) contains the Gradio Space app and
  its runtime dependencies.
- [`huggingface/model_card/`](huggingface/model_card/) contains the model card
  uploaded to the Hugging Face model repository.

Run `scripts/deploy_to_space.py` and `scripts/upload_to_hf.py` from the repository root.
