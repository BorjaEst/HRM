# Requirements Specification (Canonical)

This document defines repo-level requirements and constraints. It is binding under strict mode.

## Agent/Automation Gating (Strict)

- Do not invent schemas, dataset formats, or CLI contracts not present in these specs or in code.
- If a change would introduce a new top-level component or boundary, update the specs first.
- Keep the target taxonomy consistent with [spec/spec-architecture.md](spec-architecture.md).

## Packaging & Layout

- The repository shall be structured as a Python package using a `src/` layout.
- Canonical import root: `hrm_sn` (e.g., `from hrm_sn.models import ...`).
- Experiments shall live under `experiments/` and import from `hrm_sn`.

## Experiment Framework

- Experiments shall use **PyTorch Lightning** for training and evaluation loops.
- Logging shall use **TensorBoard**.
- Weights & Biases (`wandb`) is **not required** and should not be treated as a mandatory dependency for running experiments.

## Data Handling

- Datasets shall be stored under:
  - `data/raw/`
  - `data/interim/`
  - `data/processed/`
- The only dataset pipeline in-scope for new work is **Maze**.
- Dataset consumption at training time should use Lightning DataModules (or equivalent), not bespoke global state.

## Attention Implementation

- Use **vanilla PyTorch attention** only.
- Do not require FlashAttention, xFormers, or other external attention kernels.

## External Integrations

- Hugging Face is allowed for dataset download and model/checkpoint distribution (e.g., `huggingface_hub`).
- If Hugging Face is used, configuration must allow offline/local paths for reproducibility.

## Reproducibility

- Experiments must be seedable from configuration.
- Output artifacts must be written to an explicit run directory (Hydra run dir or a configured output path).
