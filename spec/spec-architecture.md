# Architecture Specification (Canonical)

This repository is evolving into a **research codebase template** for building and running experiments with **HRM for Spatial Navigation**.

Canonical package name: **`hrm_sn`** (Hierarchical Reasoning Model for Spatial Navigation).

This document defines the vocabulary, boundaries, and intended layout. Where the current repository differs, treat the current layout as **legacy/transitional**; new work should follow the target structure described here.

## Goals

- Provide a reusable Python package (`src/hrm_sn/`) that contains model and data utilities.
- Run experiments from `experiments/` using **PyTorch Lightning**.
- Standardize dataset storage under `data/raw`, `data/interim`, `data/processed`.
- Keep visualization utilities in a package submodule so they can be imported from notebooks and experiments.

## Repository Taxonomy (Target State)

### Package (library code)

- **Location**: `src/hrm_sn/`
- **Responsibility**: reusable, importable code only.
- **Examples of what belongs here**:
  - Model modules (Lightning `LightningModule` or plain `nn.Module` components)
  - Layers, losses, embeddings, attention blocks (vanilla PyTorch attention only)
  - Data utilities and Lightning `DataModule` implementations
  - Visualization helpers (see `Figures` below)

**Boundary rule**: nothing in `src/hrm_sn/` should assume a specific experiment run directory, CLI invocation, or hard-coded dataset paths beyond what is passed in via config.

### Experiments (entry points)

- **Location**: `experiments/`
- **Responsibility**: runnable training/evaluation scripts and experiment-specific configuration.
- **Expected contents**:
  - One or more experiment folders/modules with:
    - a runnable script (e.g., `train.py`, `eval.py`)
    - Hydra config (either centralized under `config/` or per-experiment)
    - logging setup (TensorBoard)

**Boundary rule**: experiment code may assemble components and wire configs, but core logic must live in `src/hrm_sn/`.

### Config (Hydra or TOML + Pydantic Settings)

- **Location**: `config/` (or `experiments/<name>/config/`)
- **Responsibility**: declarative configuration of experiments.
- **Conventions**:
  - Hydra is supported, but TOML + Pydantic Settings is also a first-class option.
  - For TOML + Pydantic Settings:
    - Configuration keys should be grouped by component (e.g., `shared`, `data`, `model`, `trainer`, `logger`).
    - Deterministic precedence is required: **CLI overrides TOML overrides defaults**.
    - Root experiment settings compose leaf settings near the classes that consume them.
  - Model selection should be via explicit config values rather than dynamic string imports when possible.

### Data

- **Locations**:
  - `data/raw/`: immutable inputs (downloaded datasets, original exports)
  - `data/interim/`: intermediate processing artifacts
  - `data/processed/`: training-ready datasets
- **In-scope dataset**: **Maze**
- **Out-of-scope (for now)**: ARC and Sudoku pipelines

### Models / Artifacts

To avoid confusion with “model code”, avoid storing Python modules in a top-level `models/` directory.

- **Checkpoints / weights**: store under an artifacts directory (recommended `artifacts/`) or experiment outputs.
- **Rule**: if a top-level `models/` directory exists, it should only contain **binary artifacts** (e.g., `.ckpt`, `.pt`) and must not contain importable Python source.

### Figures (Visualization)

- **Location**: `src/hrm_sn/figures/`
- **Responsibility**: plotting and display utilities (matplotlib, image grids, debug visualizations) usable from notebooks and experiments.

## Legacy / Transitional Structure (Observed Today)

The current repository contains scripts and modules such as `pretrain.py`, `evaluate.py`, `dataset/`, `puzzle_dataset.py`, and `models/` that implement training, dataset building, and model code.

These are considered **legacy** relative to the target taxonomy. The specs govern _new_ work and any refactors/migrations.

## Allowed Vocabulary

- **Package**: code under `src/hrm_sn/`
- **Experiment**: runnable code under `experiments/`
- **Builder**: dataset preprocessing job producing `data/processed/...`
- **Artifact**: outputs of experiments (checkpoints, logs, metrics)
- **Figure**: visualization output or helper under `src/hrm_sn/figures/`

## Forbidden Framings

- Do not impose generic web-app layering terms (controllers/services/repositories) unless the repo explicitly adopts them.
- Do not require or assume external attention implementations (FlashAttention/xFormers). Use vanilla PyTorch attention.

## The Canonical Structure for a Python + PyTorch + Lightning Project

```plaintext
my_project/
│
├── pyproject.toml
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── data/
│       ├── models/
│       ├── modules/
│       ├── loss/
│       ├── utils/
│       ├── callbacks/
│       ├── training/
│       ├── figures/
│       └── config/
│
└── experiments/
    ├── exp01_baseline.py
    ├── exp02_big_model.py
    └── expXX_*.py
```

## What Should NOT Go in experiments/

- LightningModules
- DataModules
- Loss functions
- Utility functions
- Model architectures
- Reusable callbacks
  These should live under src/ so they can be imported elsewhere.
