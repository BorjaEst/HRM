# Standards Specification (Canonical)

This document defines standards for code, experiments, and documentation.

## Code Standards (Python)

- `src/hrm_sn/` must be importable without triggering training runs, dataset downloads, or GPU initialization.
- Prefer typed, explicit APIs over dynamic import strings. Where dynamic loading is used, it must be documented and validated.
- Public modules should have clear, stable interfaces.

## Experiment Standards

- Experiments live in `experiments/` and should be runnable via a single entry script (e.g., `python experiments/<exp>/train.py`).
- Experiments use Hydra for configuration.
- Experiments log to TensorBoard and write checkpoints/metrics to an explicit run directory.
- Experiments must not duplicate reusable model/data logic that belongs in `src/hrm_sn/`.

## Data/Builder Standards

- Builders must read from `data/raw` (and optionally `data/interim`) and write to `data/processed`.
- Builders should be deterministic given the same inputs and seed.
- Processed datasets should include enough metadata to load and validate shapes/dtypes.

## Visualization Standards

- Visualization utilities belong in `src/hrm_sn/figures/`.
- Notebooks may live at repo root or under `docs/` or `notebooks/`, but they should import visualization helpers from the package rather than reimplementing them.

## Documentation Standards

- Canonical documentation should live under `docs/`.
- When code changes alter experiment usage, dataset layout, or expected outputs, update docs accordingly.

## Spec Maintenance

- Specs in `spec/` are canonical.
- Any change that affects taxonomy, required dependencies, or directory conventions requires updating the specs first.
