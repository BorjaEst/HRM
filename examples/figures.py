""" """

from __future__ import annotations

import logging
import os
import tomllib
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from lightning.pytorch import Trainer, seed_everything
from matplotlib.figure import Figure
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from hrm_sn import figures
from hrm_sn.callbacks.checkpoint import CheckpointCallback, CheckpointSettings
from hrm_sn.callbacks.figures import FigureCallbackSettings, FiguresCallback
from hrm_sn.data.puzzle_datamodule import PuzzleDatamodule, PuzzleDatamoduleConfig
from hrm_sn.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetRuntime, PuzzleDatasetSettings
from hrm_sn.figures.registry import FigureContext
from hrm_sn.logging.tensorboard import Logger, LoggerSettings
from hrm_sn.loss.act_head import ACTLossConfig
from hrm_sn.models.hrm_v1 import Model, ModelConfig_HRM_V1
from hrm_sn.modules.hrm import HRMConfig
from hrm_sn.rollouts.collect import TraceCollector
from hrm_sn.rollouts.trace_tree import TraceTree
from hrm_sn.training.act_controller import ACTControllerConfig
from hrm_sn.training.optim import AdamATan2Config
from hrm_sn.training.rollout import EvaluationLoop
from hrm_sn.training.schedules import SchedulerConfig

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("medium")
CONFIGURATION_PATH = os.environ.get("EXP01_CONFIGURATION_PATH", "config/defaults.toml")


NAME = __file__.split("/")[-1].replace(".py", "")
logger = logging.getLogger(NAME)


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleArguments(BaseSettings, extra="ignore", cli_parse_args=True, cli_prog_name="run"):
    """ """

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
        """ """
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    architecture: HRMConfig = Field(
        ...,
        description=(
            "Architecture config for the HRM model. The keys in `architecture` are passed to the"
            "HRModel constructor."
        ),
    )

    act_controller: ACTControllerConfig = Field(
        ...,
        description="ACT controller config for the HRM model.",
    )

    loss: ACTLossConfig = Field(
        ...,
        description="Loss config for the HRM model.",
    )

    optimizer: AdamATan2Config = Field(
        ...,
        description="Optimizer config for the HRM model.",
    )

    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Scheduler config for the HRM model.",
    )

    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices.",
    )

    checkpoint: Path | None = Field(
        default=None,
        description="Path to model checkpoint (optional). If None, uses random init.",
    )

    dataset: PuzzleDatasetSettings = Field(
        ...,
        description=(
            "Configuration for the PuzzleDataset. "
            "This includes parameters like dataset path, random seed, etc."
        ),
    )

    output_dir: Path = Field(
        default=Path("outputs/model_rollout"),
        description="Directory for saving plots",
    )

    @property
    def model(self) -> ModelConfig_HRM_V1:
        """Construct a ModelConfig_HRM_V1 from the provided arguments."""
        return ModelConfig_HRM_V1.model_validate(self, from_attributes=True)

    @property
    def runtime(self) -> PuzzleDatasetRuntime:
        return PuzzleDatasetRuntime.model_validate(self, from_attributes=True)

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output_dir if it does not exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


# ==============================================================================
# Main Experiment
# ==============================================================================
def main() -> None:
    """ """

    # Step _: Parse settings (CLI overrides TOML overrides defaults)
    defaults_from_path = tomllib.load(Path(CONFIGURATION_PATH).open("rb"))
    args = ExampleArguments(**defaults_from_path)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=args.log_level.upper())

    print("=" * 80)
    print("HRM Model Rollout Figure Generation")
    print("=" * 80)
    print(f" - Log level: {args.log_level.upper()}")
    print(f" - Checkpoint: {args.checkpoint or 'None (random init)'}")
    print(f" - Output directory: {args.output_dir}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Instantiate Dataset and build runtime objects.
    # ------------------------------------------------------------------
    dataset = PuzzleDataset(args.dataset, args.runtime, mode="eval")

    print("Step 1: Dataset generation complete.")
    print()

    # ------------------------------------------------------------------
    # Step 2: Initialize HRM model.
    # ------------------------------------------------------------------
    model = Model(args.model)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        hrm_sd = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(hrm_sd, strict=False)
        print("Checkpoint loaded.")
    else:
        print("Using random initialization (no checkpoint provided).")

    model.eval()  # Set to evaluation mode
    print("Step 2: HRM model initialized.")
    print()

    # ------------------------------------------------------------------
    # Step 3: Collect rollout trace from the model on the dataset.
    # ------------------------------------------------------------------
    set_name, batch, effective_bs = next(iter(dataset))
    collector = TraceCollector(TraceTree(), model.trace_specs)
    max_steps = model.config.act_controller.halt_max_steps
    with torch.no_grad():
        for t, step in EvaluationLoop(model.loss_head, repeat(batch), max_steps=max_steps):
            collector.append(t, step)
    trace = collector.tree
    trace.finalize()

    print("Step 3: Collecting rollout trace complete.")
    print(f" - Split: {set_name}")
    print(f" - Effective batch size: {effective_bs}")
    print(f" - Local batch size: {batch['inputs'].shape[0]}")
    # TODO: print more things
    print()

    # ------------------------------------------------------------------
    # Step 4: Generate diagnostic visualizations.
    # ------------------------------------------------------------------
    ctx = FigureContext(
        extras={
            "inputs": batch["inputs"].cpu().numpy(),
            "labels": batch["labels"].cpu().numpy(),
        }
    )
    figs: list[tuple[str, Figure]] = [
        (f"00.0_dummy_figure.png", figures.dummy.plot(trace, ctx)),
        (f"01.1_maze_overlay.png", figures.overlay.plot(trace, ctx)),
    ]

    print(f"Step 4: Generated {len(figs)} figure(s).")
    print()

    # ------------------------------------------------------------------
    # Step 5: Save and/or show plots.
    # ------------------------------------------------------------------
    for filename, fig in figs:
        fig.savefig(args.output_dir / filename, dpi=150, bbox_inches="tight")
        print(f"Saved {len(figs)} figure(s) to: {args.output_dir}")
    plt.close("all")

    print("Example completed.")
    print("=" * 80)


# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    main()
