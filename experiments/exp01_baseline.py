from __future__ import annotations

import tomllib
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from lightning.pytorch import Trainer, seed_everything
from pydantic import Field
from pydantic_settings import BaseSettings

from hrm_sn.callbacks.checkpoint import CheckpointCallback, CheckpointSettings
from hrm_sn.callbacks.figures import FiguresCallback, FiguresSettings
from hrm_sn.data.puzzle_datamodule import PuzzleDatamodule, PuzzleDatamoduleSettings
from hrm_sn.logging.tensorboard import Logger, LoggerSettings
from hrm_sn.models.hrm_v1 import Model, ModelConfig

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("medium")


# ============================================================================
# Settings Model
# ============================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True, cli_prog_name="run"):

    # =========================================================================
    # Core settings for model, data, and training configuration (passed as configs to modules)
    # =========================================================================
    configuration_path: Optional[Path] = Field(
        default=None,
        description="Path to a TOML configuration file. Overrides CLI arguments and environment variables.",
    )
    logger: Optional[LoggerSettings] = Field(
        default_factory=LoggerSettings,
        description="TensorBoard logger settings.",
    )
    checkpoint: Optional[CheckpointSettings] = Field(
        default_factory=CheckpointSettings,
        description="Model checkpoint settings.",
    )
    figures: Optional[FiguresSettings] = Field(
        default_factory=FiguresSettings,
        description="Figure generation callback settings.",
    )

    # =========================================================================
    # Training control settings (passed as kwargs to Lightning Trainer)
    # =========================================================================
    max_steps: int = Field(
        default=20000,
        description="Maximum training steps.",
    )
    val_check_interval: int = Field(
        default=100,
        description="Validation check interval (in training steps).",
    )
    log_every_n_steps: int = Field(
        default=10,
        description="Log metrics every N steps.",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show progress bar during training.",
    )

    # =========================================================================
    # Extra options for training control (e.g., resuming from checkpoint)
    # =========================================================================
    ckpt_path: Optional[Path] = Field(
        default=None,
        description="Path to checkpoint file to resume from.",
    )

    # =========================================================================
    # Aggregate settings (compose leaf settings for modules)
    # =========================================================================
    @property
    def model(self) -> ModelConfig:
        """Compose TEMConfig from leaf settings.
        Creates the aggregate model configuration consumed by Model.
        """
        if self.configuration_path is None:
            return ModelConfig(model_name="hrm_v1", model_version="default")

        with open(self.configuration_path, "rb") as f:
            config_dict = tomllib.load(f)

        return ModelConfig.model_validate(config_dict, from_attributes=True)

    @property
    def data(self) -> PuzzleDatamoduleSettings:
        """Compose DataConfig from leaf settings.
        Creates the aggregate data configuration consumed by DataModule.
        """
        return PuzzleDatamoduleSettings.model_validate(self, from_attributes=True)


# ============================================================================
# Main Entrypoint
# ============================================================================
if __name__ == "__main__":
    """Here goes a description
    # TODO: write a detailed docstring describing the main entrypoint, the training loop,
    #       and how the different components interact.
    """

    # Step _: Parse all settings from CLI and environment
    # Pydantic Settings will automatically parse sys.argv when cli_parse_args=True
    args = RunArguments()

    # Step _:
    #
    hrm_model = Model(args.model)

    # Step _:
    # Preparation of callbacks list: checkpointing + optional figure generation
    callbacks_list = []
    if args.checkpoint is not None:
        callbacks_list.append(CheckpointCallback(args.checkpoint))
    if args.figures is not None:
        callbacks_list.append(FiguresCallback(args.figures))

    # Step _: Build the PyTorch Lightning Trainer
    # This wires together logging, checkpointing, and training control
    trainer = Trainer(
        # TensorBoard logger for metrics and hyperparameters
        logger=Logger(args.logger) if args.logger is not None else None,
        # Callbacks: checkpointing + optional figure generation
        callbacks=callbacks_list if callbacks_list else None,
        # Lightning Trainer kwargs (extracted from config)
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=args.enable_progress_bar,
    )

    # Step _: Start training
    # The LightningModule wraps the TEM model and defines the training loop
    # The DataModule generates batches of walk data on-the-fly
    trainer.fit(
        # Lightning module: training step, optimizer, schedule computation
        model=Model(args.model),
        # Data module: generates environment walks and batches
        datamodule=PuzzleDatamodule(args.data),
        # Optional: resume from checkpoint
        ckpt_path=str(args.ckpt_path) if args.ckpt_path else None,
    )
