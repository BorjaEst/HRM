from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from lightning.pytorch import Trainer, seed_everything
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from hrm_sn.callbacks.checkpoint import CheckpointCallback, CheckpointSettings
from hrm_sn.callbacks.figures import FigureCallbackSettings, FiguresCallback
from hrm_sn.data.puzzle_datamodule import PuzzleDatamodule, PuzzleDatamoduleConfig
from hrm_sn.data.puzzle_dataset import PuzzleDatasetSettings
from hrm_sn.logging.tensorboard import Logger, LoggerSettings
from hrm_sn.loss.act_head import ACTLossConfig
from hrm_sn.models.hrm_v1 import Model, ModelConfig_HRM_V1
from hrm_sn.modules.hrm import HRMConfig
from hrm_sn.training.act_controller import ACTControllerConfig
from hrm_sn.training.optim import AdamATan2Config
from hrm_sn.training.schedules import SchedulerConfig

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("medium")
CONFIGURATION_PATH = os.environ.get("EXP01_CONFIGURATION_PATH", "config/defaults.toml")


# =================================================================================================
# Settings Model
# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True, cli_prog_name="run"):
    """ """

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
        """ """
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    # ---------------------------------------------------------------------------------------------
    # Names and tracking
    project_name: Optional[str] = Field(
        default=None,
        description=(
            "Project name. If not set, it defaults to the capitalized name of the dataset "
            "(e.g. `MATH` -> `Math ACT-torch`)."
        ),
    )
    run_name: Optional[str] = Field(
        default=None,
        description=(
            "Run name. If not set, it defaults to `<arch_name> <random_slug>` "
            "(e.g. `HrmV1 2x128 4L 16H 0.1D ACT-torch cool-slug`)."
        ),
    )

    # ---------------------------------------------------------------------------------------------
    # Model architecture and data
    architecture: HRMConfig = Field(
        ...,
        description=(
            "Architecture config for the HRM model. The keys in `architecture` are passed to the"
            "HRModel constructor."
        ),
    )
    act_controller: ACTControllerConfig = Field(
        ...,
        description=(
            "Configuration for the ACT controller, which manages halting and partial resets"
            "during training. "
            "The keys in `act_controller` are passed to the ACTController constructor."
        ),
    )
    loss: ACTLossConfig = Field(
        ...,
        description="Loss config. The keys in `loss` are passed to the loss head constructor.",
    )
    optimizer: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description=(
            "Main optimizer config for model parameters (e.g. Adam). "
            "The keys in `optim_main` are passed to the optimizer constructor."
        ),
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description=(
            "Learning rate scheduler config. If not set, no learning rate scheduling is applied. "
            "The keys in `scheduler` are passed to the scheduler constructor."
        ),
    )

    # ---------------------------------------------------------------------------------------------
    # Data settings (passed as configs to DataModule)
    dataset: PuzzleDatasetSettings = Field(
        ...,
        description=(
            "Configuration for the PuzzleDataset. "
            "This includes parameters like dataset path, random seed, etc."
        ),
    )
    global_batch_size: int = Field(
        ...,
        description=(
            "Global batch size across all devices. "
            "The per-device batch size is computed as `global_batch_size // world_size`."
        ),
    )
    num_workers: int = Field(
        1,
        description="Number of workers for DataLoader, currently expects 1 worker.",
    )
    prefetch_factor: int = Field(
        8,
        description="Number of batches to prefetch per worker.",
    )
    pin_memory: bool = Field(
        True,
        description="Whether to pin memory in DataLoader.",
    )
    persistent_workers: bool = Field(
        True,
        description="Whether to keep DataLoader workers alive between epochs.",
    )

    # ---------------------------------------------------------------------------------------------
    # Training control settings (passed as top-level settings for ease of CLI overrides)
    epochs: int = Field(..., description="Total number of epochs to train.")

    # ---------------------------------------------------------------------------------------------
    # Core settings for model, data, and training configuration (passed as configs to modules)
    logger: Optional[LoggerSettings] = Field(
        default_factory=LoggerSettings,
        description="TensorBoard logger settings.",
    )
    checkpoint: Optional[CheckpointSettings] = Field(
        default_factory=CheckpointSettings,
        description="Model checkpoint settings.",
    )
    figures: Optional[FigureCallbackSettings] = Field(
        default_factory=FigureCallbackSettings,
        description="Figure generation callback settings.",
    )

    # ---------------------------------------------------------------------------------------------
    # Training control settings (passed as kwargs to Lightning Trainer)
    max_steps: int = Field(
        default=80000,
        description="Maximum training steps.",
    )
    log_every_n_steps: int = Field(
        default=20,
        description="Log metrics every N steps.",
    )
    val_check_interval: int = Field(
        default=1000,
        description="Validation check interval (in training steps).",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show progress bar during training.",
    )

    # ---------------------------------------------------------------------------------------------
    # Checkpointing and evaluation settings (passed as kwargs to Trainer and Checkpoint callback)
    checkpoint_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to save checkpoints and logs. "
            "If not set, it defaults to `checkpoints/<project_name>/<run_name>`."
        ),
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Whether to checkpoint the model after every evaluation.",
    )
    limit_val_batches: int = Field(
        default=10,
        description="Cap validation to N batches per validation run.",
    )
    eval_save_outputs: List[str] = Field(
        default_factory=list,
        description="Evaluation output keys saved as tensors in the checkpoint directory.",
    )

    # ---------------------------------------------------------------------------------------------
    # Aggregate settings (compose leaf settings for modules)
    @property
    def model(self) -> ModelConfig_HRM_V1:
        """Compose ModelConfig_HRM_V1 from leaf settings."""
        return ModelConfig_HRM_V1.model_validate(self, from_attributes=True)

    @property
    def datamodule(self) -> PuzzleDatamoduleConfig:
        """Compose PuzzleDatamoduleConfig from leaf settings."""
        return PuzzleDatamoduleConfig.model_validate(self, from_attributes=True)


# =================================================================================================
# Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    """Here goes a description
    # TODO: write a detailed docstring describing the main entrypoint, the training loop,
    #       and how the different components interact.
    """

    # Step _: Parse settings (CLI overrides TOML overrides defaults)
    defaults_from_path = tomllib.load(Path(CONFIGURATION_PATH).open("rb"))
    settings = RunArguments(**defaults_from_path)

    # Step _: Seed everything for reproducibility
    seed_everything(settings.dataset.seed)

    # Step _:
    # Preparation of callbacks list: checkpointing + optional figure generation
    callbacks_list = []
    if settings.checkpoint is not None:
        callbacks_list.append(CheckpointCallback(settings.checkpoint))
    if settings.figures is not None and settings.figures.enabled:
        callbacks_list.append(FiguresCallback(settings.figures))

    # Step _: Build the PyTorch Lightning Trainer
    # This wires together logging, checkpointing, and training control
    trainer = Trainer(
        # Callbacks and TensorBoard logger for metrics and hyperparameters
        logger=Logger(settings.logger) if settings.logger is not None else None,
        callbacks=callbacks_list if callbacks_list else None,
        # Lightning Trainer kwargs (extracted from config)
        max_steps=settings.max_steps,
        val_check_interval=settings.val_check_interval,
        limit_val_batches=settings.limit_val_batches,
        # Validation check every N steps (can also be set to a fraction for epoch-based checking)
        log_every_n_steps=settings.log_every_n_steps,
        enable_progress_bar=settings.enable_progress_bar,
    )

    # Step _: Start training
    # The LightningModule wraps the TEM model and defines the training loop
    # The DataModule generates batches of walk data on-the-fly
    trainer.fit(
        # Lightning module: training step, optimizer, schedule computation
        model=Model(settings.model),
        # Data module: generates environment walks and batches
        datamodule=PuzzleDatamodule(settings.datamodule),
        # Optional: resume from checkpoint
        ckpt_path=settings.checkpoint_path,
    )
