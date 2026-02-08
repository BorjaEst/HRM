from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from lightning.pytorch import Trainer, seed_everything
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings

from hrm_sn.callbacks.checkpoint import CheckpointCallback, CheckpointSettings
from hrm_sn.callbacks.figures import FiguresCallback, FiguresSettings
from hrm_sn.data.puzzle_datamodule import PuzzleDatamodule
from hrm_sn.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetMetadata, PuzzleDatasetSettings
from hrm_sn.logging.tensorboard import Logger, LoggerSettings
from hrm_sn.loss import LossConfig
from hrm_sn.loss.act_head import ACTLossHead
from hrm_sn.models.hrm_v1 import Model, ModelConfig_HRM_V1
from hrm_sn.modules.hrm import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config
from hrm_sn.training.buffers import FifoBuffer
from hrm_sn.training.optim import AdamATan2, AdamATan2Config, CastedSparseEmbeddingSignSGD_Distributed, CastedSparseEmbeddingSignSGDConfig
from hrm_sn.training.partial_reset import PartialResetBatchAssembler
from hrm_sn.training.rollout import EvaluationLoop, RolloutLoop
from hrm_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("medium")


# ============================================================================
# Settings Model
# ============================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True, cli_prog_name="run"):

    # =========================================================================
    # Names and tracking
    # =========================================================================
    project_name: Optional[str] = Field(
        default=None,
        description="Project name. If not set, it defaults to the capitalized name of the dataset (e.g. `MATH` -> `Math ACT-torch`).",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Run name. If not set, it defaults to `<arch_name> <random_slug>` (e.g. `HrmV1 2x128 4L 16H 0.1D ACT-torch cool-slug`).",
    )

    # =========================================================================
    # Model architecture and data
    # =========================================================================
    arch: HierarchicalReasoningModel_ACTV1Config = Field(
        ...,
        description="Architecture config. The keys in `arch.__pydantic_extra__` are passed to the model constructor.",
    )
    dataset: PuzzleDatasetSettings = Field(
        default_factory=PuzzleDatasetSettings,
        description="Configuration for the PuzzleDataset. This includes parameters like dataset path, batch size, random seed, etc.",
    )
    loss: LossConfig = Field(
        ...,
        description="Loss config. The keys in `loss` are passed to the loss head constructor.",
    )
    optim_main: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Main optimizer config for model parameters (e.g. Adam). The keys in `optim_main` are passed to the optimizer constructor.",
    )
    optim_emb: CastedSparseEmbeddingSignSGDConfig = Field(
        default_factory=CastedSparseEmbeddingSignSGDConfig,
        description="Puzzle embedding optimizer config (e.g. SignSGD). The keys in `optim_emb` are passed to the optimizer constructor.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler config. If not set, no learning rate scheduling is applied. The keys in `scheduler` are passed to the scheduler constructor.",
    )

    # =========================================================================
    # Hyperparameters
    # =========================================================================
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices. The per-device batch size is computed as `global_batch_size // world_size`.",
    )
    epochs: int = Field(..., description="Total number of epochs to train.")

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
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to save checkpoints and logs. If not set, it defaults to `checkpoints/<project_name>/<run_name>`.",
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Whether to checkpoint the model after every evaluation.",
    )
    eval_interval: Optional[int] = Field(
        default=None,
        description="Number of epochs between evaluations. If not set, it defaults to evaluating only at the end of training.",
    )
    eval_save_outputs: List[str] = Field(
        default_factory=list,
        description="Evaluation output keys saved as tensors in the checkpoint directory.",
    )

    # =========================================================================
    # Aggregate settings (compose leaf settings for modules)
    # =========================================================================
    @property
    def model(self) -> ModelConfig_HRM_V1:
        """Compose TEMConfig from leaf settings.
        Creates the aggregate model configuration consumed by Model.
        """
        if self.configuration_path is None:
            model_config = ModelConfig_HRM_V1.model_validate(self, from_attributes=True)
        else:
            with open(self.configuration_path, "rb") as f:
                config_dict = tomllib.load(f)
            model_config = ModelConfig_HRM_V1.model_validate(config_dict, from_attributes=True)

        metadata = self._load_dataset_metadata(model_config.dataset.dataset_path, split="train")
        arch = model_config.arch.model_copy(
            update={
                "seq_len": metadata.seq_len,
                "vocab_size": metadata.vocab_size,
            }
        )
        return model_config.model_copy(update={"arch": arch})

    @staticmethod
    def _load_dataset_metadata(dataset_path: str, *, split: str) -> PuzzleDatasetMetadata:
        metadata_path = Path(dataset_path) / split / "dataset.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found at {metadata_path}. Ensure the dataset is built before training.")

        with metadata_path.open("r", encoding="utf-8") as f:
            return PuzzleDatasetMetadata.model_validate(json.load(f))

    @property
    def datamodule(self) -> PuzzleDatamoduleSettings:
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

    # Step _: Parse settings (CLI overrides TOML overrides defaults)
    settings = ExperimentSettings.load()

    # Step _: Seed everything for reproducibility
    seed_everything(settings.shared.seed, workers=True)

    # Step _:
    # Preparation of callbacks list: checkpointing + optional figure generation
    callbacks_list = []
    if settings.checkpoint is not None:
        callbacks_list.append(CheckpointCallback(settings.checkpoint))
    if settings.figures is not None and settings.figures.enabled:
        callbacks_list.append(FiguresCallback(settings.figures))

    trainer_settings = settings.trainer_settings

    # Step _: Build the PyTorch Lightning Trainer
    # This wires together logging, checkpointing, and training control
    trainer = Trainer(
        # TensorBoard logger for metrics and hyperparameters
        logger=Logger(settings.logger_settings) if settings.logger_settings is not None else None,
        # Callbacks: checkpointing + optional figure generation
        callbacks=callbacks_list if callbacks_list else None,
        # Lightning Trainer kwargs (extracted from config)
        max_steps=trainer_settings.max_steps,
        val_check_interval=trainer_settings.val_check_interval,
        log_every_n_steps=trainer_settings.log_every_n_steps,
        enable_progress_bar=trainer_settings.enable_progress_bar,
        default_root_dir=str(trainer_settings.default_root_dir) if trainer_settings.default_root_dir else None,
    )

    # Step _: Start training
    # The LightningModule wraps the TEM model and defines the training loop
    # The DataModule generates batches of walk data on-the-fly
    trainer.fit(
        # Lightning module: training step, optimizer, schedule computation
        model=Model(settings.resolved_model_config),
        # Data module: generates environment walks and batches
        datamodule=PuzzleDatamodule(settings.datamodule_settings),
        # Optional: resume from checkpoint
        ckpt_path=str(settings.ckpt_path) if settings.ckpt_path else None,
    )
