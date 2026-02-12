import itertools
import math
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias

import lightning as L
import torch
from adam_atan2_pytorch import AdamAtan2 as AdamATan2
from pydantic import BaseModel, Field
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hrm_sn import metrics
from hrm_sn.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetMetadata, PuzzleDatasetSettings
from hrm_sn.loss.act_head import ACTLossConfig, ACTLossHead
from hrm_sn.modules.hrm import HRMConfig, HRModel
from hrm_sn.training.act_controller import ACTController, ACTControllerConfig
from hrm_sn.training.buffers import FifoBuffer
from hrm_sn.training.collector import PartialResetCollector
from hrm_sn.training.optim import AdamATan2, AdamATan2Config, CastedSparseEmbeddingSignSGD_Distributed, CastedSparseEmbeddingSignSGDConfig
from hrm_sn.training.partial_reset import PartialResetBatchAssembler
from hrm_sn.training.rollout import EvaluationLoop, RolloutLoop
from hrm_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR

# TODO: Move later to types.py
Batch: TypeAlias = Tuple[str, Dict[str, Tensor], int]  # (set_name, batch_dict, global_effective_batch_size)
Device = torch.device


class ModelConfig_HRM_V1(BaseModel, extra="forbid"):

    # Model architecture and data
    architecture: HRMConfig = Field(
        ...,
        description="Architecture config for the HRM model. The keys in `architecture` are passed to the HRModel constructor.",
    )
    act_controller: ACTControllerConfig = Field(
        ...,
        description="Configuration for the ACT controller, which manages halting and partial resets during training. The keys in `act_controller` are passed to the ACTController constructor.",
    )
    loss: ACTLossConfig = Field(
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

    # Extra
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices. The per-device batch size is computed as `global_batch_size // world_size`.",
    )  # TODO: consider moving to BufferSettings or similar


@dataclass
class ModelState:
    carry: Any  # Model carry/state that persists across batches, initialized as None and set by the first batch
    halt_logits: Optional[Tensor] = None  # Optional tensor of shape (batch_size,) with the halt logits for the question, used for loss computation and evaluation
    continue_logits: Optional[Tensor] = None  # Optional tensor of shape (batch_size,) with the continue logits for the question, used for loss computation and evaluation
    steps: Optional[Tensor] = None  # Optional tensor of shape (batch_size,) with the number of steps taken for each example, used for evaluation

    # Metadata
    step_id: Optional[int] = None  # Optional step identifier
    batch_id: Optional[int] = None  # Optional batch identifier
    set_name: Optional[str] = None  # Optional dataset split name (e.g. "train", "test", etc.)


class Model(L.LightningModule):
    def __init__(self, config: ModelConfig_HRM_V1):
        super().__init__()
        self.model = HRModel(config.architecture)
        self.controller = ACTController(self.model, config.act_controller)
        self.loss_head = ACTLossHead(self.controller, config.loss)
        self._config = config

        # one backward, step two opts (legacy parity)
        self.automatic_optimization = False
        self._train_carry = None
        self._val_carry = None

        # init buffers and assemblers for partial reset logic in training_step
        self._train_buffer = FifoBuffer(
            capacity_rows=4 * config.global_batch_size,  # or local batch size if you prefer
            keys=("inputs", "labels"),
            pin_memory=True,
        )
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer,
            keys=("inputs", "labels"),
        )

    @property
    def config(self) -> ModelConfig_HRM_V1:
        return self._config

    def init_state(self, batch: Batch) -> ModelState:
        set_name, batch_dict, global_effective_bs = batch
        # TODO: properly use batch_dict and global_effective_bs if needed for state initialization
        return ModelState(carry=None, set_name=set_name)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[SequentialLR]]:
        optimizers = self.build_optimizers()
        schedulers = self.schedulers(optimizers)

        return optimizers, schedulers

    def build_optimizers(self) -> List[Optimizer]:
        # Main optimizer: all trainable parameters
        main_params = [p for p in self.model.parameters() if p.requires_grad]
        # Dummy secondary optimizer to preserve scheduler/step structure
        optimizer_emb = CastedSparseEmbeddingSignSGD_Distributed([], self.config.optim_emb)
        optimizer_main = AdamATan2(main_params, self.config.optim_main)

        return [optimizer_main, optimizer_emb]

    def schedulers(self, optimizers: List[Optimizer]) -> List[SequentialLR]:
        total_steps = int(self.trainer.estimated_stepping_batches)
        config = self.config.scheduler
        return [CosineAnnealingLRWithWarmup(opt, total_steps, config) for opt in optimizers]

    def transfer_batch_to_device(self, batch: Batch, device: Device, dataloader_idx: int = 0) -> Batch:
        set_name, batch_dict, global_effective_bs = batch
        batch_dict = {k: v.to(device, non_blocking=True) for k, v in batch_dict.items()}
        return set_name, batch_dict, global_effective_bs

    def on_train_epoch_start(self) -> None:
        self._train_carry = None
        self._train_buffer.clear()

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        set_name, batch_dict, global_effective_bs = batch

        # Initialize carry/state on the first batch
        if self._train_carry is None:
            self._train_carry = self.loss_head.initial_carry(batch_dict)

        # Partial reset logic: determine which examples in the batch are "reset" (halted)
        # and which are "keep" (continue), then assemble the step batch accordingly
        step_batch = self._train_batch_assembler.make_step_batch(
            incoming=batch_dict,
            reset_mask=self._train_carry.halted,  # vectorized done flags
        )

        # Horizon=1 matches legacy behavior: exactly one ACT step per mini-batch.
        step_batches = repeat(step_batch, 1)

        step = None
        for step in RolloutLoop(self.loss_head, step_batches, carry0=self._train_carry):
            pass  # TODO: Sum loss across steps if horizon > 1
        if step is None:
            raise ValueError("RolloutLoop did not yield any steps, cannot proceed with training step.")
        self._train_carry = step.carry

        # scaling: (1/global_batch_size) * loss, then backward TODO: use torch's built-in support for scaling
        loss = step.loss / float(global_effective_bs)
        self.manual_backward(loss)

        opt_main, opt_emb = self.optimizers()  # type: ignore
        opt_main.step(); opt_main.zero_grad(set_to_none=True)  # fmt: skip
        opt_emb.step(); opt_emb.zero_grad(set_to_none=True)  # fmt: skip

        sch_main, sch_emb = self.lr_schedulers()  # type: ignore
        sch_main.step()  # type: ignore
        sch_emb.step()  # type: ignore

        # log: ACTLossHead metrics are sums; normalize like legacy
        log_metrics = metrics.to_log_dict(step.metrics, prefix="train/", global_batch_size=global_effective_bs)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/accuracy", log_metrics["train/accuracy"], on_step=True, prog_bar=True)
        self.log("train/exact_accuracy", log_metrics["train/exact_accuracy"], on_step=True)
        self.log("train/steps", log_metrics["train/steps"], on_step=True)
        self.log("train/lm_loss", log_metrics["train/lm_loss"], on_step=True)
        self.log("train/q_halt_loss", log_metrics["train/q_halt_loss"], on_step=True)
        self.log("train/q_continue_loss", log_metrics["train/q_continue_loss"], on_step=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        set_name, batch_dict, global_effective_bs = batch

        # Horizon=1 matches legacy behavior: exactly one ACT step per mini-batch.
        step_batches = repeat(batch_dict, 1)

        # Initialize carry/state on the first batch
        step = None
        for step in EvaluationLoop(self.loss_head, step_batches):
            pass  # TODO: Sum loss across steps?
        if step is None:
            raise ValueError("Evaluation loop did not yield any steps, cannot log metrics.")
        log_metrics = metrics.to_log_dict(step.metrics, prefix=f"{set_name}/", global_batch_size=global_effective_bs)

        self.log(f"{set_name}/accuracy", log_metrics[f"{set_name}/accuracy"], on_step=False, on_epoch=True)
        self.log(f"{set_name}/exact_accuracy", log_metrics[f"{set_name}/exact_accuracy"], on_step=False, on_epoch=True)
        self.log(f"{set_name}/lm_loss", log_metrics[f"{set_name}/lm_loss"], on_step=False, on_epoch=True)
        self.log(f"{set_name}/q_halt_loss", log_metrics[f"{set_name}/q_halt_loss"], on_step=False, on_epoch=True)
        self.log(f"{set_name}/q_continue_loss", log_metrics[f"{set_name}/q_continue_loss"], on_step=False, on_epoch=True)


def cosine_lr(step: int, *, base_lr: float, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * math.pi)).item())))
