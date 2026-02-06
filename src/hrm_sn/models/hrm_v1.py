import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias

import lightning as L
import torch
from adam_atan2_pytorch import AdamAtan2 as AdamATan2
from pydantic import BaseModel, Field
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hrm_sn.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetMetadata, PuzzleDatasetSettings
from hrm_sn.loss import LossConfig
from hrm_sn.loss.act_head import ACTLossHead
from hrm_sn.modules.hrm import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config
from hrm_sn.training.buffers import FifoBuffer
from hrm_sn.training.optim import AdamATan2, AdamATan2Config, CastedSparseEmbeddingSignSGD_Distributed, CastedSparseEmbeddingSignSGDConfig
from hrm_sn.training.partial_reset import PartialResetBatchAssembler
from hrm_sn.training.rollout import EvaluationLoop, RolloutLoop
from hrm_sn.training.schedules import CosineAnnealingLR, LinearLR, SchedulerConfig, SequentialLR

# TODO: Move later to types.py
Batch: TypeAlias = Tuple[str, Dict[str, torch.Tensor], int]  # (set_name, batch_dict, global_effective_batch_size)
Device = torch.device


class ModelConfig(BaseModel, extra="forbid"):

    # Model architecture and data
    arch: HierarchicalReasoningModel_ACTV1Config = Field(
        ...,
        description="Architecture config. The keys in `arch.__pydantic_extra__` are passed to the model constructor.",
    )
    data: PuzzleDatasetSettings = Field(
        ...,
        description="Data config. The keys in `data` are passed to the dataset constructor.",
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

    # Hyperparameters
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices. The per-device batch size is computed as `global_batch_size // world_size`.",
    )
    epochs: int = Field(..., description="Total number of epochs to train.")

    lr: float = Field(
        ...,
        description="Base learning rate for the main optimizer (e.g. Adam). The learning rate for the puzzle embedding optimizer is set by `puzzle_emb_lr`.",
    )
    lr_min_ratio: float = Field(
        default=0.0,
        description="Minimum learning rate ratio for cosine decay. The learning rate will decay to `base_lr * lr_min_ratio` at the end of training.",
    )
    lr_warmup_steps: int = Field(
        default=0,
        description="Number of warmup steps for learning rate scheduling.",
    )

    weight_decay: float = Field(
        default=0.0,
        description="Weight decay for the main optimizer (e.g. Adam). Decay for puzzle embedding optimizer is set by `puzzle_emb_weight_decay`.",
    )
    beta1: float = Field(
        default=0.9,
        description="Beta 1 for Adam optimizer.",
    )
    beta2: float = Field(
        default=0.98,
        description="Beta 2 for Adam optimizer.",
    )

    # Puzzle embedding
    emb_lr: float = Field(
        ...,
        description="Base learning rate for the puzzle embedding optimizer (e.g. SignSGD).",
    )
    emb_weight_decay: float = Field(
        default=0.0,
        description="Weight decay for the puzzle embedding optimizer (e.g. SignSGD).",
    )

    # Names and tracking
    project_name: Optional[str] = Field(
        default=None,
        description="Project name. If not set, it defaults to the capitalized name of the dataset (e.g. `MATH` -> `Math ACT-torch`).",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Run name. If not set, it defaults to `<arch_name> <random_slug>` (e.g. `HrmV1 2x128 4L 16H 0.1D ACT-torch cool-slug`).",
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to save checkpoints and logs. If not set, it defaults to `checkpoints/<project_name>/<run_name>`.",
    )

    # Extras
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


@dataclass
class ModelState:
    carry: Any  # Model carry/state that persists across batches, initialized as None and set by the first batch
    q_halt_logits: Optional[torch.Tensor] = None  # Optional tensor of shape (batch_size,) with the halt logits for the question, used for loss computation and evaluation
    q_continue_logits: Optional[torch.Tensor] = None  # Optional tensor of shape (batch_size,) with the continue logits for the question, used for loss computation and evaluation
    steps: Optional[torch.Tensor] = None  # Optional tensor of shape (batch_size,) with the number of steps taken for each example, used for evaluation

    # Metadata
    step_id: Optional[int] = None  # Optional step identifier
    batch_id: Optional[int] = None  # Optional batch identifier
    set_name: Optional[str] = None  # Optional dataset split name (e.g. "train", "test", etc.)


class Model(L.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = HierarchicalReasoningModel_ACTV1(config.arch)
        self.loss_head = ACTLossHead(self.model, config.loss.name)
        self._config = config

        # one backward, step two opts (legacy parity)
        self.automatic_optimization = False
        self._train_carry = None
        self._val_carry = None

        # init buffers and assemblers for partial reset logic in training_step
        self._train_buffer = FifoBuffer(
            capacity_rows=4 * config.global_batch_size,  # or local batch size if you prefer
            keys=("inputs", "labels", "puzzle_identifiers"),
            pin_memory=True,
        )
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer,
            keys=("inputs", "labels", "puzzle_identifiers"),
        )

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def puzzle_emb(self):
        """Access to puzzle embeddings for optimizer configuration."""
        return self.model.puzzle_emb if hasattr(self.model, "puzzle_emb") else None

    def init_state(self, batch: Batch) -> ModelState:
        set_name, batch_dict, global_effective_bs = batch
        # TODO: properly use batch_dict and global_effective_bs if needed for state initialization
        return ModelState(carry=None, set_name=set_name)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[SequentialLR]]:
        optimizers = self.build_optimizers()
        schedulers = self.schedulers(optimizers)

        return optimizers, schedulers

    def build_optimizers(self) -> List[Optimizer]:
        # Main optimizer: all parameters except puzzle embeddings
        main_params = [p for n, p in self.model.named_parameters() if "puzzle_emb" not in n and p.requires_grad]

        # Sparse embedding optimizer: puzzle embedding buffers/parameters
        # The optimizer expects 3 params: local_ids (no grad), local_weights (with grad), and weights (no grad)
        if hasattr(self.model, "puzzle_emb") and self.model.puzzle_emb is not None:
            emb_params = [
                self.model.puzzle_emb.local_ids,  # local_ids, no grad
                self.model.puzzle_emb.local_weights,  # local_weights, requires_grad
                self.model.puzzle_emb.weights,  # global_weights, no grad
            ]
            optimizer_emb = CastedSparseEmbeddingSignSGD_Distributed(emb_params, self.config.optim_emb)
        else:
            # No puzzle embeddings, create dummy optimizer with empty params
            optimizer_emb = CastedSparseEmbeddingSignSGD_Distributed([], self.config.optim_emb)

        optimizer_main = AdamATan2(main_params, self.config.optim_main)

        return [optimizer_main, optimizer_emb]

    def schedulers(self, optimizers: List[Optimizer]) -> List[SequentialLR]:
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = self.config.scheduler.warmup_steps
        min_ratio = self.config.scheduler.min_ratio
        cosine_steps = max(1, total_steps - warmup_steps)

        def make_scheduler(optimizer):
            warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=self.config.lr * min_ratio)
            return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        return [make_scheduler(optimizer) for optimizer in optimizers]

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
        step = None
        for step in RolloutLoop(self.loss_head, self._train_carry, step_batch, horizon=1):
            pass  # TODO: Sum loss across steps if horizon > 1
        if step is None:
            raise ValueError("RolloutLoop did not yield any steps, cannot proceed with training step.")
        self._train_carry = step.carry

        # scaling: (1/global_batch_size) * loss, then backward TODO: use torch's built-in support for scaling
        loss = step.loss_sum / float(global_effective_bs)
        self.manual_backward(loss)

        opt_main, opt_emb = self.optimizers()  # type: ignore
        opt_main.step(); opt_main.zero_grad(set_to_none=True)  # fmt: skip
        opt_emb.step(); opt_emb.zero_grad(set_to_none=True)  # fmt: skip

        sch_main, sch_emb = self.lr_schedulers()  # type: ignore
        sch_main.step()  # type: ignore
        sch_emb.step()  # type: ignore

        # log: ACTLossHead metrics are sums; normalize like legacy
        count = step.metrics["count"].clamp_min(1)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/accuracy", step.metrics["accuracy"] / count, on_step=True, prog_bar=True)
        self.log("train/exact_accuracy", step.metrics["exact_accuracy"] / count, on_step=True)
        self.log("train/steps", step.metrics["steps"] / count, on_step=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        set_name, batch_dict, global_effective_bs = batch

        # Initialize carry/state on the first batch
        step = None
        for step in EvaluationLoop(self.loss_head, batch_dict, return_keys=self.config.eval_save_outputs):
            pass  # TODO: Sum loss across steps?
        if step is None:
            raise ValueError("Evaluation loop did not yield any steps, cannot log metrics.")
        count = step.metrics["count"].clamp_min(1)

        self.log(f"{set_name}/accuracy", step.metrics["accuracy"] / count, on_step=False, on_epoch=True)
        self.log(f"{set_name}/exact_accuracy", step.metrics["exact_accuracy"] / count, on_step=False, on_epoch=True)


def cosine_lr(step: int, *, base_lr: float, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * math.pi)).item())))
