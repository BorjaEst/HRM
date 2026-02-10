from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

import hrm_sn.loss.cross_entropy as cross_entropy_module
from hrm_sn.loss.cross_entropy import LossType
from hrm_sn.metrics import Metric
from hrm_sn.training.act_controller import ACTController, ACTOutput, ACTState

IGNORE_LABEL_ID = -100


class ACTLossConfig(BaseModel, extra="forbid"):
    function: LossType = "stablemax_cross_entropy"


@dataclass(frozen=True)
class CorrectnessStats:
    mask: Tensor
    loss_counts: Tensor
    loss_divisor: Tensor
    is_correct: Tensor
    seq_is_correct: Tensor


@dataclass
class LossHeadOutput:
    carry: ACTState
    loss: Tensor
    metrics: Metric
    outputs: ACTOutput
    all_finished: Tensor


@dataclass
class Losses:
    loss: Tensor  # Modeling loss (e.g., cross-entropy)
    q_halt_loss: Tensor  # Loss for the halting decision
    q_continue_loss: Optional[Tensor] = None  # Loss for the continue decision

    @property
    def total(self) -> Tensor:
        """ """
        q_continue_loss = self.q_continue_loss or torch.tensor(0.0, device=self.loss.device)
        return self.loss + 0.5 * (self.q_halt_loss + q_continue_loss)


# =================================================================================================
class ACTLossHead(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, controller: ACTController, config: ACTLossConfig,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._controller = controller
        self._config = config

    @property
    def controller(self) -> ACTController:
        """ """
        return self._controller

    @property
    def config(self) -> ACTLossConfig:
        """ """
        return self._config

    @property
    def loss_fn(self) -> Any:
        """ """
        return getattr(cross_entropy_module, self._config.function)

    def initial_carry(  # -------------------------------------------------------------------------
        self, *args, **kwargs
    ) -> ACTState:  # fmt: skip
        """ """
        return self.controller.initial_state(*args, **kwargs)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Dict[str, Tensor], carry: ACTState,
    ) -> LossHeadOutput:  # fmt: skip
        """ """
        if self.training:
            carry, outputs = self.controller.step(carry, batch, allow_halt=True, explore=True, compute_targets=True)
        else:
            carry, outputs = self.controller.step(carry, batch, allow_halt=True, explore=False)
        labels = carry.data["labels"]

        with torch.no_grad():
            stats = self.compute_correctness(outputs, labels)
            losses = self.compute_losses(outputs, labels, stats)

        return LossHeadOutput(
            carry=carry,
            loss=losses.total,
            metrics=self.compute_metrics(carry, outputs, stats, losses),
            outputs=outputs.detach(),
            all_finished=carry.halted.all(),
        )

    def compute_correctness(  # ------------------------------------------------------------------
        self, outputs: Any, labels: Tensor
    ) -> CorrectnessStats:  # fmt: skip
        """ """
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        is_correct = mask & (torch.argmax(outputs.logits, dim=-1) == labels)

        return CorrectnessStats(
            mask=mask,
            loss_counts=loss_counts,
            loss_divisor=loss_counts.clamp_min(1).unsqueeze(-1),  # Avoid NaNs in division
            is_correct=is_correct,
            seq_is_correct=is_correct.sum(-1) == loss_counts,
        )

    def compute_metrics(  # -----------------------------------------------------------------------
        self, state: ACTState, outputs: Any, stats: CorrectnessStats, losses: Losses,
    ) -> Metric:  # fmt: skip
        """ """
        valid_metrics = state.halted & (stats.loss_counts > 0)
        return Metric(
            count=valid_metrics.sum(),
            steps=torch.where(valid_metrics, state.steps, 0).sum(),
            accuracy=torch.where(valid_metrics, (stats.is_correct.to(torch.float32) / stats.loss_divisor).sum(-1), 0).sum(),
            exact_accuracy=(valid_metrics & stats.seq_is_correct).sum(),
            q_halt_accuracy=(valid_metrics & ((outputs.halt_logits >= 0) == stats.seq_is_correct)).sum(),
            q_continue_accuracy=(
                (valid_metrics & ((outputs.continue_logits >= 0) == stats.seq_is_correct)).sum()
                if outputs.target_continue is not None
                else torch.tensor(0, device=state.steps.device)
            ),
            loss_current=losses.loss.detach(),
            q_halt_loss=losses.q_halt_loss.detach(),
            q_continue_loss=losses.q_continue_loss.detach() if losses.q_continue_loss is not None else torch.tensor(0.0, device=losses.loss.device),
        )

    def compute_losses(  # -----------------------------------------------------------------------
        self, outputs: ACTOutput, labels: Tensor, stats: CorrectnessStats
    ) -> Losses:  # fmt: skip
        """ """
        loss = (self.loss_fn(outputs.logits, labels, ignore_index=IGNORE_LABEL_ID) / stats.loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs.halt_logits,
            stats.seq_is_correct.to(outputs.halt_logits.dtype),
            reduction="sum",
        )

        q_continue_loss = torch.tensor(0.0, device=loss.device)
        if outputs.target_continue is not None:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs.continue_logits,
                outputs.target_continue,
                reduction="sum",
            )

        return Losses(
            loss=loss,
            q_halt_loss=q_halt_loss,
            q_continue_loss=q_continue_loss,
        )
