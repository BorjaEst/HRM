from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

import hrm_sn.loss.cross_entropy as cross_entropy_module
from hrm_sn.loss.cross_entropy import LossType
from hrm_sn.metrics import HaltedAgg, LossAgg, StepMetrics, TokenAgg
from hrm_sn.training.act_controller import ACTController, ACTOutput, ACTState

IGNORE_LABEL_ID = -100


class ACTLossConfig(BaseModel, extra="forbid"):
    function: LossType = Field(default="stablemax_cross_entropy", description="The loss function to use for the modeling loss.")


@dataclass(frozen=True)
class CorrectnessStats:
    """ """

    mask: Tensor  # Boolean tensor indicating which tokens contribute to the loss (e.g., non-padding tokens).

    @property
    def loss_counts(self) -> Tensor:
        """Number of tokens contributing to the loss for each sequence (i.e., non-padding tokens)."""
        return self.mask.sum(-1)

    @property
    def loss_divisor(self) -> Tensor:
        """Avoid NaNs in division"""
        return self.loss_counts.clamp_min(1).unsqueeze(-1)

    is_correct: Tensor  # Boolean tensor indicating which tokens were predicted correctly (after masking).

    @property
    def seq_is_correct(self) -> Tensor:
        """Whether the entire sequence is correct (ignoring padding)"""
        return self.is_correct.sum(-1) == self.loss_counts


@dataclass(frozen=True)
class Losses:
    """ """

    loss_sum: Tensor  # Per-step loss sum for the main task
    q_halt_loss_sum: Tensor  # Loss sum for the halting decision
    q_continue_loss_sum: Optional[Tensor]  # Loss sum for the continue decision (if applicable)

    @property
    def total(self) -> Tensor:
        """ """
        q_continue_loss_sum = self.q_continue_loss_sum
        if q_continue_loss_sum is None:
            q_continue_loss_sum = torch.tensor(0.0, device=self.loss_sum.device)
        return self.loss_sum + 0.5 * (self.q_halt_loss_sum + q_continue_loss_sum)


@dataclass(frozen=True)
class StepResult:
    """Per-step output contract for loss heads and rollout loops."""

    loss: Tensor
    carry: ACTState
    metrics: StepMetrics
    outputs: Optional[ACTOutput] = None

    @property
    def all_finished(self) -> Tensor:
        """Whether all sequences in the batch have halted."""
        return self.carry.halted.all()


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
    ) -> StepResult:  # fmt: skip
        """ """
        if self.training:
            carry, outputs = self.controller.step(carry, batch, allow_halt=True, explore=True, compute_targets=True)
        else:
            carry, outputs = self.controller.step(carry, batch, allow_halt=False, explore=False, compute_targets=False)
        labels = carry.data["labels"]

        with torch.no_grad():
            stats = self.compute_correctness(outputs, labels)

        losses = self.compute_losses(outputs, labels, stats)
        metrics = self.compute_metrics(carry, outputs, stats, losses)

        return StepResult(loss=losses.total, carry=carry, metrics=metrics, outputs=outputs)

    def compute_correctness(  # ------------------------------------------------------------------
        self, outputs: ACTOutput, labels: Tensor
    ) -> CorrectnessStats:  # fmt: skip
        """ """
        mask = labels != IGNORE_LABEL_ID
        is_correct = mask & (torch.argmax(outputs.logits, dim=-1) == labels)
        return CorrectnessStats(mask=mask, is_correct=is_correct)

    def compute_metrics(  # -----------------------------------------------------------------------
        self, state: ACTState, outputs: ACTOutput, stats: CorrectnessStats, losses: Losses,
    ) -> StepMetrics:  # fmt: skip
        """ """
        halted_mask = state.halted & (stats.loss_counts > 0)  # (B,)
        halted_weights = halted_mask.to(torch.float32)

        token_correct_per_seq = stats.is_correct.to(torch.float32).sum(-1)  # (B,)
        token_count_per_seq = stats.loss_counts.clamp_min(1).to(torch.float32)  # (B,)
        seq_accuracy = token_correct_per_seq / token_count_per_seq  # (B,)

        pred_halt = outputs.halt_logits > outputs.continue_logits  # (B,)
        q_halt_correct = pred_halt == stats.seq_is_correct  # (B,)

        q_continue_correct: Optional[Tensor] = None
        if outputs.target_continue is not None:
            pred_continue = outputs.continue_logits >= 0  # (B,)
            q_continue_correct = pred_continue == stats.seq_is_correct  # (B,)

        halted = self._build_halted_agg(state, stats, halted_mask, halted_weights, seq_accuracy, q_halt_correct, q_continue_correct)
        tokens = self._build_token_agg(token_correct_per_seq, token_count_per_seq, halted_weights)
        loss = self._build_loss_agg(losses, batch_size=outputs.logits.shape[0])

        return StepMetrics(halted=halted, tokens=tokens, loss=loss)

    def _build_halted_agg(  # --------------------------------------------------------------------
        self, state: ACTState, stats: CorrectnessStats, halted_mask: Tensor, halted_weights: Tensor,
        seq_accuracy: Tensor, q_halt_correct: Tensor, q_continue_correct: Optional[Tensor]=None,
    ) -> HaltedAgg:  # fmt: skip
        """Build aggregated metrics for halted sequences."""
        if q_continue_correct is None:
            q_continue_correct = torch.zeros_like(halted_mask)  # (B,) bool

        return HaltedAgg(
            halted_count=halted_weights.sum(),
            accuracy_sum=(seq_accuracy * halted_weights).sum(),
            exact_sum=(stats.seq_is_correct & halted_mask).to(torch.float32).sum(),
            steps_sum=(state.steps * halted_weights.to(state.steps.dtype)).sum(),
            q_halt_correct_sum=(q_halt_correct & halted_mask).to(torch.float32).sum(),
            q_continue_correct_sum=(q_continue_correct & halted_mask).to(torch.float32).sum(),
        )

    def _build_token_agg(  # ---------------------------------------------------------------------
        self, token_correct_per_seq: Tensor, token_count_per_seq: Tensor, halted_weights: Tensor,
    ) -> TokenAgg:  # fmt: skip
        """Build token-level aggregates for halted sequences."""
        return TokenAgg(
            token_correct_sum=(token_correct_per_seq * halted_weights).sum(),
            token_count_sum=(token_count_per_seq * halted_weights).sum(),
        )

    def _build_loss_agg(  # ----------------------------------------------------------------------
        self, losses: Losses, *, batch_size: int,
    ) -> LossAgg:  # fmt: skip
        """Build loss aggregates with consistent device/dtype handling."""
        q_continue_loss_sum = losses.q_continue_loss_sum
        if q_continue_loss_sum is None:
            q_continue_loss_sum = losses.loss_sum.new_zeros(())

        return LossAgg(
            lm_loss_sum=losses.loss_sum.detach(),
            q_halt_loss_sum=losses.q_halt_loss_sum.detach(),
            q_continue_loss_sum=q_continue_loss_sum.detach(),
            batch_count=losses.loss_sum.new_tensor(batch_size, dtype=torch.float32),
        )

    def compute_losses(  # -----------------------------------------------------------------------
        self, outputs: ACTOutput, labels: Tensor, stats: CorrectnessStats
    ) -> Losses:  # fmt: skip
        """ """
        loss_per_token = self.loss_fn(outputs.logits, labels, ignore_index=IGNORE_LABEL_ID)
        loss_per_seq = loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)
        loss_sum = loss_per_seq.sum()

        # Halting loss: encourage the model to halt when the sequence is correct, and continue otherwise
        q_halt_loss = F.binary_cross_entropy_with_logits(
            input=outputs.halt_logits,
            target=stats.seq_is_correct.to(outputs.halt_logits.dtype),
            reduction="sum",
        )

        # Continue loss: if targets provided encourage the model to continue when the sequence is incorrect
        if outputs.target_continue is not None:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                input=outputs.continue_logits,
                target=outputs.target_continue,
                reduction="sum",
            )
        else:
            q_continue_loss = None

        # Return all losses in a structured way
        return Losses(loss_sum, q_halt_loss, q_continue_loss)
