"""Adaptive Computation Time (ACT) loss head.

This module defines the loss/metrics head used together with :class:`hrm_sn.training.act_controller.ACTController`.

At each ACT step the controller produces:

- token logits for the main task (language-model-style cross entropy over tokens)
- halting/continuation logits for a per-sequence decision

The loss head:

1) computes masked token-level modeling loss
2) derives per-sequence correctness from token predictions
3) trains the halting decision to match that correctness signal
4) aggregates step metrics, focusing on sequences that have *actually halted*

Notes
-----
- Labels use ``IGNORE_LABEL_ID`` (default ``-100``) to mark padding / non-loss tokens.
- Metrics intentionally only count halted sequences to avoid reporting partial-rollout performance.
"""

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

Batch = Dict[str, Tensor]  # Generic batch type, can be specialized as needed
IGNORE_LABEL_ID = -100


# =================================================================================================
class ACTLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTLossHead`.

    Attributes
    ----------
    function:
        Name of the token-level modeling loss function to use. This is resolved as an
        attribute on :mod:`hrm_sn.loss.cross_entropy`.
    """

    function: LossType = Field(
        default="stablemax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )


# =================================================================================================
@dataclass(frozen=True)
class CorrectnessStats:
    """Derived correctness signals used for halting supervision and metrics.

    This structure is computed under ``torch.no_grad()`` because it is used as a
    training target for the halting/continuation heads and for logging.
    """

    mask: Tensor  # Boolean tensor indicating which tokens contribute to the loss (e.g., non-padding tokens).

    @property
    def loss_counts(self) -> Tensor:
        """Number of tokens contributing to the loss for each sequence (i.e., non-padding tokens)."""
        return self.mask.sum(-1)

    @property
    def loss_divisor(self) -> Tensor:
        """Divisor used for per-sequence normalization.

        Clamps to at least 1 to avoid division-by-zero when a sequence has no
        supervised tokens.
        """
        return self.loss_counts.clamp_min(1).unsqueeze(-1)

    is_correct: Tensor  # Boolean tensor indicating which tokens were predicted correctly (after masking).

    @property
    def seq_is_correct(self) -> Tensor:
        """Whether the entire sequence is correct (ignoring masked/padding tokens)."""
        return self.is_correct.sum(-1) == self.loss_counts


# =================================================================================================
@dataclass(frozen=True)
class Losses:
    """Structured loss components produced by :class:`ACTLossHead`.

    All fields are *sums* over the batch (and for ``loss_sum`` also over tokens via
    per-sequence normalization).
    """

    loss_sum: Tensor  # Per-step loss sum for the main task
    q_halt_loss_sum: Tensor  # Loss sum for the halting decision
    q_continue_loss_sum: Optional[Tensor]  # Loss sum for the continue decision (if applicable)

    @property
    def total(self) -> Tensor:
        """Total scalar loss used for back-propagation.

        The halting and continuation losses are down-weighted (0.5 each) relative
        to the modeling loss.
        """
        q_continue_loss_sum = self.q_continue_loss_sum
        if q_continue_loss_sum is None:
            q_continue_loss_sum = torch.tensor(0.0, device=self.loss_sum.device)
        return self.loss_sum + 0.5 * (self.q_halt_loss_sum + q_continue_loss_sum)


# =================================================================================================
@dataclass(frozen=True)
class ACTStepOutput:
    """Per-step output contract for loss heads and rollout loops."""

    loss: Tensor  # Scalar loss for this step, used for back-propagation
    carry: ACTState  # Updated carry/state after this step, used for the next step's input
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging

    t: Optional[int] = None  # Step index (optional, may be set by the rollout loop)
    outputs: Optional[ACTOutput] = None  # Raw controller outputs

    @property
    def all_finished(self) -> Tensor:
        """Whether all sequences in the batch have halted."""
        return self.carry.halted.all()


# =================================================================================================
class ACTLossHead(nn.Module):
    """Loss head for ACT rollouts.

    The head is called once per rollout step. It delegates state transitions to the
    :class:`~hrm_sn.training.act_controller.ACTController`, computes losses, and
    returns both the scalar loss and aggregated metrics.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, controller: ACTController, config: ACTLossConfig,
    ) -> None:  # fmt: skip
        """Create an ACT loss head.

        Parameters
        ----------
        controller:
            The ACT controller responsible for maintaining rollout state and producing
            logits/targets.
        config:
            Loss configuration (primarily which token-loss function to use).
        """
        super().__init__()
        self._controller = controller
        self._config = config

    @property
    def controller(self) -> ACTController:
        """The underlying :class:`ACTController` used to step the rollout."""
        return self._controller

    @property
    def config(self) -> ACTLossConfig:
        """Configuration object for this loss head."""
        return self._config

    @property
    def loss_fn(self) -> Any:
        """Resolved token-level loss function.

        The config stores a string/enum name which is resolved on
        :mod:`hrm_sn.loss.cross_entropy`.
        """
        return getattr(cross_entropy_module, self._config.function)

    def initial_carry(  # -------------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> ACTState:  # fmt: skip
        """Create the initial :class:`ACTState` for a new rollout."""
        return self.controller.initial_state(batch_sample)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, carry: ACTState, t: Optional[int]=None
    ) -> ACTStepOutput:  # fmt: skip
        """Run one ACT step, returning the step loss, updated state, and metrics.

        Training vs eval behavior:

        - **Training**: allow halting, enable exploration, and compute controller targets
          (e.g. continuation supervision).
        - **Eval**: do not allow halting and do not compute targets; the loop/driver is
          expected to manage termination and reporting separately.
        """
        if self.training:
            carry, outputs = self.controller.step(carry, batch, allow_halt=True, explore=True, compute_targets=True)
        else:
            carry, outputs = self.controller.step(carry, batch, allow_halt=False, explore=False, compute_targets=False)
        labels = carry.data["labels"]

        with torch.no_grad():
            # Correctness is used as a supervision signal for halting/continuation.
            # Keeping it out of the graph avoids gradients flowing through argmax.
            stats = self.compute_correctness(outputs, labels)

        losses = self.compute_losses(outputs, labels, stats)
        metrics = self.compute_metrics(carry, outputs, stats, losses)

        return ACTStepOutput(loss=losses.total, carry=carry, metrics=metrics, t=t, outputs=outputs)

    def compute_correctness(  # ------------------------------------------------------------------
        self, outputs: ACTOutput, labels: Tensor
    ) -> CorrectnessStats:  # fmt: skip
        """Compute token- and sequence-level correctness for the current step.

        Correct tokens are those whose argmax prediction matches the label, restricted
        to non-ignored label positions.
        """
        mask = labels != IGNORE_LABEL_ID
        is_correct = mask & (torch.argmax(outputs.logits, dim=-1) == labels)
        return CorrectnessStats(mask=mask, is_correct=is_correct)

    def compute_metrics(  # -----------------------------------------------------------------------
        self, state: ACTState, outputs: ACTOutput, stats: CorrectnessStats, losses: Losses,
    ) -> StepMetrics:  # fmt: skip
        """Aggregate step metrics.

        Metrics focus on sequences that have halted in the *current* carry. This keeps
        logged accuracy and loss aligned with the ACT decision process (partial sequences
        still computing are excluded).
        """
        halted_mask = state.halted & (stats.loss_counts > 0)  # (B,)
        halted_weights = halted_mask.to(torch.float32)

        token_correct_per_seq = stats.is_correct.to(torch.float32).sum(-1)  # (B,)
        token_count_per_seq = stats.loss_counts.clamp_min(1).to(torch.float32)  # (B,)
        seq_accuracy = token_correct_per_seq / token_count_per_seq  # (B,)

        pred_halt = outputs.halt_logits > outputs.continue_logits  # (B,)
        q_halt_correct = pred_halt == stats.seq_is_correct  # (B,)

        q_continue_correct: Optional[Tensor] = None
        if outputs.target_continue is not None:
            # The continue head uses a dedicated target produced by the controller.
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
        """Compute modeling + ACT decision losses.

        - **Modeling loss**: token-level cross entropy over non-ignored labels, normalized
            per sequence and then summed over the batch.
        - **Halting loss**: binary cross entropy encouraging the model to halt when the
            entire sequence is correct, and not halt otherwise.
        - **Continuation loss**: optional auxiliary BCE loss, only computed when the
            controller provides ``outputs.target_continue``.
        """
        loss_per_token = self.loss_fn(outputs.logits, labels, ignore_index=IGNORE_LABEL_ID)
        loss_per_seq = loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)
        loss_sum = loss_per_seq.sum()

        # Halting loss: match "halt" to sequence correctness.
        # Using seq-level supervision avoids rewarding early halting on partially-correct sequences.
        q_halt_loss = F.binary_cross_entropy_with_logits(
            input=outputs.halt_logits,
            target=stats.seq_is_correct.to(outputs.halt_logits.dtype),
            reduction="sum",
        )

        # Continue loss: optional auxiliary supervision from the controller.
        if outputs.target_continue is not None:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                input=outputs.continue_logits,
                target=outputs.target_continue,
                reduction="sum",
            )
        else:
            q_continue_loss = None

        # Return all losses in a structured way for consistent logging.
        return Losses(loss_sum, q_halt_loss, q_continue_loss)
