from typing import Dict, Optional

import torch
from torch import Tensor

from hrm_sn.metrics.types import HaltedAgg, LossAgg, StepMetrics, TokenAgg

__all__ = ["HaltedAgg", "LossAgg", "StepMetrics", "TokenAgg", "flatten_raw", "to_log_dict"]


def flatten_raw(  # ---------------------------------------------------------------------------
    step_metrics: StepMetrics, *, prefix: str = "",
) -> Dict[str, Tensor]:  # fmt: skip
    """Return a flat dict of raw sums/denominators for debugging or reduction."""
    return {
        f"{prefix}halted_count": step_metrics.halted.halted_count,
        f"{prefix}accuracy_sum": step_metrics.halted.accuracy_sum,
        f"{prefix}exact_sum": step_metrics.halted.exact_sum,
        f"{prefix}steps_sum": step_metrics.halted.steps_sum,
        f"{prefix}q_halt_correct_sum": step_metrics.halted.q_halt_correct_sum,
        f"{prefix}q_continue_correct_sum": step_metrics.halted.q_continue_correct_sum,
        f"{prefix}token_correct_sum": step_metrics.tokens.token_correct_sum,
        f"{prefix}token_count_sum": step_metrics.tokens.token_count_sum,
        f"{prefix}lm_loss_sum": step_metrics.loss.lm_loss_sum,
        f"{prefix}q_halt_loss_sum": step_metrics.loss.q_halt_loss_sum,
        f"{prefix}q_continue_loss_sum": step_metrics.loss.q_continue_loss_sum,
        f"{prefix}batch_count": step_metrics.loss.batch_count,
    }


def to_log_dict(  # ---------------------------------------------------------------------------
    step_metrics: StepMetrics, *, prefix: str, global_batch_size: Optional[int | Tensor] = None,
) -> Dict[str, Tensor]:  # fmt: skip
    """Return normalized log scalars with deterministic key names."""
    device = step_metrics.loss.lm_loss_sum.device
    dtype = step_metrics.loss.lm_loss_sum.dtype

    halted_denom = step_metrics.halted.halted_count.clamp_min(1)

    if global_batch_size is None:
        batch_denom = step_metrics.loss.batch_count
    elif isinstance(global_batch_size, Tensor):
        batch_denom = global_batch_size.to(device=device, dtype=dtype)
    else:
        batch_denom = torch.tensor(global_batch_size, device=device, dtype=dtype)

    batch_denom = batch_denom.clamp_min(1)

    return {
        f"{prefix}accuracy": step_metrics.halted.accuracy_sum / halted_denom,
        f"{prefix}exact_accuracy": step_metrics.halted.exact_sum / halted_denom,
        f"{prefix}steps": step_metrics.halted.steps_sum / halted_denom,
        f"{prefix}q_halt_accuracy": step_metrics.halted.q_halt_correct_sum / halted_denom,
        f"{prefix}q_continue_accuracy": step_metrics.halted.q_continue_correct_sum / halted_denom,
        f"{prefix}lm_loss": step_metrics.loss.lm_loss_sum / batch_denom,
        f"{prefix}q_halt_loss": step_metrics.loss.q_halt_loss_sum / batch_denom,
        f"{prefix}q_continue_loss": step_metrics.loss.q_continue_loss_sum / batch_denom,
    }
