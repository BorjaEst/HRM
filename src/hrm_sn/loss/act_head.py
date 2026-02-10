from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import hrm_sn.loss.cross_entropy as cross_entropy_module
from hrm_sn.loss.cross_entropy import LossType
from hrm_sn.training.act_controller import ACTController, ACTState

IGNORE_LABEL_ID = -100


# =================================================================================================
class ACTLossHead(nn.Module):
    def __init__(  # ------------------------------------------------------------------------------
            self, controller: ACTController, loss_type: LossType = "stablemax_cross_entropy"
        ) -> None:  # fmt: skip
        super().__init__()
        self.controller = controller
        self.loss_fn = getattr(cross_entropy_module, loss_type)

    def initial_state(  # -------------------------------------------------------------------------
            self, *args, **kwargs
        ) -> ACTState:  # fmt: skip
        return self.controller.initial_state(*args, **kwargs)

    def forward(  # -------------------------------------------------------------------------------
        self, return_keys: Sequence[str], state: ACTState, batch: Dict[str, Tensor],
    ) -> Tuple[Any, Tensor, Dict[str, Tensor], Optional[Dict[str, Tensor]], Tensor]:  # fmt: skip

        labels = state.data["labels"]
        if self.training:  # Inherited from nn.Module, automatically set by .train() and .eval()
            state, outputs = self.controller.step(state, batch, allow_halt=True, explore=True, compute_targets=True)
        else:
            state, outputs = self.controller.step(state, batch, allow_halt=True, explore=False)

        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs.logits, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = state.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs.halt_logits >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, state.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs.logits, labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs.halt_logits,
            seq_is_correct.to(outputs.halt_logits.dtype),
            reduction="sum",
        )

        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if outputs.target_continue is not None:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs.continue_logits,
                outputs.target_continue,
                reduction="sum",
            )

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        output_map = {
            "logits": outputs.logits,
            "halt_logits": outputs.halt_logits,
            "continue_logits": outputs.continue_logits,
            "action": outputs.action,
            "target_continue": outputs.target_continue,
        }
        detached_outputs = {k: output_map[k].detach() for k in return_keys if k in output_map and output_map[k] is not None}

        return (
            state,
            lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            state.halted.all(),
        )
