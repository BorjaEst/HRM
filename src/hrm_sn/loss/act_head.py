from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hrm_sn.training.act_controller import ACTController, ACTControllerCarry

IGNORE_LABEL_ID = -100


class ACTLossHead(nn.Module):
    def __init__(self, controller: ACTController, loss_type: str):
        super().__init__()
        self.controller = controller
        self.loss_fn = globals()[loss_type]  # TODO: Avoid using globals() for loss function lookup, consider a more explicit mapping or factory pattern

    def initial_carry(self, *args, **kwargs):
        return self.controller.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, Tensor, Dict[str, Tensor], Optional[Dict[str, Tensor]], Tensor]:
        carry: ACTControllerCarry = model_kwargs["carry"]
        batch: Dict[str, Tensor] = model_kwargs["batch"]
        new_carry, outputs = self.controller.step(carry, batch, training=self.controller.model.training)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["halt_logits"],
            seq_is_correct.to(outputs["halt_logits"].dtype),
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
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return (
            new_carry,
            lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            new_carry.halted.all(),
        )
