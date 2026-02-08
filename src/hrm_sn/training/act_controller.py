from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor


class ACTNetwork(Protocol):
    training: bool

    def empty_carry(self, batch_size: int) -> Any: ...

    def reset_carry(self, reset_flag: Tensor, carry: Any) -> Any: ...

    def __call__(self, carry: Any, batch: Dict[str, Tensor]) -> Tuple[Any, Tensor, Tuple[Tensor, Tensor]]: ...


class ACTControllerConfig(BaseModel, extra="forbid"):
    halt_max_steps: int = Field(
        ...,
        ge=1,
        description="Maximum number of ACT steps.",
    )
    halt_exploration_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Exploration probability for ACT halting.",
    )


@dataclass
class ACTControllerCarry:
    inner_carry: Any
    steps: Tensor
    halted: Tensor
    current_data: Dict[str, Tensor]


class ACTController:
    """Training-time ACT controller for halting and partial-reset batch slots."""

    def __init__(self, model: ACTNetwork, config: ACTControllerConfig) -> None:
        self._model = model
        self._config = config

    @property
    def model(self) -> ACTNetwork:
        return self._model

    @property
    def config(self) -> ACTControllerConfig:
        return self._config

    def initial_carry(self, batch: Dict[str, Tensor]) -> ACTControllerCarry:
        batch_size = batch["inputs"].shape[0]
        return ACTControllerCarry(
            inner_carry=self.model.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def step(self, carry: ACTControllerCarry, batch: Dict[str, Tensor], *, training: bool) -> Tuple[ACTControllerCarry, Dict[str, Tensor]]:
        new_inner_carry = self.model.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (halt_logits, continue_logits) = self.model(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "halt_logits": halt_logits,
            "continue_logits": continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if training and (self.config.halt_max_steps > 1):
                halted = halted | (halt_logits > continue_logits)

                min_halt_steps = (torch.rand_like(halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                next_q_halt_logits, next_q_continue_logits = self.model(new_inner_carry, new_current_data)[-1]

                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return ACTControllerCarry(new_inner_carry, new_steps, halted, new_current_data), outputs


__all__ = ["ACTController", "ACTControllerCarry", "ACTControllerConfig"]
