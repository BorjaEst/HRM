from __future__ import annotations

"""RL-style controller for ACT halting with explicit TD bootstrapping.

This controller mirrors DQN-style structure:
- state: recurrent model state + per-slot step counters
- action: halt or continue (argmax over Q logits)
- done: episode termination (max steps or halt action)
- TD target: bootstrap from Q at next state

Note: there is no explicit per-step reward. Correctness supervision is
provided by the loss head, while the continue Q target is bootstrapped
from the next recurrent state.
"""

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor


class ACTControllerConfig(BaseModel, extra="forbid"):
    exploration_prob: float = Field(..., ge=0.0, le=1.0, description="Exploration probability for ACT halting.")
    halt_max_steps: int = Field(..., ge=1, description="Maximum number of ACT steps.")


class ACTNetwork(Protocol):
    """Minimal interface required by the ACT controller.

    Matches HRModel.forward(inputs, state=None).
    """

    training: bool

    def init_state(self, batch_size: int) -> Any: ...

    def reset_state(self, reset_flag: Tensor, state: Any) -> Any: ...

    def __call__(self, inputs: Tensor, state: Any | None = None) -> Tuple[Any, Tensor, Tuple[Tensor, Tensor]]: ...


@dataclass
class ACTState:
    model_state: Any  # Recurrent state of the model (e.g. LSTM hidden states)
    steps: Tensor  # Per-slot step counters
    halted: Tensor  # Per-slot done/halting flags
    data: Dict[str, Tensor]  # Per-slot data buffers (e.g. inputs) for refreshing on reset


@dataclass
class ACTOutput:
    logits: Tensor  # Main output logits for loss/metrics (e.g. action logits)
    halt_logits: Tensor  # Q logits for halting action
    continue_logits: Tensor  # Q logits for continuing action
    action: Tensor  # Selected action (0=halt, 1=continue)
    target_continue: Tensor | None = None  # TD target for continue head (training only)


class ACTController:
    """Training-time ACT controller for halting and partial-reset batch slots."""

    HALT_ACTION = 0
    CONTINUE_ACTION = 1

    def __init__(self, model: ACTNetwork, config: ACTControllerConfig) -> None:
        self._model = model
        self._config = config

    @property
    def model(self) -> ACTNetwork:
        return self._model

    @property
    def config(self) -> ACTControllerConfig:
        return self._config

    def initial_state(self, batch: Dict[str, Tensor]) -> ACTState:
        batch_size = batch["inputs"].shape[0]
        return ACTState(
            model_state=self.model.init_state(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def step(self, state: ACTState, batch: Dict[str, Tensor], *, training: bool) -> Tuple[ACTState, ACTOutput]:
        """Execute one ACT step and return new state plus outputs for loss/metrics."""
        config = self.config  # convenience alias

        data = self.refresh_slot_data(batch, state)
        model_state = self.reset_where_done(state)
        model_state, logits, (q_halt, q_continue) = self.model(data["inputs"], model_state)

        steps = torch.where(state.halted, 0, state.steps) + 1
        action, done, is_last_step = self._select_action_and_done(q_halt, q_continue, steps, training)

        state = ACTState(model_state=model_state, steps=steps, halted=done, data=data)
        output = ACTOutput(logits=logits, halt_logits=q_halt, continue_logits=q_continue, action=action)

        with torch.no_grad():
            if training and (config.halt_max_steps > 1):
                output.target_continue = self.td_target_continue(data, state, is_last_step)

        return state, output

    def refresh_slot_data(self, batch: Dict[str, Tensor], state: ACTState) -> Dict[str, Tensor]:
        """Replace finished slots with new batch data (episode reset).

        Args:
            batch: New batch data.
            state: Current ACT state with done/halted flags.

        Returns:
            A dict with per-key tensors updated for done slots.
        """
        data, halted = state.data, state.halted
        return {k: torch.where(halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], data[k]) for k in batch}

    def reset_where_done(self, state: ACTState) -> Any:
        """Reset recurrent state for done slots (episode reset).

        Args:
            state: Current ACT state with done/halted flags.

        Returns:
            New recurrent state with done slots reset.
        """
        model_state, halted = state.model_state, state.halted
        return self.model.reset_state(halted, model_state)

    def _select_action_and_done(self, q_halt: Tensor, q_continue: Tensor, steps: Tensor, training: bool) -> Tuple[Tensor, Tensor, Tensor]:
        """Select halt/continue action and compute done mask.

        Action is greedy (argmax over Q logits). Done is triggered by:
        - reaching max steps
        - halting action (training only)
        Optional exploration delays halting by enforcing a random
        minimum halting step for a subset of slots.
        """
        config = self.config  # convenience alias
        action = torch.where(q_halt > q_continue, self.HALT_ACTION, self.CONTINUE_ACTION)
        done = is_last_step = steps >= config.halt_max_steps

        if training and (config.halt_max_steps > 1):
            exploration_flag = torch.rand_like(q_halt) < config.exploration_prob
            min_halt_steps = exploration_flag * torch.randint_like(steps, low=2, high=config.halt_max_steps + 1)

            done = done | (action == self.HALT_ACTION)  # halt action triggers done
            done = done & (steps >= min_halt_steps)  # Enforce min steps under exploration

        return action, done, is_last_step

    def td_target_continue(self, batch: Dict[str, Tensor], state: ACTState, is_last_step: Tensor) -> Tensor:
        """Compute TD(0) target for the continue head.

        Target: sigmoid(max_a Q(s_{t+1}, a)), with terminal handling.
        """
        inputs, model_state = batch["inputs"], state.model_state

        next_q_halt, next_q_continue = self.model(inputs, model_state)[-1]
        next_q_best = torch.maximum(next_q_halt, next_q_continue)

        logits = torch.where(is_last_step, next_q_halt, next_q_best)
        return torch.sigmoid(logits)


__all__ = ["ACTControllerConfig", "ACTNetwork", "ACTController", "ACTState"]
