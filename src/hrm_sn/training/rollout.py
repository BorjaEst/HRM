from __future__ import annotations

from itertools import tee
from typing import Any, Dict, Iterator, Optional

from torch import Tensor

from hrm_sn.loss.act_head import ACTLossHead, ACTStepOutput

Batch = Dict[str, Tensor]  # Generic batch type, can be specialized as needed
StepBatchSource = Iterator[Batch]


# =================================================================================================
class RolloutLoop(Iterator[ACTStepOutput]):
    """
    Repeatedly calls loss_head(carry=..., batch=..., return_keys=...).

    Batch inputs are provided by a step-batch source (iterator) and may optionally
    accept carry updates via an ``update(carry=...)`` method.

    Stop conditions:
    - t reaches horizon (default 1)  => matches current training behavior
    - optionally stop early if all_finish (useful for eval)
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, loss_head: ACTLossHead, batch: StepBatchSource,
        *,
        carry0: Optional[Any]=None, max_steps: Optional[int] = None, stop_on_all_finish: bool = True,
    ) -> None:  # fmt: skip
        self.loss_head = loss_head
        batch_iter = enumerate(batch)  # Add time step enumeration to the batch source

        # Use tee to peek without consuming
        peek_iter, self.batch_iter = tee(batch_iter, 2)
        _, first_batch = next(peek_iter)  # Peek first batch

        # Initialize state and previous action
        self.carry = carry0 or loss_head.initial_carry(first_batch)
        self.max_steps = max_steps if max_steps is not None else float("inf")
        self.stop_on_all_finish = stop_on_all_finish
        self._done = False

    def __iter__(self) -> RolloutLoop:
        return self

    def __next__(self) -> ACTStepOutput:
        if self._done:  # Check if we've already stopped due to all_finished
            raise StopIteration

        # t_rollout is 0-indexed, so the first batch corresponds to t=0
        t_rollout, batch = next(self.batch_iter)  # enumerated, returns t in first position
        output: ACTStepOutput = self.loss_head(batch, self.carry, t=t_rollout)
        self.carry = output.carry

        # Check stop conditions after updating carry/state
        if self.stop_on_all_finish and output.all_finished:
            self._done = True
        if t_rollout >= self.max_steps - 1:  # t_rollout is 0-indexed
            self._done = True

        return output


class EvaluationLoop(Iterator[ACTStepOutput]):
    """
    Convenience: initialize carry from batch, then run until all_finish.
    TODO: This is currently unused since eval also needs trace collection
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, loss_head: ACTLossHead, batch: StepBatchSource,
        *,
        carry0: Optional[Any]=None, max_steps: Optional[int] = None, stop_on_all_finish: bool = True,
    ) -> None:  # fmt: skip
        self.loss_head = loss_head
        batch_iter = enumerate(batch)  # Add time step enumeration to the batch source

        # Use tee to peek without consuming
        peek_iter, self.batch_iter = tee(batch_iter, 2)
        _, first_batch = next(peek_iter)  # Peek first batch

        # Initialize state and previous action
        self.carry = carry0 or loss_head.initial_carry(first_batch)
        self.max_steps = max_steps if max_steps is not None else float("inf")
        self.stop_on_all_finish = stop_on_all_finish
        self._done = False

    def __iter__(self) -> EvaluationLoop:
        return self

    def __next__(self) -> ACTStepOutput:
        if self._done:  # Check if we've already stopped due to all_finished
            raise StopIteration

        # t_rollout is 0-indexed, so the first batch corresponds to t=0
        t_rollout, batch = next(self.batch_iter)  # enumerated, returns t in first position
        output: ACTStepOutput = self.loss_head(batch, self.carry, t=t_rollout)
        self.carry = output.carry

        # Check stop conditions after updating carry/state
        if self.stop_on_all_finish and output.all_finished:
            self._done = True
        if t_rollout >= self.max_steps - 1:  # t_rollout is 0-indexed
            self._done = True

        return output
