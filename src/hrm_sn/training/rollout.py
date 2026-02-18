from __future__ import annotations

from dataclasses import dataclass
from itertools import tee
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from torch import Tensor

from hrm_sn.loss.act_head import ACTLossHead, ACTStepOutput

Batch = Dict[str, Tensor]  # Generic batch type, can be specialized as needed
StepBatchSource = Iterator[Batch]


# =================================================================================================
@dataclass(frozen=True)
class StepContext:
    """ """

    batch: Mapping[str, Tensor]  # current step batch
    carry: Any  # controller/model carry after step
    outputs: Any  # controller outputs after step


# =================================================================================================
class RolloutLoop(Iterator[Tuple[int, StepContext]]):
    """ """

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

    def __iter__(  # ------------------------------------------------------------------------------
        self,
    ) -> RolloutLoop:  # fmt: skip
        return self

    def __next__(  # ------------------------------------------------------------------------------
        self,
    ) -> Tuple[int, StepContext]:  # fmt: skip
        if self._done:  # Check if we've already stopped due to all_finished
            raise StopIteration

        # t_rollout is 0-indexed, so the first batch corresponds to t=0
        t_rollout, batch = next(self.batch_iter)  # enumerated, returns t in first position
        options = {"allow_halt": True, "explore": True, "compute_targets": True}
        outputs, carry, done = self.loss_head(batch, self.carry, **options)
        self.carry = carry

        # Check stop conditions after updating carry/state
        if self.stop_on_all_finish and done:
            self._done = True
        if t_rollout >= self.max_steps - 1:  # t_rollout is 0-indexed
            self._done = True

        context = StepContext(batch=batch, carry=carry, outputs=outputs)
        return t_rollout, context


# =================================================================================================
class EvaluationLoop(Iterator[Tuple[int, StepContext]]):
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

    def __iter__(  # ------------------------------------------------------------------------------
        self,
    ) -> EvaluationLoop:  # fmt: skip
        return self

    def __next__(  # ------------------------------------------------------------------------------
        self,
    ) -> Tuple[int, StepContext]:  # fmt: skip
        if self._done:  # Check if we've already stopped due to all_finished
            raise StopIteration

        # t_rollout is 0-indexed, so the first batch corresponds to t=0
        t_rollout, batch = next(self.batch_iter)  # enumerated, returns t in first position
        options = {"allow_halt": False, "explore": False, "compute_targets": False}
        outputs, carry, done = self.loss_head(batch, self.carry, **options)
        self.carry = carry

        # Check stop conditions after updating carry/state
        if self.stop_on_all_finish and done:
            self._done = True
        if t_rollout >= self.max_steps - 1:  # t_rollout is 0-indexed
            self._done = True

        context = StepContext(batch=batch, carry=carry, outputs=outputs)
        return t_rollout, context
