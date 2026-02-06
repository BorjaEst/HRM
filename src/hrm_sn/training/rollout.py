from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Union

import torch
from torch import Tensor

from hrm_sn.loss.act_head import ACTLossHead

BatchDict = Dict[str, Tensor]
BatchProvider = Callable[[int, Any], BatchDict]  # (t, carry) -> batch


@dataclass(frozen=True)
class HRMStep:
    t: int
    carry: Any
    loss_sum: Tensor
    metrics: Dict[str, Tensor]
    outputs: Optional[Dict[str, Tensor]]
    all_finish: bool


class RolloutLoop(Iterator[HRMStep]):
    """
    Repeatedly calls loss_head(carry=..., batch=..., return_keys=...).

    Stop conditions:
    - t reaches horizon (default 1)  => matches current training behavior
    - optionally stop early if all_finish (useful for eval)
    """

    def __init__(
        self,
        loss_head: ACTLossHead,
        carry0: Any,
        batch: Union[BatchDict, BatchProvider],
        *,
        return_keys: Sequence[str] = (),
        horizon: Optional[int] = 1,
        stop_on_all_finish: bool = False,
    ) -> None:
        self.loss_head = loss_head
        self.carry = carry0
        self.batch = batch
        self.return_keys = list(return_keys)
        self.horizon = horizon
        self.stop_on_all_finish = stop_on_all_finish

        self._t = 0
        self._done = False

    def __iter__(self) -> "RolloutLoop":
        return self

    def __next__(self) -> HRMStep:
        if self._done:
            raise StopIteration

        if self.horizon is not None and self._t >= self.horizon:
            self._done = True
            raise StopIteration

        batch_dict = self.batch if isinstance(self.batch, dict) else self.batch(self._t, self.carry)

        new_carry, loss_sum, metrics, outputs, all_finish = self.loss_head(
            carry=self.carry,
            batch=batch_dict,
            return_keys=self.return_keys,
        )
        self.carry = new_carry

        step = HRMStep(
            t=self._t,
            carry=new_carry,
            loss_sum=loss_sum,
            metrics=metrics,
            outputs=outputs,
            all_finish=bool(all_finish),
        )
        self._t += 1

        if self.stop_on_all_finish and step.all_finish:
            self._done = True

        return step


class EvaluationLoop(Iterator[HRMStep]):
    """
    Convenience: initialize carry from batch, then run until all_finish
    (optionally capped by max_steps).
    """

    def __init__(
        self,
        loss_head: ACTLossHead,
        batch: BatchDict,
        *,
        return_keys: Sequence[str] = (),
        max_steps: Optional[int] = None,  # safety cap
    ) -> None:
        carry0 = loss_head.initial_carry(batch)
        self._inner = RolloutLoop(
            loss_head=loss_head,
            carry0=carry0,
            batch=batch,
            return_keys=return_keys,
            horizon=max_steps,  # None => uncapped
            stop_on_all_finish=True,
        )

    def __iter__(self) -> "EvaluationLoop":
        return self

    def __next__(self) -> HRMStep:
        return next(self._inner)
