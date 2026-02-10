from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from torch import Tensor


@dataclass(frozen=True)
class Metric:
    """ """

    # Metrics for ACT training
    count: int | Tensor
    steps: int | Tensor

    # Acc
    accuracy: float | Tensor
    exact_accuracy: int | Tensor
    q_halt_accuracy: int | Tensor
    q_continue_accuracy: int | Tensor

    # Losses
    loss_current: float | Tensor
    q_halt_loss: float | Tensor
    q_continue_loss: float | Tensor
