from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class SchedulerConfig(BaseModel, extra="forbid"):
    warmup_steps: int = Field(
        default=0,
        description="Number of warmup steps for learning rate scheduling.",
    )
    min_ratio: float = Field(
        default=0.0,
        description="Minimum learning rate ratio for cosine decay. The learning rate will decay to `base_lr * min_ratio` at the end of training.",
    )


__all__ = ["SchedulerConfig", "SequentialLR", "CosineAnnealingLR", "LinearLR"]
