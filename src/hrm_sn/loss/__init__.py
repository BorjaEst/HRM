"""Loss functions, metric utilities, wrapper"""

from pydantic import BaseModel, ConfigDict

from hrm_sn.loss.act_head import ACTLossHead
from hrm_sn.loss.cross_entropy import softmax_cross_entropy, stablemax_cross_entropy

__all__ = [
    "ACTLossHead",
    "stablemax_cross_entropy",
    "softmax_cross_entropy",
    "LossConfig",
]


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    loss_type: str
