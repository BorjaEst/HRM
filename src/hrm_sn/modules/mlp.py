"""MLP-related modules for HRM models.

This module currently provides the feed-forward block used in HRM-style
Transformer layers.
"""

from typing import Optional

import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

from hrm_sn.types import Device, Dtype
from hrm_sn.utils import _find_multiple


# =================================================================================================
class MLPConfig(BaseModel, extra="forbid"):
    """Configuration for the MLP (feed-forward) block used in HRM transformer layers."""

    hidden_size: int = Field(
        ...,
        ge=32,
        frozen=True,
        description="Hidden size of the MLP block.",
    )
    expansion: float = Field(
        default=4.0,
        gt=1.0,
        description="Expansion factor for the MLP layers in the transformer blocks.",
    )


# =================================================================================================
class SwiGLU(nn.Module):
    """SwiGLU feed-forward (MLP) block with a gated activation.

    This implements the common SwiGLU pattern:

    - Project from ``hidden_size`` to an intermediate width and split into
      ``gate`` and ``up`` parts.
    - Apply ``silu`` to the gate and multiply elementwise with the up branch.
    - Project back to ``hidden_size``.

    The intermediate width is computed from ``expansion`` and rounded/aligned
    to a multiple of 256 for efficiency.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: MLPConfig, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize the SwiGLU block.

        Args:
            hidden_size: Input and output hidden dimension.
            expansion: Expansion multiplier used to compute the intermediate width.
                The internal width is derived as ``round(expansion * hidden_size * 2/3)``
                and then aligned to a multiple of 256.
        """
        super().__init__()
        self._config = config

        inter = _find_multiple(round(config.expansion * config.hidden_size * 2 / 3), 256)
        self.gate_up_proj = nn.Linear(config.hidden_size, inter * 2, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False, device=device, dtype=dtype)

    @property
    def config(self) -> MLPConfig:
        """Configuration of the SwiGLU block."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor,
    ) -> Tensor:  # fmt: skip
        """Apply the SwiGLU transformation.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Tensor with the same shape and dtype as the input.

        Notes:
            - This module is shape-preserving in the last dimension.
            - The underlying projections are performed by
              :class:`~hrm_sn.modules.projections.CastedLinear`.
        """
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
