"""MLP-related modules for HRM models.

This module currently provides the feed-forward block used in HRM-style
Transformer layers.
"""

import torch.nn.functional as F
from torch import nn

from hrm_sn.modules.projections import CastedLinear
from hrm_sn.utils import _find_multiple


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

    def __init__(self, hidden_size: int, expansion: float):
        """Initialize the SwiGLU block.

        Args:
            hidden_size: Input and output hidden dimension.
            expansion: Expansion multiplier used to compute the intermediate width.
                The internal width is derived as ``round(expansion * hidden_size * 2/3)``
                and then aligned to a multiple of 256.
        """
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
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
