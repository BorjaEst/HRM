"""Projection layers used throughout the model.

The layers in this module are small wrappers around common PyTorch operations
that standardize initialization and dtype handling.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from hrm_sn.utils import trunc_normal_init_


class CastedLinear(nn.Module):
    """A linear projection that casts parameters to the input dtype at runtime.

    This layer stores parameters as ``nn.Parameter`` tensors (typically created in
    PyTorch's default floating dtype), but on every forward pass it casts
    ``weight`` (and ``bias`` when present) to ``input.dtype`` before calling
    :func:`torch.nn.functional.linear`.

    This is useful in mixed-precision training/inference to avoid manual casts at
    call sites while still keeping initialization and parameter storage simple.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
        param_dtype: torch.dtype | None = None,
    ):
        """Create a casted linear layer.

        Args:
            in_features: Size of each input sample (last dimension of the input).
            out_features: Size of each output sample (last dimension of the output).
            bias: If ``True``, include a bias term (initialized to zeros).

        Notes:
            - Weights are initialized with a truncated LeCun normal distribution.
            - Bias, when enabled, is initialized to zeros.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.param_dtype = torch.get_default_dtype() if param_dtype is None else param_dtype

        factory_kwargs = {"device": device, "dtype": self.param_dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters.

        Mirrors the common PyTorch pattern of factoring initialization into a
        dedicated method, while using this repo's truncated LeCun normal.
        """
        trunc_normal_init_(self.weight, std=1.0 / math.sqrt(self.in_features))
        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, param_dtype={self.param_dtype}"
        )  # fmt: skip

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the linear projection.

        Args:
            input: Tensor of shape ``(..., in_features)``.

        Returns:
            Tensor of shape ``(..., out_features)``.

        Notes:
            - ``weight`` and ``bias`` are cast to ``input.dtype`` each call.
            - Device placement follows the parameter tensors; the input must be on
              a compatible device.
        """
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )
