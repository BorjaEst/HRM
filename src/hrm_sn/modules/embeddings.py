"""Embedding layers used throughout the model.

This module provides small, explicit wrappers around :func:`torch.nn.functional.embedding`
that mirror common PyTorch conventions (``reset_parameters`` / ``extra_repr``) while
supporting the initialization and dtype/casting patterns used in this repo.

This module provides a single dense embedding implementation:

- :class:`CastedEmbedding`: a standard dense embedding table (learnable ``weight``)
    with explicit dtype casting.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hrm_sn.utils import trunc_normal_init_

# TODO: Move to types.py
DTypeName = Literal["float16", "bfloat16", "float32"]


class CastedEmbedding(nn.Module):
    """Dense embedding table with explicit dtype casting.

    This is analogous to ``torch.nn.Embedding`` but uses this repo's truncated
    LeCun normal initialization and casts ``weight`` to a configured compute
    dtype on each forward pass.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_std: float = 0.02,
        dtype: DTypeName = "bfloat16",
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.init_std = init_std
        self._dtype = dtype

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype used for lookup results."""
        return getattr(torch, self._dtype)

    def reset_parameters(self) -> None:
        """Initialize parameters.

        Mirrors the common PyTorch pattern of factoring initialization into a
        dedicated method.
        """
        trunc_normal_init_(self.weight, std=self.init_std)

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"dtype={self.dtype}"
        ) # fmt: skip

    def forward(self, input: Tensor) -> Tensor:
        """Lookup embeddings.

        Args:
            input: Integer indices of arbitrary shape.

        Returns:
            Embedded vectors of shape ``(*input.shape, embedding_dim)``.
        """
        return F.embedding(input, self.weight.to(self.dtype))
