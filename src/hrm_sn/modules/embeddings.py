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
from pydantic import BaseModel, Field
from torch import Tensor, nn

from hrm_sn.utils import trunc_normal_init_

# TODO: Move to types.py
DTypeName = Literal["float16", "bfloat16", "float32"]


class EmbeddingsConfig(BaseModel):
    """Configuration for embedding layers.

    This config intentionally mirrors the "core" subset of ``torch.nn.Embedding``
    constructor parameters that HRM actually needs.

    Notes:
            - ``dtype`` is represented as a string literal so it can be loaded from
                TOML/JSON config sources. Modules convert it to a ``torch.dtype``.
            - Padding behavior (``padding_idx``) is intentionally not implemented
                yet to keep the implementation minimal.
    """

    num_embeddings: int = Field(
        ...,
        ge=1,
        description="Number of embeddings",
    )
    embedding_dim: int = Field(
        ...,
        ge=1,
        description="Dimension of each embedding",
    )
    init_std: float = Field(
        default=0.02,
        ge=0.0,
        description="Standard deviation for truncated normal initialization",
    )
    dtype: DTypeName = Field(
        default="bfloat16",
        description="Data type for embeddings",
    )


class CastedEmbedding(nn.Module):
    """Dense embedding table with explicit dtype casting.

    This is analogous to ``torch.nn.Embedding`` but uses this repo's truncated
    LeCun normal initialization and casts ``weight`` to a configured compute
    dtype on each forward pass.
    """

    def __init__(self, config: EmbeddingsConfig):
        super().__init__()
        self._config = config

        self.weight = nn.Parameter(torch.empty(config.num_embeddings, config.embedding_dim))
        self.reset_parameters()

    @property
    def config(self) -> EmbeddingsConfig:
        return self._config

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype used for lookup results."""
        return getattr(torch, self.config.dtype)

    def reset_parameters(self) -> None:
        """Initialize parameters.

        Mirrors the common PyTorch pattern of factoring initialization into a
        dedicated method.
        """
        trunc_normal_init_(self.weight, std=self.config.init_std)

    def extra_repr(self) -> str:
        return (
            f"{self.config.num_embeddings}, {self.config.embedding_dim}, "
            f"dtype={self.config.dtype}"
        ) # fmt: skip

    def forward(self, input: Tensor) -> Tensor:
        """Lookup embeddings.

        Args:
            input: Integer indices of arbitrary shape.

        Returns:
            Embedded vectors of shape ``(*input.shape, embedding_dim)``.
        """
        return F.embedding(input, self.weight.to(self.dtype))
