"""Embedding layers used throughout the model.

This module provides small, explicit wrappers around :func:`torch.nn.functional.embedding`
that mirror common PyTorch conventions (``reset_parameters`` / ``extra_repr``) while
supporting the initialization and dtype/casting patterns used in this repo.

This module provides a single dense embedding implementation:

- :class:`LearnedPosEmbedding`: a standard dense embedding table (learnable ``weight``)
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


class EmbeddingConfig(BaseModel, extra="forbid"):

    num_embeddings: int = Field(
        ...,
        ge=1,
        description="Number of embeddings (vocab size).",
    )
    embedding_dim: int = Field(
        ...,
        ge=1,
        description="Dimensionality of each embedding vector.",
    )
    init_std: float = Field(
        default=0.02,
        ge=0.0,
        description="Standard deviation of the truncated normal distribution used for weight initialization.",
    )


class Embedding(nn.Embedding):
    """Embedding layer with truncated normal initialization and explicit dtype casting.

    This is a thin wrapper around ``torch.nn.Embedding`` that uses this repo's
    truncated normal initialization and casts the embedding weights to a
    configured compute dtype on each forward pass.
    """

    def __init__(self, config: EmbeddingConfig, device=None, dtype=None):
        super().__init__(config.num_embeddings, config.embedding_dim, device=device, dtype=dtype)
        self._config = config
        self.reset_parameters()

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    def reset_parameters(self) -> None:
        """Initialize parameters.

        Mirrors the common PyTorch pattern of factoring initialization into a
        dedicated method.
        """
        trunc_normal_init_(self.weight, std=self.config.init_std)

    def extra_repr(self) -> str:
        return (
            f"{self.config.num_embeddings}, {self.config.embedding_dim}, "
            f"dtype={self.dtype}"
        ) # fmt: skip

    def forward(self, input: Tensor) -> Tensor:
        """Lookup embeddings.

        Args:
            input: Integer indices of arbitrary shape.

        Returns:
            Embedded vectors of shape ``(*input.shape, embedding_dim)``.
        """
        return F.embedding(input, self.weight.to(self.weight.dtype))


class LearnedPosEmbeddingConfig(EmbeddingConfig):
    pass


class LearnedPosEmbedding(Embedding):
    """Dense embedding table with explicit dtype casting.

    This is analogous to ``torch.nn.Embedding`` but uses this repo's truncated
    LeCun normal initialization and casts ``weight`` to a configured compute
    dtype on each forward pass.
    """

    def __init__(self, config: LearnedPosEmbeddingConfig, device=None, dtype=None):
        super().__init__(config, device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        # # scale by 1/sqrt(2) to maintain forward variance
        # input = 0.707106781 * (input + self.weight)
        return super().forward(input)


__all__ = ["LearnedPosEmbedding", "LearnedPosEmbeddingConfig", "Embedding", "EmbeddingConfig"]
