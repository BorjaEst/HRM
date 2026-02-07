"""Embedding layers used throughout the model.

This module provides small, explicit wrappers around :func:`torch.nn.functional.embedding`
that mirror common PyTorch conventions (``reset_parameters`` / ``extra_repr``) while
supporting the initialization and dtype/casting patterns used in this repo.

Two embedding implementations exist:

- :class:`CastedEmbedding`: a standard dense embedding table (learnable ``weight``).
- :class:`SparseUpdateEmbedding`: an HRM-specific embedding that trains via a *local
    slice* (dense gradients on a per-batch buffer) and a custom optimizer that writes
    updates back to a global table.

The sparse-update variant is intentionally not a drop-in replacement for
``torch.nn.Embedding(sparse=True)``; the "sparse" aspect refers to the *update
mechanism*, not sparse gradients produced by autograd.
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


class SparseUpdateEmbeddingConfig(EmbeddingsConfig):
    """Configuration for :class:`SparseUpdateEmbedding`.

    The sparse-update embedding keeps a global table but trains via per-batch
    *local buffers*. ``local_batch_size`` defines the size of those buffers.
    """

    local_batch_size: int = Field(
        ...,
        ge=1,
        description="Batch size for local embedding buffers",
    )


class SparseUpdateEmbedding(nn.Module):
    """Embedding table trained via a local-slice + custom optimizer.

    During training, this module:

    1) Copies the rows indexed by ``inputs`` from a persistent global table
       (``weight`` buffer) into ``local_weight``.
    2) Returns ``local_weight`` so gradients flow into that local buffer.
    3) A custom optimizer (SignSGD-style) uses ``local_indices`` and
       ``local_weight.grad`` to update the global table.

    This approach avoids dense gradients on the full table while keeping the
    forward path simple.
    """

    def __init__(self, config: SparseUpdateEmbeddingConfig):
        super().__init__()
        self._config = config

        # Global table: persistent storage updated by a custom optimizer.
        self.weight = nn.Buffer(torch.empty(config.num_embeddings, config.embedding_dim))
        self.local_weight = nn.Buffer(
            torch.zeros(config.local_batch_size, config.embedding_dim, requires_grad=True),
            persistent=False,
        )
        self.local_indices = nn.Buffer(
            torch.zeros(config.local_batch_size, dtype=torch.int32),
            persistent=False,
        )
        self.reset_parameters()

    @property
    def config(self) -> SparseUpdateEmbeddingConfig:
        return self._config

    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.config.dtype)

    def reset_parameters(self) -> None:
        """Initialize the global table."""
        trunc_normal_init_(self.weight, std=self.config.init_std)

    def extra_repr(self) -> str:
        return (
            f"{self.config.num_embeddings}, {self.config.embedding_dim}, sparse_update=hrm,"
            f"local_batch_size={self.config.local_batch_size}, dtype={self.config.dtype}"
        )  # fmt: skip

    def forward(self, inputs: Tensor) -> Tensor:
        """Lookup embeddings.

        Args:
            inputs: Tensor of shape ``(local_batch_size,)`` containing integer
                identifiers.

        Returns:
            Tensor of shape ``(local_batch_size, embedding_dim)``.
        """
        if not self.training:
            return self.weight[inputs].to(self.dtype)

        with torch.no_grad():
            # Snapshot the selected rows into the local buffer and remember ids
            # so the optimizer can apply sparse updates back to the global table.
            self.local_weight.copy_(self.weight[inputs])
            self.local_indices.copy_(inputs)

        return self.local_weight.to(self.dtype)
