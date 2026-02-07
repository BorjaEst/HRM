from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

from hrm_sn.utils import trunc_normal_init_

# TODO: Move to types.py
DTypeName = Literal["float16", "bfloat16", "float32"]


class EmbeddingsConfig(BaseModel):
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
        return getattr(torch, self.config.dtype)

    def reset_parameters(self) -> None:
        trunc_normal_init_(self.weight, std=self.config.init_std)

    def extra_repr(self) -> str:
        return (
            f"{self.config.num_embeddings}, {self.config.embedding_dim},"
            f"dtype={self.config.dtype}"
        ) # fmt: skip

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.weight.to(self.config.dtype))


class SparseUpdateEmbeddingConfig(EmbeddingsConfig):
    local_batch_size: int = Field(
        ...,
        ge=1,
        description="Batch size for local embedding buffers",
    )


class SparseUpdateEmbedding(nn.Module):
    def __init__(self, config: SparseUpdateEmbeddingConfig):
        super().__init__()
        self._config = config

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
        trunc_normal_init_(self.weight, std=self.config.init_std)

    def extra_repr(self) -> str:
        return (
            f"{self.config.num_embeddings}, {self.config.embedding_dim}, sparse_update=hrm,"
            f"local_batch_size={self.config.local_batch_size}, dtype={self.config.dtype}"
        )  # fmt: skip

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training:
            return self.weight[inputs].to(self.dtype)

        with torch.no_grad():
            self.local_weight.copy_(self.weight[inputs])
            self.local_indices.copy_(inputs)

        return self.local_weight.to(self.dtype)
