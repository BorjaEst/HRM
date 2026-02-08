"""Attention building block.

This module implements a standard multi-head attention (MHA) layer with optional:

- GQA-style head sharing (a.k.a. grouped-query attention) via `num_kv_heads`.
- PyTorch SDPA (scaled dot-product attention) backend.

The implementation is intentionally small and shape-explicit since it is a core
primitive used throughout the model.
"""

from typing import Optional

import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn

from hrm_sn.modules.projections import CastedLinear


class AttentionConfig(BaseModel, extra="forbid"):
    """Configuration for the :class:`Attention` module.

    Notes:
        - `embedding_dim` must be divisible by `num_heads`.
        - `num_kv_heads` enables grouped-query attention when it is smaller than
          `num_heads` (keys/values are shared across multiple query heads).
    """

    embedding_dim: int = Field(
        ...,
        ge=32,
        frozen=True,
        description="Hidden size of the attention module.",
    )
    num_heads: int = Field(
        ...,
        ge=1,
        frozen=True,
        description="Number of attention heads.",
    )

    @field_validator("embedding_dim", mode="before")
    def _check_embed_dim(cls, v, values):
        num_heads = values.get("num_heads")
        if num_heads is not None and v % num_heads != 0:
            raise ValueError(f"embedding_dim ({v}) must be divisible by num_heads ({num_heads}).")
        return v

    num_kv_heads: Optional[int] = Field(
        default=None,
        frozen=True,
        description="Number of key/value heads for grouped-query attention. If None, defaults to `num_heads` (no sharing).",
    )

    @field_validator("num_kv_heads", mode="before")
    def _set_num_kv_heads(cls, v, values):
        return v if v is not None else values.get("num_heads")

    is_causal: bool = Field(
        default=False,
        description="Whether to apply causal masking in attention.",
    )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.embedding_dim // self.num_heads

    @property
    def output_size(self) -> int:
        """Output dimension of the attention module."""
        return self.head_dim * self.num_heads


class Attention(nn.Module):
    """Multi-head attention with grouped-query attention support.

    Inputs and outputs use the common Transformer layout `[batch, seq, embed]`.
    Internally, tensors are reshaped to per-head form and fed through
    `torch.nn.functional.scaled_dot_product_attention`.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self._config = config

        self._qkv_head_count = config.num_heads + 2 * config.num_kv_heads  # type: ignore[assignment]
        self._qkv_proj_out_dim = self._qkv_head_count * config.head_dim
        self.in_proj = CastedLinear(config.embedding_dim, self._qkv_proj_out_dim, bias=False)
        self.out_proj = CastedLinear(config.output_size, config.embedding_dim, bias=False)

    @property
    def config(self) -> AttentionConfig:
        return self._config

    def _compute_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project and split inputs into q, k, and v tensors.

        Args:
            x: Hidden states of shape `[batch, seq_len, embedding_dim]`.

        Returns:
            Tuple `(q, k, v)` with shapes:

            - `q`: `[batch, seq_len, num_heads, head_dim]`
            - `k`: `[batch, seq_len, num_kv_heads, head_dim]`
            - `v`: `[batch, seq_len, num_kv_heads, head_dim]`
        """
        config, qkv_head_count = self.config, self._qkv_head_count
        batch_size, seq_len, _ = x.shape

        # Project once, then slice into (q, k, v) head groups.
        qkv = self.in_proj(x)

        # Reshape to per-head representation.
        qkv = qkv.view(batch_size, seq_len, qkv_head_count, config.head_dim)
        q = qkv[:, :, : config.num_heads]
        k = qkv[:, :, config.num_heads : config.num_heads + config.num_kv_heads]  # type: ignore[assignment]
        v = qkv[:, :, config.num_heads + config.num_kv_heads :]  # type: ignore[assignment]
        return q, k, v

    def _expand_kv_heads(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Repeat k/v heads for grouped-query attention when needed.

        SDPA expects matching head counts for Q/K/V. When `num_kv_heads < num_heads`,
        we repeat keys/values across query heads.
        """
        config = self.config
        if config.num_kv_heads == config.num_heads:
            return k, v
        repeat = config.num_heads // config.num_kv_heads  # type: ignore[assignment]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        return k, v

    def forward(self, x: Tensor, *, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Compute attention outputs for a batch of sequences.

        Args:
            x: Hidden states of shape `[batch, seq_len, embedding_dim]`.
            attn_mask: Optional attention mask passed through to PyTorch SDPA.
                Shape and dtype semantics follow `scaled_dot_product_attention`.

        Returns:
            Attention outputs of shape `[batch, seq_len, embedding_dim]`.
        """
        config = self.config
        batch_size, seq_len, _ = x.shape
        q, k, v = self._compute_qkv(x)

        # SDPA expects `[batch, heads, seq, head_dim]`.
        q = q.transpose(1, 2)  # [bs, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k, v = self._expand_kv_heads(k, v)

        # `is_causal` is handled by SDPA; `attn_mask` is optional and may encode
        # padding, block masks, etc.
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask, is_causal=config.is_causal)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, config.output_size)
        return self.out_proj(attn)
