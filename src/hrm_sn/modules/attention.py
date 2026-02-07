from typing import Optional

import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn

from hrm_sn.modules.projections import CastedLinear
from hrm_sn.modules.rotary import apply_rotary_pos_emb
from hrm_sn.types import CosSin


class AttentionConfig(BaseModel, extra="forbid"):

    embed_dim: int = Field(
        ...,
        frozen=True,
        description="Hidden size of the attention module.",
    )
    num_heads: int = Field(
        ...,
        frozen=True,
        description="Number of attention heads.",
    )

    num_kv_heads: int = Field(
        ...,
        frozen=True,
        description="Number of k/v heads. If different from num_heads, keys and values are shared across heads.",
    )
    is_causal: bool = Field(
        default=False,
        description="Whether to apply causal masking in attention.",
    )

    @field_validator("embed_dim")
    def _check_embed_dim(cls, v, values):
        num_heads = values.get("num_heads")
        if num_heads is not None and v % num_heads != 0:
            raise ValueError(f"embed_dim ({v}) must be divisible by num_heads ({num_heads}).")
        return v

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.embed_dim // self.num_heads

    @property
    def output_size(self) -> int:
        """Output dimension of the attention module."""
        return self.head_dim * self.num_heads


class Attention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self._config = config

        self._qkv_head_count = config.num_heads + 2 * config.num_kv_heads
        self._qkv_proj_out_dim = self._qkv_head_count * config.head_dim
        self.in_proj = CastedLinear(config.embed_dim, self._qkv_proj_out_dim, bias=False)
        self.out_proj = CastedLinear(config.output_size, config.embed_dim, bias=False)

    @property
    def config(self) -> AttentionConfig:
        return self._config

    def _compute_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project and split inputs into q, k, and v tensors."""
        config, qkv_head_count = self.config, self._qkv_head_count
        batch_size, seq_len, _ = x.shape

        # hidden_states: [bs, seq_len, embed_dim]
        qkv = self.in_proj(x)

        # Split head
        qkv = qkv.view(batch_size, seq_len, qkv_head_count, config.head_dim)
        q = qkv[:, :, : config.num_heads]
        k = qkv[:, :, config.num_heads : config.num_heads + config.num_kv_heads]
        v = qkv[:, :, config.num_heads + config.num_kv_heads :]
        return q, k, v

    def _apply_rope(self, q: Tensor, k: Tensor, cos_sin: Optional[CosSin]) -> tuple[Tensor, Tensor]:
        """Apply rotary positional embeddings when provided."""
        if cos_sin is None:
            return q, k
        seq_len = q.shape[1]
        cos, sin = cos_sin[0][:seq_len], cos_sin[1][:seq_len]
        return apply_rotary_pos_emb(q, k, cos, sin)

    def _expand_kv_heads(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Repeat k/v heads for GQA when needed."""
        config = self.config
        if config.num_kv_heads == config.num_heads:
            return k, v
        repeat = config.num_heads // config.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        return k, v

    def forward(self, x: Tensor, *, cos_sin: Optional[CosSin] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Compute attention outputs for a batch of sequences."""
        config = self.config
        batch_size, seq_len, _ = x.shape
        q, k, v = self._compute_qkv(x)

        # RoPE
        q, k = self._apply_rope(q, k, cos_sin)

        # Vanilla attention via PyTorch SDPA
        q = q.transpose(1, 2)  # [bs, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k, v = self._expand_kv_heads(k, v)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask, is_causal=config.is_causal)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, config.output_size)
        return self.out_proj(attn)
