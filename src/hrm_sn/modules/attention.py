from typing import Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

from hrm_sn.modules.projections import CastedLinear
from hrm_sn.modules.rotary import apply_rotary_pos_emb
from hrm_sn.types import CosSin


class AttentionConfig(BaseModel, extra="forbid"):
    hidden_size: int = Field(
        ...,
        frozen=True,
        description="Hidden size of the attention module.",
    )

    head_dim: int = Field(
        ...,
        frozen=True,
        description="Dimension of each attention head.",
    )
    num_heads: int = Field(
        ...,
        frozen=True,
        description="Number of attention heads.",
    )

    num_key_value_heads: int = Field(
        ...,
        frozen=True,
        description="Number of key/value heads. If different from num_heads, keys and values are shared across heads.",
    )
    causal: bool = Field(
        default=False,
        description="Whether to apply causal masking in attention.",
    )


class Attention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self._config = config

        self._qkv_out = (config.num_heads + 2 * config.num_key_value_heads) * config.head_dim
        self.qkv_proj = CastedLinear(config.hidden_size, self._qkv_out, bias=False)
        self.o_proj = CastedLinear(self.output_size, config.hidden_size, bias=False)

    @property
    def config(self) -> AttentionConfig:
        return self._config

    @property
    def output_size(self) -> int:
        return self.config.head_dim * self.config.num_heads

    def _project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project and split inputs into query, key, and value tensors."""
        config, qkv_out = self.config, self._qkv_out
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, hidden_size]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, qkv_out, config.head_dim)
        query = qkv[:, :, : config.num_heads]
        key = qkv[:, :, config.num_heads : config.num_heads + config.num_key_value_heads]
        value = qkv[:, :, config.num_heads + config.num_key_value_heads :]
        return query, key, value

    def _apply_rope(self, query: Tensor, key: Tensor, cos_sin: Optional[CosSin]) -> tuple[Tensor, Tensor]:
        """Apply rotary positional embeddings when provided."""
        if cos_sin is None:
            return query, key
        seq_len = query.shape[1]
        cos, sin = cos_sin[0][:seq_len], cos_sin[1][:seq_len]
        return apply_rotary_pos_emb(query, key, cos, sin)

    def _repeat_kv(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Repeat key/value heads for GQA when needed."""
        config = self.config
        if config.num_key_value_heads == config.num_heads:
            return key, value
        repeat = config.num_heads // config.num_key_value_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        return key, value

    def forward(self, hidden_states: Tensor, *, cos_sin: Optional[CosSin] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Compute attention outputs for a batch of sequences."""
        config = self.config
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value = self._project_qkv(hidden_states)

        # RoPE
        query, key = self._apply_rope(query, key, cos_sin)

        # Vanilla attention via PyTorch SDPA
        query = query.transpose(1, 2)  # [bs, heads, seq, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        key, value = self._repeat_kv(key, value)

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask, is_causal=config.causal)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)
