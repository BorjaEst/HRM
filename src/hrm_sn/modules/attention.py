import torch
import torch.nn.functional as F
from torch import nn

from hrm_sn.modules.projections import CastedLinear
from hrm_sn.modules.rotary import apply_rotary_pos_emb
from hrm_sn.types import CosSin


class Attention(nn.Module):
    def __init__(
        self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(
            batch_size,
            seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Vanilla attention via PyTorch SDPA
        query = query.transpose(1, 2)  # [bs, heads, seq, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.num_key_value_heads != self.num_heads:
            if self.num_heads % self.num_key_value_heads != 0:
                raise ValueError(
                    "num_heads must be divisible by num_key_value_heads for SDPA"
                )
            repeat = self.num_heads // self.num_key_value_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, is_causal=self.causal
        )
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.output_size)
        )
        return self.o_proj(attn_output)
