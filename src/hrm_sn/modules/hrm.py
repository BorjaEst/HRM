import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from hrm_sn.modules.attention import Attention, AttentionConfig
from hrm_sn.modules.embeddings import CastedEmbedding, EmbeddingsConfig
from hrm_sn.modules.mlp import SwiGLU
from hrm_sn.modules.norms import rms_norm
from hrm_sn.modules.projections import CastedLinear
from hrm_sn.modules.rotary import RotaryEmbedding
from hrm_sn.types import CosSin
from hrm_sn.utils import trunc_normal_init_


class HRMArchConfig(BaseModel):
    attention: AttentionConfig = Field(
        default_factory=AttentionConfig,
        description="Configuration for the attention modules used in the transformer layers.",
    )
    pos_encodings: Literal["RoPE", "learned"] = Field(
        default="RoPE",
        description="Type of positional encodings to use. Options are 'RoPE' for Rotary Positional Encodings and 'learned' for Learned Positional Embeddings.",
    )
    rope_theta: float = Field(
        default=10000.0,
        description="Base period for rotary positional embeddings. The period of the rotary embeddings is computed as `rope_theta ** (dim / (hidden_size // num_heads))`, where `dim` is the dimension of the rotary embeddings (i.e. `hidden_size // num_heads`).",
    )
    rms_norm_eps: float = Field(
        default=1e-5,
        description="Epsilon value for RMS normalization layers.",
    )
    expansion: float = Field(
        default=4.0,
        gt=1.0,
        description="Expansion factor for the MLP layers in the transformer blocks. The MLP hidden size is computed as `hidden_size * expansion`.",
    )

    H_layers: int = Field(
        default=4,
        ge=1,
        description="Number of transformer layers in the H-level reasoning module.",
    )
    L_layers: int = Field(
        default=2,
        ge=1,
        description="Number of transformer layers in the L-level reasoning module.",
    )

    H_cycles: int = Field(
        default=4,
        ge=1,
        description="Number of H-level cycles per forward pass. The total number of H-level iterations is `H_cycles * L_cycles`.",
    )
    L_cycles: int = Field(
        default=2,
        ge=1,
        description="Number of L-level cycles per H-level cycle. The total number of L-level iterations is `H_cycles * L_cycles`.",
    )


class ACTConfig(BaseModel):
    halt_exploration_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of taking a random action (halt or continue) for exploration during training when using ACT.",
    )
    halt_max_steps: int = Field(
        default=16,
        ge=1,
        description="Maximum number of reasoning steps before forced halting.",
    )


class NumericConfig(BaseModel):
    forward_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Data type for forward pass. Options are 'float16', 'bfloat16', and 'float32'.",
    )


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: Tensor
    z_L: Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry

    steps: Tensor
    halted: Tensor

    current_data: Dict[str, Tensor]


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SettingsHRM11(BaseModel):
    attention: AttentionConfig = Field(
        default_factory=AttentionConfig,
        description="Configuration for the attention modules used in the transformer layers.",
    )
    rms_norm_eps: float = Field(
        default=1e-5,
        description="Epsilon value for RMS normalization layers.",
    )
    expansion: float = Field(
        default=4.0,
        gt=1.0,
        description="Expansion factor for the MLP layers in the transformer blocks. The MLP hidden size is computed as `hidden_size * expansion`.",
    )


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: SettingsHRM11) -> None:
        super().__init__()
        self._config = config

        self.self_attn = Attention(config.attention)
        self.mlp = SwiGLU(config.attention.embed_dim, config.expansion)
        self.norm_eps = config.rms_norm_eps

    @property
    def config(self) -> SettingsHRM11:
        return self._config

    def forward(self, cos_sin: CosSin, hidden_states: Tensor) -> Tensor:
        attention = self.self_attn(hidden_states, cos_sin=cos_sin)
        hidden_states = rms_norm(hidden_states + attention, variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: Tensor, input_injection: Tensor, **kwargs) -> Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SettingsHRM12(BaseModel):
    embeddings: EmbeddingsConfig = Field(
        default_factory=EmbeddingsConfig,
        description="Configuration for the input embedding layer.",
    )

    forward_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Data type for forward pass. Options are 'float16', 'bfloat16', and 'float32'.",
    )


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: SettingsHRM12) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(config.embeddings)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                dtype=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: Tensor) -> Tensor:
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(
        self,
        reset_flag: Tensor,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
    ):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, Tensor],
    ) -> Tuple[
        HierarchicalReasoningModel_ACTV1InnerCarry,
        Tensor,
        Tuple[Tensor, Tensor],
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config):
        super().__init__()
        self.config = config
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(config)

    def initial_carry(self, batch: Dict[str, Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, Tensor],
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]

                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return (
            HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data),
            outputs,
        )
