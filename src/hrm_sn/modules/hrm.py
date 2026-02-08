import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from hrm_sn.modules.attention import Attention, AttentionConfig
from hrm_sn.modules.embeddings import Embedding, EmbeddingConfig
from hrm_sn.modules.mlp import SwiGLU
from hrm_sn.modules.norms import rms_norm
from hrm_sn.modules.projections import CastedLinear as Linear
from hrm_sn.utils import trunc_normal_init_


@dataclass
class InnerState:
    z_H: Tensor
    z_L: Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: InnerState

    steps: Tensor
    halted: Tensor

    current_data: Dict[str, Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel, extra="forbid"):
    """Configuration for HRM ACT v1 with learned absolute positional embeddings."""

    seq_len: int = Field(
        ...,
        ge=1,
        description="Sequence length for the model (number of tokens per example).",
    )
    vocab_size: int = Field(
        ...,
        ge=1,
        description="Vocabulary size for token embeddings and LM head.",
    )
    hidden_size: int = Field(
        ...,
        ge=1,
        description="Embedding dimension for token and positional embeddings.",
    )
    num_heads: int = Field(
        ...,
        ge=1,
        description="Number of attention heads.",
    )
    num_kv_heads: Optional[int] = Field(
        default=None,
        description="Number of key/value heads for grouped-query attention. Defaults to num_heads.",
    )
    is_causal: bool = Field(
        default=False,
        description="Whether attention is causal.",
    )
    expansion: float = Field(
        default=4.0,
        gt=1.0,
        description="Expansion factor for MLP layers.",
    )
    rms_norm_eps: float = Field(
        default=1e-5,
        description="Epsilon value for RMS normalization layers.",
    )
    H_cycles: int = Field(
        ...,
        ge=1,
        description="Number of H-level reasoning cycles.",
    )
    L_cycles: int = Field(
        ...,
        ge=1,
        description="Number of L-level reasoning cycles.",
    )
    H_layers: int = Field(
        default=4,
        ge=1,
        description="Number of transformer layers in the H-level reasoning module.",
    )
    L_layers: int = Field(
        default=4,
        ge=1,
        description="Number of transformer layers in the L-level reasoning module.",
    )
    halt_max_steps: int = Field(
        ...,
        ge=1,
        description="Maximum number of ACT steps.",
    )
    halt_exploration_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Exploration probability for ACT halting.",
    )


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SettingsHRM11(BaseModel, extra="forbid"):

    attention: AttentionConfig = Field(
        ...,
        description="Attention configuration used for constructing the attention modules in the transformer layers. The keys in `attention` are passed to the attention constructor.",
    )

    @property
    def hidden_size(self) -> int:
        return self.attention.embedding_dim

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
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps

    @property
    def config(self) -> SettingsHRM11:
        return self._config

    def forward(self, x: Tensor) -> Tensor:
        attention = self.self_attn(x)
        x = rms_norm(x + attention, variance_epsilon=self.norm_eps)
        x = rms_norm(x + self.mlp(x), variance_epsilon=self.norm_eps)
        return x


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[SettingsHRM11]):
        super().__init__()

        modules = [HierarchicalReasoningModel_ACTV1Block(config) for config in layers]
        self.layers = torch.nn.ModuleList(modules)

    def forward(self, x: Tensor, input_injection: Tensor) -> Tensor:
        x = x + input_injection
        for layer in self.layers:
            x = layer(x=x)
        return x


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SettingsHRM12(SettingsHRM11, BaseModel, extra="forbid"):

    @property
    def settings_hrm11(self) -> SettingsHRM11:
        return SettingsHRM11.model_validate(self, from_attributes=True)

    token_embeddings: EmbeddingConfig = Field(
        ...,
        description="Configuration for the token embedding layer. The keys in `token_embeddings` are passed to the embedding constructor.",
    )

    @property
    def vocab_size(self) -> int:
        return self.token_embeddings.num_embeddings

    @property
    def hidden_size(self) -> int:
        return self.token_embeddings.embedding_dim

    pos_encodings: PosEncodingConfig = Field(
        ...,
        description="Configuration for positional encodings. The keys in `pos_encodings` are passed to the positional embedding constructor.",
    )

    H_layers: int = Field(
        default=4,
        ge=1,
        description="Number of transformer layers in the H-level reasoning module.",
    )
    L_layers: int = Field(
        default=4,
        ge=1,
        description="Number of transformer layers in the L-level reasoning module.",
    )


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        token_embed_config = EmbeddingConfig(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_std=embed_init_std,
        )
        pos_embed_config = EmbeddingConfig(
            num_embeddings=self.config.seq_len,
            embedding_dim=self.config.hidden_size,
            init_std=embed_init_std,
        )

        self.embed_tokens = Embedding(token_embed_config)
        self.embed_pos = Embedding(pos_embed_config)
        self.lm_head = Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = Linear(self.config.hidden_size, 2, bias=True)

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[self.config.block_config for _i in range(self.config.H_layers)],
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[self.config.block_config for _i in range(self.config.L_layers)],
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=dtype), std=1),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def empty_carry(self, batch_size: int) -> InnerState:
        config = self.config
        return InnerState(
            z_H=torch.empty(batch_size, config.seq_len, config.hidden_size),
            # dtype=self.dtype,  # TODO: resolve dtype correctly across the model
            z_L=torch.empty(batch_size, config.seq_len, config.hidden_size),
            # dtype=self.dtype,  # TODO: resolve dtype correctly across the model
        )

    def reset_carry(self, reset_flag: Tensor, carry: InnerState):
        return InnerState(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: InnerState, batch: Dict[str, Tensor]) -> Tuple[InnerState, Tensor, Tuple[Tensor, Tensor]]:
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings)
        z_H = self.H_level(z_H, z_L)

        # LM Outputs
        new_carry = InnerState(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

    def _input_embeddings(self, input: Tensor) -> Tensor:
        if input.ndim != 2:
            raise ValueError(f"Expected inputs with shape [batch, seq_len], got {tuple(input.shape)}")

        seq_len = input.shape[1]
        if seq_len > self.config.seq_len:
            raise ValueError(f"Input seq_len ({seq_len}) exceeds configured seq_len ({self.config.seq_len}).")

        token_embeddings = self.embed_tokens(input.to(torch.int32))
        positions = torch.arange(seq_len, device=input.device)
        pos_embeddings = self.embed_pos(positions).unsqueeze(0)

        # scale by 1/sqrt(2) to maintain forward variance
        embeddings = 0.707106781 * (token_embeddings + pos_embeddings)

        # Scale
        return self.embed_scale * embeddings


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
