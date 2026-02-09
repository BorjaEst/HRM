"""Hierarchical Reasoning Model (HRM) building blocks.

This module implements the core HRM architecture used in this repository:

- A small Transformer-style block (attention + MLP with RMSNorm residuals).
- A *ReasoningModule* wrapper that applies a stack of blocks with an additive
    "input injection".
- The *HRModel*, which maintains two latent state tensors (high-level $z_H$ and
    low-level $z_L$) and alternates update cycles between them.

The model is written as a plain :class:`torch.nn.Module` so it can be reused in
experiments (e.g., Lightning modules) without side effects.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from hrm_sn.modules.attention import Attention, AttentionConfig
from hrm_sn.modules.embeddings import Embedding, EmbeddingConfig
from hrm_sn.modules.mlp import SwiGLU
from hrm_sn.modules.norms import rms_norm
from hrm_sn.modules.projections import CastedLinear as Linear
from hrm_sn.utils import trunc_normal_init_


class TransformerBlockConfig(BaseModel, extra="forbid"):
    """Configuration for a single Transformer-style block.

    The hidden size is derived from the attention config (embedding dimension).
    """

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


class TransformerBlock(nn.Module):
    """A minimal Transformer block with RMSNorm residuals.

    Structure:
    1) Self-attention
    2) Residual + RMSNorm
    3) SwiGLU MLP
    4) Residual + RMSNorm

    Notes:
    - This block is intentionally small and dependency-light (vanilla PyTorch).
    - RMSNorm is applied *after* the residual add (post-norm style).
    """

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self._config = config

        self.self_attn = Attention(config.attention)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps

    @property
    def config(self) -> TransformerBlockConfig:
        return self._config

    def forward(self, x: Tensor) -> Tensor:
        attention = self.self_attn(x)
        x = rms_norm(x + attention, variance_epsilon=self.norm_eps)
        x = rms_norm(x + self.mlp(x), variance_epsilon=self.norm_eps)
        return x


class ReasoningModule(nn.Module):
    """A stack of :class:`TransformerBlock` layers with additive input injection.

    The HRM alternates between high-level and low-level reasoning modules.
    Each module receives the current state tensor and an "injection" tensor that
    anchors computation to the inputs and/or the other level's state.
    """

    def __init__(self, layers: List[TransformerBlockConfig]):
        super().__init__()

        modules = [TransformerBlock(config) for config in layers]
        self.layers = torch.nn.ModuleList(modules)

    def forward(self, x: Tensor, input_injection: Tensor) -> Tensor:
        x = x + input_injection
        for layer in self.layers:
            x = layer(x=x)
        return x


class HRMConfig(BaseModel, extra="allow"):
    """Top-level configuration for :class:`HRModel`.

    This config intentionally exposes derived convenience properties (e.g.
    ``hidden_size``, embedding init std) to reduce duplication across components.

    ``extra=allow`` is used to tolerate experiment-level config keys that are not
    consumed by the core model.
    """

    transformer_block: TransformerBlockConfig = Field(
        ...,
        description="Base transformer block configuration for constructing the reasoning modules at both H and L levels.",
    )

    @property
    def hidden_size(self) -> int:
        """Convenience property to access hidden size from the attention config."""
        return self.transformer_block.hidden_size

    @property
    def embedding_dim(self) -> int:
        """Convenience property to access embedding dimension from the attention config."""
        return self.hidden_size

    @property
    def embedding_scale(self) -> float:
        """Convenience property for scaling embeddings, computed as sqrt of hidden size divided by sqrt(2) to maintain variance."""
        # scale by 1/sqrt(2) to maintain forward variance
        return 0.707106781 * math.sqrt(self.hidden_size)

    @property
    def init_std(self) -> float:
        """Convenience property for standard deviation of truncated normal initialization, computed as the inverse of the square root of hidden size."""
        return 1.0 / math.sqrt(self.hidden_size)

    # Model parameters for input
    vocab_size: int = Field(
        ...,
        ge=1,
        description="Vocabulary size for token embeddings and LM head.",
    )

    @property
    def token_embeddings(self) -> EmbeddingConfig:
        """Convenience function to construct token embedding configuration from the model config."""
        return EmbeddingConfig(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            init_std=self.init_std,
        )

    seq_len: int = Field(
        ...,
        ge=1,
        description="Sequence length for the model (number of tokens per example).",
    )

    @property
    def pos_embeddings(self) -> EmbeddingConfig:
        """Convenience function to construct positional embedding configuration from the model config."""
        return EmbeddingConfig(
            num_embeddings=self.seq_len,
            embedding_dim=self.embedding_dim,
            init_std=self.init_std,
        )

    # Reasoning module parameters for high level
    H_layers: int = Field(
        default=4,
        ge=1,
        description="Number of transformer layers in the H-level reasoning module.",
    )
    H_cycles: int = Field(
        default=4,
        ge=1,
        description="Number of cycles for the H-level reasoning module.",
    )

    # Reasoning module parameters for low level
    L_layers: int = Field(
        default=4,
        ge=1,
        description="Number of transformer layers in the L-level reasoning module.",
    )
    L_cycles: int = Field(
        default=4,
        ge=1,
        description="Number of cycles for the L-level reasoning module.",
    )


@dataclass
class HRMState:
    """Recurrent state carried between forward passes.

    Attributes:
        z_H: High-level state of shape ``[batch, seq_len, hidden_size]``.
        z_L: Low-level state of shape ``[batch, seq_len, hidden_size]``.
    """

    z_H: Tensor  # Higher-level state tensor of shape [batch, seq_len, hidden_size].
    z_L: Tensor  # Lower-level state tensor of shape [batch, seq_len, hidden_size].


class HRModel(nn.Module):
    """Hierarchical Reasoning Model (HRM).

    The model maintains two latent sequences, a high-level state ``z_H`` and a
    low-level state ``z_L``. Computation proceeds by alternating *cycles* of
    updates between these two levels.

    Implementation detail: to reduce memory usage, the bulk of the iterative
    updates are executed under ``torch.no_grad()`` and the final update at each
    level is executed with gradients ("one-step grad"). This is a deliberate
    trade-off between compute, memory, and training signal.
    """

    def __init__(self, config: HRMConfig, device=None, dtype=None) -> None:
        super().__init__()
        self._config = config

        self.embed_tokens = Embedding(config.token_embeddings)
        self.embed_pos = Embedding(config.pos_embeddings)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.halt_q_head = Linear(config.hidden_size, 2, bias=True)

        # Reasoning Layers
        self.high_level = ReasoningModule([config.transformer_block for _ in range(config.H_layers)])
        self.low_level = ReasoningModule([config.transformer_block for _ in range(config.L_layers)])

        # Initial states
        self.register_buffer("high_init", torch.empty((config.hidden_size,)), persistent=True)
        self.register_buffer("low_init", torch.empty((config.hidden_size,)), persistent=True)
        self.high_init = cast(Tensor, self.high_init)
        self.low_init = cast(Tensor, self.low_init)

        self.reset_parameters()

    @property
    def config(self) -> HRMConfig:
        return self._config

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers.

        Only model-owned buffers/heads are initialized here. Submodules (e.g.
        attention/MLP) initialize themselves.
        """
        trunc_normal_init_(self.high_init, std=1)
        trunc_normal_init_(self.low_init, std=1)

        # Initialize Q head near-zero so early training behaves predictably.
        with torch.no_grad():
            self.halt_q_head.weight.zero_()
            self.halt_q_head.bias.fill_(-5)  # type: ignore

    def empty_carry(self, batch_size: int) -> HRMState:
        """Allocate an uninitialized state state with the right shape/dtype/device."""
        config = self.config
        return HRMState(
            z_H=self.high_init.new_empty(batch_size, config.seq_len, config.hidden_size),
            z_L=self.low_init.new_empty(batch_size, config.seq_len, config.hidden_size),
        )

    def reset_carry(self, reset_flag: Tensor, state: HRMState) -> HRMState:
        """Reset selected batch elements of the state to learned initial states.

        Args:
            reset_flag: Boolean-ish tensor of shape ``[batch]`` (or broadcastable
                to it). True entries reset the corresponding state sequences.
            state: Current state.
        """
        init_H = self.high_init.view(1, 1, -1).expand_as(state.z_H)
        init_L = self.low_init.view(1, 1, -1).expand_as(state.z_L)
        mask = reset_flag.view(-1, 1, 1)
        return HRMState(
            z_H=torch.where(mask, init_H, state.z_H),
            z_L=torch.where(mask, init_L, state.z_L),
        )

    def forward_act(self, state: HRMState, batch: Dict[str, Tensor]) -> Tuple[HRMState, Tensor, Tuple[Tensor, Tensor]]:
        """Convenience wrapper for ACT-style training loops.

        Expects ``batch`` to contain ``"inputs"`` with shape ``[batch, seq_len]``.
        """
        return self(batch["inputs"], state=state)

    def forward(self, input_ids: Tensor, state: Optional[HRMState] = None) -> Tuple[HRMState, Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass.

        Args:
            input_ids: Token ids with shape ``[batch, seq_len]``.
            state: Optional previous :class:`HRMState`. If omitted, an empty
                state is allocated.

        Returns:
            ``(new_carry, lm_logits, (halt_q0, halt_q1))`` where:
            - ``new_carry`` contains detached states to state to the next step.
            - ``lm_logits`` has shape ``[batch, seq_len, vocab_size]``.
            - Halt Q logits are per-example (computed from position 0).
        """
        state = state or self.empty_carry(batch_size=input_ids.shape[0])
        x = self.embed_inputs(input_ids)

        # Forward iterations without grad for memory efficiency.
        # The final update at each level is executed with gradients below.
        with torch.no_grad():
            self.run_low_cycles(x, state)
            self.run_high_cycles(x, state, n_cycles=self.config.H_cycles - 1)
            self.run_low_cycles(x, state, n_cycles=self.config.L_cycles - 1)

        # One-step grad: provide a training signal while keeping memory bounded.
        z_L = self.low_level(state.z_L, state.z_H + x)
        z_H = self.high_level(state.z_H, state.z_L)

        # Carry is detached so the next step does not backprop through time.
        new_carry = HRMState(z_H=z_H.detach(), z_L=z_L.detach())
        # Language-modeling head predicts a token distribution at each position.
        output = self.lm_head(z_H)
        # Halt Q head is computed from a single "summary" token (position 0).
        halt_q_logits = self.halt_q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (halt_q_logits[..., 0], halt_q_logits[..., 1])

    def run_high_cycles(self, x: Tensor, state: HRMState, n_cycles: Optional[int] = None) -> Tensor:
        """Iterate high-level cycles, interleaving low-level updates."""
        for _ in range(n_cycles or self.config.H_cycles):
            state.z_H = self.high_level(state.z_H, state.z_L)
            state.z_L = self.run_low_cycles(x, state)
        return state.z_H

    def run_low_cycles(self, x: Tensor, state: HRMState, n_cycles: Optional[int] = None) -> Tensor:
        """Iterate low-level cycles (conditioned on high-level state and inputs)."""
        for _ in range(n_cycles or self.config.L_cycles):
            state.z_L = self.low_level(state.z_L, state.z_H + x)
        return state.z_L

    def embed_inputs(self, input: Tensor) -> Tensor:
        """Embed token ids and add positional embeddings.

        Args:
            input: Integer token ids with shape ``[batch, seq_len]``.

        Returns:
            Embedded inputs with shape ``[batch, seq_len, hidden_size]``.

        Raises:
            ValueError: If input dimensionality is not 2D or if ``seq_len``
                exceeds the configured maximum.
        """
        if input.ndim != 2:
            raise ValueError(f"Expected inputs with shape [batch, seq_len], got {tuple(input.shape)}")

        seq_len = input.shape[1]
        if seq_len > self.config.seq_len:
            raise ValueError(f"Input seq_len ({seq_len}) exceeds configured seq_len ({self.config.seq_len}).")

        token_embeddings = self.embed_tokens(input.to(torch.int32))
        positions = torch.arange(seq_len, device=input.device)
        pos_embeddings = self.embed_pos(positions).unsqueeze(0)

        # Scale embeddings to keep activations in a reasonable range.
        return self.config.embedding_scale * (token_embeddings + pos_embeddings)
