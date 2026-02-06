from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from adam_atan2_pytorch import AdamAtan2
from pydantic import BaseModel, Field
from torch.optim.optimizer import Optimizer, ParamsT

__all__ = ["AdamATan2Config", "AdamATan2", "CastedSparseEmbeddingSignSGDConfig", "CastedSparseEmbeddingSignSGD_Distributed"]


class AdamATan2Config(BaseModel, extra="forbid"):
    lr: float = Field(
        default=1e-4,
        description="Base learning rate for the main optimizer (e.g. Adam). The learning rate for the puzzle embedding optimizer is set by `puzzle_emb_lr`.",
    )
    weight_decay: float = Field(
        default=1e-2,
        description="Weight decay for the main optimizer (e.g. Adam). The weight decay for the puzzle embedding optimizer is set by `emb_weight_decay`.",
    )
    betas: Tuple[float, float] = Field(
        default=(0.9, 0.98),
        description="Betas for Adam optimizer. The betas for the puzzle embedding optimizer are not set by default since Adam is not used for the puzzle embedding optimizer.",
    )


class AdamATan2(AdamAtan2):
    def __init__(self, params: ParamsT, config: Optional[AdamATan2Config] = None):
        config = config or AdamATan2Config()
        super().__init__(params, **config.model_dump())


class CastedSparseEmbeddingSignSGDConfig(BaseModel, extra="forbid"):
    lr: float = Field(
        default=1e-3,
        ge=0.0,
        le=1.0,
        description="Base learning rate for the puzzle embedding optimizer (e.g. SignSGD). The learning rate for the main optimizer is set by `lr`.",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight decay for the puzzle embedding optimizer (e.g. SignSGD). The weight decay for the main optimizer is set by `weight_decay`.",
    )
    world_size: int = Field(
        default=1,
        description="World size for distributed training. If greater than 1, the optimizer performs all-gather and unique operations across all processes to ensure consistent updates of the sparse embedding weights.",
    )


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(self, params: ParamsT, config: Optional[CastedSparseEmbeddingSignSGDConfig] = None):
        config = config or CastedSparseEmbeddingSignSGDConfig()
        super().__init__(params, config.model_dump())

    @torch.no_grad
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            # Find the sparse embedding weights
            local_weights_grad = None
            local_ids = None
            weights = None

            assert len(group["params"]) == 3
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False

            assert local_weights_grad is not None
            assert local_ids is not None
            assert weights is not None

            # Apply SignSGD
            # Adam ≈ SignSGD if gradient is very sparse
            self._sparse_emb_signsgd_dist(
                local_weights_grad,
                local_ids,
                weights,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                world_size=group["world_size"],
            )

    @staticmethod
    def _sparse_emb_signsgd_dist(
        local_weights_grad: torch.Tensor,
        local_ids: torch.Tensor,
        weights: torch.Tensor,
        lr: float,
        weight_decay: float,
        world_size: int,
    ) -> None:
        N, D = local_weights_grad.shape

        # All-gather
        all_weights_grad = local_weights_grad
        all_ids = local_ids

        if world_size > 1:
            all_weights_grad = torch.empty(
                (world_size * N, D),
                dtype=local_weights_grad.dtype,
                device=local_weights_grad.device,
            )
            all_ids = torch.empty(world_size * N, dtype=local_ids.dtype, device=local_ids.device)

            dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
            dist.all_gather_into_tensor(all_ids, local_ids)

        # Unique
        grad_ids, inv = all_ids.unique(return_inverse=True)

        grad = torch.zeros(
            (grad_ids.shape[0], D),
            dtype=all_weights_grad.dtype,
            device=all_weights_grad.device,
        )
        grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

        # SignSGD with decoupled weight decay
        p = weights[grad_ids]

        p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

        # Write updated slices back
        weights[grad_ids] = p
