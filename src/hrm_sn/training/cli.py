import importlib
import inspect
import math
import os
import shutil
from typing import Any, List, Optional, Sequence

import coolname
import hydra
import pydantic
import torch
import torch.distributed as dist
import tqdm
import yaml
from adam_atan2_pytorch import AdamAtan2 as AdamATan2
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hrm_sn import loss
from hrm_sn.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetMetadata, PuzzleDatasetSettings
from hrm_sn.modules import hrm
from hrm_sn.training.optim import CastedSparseEmbeddingSignSGD_Distributed


def create_dataloader(config: PuzzleDatasetSettings, split: str):
    dataset = PuzzleDataset(config, split)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1, prefetch_factor=8, pin_memory=True, persistent_workers=True)
    return dataloader, dataset.metadata


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log(
            {"num_params": sum(x.numel() for x in train_state.model.parameters())},
            step=0,
        )
        save_code_and_config(config)

    # Training Loop
    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

        ############ Evaluation
        train_state.model.eval()
        metrics = evaluate(
            config,
            train_state,
            eval_loader,
            eval_metadata,
            rank=RANK,
            world_size=WORLD_SIZE,
        )

        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)

        ############ Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
