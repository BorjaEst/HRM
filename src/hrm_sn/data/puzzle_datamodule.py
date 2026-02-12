import lightning as L
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from hrm_sn.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetMode, PuzzleDatasetRuntime, PuzzleDatasetSettings


class PuzzleDatamoduleConfig(BaseModel):
    dataset: PuzzleDatasetSettings = Field(
        ...,
        description="Configuration for the PuzzleDataset. This includes parameters like dataset path, random seed, and sampling settings.",
    )
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices. The per-device batch size is computed as `global_batch_size // world_size`.",
    )
    num_workers: int = Field(
        1,
        description="Number of workers for DataLoader. PuzzleDataset currently expects 1 worker.",
    )
    prefetch_factor: int = Field(
        8,
        description="Number of batches to prefetch per worker.",
    )
    pin_memory: bool = Field(
        True,
        description="Whether to pin memory in DataLoader.",
    )
    persistent_workers: bool = Field(
        True,
        description="Whether to keep DataLoader workers alive between epochs.",
    )


class PuzzleDatamodule(L.LightningDataModule):
    def __init__(self, settings: PuzzleDatamoduleConfig):
        super().__init__()
        self._settings = settings
        self._train_dataset = None
        self._val_dataset = None

    @property
    def settings(self) -> PuzzleDatamoduleConfig:
        return self._settings

    def _make_runtime(self) -> PuzzleDatasetRuntime:
        if self.trainer is None:
            raise RuntimeError("Trainer is not attached; call setup() via a Lightning Trainer.")
        return PuzzleDatasetRuntime(
            global_batch_size=self.settings.global_batch_size,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
        )

    def _make_dataset(self, split: str, mode: PuzzleDatasetMode) -> PuzzleDataset:
        return PuzzleDataset(
            settings=self.settings.dataset,
            runtime=self._make_runtime(),
            mode=mode,
            split=split,
        )

    def setup(self, stage: str):
        """Setup datasets for train/val/test stages."""
        if stage == "fit":
            self._train_dataset = self._make_dataset(split="train", mode="train")
            self._val_dataset = self._make_dataset(split="test", mode="eval")
        elif stage == "validate":
            self._val_dataset = self._make_dataset(split="test", mode="eval")
        elif stage == "test":
            self._val_dataset = self._make_dataset(split="test", mode="eval")

    def train_dataloader(self):
        """Return training dataloader."""
        if self._train_dataset is None:
            raise ValueError("Training dataset not initialized. Call setup('fit') first.")

        return DataLoader(
            self._train_dataset,
            batch_size=None,  # PuzzleDataset yields full batches
            num_workers=self.settings.num_workers,
            prefetch_factor=self.settings.prefetch_factor,
            pin_memory=self.settings.pin_memory,
            persistent_workers=self.settings.persistent_workers,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        if self._val_dataset is None:
            raise ValueError("Validation dataset not initialized. Call setup('validate') or setup('fit') first.")

        return DataLoader(
            self._val_dataset,
            batch_size=None,
            num_workers=self.settings.num_workers,
            prefetch_factor=self.settings.prefetch_factor,
            pin_memory=self.settings.pin_memory,
            persistent_workers=self.settings.persistent_workers,
        )
