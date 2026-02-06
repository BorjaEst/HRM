import lightning as L
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from hrm_sn.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetSettings


class PuzzleDatamoduleSettings(BaseModel):
    # TODO: Review responsibilities with PuzzleDatasetSettings
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
    )
    dataset_path: str = Field(
        ...,
        description="Path to the dataset directory containing `train/`, `test/`, etc. subdirectories.",
    )
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices. The per-device batch size is computed as `global_batch_size // num_replicas`.",
    )
    test_set_mode: bool = Field(
        default=False,
        description="Iterate in test set mode through the dataset without randomization, yields puzzle identifiers for each batch.",
    )

    epochs_per_iter: int = Field(
        default=1,
        description="Epochs to iterate in each iteration. This is used to reduce overhead of randomization and shuffling.",
    )

    rank: int = Field(
        default=0,
        description="Rank of the current process for distributed training. Should be in the range [0, num_replicas - 1].",
    )
    num_replicas: int = Field(
        default=1,
        description="Total number of processes for distributed training. The dataset is split across these processes according to `rank`.",
    )


class PuzzleDatamodule(L.LightningDataModule):
    def __init__(self, settings: PuzzleDatamoduleSettings):
        super().__init__()
        self._settings = settings
        self._train_dataset = None
        self._val_dataset = None

    @property
    def settings(self) -> PuzzleDatamoduleSettings:
        return self._settings

    def _make_dataset_config(self, test_set_mode: bool = False) -> PuzzleDatasetSettings:
        """Helper to create PuzzleDatasetSettings from PuzzleDatamoduleSettings."""
        return PuzzleDatasetSettings(
            seed=self._settings.seed,
            dataset_path=self._settings.dataset_path,
            global_batch_size=self._settings.global_batch_size,
            test_set_mode=test_set_mode,
            epochs_per_iter=self._settings.epochs_per_iter,
            rank=self._settings.rank,
            num_replicas=self._settings.num_replicas,
        )

    def setup(self, stage: str):
        """Setup datasets for train/val/test stages."""
        if stage == "fit":
            # Training dataset: test_set_mode=False (randomized)
            self._train_dataset = PuzzleDataset(
                config=self._make_dataset_config(test_set_mode=False),
                split="train",
            )
            # Validation dataset: test_set_mode=True (deterministic)
            self._val_dataset = PuzzleDataset(
                config=self._make_dataset_config(test_set_mode=True),
                split="test",
            )
        elif stage == "validate":
            self._val_dataset = PuzzleDataset(
                config=self._make_dataset_config(test_set_mode=True),
                split="test",
            )
        elif stage == "test":
            self._val_dataset = PuzzleDataset(
                config=self._make_dataset_config(test_set_mode=True),
                split="test",
            )

    def train_dataloader(self):
        """Return training dataloader."""
        if self._train_dataset is None:
            raise ValueError("Training dataset not initialized. Call setup('fit') first.")

        return DataLoader(
            self._train_dataset,
            batch_size=None,  # PuzzleDataset yields full batches
            num_workers=1,
            prefetch_factor=8,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        if self._val_dataset is None:
            raise ValueError("Validation dataset not initialized. Call setup('validate') or setup('fit') first.")

        return DataLoader(
            self._val_dataset,
            batch_size=None,
            num_workers=1,
            prefetch_factor=8,
            pin_memory=True,
            persistent_workers=True,
        )
