import lightning as L
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from hrm_sn.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetSettings


class PuzzleDatamoduleConfig(BaseModel):
    # TODO: Review responsibilities with PuzzleDatasetSettings
    dataset: PuzzleDatasetSettings = Field(
        default_factory=PuzzleDatasetSettings,
        description="Configuration for the PuzzleDataset. This includes parameters like dataset path, batch size, random seed, etc.",
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
        return self.settings.dataset

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
