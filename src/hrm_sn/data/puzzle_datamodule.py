import lightning as L
from pydantic import BaseModel, Field


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

    @property
    def settings(self) -> PuzzleDatamoduleSettings:
        return self._settings

    # TODO: Complete
