from lightning.pytorch.callbacks import Callback
from pydantic import BaseModel, Field


class FiguresSettings(BaseModel, extra="forbid"):
    """Settings for figure generation during training."""

    # TODO: implement settings for figure generation (e.g., frequency, types of figures, etc.)


class FiguresCallback(Callback):
    """Custom ModelCheckpoint that accepts CheckpointSettings."""

    def __init__(self, settings: FiguresSettings):
        super().__init__(**settings.model_dump())


__all__ = ["FiguresSettings", "FiguresCallback"]
