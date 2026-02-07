from lightning.pytorch.callbacks import Callback
from pydantic import BaseModel, Field


class FiguresSettings(BaseModel, extra="forbid"):
    """Settings for figure generation during training."""

    enabled: bool = Field(
        default=False,
        description="Whether to enable the figures callback.",
    )


class FiguresCallback(Callback):
    """Callback placeholder for future figure generation."""

    def __init__(self, settings: FiguresSettings):
        super().__init__()
        self.settings = settings


__all__ = ["FiguresSettings", "FiguresCallback"]
