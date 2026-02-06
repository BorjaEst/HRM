import hydra
import lightning as L
from omegaconf import DictConfig

from hrm_sn.data.puzzle_datamodule import PuzzleDatamodule, PuzzleDatamoduleSettings
from hrm_sn.models.hrm_v1 import Model, ModelConfig


def make_trainer(max_epochs: int):
    trainer = L.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=10,
    )
    return trainer


@hydra.main(config_path="config", config_name="defaults", version_base=None)
def start(hydra_config: DictConfig):

    dm = PuzzleDatamodule(batch_size=32)
    model = Model(ModelConfig(**hydra_config))
    trainer = make_trainer(max_epochs=10)

    trainer.fit(model, dm)


if __name__ == "__main__":
    start()
