from hrm_sn.data.maze_dataset import MazeDataModule
from hrm_sn.models.navigation import HRMNavModel
from hrm_sn.training.trainer_factory import make_trainer

dm = MazeDataModule(batch_size=32)
model = HRMNavModel(lr=1e-3)
trainer = make_trainer(gpus=1, max_epochs=10)

trainer.fit(model, dm)
