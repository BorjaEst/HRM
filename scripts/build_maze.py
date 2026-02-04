from typing import List, Optional

from argdantic import ArgParser

from hrm_sn.data.maze_dataset import DataProcessConfig, convert_subset

cli = ArgParser()


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
