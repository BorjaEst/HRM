from typing import List, Optional

from pydantic import BaseModel


class PuzzleDatasetMetadata(BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]

    vocab_size: int
    seq_len: int

    total_groups: int
    mean_puzzle_examples: float

    sets: List[str]
