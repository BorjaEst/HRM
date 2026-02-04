from typing import List, Optional

from pydantic import BaseModel


class PuzzleDatasetMetadata(BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int

    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int

    total_groups: int
    mean_puzzle_examples: float

    sets: List[str]
