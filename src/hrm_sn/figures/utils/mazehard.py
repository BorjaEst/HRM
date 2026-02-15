"""MazeHard figure utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np

GRID_N = 30
SEQ_LEN = GRID_N * GRID_N
O_ID = 5


def reshape_grid(seq: np.ndarray) -> np.ndarray:
    """Reshape a flat sequence into a GRID_N x GRID_N grid."""
    arr = np.asarray(seq)
    if arr.size != SEQ_LEN:
        raise ValueError(f"Expected sequence length {SEQ_LEN}, got {arr.size}")
    return arr.reshape((GRID_N, GRID_N))


def first_halt_index(halted_t: Iterable[bool]) -> int:
    """Return the first index where halted is True, or the last index if none."""
    halted_arr = np.asarray(list(halted_t), dtype=bool)
    if halted_arr.size == 0:
        return 0
    if halted_arr.any():
        return int(np.argmax(halted_arr))
    return int(halted_arr.size - 1)
