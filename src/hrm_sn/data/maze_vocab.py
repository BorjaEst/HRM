"""Maze token vocabulary and IDs."""

from __future__ import annotations

from typing import Dict

MAZE_CHARSET = "# SGo"

MAZE_CHAR_TO_ID: Dict[str, int] = {ch: idx + 1 for idx, ch in enumerate(MAZE_CHARSET)}
MAZE_ID_TO_CHAR: Dict[int, str] = {idx: ch for ch, idx in MAZE_CHAR_TO_ID.items()}

PAD_ID = 0
WALL_ID = MAZE_CHAR_TO_ID["#"]
SPACE_ID = MAZE_CHAR_TO_ID[" "]
START_ID = MAZE_CHAR_TO_ID["S"]
GOAL_ID = MAZE_CHAR_TO_ID["G"]
O_ID = MAZE_CHAR_TO_ID["o"]

__all__ = [
    "MAZE_CHARSET", "MAZE_CHAR_TO_ID", "MAZE_ID_TO_CHAR", "PAD_ID", "WALL_ID", "SPACE_ID",
    "START_ID", "GOAL_ID", "O_ID",
]  # fmt: skip
