"""MazeHard plot helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap


def _maze_cmap() -> ListedColormap:
    return ListedColormap(
        [
            "#f7f4ef",  # 0 (unused)
            "#1b1f24",  # 1 wall '#'
            "#f7f4ef",  # 2 space ' '
            "#2b6cb0",  # 3 start 'S'
            "#d69e2e",  # 4 goal 'G'
            "#f7f4ef",  # 5 path 'o' (not used in base)
        ]
    )


def plot_maze_with_overlay(
    ax: Axes,
    inputs_grid: np.ndarray,
    overlay_mask: np.ndarray,
    *,
    title: Optional[str] = None,
) -> None:
    """Render maze inputs with a semi-transparent overlay mask."""
    base = np.asarray(inputs_grid)
    overlay = np.asarray(overlay_mask).astype(float)

    ax.imshow(base, cmap=_maze_cmap(), vmin=0, vmax=5, interpolation="nearest")
    ax.imshow(overlay, cmap=ListedColormap(["none", "#e53e3e"]), alpha=0.6, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
