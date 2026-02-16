"""MazeHard figure: prediction evolution over rollout steps.

This figure renders a single MazeHard puzzle and shows how the model's binary
argmax overlay prediction (`pred/is_o`) evolves over time. The first column
shows ground-truth overlay (`labels == O_ID`), followed by per-step predictions
up to the halt step.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hrm_sn.data.maze_vocab import O_ID
from hrm_sn.figures.figures.base import BaseFigureTemplate
from hrm_sn.figures.figures.panels import panel
from hrm_sn.figures.plots.mazehard import plot_maze_with_overlay
from hrm_sn.figures.registry import FigureContext
from hrm_sn.figures.utils.axes import subdivide_axes
from hrm_sn.figures.utils.mazehard import first_halt_index, reshape_grid
from hrm_sn.rollouts.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Entry point used by the figure registry."""
    return PredictionEvolutionFigure(trace, ctx).plot()


class PredictionEvolutionFigure(BaseFigureTemplate):
    """Render GT + per-step argmax overlays for one MazeHard puzzle."""

    HEIGHT_FRAC: float = 0.12
    MOSAIC = [["evolution"]]
    K_MAX: int = 12
    WSPACE: float = 0.02

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)
        self.inputs, self.labels = _get_required_extras(ctx)
        self.pred_is_o, self.halted = _get_required_trace(trace)
        self._validate_shapes()
        self.sample_idx = 0
        self.t_halt = first_halt_index(self.halted[:, self.sample_idx])
        self.t_indices = _select_timesteps(self.t_halt, self.K_MAX)

    def _validate_shapes(self) -> None:
        if self.inputs.ndim != 2:
            raise ValueError("inputs must have shape [B, N]")
        if self.labels.ndim != 2:
            raise ValueError("labels must have shape [B, N]")
        if self.pred_is_o.ndim != 3:
            raise ValueError("pred/is_o must have shape [T, B, N]")
        if self.halted.ndim != 2:
            raise ValueError("act/halted must have shape [T, B]")
        if self.inputs.shape[0] < 1 or self.labels.shape[0] < 1:
            raise ValueError("inputs and labels must contain at least one sample")
        if self.pred_is_o.shape[1] < 1:
            raise ValueError("pred/is_o must contain at least one sample")
        if self.inputs.shape[1] != self.labels.shape[1]:
            raise ValueError("inputs and labels must share the same flattened length")
        if self.inputs.shape[1] != self.pred_is_o.shape[2]:
            raise ValueError("inputs length must match pred/is_o grid size")

    @panel()
    def evolution(self, ax: Axes) -> None:
        """Plot GT + model overlays across selected rollout timesteps."""
        ncols = 1 + len(self.t_indices)
        axs = subdivide_axes(ax, nrows=1, ncols=ncols, wspace=self.WSPACE)

        input_grid = reshape_grid(self.inputs[self.sample_idx])
        gt_overlay = reshape_grid(self.labels[self.sample_idx] == O_ID)

        plot_maze_with_overlay(axs[0, 0], input_grid, gt_overlay, title="GT")

        for col, t in enumerate(self.t_indices, start=1):
            overlay = reshape_grid(self.pred_is_o[t, self.sample_idx])
            title = f"t={t}"
            if t == self.t_halt:
                title = f"{title} (halt)"
            plot_maze_with_overlay(axs[0, col], input_grid, overlay, title=title)


def _get_required_extras(ctx: FigureContext) -> tuple[np.ndarray, np.ndarray]:
    inputs = ctx.extras.get("inputs")
    labels = ctx.extras.get("labels")
    if inputs is None or labels is None:
        raise ValueError("evolution requires extras: inputs, labels")
    return np.asarray(inputs), np.asarray(labels)


def _get_required_trace(trace: TraceTree) -> tuple[np.ndarray, np.ndarray]:
    halted = np.asarray(trace.get("act/halted"))
    pred_is_o = np.asarray(trace.get("pred/is_o"))
    return pred_is_o, halted


def _select_timesteps(t_halt: int, k_max: int) -> list[int]:
    if t_halt < 0:
        return [0]
    if t_halt + 1 <= k_max:
        return list(range(t_halt + 1))
    indices = np.linspace(0, t_halt, num=k_max, dtype=int)
    t_indices = sorted({int(idx) for idx in indices})
    if t_halt not in t_indices:
        t_indices.append(t_halt)
        t_indices.sort()
    return t_indices


__all__ = ["plot", "PredictionEvolutionFigure"]
