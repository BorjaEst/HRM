"""MazeHard paper figure: 5 samples with GT vs model overlays."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pub_ready_plots as prp
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hrm_sn.figures.figures.base import BaseFigureTemplate
from hrm_sn.figures.figures.panels import colorbar, panel
from hrm_sn.figures.plots.mazehard import plot_maze_with_overlay
from hrm_sn.figures.registry import FigureContext
from hrm_sn.figures.utils.mazehard import O_ID, first_halt_index, reshape_grid
from hrm_sn.rollouts.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """ """
    return OverlayFigure(trace, ctx).plot()


class OverlayFigure(BaseFigureTemplate):
    """ """

    HEIGHT_FRAC: float = 0.25
    MOSAIC = [
        ["map_a1", "map_a2", "map_a3", "map_a4", "map_a5"],
        ["map_b1", "map_b2", "map_b3", "map_b4", "map_b5"],
    ]

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)
        self.inputs, self.labels = _get_required_extras(ctx)
        self.gt_overlays = _select_gt_overlays(self.labels)
        self.model_overlays = _select_model_overlays(trace)

    def _plot_label_map(self, ax: Axes, row: int) -> None:
        input_grid = reshape_grid(self.inputs[row])
        gt_grid = reshape_grid(self.gt_overlays[row])
        plot_maze_with_overlay(ax, input_grid, gt_grid)

    @panel()  # FIXME: We need to add args to define the panel or so
    def map_a1(self, ax: Axes) -> None:
        self._plot_label_map(ax, row=0)

    @panel()
    def map_a2(self, ax: Axes) -> None:
        self._plot_label_map(ax, row=1)

    @panel()
    def map_a3(self, ax: Axes) -> None:
        self._plot_label_map(ax, row=2)

    @panel()
    def map_a4(self, ax: Axes) -> None:
        self._plot_label_map(ax, row=3)

    @panel()
    def map_a5(self, ax: Axes) -> None:
        self._plot_label_map(ax, row=4)

    def _plot_model_map(self, ax: Axes, row: int) -> None:
        input_grid = reshape_grid(self.inputs[row])
        model_grid = reshape_grid(self.model_overlays[row])
        plot_maze_with_overlay(ax, input_grid, model_grid)

    @panel()
    def map_b1(self, ax: Axes) -> None:
        self._plot_model_map(ax, row=0)

    @panel()
    def map_b2(self, ax: Axes) -> None:
        self._plot_model_map(ax, row=1)

    @panel()
    def map_b3(self, ax: Axes) -> None:
        self._plot_model_map(ax, row=2)

    @panel()
    def map_b4(self, ax: Axes) -> None:
        self._plot_model_map(ax, row=3)

    @panel()
    def map_b5(self, ax: Axes) -> None:
        self._plot_model_map(ax, row=4)


def _get_required_extras(ctx: FigureContext) -> tuple[np.ndarray, np.ndarray]:
    inputs = ctx.extras.get("inputs")
    labels = ctx.extras.get("labels")
    if inputs is None or labels is None:
        raise ValueError("overlay requires extras: inputs, labels")
    inputs_arr = np.asarray(inputs)
    labels_arr = np.asarray(labels)
    if inputs_arr.shape[0] < 5 or labels_arr.shape[0] < 5:
        raise ValueError("overlay requires at least 5 samples in extras")
    return inputs_arr[:5], labels_arr[:5]


def _select_model_overlays(trace: TraceTree) -> np.ndarray:
    halted = np.asarray(trace.get("act/halted"))
    pred_is_o = np.asarray(trace.get("pred/is_o"))
    if halted.ndim != 2:
        raise ValueError("act/halted must have shape [T, 5]")
    if pred_is_o.ndim != 3:
        raise ValueError("pred/is_o must have shape [T, 5, 900]")

    overlays = []
    for b in range(min(5, halted.shape[1])):
        t_halt = first_halt_index(halted[:, b])
        overlays.append(pred_is_o[t_halt, b])
    return np.stack(overlays, axis=0)


def _select_gt_overlays(labels: np.ndarray) -> np.ndarray:
    return labels == O_ID
