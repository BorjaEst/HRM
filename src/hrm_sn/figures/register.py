"""Registration helpers for built-in figures."""

from __future__ import annotations

from hrm_sn.figures.modules import dummy, overlay
from hrm_sn.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register built-in figure specifications."""

    # Overall dummy figure
    if not REGISTRY.has("dummy"):
        REGISTRY.register(
            FigureSpec(
                name="dummy",
                description="dummy figure for testing",
                plot=dummy.plot,
                default_filename="dummy",
                tags={"episode"},
                trace_keys=set(),
                extras_keys=set(),
            )
        )

    if not REGISTRY.has("overlay"):
        REGISTRY.register(
            FigureSpec(
                name="overlay",
                description="MazeHard overlays: 5 samples with GT vs model paths",
                plot=overlay.plot,
                default_filename="overlay",
                tags={"paper", "mazehard"},
                trace_keys={"act/halted", "pred/is_o"},
                extras_keys={"inputs", "labels"},
            )
        )
