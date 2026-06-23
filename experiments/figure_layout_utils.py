from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox


DEFAULT_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
)

DEFAULT_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")
DEFAULT_LINESTYLES = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
)


def style_triplet(index: int) -> dict[str, object]:
    return {
        "color": DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
        "marker": DEFAULT_MARKERS[index % len(DEFAULT_MARKERS)],
        "linestyle": DEFAULT_LINESTYLES[index % len(DEFAULT_LINESTYLES)],
    }


def make_style_map(keys: Sequence[object]) -> dict[object, dict[str, object]]:
    return {key: style_triplet(idx) for idx, key in enumerate(keys)}


def _ensure_path(path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def save_axes_group_panel(
    fig: plt.Figure,
    axes_group: Sequence[Axes],
    path: Path | str,
    pad_inches: float = 0.04,
) -> None:
    out = _ensure_path(path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [ax.get_tightbbox(renderer) for ax in axes_group if ax.get_visible()]
    if not bboxes:
        raise ValueError("No visible axes were provided for panel export.")
    bbox = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
    bbox = Bbox.from_extents(
        bbox.x0 - pad_inches,
        bbox.y0 - pad_inches,
        bbox.x1 + pad_inches,
        bbox.y1 + pad_inches,
    )
    fig.savefig(out, bbox_inches=bbox)


def save_legend_figure(
    handles: Sequence[object],
    labels: Sequence[str],
    path: Path | str,
    ncol: int | None = None,
    fontsize: int = 9,
) -> None:
    out = _ensure_path(path)
    if not handles or not labels:
        raise ValueError("Legend export requires nonempty handles and labels.")
    ncol_eff = ncol if ncol is not None else len(handles)
    nrows = max(1, math.ceil(len(handles) / max(1, ncol_eff)))
    width = max(3.2, 1.6 * ncol_eff)
    height = max(0.8, 0.55 * nrows + 0.15)
    fig = plt.figure(figsize=(width, height))
    fig.legend(handles, labels, loc="center", ncol=ncol_eff, frameon=False, fontsize=fontsize)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def remove_legend(ax: Axes) -> None:
    legend: Legend | None = ax.get_legend()
    if legend is not None:
        legend.remove()
