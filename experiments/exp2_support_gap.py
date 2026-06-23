from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_layout_utils import save_axes_group_panel, save_legend_figure
from sklearn.kernel_ridge import KernelRidge


DIAGNOSTIC_SCALED = "scaled_bump_theorem_realization"
DIAGNOSTIC_FIXED = "fixed_target_hole_enlargement"

DEFAULT_H_VALUES_SCALED = (0.04, 0.06, 0.08, 0.12, 0.16, 0.24)
DEFAULT_H_VALUES_FIXED = (0.02, 0.035, 0.05, 0.075, 0.10, 0.14, 0.18)
DEFAULT_S_VALUES = (1, 2)
DEFAULT_X0_SCALED = np.asarray([0.0, 0.0], dtype=np.float64)
DEFAULT_X0_FIXED = np.asarray([0.72, 0.72], dtype=np.float64)
ILLUSTRATION_H_VALUES = (0.08, 0.16, 0.24)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: support-gap scaling and local error floors."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--fast", action="store_true")

    parser.add_argument("--n-train-scaled", type=int, default=1000)
    parser.add_argument("--n-reps-scaled", type=int, default=60)
    parser.add_argument("--n-eval-local-scaled", type=int, default=5000)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--scaled-alpha", type=float, default=1e-6)
    parser.add_argument("--scaled-gamma", type=float, default=15.0)

    parser.add_argument("--n-train-fixed", type=int, default=800)
    parser.add_argument("--n-reps-fixed", type=int, default=30)
    parser.add_argument("--n-eval-local-fixed", type=int, default=4000)
    parser.add_argument("--n-test-global", type=int, default=8000)
    parser.add_argument("--fixed-r-eval", type=float, default=0.06)
    parser.add_argument("--fixed-sigma", type=float, default=0.03)
    parser.add_argument("--fixed-gamma", type=float, default=20.0)
    parser.add_argument("--fixed-lambda", type=float, default=1e-3)
    parser.add_argument("--fixed-bump-amplitude", type=float, default=1.0)
    parser.add_argument("--fixed-bump-alpha", type=float, default=80.0)
    parser.add_argument("--k-support", type=int, default=10)
    return parser.parse_args()


def apply_fast_mode(args: argparse.Namespace) -> None:
    if not args.fast:
        return
    args.n_reps_scaled = min(args.n_reps_scaled, 20)
    args.n_eval_local_scaled = min(args.n_eval_local_scaled, 2000)
    args.n_reps_fixed = min(args.n_reps_fixed, 10)
    args.n_eval_local_fixed = min(args.n_eval_local_fixed, 1500)
    args.n_test_global = min(args.n_test_global, 3000)


def resolve_outdir(outdir: str) -> Path:
    path = Path(outdir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def ensure_dirs(base: Path) -> Tuple[Path, Path, Path]:
    figures = base / "figures"
    tables = base / "tables"
    results = base / "results"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return figures, tables, results


def sample_uniform_box(
    n: int,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.uniform(low, high, size=(n, 2)).astype(np.float64)


def sample_uniform_with_hole(
    n: int,
    h: float,
    x0: np.ndarray,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    accepted: List[np.ndarray] = []
    remaining = int(n)
    while remaining > 0:
        batch = sample_uniform_box(max(remaining * 2, 1024), low=low, high=high, rng=rng)
        distances = np.linalg.norm(batch - x0[None, :], axis=1)
        keep = batch[distances >= h]
        if keep.size == 0:
            continue
        accepted.append(keep[:remaining])
        remaining -= min(remaining, keep.shape[0])
    X = np.vstack(accepted)
    return X[:n]


def sample_points_in_ball(
    n: int,
    center: np.ndarray,
    radius: float,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    direction = rng.normal(size=(n, dim))
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    zero_mask = norm.squeeze(-1) == 0.0
    while np.any(zero_mask):
        direction[zero_mask] = rng.normal(size=(int(np.sum(zero_mask)), dim))
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        zero_mask = norm.squeeze(-1) == 0.0
    direction = direction / norm
    radial = radius * np.power(rng.uniform(size=n), 1.0 / dim)
    return center[None, :] + direction * radial[:, None]


def radial_bump(X: np.ndarray, h: float, s: int, x0: np.ndarray) -> np.ndarray:
    U = (np.asarray(X, dtype=np.float64) - x0[None, :]) / float(h)
    r2 = np.sum(U * U, axis=1)
    out = np.zeros(X.shape[0], dtype=np.float64)
    mask = r2 < 1.0
    if np.any(mask):
        out[mask] = np.exp(1.0 - 1.0 / (1.0 - r2[mask]))
    return (float(h) ** int(s)) * out


def fixed_target(
    X: np.ndarray,
    x0: np.ndarray,
    amplitude: float,
    alpha: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]
    bump = amplitude * np.exp(-alpha * ((x1 - x0[0]) ** 2 + (x2 - x0[1]) ** 2))
    return np.sin(2.0 * np.pi * x1) * np.cos(2.0 * np.pi * x2) + bump + 0.25 * x1


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    sqA = np.sum(A * A, axis=1)[:, None]
    sqB = np.sum(B * B, axis=1)[None, :]
    sqdist = np.maximum(sqA + sqB - 2.0 * A @ B.T, 0.0)
    return np.exp(-float(gamma) * sqdist)


def train_reference_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    gamma: float,
) -> Dict[str, np.ndarray | float | KernelRidge]:
    try:
        model = KernelRidge(alpha=float(alpha), kernel="rbf", gamma=float(gamma))
        model.fit(X_train, y_train)
        return {
            "backend": "sklearn",
            "model": model,
            "alpha_effective": float(alpha),
        }
    except Exception:
        pass

    K = rbf_kernel(X_train, X_train, gamma=gamma)
    n = K.shape[0]
    eye = np.eye(n, dtype=np.float64)
    ridge = float(alpha)
    last_error: Exception | None = None
    for _ in range(5):
        try:
            dual = np.linalg.solve(K + ridge * eye, y_train)
            return {
                "backend": "manual",
                "X_train": X_train,
                "dual_coef": dual,
                "gamma": float(gamma),
                "alpha_effective": ridge,
            }
        except np.linalg.LinAlgError as exc:
            last_error = exc
            ridge *= 10.0
    raise RuntimeError("Failed to solve KRR system after jitter escalation.") from last_error


def predict_reference_model(
    model: Dict[str, np.ndarray | float | KernelRidge],
    X_query: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    if model["backend"] == "sklearn":
        predictor = model["model"]
        outputs: List[np.ndarray] = []
        for start in range(0, X_query.shape[0], batch_size):
            stop = min(start + batch_size, X_query.shape[0])
            pred = np.asarray(predictor.predict(X_query[start:stop]), dtype=np.float64)
            if pred.ndim == 1:
                pred = pred[:, None]
            outputs.append(pred)
        return np.vstack(outputs)

    X_train = np.asarray(model["X_train"], dtype=np.float64)
    dual = np.asarray(model["dual_coef"], dtype=np.float64)
    gamma = float(model["gamma"])
    outputs: List[np.ndarray] = []
    for start in range(0, X_query.shape[0], batch_size):
        stop = min(start + batch_size, X_query.shape[0])
        K_q = rbf_kernel(np.asarray(X_query[start:stop], dtype=np.float64), X_train, gamma=gamma)
        pred = K_q @ dual
        if pred.ndim == 1:
            pred = pred[:, None]
        outputs.append(pred)
    return np.vstack(outputs)


def compute_knn_radius(
    X_query: np.ndarray,
    X_train: np.ndarray,
    k: int,
    batch_size: int = 2048,
) -> np.ndarray:
    X_query = np.asarray(X_query, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    k_eff = min(max(int(k), 1), X_train.shape[0])
    out: List[np.ndarray] = []
    sq_train = np.sum(X_train * X_train, axis=1)[None, :]
    for start in range(0, X_query.shape[0], batch_size):
        stop = min(start + batch_size, X_query.shape[0])
        Q = X_query[start:stop]
        sq_q = np.sum(Q * Q, axis=1)[:, None]
        sqdist = np.maximum(sq_q + sq_train - 2.0 * Q @ X_train.T, 0.0)
        kth = np.partition(sqdist, kth=k_eff - 1, axis=1)[:, k_eff - 1]
        out.append(np.sqrt(kth))
    return np.concatenate(out, axis=0)


def fit_loglog_slope(h_values: Sequence[float], mse_values: Sequence[float]) -> Tuple[float, float]:
    x = np.log(np.asarray(h_values, dtype=np.float64))
    y = np.log(np.maximum(np.asarray(mse_values, dtype=np.float64), 1e-300))
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = math.sqrt(float(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def rank_values(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson_corr(rank_values(x), rank_values(y))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)


def summarize_array(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if arr.shape[0] > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "se": float(std / math.sqrt(arr.shape[0])) if arr.shape[0] > 1 else 0.0,
        "median": float(np.median(arr)),
        "n": int(arr.shape[0]),
    }


def make_support_hole_figure(
    figures_dir: Path,
    illustration_designs: Dict[float, np.ndarray],
    rho: float,
    x0: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    panel_handles = None
    panel_labels = None
    for ax, h in zip(axes, ILLUSTRATION_H_VALUES):
        X = illustration_designs[float(h)]
        pts = ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.55, color="#4c78a8", label="Training samples")
        center = ax.scatter([x0[0]], [x0[1]], s=45, color="#111111", marker="x", linewidths=1.5, label=r"$x_0$")
        hole = plt.Circle((x0[0], x0[1]), radius=float(h), fill=False, color="#e45756", linewidth=2.0, label=r"Support hole $B(x_0,h)$")
        local = plt.Circle((x0[0], x0[1]), radius=float(rho * h), fill=False, color="#54a24b", linestyle="--", linewidth=2.0, label=r"Local region $G_h$")
        ax.add_patch(hole)
        ax.add_patch(local)
        ax.set_xlim(-1.03, 1.03)
        ax.set_ylim(-1.03, 1.03)
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
        ax.set_title(rf"$h={h:.2f}$")
        ax.set_xlabel(r"$x_1$")
        if ax is axes[0]:
            ax.set_ylabel(r"$x_2$")
        panel_handles = [pts, center, hole, local]
        panel_labels = [hnd.get_label() for hnd in panel_handles]

    for ax, h in zip(axes, ILLUSTRATION_H_VALUES):
        save_axes_group_panel(
            fig,
            [ax],
            figures_dir / f"exp2_support_holes_panel_h{int(round(100 * h)):03d}.pdf",
        )
    if panel_handles is not None and panel_labels is not None:
        save_legend_figure(
            panel_handles,
            panel_labels,
            figures_dir / "exp2_support_holes_legend.pdf",
            ncol=4,
        )
    plt.close(fig)


def make_loglog_scaling_figure(
    figures_dir: Path,
    h_values: Sequence[float],
    per_h_summary: Dict[Tuple[str, int], Dict[float, Dict[str, float]]],
    slope_summary: Dict[Tuple[str, int], Dict[str, float]],
) -> None:
    styles = {
        "hole": {"color": "#e45756", "marker": "o", "linestyle": "-"},
        "dense": {"color": "#4c78a8", "marker": "s", "linestyle": "--"},
        "reference": {"color": "#222222", "marker": "D", "linestyle": "-."},
    }
    labels = {"hole": "Hole design", "dense": "Dense reference"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    for ax, s in zip(axes, DEFAULT_S_VALUES):
        x = np.asarray(h_values, dtype=np.float64)
        for design in ("hole", "dense"):
            summary = per_h_summary[(design, s)]
            y = np.asarray([summary[float(h)]["mse_mean"] for h in h_values], dtype=np.float64)
            yerr = np.asarray([summary[float(h)]["mse_se"] for h in h_values], dtype=np.float64)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=styles[design]["color"],
                linestyle=styles[design]["linestyle"],
                marker=styles[design]["marker"],
                markersize=4.5,
                linewidth=2.0,
                capsize=3,
                label=labels[design],
            )
            slope = slope_summary[(design, s)]["slope_mean"]
            intercept = slope_summary[(design, s)]["intercept_mean"]
            fit = np.exp(intercept) * np.power(x, slope)
            ax.plot(
                x,
                fit,
                color=styles[design]["color"],
                linestyle=":",
                linewidth=1.5,
                marker=styles[design]["marker"],
                markersize=3.8,
                markevery=(1, 2),
            )

        ref_slope = 2 * s
        hole_y = np.asarray([per_h_summary[("hole", s)][float(h)]["mse_mean"] for h in h_values], dtype=np.float64)
        ref_idx = len(h_values) // 2
        ref_const = hole_y[ref_idx] / (x[ref_idx] ** ref_slope)
        ax.plot(
            x,
            ref_const * np.power(x, ref_slope),
            color=styles["reference"]["color"],
            linestyle=styles["reference"]["linestyle"],
            marker=styles["reference"]["marker"],
            markersize=3.8,
            linewidth=1.6,
            label=r"Reference slope $2s$",
        )

        text = (
            rf"Predicted $2s={2*s}$" "\n"
            rf"Hole: {slope_summary[('hole', s)]['slope_mean']:.2f}"
            rf" $\pm$ {slope_summary[('hole', s)]['slope_se']:.2f}" "\n"
            rf"Dense: {slope_summary[('dense', s)]['slope_mean']:.2f}"
        )
        ax.text(
            0.04,
            0.05,
            text,
            transform=ax.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        ax.set_title(rf"Smoothness $s={s}$")
        ax.set_xlabel(r"Support-gap radius $h$")
        if ax is axes[0]:
            ax.set_ylabel(r"Local MSE$_{G_h}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.2, which="both")

    handles, labels_out = axes[0].get_legend_handles_labels()
    for ax, s in zip(axes, DEFAULT_S_VALUES):
        save_axes_group_panel(fig, [ax], figures_dir / f"exp2_loglog_scaling_panel_s{s}.pdf")
    save_legend_figure(handles, labels_out, figures_dir / "exp2_loglog_scaling_legend.pdf", ncol=3)
    plt.close(fig)


def make_scaled_bump_summary_panel(
    figures_dir: Path,
    h_values: Sequence[float],
    per_h_summary: Dict[Tuple[str, int], Dict[float, Dict[str, float]]],
    slope_summary: Dict[Tuple[str, int], Dict[str, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.3), constrained_layout=True)
    handles, labels_out = plot_scaled_bump_summary_axis(
        ax=ax,
        h_values=h_values,
        per_h_summary=per_h_summary,
        slope_summary=slope_summary,
        include_reference_slopes=False,
    )
    save_axes_group_panel(fig, [ax], figures_dir / "exp2_support_gap_combined_panel_a.pdf")
    save_legend_figure(handles, labels_out, figures_dir / "exp2_support_gap_combined_legend.pdf", ncol=3)
    plt.close(fig)


def plot_scaled_bump_summary_axis(
    ax: plt.Axes,
    h_values: Sequence[float],
    per_h_summary: Dict[Tuple[str, int], Dict[float, Dict[str, float]]],
    slope_summary: Dict[Tuple[str, int], Dict[str, float]],
    include_reference_slopes: bool = True,
) -> Tuple[List[object], List[str]]:
    styles = {
        ("hole", 1): {"color": "#e45756", "marker": "o", "linestyle": "-"},
        ("dense", 1): {"color": "#4c78a8", "marker": "s", "linestyle": "--"},
        ("hole", 2): {"color": "#f58518", "marker": "^", "linestyle": "-"},
        ("dense", 2): {"color": "#54a24b", "marker": "D", "linestyle": "--"},
    }
    labels = {
        ("hole", 1): r"Hole design ($s=1$)",
        ("dense", 1): r"Dense reference ($s=1$)",
        ("hole", 2): r"Hole design ($s=2$)",
        ("dense", 2): r"Dense reference ($s=2$)",
    }
    x = np.asarray(h_values, dtype=np.float64)
    handles: List[object] = []
    labels_out: List[str] = []
    for key in (("hole", 1), ("dense", 1), ("hole", 2), ("dense", 2)):
        summary = per_h_summary[key]
        y = np.asarray([summary[float(h)]["mse_mean"] for h in h_values], dtype=np.float64)
        yerr = np.asarray([summary[float(h)]["mse_se"] for h in h_values], dtype=np.float64)
        style = styles[key]
        handle = ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=4.8,
            capsize=3,
            label=labels[key],
        )
        handles.append(handle)
        labels_out.append(labels[key])

    if include_reference_slopes:
        for s, ref_color in ((1, "#222222"), (2, "#7f7f7f")):
            ref_slope = 2 * s
            hole_y = np.asarray([per_h_summary[("hole", s)][float(h)]["mse_mean"] for h in h_values], dtype=np.float64)
            ref_idx = len(h_values) // 2
            ref_const = hole_y[ref_idx] / max(x[ref_idx] ** ref_slope, EPS)
            (line,) = ax.plot(
                x,
                ref_const * np.power(x, ref_slope),
                color=ref_color,
                linestyle=":",
                linewidth=1.8,
                label=rf"Reference slope {2*s}",
            )
            handles.append(line)
            labels_out.append(rf"Reference slope {2*s}")

    ax.text(
        0.03,
        0.04,
        (
            rf"$s=1$: {slope_summary[('hole', 1)]['slope_mean']:.2f}" "\n"
            rf"$s=2$: {slope_summary[('hole', 2)]['slope_mean']:.2f}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Hole radius $h$")
    ax.set_ylabel(r"Local MSE")
    ax.grid(alpha=0.2, which="both")
    return handles, labels_out


def make_fixed_target_main_figure(
    figures_dir: Path,
    h_values: Sequence[float],
    per_h_summary: Dict[str, Dict[float, Dict[str, float]]],
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.3), constrained_layout=True)
    handles, labels_out = plot_fixed_target_axis(ax=ax, h_values=h_values, per_h_summary=per_h_summary)
    save_axes_group_panel(fig, [ax], figures_dir / "exp2_support_gap_combined_panel_b.pdf")
    save_axes_group_panel(fig, [ax], figures_dir / "exp2_fixed_target_hole_enlargement.pdf")
    save_legend_figure(handles, labels_out, figures_dir / "exp2_fixed_target_hole_enlargement_legend.pdf", ncol=2)
    plt.close(fig)


def plot_fixed_target_axis(
    ax: plt.Axes,
    h_values: Sequence[float],
    per_h_summary: Dict[str, Dict[float, Dict[str, float]]],
) -> Tuple[List[object], List[str]]:
    styles = {
        "hole": {"color": "#e45756", "marker": "o", "linestyle": "-"},
        "dense": {"color": "#4c78a8", "marker": "s", "linestyle": "--"},
    }
    labels = {"hole": "Hole design", "dense": "Dense reference"}
    x = np.asarray(h_values, dtype=np.float64)
    handles: List[object] = []
    labels_out: List[str] = []
    for design in ("hole", "dense"):
        y = np.asarray([per_h_summary[design][float(h)]["local_mse_mean"] for h in h_values], dtype=np.float64)
        yerr = np.asarray([per_h_summary[design][float(h)]["local_mse_se"] for h in h_values], dtype=np.float64)
        style = styles[design]
        handle = ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.1,
            markersize=4.8,
            capsize=3,
            label=labels[design],
        )
        handles.append(handle)
        labels_out.append(labels[design])
    ax.set_xlabel(r"Hole radius $h$")
    ax.set_ylabel(r"Local MSE on fixed $G_{\mathrm{fixed}}$")
    ax.grid(alpha=0.2)
    return handles, labels_out


def make_fixed_target_repetition_figure(
    figures_dir: Path,
    fixed_rows: Sequence[Dict[str, float | int | str]],
    h_values: Sequence[float],
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    hole_rows = [row for row in fixed_rows if row["design_type"] == "hole"]
    dense_rows = [row for row in fixed_rows if row["design_type"] == "dense"]
    reps = sorted({int(row["repetition"]) for row in fixed_rows})
    x = np.asarray(h_values, dtype=np.float64)
    for rep in reps:
        hole_curve = [float(next(row["local_mse"] for row in hole_rows if int(row["repetition"]) == rep and math.isclose(float(row["h"]), float(h)) )) for h in h_values]
        dense_curve = [float(next(row["local_mse"] for row in dense_rows if int(row["repetition"]) == rep and math.isclose(float(row["h"]), float(h)) )) for h in h_values]
        ax.plot(x, hole_curve, color="#e45756", alpha=0.10, linewidth=1.0)
        ax.plot(x, dense_curve, color="#4c78a8", alpha=0.10, linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"Hole radius $h$")
    ax.set_ylabel(r"Local MSE")
    ax.set_title("Fixed-target local MSE across repetitions")
    ax.grid(alpha=0.2)
    save_axes_group_panel(fig, [ax], figures_dir / "exp2_fixed_target_repetition_curves.pdf")
    plt.close(fig)


def make_fixed_target_ratio_figure(
    figures_dir: Path,
    h_values: Sequence[float],
    ratio_summary: Dict[float, Dict[str, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    x = np.asarray(h_values, dtype=np.float64)
    y = np.asarray([ratio_summary[float(h)]["mean"] for h in h_values], dtype=np.float64)
    yerr = np.asarray([ratio_summary[float(h)]["se"] for h in h_values], dtype=np.float64)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color="#b279a2",
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=4.6,
        capsize=3,
    )
    ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"Hole radius $h$")
    ax.set_ylabel(r"Hole / dense local MSE")
    ax.set_title("Fixed-target hole/dense local-error ratio")
    ax.grid(alpha=0.2)
    save_axes_group_panel(fig, [ax], figures_dir / "exp2_fixed_target_hole_dense_ratio.pdf")
    plt.close(fig)


def make_fixed_target_support_figure(
    figures_dir: Path,
    h_values: Sequence[float],
    per_h_summary: Dict[str, Dict[float, Dict[str, float]]],
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    x = np.asarray(h_values, dtype=np.float64)
    for design, color, marker, linestyle in (
        ("hole", "#e45756", "o", "-"),
        ("dense", "#4c78a8", "s", "--"),
    ):
        y = np.asarray([per_h_summary[design][float(h)]["median_support_radius_mean"] for h in h_values], dtype=np.float64)
        yerr = np.asarray([per_h_summary[design][float(h)]["median_support_radius_se"] for h in h_values], dtype=np.float64)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=4.6,
            capsize=3,
            label="Hole design" if design == "hole" else "Dense reference",
        )
    ax.set_xlabel(r"Hole radius $h$")
    ax.set_ylabel(r"Median support radius in $G_{\mathrm{fixed}}$")
    ax.set_title("Fixed-target support radius in the local evaluation region")
    ax.grid(alpha=0.2)
    handles, labels_out = ax.get_legend_handles_labels()
    save_axes_group_panel(fig, [ax], figures_dir / "exp2_fixed_target_support_radius.pdf")
    save_legend_figure(handles, labels_out, figures_dir / "exp2_fixed_target_support_radius_legend.pdf", ncol=2)
    plt.close(fig)


def make_combined_support_gap_figure(
    figures_dir: Path,
    h_values_scaled: Sequence[float],
    per_h_summary_scaled: Dict[Tuple[str, int], Dict[float, Dict[str, float]]],
    slope_summary_scaled: Dict[Tuple[str, int], Dict[str, float]],
    h_values_fixed: Sequence[float],
    per_h_summary_fixed: Dict[str, Dict[float, Dict[str, float]]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    plot_scaled_bump_summary_axis(
        ax=axes[0],
        h_values=h_values_scaled,
        per_h_summary=per_h_summary_scaled,
        slope_summary=slope_summary_scaled,
    )
    plot_fixed_target_axis(
        ax=axes[1],
        h_values=h_values_fixed,
        per_h_summary=per_h_summary_fixed,
    )
    fig.savefig(figures_dir / "exp2_support_gap_combined_figure.pdf")
    plt.close(fig)


def save_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_scaled_bump_tables(
    tables_dir: Path,
    table_rows: Sequence[Dict[str, float]],
) -> None:
    csv_rows = [
        {
            "smoothness_s": int(row["smoothness_s"]),
            "predicted_slope_2s": float(row["predicted_slope_2s"]),
            "hole_design_slope": float(row["hole_design_slope"]),
            "dense_design_slope": float(row["dense_design_slope"]),
            "se": float(row["se"]),
        }
        for row in table_rows
    ]
    save_csv(tables_dir / "exp2_scaling_slopes.csv", csv_rows)
    save_csv(tables_dir / "exp2_scaled_bump_results.csv", csv_rows)
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Smoothness $s$ & Predicted slope $2s$ & Hole-design slope & Dense-design slope & SE \\\\",
        "\\midrule",
    ]
    for row in csv_rows:
        lines.append(
            rf"$s={int(row['smoothness_s'])}$ & {row['predicted_slope_2s']:.0f} & "
            rf"{row['hole_design_slope']:.3f} & {row['dense_design_slope']:.3f} & {row['se']:.4f} \\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "exp2_scaling_slopes.tex").write_text("\n".join(lines) + "\n")
    (tables_dir / "exp2_scaled_bump_slope_table.tex").write_text("\n".join(lines) + "\n")


def save_fixed_target_trend_table(
    tables_dir: Path,
    results_dir: Path,
    trend_rows: Sequence[Dict[str, object]],
) -> None:
    save_csv(tables_dir / "exp2_fixed_target_trend_summary.csv", trend_rows)
    save_csv(results_dir / "exp2_fixed_target_trend_summary.csv", trend_rows)
    display_row = next(row for row in trend_rows if row["summary_level"] == "mean_curve")
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Diagnostic & Spearman $\\rho$ & Kendall $\\tau$ & Largest/smallest local MSE & Hole/dense ratio at largest $h$ \\\\",
        "\\midrule",
        (
            "Fixed target hole enlargement & "
            f"{float(display_row['spearman_rho']):.3f} & "
            f"{float(display_row['kendall_tau']):.3f} & "
            f"{float(display_row['largest_to_smallest_ratio']):.3f} & "
            f"{float(display_row['largest_h_hole_dense_ratio']):.3f} \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (tables_dir / "exp2_fixed_target_trend_table.tex").write_text("\n".join(lines) + "\n")


def save_support_gap_main_table(
    tables_dir: Path,
    scaled_table_rows: Sequence[Dict[str, float]],
    fixed_summary_row: Dict[str, object],
) -> None:
    hole_s1 = next(row for row in scaled_table_rows if int(row["smoothness_s"]) == 1)
    hole_s2 = next(row for row in scaled_table_rows if int(row["smoothness_s"]) == 2)
    lines = [
        "\\begin{tabular}{p{2.35cm}p{2.2cm}p{2.4cm}p{3.1cm}p{4.1cm}}",
        "\\toprule",
        "Diagnostic & Target & Quantity & Expected behavior & Observed summary \\\\",
        "\\midrule",
        (
            "Scaled bump & $h^s\\psi((x-x_0)/h)$, $s=1$ & Local MSE vs. $h$ & "
            "Recovers the proof-construction slope $2s=2$ & "
            f"Estimated slope {float(hole_s1['hole_design_slope']):.3f}; dense control slope {float(hole_s1['dense_design_slope']):.3f}. \\\\"
        ),
        (
            "Scaled bump & $h^s\\psi((x-x_0)/h)$, $s=2$ & Local MSE vs. $h$ & "
            "Recovers the proof-construction slope $2s=4$ & "
            f"Estimated slope {float(hole_s2['hole_design_slope']):.3f}; dense control slope {float(hole_s2['dense_design_slope']):.3f}. \\\\"
        ),
        (
            "Fixed target & Smooth sinusoid plus local bump & Local MSE on fixed $G_{\\mathrm{fixed}}$ & "
            "Local error rises as the hole enlarges relative to dense control & "
            f"Spearman $\\rho$ {float(fixed_summary_row['spearman_rho']):.3f}; "
            f"largest/smallest ratio {float(fixed_summary_row['largest_to_smallest_ratio']):.2f}; "
            f"largest-$h$ hole/dense ratio {float(fixed_summary_row['largest_h_hole_dense_ratio']):.2f}. \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (tables_dir / "exp2_support_gap_summary_table.tex").write_text("\n".join(lines) + "\n")


def save_design_stability_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_scaled_bump_theorem_realization(
    args: argparse.Namespace,
    figures_dir: Path,
    tables_dir: Path,
    results_dir: Path,
) -> Dict[str, object]:
    x0 = DEFAULT_X0_SCALED.copy()
    h_values = [float(h) for h in DEFAULT_H_VALUES_SCALED]
    s_values = [int(s) for s in DEFAULT_S_VALUES]
    raw_rows: List[Dict[str, object]] = []
    slope_rows: List[Dict[str, object]] = []
    illustration_designs: Dict[float, np.ndarray] = {}
    jitter_count = 0
    mse_by_design_s: Dict[Tuple[str, int], List[List[float]]] = {("hole", s): [] for s in s_values}
    mse_by_design_s.update({("dense", s): [] for s in s_values})

    for s in s_values:
        check = radial_bump(x0[None, :], h=0.12, s=s, x0=x0)[0]
        assert math.isclose(check, 0.12**s, rel_tol=1e-12, abs_tol=1e-12)

    for rep in range(args.n_reps_scaled):
        rep_rng = np.random.default_rng(args.seed + 10_000 * rep + 17)
        rep_mse: Dict[Tuple[str, int], List[float]] = {("hole", s): [] for s in s_values}
        rep_mse.update({("dense", s): [] for s in s_values})
        for h in h_values:
            X_hole = sample_uniform_with_hole(args.n_train_scaled, h=h, x0=x0, low=-1.0, high=1.0, rng=rep_rng)
            hole_dist = np.linalg.norm(X_hole - x0[None, :], axis=1)
            assert np.all(hole_dist >= h - 1e-12)
            if rep == 0 and h in ILLUSTRATION_H_VALUES:
                illustration_designs[h] = X_hole.copy()
            X_dense = sample_uniform_box(args.n_train_scaled, low=-1.0, high=1.0, rng=rep_rng)
            dense_dist = np.linalg.norm(X_dense - x0[None, :], axis=1)
            n_dense_support = int(np.sum(dense_dist < h))
            X_eval = sample_points_in_ball(
                n=args.n_eval_local_scaled,
                center=x0,
                radius=float(args.rho) * h,
                dim=2,
                rng=rep_rng,
            )
            eval_targets = np.column_stack([radial_bump(X_eval, h=h, s=s, x0=x0) for s in s_values])
            hole_predictions = np.zeros_like(eval_targets)
            for col, s in enumerate(s_values):
                mse_hole = float(np.mean((hole_predictions[:, col] - eval_targets[:, col]) ** 2))
                rep_mse[("hole", s)].append(mse_hole)
                raw_rows.append(
                    {
                        "diagnostic": DIAGNOSTIC_SCALED,
                        "design_type": "hole",
                        "smoothness_s": int(s),
                        "h": float(h),
                        "repetition": int(rep),
                        "mse_local": float(mse_hole),
                        "n_train_inside_support": 0,
                    }
                )

            Y_dense = np.column_stack([radial_bump(X_dense, h=h, s=s, x0=x0) for s in s_values])
            if np.allclose(Y_dense, 0.0):
                dense_predictions = np.zeros_like(eval_targets)
            else:
                model = train_reference_model(
                    X_dense,
                    Y_dense,
                    alpha=float(args.scaled_alpha),
                    gamma=float(args.scaled_gamma),
                )
                alpha_effective = float(model["alpha_effective"])
                if not math.isclose(alpha_effective, float(args.scaled_alpha)):
                    jitter_count += 1
                dense_predictions = predict_reference_model(model, X_eval)

            for col, s in enumerate(s_values):
                mse_dense = float(np.mean((dense_predictions[:, col] - eval_targets[:, col]) ** 2))
                rep_mse[("dense", s)].append(mse_dense)
                raw_rows.append(
                    {
                        "diagnostic": DIAGNOSTIC_SCALED,
                        "design_type": "dense",
                        "smoothness_s": int(s),
                        "h": float(h),
                        "repetition": int(rep),
                        "mse_local": float(mse_dense),
                        "n_train_inside_support": int(n_dense_support),
                    }
                )

        for key, values in rep_mse.items():
            design, s = key
            slope, intercept = fit_loglog_slope(h_values, values)
            slope_rows.append(
                {
                    "diagnostic": DIAGNOSTIC_SCALED,
                    "design_type": design,
                    "smoothness_s": int(s),
                    "repetition": int(rep),
                    "slope": float(slope),
                    "intercept": float(intercept),
                }
            )
            mse_by_design_s[key].append(list(values))

    per_h_summary: Dict[Tuple[str, int], Dict[float, Dict[str, float]]] = {}
    slope_summary: Dict[Tuple[str, int], Dict[str, float]] = {}
    table_rows: List[Dict[str, float]] = []

    for design in ("hole", "dense"):
        for s in s_values:
            values = np.asarray(mse_by_design_s[(design, s)], dtype=np.float64)
            per_h_summary[(design, s)] = {}
            for j, h in enumerate(h_values):
                summary = summarize_array(values[:, j])
                per_h_summary[(design, s)][float(h)] = {
                    "mse_mean": summary["mean"],
                    "mse_std": summary["std"],
                    "mse_se": summary["se"],
                    "n_success": summary["n"],
                }
            cur_slopes = [
                row for row in slope_rows if row["design_type"] == design and int(row["smoothness_s"]) == s
            ]
            slopes = [float(row["slope"]) for row in cur_slopes]
            intercepts = [float(row["intercept"]) for row in cur_slopes]
            slope_info = summarize_array(slopes)
            slope_summary[(design, s)] = {
                "slope_mean": slope_info["mean"],
                "slope_std": slope_info["std"],
                "slope_se": slope_info["se"],
                "intercept_mean": float(np.mean(np.asarray(intercepts, dtype=np.float64))),
            }

    for s in s_values:
        table_rows.append(
            {
                "smoothness_s": float(s),
                "predicted_slope_2s": float(2 * s),
                "hole_design_slope": slope_summary[("hole", s)]["slope_mean"],
                "dense_design_slope": slope_summary[("dense", s)]["slope_mean"],
                "se": slope_summary[("hole", s)]["slope_se"],
            }
        )

    make_support_hole_figure(figures_dir=figures_dir, illustration_designs=illustration_designs, rho=float(args.rho), x0=x0)
    make_loglog_scaling_figure(
        figures_dir=figures_dir,
        h_values=h_values,
        per_h_summary=per_h_summary,
        slope_summary=slope_summary,
    )
    make_scaled_bump_summary_panel(
        figures_dir=figures_dir,
        h_values=h_values,
        per_h_summary=per_h_summary,
        slope_summary=slope_summary,
    )
    save_scaled_bump_tables(tables_dir=tables_dir, table_rows=table_rows)
    save_csv(results_dir / "exp2_raw_results.csv", raw_rows)
    save_csv(results_dir / "exp2_scaled_bump_results.csv", raw_rows)

    return {
        "diagnostic": DIAGNOSTIC_SCALED,
        "h_values": h_values,
        "raw_rows": raw_rows,
        "slope_rows": slope_rows,
        "table_rows": table_rows,
        "per_h_summary": per_h_summary,
        "slope_summary": slope_summary,
        "jitter_count": jitter_count,
    }


def run_fixed_target_hole_enlargement(
    args: argparse.Namespace,
    figures_dir: Path,
    tables_dir: Path,
    results_dir: Path,
) -> Dict[str, object]:
    x0 = DEFAULT_X0_FIXED.copy()
    h_values = [float(h) for h in DEFAULT_H_VALUES_FIXED]
    raw_rows: List[Dict[str, object]] = []
    trend_rows: List[Dict[str, object]] = []
    ratio_by_h: Dict[float, List[float]] = {float(h): [] for h in h_values}
    per_h_values: Dict[str, Dict[float, Dict[str, List[float]]]] = {
        "hole": {float(h): {"local_mse": [], "global_mse": [], "median_support_radius": [], "support_contrast": []} for h in h_values},
        "dense": {float(h): {"local_mse": [], "global_mse": [], "median_support_radius": [], "support_contrast": []} for h in h_values},
    }

    for rep in range(args.n_reps_fixed):
        rep_rng = np.random.default_rng(args.seed + 50_000 + 97 * rep)
        X_local = sample_points_in_ball(
            n=args.n_eval_local_fixed,
            center=x0,
            radius=float(args.fixed_r_eval),
            dim=2,
            rng=rep_rng,
        )
        X_test = sample_uniform_box(args.n_test_global, low=0.0, high=1.0, rng=rep_rng)
        y_local_true = fixed_target(X_local, x0=x0, amplitude=float(args.fixed_bump_amplitude), alpha=float(args.fixed_bump_alpha))
        y_test_true = fixed_target(X_test, x0=x0, amplitude=float(args.fixed_bump_amplitude), alpha=float(args.fixed_bump_alpha))
        hole_curve: List[float] = []
        dense_curve: List[float] = []

        for h in h_values:
            local_mse_by_design: Dict[str, float] = {}
            for design in ("hole", "dense"):
                if design == "hole":
                    X_train = sample_uniform_with_hole(
                        args.n_train_fixed,
                        h=h,
                        x0=x0,
                        low=0.0,
                        high=1.0,
                        rng=rep_rng,
                    )
                else:
                    X_train = sample_uniform_box(args.n_train_fixed, low=0.0, high=1.0, rng=rep_rng)
                y_train_true = fixed_target(
                    X_train,
                    x0=x0,
                    amplitude=float(args.fixed_bump_amplitude),
                    alpha=float(args.fixed_bump_alpha),
                )
                y_train = y_train_true + float(args.fixed_sigma) * rep_rng.normal(size=y_train_true.shape[0])
                model = train_reference_model(
                    X_train,
                    y_train[:, None],
                    alpha=float(args.fixed_lambda),
                    gamma=float(args.fixed_gamma),
                )
                pred_local = predict_reference_model(model, X_local).reshape(-1)
                pred_test = predict_reference_model(model, X_test).reshape(-1)
                local_mse = float(np.mean((pred_local - y_local_true) ** 2))
                global_mse = float(np.mean((pred_test - y_test_true) ** 2))
                support_local = compute_knn_radius(X_local, X_train, k=int(args.k_support))
                q05 = float(np.quantile(support_local, 0.05))
                q95 = float(np.quantile(support_local, 0.95))
                support_contrast = q95 / max(q05, EPS)
                median_support_radius = float(np.median(support_local))

                per_h_values[design][float(h)]["local_mse"].append(local_mse)
                per_h_values[design][float(h)]["global_mse"].append(global_mse)
                per_h_values[design][float(h)]["median_support_radius"].append(median_support_radius)
                per_h_values[design][float(h)]["support_contrast"].append(support_contrast)
                local_mse_by_design[design] = local_mse
                if design == "hole":
                    hole_curve.append(local_mse)
                else:
                    dense_curve.append(local_mse)
                raw_rows.append(
                    {
                        "diagnostic": DIAGNOSTIC_FIXED,
                        "design_type": design,
                        "repetition": int(rep),
                        "h": float(h),
                        "local_mse": local_mse,
                        "global_mse": global_mse,
                        "median_support_radius_local": median_support_radius,
                        "support_contrast_local": support_contrast,
                        "n_train_eff": int(X_train.shape[0]),
                        "local_eval_radius": float(args.fixed_r_eval),
                    }
                )

            hole_dense_ratio = local_mse_by_design["hole"] / max(local_mse_by_design["dense"], EPS)
            ratio_by_h[float(h)].append(hole_dense_ratio)
            raw_rows.append(
                {
                    "diagnostic": DIAGNOSTIC_FIXED,
                    "design_type": "ratio",
                    "repetition": int(rep),
                    "h": float(h),
                    "local_mse": hole_dense_ratio,
                    "global_mse": "",
                    "median_support_radius_local": "",
                    "support_contrast_local": "",
                    "n_train_eff": int(args.n_train_fixed),
                    "local_eval_radius": float(args.fixed_r_eval),
                }
            )

        hole_slope, hole_intercept = fit_loglog_slope(h_values, hole_curve)
        trend_rows.append(
            {
                "diagnostic": DIAGNOSTIC_FIXED,
                "summary_level": "per_rep",
                "repetition": int(rep),
                "spearman_rho": spearman_corr(h_values, hole_curve),
                "kendall_tau": kendall_tau(h_values, hole_curve),
                "largest_to_smallest_ratio": hole_curve[-1] / max(hole_curve[0], EPS),
                "largest_h_hole_dense_ratio": hole_curve[-1] / max(dense_curve[-1], EPS),
                "descriptive_loglog_slope": hole_slope,
                "descriptive_loglog_intercept": hole_intercept,
            }
        )

    per_h_summary: Dict[str, Dict[float, Dict[str, float]]] = {"hole": {}, "dense": {}}
    for design in ("hole", "dense"):
        for h in h_values:
            local_info = summarize_array(per_h_values[design][float(h)]["local_mse"])
            global_info = summarize_array(per_h_values[design][float(h)]["global_mse"])
            support_info = summarize_array(per_h_values[design][float(h)]["median_support_radius"])
            contrast_info = summarize_array(per_h_values[design][float(h)]["support_contrast"])
            per_h_summary[design][float(h)] = {
                "local_mse_mean": local_info["mean"],
                "local_mse_std": local_info["std"],
                "local_mse_se": local_info["se"],
                "global_mse_mean": global_info["mean"],
                "global_mse_std": global_info["std"],
                "global_mse_se": global_info["se"],
                "median_support_radius_mean": support_info["mean"],
                "median_support_radius_se": support_info["se"],
                "support_contrast_mean": contrast_info["mean"],
                "support_contrast_se": contrast_info["se"],
            }

    ratio_summary = {float(h): summarize_array(ratio_by_h[float(h)]) for h in h_values}
    hole_curve_mean = [per_h_summary["hole"][float(h)]["local_mse_mean"] for h in h_values]
    dense_curve_mean = [per_h_summary["dense"][float(h)]["local_mse_mean"] for h in h_values]
    curve_slope, curve_intercept = fit_loglog_slope(h_values, hole_curve_mean)
    trend_rows.append(
        {
            "diagnostic": DIAGNOSTIC_FIXED,
            "summary_level": "mean_curve",
            "repetition": "",
            "spearman_rho": spearman_corr(h_values, hole_curve_mean),
            "kendall_tau": kendall_tau(h_values, hole_curve_mean),
            "largest_to_smallest_ratio": hole_curve_mean[-1] / max(hole_curve_mean[0], EPS),
            "largest_h_hole_dense_ratio": hole_curve_mean[-1] / max(dense_curve_mean[-1], EPS),
            "descriptive_loglog_slope": curve_slope,
            "descriptive_loglog_intercept": curve_intercept,
        }
    )
    per_rep_rows = [row for row in trend_rows if row["summary_level"] == "per_rep"]
    trend_rows.append(
        {
            "diagnostic": DIAGNOSTIC_FIXED,
            "summary_level": "per_rep_median",
            "repetition": "",
            "spearman_rho": float(np.median([float(row["spearman_rho"]) for row in per_rep_rows])),
            "kendall_tau": float(np.median([float(row["kendall_tau"]) for row in per_rep_rows])),
            "largest_to_smallest_ratio": float(np.median([float(row["largest_to_smallest_ratio"]) for row in per_rep_rows])),
            "largest_h_hole_dense_ratio": float(np.median([float(row["largest_h_hole_dense_ratio"]) for row in per_rep_rows])),
            "descriptive_loglog_slope": float(np.median([float(row["descriptive_loglog_slope"]) for row in per_rep_rows])),
            "descriptive_loglog_intercept": float(np.median([float(row["descriptive_loglog_intercept"]) for row in per_rep_rows])),
        }
    )

    make_fixed_target_main_figure(figures_dir=figures_dir, h_values=h_values, per_h_summary=per_h_summary)
    make_fixed_target_repetition_figure(figures_dir=figures_dir, fixed_rows=raw_rows, h_values=h_values)
    make_fixed_target_ratio_figure(figures_dir=figures_dir, h_values=h_values, ratio_summary=ratio_summary)
    make_fixed_target_support_figure(figures_dir=figures_dir, h_values=h_values, per_h_summary=per_h_summary)

    fixed_rows_no_ratio = [row for row in raw_rows if row["design_type"] != "ratio"]
    save_csv(results_dir / "exp2_fixed_target_hole_enlargement.csv", fixed_rows_no_ratio)
    save_fixed_target_trend_table(tables_dir=tables_dir, results_dir=results_dir, trend_rows=trend_rows)

    return {
        "diagnostic": DIAGNOSTIC_FIXED,
        "h_values": h_values,
        "raw_rows": fixed_rows_no_ratio,
        "trend_rows": trend_rows,
        "ratio_summary": ratio_summary,
        "per_h_summary": per_h_summary,
        "mean_curve_summary": next(row for row in trend_rows if row["summary_level"] == "mean_curve"),
    }


def save_combined_summary_json(
    results_dir: Path,
    args: argparse.Namespace,
    scaled: Dict[str, object],
    fixed: Dict[str, object],
) -> None:
    summary = {
        "experiment": "exp2_support_gap_scaling",
        "diagnostics": {
            DIAGNOSTIC_SCALED: {
                "parameters": {
                    "n_train": int(args.n_train_scaled),
                    "n_reps": int(args.n_reps_scaled),
                    "n_eval_local": int(args.n_eval_local_scaled),
                    "rho": float(args.rho),
                    "alpha": float(args.scaled_alpha),
                    "gamma": float(args.scaled_gamma),
                    "h_values": list(DEFAULT_H_VALUES_SCALED),
                    "s_values": list(DEFAULT_S_VALUES),
                    "x0": [float(x) for x in DEFAULT_X0_SCALED],
                },
                "table_rows": scaled["table_rows"],
                "slope_rows": scaled["slope_rows"],
                "jitter_escalations": int(scaled["jitter_count"]),
            },
            DIAGNOSTIC_FIXED: {
                "parameters": {
                    "n_train": int(args.n_train_fixed),
                    "n_reps": int(args.n_reps_fixed),
                    "n_eval_local_fixed": int(args.n_eval_local_fixed),
                    "n_test_global": int(args.n_test_global),
                    "fixed_r_eval": float(args.fixed_r_eval),
                    "sigma": float(args.fixed_sigma),
                    "gamma": float(args.fixed_gamma),
                    "lambda_reg": float(args.fixed_lambda),
                    "hole_h_values": list(DEFAULT_H_VALUES_FIXED),
                    "x0": [float(x) for x in DEFAULT_X0_FIXED],
                    "fixed_bump_amplitude": float(args.fixed_bump_amplitude),
                    "fixed_bump_alpha": float(args.fixed_bump_alpha),
                },
                "trend_summary": fixed["trend_rows"],
                "mean_curve_summary": fixed["mean_curve_summary"],
            },
        },
        "output_paths": {
            "scaled_bump_csv": str((results_dir / "exp2_scaled_bump_results.csv").resolve()),
            "fixed_target_csv": str((results_dir / "exp2_fixed_target_hole_enlargement.csv").resolve()),
            "fixed_target_trend_csv": str((results_dir / "exp2_fixed_target_trend_summary.csv").resolve()),
            "combined_figure": str((results_dir.parent / "figures" / "exp2_support_gap_combined_figure.pdf").resolve()),
            "fixed_target_figure": str((results_dir.parent / "figures" / "exp2_fixed_target_hole_enlargement.pdf").resolve()),
        },
    }
    save_design_stability_json(results_dir / "exp2_support_gap_scaling_summary.json", summary)


def main() -> None:
    args = parse_args()
    apply_fast_mode(args)
    outdir = resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    figures_dir, tables_dir, results_dir = ensure_dirs(outdir)

    scaled = run_scaled_bump_theorem_realization(
        args=args,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        results_dir=results_dir,
    )
    fixed = run_fixed_target_hole_enlargement(
        args=args,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        results_dir=results_dir,
    )
    save_support_gap_main_table(
        tables_dir=tables_dir,
        scaled_table_rows=scaled["table_rows"],
        fixed_summary_row=fixed["mean_curve_summary"],
    )
    make_combined_support_gap_figure(
        figures_dir=figures_dir,
        h_values_scaled=scaled["h_values"],
        per_h_summary_scaled=scaled["per_h_summary"],
        slope_summary_scaled=scaled["slope_summary"],
        h_values_fixed=fixed["h_values"],
        per_h_summary_fixed=fixed["per_h_summary"],
    )
    save_combined_summary_json(results_dir=results_dir, args=args, scaled=scaled, fixed=fixed)


if __name__ == "__main__":
    main()
