from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RULE_ORDER = [
    "Global-MSE selection",
    "GCV",
    "Constrained weak-MSE",
    "Constrained gap-only",
    "Constrained profile-aware",
]

RULE_STYLES = {
    "Global-MSE selection": {"marker": "o", "color": "#111111", "label": "Global-MSE"},
    "GCV": {"marker": "s", "color": "#2f4b7c", "label": "GCV"},
    "Constrained weak-MSE": {"marker": "^", "color": "#e45756", "label": "Weak-MSE"},
    "Constrained gap-only": {"marker": "D", "color": "#54a24b", "label": "Gap-only"},
    "Constrained profile-aware": {"marker": "P", "color": "#7f3c8d", "label": "Profile-aware"},
}

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_by_theta(candidate_rows: Iterable[Dict[str, str]]) -> Dict[Tuple[float, float], Dict[str, float]]:
    buckets: Dict[Tuple[float, float], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in candidate_rows:
        if row["setting"] != "stress" or row["split"] != "val":
            continue
        theta = (float(row["lambda"]), float(row["gamma"]))
        for key in ["global_mse", "weak_mse", "dense_mse", "gap", "slope", "pos_slope", "profile_var"]:
            buckets[theta][key].append(float(row[key]))
    return {
        theta: {key: float(np.mean(values)) for key, values in metrics.items()}
        for theta, metrics in buckets.items()
    }


def mode_theta_for_rule(selection_rows: Iterable[Dict[str, str]], rule: str, tau: float) -> Tuple[float, float]:
    pairs = [
        (float(row["lambda"]), float(row["gamma"]))
        for row in selection_rows
        if row["setting"] == "stress"
        and row["selection_rule"] == rule
        and math.isclose(float(row["tau"]), tau, rel_tol=0.0, abs_tol=1e-15)
    ]
    if not pairs:
        raise RuntimeError(f"No selected rows found for {rule}.")
    counts = Counter(pairs)
    # Match the experiment script tie-breaking: first by count, then by grid order.
    theta_grid = sorted(set(pairs))
    return max(theta_grid, key=lambda theta: (counts.get(theta, 0), -theta_grid.index(theta)))


def build_selected_points(
    selection_rows: List[Dict[str, str]],
    metrics_by_theta: Dict[Tuple[float, float], Dict[str, float]],
    theta_global: Tuple[float, float],
    tau: float,
) -> List[Dict[str, object]]:
    rows_tau = [
        row for row in selection_rows
        if row["setting"] == "stress" and math.isclose(float(row["tau"]), tau, rel_tol=0.0, abs_tol=1e-15)
    ]
    global_rows = [row for row in rows_tau if row["selection_rule"] == "Global-MSE selection"]
    global_mse = float(np.mean([float(row["validation_global_mse"]) for row in global_rows]))
    budget = (1.0 + tau) * global_mse
    points = []
    for rule in RULE_ORDER:
        theta_mode = mode_theta_for_rule(selection_rows, rule, tau)
        rows_rule = [row for row in rows_tau if row["selection_rule"] == rule]
        if not rows_rule:
            raise RuntimeError(f"No selected rows found for {rule}.")
        global_selected = float(np.mean([float(row["validation_global_mse"]) for row in rows_rule]))
        weak_selected = float(np.mean([float(row["validation_weak_mse"]) for row in rows_rule]))
        points.append(
            {
                "rule": rule,
                "lambda": theta_mode[0],
                "gamma": theta_mode[1],
                "global_mse": global_selected,
                "weak_mse": weak_selected,
                "excess_pct": 100.0 * (global_selected / global_mse - 1.0),
                "inside_budget": global_selected <= budget + 1e-15,
            }
        )
    return points


def save_selected_summary(path: Path, selected_points: List[Dict[str, object]]) -> None:
    fields = [
        "selection_rule",
        "lambda",
        "gamma",
        "validation_global_mse",
        "validation_weak_mse",
        "global_mse_excess_percent",
        "inside_tau005_budget",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for point in selected_points:
            writer.writerow(
                {
                    "selection_rule": point["rule"],
                    "lambda": point["lambda"],
                    "gamma": point["gamma"],
                    "validation_global_mse": point["global_mse"],
                    "validation_weak_mse": point["weak_mse"],
                    "global_mse_excess_percent": point["excess_pct"],
                    "inside_tau005_budget": point["inside_budget"],
                }
            )


def add_panel_contents(
    ax_full,
    ax_zoom,
    metrics_by_theta: Dict[Tuple[float, float], Dict[str, float]],
    selected_points: List[Dict[str, object]],
    theta_global: Tuple[float, float],
    tau: float,
) -> List[Line2D]:
    del theta_global
    selected_global = [point for point in selected_points if point["rule"] == "Global-MSE selection"][0]
    global_mse_selected = float(selected_global["global_mse"])
    weak_ref_selected = float(selected_global["weak_mse"])
    budget = (1.0 + tau) * global_mse_selected
    theta_items = sorted(metrics_by_theta.items(), key=lambda item: (item[0][1], item[0][0]))

    xs = np.asarray([metrics["global_mse"] for _, metrics in theta_items])
    ys = np.asarray([metrics["weak_mse"] for _, metrics in theta_items])
    inside = np.asarray([metrics["global_mse"] <= budget + 1e-15 for _, metrics in theta_items])
    ax_full.scatter(
        xs[~inside],
        ys[~inside],
        marker="o",
        s=18,
        color="#9aa0a6",
        alpha=0.20,
        linewidths=0.0,
    )
    ax_full.scatter(
        xs[inside],
        ys[inside],
        marker="o",
        s=34,
        color="#1f77b4",
        alpha=0.72,
        edgecolors="white",
        linewidths=0.35,
    )

    zoom_rows = [(theta, metrics) for theta, metrics in theta_items if metrics["global_mse"] <= 1.15 * global_mse_selected]
    if zoom_rows:
        zx = np.asarray([100.0 * (metrics["global_mse"] / global_mse_selected - 1.0) for _, metrics in zoom_rows])
        zy = np.asarray([metrics["weak_mse"] / weak_ref_selected for _, metrics in zoom_rows])
        zin = np.asarray([metrics["global_mse"] <= budget + 1e-15 for _, metrics in zoom_rows])
        ax_zoom.scatter(
            zx[~zin],
            zy[~zin],
            marker="o",
            s=22,
            color="#9aa0a6",
            alpha=0.22,
            linewidths=0.0,
        )
        ax_zoom.scatter(
            zx[zin],
            zy[zin],
            marker="o",
            s=40,
            color="#1f77b4",
            alpha=0.76,
            edgecolors="white",
            linewidths=0.35,
        )

    ax_full.axvline(budget, color="#222222", linestyle="--", linewidth=1.25)
    ax_zoom.axvline(100.0 * tau, color="#222222", linestyle="--", linewidth=1.25)

    # Visual jitter is used only in the zoomed panel to separate coincident selected modes.
    jitter = {
        "Global-MSE selection": (0.00, 0.000),
        "GCV": (0.25, 0.010),
        "Constrained weak-MSE": (-0.25, -0.010),
        "Constrained gap-only": (-0.25, -0.010),
        "Constrained profile-aware": (0.25, 0.010),
    }
    selected_marker_sizes = {
        "Global-MSE selection": 155,
        "GCV": 145,
        "Constrained weak-MSE": 135,
        "Constrained gap-only": 125,
        "Constrained profile-aware": 165,
    }
    for point in selected_points:
        style = RULE_STYLES[str(point["rule"])]
        x = float(point["global_mse"])
        y = float(point["weak_mse"])
        ax_full.scatter(
            [x],
            [y],
            s=selected_marker_sizes[str(point["rule"])],
            marker=style["marker"],
            facecolors="none",
            edgecolors=style["color"],
            linewidths=2.1,
            zorder=5,
        )
        zx = float(point["excess_pct"])
        zy = float(point["weak_mse"]) / weak_ref_selected
        dx, dy = jitter[str(point["rule"])]
        ax_zoom.scatter(
            [zx + dx],
            [zy + dy],
            s=selected_marker_sizes[str(point["rule"])],
            marker=style["marker"],
            facecolors="none",
            edgecolors=style["color"],
            linewidths=2.0,
            zorder=6,
        )

    ax_full.set_xlabel("Validation global MSE")
    ax_full.set_ylabel("Validation weak-support MSE")
    ax_full.grid(alpha=0.18)

    ax_zoom.set_xlabel("Global validation MSE excess (%)")
    ax_zoom.set_ylabel("Weak-support MSE / global-selected")
    ax_zoom.set_xlim(-25.0, 16.0)
    zoom_y = [
        metrics["weak_mse"] / weak_ref_selected
        for metrics in metrics_by_theta.values()
        if metrics["global_mse"] <= 1.15 * global_mse_selected
    ]
    if zoom_y:
        ymin = max(0.0, min(zoom_y) - 0.08)
        ymax = max(zoom_y) + 0.12
        ax_zoom.set_ylim(ymin, ymax)
    ax_zoom.grid(alpha=0.18)

    rule_handles = [
        Line2D([0], [0], marker=RULE_STYLES[rule]["marker"], color="none", markerfacecolor="none",
               markeredgecolor=RULE_STYLES[rule]["color"], markeredgewidth=1.8, markersize=7,
               label=RULE_STYLES[rule]["label"])
        for rule in RULE_ORDER
    ]
    budget_handle = Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.25, label=r"$5\%$ budget")
    candidate_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markeredgecolor="white",
               markersize=6, label="Inside budget"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9aa0a6", markeredgecolor="none",
               alpha=0.35, markersize=6, label="Outside budget"),
    ]
    return candidate_handles + [budget_handle] + rule_handles


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def plot_twopanel(
    figures_dir: Path,
    metrics_by_theta: Dict[Tuple[float, float], Dict[str, float]],
    selected_points: List[Dict[str, object]],
    theta_global: Tuple[float, float],
    tau: float,
) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), constrained_layout=False)
    ax_full, ax_zoom = axes
    handles = add_panel_contents(ax_full, ax_zoom, metrics_by_theta, selected_points, theta_global, tau)
    ax_full.set_title("(a) Full candidate grid", loc="left")
    ax_zoom.set_title("(b) Zoom near admissible region", loc="left")
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.28, wspace=0.34)

    out_pdf = figures_dir / "exp6_pareto_selection_twopanel.pdf"
    out_png = figures_dir / "exp6_pareto_selection_twopanel.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_split_panels(
    figures_dir: Path,
    metrics_by_theta: Dict[Tuple[float, float], Dict[str, float]],
    selected_points: List[Dict[str, object]],
    theta_global: Tuple[float, float],
    tau: float,
) -> None:
    apply_plot_style()
    fig_full, ax_full = plt.subplots(figsize=(3.7, 3.15), constrained_layout=True)
    fig_zoom, ax_zoom = plt.subplots(figsize=(3.7, 3.15), constrained_layout=True)
    handles = add_panel_contents(ax_full, ax_zoom, metrics_by_theta, selected_points, theta_global, tau)
    fig_full.savefig(figures_dir / "exp6_pareto_selection_twopanel_panel_a.pdf", bbox_inches="tight")
    fig_full.savefig(figures_dir / "exp6_pareto_selection_twopanel_panel_a.png", dpi=240, bbox_inches="tight")
    fig_zoom.savefig(figures_dir / "exp6_pareto_selection_twopanel_panel_b.pdf", bbox_inches="tight")
    fig_zoom.savefig(figures_dir / "exp6_pareto_selection_twopanel_panel_b.png", dpi=240, bbox_inches="tight")
    plt.close(fig_full)
    plt.close(fig_zoom)

    fig_legend = plt.figure(figsize=(7.2, 1.15))
    fig_legend.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig_legend.savefig(figures_dir / "exp6_pareto_selection_twopanel_legend.pdf", bbox_inches="tight")
    fig_legend.savefig(figures_dir / "exp6_pareto_selection_twopanel_legend.png", dpi=240, bbox_inches="tight")
    plt.close(fig_legend)


def main() -> None:
    outdir = PROJECT_ROOT / "outputs"
    results_dir = outdir / "results"
    figures_dir = outdir / "figures"
    tau = 0.05
    candidate_rows = read_csv(results_dir / "exp6_stress_candidate_metrics.csv")
    selection_rows = read_csv(results_dir / "exp6_stress_all_splits.csv")
    metrics_by_theta = mean_by_theta(candidate_rows)
    theta_global = mode_theta_for_rule(selection_rows, "Global-MSE selection", tau)
    selected_points = build_selected_points(selection_rows, metrics_by_theta, theta_global, tau)
    plot_twopanel(figures_dir, metrics_by_theta, selected_points, theta_global, tau)
    plot_split_panels(figures_dir, metrics_by_theta, selected_points, theta_global, tau)
    save_selected_summary(results_dir / "exp6_pareto_selection_twopanel_selected_points.csv", selected_points)

    selected_global = [point for point in selected_points if point["rule"] == "Global-MSE selection"][0]
    budget = (1.0 + tau) * float(selected_global["global_mse"])
    print("Selected candidates shown in exp6_pareto_selection_twopanel:")
    for point in selected_points:
        print(
            f"{point['rule']}: lambda={point['lambda']:.6g}, gamma={point['gamma']:.6g}, "
            f"val_global_mse={point['global_mse']:.8g}, "
            f"val_weak_mse={point['weak_mse']:.8g}, "
            f"global_excess={point['excess_pct']:.2f}%, "
            f"inside_budget={point['inside_budget']}"
        )
    zoom_rules = {"Global-MSE selection", "Constrained weak-MSE", "Constrained gap-only", "Constrained profile-aware"}
    visible = all(
        point["rule"] not in zoom_rules or float(point["global_mse"]) <= 1.15 * metrics_by_theta[theta_global]["global_mse"]
        for point in selected_points
    )
    print(f"Zoomed panel selected constrained/global markers visible: {visible}")
    print(f"Tau=0.05 validation budget: {budget:.8g}")


if __name__ == "__main__":
    main()
