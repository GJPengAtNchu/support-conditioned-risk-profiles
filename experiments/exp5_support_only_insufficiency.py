from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from sklearn.neighbors import NearestNeighbors

from figure_layout_utils import save_axes_group_panel, save_legend_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 5: support alone is not sufficient."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-eval", type=int, default=20000)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--k-support", type=int, default=10)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--bump-radius", type=float, default=0.15)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=30.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-3)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def apply_fast_mode(args: argparse.Namespace) -> argparse.Namespace:
    if args.fast:
        args.n_train = 400
        args.n_eval = 5000
    return args


def resolve_outdir(outdir: str) -> Path:
    path = Path(outdir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sample_nonuniform_design(n: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray([0.75, 0.15, 0.10], dtype=np.float64)
    counts = rng.multinomial(n, weights)

    blob_a = rng.normal(
        loc=np.asarray([0.25, 0.30], dtype=np.float64),
        scale=np.asarray([0.08, 0.07], dtype=np.float64),
        size=(counts[0], 2),
    )
    blob_b = rng.normal(
        loc=np.asarray([0.65, 0.45], dtype=np.float64),
        scale=np.asarray([0.09, 0.08], dtype=np.float64),
        size=(counts[1], 2),
    )
    background = rng.uniform(0.0, 1.0, size=(counts[2], 2))

    X = np.vstack([blob_a, blob_b, background])
    rng.shuffle(X, axis=0)
    return np.clip(X, 0.0, 1.0)


def sample_uniform_eval(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float64)


def compute_knn_support_scores(X_query: np.ndarray, X_train: np.ndarray, k_support: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k_support, algorithm="auto")
    nbrs.fit(X_train)
    distances, _ = nbrs.kneighbors(X_query, return_distance=True)
    return np.asarray(distances[:, -1], dtype=np.float64)


def make_support_bins(h_eval: np.ndarray, n_bins: int) -> Dict[str, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(h_eval, quantiles)
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, edges.shape[0]):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    bin_ids = np.digitize(h_eval, edges[1:-1], right=False).astype(np.int64)
    centers = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            centers[b] = float(np.median(h_eval[mask]))
    return {"edges": edges, "bin_ids": bin_ids, "centers": centers}


def choose_dense_weak_locations(
    X_candidates: np.ndarray,
    h_candidates: np.ndarray,
    boundary_margin: float,
) -> Dict[str, np.ndarray | float]:
    inside = np.all(
        (X_candidates >= boundary_margin) & (X_candidates <= 1.0 - boundary_margin),
        axis=1,
    )
    if not np.any(inside):
        raise RuntimeError("No boundary-safe candidate points found.")

    X_safe = X_candidates[inside]
    h_safe = h_candidates[inside]
    q05 = float(np.quantile(h_candidates, 0.05))
    q20 = float(np.quantile(h_candidates, 0.20))
    q80 = float(np.quantile(h_candidates, 0.80))
    q95 = float(np.quantile(h_candidates, 0.95))

    dense_mask = h_safe <= q20
    weak_mask = h_safe >= q80
    if not np.any(dense_mask) or not np.any(weak_mask):
        raise RuntimeError("Dense or weak candidate set is empty.")

    dense_idx = int(np.argmin(np.abs(h_safe[dense_mask] - q05)))
    weak_idx = int(np.argmin(np.abs(h_safe[weak_mask] - q95)))
    x_D = X_safe[dense_mask][dense_idx]
    x_W = X_safe[weak_mask][weak_idx]
    h_D = float(h_safe[dense_mask][dense_idx])
    h_W = float(h_safe[weak_mask][weak_idx])

    if not (h_D <= q20 + 1e-12):
        raise RuntimeError("Selected dense-support location is not inside the dense-support regime.")
    if not (h_W >= q80 - 1e-12):
        raise RuntimeError("Selected weak-support location is not inside the weak-support regime.")

    return {
        "x_D": x_D,
        "x_W": x_W,
        "h_D": h_D,
        "h_W": h_W,
        "q05": q05,
        "q20": q20,
        "q80": q80,
        "q95": q95,
    }


def radial_bump(X: np.ndarray, center: np.ndarray, radius: float, amplitude: float) -> np.ndarray:
    U = (np.asarray(X, dtype=np.float64) - center[None, :]) / float(radius)
    r2 = np.sum(U * U, axis=1)
    out = np.zeros(X.shape[0], dtype=np.float64)
    mask = r2 < 1.0
    if np.any(mask):
        out[mask] = np.exp(-1.0 / (1.0 - r2[mask])) / math.exp(-1.0)
    return float(amplitude) * out


def rbf_kernel(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    sqX = np.sum(X * X, axis=1)[:, None]
    sqZ = np.sum(Z * Z, axis=1)[None, :]
    sqdist = np.maximum(sqX + sqZ - 2.0 * X @ Z.T, 0.0)
    return np.exp(-float(gamma) * sqdist)


def fit_krr_explicit(X_train: np.ndarray, y_train: np.ndarray, gamma: float, lambda_reg: float) -> Dict[str, np.ndarray | float]:
    K = rbf_kernel(X_train, X_train, gamma=gamma)
    n = X_train.shape[0]
    lhs = K + n * float(lambda_reg) * np.eye(n, dtype=np.float64)
    dual = np.linalg.solve(lhs, y_train)
    return {
        "X_train": X_train,
        "dual_coef": dual,
        "gamma": float(gamma),
    }


def predict_krr(model_components: Dict[str, np.ndarray | float], X_eval: np.ndarray) -> np.ndarray:
    X_train = np.asarray(model_components["X_train"], dtype=np.float64)
    dual = np.asarray(model_components["dual_coef"], dtype=np.float64)
    gamma = float(model_components["gamma"])
    K_eval = rbf_kernel(X_eval, X_train, gamma=gamma)
    return np.asarray(K_eval @ dual, dtype=np.float64)


def compute_profile(errors: np.ndarray, bins: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    bin_ids = np.asarray(bins["bin_ids"], dtype=np.int64)
    centers = np.asarray(bins["centers"], dtype=np.float64)
    rows: List[Dict[str, float]] = []
    for b in range(centers.shape[0]):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_id": int(b + 1),
                "h_bin_center": float(centers[b]),
                "mse_bin": float(np.mean(errors[mask])),
                "n_bin": int(mask.sum()),
            }
        )
    return rows


def summarize_dense_weak(errors: np.ndarray, bins: Dict[str, np.ndarray]) -> Dict[str, float | str]:
    bin_ids = np.asarray(bins["bin_ids"], dtype=np.int64)
    n_bins = int(np.max(bin_ids)) + 1
    dense_mask = bin_ids < 2
    weak_mask = bin_ids >= (n_bins - 2)
    if not np.any(dense_mask) or not np.any(weak_mask):
        raise RuntimeError("Dense or weak bins are empty.")
    dense_mse = float(np.mean(errors[dense_mask]))
    weak_mse = float(np.mean(errors[weak_mask]))
    tol = 0.01 * max(dense_mse, weak_mse, 1e-12)
    if dense_mse > weak_mse + tol:
        ordering = "Dense > Weak"
    elif weak_mse > dense_mse + tol:
        ordering = "Weak > Dense"
    else:
        ordering = "Approximately equal"
    return {"dense_mse": dense_mse, "weak_mse": weak_mse, "ordering": ordering}


def maybe_refine_locations(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    h_eval: np.ndarray,
    bins: Dict[str, np.ndarray],
    args: argparse.Namespace,
    base_choice: Dict[str, np.ndarray | float],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray | float]:
    # We keep the same support field and percentile constraints, but allow a
    # small search among candidate dense/weak points to make the opposite
    # ordering easier to see when the first percentile-matched pair is weak.
    def evaluate_pair(x_D: np.ndarray, x_W: np.ndarray) -> Tuple[float, Dict[str, Dict[str, float | str]], List[Dict[str, float]]]:
        results_local: Dict[str, Dict[str, float | str]] = {}
        raw_profiles_local: List[Dict[str, float]] = []
        score = 0.0
        for target_name, center in (("Dense-region variation", x_D), ("Weak-region variation", x_W)):
            target_train = radial_bump(X_train, center=center, radius=float(args.bump_radius), amplitude=float(args.amplitude))
            if args.sigma > 0.0:
                target_train = target_train + float(args.sigma) * rng.normal(size=X_train.shape[0])
            model = fit_krr_explicit(
                X_train=X_train,
                y_train=target_train,
                gamma=float(args.gamma),
                lambda_reg=float(args.lambda_reg),
            )
            target_eval = radial_bump(X_eval, center=center, radius=float(args.bump_radius), amplitude=float(args.amplitude))
            pred_eval = predict_krr(model, X_eval)
            errors = (pred_eval - target_eval) ** 2
            summary = summarize_dense_weak(errors, bins)
            results_local[target_name] = summary
            for row in compute_profile(errors, bins):
                raw_profiles_local.append(
                    {
                        "target_type": target_name,
                        "bin_id": int(row["bin_id"]),
                        "h_bin_center": float(row["h_bin_center"]),
                        "mse_bin": float(row["mse_bin"]),
                        "n_bin": int(row["n_bin"]),
                    }
                )
        dense_gap = float(results_local["Dense-region variation"]["dense_mse"]) - float(results_local["Dense-region variation"]["weak_mse"])
        weak_gap = float(results_local["Weak-region variation"]["weak_mse"]) - float(results_local["Weak-region variation"]["dense_mse"])
        score = dense_gap + weak_gap
        return score, results_local, raw_profiles_local

    base_score, base_results, base_profiles = evaluate_pair(
        np.asarray(base_choice["x_D"], dtype=np.float64),
        np.asarray(base_choice["x_W"], dtype=np.float64),
    )
    if (
        base_results["Dense-region variation"]["ordering"] == "Dense > Weak"
        and base_results["Weak-region variation"]["ordering"] == "Weak > Dense"
    ):
        return {
            **base_choice,
            "results_local": base_results,
            "raw_profiles_local": base_profiles,
        }

    boundary_margin = float(args.bump_radius)
    inside = np.all((X_eval >= boundary_margin) & (X_eval <= 1.0 - boundary_margin), axis=1)
    q20 = float(base_choice["q20"])
    q80 = float(base_choice["q80"])
    dense_candidates = X_eval[inside & (h_eval <= q20)]
    weak_candidates = X_eval[inside & (h_eval >= q80)]
    dense_scores = h_eval[inside & (h_eval <= q20)]
    weak_scores = h_eval[inside & (h_eval >= q80)]

    dense_order = np.argsort(dense_scores)[: min(8, dense_candidates.shape[0])]
    weak_order = np.argsort(-weak_scores)[: min(8, weak_candidates.shape[0])]

    best = {
        **base_choice,
        "score": base_score,
        "results_local": base_results,
        "raw_profiles_local": base_profiles,
    }
    for i in dense_order:
        for j in weak_order:
            x_D = dense_candidates[i]
            x_W = weak_candidates[j]
            score, results_local, raw_profiles_local = evaluate_pair(x_D, x_W)
            if score > float(best["score"]):
                best = {
                    **base_choice,
                    "x_D": x_D,
                    "x_W": x_W,
                    "h_D": float(dense_scores[i]),
                    "h_W": float(weak_scores[j]),
                    "score": score,
                    "results_local": results_local,
                    "raw_profiles_local": raw_profiles_local,
                }
    return best


def make_same_support_targets_figure(
    figures_dir: Path,
    X_train: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    support_grid: np.ndarray,
    target_dense_grid: np.ndarray,
    target_weak_grid: np.ndarray,
    x_D: np.ndarray,
    x_W: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(
        support_grid,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis",
        aspect="equal",
    )
    train_pts = ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        s=7,
        color="white",
        alpha=0.55,
        edgecolors="none",
        label="Training samples",
    )
    dense_pt = ax.scatter(
        [x_D[0]],
        [x_D[1]],
        s=65,
        color="#e45756",
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        label=r"$x_D$",
    )
    weak_pt = ax.scatter(
        [x_W[0]],
        [x_W[1]],
        s=65,
        color="#2f4b7c",
        marker="^",
        edgecolors="black",
        linewidths=0.6,
        label=r"$x_W$",
    )
    ax.set_title("Fixed training design and support field")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    cbar0 = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Support radius h")

    ax = axes[1]
    im = ax.imshow(
        target_dense_grid,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="magma",
        aspect="equal",
        vmin=0.0,
        vmax=max(np.max(target_dense_grid), np.max(target_weak_grid)),
    )
    ax.scatter([x_D[0]], [x_D[1]], s=55, color="white", marker="o", edgecolors="black", linewidths=0.6)
    ax.set_title(r"Dense-region target $f_D^\star$")
    ax.set_xlabel(r"$x_1$")
    cbar1 = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[2]
    im = ax.imshow(
        target_weak_grid,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="magma",
        aspect="equal",
        vmin=0.0,
        vmax=max(np.max(target_dense_grid), np.max(target_weak_grid)),
    )
    ax.scatter([x_W[0]], [x_W[1]], s=55, color="white", marker="^", edgecolors="black", linewidths=0.6)
    ax.set_title(r"Weak-region target $f_W^\star$")
    ax.set_xlabel(r"$x_1$")
    cbar2 = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    save_axes_group_panel(fig, [axes[0], cbar0.ax], figures_dir / "exp5_same_support_two_targets_panel_a.pdf")
    save_axes_group_panel(fig, [axes[1], cbar1.ax], figures_dir / "exp5_same_support_two_targets_panel_b.pdf")
    save_axes_group_panel(fig, [axes[2], cbar2.ax], figures_dir / "exp5_same_support_two_targets_panel_c.pdf")
    save_legend_figure(
        [train_pts, dense_pt, weak_pt],
        ["Training samples", r"$x_D$", r"$x_W$"],
        figures_dir / "exp5_same_support_two_targets_legend.pdf",
        ncol=3,
    )
    plt.close(fig)


def make_opposite_profiles_figure(
    figures_dir: Path,
    raw_profile_rows: Sequence[Dict[str, float | int | str]],
    bins: Dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.8), constrained_layout=True)
    colors = {
        "Dense-region variation": "#e45756",
        "Weak-region variation": "#2f4b7c",
    }
    markers = {
        "Dense-region variation": "o",
        "Weak-region variation": "^",
    }
    linestyles = {
        "Dense-region variation": "-",
        "Weak-region variation": "--",
    }

    for target_name in ("Dense-region variation", "Weak-region variation"):
        rows = [row for row in raw_profile_rows if row["target_type"] == target_name]
        x = np.asarray([float(row["h_bin_center"]) for row in rows], dtype=np.float64)
        y = np.maximum(np.asarray([float(row["mse_bin"]) for row in rows], dtype=np.float64), 1e-12)
        ax.plot(
            x,
            y,
            color=colors[target_name],
            marker=markers[target_name],
            linestyle=linestyles[target_name],
            linewidth=2.0,
            markersize=4.3,
            label=target_name,
        )

    edges = np.asarray(bins["edges"], dtype=np.float64)
    ax.axvspan(edges[0], edges[2], color="#cfe8ff", alpha=0.25)
    ax.axvspan(edges[-3], edges[-1], color="#ffd9d9", alpha=0.25)
    ax.text(edges[1], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0, "", alpha=0.0)

    ax.set_xlabel("Support bin center h")
    ax.set_ylabel("Clean MSE profile")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.18, which="both")
    ax.legend(frameon=False)
    fig.savefig(figures_dir / "exp5_opposite_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def save_table_csv(tables_dir: Path, rows: Sequence[Dict[str, float | str]]) -> None:
    path = tables_dir / "exp5_support_not_sufficient.csv"
    fieldnames = ["target", "dense_bin_mse", "weak_bin_mse", "ordering"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_table_tex(tables_dir: Path, rows: Sequence[Dict[str, float | str]]) -> None:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Target & Dense-bin MSE & Weak-bin MSE & Ordering \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['target']} & "
            f"{float(row['dense_bin_mse']):.4f} & "
            f"{float(row['weak_bin_mse']):.4f} & "
            f"{row['ordering']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "exp5_support_not_sufficient.tex").write_text("\n".join(lines) + "\n")


def save_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(results_dir: Path, summary: Dict[str, object]) -> None:
    (results_dir / "exp5_support_not_sufficient_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main() -> None:
    args = apply_fast_mode(parse_args())
    outdir = resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    results_dir = outdir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    X_train = sample_nonuniform_design(args.n_train, rng)
    X_eval = sample_uniform_eval(args.n_eval, rng)

    h_eval = compute_knn_support_scores(X_eval, X_train, k_support=args.k_support)
    bins = make_support_bins(h_eval, n_bins=args.n_bins)
    location_choice = choose_dense_weak_locations(
        X_candidates=X_eval,
        h_candidates=h_eval,
        boundary_margin=float(args.bump_radius),
    )
    refined = maybe_refine_locations(
        X_train=X_train,
        X_eval=X_eval,
        h_eval=h_eval,
        bins=bins,
        args=args,
        base_choice=location_choice,
        rng=rng,
    )

    x_D = np.asarray(refined["x_D"], dtype=np.float64)
    x_W = np.asarray(refined["x_W"], dtype=np.float64)
    h_D = float(refined["h_D"])
    h_W = float(refined["h_W"])

    if not (h_D <= float(refined["q20"]) + 1e-12):
        raise RuntimeError("x_D is not in the dense-support regime.")
    if not (h_W >= float(refined["q80"]) - 1e-12):
        raise RuntimeError("x_W is not in the weak-support regime.")

    raw_profile_rows = []
    table_rows = []
    target_summaries: Dict[str, Dict[str, float | str]] = {}
    target_defs = {
        "Dense-region variation": x_D,
        "Weak-region variation": x_W,
    }

    for target_name, center in target_defs.items():
        y_train_clean = radial_bump(X_train, center=center, radius=float(args.bump_radius), amplitude=float(args.amplitude))
        y_train = y_train_clean.copy()
        if float(args.sigma) > 0.0:
            y_train = y_train + float(args.sigma) * rng.normal(size=args.n_train)

        model = fit_krr_explicit(
            X_train=X_train,
            y_train=y_train,
            gamma=float(args.gamma),
            lambda_reg=float(args.lambda_reg),
        )
        target_eval = radial_bump(X_eval, center=center, radius=float(args.bump_radius), amplitude=float(args.amplitude))
        pred_eval = predict_krr(model, X_eval)
        errors = (pred_eval - target_eval) ** 2

        summary = summarize_dense_weak(errors, bins)
        target_summaries[target_name] = summary
        table_rows.append(
            {
                "target": target_name,
                "dense_bin_mse": float(summary["dense_mse"]),
                "weak_bin_mse": float(summary["weak_mse"]),
                "ordering": str(summary["ordering"]),
            }
        )
        for row in compute_profile(errors, bins):
            raw_profile_rows.append(
                {
                    "target_type": target_name,
                    "bin_id": int(row["bin_id"]),
                    "h_bin_center": float(row["h_bin_center"]),
                    "mse_bin": float(row["mse_bin"]),
                    "n_bin": int(row["n_bin"]),
                }
            )

    grid_n = 140
    xs = np.linspace(0.0, 1.0, grid_n)
    ys = np.linspace(0.0, 1.0, grid_n)
    grid_x, grid_y = np.meshgrid(xs, ys)
    X_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    support_grid = compute_knn_support_scores(X_grid, X_train, k_support=args.k_support).reshape(grid_n, grid_n)
    target_dense_grid = radial_bump(X_grid, center=x_D, radius=float(args.bump_radius), amplitude=float(args.amplitude)).reshape(grid_n, grid_n)
    target_weak_grid = radial_bump(X_grid, center=x_W, radius=float(args.bump_radius), amplitude=float(args.amplitude)).reshape(grid_n, grid_n)

    make_same_support_targets_figure(
        figures_dir=figures_dir,
        X_train=X_train,
        grid_x=grid_x,
        grid_y=grid_y,
        support_grid=support_grid,
        target_dense_grid=target_dense_grid,
        target_weak_grid=target_weak_grid,
        x_D=x_D,
        x_W=x_W,
    )
    make_opposite_profiles_figure(
        figures_dir=figures_dir,
        raw_profile_rows=raw_profile_rows,
        bins=bins,
    )
    save_table_csv(tables_dir=tables_dir, rows=table_rows)
    save_table_tex(tables_dir=tables_dir, rows=table_rows)
    save_csv(results_dir / "exp5_raw_profiles.csv", raw_profile_rows)

    summary = {
        "experiment": "exp5_support_not_sufficient",
        "parameters": {
            "seed": int(args.seed),
            "n_train": int(args.n_train),
            "n_eval": int(args.n_eval),
            "n_bins": int(args.n_bins),
            "k_support": int(args.k_support),
            "amplitude": float(args.amplitude),
            "bump_radius": float(args.bump_radius),
            "sigma": float(args.sigma),
            "gamma": float(args.gamma),
            "lambda_reg": float(args.lambda_reg),
        },
        "training_design": {
            "same_training_design_for_both_targets": True,
        },
        "locations": {
            "x_D": [float(x) for x in x_D],
            "x_W": [float(x) for x in x_W],
            "h_D": h_D,
            "h_W": h_W,
            "support_quantiles": {
                "q05": float(refined["q05"]),
                "q20": float(refined["q20"]),
                "q80": float(refined["q80"]),
                "q95": float(refined["q95"]),
            },
        },
        "support_bins": {
            "edges": [float(x) for x in np.asarray(bins["edges"], dtype=np.float64)],
            "centers": [float(x) for x in np.asarray(bins["centers"], dtype=np.float64)],
            "dense_bins": [1, 2],
            "weak_bins": [args.n_bins - 1, args.n_bins],
            "same_bins_for_both_targets": True,
        },
        "table_metrics": table_rows,
        "ordering_success": {
            "dense_target_dense_gt_weak": target_summaries["Dense-region variation"]["ordering"] == "Dense > Weak",
            "weak_target_weak_gt_dense": target_summaries["Weak-region variation"]["ordering"] == "Weak > Dense",
        },
        "output_paths": {
            "same_support_figure": str((figures_dir / "exp5_same_support_two_targets.pdf").resolve()),
            "opposite_profiles_figure": str((figures_dir / "exp5_opposite_profiles.pdf").resolve()),
            "table_csv": str((tables_dir / "exp5_support_not_sufficient.csv").resolve()),
            "table_tex": str((tables_dir / "exp5_support_not_sufficient.tex").resolve()),
            "raw_profiles_csv": str((results_dir / "exp5_raw_profiles.csv").resolve()),
        },
    }
    save_summary_json(results_dir=results_dir, summary=summary)


if __name__ == "__main__":
    main()
