from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_layout_utils import make_style_map, style_triplet

try:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover - the environment already has sklearn
    raise RuntimeError("This script requires scikit-learn.") from exc


DEFAULT_HELDOUT_SIZES = (500, 1000, 2000, 5000, 10000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1: stability of the support-conditioned profile estimator."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-oracle", type=int, default=100000)
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--k-support", type=int, default=10)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="outputs")
    return parser.parse_args()


def resolve_outdir(outdir: str) -> Path:
    path = Path(outdir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sample_nonuniform_design(n: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray([0.70, 0.20, 0.10], dtype=np.float64)
    counts = rng.multinomial(n, weights)

    blob_a = rng.normal(
        loc=np.asarray([0.25, 0.30], dtype=np.float64),
        scale=np.asarray([0.09, 0.08], dtype=np.float64),
        size=(counts[0], 2),
    )
    blob_b = rng.normal(
        loc=np.asarray([0.65, 0.45], dtype=np.float64),
        scale=np.asarray([0.10, 0.09], dtype=np.float64),
        size=(counts[1], 2),
    )
    background = rng.uniform(0.0, 1.0, size=(counts[2], 2))

    X = np.vstack([blob_a, blob_b, background])
    rng.shuffle(X, axis=0)
    return np.clip(X, 0.0, 1.0)


def f_star(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]
    return (
        np.sin(2.0 * math.pi * x1) * np.cos(2.0 * math.pi * x2)
        + 0.75 * np.exp(-35.0 * ((x1 - 0.82) ** 2 + (x2 - 0.78) ** 2))
        + 0.25 * x1
    )


def train_reference_model(X_train: np.ndarray, y_train: np.ndarray) -> KernelRidge:
    model = KernelRidge(alpha=1e-3, kernel="rbf", gamma=20.0)
    model.fit(X_train, y_train)
    return model


def batched_predict(model: KernelRidge, X: np.ndarray, batch_size: int = 5000) -> np.ndarray:
    preds: List[np.ndarray] = []
    for start in range(0, X.shape[0], batch_size):
        stop = min(start + batch_size, X.shape[0])
        preds.append(np.asarray(model.predict(X[start:stop]), dtype=np.float64))
    return np.concatenate(preds, axis=0)


def compute_knn_support_scores(
    X_query: np.ndarray,
    X_train: np.ndarray,
    k: int,
) -> np.ndarray:
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    knn.fit(X_train)
    distances, _ = knn.kneighbors(X_query, return_distance=True)
    return np.asarray(distances[:, -1], dtype=np.float64)


def assign_bins(h_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(h_values, bins=edges[1:-1], right=False).astype(np.int64)


def make_support_bins(h_oracle: np.ndarray, K: int) -> Dict[str, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, K + 1)
    edges = np.quantile(h_oracle, quantiles)
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, edges.shape[0]):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)

    bin_ids = assign_bins(h_oracle, edges)
    centers = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        mask = bin_ids == k
        if np.any(mask):
            centers[k] = float(np.median(h_oracle[mask]))

    return {"edges": edges, "centers": centers}


def compute_profile(
    errors: np.ndarray,
    h_values: np.ndarray,
    bins: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    edges = bins["edges"]
    n_bins = edges.shape[0] - 1
    bin_ids = assign_bins(h_values, edges)
    counts = np.bincount(bin_ids, minlength=n_bins).astype(np.int64)
    sums = np.bincount(bin_ids, weights=errors, minlength=n_bins).astype(np.float64)
    profile = np.full(n_bins, np.nan, dtype=np.float64)
    nonempty = counts > 0
    profile[nonempty] = sums[nonempty] / counts[nonempty]
    return profile, counts


def run_repeated_subsampling(
    oracle_errors: np.ndarray,
    h_oracle: np.ndarray,
    bins: Dict[str, np.ndarray],
    oracle_profile: np.ndarray,
    heldout_sizes: Sequence[int],
    n_reps: int,
    seed: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    results: Dict[int, Dict[str, np.ndarray]] = {}
    n_bins = oracle_profile.shape[0]

    for N in heldout_sizes:
        profiles = np.full((n_reps, n_bins), np.nan, dtype=np.float64)
        counts = np.full((n_reps, n_bins), np.nan, dtype=np.float64)
        abs_errors = np.full((n_reps, n_bins), np.nan, dtype=np.float64)

        for rep in range(n_reps):
            sample_idx = rng.choice(oracle_errors.shape[0], size=N, replace=False)
            profile, count = compute_profile(
                oracle_errors[sample_idx],
                h_oracle[sample_idx],
                bins,
            )
            profiles[rep] = profile
            counts[rep] = count
            abs_errors[rep] = np.abs(profile - oracle_profile)

        results[int(N)] = {
            "profiles": profiles,
            "counts": counts,
            "abs_errors": abs_errors,
        }

    return results


def compute_table_metrics(
    oracle_profile: np.ndarray,
    repeated: Dict[int, Dict[str, np.ndarray]],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for N in sorted(repeated):
        profiles = repeated[N]["profiles"]
        abs_errors = repeated[N]["abs_errors"]

        mean_abs_by_bin = np.nanmean(abs_errors, axis=0)
        weak_bin = abs_errors[:, -1]

        lower = np.nanquantile(profiles, 0.025, axis=0)
        upper = np.nanquantile(profiles, 0.975, axis=0)
        coverage = float(np.mean((oracle_profile >= lower) & (oracle_profile <= upper)))

        rows.append(
            {
                "N": int(N),
                "mean_bin_error": float(np.nanmean(abs_errors)),
                # Use the maximum mean absolute bin error across bins to keep the
                # reported value stable while still highlighting the hardest bin.
                "max_bin_error": float(np.nanmax(mean_abs_by_bin)),
                "weak_bin_error": float(np.nanmean(weak_bin)),
                "ci_coverage": coverage,
            }
        )
    return rows


def make_figures(
    outdir: Path,
    bins: Dict[str, np.ndarray],
    oracle_profile: np.ndarray,
    repeated: Dict[int, Dict[str, np.ndarray]],
) -> None:
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_sizes = [N for N in (500, 2000, 10000) if N in repeated]
    series_styles = make_style_map(plot_sizes)

    centers = bins["centers"]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(
        centers,
        oracle_profile,
        color="black",
        linestyle="-",
        linewidth=2.2,
        marker="X",
        markersize=4.5,
        label="Oracle",
    )
    for N in plot_sizes:
        profiles = repeated[N]["profiles"]
        mean_profile = np.nanmean(profiles, axis=0)
        lower = np.nanquantile(profiles, 0.025, axis=0)
        upper = np.nanquantile(profiles, 0.975, axis=0)
        style = series_styles[N]
        ax.plot(
            centers,
            mean_profile,
            color=str(style["color"]),
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker=str(style["marker"]),
            markersize=4.0,
            label=f"N={N}",
        )
        ax.fill_between(centers, lower, upper, color=str(style["color"]), alpha=0.18)

    ax.set_xlabel("Support score h")
    ax.set_ylabel("Clean squared error profile")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(figures_dir / "exp1_profile_stability.pdf", bbox_inches="tight")
    plt.close(fig)

    all_counts = []
    all_abs_errors = []
    for N in sorted(repeated):
        counts = repeated[N]["counts"].reshape(-1)
        abs_errors = repeated[N]["abs_errors"].reshape(-1)
        valid = np.isfinite(counts) & np.isfinite(abs_errors) & (counts > 0) & (abs_errors > 0)
        all_counts.append(counts[valid])
        all_abs_errors.append(abs_errors[valid])

    flat_counts = np.concatenate(all_counts, axis=0)
    flat_abs_errors = np.concatenate(all_abs_errors, axis=0)

    slope, intercept = np.polyfit(np.log(flat_counts), np.log(flat_abs_errors), deg=1)
    x_fit = np.linspace(flat_counts.min(), flat_counts.max(), 300)
    y_fit = np.exp(intercept) * np.power(x_fit, slope)

    ref_anchor_x = float(np.median(flat_counts))
    ref_anchor_y = float(np.median(flat_abs_errors))
    ref_const = ref_anchor_y * math.sqrt(ref_anchor_x)
    y_ref = ref_const / np.sqrt(x_fit)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.scatter(
        flat_counts,
        flat_abs_errors,
        s=12,
        alpha=0.10,
        color="#4c78a8",
        edgecolors="none",
        label="All bins and repetitions",
    )
    ax.plot(
        x_fit,
        y_fit,
        color=str(style_triplet(0)["color"]),
        linewidth=2.2,
        linestyle=style_triplet(0)["linestyle"],
        marker=str(style_triplet(0)["marker"]),
        markersize=3.8,
        markevery=30,
        label=f"Fitted slope = {slope:.2f}",
    )
    ax.plot(
        x_fit,
        y_ref,
        color=str(style_triplet(1)["color"]),
        linewidth=1.8,
        linestyle=style_triplet(1)["linestyle"],
        marker=str(style_triplet(1)["marker"]),
        markersize=3.8,
        markevery=34,
        label="Reference slope = -0.50",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective bin size N_k")
    ax.set_ylabel("Absolute bin error")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False)
    fig.savefig(figures_dir / "exp1_bin_error_vs_size.pdf", bbox_inches="tight")
    plt.close(fig)


def save_tables(outdir: Path, rows: Sequence[Dict[str, float]]) -> None:
    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "exp1_profile_stability.csv"
    header = ["Held-out size N", "Mean bin error", "Max bin error", "Weak-bin error", "CI coverage"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(int(row["N"])),
                    f"{row['mean_bin_error']:.6f}",
                    f"{row['max_bin_error']:.6f}",
                    f"{row['weak_bin_error']:.6f}",
                    f"{row['ci_coverage']:.3f}",
                ]
            )
        )
    csv_path.write_text("\n".join(lines) + "\n")

    tex_lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Held-out size $N$ & Mean bin error & Max bin error & Weak-bin error & CI coverage \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{int(row['N'])} & "
            f"{row['mean_bin_error']:.4f} & "
            f"{row['max_bin_error']:.4f} & "
            f"{row['weak_bin_error']:.4f} & "
            f"{row['ci_coverage']:.2f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "exp1_profile_stability.tex").write_text("\n".join(tex_lines) + "\n")


def save_summary_json(
    outdir: Path,
    args: argparse.Namespace,
    oracle_profile: np.ndarray,
    bins: Dict[str, np.ndarray],
    table_rows: Sequence[Dict[str, float]],
) -> None:
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "exp1_profile_stability",
        "parameters": {
            "seed": int(args.seed),
            "n_train": int(args.n_train),
            "n_oracle": int(args.n_oracle),
            "n_reps": int(args.n_reps),
            "k_support": int(args.k_support),
            "n_bins": int(args.n_bins),
            "heldout_sizes": list(DEFAULT_HELDOUT_SIZES),
            "noise_sigma": 0.05,
            "kernel_ridge_alpha": 1e-3,
            "kernel_ridge_gamma": 20.0,
        },
        "oracle_profile": [float(x) for x in oracle_profile],
        "support_bin_centers": [float(x) for x in bins["centers"]],
        "table_metrics": [
            {
                "N": int(row["N"]),
                "mean_bin_error": float(row["mean_bin_error"]),
                "max_bin_error": float(row["max_bin_error"]),
                "weak_bin_error": float(row["weak_bin_error"]),
                "ci_coverage": float(row["ci_coverage"]),
            }
            for row in table_rows
        ],
        "output_paths": {
            "figure_profile": str((outdir / "figures" / "exp1_profile_stability.pdf").resolve()),
            "figure_bin_error": str((outdir / "figures" / "exp1_bin_error_vs_size.pdf").resolve()),
            "table_csv": str((outdir / "tables" / "exp1_profile_stability.csv").resolve()),
            "table_tex": str((outdir / "tables" / "exp1_profile_stability.tex").resolve()),
        },
    }
    (results_dir / "exp1_profile_stability_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    outdir = resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    X_train = sample_nonuniform_design(args.n_train, rng)
    y_train = f_star(X_train) + 0.05 * rng.normal(size=args.n_train)

    model = train_reference_model(X_train, y_train)

    X_oracle = sample_nonuniform_design(args.n_oracle, rng)
    y_oracle_true = f_star(X_oracle)
    y_oracle_pred = batched_predict(model, X_oracle, batch_size=5000)
    oracle_errors = np.square(y_oracle_pred - y_oracle_true)
    h_oracle = compute_knn_support_scores(X_oracle, X_train, k=args.k_support)

    bins = make_support_bins(h_oracle, K=args.n_bins)
    oracle_profile, _ = compute_profile(oracle_errors, h_oracle, bins)

    repeated = run_repeated_subsampling(
        oracle_errors=oracle_errors,
        h_oracle=h_oracle,
        bins=bins,
        oracle_profile=oracle_profile,
        heldout_sizes=DEFAULT_HELDOUT_SIZES,
        n_reps=args.n_reps,
        seed=args.seed + 1,
    )

    table_rows = compute_table_metrics(oracle_profile=oracle_profile, repeated=repeated)
    make_figures(outdir=outdir, bins=bins, oracle_profile=oracle_profile, repeated=repeated)
    save_tables(outdir=outdir, rows=table_rows)
    save_summary_json(
        outdir=outdir,
        args=args,
        oracle_profile=oracle_profile,
        bins=bins,
        table_rows=table_rows,
    )


if __name__ == "__main__":
    main()
