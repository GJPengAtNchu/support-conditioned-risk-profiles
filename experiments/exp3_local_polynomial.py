from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
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
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from exp3_rate_controlled import run_rate_controlled
from figure_layout_utils import make_style_map, save_axes_group_panel, save_legend_figure


DEFAULT_S_VALUES = (1, 2)
DEFAULT_K_VALUES = (5, 10, 20, 40, 80)
DEFAULT_SIGMA_VALUES = (0.0, 0.03, 0.08, 0.15)
PROFILE_PLOT_K = (10, 40, 80)
PROFILE_PLOT_SIGMA = (0.0, 0.15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: local polynomial upper bound."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--setting",
        type=str,
        default="all",
        choices=("all", "legacy", "rate_controlled"),
        help="Run the legacy random-design diagnostic, the new rate-controlled diagnostic, or both.",
    )
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=4000)
    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--approx-k", type=int, default=80)
    parser.add_argument("--approx-reps", type=int, default=200)
    parser.add_argument("--variance-design-reps", type=int, default=100)
    parser.add_argument("--variance-noise-reps", type=int, default=100)
    parser.add_argument("--two-term-design-reps", type=int, default=60)
    parser.add_argument("--two-term-noise-reps", type=int, default=100)
    parser.add_argument(
        "--rate-geometry",
        type=str,
        default="half_ball",
        choices=("half_ball", "cone"),
        help="Controlled local-neighborhood geometry for the rate diagnostic.",
    )
    parser.add_argument("--disable-cone-fallback", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-ablation", dest="run_ablation", action="store_true")
    parser.add_argument("--use-diagnostic-targets", dest="use_diagnostic_targets", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.set_defaults(run_ablation=True, use_diagnostic_targets=True)
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


def f_star(X: np.ndarray, s: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]
    if int(s) == 1:
        return np.sin(2.0 * math.pi * x1) + 0.5 * np.cos(2.0 * math.pi * x2) + 0.25 * x1
    if int(s) == 2:
        return (
            np.sin(2.0 * math.pi * x1) * np.cos(2.0 * math.pi * x2)
            + 0.5 * np.exp(-20.0 * ((x1 - 0.75) ** 2 + (x2 - 0.70) ** 2))
            + 0.25 * x1 * x2
        )
    raise ValueError(f"Unsupported smoothness s={s}")


def f_star_diagnostic(X: np.ndarray, s: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray([0.75, 0.70], dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]
    if int(s) == 1:
        delta = 1e-2
        radial = np.sqrt(np.sum((X - c[None, :]) ** 2, axis=1) + delta**2) - delta
        return radial / 1.05
    if int(s) == 2:
        return ((x1 - c[0]) ** 2 + 0.7 * (x2 - c[1]) ** 2 + 0.25 * x1) / 1.2
    raise ValueError(f"Unsupported smoothness s={s}")


def build_poly_features(U: np.ndarray, degree: int) -> np.ndarray:
    if degree == 0:
        return np.ones((U.shape[0], U.shape[1], 1), dtype=np.float64)
    if degree == 1:
        ones = np.ones((U.shape[0], U.shape[1], 1), dtype=np.float64)
        return np.concatenate([ones, U], axis=2)
    raise ValueError(f"Unsupported polynomial degree={degree}")


def make_quantile_bins(h_values: np.ndarray, n_bins: int) -> Dict[str, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(h_values, quantiles)
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, edges.shape[0]):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)

    bin_ids = np.digitize(h_values, edges[1:-1], right=False).astype(np.int64)
    centers = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            centers[b] = float(np.median(h_values[mask]))
    return {"edges": edges, "bin_ids": bin_ids, "centers": centers}


def compute_binned_profile(
    errors: np.ndarray,
    h_values: np.ndarray,
    bins: Dict[str, np.ndarray],
) -> List[Dict[str, float]]:
    _ = h_values  # kept for signature clarity
    bin_ids = np.asarray(bins["bin_ids"], dtype=np.int64)
    centers = np.asarray(bins["centers"], dtype=np.float64)
    n_bins = centers.shape[0]
    rows: List[Dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        mse_bin = float(np.mean(errors[mask]))
        rows.append(
            {
                "bin_id": int(b + 1),
                "count": int(mask.sum()),
                "h_bin_center": float(centers[b]),
                "mse_bin": mse_bin,
            }
        )
    return rows


def compute_knn_radius(distances: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(distances[:, int(k) - 1], dtype=np.float64)


def predict_local_polynomial(
    y_train: np.ndarray,
    neighbor_indices: np.ndarray,
    degree: int,
    linear_weights: np.ndarray | None = None,
) -> np.ndarray:
    y_local = np.asarray(y_train[neighbor_indices], dtype=np.float64)
    if degree == 0:
        return np.mean(y_local, axis=1)
    if degree == 1:
        if linear_weights is None:
            raise ValueError("linear_weights are required for degree 1 local polynomial prediction.")
        return np.sum(linear_weights * y_local, axis=1)
    raise ValueError(f"Unsupported degree={degree}")


def predict_nearest_neighbor(y_train: np.ndarray, nn_indices: np.ndarray) -> np.ndarray:
    return np.asarray(y_train[nn_indices[:, 0]], dtype=np.float64)


def prepare_local_geometry(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    k_values: Sequence[int],
    n_bins: int,
    ridge: float,
) -> Dict[int, Dict[str, np.ndarray]]:
    max_k = int(max(k_values))
    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm="auto")
    nbrs.fit(X_train)
    distances, indices = nbrs.kneighbors(X_eval, return_distance=True)

    geometry: Dict[int, Dict[str, np.ndarray]] = {}
    for k in k_values:
        idx_k = np.asarray(indices[:, :k], dtype=np.int64)
        h_k = compute_knn_radius(distances, k=k)
        bins = make_quantile_bins(h_k, n_bins=n_bins)

        X_neighbors = X_train[idx_k]
        U = X_neighbors - X_eval[:, None, :]

        linear_weights = None
        if k >= 2:
            Phi = build_poly_features(U, degree=1)
            gram = np.einsum("nki,nkj->nij", Phi, Phi)
            gram[:, np.arange(gram.shape[1]), np.arange(gram.shape[2])] += ridge
            inv_gram = np.linalg.inv(gram)
            intercept_direction = inv_gram[:, :, 0]
            linear_weights = np.einsum("nkp,np->nk", Phi, intercept_direction)

        geometry[int(k)] = {
            "neighbor_indices": idx_k,
            "h_values": h_k,
            "bin_ids": np.asarray(bins["bin_ids"], dtype=np.int64),
            "bin_edges": np.asarray(bins["edges"], dtype=np.float64),
            "bin_centers": np.asarray(bins["centers"], dtype=np.float64),
            "linear_weights": linear_weights,
        }
    return geometry


def run_experiment_rows(
    *,
    seed: int,
    n_train: int,
    n_eval: int,
    n_reps: int,
    n_bins: int,
    ridge: float,
    sigma_values: Sequence[float],
    target_fn,
    target_label: str,
) -> List[Dict[str, float | int | str]]:
    raw_rows: List[Dict[str, float | int | str]] = []

    for rep in range(n_reps):
        rng = np.random.default_rng(seed + 1000 * rep + 7)
        X_train = sample_nonuniform_design(n_train, rng)
        X_eval = sample_nonuniform_design(n_eval, rng)

        geometry = prepare_local_geometry(
            X_train=X_train,
            X_eval=X_eval,
            k_values=DEFAULT_K_VALUES,
            n_bins=n_bins,
            ridge=float(ridge),
        )

        nn_indices = geometry[int(min(DEFAULT_K_VALUES))]["neighbor_indices"][:, :1]
        targets_eval = {s: target_fn(X_eval, s=s) for s in DEFAULT_S_VALUES}
        targets_train_clean = {s: target_fn(X_train, s=s) for s in DEFAULT_S_VALUES}

        for s in DEFAULT_S_VALUES:
            degree = int(math.ceil(s) - 1)
            base_noise = rng.normal(size=n_train)
            for sigma in sigma_values:
                y_train = targets_train_clean[s] + float(sigma) * base_noise
                nn_predictions = predict_nearest_neighbor(y_train=y_train, nn_indices=nn_indices)

                for k in DEFAULT_K_VALUES:
                    h_values = geometry[k]["h_values"]
                    bins = {
                        "bin_ids": geometry[k]["bin_ids"],
                        "centers": geometry[k]["bin_centers"],
                    }
                    neighbor_indices = geometry[k]["neighbor_indices"]
                    lp_predictions = predict_local_polynomial(
                        y_train=y_train,
                        neighbor_indices=neighbor_indices,
                        degree=degree,
                        linear_weights=geometry[k]["linear_weights"],
                    )

                    for method, preds in (
                        ("local_poly", lp_predictions),
                        ("nearest_neighbor", nn_predictions),
                    ):
                        errors = (preds - targets_eval[s]) ** 2
                        binned = compute_binned_profile(errors=errors, h_values=h_values, bins=bins)
                        for row in binned:
                            raw_rows.append(
                                {
                                    "target_label": target_label,
                                    "method": method,
                                    "smoothness_s": int(s),
                                    "k": int(k),
                                    "sigma": float(sigma),
                                    "repetition": int(rep),
                                    "bin_id": int(row["bin_id"]),
                                    "h_bin_center": float(row["h_bin_center"]),
                                    "mse_bin": float(row["mse_bin"]),
                                    "h_power_term": float(row["h_bin_center"] ** (2 * s)),
                                    "variance_term": float((sigma ** 2) / k),
                                    "count": int(row["count"]),
                                }
                            )
    return raw_rows


def fit_rate_model(
    raw_rows: Sequence[Dict[str, float | int | str]],
    method: str,
    s: int,
) -> Dict[str, float]:
    subset = [
        row for row in raw_rows
        if row["method"] == method and int(row["smoothness_s"]) == int(s)
    ]
    if not subset:
        raise ValueError(f"No rows available for method={method}, s={s}")

    X = np.asarray(
        [
            [float(row["h_power_term"]), float(row["variance_term"])]
            for row in subset
        ],
        dtype=np.float64,
    )
    y = np.asarray([float(row["mse_bin"]) for row in subset], dtype=np.float64)

    try:
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X, y)
        coef = np.asarray(reg.coef_, dtype=np.float64)
    except TypeError:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        coef = np.asarray(coef, dtype=np.float64)

    y_pred = X @ coef
    sse = float(np.sum((y - y_pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0

    return {
        "A": float(coef[0]),
        "B": float(coef[1]),
        "R2": float(r2),
        "n_obs": int(y.shape[0]),
        "predicted": y_pred.tolist(),
        "observed": y.tolist(),
    }


def fit_loglog_slope(x_values: np.ndarray, y_values: np.ndarray) -> Dict[str, float]:
    x = np.maximum(np.asarray(x_values, dtype=np.float64), 1e-12)
    y = np.maximum(np.asarray(y_values, dtype=np.float64), 1e-12)
    log_x = np.log(x)
    log_y = np.log(y)
    X = np.column_stack([np.ones(log_x.shape[0], dtype=np.float64), log_x])
    coef, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    fitted = X @ coef
    sse = float(np.sum((log_y - fitted) ** 2))
    sst = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0
    return {"intercept": intercept, "slope": slope, "R2": r2}


def fit_no_intercept_line(x_values: np.ndarray, y_values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    denom = float(np.dot(x, x))
    coef = float(np.dot(x, y) / denom) if denom > 0.0 else 0.0
    fitted = coef * x
    sse = float(np.sum((y - fitted) ** 2))
    sst0 = float(np.sum(y ** 2))
    # Use the no-intercept/uncentered R^2 because the variance diagnostic is
    # intentionally constrained to pass through the origin at sigma=0.
    r2 = 1.0 - sse / sst0 if sst0 > 0.0 else 1.0
    return {"coef": coef, "R2": r2, "sse": sse, "sst0": sst0}


def aggregate_profile_rows(
    raw_rows: Sequence[Dict[str, float | int | str]],
    s: int,
    method: str,
    k: int,
    sigma: float,
) -> List[Dict[str, float]]:
    subset = [
        row for row in raw_rows
        if row["method"] == method
        and int(row["smoothness_s"]) == int(s)
        and int(row["k"]) == int(k)
        and math.isclose(float(row["sigma"]), float(sigma), rel_tol=0.0, abs_tol=1e-12)
    ]
    if not subset:
        return []

    bin_ids = sorted(set(int(row["bin_id"]) for row in subset))
    out: List[Dict[str, float]] = []
    for b in bin_ids:
        cur = [row for row in subset if int(row["bin_id"]) == b]
        out.append(
            {
                "bin_id": int(b),
                "h_bin_center": float(np.mean([float(row["h_bin_center"]) for row in cur])),
                "mse_bin": float(np.mean([float(row["mse_bin"]) for row in cur])),
            }
        )
    return out


def summarize_approximation_slopes(
    raw_rows: Sequence[Dict[str, float | int | str]],
) -> Tuple[
    Dict[Tuple[str, int], Dict[str, float]],
    List[Dict[str, float | int | str]],
]:
    grouped: Dict[Tuple[str, int, int, int], List[Dict[str, float | int | str]]] = {}
    for row in raw_rows:
        if not math.isclose(float(row["sigma"]), 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        key = (
            str(row["method"]),
            int(row["smoothness_s"]),
            int(row["k"]),
            int(row["repetition"]),
        )
        grouped.setdefault(key, []).append(row)

    per_group_rows: List[Dict[str, float | int | str]] = []
    summary: Dict[Tuple[str, int], Dict[str, float]] = {}
    for (method, s, k, rep), rows in grouped.items():
        x = np.asarray([float(row["h_bin_center"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["mse_bin"]) for row in rows], dtype=np.float64)
        fit = fit_loglog_slope(x, y)
        per_group_rows.append(
            {
                "method": method,
                "smoothness_s": s,
                "k": k,
                "repetition": rep,
                "approx_slope": float(fit["slope"]),
                "approx_r2": float(fit["R2"]),
            }
        )

    for method in ("local_poly", "nearest_neighbor"):
        for s in DEFAULT_S_VALUES:
            slopes = np.asarray(
                [
                    float(row["approx_slope"])
                    for row in per_group_rows
                    if row["method"] == method and int(row["smoothness_s"]) == int(s)
                ],
                dtype=np.float64,
            )
            if slopes.size == 0:
                raise ValueError(f"No approximation slopes available for method={method}, s={s}")
            summary[(method, s)] = {
                "mean_slope": float(np.mean(slopes)),
                "std_slope": float(np.std(slopes, ddof=1)) if slopes.size > 1 else 0.0,
                "se_slope": float(np.std(slopes, ddof=1) / math.sqrt(slopes.size)) if slopes.size > 1 else 0.0,
                "n_fits": int(slopes.size),
            }
    return summary, per_group_rows


def compute_excess_error_rows(
    raw_rows: Sequence[Dict[str, float | int | str]],
) -> List[Dict[str, float | int | str]]:
    sigma0_map: Dict[Tuple[str, int, int, int, int], float] = {}
    h_center_map: Dict[Tuple[str, int, int, int, int], float] = {}
    for row in raw_rows:
        if math.isclose(float(row["sigma"]), 0.0, rel_tol=0.0, abs_tol=1e-12):
            key = (
                str(row["method"]),
                int(row["smoothness_s"]),
                int(row["k"]),
                int(row["repetition"]),
                int(row["bin_id"]),
            )
            sigma0_map[key] = float(row["mse_bin"])
            h_center_map[key] = float(row["h_bin_center"])

    excess_rows: List[Dict[str, float | int | str]] = []
    for row in raw_rows:
        sigma = float(row["sigma"])
        if math.isclose(sigma, 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        key = (
            str(row["method"]),
            int(row["smoothness_s"]),
            int(row["k"]),
            int(row["repetition"]),
            int(row["bin_id"]),
        )
        if key not in sigma0_map:
            continue
        baseline = sigma0_map[key]
        excess_rows.append(
            {
                "method": str(row["method"]),
                "smoothness_s": int(row["smoothness_s"]),
                "k": int(row["k"]),
                "sigma": sigma,
                "repetition": int(row["repetition"]),
                "bin_id": int(row["bin_id"]),
                "h_bin_center": float(h_center_map[key]),
                "baseline_mse_bin": float(baseline),
                "mse_bin": float(row["mse_bin"]),
                "excess_error": float(float(row["mse_bin"]) - baseline),
                "variance_term": float(row["variance_term"]),
            }
        )
    return excess_rows


def summarize_ablation_approximation(
    raw_rows: Sequence[Dict[str, float | int | str]],
    min_valid_bins: int = 4,
) -> Tuple[
    Dict[Tuple[str, int], Dict[str, float]],
    List[Dict[str, float | int | str]],
    List[Dict[str, float | int | str]],
]:
    grouped: Dict[Tuple[str, int, int, int], List[Dict[str, float | int | str]]] = {}
    for row in raw_rows:
        if not math.isclose(float(row["sigma"]), 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        key = (
            str(row["method"]),
            int(row["smoothness_s"]),
            int(row["k"]),
            int(row["repetition"]),
        )
        grouped.setdefault(key, []).append(row)

    fit_rows: List[Dict[str, float | int | str]] = []
    curve_rows: List[Dict[str, float | int | str]] = []
    for (method, s, k, rep), rows in grouped.items():
        valid = [
            row
            for row in rows
            if float(row["mse_bin"]) > 0.0
            and float(row["h_bin_center"]) > 0.0
            and int(row["count"]) > 0
        ]
        if len(valid) < min_valid_bins:
            warnings.warn(
                f"Approximation ablation has only {len(valid)} valid bins for method={method}, s={s}, k={k}, rep={rep}.",
                RuntimeWarning,
            )
            continue
        x = np.asarray([float(row["h_bin_center"]) for row in valid], dtype=np.float64)
        y = np.asarray([float(row["mse_bin"]) for row in valid], dtype=np.float64)
        fit = fit_loglog_slope(x, y)
        fit_rows.append(
            {
                "method": method,
                "smoothness_s": s,
                "k": k,
                "repetition": rep,
                "approx_slope": float(fit["slope"]),
                "approx_r2": float(fit["R2"]),
                "valid_bins": int(len(valid)),
            }
        )

    for method in ("local_poly", "nearest_neighbor"):
        for s in DEFAULT_S_VALUES:
            for k in DEFAULT_K_VALUES:
                subset = [
                    row
                    for row in raw_rows
                    if row["method"] == method
                    and int(row["smoothness_s"]) == int(s)
                    and int(row["k"]) == int(k)
                    and math.isclose(float(row["sigma"]), 0.0, rel_tol=0.0, abs_tol=1e-12)
                ]
                if not subset:
                    continue
                for bin_id in sorted(set(int(row["bin_id"]) for row in subset)):
                    cur = [row for row in subset if int(row["bin_id"]) == bin_id]
                    curve_rows.append(
                        {
                            "method": method,
                            "smoothness_s": int(s),
                            "k": int(k),
                            "bin_id": int(bin_id),
                            "h_bin_center": float(np.mean([float(row["h_bin_center"]) for row in cur])),
                            "mse_bin": float(np.mean([float(row["mse_bin"]) for row in cur])),
                            "count": int(np.mean([int(row["count"]) for row in cur])),
                        }
                    )

    summary: Dict[Tuple[str, int], Dict[str, float]] = {}
    for method in ("local_poly", "nearest_neighbor"):
        for s in DEFAULT_S_VALUES:
            subset = [
                row
                for row in fit_rows
                if row["method"] == method and int(row["smoothness_s"]) == int(s)
            ]
            if not subset:
                raise ValueError(f"No approximation ablation fits available for method={method}, s={s}")
            slopes = np.asarray([float(row["approx_slope"]) for row in subset], dtype=np.float64)
            r2s = np.asarray([float(row["approx_r2"]) for row in subset], dtype=np.float64)
            valid_bins = np.asarray([int(row["valid_bins"]) for row in subset], dtype=np.float64)
            summary[(method, s)] = {
                "mean_slope": float(np.mean(slopes)),
                "std_slope": float(np.std(slopes, ddof=1)) if slopes.size > 1 else 0.0,
                "se_slope": float(np.std(slopes, ddof=1) / math.sqrt(slopes.size)) if slopes.size > 1 else 0.0,
                "mean_r2": float(np.mean(r2s)),
                "median_r2": float(np.median(r2s)),
                "mean_valid_bins": float(np.mean(valid_bins)),
                "n_fits": int(slopes.size),
            }
    return summary, fit_rows, curve_rows


def summarize_ablation_variance(
    raw_rows: Sequence[Dict[str, float | int | str]],
) -> Tuple[
    Dict[Tuple[str, int], Dict[str, float]],
    List[Dict[str, float | int | str]],
    List[Dict[str, float | int | str]],
]:
    excess_rows = compute_excess_error_rows(raw_rows=raw_rows)
    rep_grouped: Dict[Tuple[str, int, int, float, int], List[Dict[str, float | int | str]]] = {}
    for row in excess_rows:
        key = (
            str(row["method"]),
            int(row["smoothness_s"]),
            int(row["k"]),
            float(row["sigma"]),
            int(row["repetition"]),
        )
        rep_grouped.setdefault(key, []).append(row)

    rep_rows: List[Dict[str, float | int | str]] = []
    for (method, s, k, sigma, rep), rows in rep_grouped.items():
        vals = np.asarray([float(row["excess_error"]) for row in rows], dtype=np.float64)
        rep_rows.append(
            {
                "method": method,
                "smoothness_s": int(s),
                "k": int(k),
                "sigma": float(sigma),
                "repetition": int(rep),
                "variance_term": float((sigma ** 2) / k),
                "mean_excess_mse": float(np.mean(vals)),
                "mean_excess_mse_clipped": float(max(np.mean(vals), 0.0)),
                "n_bins": int(vals.shape[0]),
            }
        )

    mean_grouped: Dict[Tuple[str, int, int, float], List[Dict[str, float | int | str]]] = {}
    for row in rep_rows:
        key = (
            str(row["method"]),
            int(row["smoothness_s"]),
            int(row["k"]),
            float(row["sigma"]),
        )
        mean_grouped.setdefault(key, []).append(row)

    mean_rows: List[Dict[str, float | int | str]] = []
    for (method, s, k, sigma), rows in mean_grouped.items():
        mean_rows.append(
            {
                "method": method,
                "smoothness_s": int(s),
                "k": int(k),
                "sigma": float(sigma),
                "variance_term": float((sigma ** 2) / k),
                "mean_excess_mse": float(np.mean([float(row["mean_excess_mse"]) for row in rows])),
                "mean_excess_mse_clipped": float(
                    np.mean([float(row["mean_excess_mse_clipped"]) for row in rows])
                ),
                "std_excess_mse": float(np.std([float(row["mean_excess_mse"]) for row in rows], ddof=0)),
                "n_reps": int(len(rows)),
            }
        )

    summary: Dict[Tuple[str, int], Dict[str, float]] = {}
    for method in ("local_poly", "nearest_neighbor"):
        for s in DEFAULT_S_VALUES:
            subset = [
                row
                for row in mean_rows
                if row["method"] == method and int(row["smoothness_s"]) == int(s)
            ]
            if not subset:
                raise ValueError(f"No variance ablation rows available for method={method}, s={s}")
            x = np.asarray([float(row["variance_term"]) for row in subset], dtype=np.float64)
            y = np.asarray([float(row["mean_excess_mse"]) for row in subset], dtype=np.float64)
            fit = fit_no_intercept_line(x, y)
            summary[(method, s)] = {
                "slope": float(fit["coef"]),
                "R2": float(fit["R2"]),
                "n_obs": int(x.shape[0]),
                "negative_means": int(np.sum(y < 0.0)),
            }
    return summary, rep_rows, mean_rows


def fit_variance_model(
    excess_rows: Sequence[Dict[str, float | int | str]],
    method: str,
    s: int,
) -> Dict[str, float]:
    subset = [
        row for row in excess_rows
        if row["method"] == method and int(row["smoothness_s"]) == int(s)
    ]
    if not subset:
        raise ValueError(f"No excess-error rows available for method={method}, s={s}")

    x = np.asarray([float(row["variance_term"]) for row in subset], dtype=np.float64)
    y = np.asarray([float(row["excess_error"]) for row in subset], dtype=np.float64)
    fit = fit_no_intercept_line(x, y)
    return {
        "B": float(fit["coef"]),
        "R2": float(fit["R2"]),
        "n_obs": int(x.shape[0]),
        "predicted": (fit["coef"] * x).tolist(),
        "observed": y.tolist(),
    }


def make_components_figure(
    figures_dir: Path,
    raw_rows: Sequence[Dict[str, float | int | str]],
    excess_rows: Sequence[Dict[str, float | int | str]],
    approx_summary: Dict[Tuple[str, int], Dict[str, float]],
    variance_results: Dict[Tuple[str, int], Dict[str, float]],
) -> None:
    colors = {"local_poly": "#4c78a8", "nearest_neighbor": "#e45756"}
    markers = {"local_poly": "o", "nearest_neighbor": "^"}
    linestyles = {"local_poly": "-", "nearest_neighbor": "--", "reference": "-."}
    labels = {"local_poly": "Local polynomial", "nearest_neighbor": "Nearest neighbor"}

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
    for row_idx, s in enumerate(DEFAULT_S_VALUES):
        ax_approx = axes[row_idx, 0]
        ax_var = axes[row_idx, 1]

        pooled_x = []
        pooled_y = []
        for method in ("local_poly", "nearest_neighbor"):
            subset = [
                row for row in raw_rows
                if row["method"] == method
                and int(row["smoothness_s"]) == int(s)
                and math.isclose(float(row["sigma"]), 0.0, rel_tol=0.0, abs_tol=1e-12)
            ]
            x = np.asarray([float(row["h_bin_center"]) for row in subset], dtype=np.float64)
            y = np.maximum(np.asarray([float(row["mse_bin"]) for row in subset], dtype=np.float64), 1e-12)
            pooled_x.append(x)
            pooled_y.append(y)
            ax_approx.scatter(
                x,
                y,
                s=11,
                alpha=0.12,
                color=colors[method],
                marker=markers[method],
                edgecolors="none",
                label=labels[method],
            )
            fit = fit_loglog_slope(x, y)
            line_x = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
            line_y = np.exp(fit["intercept"]) * (line_x ** fit["slope"])
            ax_approx.plot(
                line_x,
                line_y,
                color=colors[method],
                linestyle=linestyles[method],
                marker=markers[method],
                markersize=3.8,
                markevery=28,
                linewidth=2.0,
            )

        x_all = np.concatenate(pooled_x, axis=0)
        y_all = np.concatenate(pooled_y, axis=0)
        ref_intercept = float(np.mean(np.log(y_all)) - (2 * s) * np.mean(np.log(x_all)))
        ref_x = np.geomspace(float(np.min(x_all)), float(np.max(x_all)), 200)
        ref_y = np.exp(ref_intercept) * (ref_x ** (2 * s))
        ax_approx.plot(
            ref_x,
            ref_y,
            color="#222222",
            linestyle=linestyles["reference"],
            marker="D",
            markersize=3.8,
            markevery=34,
            linewidth=1.8,
            label=rf"Reference slope $2s={2*s}$",
        )
        text_approx = (
            rf"LP mean slope $={approx_summary[('local_poly', s)]['mean_slope']:.2f}$" "\n"
            rf"NN mean slope $={approx_summary[('nearest_neighbor', s)]['mean_slope']:.2f}$"
        )
        ax_approx.text(
            0.05,
            0.94,
            text_approx,
            transform=ax_approx.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        ax_approx.set_xscale("log")
        ax_approx.set_yscale("log")
        ax_approx.set_xlabel("Support radius h")
        ax_approx.set_ylabel("Noiseless clean MSE")
        ax_approx.set_title(rf"Approximation scaling ($s={s}$)")
        ax_approx.grid(alpha=0.18, which="both")

        for method in ("local_poly", "nearest_neighbor"):
            subset = [
                row for row in excess_rows
                if row["method"] == method and int(row["smoothness_s"]) == int(s)
            ]
            x = np.asarray([float(row["variance_term"]) for row in subset], dtype=np.float64)
            y = np.asarray([float(row["excess_error"]) for row in subset], dtype=np.float64)
            unique_x = np.unique(x)
            mean_y = np.asarray([float(np.mean(y[np.isclose(x, x0)])) for x0 in unique_x], dtype=np.float64)
            ax_var.scatter(
                unique_x,
                mean_y,
                s=30,
                color=colors[method],
                marker=markers[method],
                alpha=0.9,
                label=labels[method],
            )
            B = variance_results[(method, s)]["B"]
            line_x = np.linspace(0.0, float(np.max(unique_x) * 1.02), 200)
            ax_var.plot(
                line_x,
                B * line_x,
                color=colors[method],
                linestyle=linestyles[method],
                marker=markers[method],
                markersize=3.8,
                markevery=28,
                linewidth=2.0,
            )

        text_var = (
            rf"LP $R^2={variance_results[('local_poly', s)]['R2']:.3f}$" "\n"
            rf"NN $R^2={variance_results[('nearest_neighbor', s)]['R2']:.3f}$"
        )
        ax_var.text(
            0.05,
            0.94,
            text_var,
            transform=ax_var.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        ax_var.set_xlabel(r"Variance term $\sigma^2/k$")
        ax_var.set_ylabel(r"Excess error $\Delta_{\sigma,k,\mathrm{bin}}$")
        ax_var.set_title(rf"Variance diagnostic ($s={s}$)")
        ax_var.grid(alpha=0.18)

    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    panel_names = {
        (0, 0): "exp3_components_panel_approx_s1.pdf",
        (0, 1): "exp3_components_panel_variance_s1.pdf",
        (1, 0): "exp3_components_panel_approx_s2.pdf",
        (1, 1): "exp3_components_panel_variance_s2.pdf",
    }
    for (i, j), name in panel_names.items():
        save_axes_group_panel(fig, [axes[i, j]], figures_dir / name)
    save_legend_figure(handles, labels_out, figures_dir / "exp3_components_legend.pdf", ncol=3)
    plt.close(fig)


def make_local_poly_profiles_figure(
    figures_dir: Path,
    raw_rows: Sequence[Dict[str, float | int | str]],
) -> None:
    method_label = {"local_poly": "LP", "nearest_neighbor": "NN"}
    style_keys = [
        (method, k, sigma)
        for method in ("local_poly", "nearest_neighbor")
        for k in PROFILE_PLOT_K
        for sigma in PROFILE_PLOT_SIGMA
    ]
    style_map = make_style_map(style_keys)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for ax, s in zip(axes, DEFAULT_S_VALUES):
        for method in ("local_poly", "nearest_neighbor"):
            for k in PROFILE_PLOT_K:
                for sigma in PROFILE_PLOT_SIGMA:
                    profile = aggregate_profile_rows(
                        raw_rows=raw_rows,
                        s=s,
                        method=method,
                        k=k,
                        sigma=sigma,
                    )
                    if not profile:
                        continue
                    x = np.asarray([float(row["h_bin_center"]) for row in profile], dtype=np.float64)
                    y = np.maximum(np.asarray([float(row["mse_bin"]) for row in profile], dtype=np.float64), 1e-12)
                    style = style_map[(method, k, sigma)]
                    ax.plot(
                        x,
                        y,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markersize=3.8,
                        linewidth=1.5,
                        label=rf"{method_label[method]} $k={k}$, $\sigma={sigma:.2f}$",
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Support radius h")
        if ax is axes[0]:
            ax.set_ylabel("Clean MSE profile")
        ax.set_title(rf"Smoothness $s={s}$")
        ax.grid(alpha=0.18, which="both")

    handles, labels = axes[0].get_legend_handles_labels()
    for ax, s in zip(axes, DEFAULT_S_VALUES):
        save_axes_group_panel(fig, [ax], figures_dir / f"exp3_local_poly_profiles_panel_s{s}.pdf")
    save_legend_figure(
        handles,
        labels,
        figures_dir / "exp3_local_poly_profiles_legend.pdf",
        ncol=3,
        fontsize=8,
    )
    plt.close(fig)


def make_ablation_approximation_figure(
    figures_dir: Path,
    curve_rows: Sequence[Dict[str, float | int | str]],
    approx_summary: Dict[Tuple[str, int], Dict[str, float]],
) -> None:
    colors = {"local_poly": "#4c78a8", "nearest_neighbor": "#e45756", "reference": "#222222"}
    labels = {"local_poly": "Local polynomial", "nearest_neighbor": "Nearest neighbor"}
    markers = {10: "o", 20: "s", 40: "^", 80: "D"}
    line_markers = {"local_poly": "P", "nearest_neighbor": "X", "reference": "D"}
    line_styles = {"local_poly": "-", "nearest_neighbor": "--", "reference": "-."}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, s in zip(axes, DEFAULT_S_VALUES):
        pooled_x = []
        pooled_y = []
        for method in ("local_poly", "nearest_neighbor"):
            method_subset = [
                row for row in curve_rows if row["method"] == method and int(row["smoothness_s"]) == int(s)
            ]
            for k in (10, 20, 40, 80):
                subset = [row for row in method_subset if int(row["k"]) == int(k)]
                if not subset:
                    continue
                x = np.asarray([float(row["h_bin_center"]) for row in subset], dtype=np.float64)
                y = np.maximum(np.asarray([float(row["mse_bin"]) for row in subset], dtype=np.float64), 1e-12)
                pooled_x.append(x)
                pooled_y.append(y)
                ax.scatter(
                    x,
                    y,
                    color=colors[method],
                    alpha=0.30 if method == "local_poly" else 0.22,
                    s=30,
                    marker=markers[k],
                    edgecolors="none",
                )

            all_x = np.asarray(
                [float(row["h_bin_center"]) for row in method_subset],
                dtype=np.float64,
            )
            all_y = np.maximum(
                np.asarray([float(row["mse_bin"]) for row in method_subset], dtype=np.float64),
                1e-12,
            )
            fit = fit_loglog_slope(all_x, all_y)
            line_x = np.geomspace(float(np.min(all_x)), float(np.max(all_x)), 200)
            line_y = np.exp(fit["intercept"]) * (line_x ** fit["slope"])
            ax.plot(
                line_x,
                line_y,
                color=colors[method],
                linestyle=line_styles[method],
                marker=line_markers[method],
                markevery=36,
                linewidth=2.2,
                label=labels[method],
            )

        x_all = np.concatenate(pooled_x, axis=0)
        y_all = np.concatenate(pooled_y, axis=0)
        ref_intercept = float(np.mean(np.log(y_all)) - (2 * s) * np.mean(np.log(x_all)))
        ref_x = np.geomspace(float(np.min(x_all)), float(np.max(x_all)), 200)
        ref_y = np.exp(ref_intercept) * (ref_x ** (2 * s))
        ax.plot(
            ref_x,
            ref_y,
            color=colors["reference"],
            linestyle=line_styles["reference"],
            marker=line_markers["reference"],
            markevery=42,
            linewidth=1.8,
            label=rf"Reference $2s={2*s}$",
        )

        text = (
            rf"LP slope $={approx_summary[('local_poly', s)]['mean_slope']:.2f}$, $R^2={approx_summary[('local_poly', s)]['mean_r2']:.2f}$" "\n"
            rf"NN slope $={approx_summary[('nearest_neighbor', s)]['mean_slope']:.2f}$, $R^2={approx_summary[('nearest_neighbor', s)]['mean_r2']:.2f}$"
        )
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Support radius h")
        if ax is axes[0]:
            ax.set_ylabel("Approximation-only clean MSE")
        ax.set_title(rf"Approximation ablation ($s={s}$)")
        ax.grid(alpha=0.18, which="both")

    handles, labels_out = axes[0].get_legend_handles_labels()
    for ax, s in zip(axes, DEFAULT_S_VALUES):
        save_axes_group_panel(
            fig,
            [ax],
            figures_dir / f"exp3_ablation_approximation_scaling_panel_s{s}.pdf",
        )
    save_legend_figure(
        handles,
        labels_out,
        figures_dir / "exp3_ablation_approximation_scaling_legend.pdf",
        ncol=3,
    )
    plt.close(fig)


def make_ablation_variance_figure(
    figures_dir: Path,
    mean_rows: Sequence[Dict[str, float | int | str]],
    variance_summary: Dict[Tuple[str, int], Dict[str, float]],
) -> None:
    colors = {"local_poly": "#4c78a8", "nearest_neighbor": "#e45756"}
    labels = {"local_poly": "Local polynomial", "nearest_neighbor": "Nearest neighbor"}
    markers = {5: "o", 10: "s", 20: "^", 40: "D", 80: "P"}
    line_markers = {"local_poly": "X", "nearest_neighbor": "v"}
    line_styles = {"local_poly": "-", "nearest_neighbor": "--"}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, s in zip(axes, DEFAULT_S_VALUES):
        for method in ("local_poly", "nearest_neighbor"):
            subset = [
                row for row in mean_rows if row["method"] == method and int(row["smoothness_s"]) == int(s)
            ]
            x = np.asarray([float(row["variance_term"]) for row in subset], dtype=np.float64)
            y = np.asarray([float(row["mean_excess_mse"]) for row in subset], dtype=np.float64)
            y_plot = np.maximum(y, 0.0)
            for row, xv, yv in zip(subset, x, y_plot):
                ax.scatter(
                    [xv],
                    [yv],
                    color=colors[method],
                    alpha=0.85,
                    s=42,
                    marker=markers[int(row["k"])],
                    edgecolors="none",
                )
            coef = variance_summary[(method, s)]["slope"]
            line_x = np.linspace(0.0, float(np.max(x) * 1.03), 200)
            ax.plot(
                line_x,
                coef * line_x,
                color=colors[method],
                linestyle=line_styles[method],
                marker=line_markers[method],
                markevery=30,
                linewidth=2.0,
                label=labels[method],
            )

        text = (
            rf"LP $R^2={variance_summary[('local_poly', s)]['R2']:.2f}$" "\n"
            rf"NN $R^2={variance_summary[('nearest_neighbor', s)]['R2']:.2f}$"
        )
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        ax.set_xlabel(r"$\sigma^2/k$")
        if ax is axes[0]:
            ax.set_ylabel("Excess MSE")
        ax.set_title(rf"Variance ablation ($s={s}$)")
        ax.grid(alpha=0.18)

    handles, labels_out = axes[0].get_legend_handles_labels()
    for ax, s in zip(axes, DEFAULT_S_VALUES):
        save_axes_group_panel(
            fig,
            [ax],
            figures_dir / f"exp3_ablation_variance_scaling_panel_s{s}.pdf",
        )
    save_legend_figure(
        handles,
        labels_out,
        figures_dir / "exp3_ablation_variance_scaling_legend.pdf",
        ncol=2,
    )
    plt.close(fig)


def make_predicted_vs_observed_figure(
    figures_dir: Path,
    raw_rows: Sequence[Dict[str, float | int | str]],
    fit_results: Dict[Tuple[str, int], Dict[str, float]],
) -> None:
    colors = {"local_poly": "#4c78a8", "nearest_neighbor": "#e45756"}
    markers = {"local_poly": "o", "nearest_neighbor": "^"}
    labels = {"local_poly": "Local polynomial", "nearest_neighbor": "Nearest neighbor"}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, s in zip(axes, DEFAULT_S_VALUES):
        xy_all = []
        for method in ("local_poly", "nearest_neighbor"):
            subset = [
                row for row in raw_rows
                if row["method"] == method and int(row["smoothness_s"]) == int(s)
            ]
            A = fit_results[(method, s)]["A"]
            B = fit_results[(method, s)]["B"]
            predicted = np.asarray(
                [
                    A * float(row["h_power_term"]) + B * float(row["variance_term"])
                    for row in subset
                ],
                dtype=np.float64,
            )
            observed = np.maximum(
                np.asarray([float(row["mse_bin"]) for row in subset], dtype=np.float64),
                1e-12,
            )
            predicted = np.maximum(predicted, 1e-12)
            xy_all.append(predicted)
            xy_all.append(observed)
            ax.scatter(
                predicted,
                observed,
                s=12,
                alpha=0.18,
                color=colors[method],
                marker=markers[method],
                edgecolors="none",
                label=labels[method],
            )

        stacked = np.concatenate(xy_all, axis=0)
        lo = float(np.min(stacked))
        hi = float(np.max(stacked))
        diag = np.geomspace(lo, hi, 200)
        ax.plot(
            diag,
            diag,
            color="#222222",
            linestyle="-.",
            marker="D",
            markevery=34,
            linewidth=1.5,
            label="Diagonal",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Predicted MSE from two-term fit")
        if ax is axes[0]:
            ax.set_ylabel("Observed MSE")
        ax.set_title(rf"Smoothness $s={s}$")
        ax.grid(alpha=0.18, which="both")
        text = (
            rf"LP $R^2={fit_results[('local_poly', s)]['R2']:.3f}$" "\n"
            rf"NN $R^2={fit_results[('nearest_neighbor', s)]['R2']:.3f}$"
        )
        ax.text(
            0.05,
            0.90,
            text,
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    handles, labels_out = axes[0].get_legend_handles_labels()
    for ax, s in zip(axes, DEFAULT_S_VALUES):
        save_axes_group_panel(
            fig,
            [ax],
            figures_dir / f"exp3_predicted_vs_observed_panel_s{s}.pdf",
        )
    save_legend_figure(
        handles,
        labels_out,
        figures_dir / "exp3_predicted_vs_observed_legend.pdf",
        ncol=3,
    )
    plt.close(fig)


def save_table_csv(tables_dir: Path, rows: Sequence[Dict[str, float | str]]) -> None:
    path = tables_dir / "exp3_local_poly_fit.csv"
    fieldnames = [
        "method",
        "smoothness_s",
        "approx_slope",
        "predicted_slope_2s",
        "variance_fit_r2",
        "two_term_r2",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_table_tex(tables_dir: Path, rows: Sequence[Dict[str, float | str]]) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Smoothness $s$ & Approx. slope & Predicted slope $2s$ & Variance fit $R^2$ & Two-term $R^2$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = str(row["method"])
        s = int(row["smoothness_s"])
        approx_slope = float(row["approx_slope"])
        predicted_slope = float(row["predicted_slope_2s"])
        variance_fit_r2 = float(row["variance_fit_r2"])
        two_term_r2 = float(row["two_term_r2"])
        lines.append(
            f"{method} & $s={s}$ & {approx_slope:.3f} & {predicted_slope:.0f} & {variance_fit_r2:.3f} & {two_term_r2:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "exp3_local_poly_fit.tex").write_text("\n".join(lines) + "\n")


def save_raw_results(results_dir: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    path = results_dir / "exp3_raw_results.csv"
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_ablation_table_csv(tables_dir: Path, rows: Sequence[Dict[str, float | str]]) -> None:
    path = tables_dir / "exp3_ablation_diagnostics.csv"
    fieldnames = [
        "method",
        "smoothness_s",
        "approx_slope",
        "predicted_approx_slope",
        "approx_r2",
        "variance_slope",
        "variance_r2",
        "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_ablation_table_tex(tables_dir: Path, rows: Sequence[Dict[str, float | str]]) -> None:
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Method & Smoothness $s$ & Approx. slope & Pred. slope & Approx. $R^2$ & Var. slope & Var. $R^2$ & Notes \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & $s={int(row['smoothness_s'])}$ & "
            f"{float(row['approx_slope']):.3f} & {float(row['predicted_approx_slope']):.0f} & "
            f"{float(row['approx_r2']):.3f} & {float(row['variance_slope']):.4f} & "
            f"{float(row['variance_r2']):.3f} & {row['notes']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "exp3_ablation_diagnostics.tex").write_text("\n".join(lines) + "\n")


def save_ablation_raw_results(results_dir: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    path = results_dir / "exp3_ablation_raw.csv"
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_ablation_json(
    results_dir: Path,
    args: argparse.Namespace,
    approx_summary: Dict[Tuple[str, int], Dict[str, float]],
    approx_fit_rows: Sequence[Dict[str, float | int | str]],
    variance_summary: Dict[Tuple[str, int], Dict[str, float]],
    variance_rep_rows: Sequence[Dict[str, float | int | str]],
    variance_mean_rows: Sequence[Dict[str, float | int | str]],
    table_rows: Sequence[Dict[str, float | str]],
) -> None:
    payload = {
        "experiment": "exp3_ablation_diagnostics",
        "parameters": {
            "seed": int(args.seed),
            "n_train": int(args.n_train),
            "n_eval": int(args.n_eval),
            "n_reps": int(args.n_reps),
            "n_bins": int(args.n_bins),
            "ridge": float(args.ridge),
            "k_values": list(DEFAULT_K_VALUES),
            "sigma_values": list(DEFAULT_SIGMA_VALUES),
            "diagnostic_targets": bool(args.use_diagnostic_targets),
        },
        "interpretation_notes": [
            "The approximation-only ablation isolates the sigma=0 component and provides a cleaner diagnostic for h^{2s} than the mixed two-term regression.",
            "The variance-only ablation checks whether excess error grows with sigma^2/k after subtracting the sigma=0 baseline.",
            "Fitted slopes are diagnostics for consistency with the upper-bound rate structure, not exact theorem verification.",
            "The primary evidence remains the two-term structure and the comparison between local polynomial and the lower-order baseline.",
        ],
        "table_rows": list(table_rows),
        "approximation_summary": {
            f"{method}_s{s}": {
                "mean_slope": float(result["mean_slope"]),
                "mean_r2": float(result["mean_r2"]),
                "predicted_slope_2s": int(2 * s),
                "mean_valid_bins": float(result["mean_valid_bins"]),
                "n_fits": int(result["n_fits"]),
            }
            for (method, s), result in approx_summary.items()
        },
        "variance_summary": {
            f"{method}_s{s}": {
                "variance_slope": float(result["slope"]),
                "variance_r2": float(result["R2"]),
                "n_obs": int(result["n_obs"]),
                "negative_means": int(result["negative_means"]),
            }
            for (method, s), result in variance_summary.items()
        },
        "approximation_fit_rows": list(approx_fit_rows),
        "variance_rep_rows": list(variance_rep_rows),
        "variance_mean_rows": list(variance_mean_rows),
        "output_paths": {
            "approximation_figure": str((results_dir.parent / "figures" / "exp3_ablation_approximation_scaling.pdf").resolve()),
            "variance_figure": str((results_dir.parent / "figures" / "exp3_ablation_variance_scaling.pdf").resolve()),
            "table_csv": str((results_dir.parent / "tables" / "exp3_ablation_diagnostics.csv").resolve()),
            "table_tex": str((results_dir.parent / "tables" / "exp3_ablation_diagnostics.tex").resolve()),
            "raw_csv": str((results_dir / "exp3_ablation_raw.csv").resolve()),
        },
    }
    (results_dir / "exp3_ablation_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")


def save_summary_json(
    results_dir: Path,
    args: argparse.Namespace,
    approx_summary: Dict[Tuple[str, int], Dict[str, float]],
    approx_rows: Sequence[Dict[str, float | int | str]],
    variance_results: Dict[Tuple[str, int], Dict[str, float]],
    excess_rows: Sequence[Dict[str, float | int | str]],
    two_term_results: Dict[Tuple[str, int], Dict[str, float]],
    table_rows: Sequence[Dict[str, float | str]],
    raw_rows: Sequence[Dict[str, float | int | str]],
) -> None:
    payload = {
        "experiment": "exp3_local_poly_upper",
        "parameters": {
            "seed": int(args.seed),
            "n_train": int(args.n_train),
            "n_eval": int(args.n_eval),
            "n_reps": int(args.n_reps),
            "n_bins": int(args.n_bins),
            "ridge": float(args.ridge),
            "k_values": list(DEFAULT_K_VALUES),
            "sigma_values": list(DEFAULT_SIGMA_VALUES),
            "s_values": list(DEFAULT_S_VALUES),
            "run_ablation": bool(args.run_ablation),
            "ablation_only": bool(args.ablation_only),
        },
        "table_rows": list(table_rows),
        "approximation_summary": {
            f"{method}_s{s}": {
                "mean_slope": float(result["mean_slope"]),
                "std_slope": float(result["std_slope"]),
                "se_slope": float(result["se_slope"]),
                "predicted_slope_2s": int(2 * s),
                "n_fits": int(result["n_fits"]),
            }
            for (method, s), result in approx_summary.items()
        },
        "variance_fit_results": {
            f"{method}_s{s}": {
                "B": float(result["B"]),
                "R2": float(result["R2"]),
                "n_obs": int(result["n_obs"]),
                "r2_definition": "uncentered_no_intercept",
            }
            for (method, s), result in variance_results.items()
        },
        "two_term_fit_results": {
            f"{method}_s{s}": {
                "A": float(result["A"]),
                "B": float(result["B"]),
                "R2": float(result["R2"]),
                "n_obs": int(result["n_obs"]),
            }
            for (method, s), result in two_term_results.items()
        },
        "raw_result_count": len(raw_rows),
        "approximation_fit_count": len(approx_rows),
        "excess_error_count": len(excess_rows),
        "output_paths": {
            "components_figure": str((results_dir.parent / "figures" / "exp3_components.pdf").resolve()),
            "profiles_figure": str((results_dir.parent / "figures" / "exp3_local_poly_profiles.pdf").resolve()),
            "predicted_observed_figure": str((results_dir.parent / "figures" / "exp3_predicted_vs_observed.pdf").resolve()),
            "table_csv": str((results_dir.parent / "tables" / "exp3_local_poly_fit.csv").resolve()),
            "table_tex": str((results_dir.parent / "tables" / "exp3_local_poly_fit.tex").resolve()),
            "raw_csv": str((results_dir / "exp3_raw_results.csv").resolve()),
        },
    }
    (results_dir / "exp3_local_poly_upper_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def apply_fast_mode(args: argparse.Namespace) -> argparse.Namespace:
    if args.fast:
        args.n_train = 600
        args.n_eval = 1500
        args.n_reps = 10
        args.approx_reps = 40
        args.variance_design_reps = 25
        args.variance_noise_reps = 40
        args.two_term_design_reps = 20
        args.two_term_noise_reps = 40
    return args


def run_legacy_pipeline(
    args: argparse.Namespace,
    outdir: Path,
    figures_dir: Path,
    tables_dir: Path,
    results_dir: Path,
) -> None:
    if not args.ablation_only:
        raw_rows = run_experiment_rows(
            seed=int(args.seed),
            n_train=int(args.n_train),
            n_eval=int(args.n_eval),
            n_reps=int(args.n_reps),
            n_bins=int(args.n_bins),
            ridge=float(args.ridge),
            sigma_values=DEFAULT_SIGMA_VALUES,
            target_fn=f_star,
            target_label="main",
        )

        approx_summary, approx_rows = summarize_approximation_slopes(raw_rows=raw_rows)
        excess_rows = compute_excess_error_rows(raw_rows=raw_rows)
        variance_results: Dict[Tuple[str, int], Dict[str, float]] = {}
        two_term_results: Dict[Tuple[str, int], Dict[str, float]] = {}
        table_rows: List[Dict[str, float | str]] = []
        for method in ("local_poly", "nearest_neighbor"):
            for s in DEFAULT_S_VALUES:
                variance_fit = fit_variance_model(excess_rows=excess_rows, method=method, s=s)
                two_term_fit = fit_rate_model(raw_rows=raw_rows, method=method, s=s)
                variance_results[(method, s)] = variance_fit
                two_term_results[(method, s)] = two_term_fit
                table_rows.append(
                    {
                        "method": "Local polynomial" if method == "local_poly" else "Nearest neighbor",
                        "smoothness_s": int(s),
                        "approx_slope": float(approx_summary[(method, s)]["mean_slope"]),
                        "predicted_slope_2s": float(2 * s),
                        "variance_fit_r2": float(variance_fit["R2"]),
                        "two_term_r2": float(two_term_fit["R2"]),
                    }
                )

        make_components_figure(
            figures_dir=figures_dir,
            raw_rows=raw_rows,
            excess_rows=excess_rows,
            approx_summary=approx_summary,
            variance_results=variance_results,
        )
        make_local_poly_profiles_figure(
            figures_dir=figures_dir,
            raw_rows=raw_rows,
        )
        make_predicted_vs_observed_figure(
            figures_dir=figures_dir,
            raw_rows=raw_rows,
            fit_results=two_term_results,
        )
        save_table_csv(tables_dir=tables_dir, rows=table_rows)
        save_table_tex(tables_dir=tables_dir, rows=table_rows)
        save_raw_results(results_dir=results_dir, rows=raw_rows)
        save_summary_json(
            results_dir=results_dir,
            args=args,
            approx_summary=approx_summary,
            approx_rows=approx_rows,
            variance_results=variance_results,
            excess_rows=excess_rows,
            two_term_results=two_term_results,
            table_rows=table_rows,
            raw_rows=raw_rows,
        )

    if args.run_ablation:
        ablation_target_fn = f_star_diagnostic if args.use_diagnostic_targets else f_star
        ablation_target_label = "diagnostic" if args.use_diagnostic_targets else "main_target"
        ablation_rows = run_experiment_rows(
            seed=int(args.seed),
            n_train=int(args.n_train),
            n_eval=int(args.n_eval),
            n_reps=int(args.n_reps),
            n_bins=int(args.n_bins),
            ridge=float(args.ridge),
            sigma_values=DEFAULT_SIGMA_VALUES,
            target_fn=ablation_target_fn,
            target_label=ablation_target_label,
        )
        approx_ablation_summary, approx_ablation_fit_rows, approx_curve_rows = summarize_ablation_approximation(
            raw_rows=ablation_rows
        )
        variance_ablation_summary, variance_rep_rows, variance_mean_rows = summarize_ablation_variance(
            raw_rows=ablation_rows
        )
        ablation_excess_rows = compute_excess_error_rows(raw_rows=ablation_rows)
        excess_lookup = {
            (
                str(row["method"]),
                int(row["smoothness_s"]),
                int(row["k"]),
                float(row["sigma"]),
                int(row["repetition"]),
                int(row["bin_id"]),
            ): row
            for row in ablation_excess_rows
        }
        ablation_csv_rows: List[Dict[str, float | int | str]] = []
        for row in ablation_rows:
            key = (
                str(row["method"]),
                int(row["smoothness_s"]),
                int(row["k"]),
                float(row["sigma"]),
                int(row["repetition"]),
                int(row["bin_id"]),
            )
            extra = excess_lookup.get(key, None)
            ablation_csv_rows.append(
                {
                    **row,
                    "baseline_sigma0_mse_bin": float(extra["baseline_mse_bin"]) if extra is not None else "",
                    "excess_error": float(extra["excess_error"]) if extra is not None else "",
                }
            )

        ablation_table_rows: List[Dict[str, float | str]] = []
        for method in ("local_poly", "nearest_neighbor"):
            for s in DEFAULT_S_VALUES:
                notes = (
                    "degree 0" if method == "local_poly" and int(s) == 1 else
                    "degree 1" if method == "local_poly" and int(s) == 2 else
                    "lower-order baseline"
                )
                ablation_table_rows.append(
                    {
                        "method": "Local polynomial" if method == "local_poly" else "Nearest neighbor",
                        "smoothness_s": int(s),
                        "approx_slope": float(approx_ablation_summary[(method, s)]["mean_slope"]),
                        "predicted_approx_slope": float(2 * s),
                        "approx_r2": float(approx_ablation_summary[(method, s)]["mean_r2"]),
                        "variance_slope": float(variance_ablation_summary[(method, s)]["slope"]),
                        "variance_r2": float(variance_ablation_summary[(method, s)]["R2"]),
                        "notes": notes,
                    }
                )

        make_ablation_approximation_figure(
            figures_dir=figures_dir,
            curve_rows=approx_curve_rows,
            approx_summary=approx_ablation_summary,
        )
        make_ablation_variance_figure(
            figures_dir=figures_dir,
            mean_rows=variance_mean_rows,
            variance_summary=variance_ablation_summary,
        )
        save_ablation_table_csv(tables_dir=tables_dir, rows=ablation_table_rows)
        save_ablation_table_tex(tables_dir=tables_dir, rows=ablation_table_rows)
        save_ablation_raw_results(results_dir=results_dir, rows=ablation_csv_rows)
        save_ablation_json(
            results_dir=results_dir,
            args=args,
            approx_summary=approx_ablation_summary,
            approx_fit_rows=approx_ablation_fit_rows,
            variance_summary=variance_ablation_summary,
            variance_rep_rows=variance_rep_rows,
            variance_mean_rows=variance_mean_rows,
            table_rows=ablation_table_rows,
        )


def main() -> None:
    args = apply_fast_mode(parse_args())
    if args.ablation_only:
        args.run_ablation = True
        args.setting = "legacy"
    if max(DEFAULT_K_VALUES) >= args.n_train:
        raise ValueError("Each k must be smaller than n_train.")

    outdir = resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    results_dir = outdir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.setting in ("all", "legacy"):
        run_legacy_pipeline(
            args=args,
            outdir=outdir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            results_dir=results_dir,
        )

    if args.setting in ("all", "rate_controlled"):
        run_rate_controlled(args=args, outdir=outdir)


if __name__ == "__main__":
    main()
