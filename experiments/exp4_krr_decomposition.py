from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
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
from sklearn.neighbors import NearestNeighbors

from figure_layout_utils import save_axes_group_panel, save_legend_figure


DEFAULT_LAMBDA_GRID = tuple(np.logspace(-7.0, 0.0, 41))
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4: support-conditioned bias-variance reshaping in KRR."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-eval", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=8000)
    parser.add_argument("--n-reps", type=int, default=20)
    parser.add_argument("--n-noise-reps", type=int, default=20)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--k-support", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def apply_fast_mode(args: argparse.Namespace) -> argparse.Namespace:
    if args.fast:
        args.n_train = 300
        args.n_eval = 1500
        args.n_test = 2500
        args.n_reps = 5
        args.n_noise_reps = 8
    return args


def resolve_outdir(outdir: str) -> Path:
    path = Path(outdir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sample_nonuniform_design(
    n: int,
    rng: np.random.Generator,
    weights: Sequence[float],
    centers: Sequence[Tuple[float, float]],
    sds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    counts = rng.multinomial(n, np.asarray(weights, dtype=np.float64))
    parts: List[np.ndarray] = []
    for count, center, sd in zip(counts[:-1], centers, sds):
        parts.append(
            rng.normal(
                loc=np.asarray(center, dtype=np.float64),
                scale=np.asarray(sd, dtype=np.float64),
                size=(count, 2),
            )
        )
    parts.append(rng.uniform(0.0, 1.0, size=(counts[-1], 2)))
    X = np.vstack(parts)
    rng.shuffle(X, axis=0)
    return np.clip(X, 0.0, 1.0)


def f_star(
    X: np.ndarray,
    bump_amplitude: float,
    bump_sharpness: float,
    bump_center: Tuple[float, float],
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]
    c1, c2 = bump_center
    return (
        np.sin(2.0 * math.pi * x1) * np.cos(2.0 * math.pi * x2)
        + float(bump_amplitude) * np.exp(-float(bump_sharpness) * ((x1 - c1) ** 2 + (x2 - c2) ** 2))
        + 0.25 * x1
    )


def rbf_kernel(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    sqX = np.sum(X * X, axis=1)[:, None]
    sqZ = np.sum(Z * Z, axis=1)[None, :]
    sqdist = np.maximum(sqX + sqZ - 2.0 * X @ Z.T, 0.0)
    return np.exp(-float(gamma) * sqdist)


def fit_krr_eigendecomposition(X_train: np.ndarray, gamma: float) -> Dict[str, np.ndarray]:
    K = rbf_kernel(X_train, X_train, gamma=gamma)
    if not np.allclose(K, K.T, atol=1e-10):
        raise RuntimeError("Kernel matrix is not numerically symmetric.")
    eigvals, eigvecs = np.linalg.eigh(K)
    return {
        "K_train": K,
        "eigvals": np.maximum(eigvals, 0.0),
        "eigvecs": eigvecs,
    }


def compute_support_scores(X_query: np.ndarray, X_train: np.ndarray, k_support: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=min(k_support, X_train.shape[0]), algorithm="auto")
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
    eval_bin_ids = assign_bins(h_eval, edges)
    centers = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        mask = eval_bin_ids == b
        if np.any(mask):
            centers[b] = float(np.median(h_eval[mask]))
    weak_bins = np.arange(max(0, n_bins - max(1, n_bins // 5)), n_bins, dtype=np.int64)
    dense_bins = np.arange(0, max(1, n_bins // 5), dtype=np.int64)
    return {
        "edges": edges,
        "eval_bin_ids": eval_bin_ids,
        "centers": centers,
        "weak_bins": weak_bins,
        "dense_bins": dense_bins,
    }


def assign_bins(h_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(h_values, edges[0], np.nextafter(edges[-1], -np.inf))
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int64)


def slope_loglog(h_centers: np.ndarray, mse_values: np.ndarray) -> float:
    mask = (h_centers > 0.0) & (mse_values > 0.0)
    if int(np.sum(mask)) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(h_centers[mask]), np.log(mse_values[mask]), deg=1)
    return float(coeffs[0])


def positive_slope(slope: float) -> float:
    if not math.isfinite(float(slope)):
        return float("inf")
    return max(float(slope), 0.0)


def gcv_score(y_eig: np.ndarray, eigvals: np.ndarray, lambda_: float, n_train: int) -> float:
    inv = 1.0 / (eigvals + n_train * lambda_)
    smoother_eigs = eigvals * inv
    resid_eig = (1.0 - smoother_eigs) * y_eig
    rss = float(np.sum(resid_eig**2))
    tr_s = float(np.sum(smoother_eigs))
    denom = max((1.0 - tr_s / n_train) ** 2, EPS)
    return (rss / n_train) / denom


def estimate_observed_error_monte_carlo(
    bias: np.ndarray,
    variance: np.ndarray,
    n_noise_reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    std = np.sqrt(np.maximum(variance, 0.0))
    z = rng.normal(size=(bias.shape[0], n_noise_reps))
    err = (bias[:, None] + std[:, None] * z) ** 2
    observed = np.mean(err, axis=1)
    return np.maximum(observed, 0.0)


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


def compute_bin_profiles(
    observed: np.ndarray,
    bias_sq: np.ndarray,
    variance: np.ndarray,
    bin_ids: np.ndarray,
    bin_centers: np.ndarray,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    obs_vals: List[float] = []
    bias_vals: List[float] = []
    var_vals: List[float] = []
    sum_vals: List[float] = []
    for b in range(bin_centers.shape[0]):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        obs_mean = float(np.mean(observed[mask]))
        bias_mean = float(np.mean(bias_sq[mask]))
        var_mean = float(np.mean(variance[mask]))
        sum_mean = bias_mean + var_mean
        rows.append(
            {
                "bin_id": int(b + 1),
                "bin_index0": int(b),
                "h_bin_center": float(bin_centers[b]),
                "observed_mse": obs_mean,
                "bias_sq": bias_mean,
                "variance": var_mean,
                "sum_bias_variance": sum_mean,
                "count": int(mask.sum()),
            }
        )
        obs_vals.append(obs_mean)
        bias_vals.append(bias_mean)
        var_vals.append(var_mean)
        sum_vals.append(sum_mean)

    h = np.asarray([row["h_bin_center"] for row in rows], dtype=np.float64)
    mse = np.asarray([row["observed_mse"] for row in rows], dtype=np.float64)
    profile_var = float(np.var(mse, ddof=0)) if mse.size else float("nan")
    slope = slope_loglog(h, mse)

    return rows, {
        "profile_var": profile_var,
        "slope": slope,
    }


def compute_pointwise_metrics(
    observed: np.ndarray,
    bin_ids: np.ndarray,
    weak_bins: np.ndarray,
    dense_bins: np.ndarray,
) -> Dict[str, float]:
    weak_mask = np.isin(bin_ids, weak_bins)
    dense_mask = np.isin(bin_ids, dense_bins)
    if not np.any(weak_mask) or not np.any(dense_mask):
        raise RuntimeError("Weak or dense support region is empty.")
    global_mse = float(np.mean(observed))
    weak_mse = float(np.mean(observed[weak_mask]))
    dense_mse = float(np.mean(observed[dense_mask]))
    gap = float(weak_mse / max(dense_mse, EPS))
    return {
        "global_mse": global_mse,
        "weak_mse": weak_mse,
        "dense_mse": dense_mse,
        "gap": gap,
    }


def compute_bias_variance_shares(
    bin_rows: Sequence[Dict[str, float]],
    weak_bins: np.ndarray,
    dense_bins: np.ndarray,
) -> Dict[str, float]:
    def share(subset: Iterable[Dict[str, float]], key: str) -> float:
        rows = list(subset)
        denom = sum(float(row["sum_bias_variance"]) for row in rows)
        if denom <= 0.0:
            return float("nan")
        return float(sum(float(row[key]) for row in rows) / denom)

    overall_bias = share(bin_rows, "bias_sq")
    overall_var = share(bin_rows, "variance")
    weak_rows = [row for row in bin_rows if int(row["bin_index0"]) in set(int(x) for x in weak_bins)]
    dense_rows = [row for row in bin_rows if int(row["bin_index0"]) in set(int(x) for x in dense_bins)]
    weak_bias = share(weak_rows, "bias_sq")
    weak_var = share(weak_rows, "variance")
    dense_bias = share(dense_rows, "bias_sq")
    dense_var = share(dense_rows, "variance")
    return {
        "bias_share": overall_bias,
        "variance_share": overall_var,
        "weak_bin_bias_share": weak_bias,
        "weak_bin_variance_share": weak_var,
        "dense_bin_bias_share": dense_bias,
        "dense_bin_variance_share": dense_var,
    }


def mode_value(values: Sequence[float], grid: Sequence[float]) -> float:
    counts = Counter(values)
    return float(max(grid, key=lambda lam: (counts[lam], -list(grid).index(lam))))


def argmin_key(
    score_map: Dict[float, float],
    grid: Sequence[float],
    subset: Sequence[float] | None = None,
    tol: float = 1e-14,
) -> float:
    candidates = list(subset) if subset is not None else list(grid)
    best = min(float(score_map[lam]) for lam in candidates)
    near = [lam for lam in candidates if float(score_map[lam]) <= best + tol]
    return float(min(near))


def get_setting_configs(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    return {
        "current": {
            "name": "current",
            "weights": (0.70, 0.20, 0.10),
            "centers": ((0.25, 0.30), (0.65, 0.45)),
            "sds": ((0.09, 0.08), (0.10, 0.09)),
            "bump_amplitude": 0.6,
            "bump_sharpness": 25.0,
            "bump_center": (0.78, 0.72),
            "sigma": float(args.sigma),
            "description": "Original mild/current decomposition setting.",
        },
        "reshaping_stress": {
            "name": "reshaping_stress",
            "weights": (0.85, 0.10, 0.05),
            "centers": ((0.25, 0.30), (0.60, 0.45)),
            "sds": ((0.07, 0.07), (0.07, 0.07)),
            "bump_amplitude": 0.8,
            "bump_sharpness": 35.0,
            "bump_center": (0.82, 0.78),
            "sigma": 0.08,
            "description": "Stress setting with stronger support heterogeneity and variance.",
        },
    }


def run_setting(
    setting_name: str,
    setting: Dict[str, object],
    args: argparse.Namespace,
    lambda_grid: Sequence[float],
) -> Dict[str, object]:
    raw_profile_rows: List[Dict[str, float | int | str]] = []
    path_rows: List[Dict[str, float | int | str]] = []
    per_rep_selection_rows: List[Dict[str, float | int | str]] = []

    for rep in range(args.n_reps):
        rng = np.random.default_rng(args.seed + 10000 * rep + (0 if setting_name == "current" else 500_000))
        X_train = sample_nonuniform_design(
            args.n_train,
            rng,
            weights=setting["weights"],
            centers=setting["centers"],
            sds=setting["sds"],
        )
        X_eval = sample_nonuniform_design(
            args.n_eval,
            rng,
            weights=setting["weights"],
            centers=setting["centers"],
            sds=setting["sds"],
        )
        X_test = sample_nonuniform_design(
            args.n_test,
            rng,
            weights=setting["weights"],
            centers=setting["centers"],
            sds=setting["sds"],
        )

        f_train = f_star(
            X_train,
            bump_amplitude=float(setting["bump_amplitude"]),
            bump_sharpness=float(setting["bump_sharpness"]),
            bump_center=setting["bump_center"],
        )
        f_eval = f_star(
            X_eval,
            bump_amplitude=float(setting["bump_amplitude"]),
            bump_sharpness=float(setting["bump_sharpness"]),
            bump_center=setting["bump_center"],
        )
        f_test = f_star(
            X_test,
            bump_amplitude=float(setting["bump_amplitude"]),
            bump_sharpness=float(setting["bump_sharpness"]),
            bump_center=setting["bump_center"],
        )

        sigma = float(setting["sigma"])
        y_train = f_train + sigma * rng.normal(size=args.n_train)

        h_eval = compute_support_scores(X_eval, X_train, k_support=args.k_support)
        bins = make_support_bins(h_eval, n_bins=args.n_bins)
        h_test = compute_support_scores(X_test, X_train, k_support=args.k_support)
        eval_bin_ids = np.asarray(bins["eval_bin_ids"], dtype=np.int64)
        test_bin_ids = assign_bins(h_test, np.asarray(bins["edges"], dtype=np.float64))
        bin_centers = np.asarray(bins["centers"], dtype=np.float64)
        weak_bins = np.asarray(bins["weak_bins"], dtype=np.int64)
        dense_bins = np.asarray(bins["dense_bins"], dtype=np.int64)

        components = fit_krr_eigendecomposition(X_train, gamma=float(args.gamma))
        eigvals = np.asarray(components["eigvals"], dtype=np.float64)
        eigvecs = np.asarray(components["eigvecs"], dtype=np.float64)
        f_eig = eigvecs.T @ f_train
        y_eig = eigvecs.T @ y_train

        K_eval_train = rbf_kernel(X_eval, X_train, gamma=float(args.gamma))
        K_test_train = rbf_kernel(X_test, X_train, gamma=float(args.gamma))
        Z_eval = K_eval_train @ eigvecs
        Z_test = K_test_train @ eigvecs

        eval_metric_map: Dict[float, Dict[str, float]] = {}
        test_metric_map: Dict[float, Dict[str, float]] = {}
        gcv_map: Dict[float, float] = {}

        for lambda_ in lambda_grid:
            inv = 1.0 / (eigvals + args.n_train * float(lambda_))
            mean_eval = Z_eval @ (inv * f_eig)
            mean_test = Z_test @ (inv * f_eig)
            noisy_eval = Z_eval @ (inv * y_eig)
            noisy_test = Z_test @ (inv * y_eig)

            var_eval = (sigma**2) * np.sum((Z_eval * inv[None, :]) ** 2, axis=1)
            var_test = (sigma**2) * np.sum((Z_test * inv[None, :]) ** 2, axis=1)
            bias_eval = mean_eval - f_eval
            bias_test = mean_test - f_test
            bias_sq_eval = bias_eval**2
            bias_sq_test = bias_test**2

            observed_eval = estimate_observed_error_monte_carlo(
                bias=bias_eval,
                variance=var_eval,
                n_noise_reps=args.n_noise_reps,
                rng=rng,
            )
            observed_test = estimate_observed_error_monte_carlo(
                bias=bias_test,
                variance=var_test,
                n_noise_reps=args.n_noise_reps,
                rng=rng,
            )

            eval_bin_rows, eval_profile = compute_bin_profiles(
                observed=observed_eval,
                bias_sq=bias_sq_eval,
                variance=var_eval,
                bin_ids=eval_bin_ids,
                bin_centers=bin_centers,
            )
            test_bin_rows, test_profile = compute_bin_profiles(
                observed=observed_test,
                bias_sq=bias_sq_test,
                variance=var_test,
                bin_ids=test_bin_ids,
                bin_centers=bin_centers,
            )
            eval_point_metrics = compute_pointwise_metrics(observed_eval, eval_bin_ids, weak_bins, dense_bins)
            test_point_metrics = compute_pointwise_metrics(observed_test, test_bin_ids, weak_bins, dense_bins)
            shares = compute_bias_variance_shares(test_bin_rows, weak_bins, dense_bins)
            decomp_error = float(
                np.mean(
                    [
                        abs(float(row["observed_mse"]) - float(row["sum_bias_variance"]))
                        for row in test_bin_rows
                    ]
                )
            )

            eval_metric_map[float(lambda_)] = {
                **eval_point_metrics,
                "slope": float(eval_profile["slope"]),
                "positive_slope": positive_slope(float(eval_profile["slope"])),
                "profile_var": float(eval_profile["profile_var"]),
            }
            test_metric_map[float(lambda_)] = {
                **test_point_metrics,
                "slope": float(test_profile["slope"]),
                "positive_slope": positive_slope(float(test_profile["slope"])),
                "profile_var": float(test_profile["profile_var"]),
                "bias_share": float(shares["bias_share"]),
                "variance_share": float(shares["variance_share"]),
                "weak_bin_bias_share": float(shares["weak_bin_bias_share"]),
                "weak_bin_variance_share": float(shares["weak_bin_variance_share"]),
                "dense_bin_bias_share": float(shares["dense_bin_bias_share"]),
                "dense_bin_variance_share": float(shares["dense_bin_variance_share"]),
                "decomposition_error": decomp_error,
            }
            gcv_map[float(lambda_)] = gcv_score(y_eig=y_eig, eigvals=eigvals, lambda_=float(lambda_), n_train=args.n_train)

            path_rows.append(
                {
                    "setting": setting_name,
                    "repetition": int(rep),
                    "lambda": float(lambda_),
                    "eval_global_mse": float(eval_metric_map[float(lambda_)]["global_mse"]),
                    "eval_weak_mse": float(eval_metric_map[float(lambda_)]["weak_mse"]),
                    "eval_dense_mse": float(eval_metric_map[float(lambda_)]["dense_mse"]),
                    "eval_gap": float(eval_metric_map[float(lambda_)]["gap"]),
                    "eval_slope": float(eval_metric_map[float(lambda_)]["slope"]),
                    "eval_positive_slope": float(eval_metric_map[float(lambda_)]["positive_slope"]),
                    "eval_profile_var": float(eval_metric_map[float(lambda_)]["profile_var"]),
                    "test_global_mse": float(test_metric_map[float(lambda_)]["global_mse"]),
                    "test_weak_mse": float(test_metric_map[float(lambda_)]["weak_mse"]),
                    "test_dense_mse": float(test_metric_map[float(lambda_)]["dense_mse"]),
                    "test_gap": float(test_metric_map[float(lambda_)]["gap"]),
                    "test_slope": float(test_metric_map[float(lambda_)]["slope"]),
                    "test_positive_slope": float(test_metric_map[float(lambda_)]["positive_slope"]),
                    "test_profile_var": float(test_metric_map[float(lambda_)]["profile_var"]),
                    "bias_share": float(test_metric_map[float(lambda_)]["bias_share"]),
                    "variance_share": float(test_metric_map[float(lambda_)]["variance_share"]),
                    "weak_bin_bias_share": float(test_metric_map[float(lambda_)]["weak_bin_bias_share"]),
                    "weak_bin_variance_share": float(test_metric_map[float(lambda_)]["weak_bin_variance_share"]),
                    "dense_bin_bias_share": float(test_metric_map[float(lambda_)]["dense_bin_bias_share"]),
                    "dense_bin_variance_share": float(test_metric_map[float(lambda_)]["dense_bin_variance_share"]),
                    "decomposition_error": float(test_metric_map[float(lambda_)]["decomposition_error"]),
                    "gcv_score": float(gcv_map[float(lambda_)]),
                }
            )

            for row in test_bin_rows:
                raw_profile_rows.append(
                    {
                        "setting": setting_name,
                        "repetition": int(rep),
                        "lambda": float(lambda_),
                        "bin_id": int(row["bin_id"]),
                        "bin_index0": int(row["bin_index0"]),
                        "h_bin_center": float(row["h_bin_center"]),
                        "observed_mse": float(row["observed_mse"]),
                        "bias_sq": float(row["bias_sq"]),
                        "variance": float(row["variance"]),
                        "sum_bias_variance": float(row["sum_bias_variance"]),
                        "count": int(row["count"]),
                    }
                )

        global_scores = {lam: float(eval_metric_map[lam]["global_mse"]) for lam in lambda_grid}
        weak_scores = {lam: float(eval_metric_map[lam]["weak_mse"]) for lam in lambda_grid}
        gap_scores = {lam: float(eval_metric_map[lam]["gap"]) for lam in lambda_grid}
        slope_scores = {lam: float(eval_metric_map[lam]["positive_slope"]) for lam in lambda_grid}
        lambda_global = argmin_key(global_scores, lambda_grid)
        lambda_weak = argmin_key(weak_scores, lambda_grid)
        lambda_gap = argmin_key(gap_scores, lambda_grid)
        lambda_slope = argmin_key(slope_scores, lambda_grid)
        lambda_gcv = argmin_key(gcv_map, lambda_grid)

        global_threshold = (1.0 + float(args.tau)) * global_scores[lambda_global]
        candidate_set = [lam for lam in lambda_grid if global_scores[lam] <= global_threshold + 1e-15]
        weak_ref = max(float(eval_metric_map[lambda_global]["weak_mse"]), EPS)
        gap_ref = max(float(eval_metric_map[lambda_global]["gap"]), EPS)
        slope_ref = max(float(eval_metric_map[lambda_global]["positive_slope"]), EPS)
        profile_scores = {
            lam: (
                float(eval_metric_map[lam]["weak_mse"]) / weak_ref
                + float(eval_metric_map[lam]["gap"]) / gap_ref
                + float(eval_metric_map[lam]["positive_slope"]) / slope_ref
            )
            for lam in lambda_grid
        }
        lambda_profile_unconstrained = argmin_key(profile_scores, lambda_grid)
        lambda_profile = argmin_key(profile_scores, lambda_grid, subset=candidate_set)
        constraint_active = lambda_profile != lambda_profile_unconstrained
        if candidate_set:
            assert global_scores[lambda_profile] <= global_threshold + 1e-12

        for criterion, lambda_selected in (
            ("Global MSE", lambda_global),
            ("Weak MSE", lambda_weak),
            ("Gap", lambda_gap),
            ("Positive slope", lambda_slope),
            ("Profile-aware within 5% global budget", lambda_profile),
            ("GCV", lambda_gcv),
        ):
            metrics = test_metric_map[lambda_selected]
            per_rep_selection_rows.append(
                {
                    "setting": setting_name,
                    "repetition": int(rep),
                    "criterion": criterion,
                    "lambda": float(lambda_selected),
                    "global_mse": float(metrics["global_mse"]),
                    "weak_mse": float(metrics["weak_mse"]),
                    "dense_mse": float(metrics["dense_mse"]),
                    "gap": float(metrics["gap"]),
                    "positive_slope": float(metrics["positive_slope"]),
                    "profile_var": float(metrics["profile_var"]),
                    "bias_share": float(metrics["bias_share"]),
                    "variance_share": float(metrics["variance_share"]),
                    "weak_bin_bias_share": float(metrics["weak_bin_bias_share"]),
                    "weak_bin_variance_share": float(metrics["weak_bin_variance_share"]),
                    "constraint_active": bool(constraint_active) if criterion == "Profile-aware within 5% global budget" else False,
                    "candidate_count": int(len(candidate_set)) if criterion == "Profile-aware within 5% global budget" else len(lambda_grid),
                }
            )

    agg_by_lambda: List[Dict[str, float | str]] = []
    for lambda_ in lambda_grid:
        cur = [row for row in path_rows if math.isclose(float(row["lambda"]), float(lambda_), rel_tol=0.0, abs_tol=1e-15)]
        agg_by_lambda.append(
            {
                "setting": setting_name,
                "lambda": float(lambda_),
                "eval_global_mse": float(np.mean([float(row["eval_global_mse"]) for row in cur])),
                "eval_weak_mse": float(np.mean([float(row["eval_weak_mse"]) for row in cur])),
                "eval_dense_mse": float(np.mean([float(row["eval_dense_mse"]) for row in cur])),
                "eval_gap": float(np.mean([float(row["eval_gap"]) for row in cur])),
                "eval_slope": float(np.nanmean([float(row["eval_slope"]) for row in cur])),
                "eval_positive_slope": float(np.mean([float(row["eval_positive_slope"]) for row in cur])),
                "eval_profile_var": float(np.mean([float(row["eval_profile_var"]) for row in cur])),
                "test_global_mse": float(np.mean([float(row["test_global_mse"]) for row in cur])),
                "test_weak_mse": float(np.mean([float(row["test_weak_mse"]) for row in cur])),
                "test_dense_mse": float(np.mean([float(row["test_dense_mse"]) for row in cur])),
                "test_gap": float(np.mean([float(row["test_gap"]) for row in cur])),
                "test_slope": float(np.nanmean([float(row["test_slope"]) for row in cur])),
                "test_positive_slope": float(np.mean([float(row["test_positive_slope"]) for row in cur])),
                "test_profile_var": float(np.mean([float(row["test_profile_var"]) for row in cur])),
                "bias_share": float(np.mean([float(row["bias_share"]) for row in cur])),
                "variance_share": float(np.mean([float(row["variance_share"]) for row in cur])),
                "weak_bin_bias_share": float(np.mean([float(row["weak_bin_bias_share"]) for row in cur])),
                "weak_bin_variance_share": float(np.mean([float(row["weak_bin_variance_share"]) for row in cur])),
                "dense_bin_bias_share": float(np.mean([float(row["dense_bin_bias_share"]) for row in cur])),
                "dense_bin_variance_share": float(np.mean([float(row["dense_bin_variance_share"]) for row in cur])),
                "decomposition_error": float(np.mean([float(row["decomposition_error"]) for row in cur])),
                "gcv_score": float(np.mean([float(row["gcv_score"]) for row in cur])),
            }
        )
    agg_map = {float(row["lambda"]): row for row in agg_by_lambda}
    eval_global_scores = {lam: float(agg_map[lam]["eval_global_mse"]) for lam in lambda_grid}
    eval_weak_scores = {lam: float(agg_map[lam]["eval_weak_mse"]) for lam in lambda_grid}
    eval_gap_scores = {lam: float(agg_map[lam]["eval_gap"]) for lam in lambda_grid}
    eval_slope_scores = {lam: float(agg_map[lam]["eval_positive_slope"]) for lam in lambda_grid}
    gcv_scores = {lam: float(agg_map[lam]["gcv_score"]) for lam in lambda_grid}

    lambda_global = argmin_key(eval_global_scores, lambda_grid)
    lambda_weak = argmin_key(eval_weak_scores, lambda_grid)
    lambda_gap = argmin_key(eval_gap_scores, lambda_grid)
    lambda_slope = argmin_key(eval_slope_scores, lambda_grid)
    lambda_gcv = argmin_key(gcv_scores, lambda_grid)
    threshold = (1.0 + float(args.tau)) * eval_global_scores[lambda_global]
    candidate_set = [lam for lam in lambda_grid if eval_global_scores[lam] <= threshold + 1e-15]
    weak_ref = max(float(agg_map[lambda_global]["eval_weak_mse"]), EPS)
    gap_ref = max(float(agg_map[lambda_global]["eval_gap"]), EPS)
    slope_ref = max(float(agg_map[lambda_global]["eval_positive_slope"]), EPS)
    profile_scores = {
        lam: (
            float(agg_map[lam]["eval_weak_mse"]) / weak_ref
            + float(agg_map[lam]["eval_gap"]) / gap_ref
            + float(agg_map[lam]["eval_positive_slope"]) / slope_ref
        )
        for lam in lambda_grid
    }
    lambda_profile_unconstrained = argmin_key(profile_scores, lambda_grid)
    lambda_profile = argmin_key(profile_scores, lambda_grid, subset=candidate_set)
    constraint_active = lambda_profile != lambda_profile_unconstrained

    selected_rows: List[Dict[str, float | str]] = []
    for criterion, lambda_selected in (
        ("Global MSE", lambda_global),
        ("Weak MSE", lambda_weak),
        ("Gap", lambda_gap),
        ("Positive slope", lambda_slope),
        ("Profile-aware within 5% global budget", lambda_profile),
        ("GCV", lambda_gcv),
    ):
        row = agg_map[lambda_selected]
        selected_rows.append(
            {
                "setting": setting_name,
                "criterion": criterion,
                "selected_lambda": float(lambda_selected),
                "global_mse": float(row["test_global_mse"]),
                "weak_mse": float(row["test_weak_mse"]),
                "dense_mse": float(row["test_dense_mse"]),
                "gap": float(row["test_gap"]),
                "positive_slope": float(row["test_positive_slope"]),
                "profile_var": float(row["test_profile_var"]),
                "bias_share": float(row["bias_share"]),
                "variance_share": float(row["variance_share"]),
                "weak_bin_bias_share": float(row["weak_bin_bias_share"]),
                "weak_bin_variance_share": float(row["weak_bin_variance_share"]),
                "dense_bin_bias_share": float(row["dense_bin_bias_share"]),
                "dense_bin_variance_share": float(row["dense_bin_variance_share"]),
                "decomposition_error": float(row["decomposition_error"]),
            }
        )

    rep_lambda_summary = {}
    for criterion in ("Global MSE", "Weak MSE", "Gap", "Positive slope", "Profile-aware within 5% global budget", "GCV"):
        cur = [row for row in per_rep_selection_rows if row["criterion"] == criterion]
        lambdas = [float(row["lambda"]) for row in cur]
        rep_lambda_summary[criterion] = {
            "lambda_mode": mode_value(lambdas, lambda_grid),
            "lambda_frequencies": {
                f"{lam:g}": int(sum(math.isclose(lam, x, rel_tol=0.0, abs_tol=1e-15) for x in lambdas))
                for lam in lambda_grid
            },
        }

    selected_rows_by_criterion = {str(row["criterion"]): row for row in selected_rows}
    idx_map = {float(lam): idx for idx, lam in enumerate(lambda_grid)}
    global_gap = float(selected_rows_by_criterion["Global MSE"]["gap"])
    profile_gap = float(selected_rows_by_criterion["Profile-aware within 5% global budget"]["gap"])
    global_pos_slope = float(selected_rows_by_criterion["Global MSE"]["positive_slope"])
    profile_pos_slope = float(selected_rows_by_criterion["Profile-aware within 5% global budget"]["positive_slope"])
    profile_gap_improvement = (global_gap - profile_gap) / (abs(global_gap) + 1e-12)
    profile_slope_improvement = (global_pos_slope - profile_pos_slope) / (abs(global_pos_slope) + 1e-12)

    # Promote the reshaping-stress setting whenever the mild/current regime is
    # too close to a pure identity check: either all criteria collapse to the
    # same lambda, or the constrained profile-aware reference barely changes the
    # support-conditioned shape relative to the global-risk optimum.
    need_stress = (
        lambda_global == lambda_weak == lambda_gap == lambda_slope
        or (
            abs(idx_map[lambda_gap] - idx_map[lambda_global]) < 1
            and abs(idx_map[lambda_slope] - idx_map[lambda_global]) < 1
        )
        or (
            setting_name == "current"
            and profile_gap_improvement < 0.02
            and profile_slope_improvement < 0.02
        )
    )

    return {
        "setting": setting_name,
        "config": setting,
        "raw_profile_rows": raw_profile_rows,
        "path_rows": path_rows,
        "per_rep_selection_rows": per_rep_selection_rows,
        "aggregate_path_rows": agg_by_lambda,
        "selected_rows": selected_rows,
        "selected_lambda_map": {
            "lambda_global": lambda_global,
            "lambda_weak": lambda_weak,
            "lambda_gap": lambda_gap,
            "lambda_slope": lambda_slope,
            "lambda_profile": lambda_profile,
            "lambda_gcv": lambda_gcv,
        },
        "constraint_active": bool(constraint_active),
        "candidate_count": int(len(candidate_set)),
        "need_stress": bool(need_stress),
        "profile_gap_improvement": float(profile_gap_improvement),
        "profile_slope_improvement": float(profile_slope_improvement),
        "rep_lambda_summary": rep_lambda_summary,
    }


def aggregate_profiles_for_lambda(
    profile_rows: Sequence[Dict[str, float | int | str]],
    lambda_: float,
) -> List[Dict[str, float]]:
    cur = [row for row in profile_rows if math.isclose(float(row["lambda"]), float(lambda_), rel_tol=0.0, abs_tol=1e-15)]
    by_bin: Dict[int, List[Dict[str, float | int | str]]] = defaultdict(list)
    for row in cur:
        by_bin[int(row["bin_id"])].append(row)
    out: List[Dict[str, float]] = []
    for bin_id in sorted(by_bin):
        rows = by_bin[bin_id]
        out.append(
            {
                "bin_id": int(bin_id),
                "h_bin_center": float(np.mean([float(row["h_bin_center"]) for row in rows])),
                "observed_mse": float(np.mean([float(row["observed_mse"]) for row in rows])),
                "bias_sq": float(np.mean([float(row["bias_sq"]) for row in rows])),
                "variance": float(np.mean([float(row["variance"]) for row in rows])),
                "sum_bias_variance": float(np.mean([float(row["sum_bias_variance"]) for row in rows])),
            }
        )
    return out


def choose_representative_lambdas(
    selected_lambda_map: Dict[str, float],
    lambda_grid: Sequence[float],
) -> List[Tuple[float, str]]:
    pairs = [
        (float(lambda_grid[0]), "small $\\lambda$"),
        (float(selected_lambda_map["lambda_global"]), r"$\lambda_{\mathrm{global}}$"),
        (float(selected_lambda_map["lambda_profile"]), r"$\lambda_{\mathrm{profile}}$"),
        (float(selected_lambda_map["lambda_gap"]), r"$\lambda_{\mathrm{gap}}$"),
        (float(selected_lambda_map["lambda_slope"]), r"$\lambda_{\mathrm{slope}}$"),
        (float(lambda_grid[-1]), "large $\\lambda$"),
    ]
    chosen: List[Tuple[float, str]] = []
    seen: set[float] = set()
    for lam, label in pairs:
        if lam in seen:
            continue
        seen.add(lam)
        chosen.append((lam, label))
        if len(chosen) == 3:
            break
    return chosen


def make_decomposition_profiles_figure(
    figures_dir: Path,
    setting_result: Dict[str, object],
    lambda_grid: Sequence[float],
) -> None:
    representative = choose_representative_lambdas(setting_result["selected_lambda_map"], lambda_grid)
    fig, axes = plt.subplots(1, len(representative), figsize=(4.3 * len(representative), 4.2), constrained_layout=True)
    if len(representative) == 1:
        axes = [axes]  # type: ignore[assignment]
    style_map = {
        "observed_mse": {
            "color": "#2f4b7c",
            "linestyle": "None",
            "marker": "o",
            "linewidth": 0.0,
            "markersize": 4.6,
            "markerfacecolor": "white",
            "markeredgewidth": 1.1,
            "zorder": 5,
        },
        "bias_sq": {
            "color": "#e45756",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 1.6,
            "markersize": 3.4,
            "markerfacecolor": "#e45756",
            "markeredgewidth": 0.8,
            "zorder": 3,
        },
        "variance": {
            "color": "#54a24b",
            "linestyle": ":",
            "marker": "^",
            "linewidth": 1.8,
            "markersize": 4.2,
            "markerfacecolor": "white",
            "markeredgewidth": 1.0,
            "zorder": 4,
        },
        "sum_bias_variance": {
            "color": "#111111",
            "linestyle": "-",
            "marker": None,
            "linewidth": 1.2,
            "markersize": 0.0,
            "markerfacecolor": "#111111",
            "markeredgewidth": 0.0,
            "zorder": 2,
        },
    }
    labels = {
        "observed_mse": "Observed clean error",
        "bias_sq": r"Bias$^2$",
        "variance": "Variance",
        "sum_bias_variance": r"Bias$^2$ + Variance",
    }
    legend_handles = None
    legend_labels = [labels[key] for key in ("observed_mse", "bias_sq", "variance", "sum_bias_variance")]
    for idx, (ax, (lambda_, title)) in enumerate(zip(axes, representative), start=1):
        rows = aggregate_profiles_for_lambda(setting_result["raw_profile_rows"], lambda_)
        x = np.asarray([row["h_bin_center"] for row in rows], dtype=np.float64)
        handle_by_key: Dict[str, object] = {}
        for key in ("sum_bias_variance", "bias_sq", "variance", "observed_mse"):
            y = np.maximum(np.asarray([row[key] for row in rows], dtype=np.float64), 1e-12)
            style = style_map[key]
            (handle,) = ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=float(style["linewidth"]),
                markersize=float(style["markersize"]),
                markerfacecolor=str(style["markerfacecolor"]),
                markeredgewidth=float(style["markeredgewidth"]),
                label=labels[key],
                zorder=int(style["zorder"]),
            )
            handle_by_key[key] = handle
        if legend_handles is None:
            legend_handles = [handle_by_key[key] for key in ("observed_mse", "bias_sq", "variance", "sum_bias_variance")]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.18, which="both")
        ax.set_xlabel("Support radius h")
        if idx == 1:
            ax.set_ylabel("Binwise clean error contribution")
        ax.set_title(f"{title}\n" + rf"$\lambda={lambda_:g}$")
        save_axes_group_panel(fig, [ax], figures_dir / f"exp4_krr_decomposition_profiles_panel_{idx}.pdf")
        save_axes_group_panel(fig, [ax], figures_dir / f"exp4_krr_bias_variance_panel_{idx}.pdf")
    if legend_handles is not None:
        save_legend_figure(legend_handles, legend_labels, figures_dir / "exp4_krr_decomposition_profiles_legend.pdf", ncol=4)
        save_legend_figure(legend_handles, legend_labels, figures_dir / "exp4_krr_bias_variance_legend.pdf", ncol=4)
    fig.savefig(figures_dir / "exp4_krr_decomposition_profiles.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "exp4_krr_bias_variance.pdf", bbox_inches="tight")
    plt.close(fig)


def make_regularization_path_figure(
    figures_dir: Path,
    setting_result: Dict[str, object],
    lambda_grid: Sequence[float],
) -> None:
    agg = {float(row["lambda"]): row for row in setting_result["aggregate_path_rows"]}
    x = np.asarray(lambda_grid, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    metric_specs = [
        ("eval_global_mse", "Global MSE", "#2f4b7c", "o", "-"),
        ("eval_weak_mse", "Weak MSE", "#e45756", "s", "--"),
        ("eval_gap", "Gap", "#54a24b", "^", "-."),
        ("eval_positive_slope", "Positive slope", "#f58518", "D", ":"),
    ]
    for key, label, color, marker, linestyle in metric_specs:
        vals = np.asarray([float(agg[lam][key]) for lam in lambda_grid], dtype=np.float64)
        denom = max(float(np.min(vals)), EPS)
        ax.plot(
            x,
            vals / denom,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=4.2,
            label=label,
        )
    selected = setting_result["selected_lambda_map"]
    line_colors = {
        "lambda_global": "#2f4b7c",
        "lambda_weak": "#e45756",
        "lambda_gap": "#54a24b",
        "lambda_slope": "#f58518",
        "lambda_profile": "#111111",
    }
    for key, color in line_colors.items():
        ax.axvline(float(selected[key]), color=color, linewidth=1.2, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_ylabel("Normalized validation metric (minimum = 1)")
    ax.set_title(f"Regularization-path metrics ({setting_result['setting']})")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(figures_dir / "exp4_regularization_path_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def make_bias_variance_shares_figure(
    figures_dir: Path,
    setting_result: Dict[str, object],
    lambda_grid: Sequence[float],
) -> None:
    agg = {float(row["lambda"]): row for row in setting_result["aggregate_path_rows"]}
    x = np.asarray(lambda_grid, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    specs = [
        ("bias_share", "Overall bias share", "#e45756", "o", "-"),
        ("variance_share", "Overall variance share", "#2f4b7c", "s", "--"),
        ("weak_bin_bias_share", "Weak-bin bias share", "#f58518", "^", "-."),
        ("weak_bin_variance_share", "Weak-bin variance share", "#54a24b", "D", ":"),
    ]
    for key, label, color, marker, linestyle in specs:
        ax.plot(
            x,
            np.asarray([float(agg[lam][key]) for lam in lambda_grid], dtype=np.float64),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=4.2,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Regularization $\lambda$")
    ax.set_ylabel("Share")
    ax.set_title(f"Bias-variance shares ({setting_result['setting']})")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(figures_dir / "exp4_bias_variance_shares_vs_lambda.pdf", bbox_inches="tight")
    plt.close(fig)


def make_profiles_by_lambda_figure(
    figures_dir: Path,
    setting_result: Dict[str, object],
    lambda_grid: Sequence[float],
) -> None:
    representative = choose_representative_lambdas(setting_result["selected_lambda_map"], lambda_grid)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    color_cycle = ["#2f4b7c", "#e45756", "#54a24b", "#f58518", "#7f7f7f"]
    marker_cycle = ["o", "s", "^", "D", "v"]
    for idx, (lambda_, label) in enumerate(representative):
        rows = aggregate_profiles_for_lambda(setting_result["raw_profile_rows"], lambda_)
        ax.plot(
            np.asarray([row["h_bin_center"] for row in rows], dtype=np.float64),
            np.maximum(np.asarray([row["observed_mse"] for row in rows], dtype=np.float64), 1e-12),
            color=color_cycle[idx % len(color_cycle)],
            marker=marker_cycle[idx % len(marker_cycle)],
            linestyle="-",
            linewidth=2.0,
            markersize=4.2,
            label=f"{label} ({lambda_:g})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Support radius h")
    ax.set_ylabel("Support-conditioned clean MSE")
    ax.set_title(f"Representative KRR profiles by $\\lambda$ ({setting_result['setting']})")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    fig.savefig(figures_dir / "exp4_krr_profiles_by_lambda.pdf", bbox_inches="tight")
    plt.close(fig)


def save_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: str) -> str:
    return text.replace("%", "\\%")


def save_selected_lambda_table(
    tables_dir: Path,
    rows: Sequence[Dict[str, float | str]],
) -> None:
    main_rows = [row for row in rows if row["criterion"] != "GCV"]
    save_csv(tables_dir / "exp4_selected_lambda_references.csv", main_rows)
    save_csv(tables_dir / "exp4_krr_selection.csv", main_rows)
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Criterion & Selected $\\lambda$ & Global MSE & Weak MSE & Gap & Pos. slope & Bias share & Variance share \\\\",
        "\\midrule",
    ]
    for row in main_rows:
        lines.append(
            f"{latex_escape(str(row['criterion']))} & "
            f"{float(row['selected_lambda']):g} & "
            f"{float(row['global_mse']):.4f} & "
            f"{float(row['weak_mse']):.4f} & "
            f"{float(row['gap']):.3f} & "
            f"{float(row['positive_slope']):.3f} & "
            f"{float(row['bias_share']):.3f} & "
            f"{float(row['variance_share']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "exp4_selected_lambda_table.tex").write_text("\n".join(lines) + "\n")
    (tables_dir / "exp4_krr_selection.tex").write_text("\n".join(lines) + "\n")


def save_regularization_path_summary_table(
    tables_dir: Path,
    setting_result: Dict[str, object],
    lambda_grid: Sequence[float],
) -> None:
    agg = {float(row["lambda"]): row for row in setting_result["aggregate_path_rows"]}
    selected = setting_result["selected_lambda_map"]
    candidate_lambdas = [
        float(lambda_grid[0]),
        float(selected["lambda_global"]),
        float(selected["lambda_profile"]),
        float(selected["lambda_gap"]),
        float(selected["lambda_slope"]),
        float(lambda_grid[-1]),
    ]
    unique_lambdas = []
    for lam in candidate_lambdas:
        if lam not in unique_lambdas:
            unique_lambdas.append(lam)
    rows = []
    for lam in unique_lambdas:
        row = agg[lam]
        rows.append(
            {
                "setting": setting_result["setting"],
                "lambda": lam,
                "global_mse": float(row["test_global_mse"]),
                "weak_mse": float(row["test_weak_mse"]),
                "gap": float(row["test_gap"]),
                "positive_slope": float(row["test_positive_slope"]),
                "profile_var": float(row["test_profile_var"]),
                "bias_share": float(row["bias_share"]),
                "variance_share": float(row["variance_share"]),
            }
        )
    save_csv(tables_dir / "exp4_regularization_path_summary.csv", rows)
    lines = [
        "\\begin{tabular}{cccccccc}",
        "\\toprule",
        "$\\lambda$ & Global MSE & Weak MSE & Gap & Pos. slope & Profile var. & Bias share & Variance share \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{float(row['lambda']):g} & "
            f"{float(row['global_mse']):.4f} & "
            f"{float(row['weak_mse']):.4f} & "
            f"{float(row['gap']):.3f} & "
            f"{float(row['positive_slope']):.3f} & "
            f"{float(row['profile_var']):.6f} & "
            f"{float(row['bias_share']):.3f} & "
            f"{float(row['variance_share']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "exp4_regularization_path_summary_table.tex").write_text("\n".join(lines) + "\n")


def save_decomposition_summary_tables(
    tables_dir: Path,
    setting_result: Dict[str, object],
    lambda_grid: Sequence[float],
) -> None:
    agg = {float(row["lambda"]): row for row in setting_result["aggregate_path_rows"]}
    selected = setting_result["selected_lambda_map"]
    lambdas = [
        float(lambda_grid[0]),
        float(selected["lambda_global"]),
        float(selected["lambda_profile"]),
        float(selected["lambda_gap"]),
        float(selected["lambda_slope"]),
        float(lambda_grid[-1]),
    ]
    unique = []
    for lam in lambdas:
        if lam not in unique:
            unique.append(lam)
    rows = []
    for lam in unique:
        row = agg[lam]
        rows.append(
            {
                "setting": setting_result["setting"],
                "lambda": lam,
                "global_mse": float(row["test_global_mse"]),
                "weak_mse": float(row["test_weak_mse"]),
                "bias_share": float(row["bias_share"]),
                "variance_share": float(row["variance_share"]),
                "decomp_error": float(row["decomposition_error"]),
                "weak_bin_bias_share": float(row["weak_bin_bias_share"]),
                "weak_bin_variance_share": float(row["weak_bin_variance_share"]),
                "dense_bin_bias_share": float(row["dense_bin_bias_share"]),
                "dense_bin_variance_share": float(row["dense_bin_variance_share"]),
            }
        )
    save_csv(tables_dir / "exp4_decomposition_summary.csv", rows)
    save_csv(tables_dir / "exp4_krr_decomposition.csv", rows)
    lines = [
        "\\begin{tabular}{cccccc}",
        "\\toprule",
        "$\\lambda$ & Global MSE & Weak MSE & Bias share & Variance share & Decomp. error \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{float(row['lambda']):g} & "
            f"{float(row['global_mse']):.4f} & "
            f"{float(row['weak_mse']):.4f} & "
            f"{float(row['bias_share']):.3f} & "
            f"{float(row['variance_share']):.3f} & "
            f"{float(row['decomp_error']):.6f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "exp4_decomposition_summary_table.tex").write_text("\n".join(lines) + "\n")
    (tables_dir / "exp4_krr_decomposition.tex").write_text("\n".join(lines) + "\n")


def save_summary_json(
    results_dir: Path,
    args: argparse.Namespace,
    active_setting: str,
    used_stress: bool,
    settings_results: Dict[str, Dict[str, object]],
    lambda_grid: Sequence[float],
) -> None:
    payload = {
        "experiment": "exp4_krr_bias_variance_reshaping",
        "parameters": {
            "seed": int(args.seed),
            "n_train": int(args.n_train),
            "n_eval": int(args.n_eval),
            "n_test": int(args.n_test),
            "n_reps": int(args.n_reps),
            "n_noise_reps": int(args.n_noise_reps),
            "n_bins": int(args.n_bins),
            "k_support": int(args.k_support),
            "gamma": float(args.gamma),
            "tau": float(args.tau),
            "lambda_grid": [float(x) for x in lambda_grid],
        },
        "active_setting": active_setting,
        "used_reshaping_stress": bool(used_stress),
        "settings": {},
    }
    for setting_name, result in settings_results.items():
        payload["settings"][setting_name] = {
            "config": result["config"],
            "selected_lambda_map": result["selected_lambda_map"],
            "constraint_active": bool(result["constraint_active"]),
            "candidate_count": int(result["candidate_count"]),
            "need_stress": bool(result["need_stress"]),
            "profile_gap_improvement": float(result["profile_gap_improvement"]),
            "profile_slope_improvement": float(result["profile_slope_improvement"]),
            "selected_rows": result["selected_rows"],
            "rep_lambda_summary": result["rep_lambda_summary"],
            "regularization_path_rows": result["aggregate_path_rows"],
        }
    (results_dir / "exp4_krr_decomposition_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


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

    lambda_grid = [float(x) for x in DEFAULT_LAMBDA_GRID]
    setting_configs = get_setting_configs(args)

    current_result = run_setting("current", setting_configs["current"], args, lambda_grid)
    settings_results: Dict[str, Dict[str, object]] = {"current": current_result}
    used_stress = bool(current_result["need_stress"])
    active_setting = "current"
    if used_stress:
        stress_result = run_setting("reshaping_stress", setting_configs["reshaping_stress"], args, lambda_grid)
        settings_results["reshaping_stress"] = stress_result
        active_setting = "reshaping_stress"

    active_result = settings_results[active_setting]

    make_decomposition_profiles_figure(figures_dir=figures_dir, setting_result=active_result, lambda_grid=lambda_grid)
    make_regularization_path_figure(figures_dir=figures_dir, setting_result=active_result, lambda_grid=lambda_grid)
    make_bias_variance_shares_figure(figures_dir=figures_dir, setting_result=active_result, lambda_grid=lambda_grid)
    make_profiles_by_lambda_figure(figures_dir=figures_dir, setting_result=active_result, lambda_grid=lambda_grid)

    save_selected_lambda_table(tables_dir=tables_dir, rows=active_result["selected_rows"])
    save_regularization_path_summary_table(tables_dir=tables_dir, setting_result=active_result, lambda_grid=lambda_grid)
    save_decomposition_summary_tables(tables_dir=tables_dir, setting_result=active_result, lambda_grid=lambda_grid)

    all_path_rows = []
    all_profile_rows = []
    all_selection_rows = []
    for result in settings_results.values():
        all_path_rows.extend(result["path_rows"])
        all_profile_rows.extend(result["raw_profile_rows"])
        all_selection_rows.extend(result["per_rep_selection_rows"])

    save_csv(results_dir / "exp4_regularization_path.csv", all_path_rows)
    save_csv(results_dir / "exp4_raw_profiles.csv", all_profile_rows)
    save_csv(results_dir / "exp4_selection_results.csv", all_selection_rows)
    save_csv(results_dir / "exp4_selected_lambda_references.csv", active_result["selected_rows"])
    save_csv(results_dir / "exp4_decomposition_summary.csv", [row for row in active_result["aggregate_path_rows"]])

    save_summary_json(
        results_dir=results_dir,
        args=args,
        active_setting=active_setting,
        used_stress=used_stress,
        settings_results=settings_results,
        lambda_grid=lambda_grid,
    )


if __name__ == "__main__":
    main()
