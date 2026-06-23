from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from figure_layout_utils import save_axes_group_panel, save_legend_figure


S_VALUES = (1, 2)
APPROX_H_GRID = (0.04, 0.055, 0.075, 0.105, 0.145, 0.20)
VARIANCE_H_FIXED = 0.12
VARIANCE_K_GRID = (10, 20, 40, 80, 160)
VARIANCE_SIGMA_GRID = (0.02, 0.05, 0.10, 0.15)
TWO_TERM_SIGMA_GRID = (0.0, 0.02, 0.05, 0.10, 0.15)
X0 = np.asarray([0.40, 0.40], dtype=np.float64)
DOMAIN_LOW = 0.0
DOMAIN_HIGH = 1.0
STABILITY_EIG_FLOOR = 1e-3
RIDGE_EPS = 1e-12


def target_rate_controlled(X: np.ndarray, s: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]
    if int(s) == 1:
        return x1 + 0.2 * x2
    if int(s) == 2:
        return x1 + 0.2 * x2 + 0.5 * x1**2 + 0.3 * x1 * x2 + 0.25 * x2**2
    raise ValueError(f"Unsupported smoothness s={s}")


def degree_for_s(s: int) -> int:
    return int(math.ceil(float(s)) - 1)


def sample_unit_vectors(
    n: int,
    rng: np.random.Generator,
    geometry: str,
) -> np.ndarray:
    radii = np.sqrt(rng.uniform(size=n))
    if geometry == "half_ball":
        angles = rng.uniform(-0.5 * math.pi, 0.5 * math.pi, size=n)
    elif geometry == "cone":
        angles = rng.uniform(-0.25 * math.pi, 0.25 * math.pi, size=n)
    else:
        raise ValueError(f"Unsupported local geometry {geometry!r}")
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def build_design_features(U_scaled: np.ndarray, degree: int) -> np.ndarray:
    n = U_scaled.shape[0]
    if degree == 0:
        return np.ones((n, 1), dtype=np.float64)
    if degree == 1:
        return np.column_stack([np.ones(n, dtype=np.float64), U_scaled])
    raise ValueError(f"Unsupported degree {degree}")


def local_design_stability(U_scaled: np.ndarray, degree: int) -> float:
    if degree == 0:
        return 1.0
    Phi = build_design_features(U_scaled, degree=degree)
    gram = (Phi.T @ Phi) / float(Phi.shape[0])
    eigvals = np.linalg.eigvalsh(gram)
    return float(np.min(eigvals))


def compute_prediction_weights(
    X_local: np.ndarray,
    x0: np.ndarray,
    degree: int,
    ridge: float,
) -> np.ndarray:
    U = np.asarray(X_local, dtype=np.float64) - x0[None, :]
    if degree == 0:
        return np.full(U.shape[0], 1.0 / float(U.shape[0]), dtype=np.float64)

    radii = np.linalg.norm(U, axis=1)
    r_emp = float(max(np.max(radii), RIDGE_EPS))
    U_scaled = U / r_emp
    Phi = build_design_features(U_scaled, degree=degree)
    gram = Phi.T @ Phi
    gram += float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    solution = np.linalg.solve(gram, Phi.T)
    return np.asarray(solution[0], dtype=np.float64)


def compute_nearest_neighbor_weights(X_local: np.ndarray, x0: np.ndarray) -> np.ndarray:
    U = np.asarray(X_local, dtype=np.float64) - x0[None, :]
    idx = int(np.argmin(np.linalg.norm(U, axis=1)))
    weights = np.zeros(U.shape[0], dtype=np.float64)
    weights[idx] = 1.0
    return weights


def generate_stable_local_design(
    *,
    x0: np.ndarray,
    h: float,
    k: int,
    degree: int,
    geometry: str,
    rng: np.random.Generator,
    min_eig_threshold: float = STABILITY_EIG_FLOOR,
    max_tries: int = 4000,
) -> Tuple[np.ndarray, float, float]:
    for _ in range(max_tries):
        U_unit = sample_unit_vectors(n=int(k), rng=rng, geometry=geometry)
        X_local = x0[None, :] + float(h) * U_unit
        if np.any(X_local < DOMAIN_LOW) or np.any(X_local > DOMAIN_HIGH):
            continue
        radii = np.linalg.norm(X_local - x0[None, :], axis=1)
        r_emp = float(max(np.max(radii), RIDGE_EPS))
        U_scaled = (X_local - x0[None, :]) / r_emp
        min_eig = local_design_stability(U_scaled, degree=degree)
        if degree == 0 or min_eig >= float(min_eig_threshold):
            return X_local, r_emp, float(min_eig)
    raise RuntimeError(
        f"Could not sample a stable {geometry} neighborhood for h={h}, k={k}, degree={degree}."
    )


def fit_loglog_line(x_values: np.ndarray, y_values: np.ndarray) -> Dict[str, float]:
    x = np.maximum(np.asarray(x_values, dtype=np.float64), RIDGE_EPS)
    y = np.maximum(np.asarray(y_values, dtype=np.float64), RIDGE_EPS)
    log_x = np.log(x)
    log_y = np.log(y)
    X = np.column_stack([np.ones(log_x.shape[0], dtype=np.float64), log_x])
    coef, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    fitted = X @ coef
    residual = log_y - fitted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0
    if X.shape[0] > 2:
        sigma2 = sse / float(X.shape[0] - 2)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        slope_se = float(math.sqrt(max(cov[1, 1], 0.0)))
    else:
        slope_se = 0.0
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "r2": float(r2),
        "slope_se": float(slope_se),
    }


def fit_no_intercept_line(x_values: np.ndarray, y_values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    denom = float(np.dot(x, x))
    coef = float(np.dot(x, y) / denom) if denom > 0.0 else 0.0
    fitted = coef * x
    sse = float(np.sum((y - fitted) ** 2))
    sst0 = float(np.sum(y**2))
    r2 = 1.0 - sse / sst0 if sst0 > 0.0 else 1.0
    return {"coef": float(coef), "r2": float(r2), "sse": float(sse), "sst0": float(sst0)}


def fit_two_term_model(rows: Sequence[Dict[str, float | int | str]], s: int, method: str) -> Dict[str, float]:
    subset = [
        row
        for row in rows
        if str(row["method"]) == method and int(row["smoothness_s"]) == int(s)
    ]
    if not subset:
        raise ValueError(f"No two-term rows for method={method}, s={s}")
    X = np.asarray(
        [
            [float(row["empirical_rk"]) ** (2 * int(s)), float(row["variance_term"])]
            for row in subset
        ],
        dtype=np.float64,
    )
    y = np.asarray([float(row["mse"]) for row in subset], dtype=np.float64)
    try:
        reg = LinearRegression(fit_intercept=True, positive=True)
        reg.fit(X, y)
        intercept = float(reg.intercept_)
        coefs = np.asarray(reg.coef_, dtype=np.float64)
        fitted = reg.predict(X)
    except TypeError:
        X_aug = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X])
        coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        intercept = float(coef[0])
        coefs = np.asarray(coef[1:], dtype=np.float64)
        fitted = X_aug @ coef

    sse = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0
    return {
        "c0": float(intercept),
        "c_h": float(coefs[0]),
        "c_v": float(coefs[1]),
        "r2": float(r2),
        "n_obs": int(y.shape[0]),
    }


def compute_rate_summary_statistics(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.shape[0] > 1 else 0.0,
    }


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_rate_controlled_results(args, geometry: str) -> Dict[str, object]:
    approx_reps = int(args.approx_reps)
    approx_k = int(args.approx_k)
    variance_design_reps = int(args.variance_design_reps)
    variance_noise_reps = int(args.variance_noise_reps)
    two_term_design_reps = int(args.two_term_design_reps)
    two_term_noise_reps = int(args.two_term_noise_reps)
    ridge = float(args.ridge)

    approx_rows: List[Dict[str, object]] = []
    variance_rows: List[Dict[str, object]] = []
    two_term_rows: List[Dict[str, object]] = []
    stability_rows: List[Dict[str, object]] = []

    true_x0 = {s: float(target_rate_controlled(X0[None, :], s=s)[0]) for s in S_VALUES}

    for s in S_VALUES:
        degree = degree_for_s(s)
        for rep in range(approx_reps):
            rng = np.random.default_rng(int(args.seed) + 100_000 * int(s) + rep)
            for h_idx, h in enumerate(APPROX_H_GRID):
                X_local, r_emp, min_eig = generate_stable_local_design(
                    x0=X0,
                    h=float(h),
                    k=approx_k,
                    degree=degree,
                    geometry=geometry,
                    rng=rng,
                )
                y_clean = target_rate_controlled(X_local, s=s)
                lp_weights = compute_prediction_weights(X_local, X0, degree=degree, ridge=ridge)
                nn_weights = compute_nearest_neighbor_weights(X_local, X0)
                for method, weights in (("local_poly", lp_weights), ("nearest_neighbor", nn_weights)):
                    prediction = float(np.dot(weights, y_clean))
                    mse = float((prediction - true_x0[s]) ** 2)
                    approx_rows.append(
                        {
                            "setting": "rate_controlled",
                            "geometry": geometry,
                            "component": "approximation",
                            "method": method,
                            "smoothness_s": int(s),
                            "degree": int(degree),
                            "repetition": int(rep),
                            "nominal_h": float(h),
                            "empirical_rk": float(r_emp),
                            "k": int(approx_k),
                            "mse": float(mse),
                            "prediction": float(prediction),
                            "true_value": float(true_x0[s]),
                            "min_eig": float(min_eig),
                        }
                    )
                stability_rows.append(
                    {
                        "setting": "rate_controlled",
                        "geometry": geometry,
                        "component": "approximation",
                        "smoothness_s": int(s),
                        "degree": int(degree),
                        "nominal_h": float(h),
                        "k": int(approx_k),
                        "repetition": int(rep),
                        "empirical_rk": float(r_emp),
                        "min_eig": float(min_eig),
                    }
                )

    for s in S_VALUES:
        degree = degree_for_s(s)
        for k in VARIANCE_K_GRID:
            for rep in range(variance_design_reps):
                rng = np.random.default_rng(int(args.seed) + 200_000 * int(s) + 1_000 * int(k) + rep)
                X_local, r_emp, min_eig = generate_stable_local_design(
                    x0=X0,
                    h=VARIANCE_H_FIXED,
                    k=int(k),
                    degree=degree,
                    geometry=geometry,
                    rng=rng,
                )
                y_clean = target_rate_controlled(X_local, s=s)
                noise_draws = rng.normal(size=(variance_noise_reps, int(k)))
                lp_weights = compute_prediction_weights(X_local, X0, degree=degree, ridge=ridge)
                nn_weights = compute_nearest_neighbor_weights(X_local, X0)
                for method, weights in (("local_poly", lp_weights), ("nearest_neighbor", nn_weights)):
                    pred_clean = float(np.dot(weights, y_clean))
                    mse_sigma0 = float((pred_clean - true_x0[s]) ** 2)
                    noise_projection = np.asarray(noise_draws @ weights, dtype=np.float64)
                    for sigma in VARIANCE_SIGMA_GRID:
                        noisy_preds = pred_clean + float(sigma) * noise_projection
                        mse_sigma = float(np.mean((noisy_preds - true_x0[s]) ** 2))
                        variance_rows.append(
                            {
                                "setting": "rate_controlled",
                                "geometry": geometry,
                                "component": "variance",
                                "method": method,
                                "smoothness_s": int(s),
                                "degree": int(degree),
                                "design_rep": int(rep),
                                "nominal_h": float(VARIANCE_H_FIXED),
                                "empirical_rk": float(r_emp),
                                "k": int(k),
                                "sigma": float(sigma),
                                "variance_term": float((float(sigma) ** 2) / float(k)),
                                "mse_sigma0": float(mse_sigma0),
                                "mse_sigma": float(mse_sigma),
                                "excess_mse": float(mse_sigma - mse_sigma0),
                                "min_eig": float(min_eig),
                            }
                        )
                stability_rows.append(
                    {
                        "setting": "rate_controlled",
                        "geometry": geometry,
                        "component": "variance",
                        "smoothness_s": int(s),
                        "degree": int(degree),
                        "nominal_h": float(VARIANCE_H_FIXED),
                        "k": int(k),
                        "repetition": int(rep),
                        "empirical_rk": float(r_emp),
                        "min_eig": float(min_eig),
                    }
                )

    for s in S_VALUES:
        degree = degree_for_s(s)
        for h in APPROX_H_GRID:
            for k in VARIANCE_K_GRID:
                for rep in range(two_term_design_reps):
                    rng = np.random.default_rng(
                        int(args.seed) + 300_000 * int(s) + 10_000 * int(round(1000 * float(h))) + 100 * int(k) + rep
                    )
                    X_local, r_emp, min_eig = generate_stable_local_design(
                        x0=X0,
                        h=float(h),
                        k=int(k),
                        degree=degree,
                        geometry=geometry,
                        rng=rng,
                    )
                    y_clean = target_rate_controlled(X_local, s=s)
                    noise_draws = rng.normal(size=(two_term_noise_reps, int(k)))
                    lp_weights = compute_prediction_weights(X_local, X0, degree=degree, ridge=ridge)
                    nn_weights = compute_nearest_neighbor_weights(X_local, X0)
                    for method, weights in (("local_poly", lp_weights), ("nearest_neighbor", nn_weights)):
                        pred_clean = float(np.dot(weights, y_clean))
                        noise_projection = np.asarray(noise_draws @ weights, dtype=np.float64)
                        for sigma in TWO_TERM_SIGMA_GRID:
                            noisy_preds = pred_clean + float(sigma) * noise_projection
                            mse = float(np.mean((noisy_preds - true_x0[s]) ** 2))
                            two_term_rows.append(
                                {
                                    "setting": "rate_controlled",
                                    "geometry": geometry,
                                    "component": "two_term",
                                    "method": method,
                                    "smoothness_s": int(s),
                                    "degree": int(degree),
                                    "design_rep": int(rep),
                                    "nominal_h": float(h),
                                    "empirical_rk": float(r_emp),
                                    "k": int(k),
                                    "sigma": float(sigma),
                                    "variance_term": float((float(sigma) ** 2) / float(k)),
                                    "mse": float(mse),
                                    "min_eig": float(min_eig),
                                }
                            )
                    stability_rows.append(
                        {
                            "setting": "rate_controlled",
                            "geometry": geometry,
                            "component": "two_term",
                            "smoothness_s": int(s),
                            "degree": int(degree),
                            "nominal_h": float(h),
                            "k": int(k),
                            "repetition": int(rep),
                            "empirical_rk": float(r_emp),
                            "min_eig": float(min_eig),
                        }
                    )

    approx_fit_rows: List[Dict[str, object]] = []
    approx_curve_rows: List[Dict[str, object]] = []
    approx_summary: Dict[Tuple[str, int], Dict[str, float]] = {}
    for method in ("local_poly", "nearest_neighbor"):
        for s in S_VALUES:
            rep_rows = []
            for rep in range(approx_reps):
                subset = [
                    row
                    for row in approx_rows
                    if str(row["method"]) == method
                    and int(row["smoothness_s"]) == int(s)
                    and int(row["repetition"]) == int(rep)
                ]
                fit = fit_loglog_line(
                    np.asarray([float(row["empirical_rk"]) for row in subset], dtype=np.float64),
                    np.asarray([float(row["mse"]) for row in subset], dtype=np.float64),
                )
                rep_rows.append({"slope": float(fit["slope"]), "r2": float(fit["r2"]), "slope_se": float(fit["slope_se"])})
                approx_fit_rows.append(
                    {
                        "setting": "rate_controlled",
                        "geometry": geometry,
                        "method": method,
                        "smoothness_s": int(s),
                        "repetition": int(rep),
                        "approx_slope": float(fit["slope"]),
                        "approx_r2": float(fit["r2"]),
                        "regression_slope_se": float(fit["slope_se"]),
                    }
                )
            slopes = np.asarray([float(row["slope"]) for row in rep_rows], dtype=np.float64)
            r2s = np.asarray([float(row["r2"]) for row in rep_rows], dtype=np.float64)
            for h in APPROX_H_GRID:
                subset = [
                    row
                    for row in approx_rows
                    if str(row["method"]) == method
                    and int(row["smoothness_s"]) == int(s)
                    and math.isclose(float(row["nominal_h"]), float(h), rel_tol=0.0, abs_tol=1e-12)
                ]
                approx_curve_rows.append(
                    {
                        "setting": "rate_controlled",
                        "geometry": geometry,
                        "method": method,
                        "smoothness_s": int(s),
                        "nominal_h": float(h),
                        "empirical_rk": float(np.median([float(row["empirical_rk"]) for row in subset])),
                        "mean_mse": float(np.mean([float(row["mse"]) for row in subset])),
                        "std_mse": float(np.std([float(row["mse"]) for row in subset], ddof=0)),
                    }
                )
            curve_subset = [
                row
                for row in approx_curve_rows
                if str(row["method"]) == method and int(row["smoothness_s"]) == int(s)
            ]
            aggregate_fit = fit_loglog_line(
                np.asarray([float(row["empirical_rk"]) for row in curve_subset], dtype=np.float64),
                np.asarray([float(row["mean_mse"]) for row in curve_subset], dtype=np.float64),
            )
            approx_summary[(method, s)] = {
                "mean_slope": float(np.mean(slopes)),
                "slope_se_across_reps": float(np.std(slopes, ddof=1) / math.sqrt(slopes.shape[0])) if slopes.shape[0] > 1 else 0.0,
                "mean_r2": float(np.mean(r2s)),
                "aggregate_slope": float(aggregate_fit["slope"]),
                "aggregate_r2": float(aggregate_fit["r2"]),
                "predicted_slope": float(2 * int(s)),
            }

    variance_mean_rows: List[Dict[str, object]] = []
    variance_summary: Dict[Tuple[str, int], Dict[str, float]] = {}
    for method in ("local_poly", "nearest_neighbor"):
        for s in S_VALUES:
            subset_rows = [
                row
                for row in variance_rows
                if str(row["method"]) == method and int(row["smoothness_s"]) == int(s)
            ]
            for k in VARIANCE_K_GRID:
                for sigma in VARIANCE_SIGMA_GRID:
                    cur = [
                        row
                        for row in subset_rows
                        if int(row["k"]) == int(k)
                        and math.isclose(float(row["sigma"]), float(sigma), rel_tol=0.0, abs_tol=1e-12)
                    ]
                    mean_excess = float(np.mean([float(row["excess_mse"]) for row in cur]))
                    variance_mean_rows.append(
                        {
                            "setting": "rate_controlled",
                            "geometry": geometry,
                            "method": method,
                            "smoothness_s": int(s),
                            "k": int(k),
                            "sigma": float(sigma),
                            "variance_term": float((float(sigma) ** 2) / float(k)),
                            "mean_excess_mse": float(mean_excess),
                            "mean_excess_mse_clipped": float(max(mean_excess, 0.0)),
                            "std_excess_mse": float(np.std([float(row["excess_mse"]) for row in cur], ddof=0)),
                        }
                    )
            fit_subset = [
                row for row in variance_mean_rows if str(row["method"]) == method and int(row["smoothness_s"]) == int(s)
            ]
            fit = fit_no_intercept_line(
                np.asarray([float(row["variance_term"]) for row in fit_subset], dtype=np.float64),
                np.asarray([float(row["mean_excess_mse"]) for row in fit_subset], dtype=np.float64),
            )
            variance_summary[(method, s)] = {
                "variance_coef": float(fit["coef"]),
                "variance_r2": float(fit["r2"]),
            }

    two_term_mean_rows: List[Dict[str, object]] = []
    two_term_summary: Dict[Tuple[str, int], Dict[str, float]] = {}
    for method in ("local_poly", "nearest_neighbor"):
        for s in S_VALUES:
            subset_rows = [
                row
                for row in two_term_rows
                if str(row["method"]) == method and int(row["smoothness_s"]) == int(s)
            ]
            for h in APPROX_H_GRID:
                for k in VARIANCE_K_GRID:
                    for sigma in TWO_TERM_SIGMA_GRID:
                        cur = [
                            row
                            for row in subset_rows
                            if math.isclose(float(row["nominal_h"]), float(h), rel_tol=0.0, abs_tol=1e-12)
                            and int(row["k"]) == int(k)
                            and math.isclose(float(row["sigma"]), float(sigma), rel_tol=0.0, abs_tol=1e-12)
                        ]
                        two_term_mean_rows.append(
                            {
                                "setting": "rate_controlled",
                                "geometry": geometry,
                                "method": method,
                                "smoothness_s": int(s),
                                "nominal_h": float(h),
                                "k": int(k),
                                "sigma": float(sigma),
                                "empirical_rk": float(np.median([float(row["empirical_rk"]) for row in cur])),
                                "variance_term": float((float(sigma) ** 2) / float(k)),
                                "mse": float(np.mean([float(row["mse"]) for row in cur])),
                            }
                        )
            fit = fit_two_term_model(two_term_mean_rows, s=s, method=method)
            two_term_summary[(method, s)] = fit

    stability_summary_rows: List[Dict[str, object]] = []
    for component in ("approximation", "variance", "two_term"):
        for s in S_VALUES:
            subset = [
                float(row["min_eig"])
                for row in stability_rows
                if str(row["component"]) == component and int(row["smoothness_s"]) == int(s)
            ]
            stats = compute_rate_summary_statistics(subset)
            stability_summary_rows.append(
                {
                    "setting": "rate_controlled",
                    "geometry": geometry,
                    "component": component,
                    "smoothness_s": int(s),
                    "degree": int(degree_for_s(s)),
                    "min_eig_min": float(stats["min"]),
                    "min_eig_p05": float(stats["p05"]),
                    "min_eig_median": float(stats["median"]),
                    "min_eig_mean": float(stats["mean"]),
                    "n_designs": int(len(subset)),
                }
            )

    main_table_rows: List[Dict[str, object]] = []
    two_term_table_rows: List[Dict[str, object]] = []
    for method in ("local_poly", "nearest_neighbor"):
        display = "Local polynomial" if method == "local_poly" else "Nearest neighbor"
        for s in S_VALUES:
            main_table_rows.append(
                {
                    "method": display,
                    "smoothness_s": int(s),
                    "approx_slope": float(approx_summary[(method, s)]["mean_slope"]),
                    "approx_slope_se": float(approx_summary[(method, s)]["slope_se_across_reps"]),
                    "predicted_slope": float(approx_summary[(method, s)]["predicted_slope"]),
                    "approx_r2": float(approx_summary[(method, s)]["aggregate_r2"]),
                    "variance_r2": float(variance_summary[(method, s)]["variance_r2"]),
                    "two_term_r2": float(two_term_summary[(method, s)]["r2"]),
                }
            )
            two_term_table_rows.append(
                {
                    "method": display,
                    "smoothness_s": int(s),
                    "c0": float(two_term_summary[(method, s)]["c0"]),
                    "c_h": float(two_term_summary[(method, s)]["c_h"]),
                    "c_v": float(two_term_summary[(method, s)]["c_v"]),
                    "two_term_r2": float(two_term_summary[(method, s)]["r2"]),
                }
            )

    return {
        "setting_name": "rate_controlled",
        "geometry": geometry,
        "approx_rows": approx_rows,
        "approx_fit_rows": approx_fit_rows,
        "approx_curve_rows": approx_curve_rows,
        "variance_rows": variance_rows,
        "variance_mean_rows": variance_mean_rows,
        "two_term_rows": two_term_rows,
        "two_term_mean_rows": two_term_mean_rows,
        "stability_rows": stability_rows,
        "stability_summary_rows": stability_summary_rows,
        "main_table_rows": main_table_rows,
        "two_term_table_rows": two_term_table_rows,
        "approx_summary": approx_summary,
        "variance_summary": variance_summary,
        "two_term_summary": two_term_summary,
    }


def save_rate_controlled_main_table_tex(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Smoothness $s$ & Approx. slope & Pred. slope & Approx. $R^2$ & Var.-fit $R^2$ & Two-term $R^2$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & $s={int(row['smoothness_s'])}$ & "
            f"{float(row['approx_slope']):.3f} $\\pm$ {float(row['approx_slope_se']):.3f} & "
            f"{float(row['predicted_slope']):.0f} & "
            f"{float(row['approx_r2']):.3f} & "
            f"{float(row['variance_r2']):.3f} & "
            f"{float(row['two_term_r2']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_rate_controlled_two_term_table_tex(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Smoothness $s$ & $c_0$ & $c_h$ & $c_v$ & Two-term $R^2$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & $s={int(row['smoothness_s'])}$ & "
            f"{float(row['c0']):.5f} & {float(row['c_h']):.5f} & {float(row['c_v']):.5f} & "
            f"{float(row['two_term_r2']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_rate_controlled_design_stability_table_tex(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Component & Smoothness $s$ & Geometry & 5th pct. min eig & Median min eig & Min min eig \\\\",
        "\\midrule",
    ]
    for row in rows:
        component_label = {
            "approximation": "Approximation",
            "variance": "Variance",
            "two_term": "Two-term",
        }[str(row["component"])]
        lines.append(
            f"{component_label} & $s={int(row['smoothness_s'])}$ & {row['geometry'].replace('_', ' ')} & "
            f"{float(row['min_eig_p05']):.4f} & {float(row['min_eig_median']):.4f} & {float(row['min_eig_min']):.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def make_rate_controlled_approximation_figure(
    figures_dir: Path,
    prefix: str,
    curve_rows: Sequence[Dict[str, object]],
    summary: Dict[Tuple[str, int], Dict[str, float]],
) -> None:
    colors = {"local_poly": "#4c78a8", "nearest_neighbor": "#e45756", "reference": "#222222"}
    markers = {"local_poly": "o", "nearest_neighbor": "s", "reference": "D"}
    linestyles = {"local_poly": "-", "nearest_neighbor": "--", "reference": "-."}
    labels = {"local_poly": "Local polynomial", "nearest_neighbor": "Nearest neighbor"}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, s in zip(axes, S_VALUES):
        pooled_x: List[np.ndarray] = []
        pooled_y: List[np.ndarray] = []
        for method in ("local_poly", "nearest_neighbor"):
            subset = [
                row
                for row in curve_rows
                if str(row["method"]) == method and int(row["smoothness_s"]) == int(s)
            ]
            x = np.asarray([float(row["empirical_rk"]) for row in subset], dtype=np.float64)
            y = np.maximum(np.asarray([float(row["mean_mse"]) for row in subset], dtype=np.float64), RIDGE_EPS)
            pooled_x.append(x)
            pooled_y.append(y)
            fit = fit_loglog_line(x, y)
            line_x = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
            line_y = np.exp(float(fit["intercept"])) * (line_x ** float(fit["slope"]))
            ax.plot(
                x,
                y,
                color=colors[method],
                marker=markers[method],
                linestyle=linestyles[method],
                linewidth=1.8,
                markersize=5.0,
                label=labels[method],
            )
            ax.plot(
                line_x,
                line_y,
                color=colors[method],
                marker=markers[method],
                linestyle=linestyles[method],
                markevery=38,
                linewidth=2.1,
                alpha=0.9,
            )
        x_all = np.concatenate(pooled_x, axis=0)
        y_all = np.concatenate(pooled_y, axis=0)
        ref_intercept = float(np.mean(np.log(y_all)) - (2 * int(s)) * np.mean(np.log(x_all)))
        ref_x = np.geomspace(float(np.min(x_all)), float(np.max(x_all)), 200)
        ref_y = np.exp(ref_intercept) * (ref_x ** (2 * int(s)))
        ax.plot(
            ref_x,
            ref_y,
            color=colors["reference"],
            marker=markers["reference"],
            linestyle=linestyles["reference"],
            markevery=42,
            linewidth=1.7,
            label=r"Reference slope $2s$",
        )
        text = (
            rf"LP slope $={summary[('local_poly', s)]['mean_slope']:.2f}\pm{summary[('local_poly', s)]['slope_se_across_reps']:.2f}$" "\n"
            rf"NN slope $={summary[('nearest_neighbor', s)]['mean_slope']:.2f}\pm{summary[('nearest_neighbor', s)]['slope_se_across_reps']:.2f}$"
        )
        ax.text(
            0.04,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Empirical support radius $r_k(x_0)$")
        if ax is axes[0]:
            ax.set_ylabel("Approximation-only MSE")
        ax.set_title(rf"Approximation diagnostic ($s={s}$)")
        ax.grid(alpha=0.18, which="both")

    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.savefig(figures_dir / f"{prefix}_approximation_slopes.pdf")
    for ax, s in zip(axes, S_VALUES):
        save_axes_group_panel(fig, [ax], figures_dir / f"{prefix}_approximation_slopes_panel_s{s}.pdf")
    save_legend_figure(handles, labels_out, figures_dir / f"{prefix}_approximation_slopes_legend.pdf", ncol=3)
    plt.close(fig)


def make_rate_controlled_variance_figure(
    figures_dir: Path,
    prefix: str,
    mean_rows: Sequence[Dict[str, object]],
    summary: Dict[Tuple[str, int], Dict[str, float]],
) -> None:
    colors = {"local_poly": "#4c78a8", "nearest_neighbor": "#e45756"}
    markers_by_k = {10: "o", 20: "s", 40: "^", 80: "D", 160: "P"}
    lines = {"local_poly": "-", "nearest_neighbor": "--"}
    line_markers = {"local_poly": "X", "nearest_neighbor": "v"}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, s in zip(axes, S_VALUES):
        for method in ("local_poly", "nearest_neighbor"):
            subset = [
                row
                for row in mean_rows
                if str(row["method"]) == method and int(row["smoothness_s"]) == int(s)
            ]
            for row in subset:
                ax.scatter(
                    float(row["variance_term"]),
                    max(float(row["mean_excess_mse"]), 0.0),
                    color=colors[method],
                    marker=markers_by_k[int(row["k"])],
                    s=42,
                    alpha=0.85,
                    edgecolors="none",
                )
            coef = float(summary[(method, s)]["variance_coef"])
            x = np.linspace(0.0, max(float(row["variance_term"]) for row in subset) * 1.05, 200)
            ax.plot(
                x,
                coef * x,
                color=colors[method],
                marker=line_markers[method],
                linestyle=lines[method],
                linewidth=2.0,
                markevery=30,
                label="Local polynomial" if method == "local_poly" else "Nearest neighbor",
            )
        text = (
            rf"LP $R^2={summary[('local_poly', s)]['variance_r2']:.3f}$" "\n"
            rf"NN $R^2={summary[('nearest_neighbor', s)]['variance_r2']:.3f}$"
        )
        ax.text(
            0.04,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        ax.set_xlabel(r"$\sigma^2/k$")
        if ax is axes[0]:
            ax.set_ylabel("Excess MSE")
        ax.set_title(rf"Variance diagnostic ($s={s}$)")
        ax.grid(alpha=0.18)

    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.savefig(figures_dir / f"{prefix}_variance_fit.pdf")
    for ax, s in zip(axes, S_VALUES):
        save_axes_group_panel(fig, [ax], figures_dir / f"{prefix}_variance_fit_panel_s{s}.pdf")
    save_legend_figure(handles, labels_out, figures_dir / f"{prefix}_variance_fit_legend.pdf", ncol=2)
    plt.close(fig)


def make_rate_controlled_two_term_figure(
    figures_dir: Path,
    prefix: str,
    rows: Sequence[Dict[str, object]],
    summary: Dict[Tuple[str, int], Dict[str, float]],
) -> None:
    colors = {"local_poly": "#4c78a8", "nearest_neighbor": "#e45756", "diag": "#222222"}
    markers = {"local_poly": "o", "nearest_neighbor": "s", "diag": "D"}
    linestyles = {"local_poly": "-", "nearest_neighbor": "--", "diag": "-."}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, s in zip(axes, S_VALUES):
        plotted_values: List[np.ndarray] = []
        for method in ("local_poly", "nearest_neighbor"):
            subset = [
                row
                for row in rows
                if str(row["method"]) == method and int(row["smoothness_s"]) == int(s)
            ]
            fit = summary[(method, s)]
            predicted = np.asarray(
                [
                    float(fit["c0"])
                    + float(fit["c_h"]) * (float(row["empirical_rk"]) ** (2 * int(s)))
                    + float(fit["c_v"]) * float(row["variance_term"])
                    for row in subset
                ],
                dtype=np.float64,
            )
            observed = np.maximum(np.asarray([float(row["mse"]) for row in subset], dtype=np.float64), RIDGE_EPS)
            predicted = np.maximum(predicted, RIDGE_EPS)
            plotted_values.extend([predicted, observed])
            ax.scatter(
                predicted,
                observed,
                color=colors[method],
                marker=markers[method],
                s=16,
                alpha=0.25,
                edgecolors="none",
                label="Local polynomial" if method == "local_poly" else "Nearest neighbor",
            )
        diag_vals = np.concatenate(plotted_values, axis=0)
        lo = float(np.min(diag_vals))
        hi = float(np.max(diag_vals))
        diag = np.geomspace(lo, hi, 200)
        ax.plot(
            diag,
            diag,
            color=colors["diag"],
            marker=markers["diag"],
            linestyle=linestyles["diag"],
            markevery=38,
            linewidth=1.7,
            label="Diagonal",
        )
        text = (
            rf"LP $R^2={summary[('local_poly', s)]['r2']:.3f}$" "\n"
            rf"NN $R^2={summary[('nearest_neighbor', s)]['r2']:.3f}$"
        )
        ax.text(
            0.04,
            0.93,
            text,
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Predicted MSE from two-term fit")
        if ax is axes[0]:
            ax.set_ylabel("Observed MSE")
        ax.set_title(rf"Two-term diagnostic ($s={s}$)")
        ax.grid(alpha=0.18, which="both")

    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.savefig(figures_dir / f"{prefix}_two_term_pred_vs_obs.pdf")
    for ax, s in zip(axes, S_VALUES):
        save_axes_group_panel(fig, [ax], figures_dir / f"{prefix}_two_term_pred_vs_obs_panel_s{s}.pdf")
    save_legend_figure(handles, labels_out, figures_dir / f"{prefix}_two_term_pred_vs_obs_legend.pdf", ncol=3)
    plt.close(fig)


def write_rate_controlled_outputs(
    bundle: Dict[str, object],
    *,
    args,
    figures_dir: Path,
    tables_dir: Path,
    results_dir: Path,
    prefix: str,
    initial_geometry: str,
    fallback_triggered: bool,
) -> Dict[str, object]:
    approx_rows = bundle["approx_rows"]
    variance_rows = bundle["variance_rows"]
    two_term_rows = bundle["two_term_rows"]
    stability_rows = bundle["stability_summary_rows"]
    main_table_rows = bundle["main_table_rows"]
    two_term_table_rows = bundle["two_term_table_rows"]

    save_csv(results_dir / f"{prefix}_approximation.csv", approx_rows)
    save_csv(results_dir / f"{prefix}_variance.csv", variance_rows)
    save_csv(results_dir / f"{prefix}_two_term_fit.csv", two_term_table_rows)
    save_csv(results_dir / f"{prefix}_design_stability.csv", stability_rows)

    save_rate_controlled_main_table_tex(tables_dir / f"{prefix}_main_table.tex", main_table_rows)
    save_rate_controlled_two_term_table_tex(tables_dir / f"{prefix}_two_term_table.tex", two_term_table_rows)
    save_rate_controlled_design_stability_table_tex(
        tables_dir / f"{prefix}_design_stability_table.tex",
        stability_rows,
    )

    make_rate_controlled_approximation_figure(
        figures_dir=figures_dir,
        prefix=prefix,
        curve_rows=bundle["approx_curve_rows"],
        summary=bundle["approx_summary"],
    )
    make_rate_controlled_variance_figure(
        figures_dir=figures_dir,
        prefix=prefix,
        mean_rows=bundle["variance_mean_rows"],
        summary=bundle["variance_summary"],
    )
    make_rate_controlled_two_term_figure(
        figures_dir=figures_dir,
        prefix=prefix,
        rows=bundle["two_term_mean_rows"],
        summary=bundle["two_term_summary"],
    )

    approx_summary_json = {
        f"{method}_s{s}": {
            "approx_slope_mean": float(summary["mean_slope"]),
            "approx_slope_se": float(summary["slope_se_across_reps"]),
            "approx_r2_mean": float(summary["mean_r2"]),
            "approx_r2_aggregate": float(summary["aggregate_r2"]),
            "approx_slope_aggregate": float(summary["aggregate_slope"]),
            "predicted_slope": float(summary["predicted_slope"]),
        }
        for (method, s), summary in bundle["approx_summary"].items()
    }
    variance_summary_json = {
        f"{method}_s{s}": {
            "variance_coef": float(summary["variance_coef"]),
            "variance_r2": float(summary["variance_r2"]),
        }
        for (method, s), summary in bundle["variance_summary"].items()
    }
    two_term_summary_json = {
        f"{method}_s{s}": {
            "c0": float(summary["c0"]),
            "c_h": float(summary["c_h"]),
            "c_v": float(summary["c_v"]),
            "two_term_r2": float(summary["r2"]),
        }
        for (method, s), summary in bundle["two_term_summary"].items()
    }

    summary_payload = {
        "experiment": "exp3_rate_controlled",
        "initial_geometry": initial_geometry,
        "selected_geometry": bundle["geometry"],
        "fallback_triggered": bool(fallback_triggered),
        "seed": int(args.seed),
        "approx_k": int(args.approx_k),
        "approx_reps": int(args.approx_reps),
        "variance_design_reps": int(args.variance_design_reps),
        "variance_noise_reps": int(args.variance_noise_reps),
        "two_term_design_reps": int(args.two_term_design_reps),
        "two_term_noise_reps": int(args.two_term_noise_reps),
        "h_grid": [float(x) for x in APPROX_H_GRID],
        "variance_h_fixed": float(VARIANCE_H_FIXED),
        "k_grid": [int(x) for x in VARIANCE_K_GRID],
        "variance_sigma_grid": [float(x) for x in VARIANCE_SIGMA_GRID],
        "two_term_sigma_grid": [float(x) for x in TWO_TERM_SIGMA_GRID],
        "approximation_summary": approx_summary_json,
        "variance_summary": variance_summary_json,
        "two_term_summary": two_term_summary_json,
        "design_stability_summary": stability_rows,
        "interpretation_notes": [
            "The rate-controlled diagnostic separates approximation-only and variance-only behavior.",
            "One-sided stable local neighborhoods are used to avoid accidental first-order cancellation in the s=1 local-constant case.",
            "Fitted slopes are diagnostics for the upper-bound rate structure, not exact finite-sample equalities.",
        ],
        "output_paths": {
            "approximation_csv": str((results_dir / f"{prefix}_approximation.csv").resolve()),
            "variance_csv": str((results_dir / f"{prefix}_variance.csv").resolve()),
            "two_term_csv": str((results_dir / f"{prefix}_two_term_fit.csv").resolve()),
            "design_stability_csv": str((results_dir / f"{prefix}_design_stability.csv").resolve()),
            "main_table_tex": str((tables_dir / f"{prefix}_main_table.tex").resolve()),
            "two_term_table_tex": str((tables_dir / f"{prefix}_two_term_table.tex").resolve()),
            "design_stability_table_tex": str((tables_dir / f"{prefix}_design_stability_table.tex").resolve()),
            "approximation_figure": str((figures_dir / f"{prefix}_approximation_slopes.pdf").resolve()),
            "variance_figure": str((figures_dir / f"{prefix}_variance_fit.pdf").resolve()),
            "two_term_figure": str((figures_dir / f"{prefix}_two_term_pred_vs_obs.pdf").resolve()),
        },
    }
    summary_path = results_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
    return summary_payload


def run_rate_controlled(args, outdir: Path) -> Dict[str, object]:
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    results_dir = outdir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    initial_geometry = str(args.rate_geometry)
    bundle = collect_rate_controlled_results(args, geometry=initial_geometry)
    slope_s1_lp = float(bundle["approx_summary"][("local_poly", 1)]["mean_slope"])
    fallback_triggered = False

    if (
        slope_s1_lp > 2.8
        and initial_geometry == "half_ball"
        and not bool(getattr(args, "disable_cone_fallback", False))
    ):
        fallback_bundle = collect_rate_controlled_results(args, geometry="cone")
        fallback_slope = float(fallback_bundle["approx_summary"][("local_poly", 1)]["mean_slope"])
        if 1.8 <= fallback_slope <= 2.8:
            fallback_triggered = True
            write_rate_controlled_outputs(
                bundle,
                args=args,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
                results_dir=results_dir,
                prefix="exp3_rate_controlled_halfball",
                initial_geometry=initial_geometry,
                fallback_triggered=False,
            )
            bundle = fallback_bundle

    summary_payload = write_rate_controlled_outputs(
        bundle,
        args=args,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        results_dir=results_dir,
        prefix="exp3_rate_controlled",
        initial_geometry=initial_geometry,
        fallback_triggered=fallback_triggered,
    )
    return summary_payload
