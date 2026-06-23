from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import exp6_model_selection as exp6
from experiments import exp8_real_data as exp8


SUPPORT_K_VALUES = (5, 10, 20, 40)
K_BINS_VALUES = (4, 5, 6, 8)
BINNING_SCHEMES = ("quantile", "log_width")
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supplementary sensitivity analysis for support-field construction in Experiments 6 and 8."
    )
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-exp6", action="store_true")
    parser.add_argument("--run-exp8", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--exp8-data-dir", type=str, default="")
    parser.add_argument("--exp8-models", type=str, default="krr,rf,gbrt")
    return parser.parse_args()


def resolve_outdir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def ensure_dirs(outdir: Path) -> tuple[Path, Path, Path]:
    figures_dir = outdir / "figures"
    results_dir = outdir / "results"
    tables_dir = outdir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, results_dir, tables_dir


def save_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slope_loglog(h_centers: np.ndarray, mse_values: np.ndarray) -> float:
    mask = (h_centers > 0.0) & (mse_values > 0.0)
    if int(np.sum(mask)) < 2:
        return float("nan")
    beta, _ = np.polyfit(np.log(h_centers[mask]), np.log(mse_values[mask]), deg=1)
    return float(beta)


def positive_slope(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(float(value), 0.0)


def percentage_delta(new: float, base: float, eps: float = 1.0e-12) -> float:
    return 100.0 * (float(new) - float(base)) / (abs(float(base)) + eps)


def config_rows() -> list[dict]:
    rows: list[dict] = []
    for k_support in SUPPORT_K_VALUES:
        for requested_bins in K_BINS_VALUES:
            for scheme in BINNING_SCHEMES:
                rows.append(
                    {
                        "k_support": int(k_support),
                        "requested_bins": int(requested_bins),
                        "binning_scheme": scheme,
                        "config_label": f"{'Q' if scheme == 'quantile' else 'L'}{requested_bins}",
                    }
                )
    return rows


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges[1:-1], right=False).astype(np.int64)


def build_support_bins(values: np.ndarray, requested_bins: int, scheme: str) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Support values must be a nonempty 1D array.")
    if scheme == "quantile":
        edges = np.quantile(arr, np.linspace(0.0, 1.0, int(requested_bins) + 1))
    elif scheme == "log_width":
        log_arr = np.log(np.maximum(arr, EPS))
        lo = float(np.min(log_arr))
        hi = float(np.max(log_arr))
        if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
            hi = lo + 1e-12
        log_edges = np.linspace(lo, hi, int(requested_bins) + 1)
        edges = np.exp(log_edges)
        edges[0] = min(edges[0], float(np.min(arr)))
        edges[-1] = max(edges[-1], float(np.max(arr)))
    else:
        raise ValueError(f"Unknown binning scheme: {scheme}")

    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, edges.shape[0]):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    edges[0] = min(edges[0], float(np.min(arr)))
    edges[-1] = max(edges[-1], float(np.max(arr)) + 1e-12)

    bin_ids = assign_bins(arr, edges)
    counts = np.bincount(bin_ids, minlength=int(requested_bins))
    occupied_bins = [int(idx) for idx, count in enumerate(counts) if int(count) > 0]
    if len(occupied_bins) < 2:
        raise RuntimeError(
            f"Support binning produced fewer than two occupied bins for scheme={scheme}, bins={requested_bins}."
        )
    centers = np.full(int(requested_bins), np.nan, dtype=np.float64)
    for b in occupied_bins:
        centers[b] = float(np.median(arr[bin_ids == b]))
    n_region_bins = max(1, int(math.ceil(0.2 * len(occupied_bins))))
    dense_bins = tuple(occupied_bins[:n_region_bins])
    weak_bins = tuple(occupied_bins[-n_region_bins:])
    return {
        "edges": edges,
        "bin_ids_val": bin_ids,
        "counts_val": counts,
        "centers_val": centers,
        "occupied_bins": tuple(occupied_bins),
        "dense_bins": dense_bins,
        "weak_bins": weak_bins,
        "requested_bins": int(requested_bins),
        "n_bins_effective": int(len(occupied_bins)),
        "binning_scheme": scheme,
    }


def region_mask_from_bins(
    bin_ids: np.ndarray,
    target_bins: Sequence[int],
    occupied_reference: Sequence[int],
    direction: str,
) -> np.ndarray:
    mask = np.isin(bin_ids, np.asarray(target_bins, dtype=np.int64))
    if np.any(mask):
        return mask
    if direction == "dense":
        traversal = list(occupied_reference)
    elif direction == "weak":
        traversal = list(reversed(occupied_reference))
    else:
        raise ValueError(f"Unknown direction: {direction}")
    selected: list[int] = []
    for b in traversal:
        selected.append(int(b))
        mask = np.isin(bin_ids, np.asarray(selected, dtype=np.int64))
        if np.any(mask):
            return mask
    return mask


def profile_rows_from_values(metric_values: np.ndarray, h_values: np.ndarray, edges: np.ndarray) -> list[dict]:
    metric_values = np.asarray(metric_values, dtype=np.float64)
    h_values = np.asarray(h_values, dtype=np.float64)
    bin_ids = assign_bins(h_values, edges)
    n_bins = edges.shape[0] - 1
    rows: list[dict] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_id": int(b + 1),
                "h_bin_center": float(np.median(h_values[mask])),
                "metric_bin": float(np.mean(metric_values[mask])),
                "n_bin": int(mask.sum()),
            }
        )
    return rows


def compute_profile_metrics_exp6(
    errors: np.ndarray,
    h_values: np.ndarray,
    bin_info: Mapping[str, object],
) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    h_values = np.asarray(h_values, dtype=np.float64)
    edges = np.asarray(bin_info["edges"], dtype=np.float64)
    occupied_bins = tuple(int(x) for x in bin_info["occupied_bins"])
    bin_ids = assign_bins(h_values, edges)
    weak_mask = region_mask_from_bins(bin_ids, bin_info["weak_bins"], occupied_bins, direction="weak")
    dense_mask = region_mask_from_bins(bin_ids, bin_info["dense_bins"], occupied_bins, direction="dense")
    if not np.any(weak_mask) or not np.any(dense_mask):
        raise RuntimeError("Weak or dense region is empty under the requested support construction.")
    global_mse = float(np.mean(errors))
    weak_mse = float(np.mean(errors[weak_mask]))
    dense_mse = float(np.mean(errors[dense_mask]))
    gap = float(weak_mse / (dense_mse + EPS))
    rows = profile_rows_from_values(errors, h_values, edges)
    h_centers = np.asarray([row["h_bin_center"] for row in rows], dtype=np.float64)
    mse_bins = np.asarray([row["metric_bin"] for row in rows], dtype=np.float64)
    slope = slope_loglog(h_centers, np.maximum(mse_bins, EPS))
    if not math.isfinite(float(slope)):
        slope = 0.0
    return {
        "global_mse": global_mse,
        "weak_mse": weak_mse,
        "dense_mse": dense_mse,
        "gap": gap,
        "slope": float(slope),
        "pos_slope": float(positive_slope(slope)),
        "profile_var": float(np.var(mse_bins)),
    }


def compute_profile_metrics_exp8(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    h_values: np.ndarray,
    bin_info: Mapping[str, object],
) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = (y_pred - y_true) ** 2
    y_var = max(float(np.var(y_true)), EPS)
    nmse_errors = errors / y_var
    edges = np.asarray(bin_info["edges"], dtype=np.float64)
    occupied_bins = tuple(int(x) for x in bin_info["occupied_bins"])
    bin_ids = assign_bins(np.asarray(h_values, dtype=np.float64), edges)
    weak_mask = region_mask_from_bins(bin_ids, bin_info["weak_bins"], occupied_bins, direction="weak")
    dense_mask = region_mask_from_bins(bin_ids, bin_info["dense_bins"], occupied_bins, direction="dense")
    if not np.any(weak_mask) or not np.any(dense_mask):
        raise RuntimeError("Weak or dense region is empty under the requested support construction.")
    global_mse = float(np.mean(errors))
    global_nmse = float(np.mean(nmse_errors))
    weak_mse = float(np.mean(errors[weak_mask]))
    weak_nmse = float(np.mean(nmse_errors[weak_mask]))
    dense_mse = float(np.mean(errors[dense_mask]))
    dense_nmse = float(np.mean(nmse_errors[dense_mask]))
    gap = float(weak_nmse / (dense_nmse + EPS))
    rows = profile_rows_from_values(nmse_errors, h_values, edges)
    h_centers = np.asarray([row["h_bin_center"] for row in rows], dtype=np.float64)
    nmse_bins = np.asarray([row["metric_bin"] for row in rows], dtype=np.float64)
    slope = slope_loglog(h_centers, np.maximum(nmse_bins, EPS))
    if not math.isfinite(float(slope)):
        slope = 0.0
    return {
        "global_mse": global_mse,
        "global_nmse": global_nmse,
        "weak_mse": weak_mse,
        "weak_nmse": weak_nmse,
        "dense_mse": dense_mse,
        "dense_nmse": dense_nmse,
        "gap": gap,
        "slope": float(slope),
        "pos_slope": float(positive_slope(slope)),
        "profile_var": float(np.var(nmse_bins)),
    }


def theta_mode(labels: Sequence[str]) -> tuple[str, float]:
    if not labels:
        return "", 0.0
    counts = Counter(str(x) for x in labels)
    mode_label, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return mode_label, float(count / max(1, len(labels)))


def run_exp6_sensitivity(seed: int, fast: bool) -> list[dict]:
    setting = exp6.SETTING_LIBRARY["stress"]
    n_train = min(setting.default_n_train, 300) if fast else setting.default_n_train
    n_val = min(setting.default_n_val, 1500) if fast else setting.default_n_val
    n_test = min(setting.default_n_test, 2000) if fast else setting.default_n_test
    n_reps = min(setting.default_n_reps, 6) if fast else setting.default_n_reps
    sigma = float(setting.sigma_default)
    tau = 0.05
    eta_w = 1.0
    eta_g = 1.0
    eta_s = 1.0
    theta_grid = exp6.build_theta_grid(setting)

    config_specs = config_rows()
    per_config_rows: dict[tuple[int, int, str], list[dict]] = defaultdict(list)

    for rep in range(int(n_reps)):
        print(f"[Exp6 sensitivity] repetition {rep + 1}/{n_reps}", flush=True)
        rng = np.random.default_rng(int(seed) + 1000 * rep + 17)
        X_train = exp6.sample_nonuniform_design(int(n_train), rng, setting)
        X_val = exp6.sample_nonuniform_design(int(n_val), rng, setting)
        X_test = exp6.sample_nonuniform_design(int(n_test), rng, setting)
        y_train_clean = exp6.f_star(X_train, setting)
        y_val_clean = exp6.f_star(X_val, setting)
        y_test_clean = exp6.f_star(X_test, setting)
        y_train = y_train_clean + sigma * rng.normal(size=int(n_train))

        errors_val_by_theta: dict[tuple[float, float], np.ndarray] = {}
        errors_test_by_theta: dict[tuple[float, float], np.ndarray] = {}

        for gamma in setting.gamma_grid:
            basis = exp6.fit_krr_basis(X_train, gamma=float(gamma))
            eigvals = np.asarray(basis["eigvals"], dtype=np.float64)
            eigvecs = np.asarray(basis["eigvecs"], dtype=np.float64)
            y_eig = eigvecs.T @ y_train

            K_val_train = exp6.rbf_kernel(X_val, X_train, gamma=float(gamma))
            K_test_train = exp6.rbf_kernel(X_test, X_train, gamma=float(gamma))
            Z_val = K_val_train @ eigvecs
            Z_test = K_test_train @ eigvecs

            for lambda_reg in setting.lambda_grid:
                theta = (float(lambda_reg), float(gamma))
                pred_val = exp6.predict_krr(Z_eval=Z_val, eigvals=eigvals, y_eig=y_eig, lambda_reg=float(lambda_reg), n_train=int(n_train))
                pred_test = exp6.predict_krr(Z_eval=Z_test, eigvals=eigvals, y_eig=y_eig, lambda_reg=float(lambda_reg), n_train=int(n_train))
                errors_val_by_theta[theta] = (pred_val - y_val_clean) ** 2
                errors_test_by_theta[theta] = (pred_test - y_test_clean) ** 2

        global_val_mse_by_theta = {theta: float(np.mean(err)) for theta, err in errors_val_by_theta.items()}
        theta_global = exp6.select_theta_from_metric(theta_grid, global_val_mse_by_theta)
        global_val = float(global_val_mse_by_theta[theta_global])
        admissible = [theta for theta in theta_grid if float(global_val_mse_by_theta[theta]) <= (1.0 + tau) * global_val + 1e-15]
        if theta_global not in admissible:
            raise RuntimeError("Global baseline theta must belong to the admissible set.")

        support_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        bin_cache: dict[tuple[int, int, str], dict] = {}
        for spec in config_specs:
            k_support = int(spec["k_support"])
            if k_support not in support_cache:
                h_val = exp6.compute_support_scores(X_val, X_train, k_support=k_support)
                h_test = exp6.compute_support_scores(X_test, X_train, k_support=k_support)
                support_cache[k_support] = (h_val, h_test)
            h_val, h_test = support_cache[k_support]
            key = (k_support, int(spec["requested_bins"]), str(spec["binning_scheme"]))
            if key not in bin_cache:
                bin_cache[key] = build_support_bins(h_val, int(spec["requested_bins"]), str(spec["binning_scheme"]))
            bin_info = bin_cache[key]

            val_metrics_by_theta: dict[tuple[float, float], dict] = {}
            test_metrics_by_theta: dict[tuple[float, float], dict] = {}
            for theta in admissible:
                val_metrics_by_theta[theta] = compute_profile_metrics_exp6(errors_val_by_theta[theta], h_val, bin_info)
                test_metrics_by_theta[theta] = compute_profile_metrics_exp6(errors_test_by_theta[theta], h_test, bin_info)
            theta_global_val_metrics = compute_profile_metrics_exp6(errors_val_by_theta[theta_global], h_val, bin_info)
            theta_global_test_metrics = compute_profile_metrics_exp6(errors_test_by_theta[theta_global], h_test, bin_info)

            weak_ref = max(float(theta_global_val_metrics["weak_mse"]), EPS)
            gap_ref = max(float(theta_global_val_metrics["gap"]), EPS)
            slope_ref = max(float(theta_global_val_metrics["pos_slope"]), 0.0) + EPS
            profile_score_map = {
                theta: (
                    eta_w * float(val_metrics_by_theta[theta]["weak_mse"]) / weak_ref
                    + eta_g * float(val_metrics_by_theta[theta]["gap"]) / gap_ref
                    + eta_s * float(val_metrics_by_theta[theta]["pos_slope"]) / slope_ref
                )
                for theta in admissible
            }
            theta_profile = exp6.select_theta_from_metric(admissible, profile_score_map)
            profile_test_metrics = test_metrics_by_theta[theta_profile]

            per_config_rows[key].append(
                {
                    "repetition": int(rep),
                    "theta_global": exp6.theta_label(theta_global),
                    "theta_profile": exp6.theta_label(theta_profile),
                    "changed_selection": bool(theta_profile != theta_global),
                    "n_bins_effective": int(bin_info["n_bins_effective"]),
                    "delta_global_mse_pct": percentage_delta(profile_test_metrics["global_mse"], theta_global_test_metrics["global_mse"]),
                    "delta_weak_mse_pct": percentage_delta(profile_test_metrics["weak_mse"], theta_global_test_metrics["weak_mse"]),
                    "delta_gap_pct": percentage_delta(profile_test_metrics["gap"], theta_global_test_metrics["gap"]),
                    "delta_slope_pct": percentage_delta(profile_test_metrics["pos_slope"], theta_global_test_metrics["pos_slope"]),
                    "gap_improved": float(profile_test_metrics["gap"] < theta_global_test_metrics["gap"]),
                    "slope_improved": float(profile_test_metrics["pos_slope"] < theta_global_test_metrics["pos_slope"]),
                    "weak_improved": float(profile_test_metrics["weak_mse"] < theta_global_test_metrics["weak_mse"]),
                    "global_theta_lambda": float(theta_global[0]),
                    "global_theta_gamma": float(theta_global[1]),
                    "profile_theta_lambda": float(theta_profile[0]),
                    "profile_theta_gamma": float(theta_profile[1]),
                }
            )

    summary_rows: list[dict] = []
    for spec in config_specs:
        key = (int(spec["k_support"]), int(spec["requested_bins"]), str(spec["binning_scheme"]))
        rows = per_config_rows.get(key, [])
        if not rows:
            continue
        global_mode, global_mode_frac = theta_mode([str(row["theta_global"]) for row in rows])
        profile_mode, profile_mode_frac = theta_mode([str(row["theta_profile"]) for row in rows])
        summary_rows.append(
            {
                "experiment": "exp6",
                "base_setting": "stress",
                "seed": int(seed),
                "n_reps": int(n_reps),
                "n_train": int(n_train),
                "n_val": int(n_val),
                "n_test": int(n_test),
                "sigma": float(sigma),
                "tau": float(tau),
                "eta_w": float(eta_w),
                "eta_g": float(eta_g),
                "eta_s": float(eta_s),
                "k_support": int(spec["k_support"]),
                "requested_bins": int(spec["requested_bins"]),
                "binning_scheme": str(spec["binning_scheme"]),
                "config_label": str(spec["config_label"]),
                "n_bins_effective_mean": float(np.mean([float(row["n_bins_effective"]) for row in rows])),
                "n_bins_effective_median": float(np.median([float(row["n_bins_effective"]) for row in rows])),
                "changed_selection_rate": float(np.mean([bool(row["changed_selection"]) for row in rows])),
                "fraction_gap_improved": float(np.mean([float(row["gap_improved"]) for row in rows])),
                "fraction_slope_improved": float(np.mean([float(row["slope_improved"]) for row in rows])),
                "fraction_weak_improved": float(np.mean([float(row["weak_improved"]) for row in rows])),
                "median_global_mse_change_pct": float(np.median([float(row["delta_global_mse_pct"]) for row in rows])),
                "mean_global_mse_change_pct": float(np.mean([float(row["delta_global_mse_pct"]) for row in rows])),
                "median_gap_change_pct": float(np.median([float(row["delta_gap_pct"]) for row in rows])),
                "median_slope_change_pct": float(np.median([float(row["delta_slope_pct"]) for row in rows])),
                "mode_theta_global": global_mode,
                "mode_theta_global_fraction": float(global_mode_frac),
                "mode_theta_profile": profile_mode,
                "mode_theta_profile_fraction": float(profile_mode_frac),
            }
        )
    return summary_rows


def run_exp8_sensitivity(seed: int, fast: bool, data_dir: str, models: str) -> list[dict]:
    args_like = argparse.Namespace(seed=seed, data_dir=data_dir, fast=fast)
    discovered, skipped = exp8.discover_real_datasets(args_like)
    max_samples = exp8.MAX_SAMPLES_FAST if fast else exp8.MAX_SAMPLES_DEFAULT
    cleaned: list[dict] = []
    for idx, ds in enumerate(discovered):
        ds_clean, skip_info = exp8.clean_dataset(ds, seed=seed + 97 * (idx + 1), max_samples=max_samples)
        if skip_info is not None:
            continue
        cleaned.append(ds_clean)
    if not cleaned:
        raise RuntimeError("No usable datasets were found for Experiment 8 sensitivity.")

    n_splits = 3 if fast else 20
    tau = 0.05
    eta = 0.5
    zeta = 0.25
    model_families = [x.strip() for x in models.split(",") if x.strip()]
    config_specs = config_rows()
    per_config_pair_rows: dict[tuple[int, int, str], list[dict]] = defaultdict(list)

    for ds_idx, ds in enumerate(cleaned):
        print(f"[Exp8 sensitivity] dataset {ds_idx + 1}/{len(cleaned)}: {ds['name']}", flush=True)
        X_df = ds["X"]
        y = ds["y"]
        splits = exp8.make_repeated_splits(n_samples=y.shape[0], n_splits=int(n_splits), seed=seed + 1000 * ds_idx + 31)
        for split in splits:
            split_id = int(split["split_id"])
            tr_idx = split["train"]
            val_idx = split["val"]
            te_idx = split["test"]
            X_train_df = X_df.iloc[tr_idx].reset_index(drop=True)
            X_val_df = X_df.iloc[val_idx].reset_index(drop=True)
            X_test_df = X_df.iloc[te_idx].reset_index(drop=True)
            y_train = y[tr_idx]
            y_val = y[val_idx]
            y_test = y[te_idx]

            preprocessor = exp8.make_preprocessor(X_train_df)
            preprocessor.fit(X_train_df)
            X_train = exp8.transform_dense(preprocessor, X_train_df)
            X_val = exp8.transform_dense(preprocessor, X_val_df)
            X_test = exp8.transform_dense(preprocessor, X_test_df)
            if X_train.shape[1] < 2:
                continue

            support_cache: dict[int, tuple[np.ndarray, np.ndarray, dict[str, dict]]] = {}
            for k_support in SUPPORT_K_VALUES:
                h_val = exp8.compute_support_scores(X_val, X_train, k_support=k_support)
                h_test = exp8.compute_support_scores(X_test, X_train, k_support=k_support)
                by_scheme: dict[str, dict] = {}
                for requested_bins in K_BINS_VALUES:
                    for scheme in BINNING_SCHEMES:
                        by_scheme[(requested_bins, scheme)] = build_support_bins(h_val, int(requested_bins), scheme)
                support_cache[int(k_support)] = (h_val, h_test, by_scheme)

            for family_idx, family in enumerate(model_families):
                grid = exp8.generate_candidate_grid(model_family=family, X_train=X_train, fast=bool(fast))
                pred_val_by_candidate: dict[str, np.ndarray] = {}
                pred_test_by_candidate: dict[str, np.ndarray] = {}
                metadata_by_candidate: dict[str, dict] = {}

                for cand in grid:
                    candidate_id = str(cand["candidate_id"])
                    hyperparams = dict(cand["hyperparams"])
                    try:
                        model = exp8.fit_model_family_candidate(
                            model_family=family,
                            hyperparams=hyperparams,
                            X_train=X_train,
                            y_train=y_train,
                            random_state=seed + 100000 * ds_idx + 1000 * split_id + 17 * (family_idx + 1),
                        )
                        pred_val_by_candidate[candidate_id] = exp8.predict_model(model, X_val)
                        pred_test_by_candidate[candidate_id] = exp8.predict_model(model, X_test)
                        metadata_by_candidate[candidate_id] = {
                            "hyperparams": json.dumps(hyperparams, sort_keys=True),
                            "complexity_rank": int(cand["complexity_rank"]),
                        }
                    except Exception:
                        continue
                if not pred_val_by_candidate:
                    continue

                global_metric_rows = []
                for candidate_id, pred_val in pred_val_by_candidate.items():
                    global_mse = float(np.mean((pred_val - y_val) ** 2))
                    global_metric_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "hyperparams": metadata_by_candidate[candidate_id]["hyperparams"],
                            "complexity_rank": metadata_by_candidate[candidate_id]["complexity_rank"],
                            "global_mse": global_mse,
                        }
                    )
                theta_global_base, admissible_rows_base, _ = exp8.build_admissible_global_set(global_metric_rows, tau=tau)
                global_candidate_id = str(theta_global_base["candidate_id"])
                admissible_candidate_ids = {str(row["candidate_id"]) for row in admissible_rows_base}

                for spec in config_specs:
                    key = (int(spec["k_support"]), int(spec["requested_bins"]), str(spec["binning_scheme"]))
                    h_val, h_test, by_scheme = support_cache[int(spec["k_support"])]
                    bin_info = by_scheme[(int(spec["requested_bins"]), str(spec["binning_scheme"]))]
                    val_rows: list[dict] = []
                    test_rows: list[dict] = []
                    for candidate_id, pred_val in pred_val_by_candidate.items():
                        val_metrics = compute_profile_metrics_exp8(y_val, pred_val, h_val, bin_info)
                        test_metrics = compute_profile_metrics_exp8(y_test, pred_test_by_candidate[candidate_id], h_test, bin_info)
                        row_common = {
                            "candidate_id": candidate_id,
                            "hyperparams": metadata_by_candidate[candidate_id]["hyperparams"],
                            "complexity_rank": metadata_by_candidate[candidate_id]["complexity_rank"],
                        }
                        val_rows.append({**row_common, **val_metrics})
                        test_rows.append({**row_common, **test_metrics})

                    global_row_val = next(row for row in val_rows if str(row["candidate_id"]) == global_candidate_id)
                    admissible_rows = [row for row in val_rows if str(row["candidate_id"]) in admissible_candidate_ids]
                    theta_profile = exp8.select_constrained_profile_candidate(
                        admissible_rows,
                        theta_global=global_row_val,
                        eta=eta,
                        zeta=zeta,
                    )
                    profile_candidate_id = str(theta_profile["candidate_id"])
                    test_map = {str(row["candidate_id"]): row for row in test_rows}
                    global_test = test_map[global_candidate_id]
                    profile_test = test_map[profile_candidate_id]

                    per_config_pair_rows[key].append(
                        {
                            "dataset": ds["name"],
                            "model_family": family,
                            "split_id": int(split_id),
                            "theta_global": str(global_row_val["hyperparams"]),
                            "theta_profile": str(theta_profile["hyperparams"]),
                            "selection_changed": bool(profile_candidate_id != global_candidate_id),
                            "global_nmse_global": float(global_test["global_nmse"]),
                            "global_nmse_profile": float(profile_test["global_nmse"]),
                            "weak_nmse_global": float(global_test["weak_nmse"]),
                            "weak_nmse_profile": float(profile_test["weak_nmse"]),
                            "gap_global": float(global_test["gap"]),
                            "gap_profile": float(profile_test["gap"]),
                            "slope_global": float(global_test["pos_slope"]),
                            "slope_profile": float(profile_test["pos_slope"]),
                            "profile_var_global": float(global_test["profile_var"]),
                            "profile_var_profile": float(profile_test["profile_var"]),
                            "delta_global_nmse_pct": percentage_delta(profile_test["global_nmse"], global_test["global_nmse"]),
                            "delta_weak_nmse_pct": percentage_delta(profile_test["weak_nmse"], global_test["weak_nmse"]),
                            "delta_gap_pct": percentage_delta(profile_test["gap"], global_test["gap"]),
                            "delta_positive_slope_pct": percentage_delta(profile_test["pos_slope"], global_test["pos_slope"]),
                            "delta_profile_var_pct": percentage_delta(profile_test["profile_var"], global_test["profile_var"]),
                            "n_bins_effective": int(bin_info["n_bins_effective"]),
                        }
                    )

    summary_rows: list[dict] = []
    for spec in config_specs:
        key = (int(spec["k_support"]), int(spec["requested_bins"]), str(spec["binning_scheme"]))
        rows = per_config_pair_rows.get(key, [])
        if not rows:
            continue
        pair_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            pair_groups[(str(row["dataset"]), str(row["model_family"]))].append(row)
        pair_summary_rows: list[dict] = []
        for (dataset, family), cur in sorted(pair_groups.items()):
            global_nmse_mean = float(np.mean([float(r["global_nmse_global"]) for r in cur]))
            profile_global_mean = float(np.mean([float(r["global_nmse_profile"]) for r in cur]))
            weak_nmse_mean = float(np.mean([float(r["weak_nmse_global"]) for r in cur]))
            profile_weak_mean = float(np.mean([float(r["weak_nmse_profile"]) for r in cur]))
            gap_mean = float(np.mean([float(r["gap_global"]) for r in cur]))
            profile_gap_mean = float(np.mean([float(r["gap_profile"]) for r in cur]))
            slope_mean = float(np.mean([float(r["slope_global"]) for r in cur]))
            profile_slope_mean = float(np.mean([float(r["slope_profile"]) for r in cur]))
            profile_var_mean = float(np.mean([float(r["profile_var_global"]) for r in cur]))
            profile_profile_var_mean = float(np.mean([float(r["profile_var_profile"]) for r in cur]))
            pair_summary_rows.append(
                {
                    "dataset": dataset,
                    "model_family": family,
                    "selection_changed_fraction": float(np.mean([bool(r["selection_changed"]) for r in cur])),
                    "delta_global_nmse_pct": percentage_delta(profile_global_mean, global_nmse_mean),
                    "delta_weak_nmse_pct": percentage_delta(profile_weak_mean, weak_nmse_mean),
                    "delta_gap_pct": percentage_delta(profile_gap_mean, gap_mean),
                    "delta_positive_slope_pct": percentage_delta(profile_slope_mean, slope_mean),
                    "delta_profile_var_pct": percentage_delta(profile_profile_var_mean, profile_var_mean),
                }
            )
        summary_rows.append(
            {
                "experiment": "exp8",
                "seed": int(seed),
                "n_splits": int(n_splits),
                "tau": float(tau),
                "eta": float(eta),
                "zeta": float(zeta),
                "n_datasets": int(len(cleaned)),
                "n_pairs": int(len(pair_summary_rows)),
                "models": ",".join(model_families),
                "k_support": int(spec["k_support"]),
                "requested_bins": int(spec["requested_bins"]),
                "binning_scheme": str(spec["binning_scheme"]),
                "config_label": str(spec["config_label"]),
                "n_bins_effective_mean": float(np.mean([float(r["n_bins_effective"]) for r in rows])),
                "n_bins_effective_median": float(np.median([float(r["n_bins_effective"]) for r in rows])),
                "split_level_changed_rate": float(np.mean([bool(r["selection_changed"]) for r in rows])),
                "pair_level_changed_fraction_mean": float(np.mean([float(r["selection_changed_fraction"]) for r in pair_summary_rows])),
                "fraction_improved_global_nmse": float(np.mean([float(r["delta_global_nmse_pct"]) < 0.0 for r in pair_summary_rows])),
                "fraction_improved_weak_nmse": float(np.mean([float(r["delta_weak_nmse_pct"]) < 0.0 for r in pair_summary_rows])),
                "fraction_improved_gap": float(np.mean([float(r["delta_gap_pct"]) < 0.0 for r in pair_summary_rows])),
                "fraction_improved_slope": float(np.mean([float(r["delta_positive_slope_pct"]) < 0.0 for r in pair_summary_rows])),
                "fraction_improved_profile_var": float(np.mean([float(r["delta_profile_var_pct"]) < 0.0 for r in pair_summary_rows])),
                "mean_delta_global_nmse_pct": float(np.mean([float(r["delta_global_nmse_pct"]) for r in pair_summary_rows])),
                "mean_delta_gap_pct": float(np.mean([float(r["delta_gap_pct"]) for r in pair_summary_rows])),
                "mean_delta_slope_pct": float(np.mean([float(r["delta_positive_slope_pct"]) for r in pair_summary_rows])),
            }
        )
    return summary_rows


def save_exp6_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "$k_{\\mathrm{sup}}$ & Binning & $K$ & Eff. bins & Changed & Gap improved & Slope improved & Med. $\\Delta$ Global (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        scheme_label = "quantile" if row["binning_scheme"] == "quantile" else "log-width"
        lines.append(
            f"{int(row['k_support'])} & "
            f"{scheme_label} & "
            f"{int(row['requested_bins'])} & "
            f"{float(row['n_bins_effective_mean']):.2f} & "
            f"{100.0 * float(row['changed_selection_rate']):.1f}\\% & "
            f"{100.0 * float(row['fraction_gap_improved']):.1f}\\% & "
            f"{100.0 * float(row['fraction_slope_improved']):.1f}\\% & "
            f"{float(row['median_global_mse_change_pct']):+.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_exp8_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{llcccccccc}",
        "\\toprule",
        "$k_{\\mathrm{sup}}$ & Binning & $K$ & Eff. bins & Changed & Global & Weak & Gap & Slope & Prof. var \\\\",
        "\\midrule",
    ]
    for row in rows:
        scheme_label = "quantile" if row["binning_scheme"] == "quantile" else "log-width"
        lines.append(
            f"{int(row['k_support'])} & "
            f"{scheme_label} & "
            f"{int(row['requested_bins'])} & "
            f"{float(row['n_bins_effective_mean']):.2f} & "
            f"{100.0 * float(row['pair_level_changed_fraction_mean']):.1f}\\% & "
            f"{100.0 * float(row['fraction_improved_global_nmse']):.1f}\\% & "
            f"{100.0 * float(row['fraction_improved_weak_nmse']):.1f}\\% & "
            f"{100.0 * float(row['fraction_improved_gap']):.1f}\\% & "
            f"{100.0 * float(row['fraction_improved_slope']):.1f}\\% & "
            f"{100.0 * float(row['fraction_improved_profile_var']):.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def pivot_grid(rows: Sequence[Mapping[str, object]], key: str) -> tuple[np.ndarray, list[str], list[str]]:
    k_vals = list(SUPPORT_K_VALUES)
    cols = [f"{'Q' if s == 'quantile' else 'L'}{b}" for s in BINNING_SCHEMES for b in K_BINS_VALUES]
    value_map = {}
    for row in rows:
        label = f"{'Q' if row['binning_scheme'] == 'quantile' else 'L'}{int(row['requested_bins'])}"
        value_map[(int(row["k_support"]), label)] = float(row[key])
    grid = np.full((len(k_vals), len(cols)), np.nan, dtype=np.float64)
    for i, kval in enumerate(k_vals):
        for j, label in enumerate(cols):
            if (kval, label) in value_map:
                grid[i, j] = value_map[(kval, label)]
    return grid, [str(k) for k in k_vals], cols


def add_heatmap(ax, data: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], title: str, as_percent: bool) -> None:
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not math.isfinite(float(value)):
                continue
            text = f"{100.0 * value:.0f}%" if as_percent else f"{value:+.1f}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_exp6_heatmap(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    metrics = [
        ("changed_selection_rate", "Changed-selection rate", True),
        ("fraction_gap_improved", "Gap improvement fraction", True),
        ("fraction_slope_improved", "Slope improvement fraction", True),
        ("median_global_mse_change_pct", "Median global-MSE change (%)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8))
    for ax, (key, title, as_percent) in zip(axes.flat, metrics):
        data, row_labels, col_labels = pivot_grid(rows, key)
        add_heatmap(ax, data, row_labels, col_labels, title, as_percent=as_percent)
    fig.suptitle("Experiment 6 support-construction sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_exp8_heatmap(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    metrics = [
        ("pair_level_changed_fraction_mean", "Changed-selection fraction", True),
        ("fraction_improved_global_nmse", "Global-NMSE improved fraction", True),
        ("fraction_improved_weak_nmse", "Weak-NMSE improved fraction", True),
        ("fraction_improved_gap", "Gap improved fraction", True),
        ("fraction_improved_slope", "Slope improved fraction", True),
        ("fraction_improved_profile_var", "Profile-var improved fraction", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.2))
    for ax, (key, title, as_percent) in zip(axes.flat, metrics):
        data, row_labels, col_labels = pivot_grid(rows, key)
        add_heatmap(ax, data, row_labels, col_labels, title, as_percent=as_percent)
    fig.suptitle("Experiment 8 support-construction sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_exp8_heatmap_panels(figures_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    metrics = [
        ("pair_level_changed_fraction_mean", "Changed-selection fraction", True, "changed"),
        ("fraction_improved_global_nmse", "Global-NMSE improved fraction", True, "global"),
        ("fraction_improved_weak_nmse", "Weak-NMSE improved fraction", True, "weak"),
        ("fraction_improved_gap", "Gap improved fraction", True, "gap"),
        ("fraction_improved_slope", "Slope improved fraction", True, "slope"),
        ("fraction_improved_profile_var", "Profile-var improved fraction", True, "profile_var"),
    ]
    for key, title, as_percent, suffix in metrics:
        data, row_labels, col_labels = pivot_grid(rows, key)
        fig, ax = plt.subplots(figsize=(4.35, 3.35))
        add_heatmap(ax, data, row_labels, col_labels, title, as_percent=as_percent)
        fig.tight_layout()
        fig.savefig(figures_dir / f"exp8_profile_construction_sensitivity_{suffix}.pdf", bbox_inches="tight")
        fig.savefig(figures_dir / f"exp8_profile_construction_sensitivity_{suffix}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_interpretation(path: Path, exp6_rows: Sequence[Mapping[str, object]], exp8_rows: Sequence[Mapping[str, object]]) -> None:
    lines: list[str] = [
        "Profile-construction sensitivity interpretation",
        "",
    ]
    if exp6_rows:
        exp6_df = pd.DataFrame(exp6_rows)
        lines.extend(
            [
                "Experiment 6:",
                (
                    f"- Across the support-field and binning choices, the changed-selection rate ranges from "
                    f"{100.0 * float(exp6_df['changed_selection_rate'].min()):.1f}% to "
                    f"{100.0 * float(exp6_df['changed_selection_rate'].max()):.1f}%."
                ),
                (
                    f"- The fraction of splits with gap improvement ranges from "
                    f"{100.0 * float(exp6_df['fraction_gap_improved'].min()):.1f}% to "
                    f"{100.0 * float(exp6_df['fraction_gap_improved'].max()):.1f}%."
                ),
                (
                    f"- The fraction of splits with slope improvement ranges from "
                    f"{100.0 * float(exp6_df['fraction_slope_improved'].min()):.1f}% to "
                    f"{100.0 * float(exp6_df['fraction_slope_improved'].max()):.1f}%."
                ),
                (
                    f"- Median global-MSE change remains between "
                    f"{float(exp6_df['median_global_mse_change_pct'].min()):+.2f}% and "
                    f"{float(exp6_df['median_global_mse_change_pct'].max()):+.2f}%."
                ),
                (
                    "- The qualitative conclusion is stable: support-conditioned summaries still re-rank "
                    "near-tied global-risk candidates, and the main effect is a trade-off between small "
                    "global-risk changes and improved dense--weak imbalance diagnostics."
                ),
                "",
            ]
        )
    if exp8_rows:
        exp8_df = pd.DataFrame(exp8_rows)
        lines.extend(
            [
                "Experiment 8:",
                (
                    f"- Across support constructions, the pair-level changed-selection fraction ranges from "
                    f"{100.0 * float(exp8_df['pair_level_changed_fraction_mean'].min()):.1f}% to "
                    f"{100.0 * float(exp8_df['pair_level_changed_fraction_mean'].max()):.1f}%."
                ),
                (
                    f"- The aggregate gap-improvement fraction ranges from "
                    f"{100.0 * float(exp8_df['fraction_improved_gap'].min()):.1f}% to "
                    f"{100.0 * float(exp8_df['fraction_improved_gap'].max()):.1f}%."
                ),
                (
                    f"- The aggregate slope-improvement fraction ranges from "
                    f"{100.0 * float(exp8_df['fraction_improved_slope'].min()):.1f}% to "
                    f"{100.0 * float(exp8_df['fraction_improved_slope'].max()):.1f}%."
                ),
                (
                    f"- Global-NMSE and weak-NMSE improvement fractions vary more modestly, with ranges "
                    f"{100.0 * float(exp8_df['fraction_improved_global_nmse'].min()):.1f}%--"
                    f"{100.0 * float(exp8_df['fraction_improved_global_nmse'].max()):.1f}% and "
                    f"{100.0 * float(exp8_df['fraction_improved_weak_nmse'].min()):.1f}%--"
                    f"{100.0 * float(exp8_df['fraction_improved_weak_nmse'].max()):.1f}%."
                ),
                (
                    "- The qualitative real-data conclusion remains diagnostic rather than dominant: the "
                    "profile-aware rule does not uniformly improve every metric, but support-conditioned "
                    "summaries continue to expose alternatives that global-risk selection alone does not distinguish, "
                    "especially when the support field is meaningfully heterogeneous."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_exp6_flag = bool(args.run_exp6) or (not args.run_exp6 and not args.run_exp8)
    run_exp8_flag = bool(args.run_exp8) or (not args.run_exp6 and not args.run_exp8)

    outdir = resolve_outdir(args.outdir)
    figures_dir, results_dir, tables_dir = ensure_dirs(outdir)

    exp6_rows: list[dict] = []
    exp8_rows: list[dict] = []

    if run_exp6_flag:
        exp6_rows = run_exp6_sensitivity(seed=int(args.seed), fast=bool(args.fast))
        exp6_rows = sorted(
            exp6_rows,
            key=lambda row: (int(row["k_support"]), str(row["binning_scheme"]), int(row["requested_bins"])),
        )
        save_csv(results_dir / "results_profile_construction_sensitivity_exp6.csv", exp6_rows)
        save_exp6_table(tables_dir / "table_profile_construction_sensitivity_exp6.tex", exp6_rows)
        make_exp6_heatmap(figures_dir / "exp6_profile_construction_sensitivity_heatmap.png", exp6_rows)

    if run_exp8_flag:
        exp8_rows = run_exp8_sensitivity(
            seed=int(args.seed),
            fast=bool(args.fast),
            data_dir=str(args.exp8_data_dir),
            models=str(args.exp8_models),
        )
        exp8_rows = sorted(
            exp8_rows,
            key=lambda row: (int(row["k_support"]), str(row["binning_scheme"]), int(row["requested_bins"])),
        )
        save_csv(results_dir / "results_profile_construction_sensitivity_exp8.csv", exp8_rows)
        save_exp8_table(tables_dir / "table_profile_construction_sensitivity_exp8.tex", exp8_rows)
        make_exp8_heatmap(figures_dir / "exp8_profile_construction_sensitivity_heatmap.png", exp8_rows)
        make_exp8_heatmap_panels(figures_dir, exp8_rows)

    write_interpretation(results_dir / "profile_construction_sensitivity_interpretation.txt", exp6_rows, exp8_rows)


if __name__ == "__main__":
    main()
