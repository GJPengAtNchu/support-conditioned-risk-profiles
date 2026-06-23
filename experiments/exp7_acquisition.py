from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

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
from matplotlib import patches
from sklearn.neighbors import NearestNeighbors

from figure_layout_utils import make_style_map, save_axes_group_panel, save_legend_figure


RULE_DISPLAY_ORDER = (
    "Random acquisition",
    "Support-only acquisition",
    "Error-only acquisition",
    "Posterior-variance acquisition",
    "Profile-aware acquisition",
)

REGIME_ORDER = (
    "support_limited",
    "error_support_conflict",
)

REGIME_LABELS = {
    "support_limited": "Support-limited",
    "error_support_conflict": "Error-support conflict",
    "error_support_conflict_strong": "Conflict (strong)",
}

REGIME_SHORT = {
    "support_limited": "Support-limited",
    "error_support_conflict": "Conflict",
    "error_support_conflict_strong": "Conflict (strong)",
}

RULE_TABLE_LABELS = {
    "Random acquisition": "Random acquisition",
    "Support-only acquisition": "Support-only acquisition",
    "Error-only acquisition": "Error-only acquisition",
    "Posterior-variance acquisition": "Posterior-variance",
    "Profile-aware acquisition": "Profile-aware acquisition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 7: profile-guided data acquisition."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-initial", type=int, default=250)
    parser.add_argument("--n-pool", type=int, default=5000)
    parser.add_argument("--n-eval", type=int, default=8000)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--n-reps", type=int, default=20)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--k-support", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-3)
    parser.add_argument("--gamma-acq", type=float, default=1.0)
    parser.add_argument(
        "--regimes",
        type=str,
        default="support_limited,error_support_conflict",
        help="Comma-separated list of regimes to run.",
    )
    parser.add_argument(
        "--disable-strong-conflict-fallback",
        action="store_true",
        help="Do not automatically run the stronger conflict regime when the default conflict regime is too mild.",
    )
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def apply_fast_mode(args: argparse.Namespace) -> argparse.Namespace:
    if args.fast:
        args.n_initial = 150
        args.n_pool = 2000
        args.n_eval = 2000
        args.n_rounds = 5
        args.batch_size = 20
        args.n_reps = 5
    return args


def resolve_outdir(outdir: str) -> Path:
    path = Path(outdir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_regimes(regimes: str) -> List[str]:
    requested = []
    for item in regimes.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in {"support_limited", "error_support_conflict", "error_support_conflict_strong"}:
            raise ValueError(f"Unknown regime: {key}")
        requested.append(key)
    if not requested:
        raise ValueError("At least one regime must be specified.")
    return requested


def regime_config(name: str) -> Dict[str, object]:
    if name == "support_limited":
        return {
            "name": name,
            "label": REGIME_LABELS[name],
            "weights": [0.75, 0.15, 0.10],
            "components": [
                {"kind": "gaussian", "mean": [0.25, 0.30], "sd": [0.09, 0.08]},
                {"kind": "gaussian", "mean": [0.60, 0.45], "sd": [0.10, 0.09]},
                {"kind": "uniform"},
            ],
            "bump_center": [0.82, 0.78],
            "bump_amplitude": 0.8,
            "bump_sharpness": 35.0,
            "relevant_radius": 0.15,
            "relevant_region_label": "high-error weak region",
            "distractor_region": None,
            "target_description": (
                "sincos background plus a weak-region Gaussian bump near (0.82, 0.78)"
            ),
        }
    if name == "error_support_conflict":
        return {
            "name": name,
            "label": REGIME_LABELS[name],
            "weights": [0.70, 0.15, 0.10, 0.05],
            "components": [
                {"kind": "gaussian", "mean": [0.25, 0.30], "sd": [0.08, 0.08]},
                {"kind": "gaussian", "mean": [0.65, 0.35], "sd": [0.07, 0.07]},
                {"kind": "gaussian", "mean": [0.72, 0.72], "sd": [0.055, 0.055]},
                {"kind": "uniform"},
            ],
            "bump_center": [0.72, 0.72],
            "bump_amplitude": 1.2,
            "bump_sharpness": 50.0,
            "relevant_radius": 0.15,
            "relevant_region_label": "high-error relevant region",
            "distractor_region": {"x1_min": 0.82, "x2_min": 0.82},
            "target_description": (
                "sincos background plus a high-error bump near (0.72, 0.72); "
                "the far sparse corner acts as a low-error distractor"
            ),
        }
    if name == "error_support_conflict_strong":
        return {
            "name": name,
            "label": REGIME_LABELS[name],
            "weights": [0.70, 0.15, 0.12, 0.03],
            "components": [
                {"kind": "gaussian", "mean": [0.25, 0.30], "sd": [0.08, 0.08]},
                {"kind": "gaussian", "mean": [0.65, 0.35], "sd": [0.07, 0.07]},
                {"kind": "gaussian", "mean": [0.72, 0.72], "sd": [0.050, 0.050]},
                {"kind": "uniform"},
            ],
            "bump_center": [0.78, 0.78],
            "bump_amplitude": 1.6,
            "bump_sharpness": 80.0,
            "relevant_radius": 0.14,
            "relevant_region_label": "high-error relevant region",
            "distractor_region": {"x1_min": 0.82, "x2_min": 0.82},
            "target_description": (
                "stronger conflict variant with a sharper high-error bump near (0.78, 0.78), "
                "a nearby moderate-support cluster around (0.72, 0.72), and lower uniform background mass"
            ),
        }
    raise ValueError(f"Unknown regime configuration: {name}")


def sample_truncated_gaussian(
    n: int,
    mean: Sequence[float],
    sd: Sequence[float] | float,
    rng: np.random.Generator,
) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float64)
    sd_arr = np.asarray(sd, dtype=np.float64)
    if sd_arr.ndim == 0:
        sd_arr = np.asarray([float(sd_arr), float(sd_arr)], dtype=np.float64)

    samples = np.empty((n, 2), dtype=np.float64)
    filled = 0
    while filled < n:
        need = n - filled
        batch = max(64, int(math.ceil(need * 2.5)))
        cand = rng.normal(loc=mean_arr, scale=sd_arr, size=(batch, 2))
        mask = (
            (cand[:, 0] >= 0.0)
            & (cand[:, 0] <= 1.0)
            & (cand[:, 1] >= 0.0)
            & (cand[:, 1] <= 1.0)
        )
        kept = cand[mask]
        take = min(need, kept.shape[0])
        if take > 0:
            samples[filled : filled + take] = kept[:take]
            filled += take
    return samples


def sample_mixture_design(n: int, cfg: Dict[str, object], rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray(cfg["weights"], dtype=np.float64)
    components = list(cfg["components"])
    counts = rng.multinomial(n, weights)
    blocks = []
    for count, component in zip(counts, components):
        if count == 0:
            continue
        if component["kind"] == "gaussian":
            blocks.append(
                sample_truncated_gaussian(
                    n=count,
                    mean=component["mean"],
                    sd=component["sd"],
                    rng=rng,
                )
            )
        else:
            blocks.append(rng.uniform(0.0, 1.0, size=(count, 2)))
    X = np.vstack(blocks)
    rng.shuffle(X, axis=0)
    return np.asarray(X, dtype=np.float64)


def sample_uniform_points(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(rng.uniform(0.0, 1.0, size=(n, 2)), dtype=np.float64)


def f_star(X: np.ndarray, cfg: Dict[str, object]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]
    center = np.asarray(cfg["bump_center"], dtype=np.float64)
    amp = float(cfg["bump_amplitude"])
    sharpness = float(cfg["bump_sharpness"])
    return (
        np.sin(2.0 * math.pi * x1) * np.cos(2.0 * math.pi * x2)
        + amp * np.exp(-sharpness * ((x1 - center[0]) ** 2 + (x2 - center[1]) ** 2))
        + 0.25 * x1
    )


def add_noisy_labels(y_clean: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(y_clean + float(sigma) * rng.normal(size=y_clean.shape[0]), dtype=np.float64)


def rbf_kernel(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    sqX = np.sum(X * X, axis=1)[:, None]
    sqZ = np.sum(Z * Z, axis=1)[None, :]
    sqdist = np.maximum(sqX + sqZ - 2.0 * X @ Z.T, 0.0)
    return np.exp(-float(gamma) * sqdist)


def fit_krr(X_train: np.ndarray, y_train: np.ndarray, gamma: float, lambda_reg: float) -> Dict[str, np.ndarray]:
    K = rbf_kernel(X_train, X_train, gamma=gamma)
    if not np.allclose(K, K.T, atol=1e-10):
        raise RuntimeError("Kernel matrix is not numerically symmetric.")
    n_train = X_train.shape[0]
    reg = K + (n_train * float(lambda_reg)) * np.eye(n_train, dtype=np.float64)
    dual = np.linalg.solve(reg, y_train)
    return {
        "X_train": np.asarray(X_train, dtype=np.float64),
        "dual": np.asarray(dual, dtype=np.float64),
        "gamma": np.asarray([gamma], dtype=np.float64),
        "lambda_reg": np.asarray([lambda_reg], dtype=np.float64),
        "reg": np.asarray(reg, dtype=np.float64),
    }


def predict_krr(model: Dict[str, np.ndarray], X_eval: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    X_train = np.asarray(model["X_train"], dtype=np.float64)
    dual = np.asarray(model["dual"], dtype=np.float64)
    gamma = float(np.asarray(model["gamma"], dtype=np.float64)[0])
    preds = np.empty(X_eval.shape[0], dtype=np.float64)
    for start in range(0, X_eval.shape[0], batch_size):
        stop = min(start + batch_size, X_eval.shape[0])
        K = rbf_kernel(np.asarray(X_eval[start:stop], dtype=np.float64), X_train, gamma=gamma)
        preds[start:stop] = K @ dual
    return preds


def posterior_variance_krr(model: Dict[str, np.ndarray], X_query: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    X_train = np.asarray(model["X_train"], dtype=np.float64)
    reg = np.asarray(model["reg"], dtype=np.float64)
    gamma = float(np.asarray(model["gamma"], dtype=np.float64)[0])
    out = np.empty(X_query.shape[0], dtype=np.float64)
    for start in range(0, X_query.shape[0], batch_size):
        stop = min(start + batch_size, X_query.shape[0])
        K_q = rbf_kernel(np.asarray(X_query[start:stop], dtype=np.float64), X_train, gamma=gamma)
        solved = np.linalg.solve(reg, K_q.T)
        out[start:stop] = 1.0 - np.sum(K_q * solved.T, axis=1)
    return np.maximum(out, 0.0)


def make_support_index(X_train: np.ndarray, k_support: int) -> NearestNeighbors:
    nbrs = NearestNeighbors(n_neighbors=k_support, algorithm="auto")
    nbrs.fit(X_train)
    return nbrs


def compute_support_scores(
    X_query: np.ndarray,
    X_train: np.ndarray | None = None,
    k_support: int | None = None,
    nbrs: NearestNeighbors | None = None,
) -> np.ndarray:
    if nbrs is None:
        if X_train is None or k_support is None:
            raise ValueError("Either provide nbrs or provide X_train and k_support.")
        nbrs = make_support_index(X_train, k_support)
    distances, _ = nbrs.kneighbors(X_query, return_distance=True)
    return np.asarray(distances[:, -1], dtype=np.float64)


def assign_bins(h_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(h_values, edges[1:-1], right=False).astype(np.int64)


def make_support_bins(h_eval: np.ndarray, n_bins: int) -> Dict[str, np.ndarray]:
    max_bins = min(int(n_bins), int(h_eval.shape[0]))
    for bins in range(max_bins, 1, -1):
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(h_eval, quantiles)
        edges = np.asarray(edges, dtype=np.float64)
        for i in range(1, edges.shape[0]):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        bin_ids = assign_bins(h_eval, edges)
        counts = np.bincount(bin_ids, minlength=bins)
        if int(np.min(counts)) > 0:
            centers = np.asarray(
                [float(np.median(h_eval[bin_ids == b])) for b in range(bins)],
                dtype=np.float64,
            )
            return {
                "edges": edges,
                "bin_ids": bin_ids,
                "centers": centers,
                "counts": counts.astype(np.int64),
                "n_bins": np.asarray([bins], dtype=np.int64),
            }
    raise RuntimeError("Unable to construct nonempty support bins.")


def compute_profile_rows(errors: np.ndarray, h_values: np.ndarray, edges: np.ndarray) -> List[Dict[str, float]]:
    bin_ids = assign_bins(h_values, edges)
    n_bins = edges.shape[0] - 1
    rows: List[Dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_id": int(b + 1),
                "h_bin_center": float(np.median(h_values[mask])),
                "mse_bin": float(np.mean(errors[mask])),
                "n_bin": int(mask.sum()),
            }
        )
    return rows


def slope_loglog(h_centers: np.ndarray, mse_values: np.ndarray) -> float:
    mask = (h_centers > 0.0) & (mse_values > 0.0)
    if int(np.sum(mask)) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(h_centers[mask]), np.log(mse_values[mask]), deg=1)
    return float(coeffs[0])


def compute_profile_metrics(
    errors: np.ndarray,
    h_eval: np.ndarray,
    bin_edges: np.ndarray,
    weak_bins: Sequence[int],
    dense_bins: Sequence[int],
) -> Dict[str, object]:
    bin_ids = assign_bins(h_eval, bin_edges)
    weak_mask = np.isin(bin_ids, np.asarray(weak_bins, dtype=np.int64))
    dense_mask = np.isin(bin_ids, np.asarray(dense_bins, dtype=np.int64))
    if not np.any(weak_mask) or not np.any(dense_mask):
        raise RuntimeError("Weak or dense evaluation region is empty.")

    global_mse = float(np.mean(errors))
    weak_mse = float(np.mean(errors[weak_mask]))
    dense_mse = float(np.mean(errors[dense_mask]))
    gap = float(weak_mse / (dense_mse + 1e-12))

    profile_rows = compute_profile_rows(errors=errors, h_values=h_eval, edges=bin_edges)
    h_centers = np.asarray([row["h_bin_center"] for row in profile_rows], dtype=np.float64)
    mse_bins = np.asarray([row["mse_bin"] for row in profile_rows], dtype=np.float64)
    slope = slope_loglog(h_centers, np.maximum(mse_bins, 1e-12))
    pos_slope = float(max(slope, 0.0)) if not math.isnan(slope) else float("nan")
    profile_var = float(np.var(mse_bins))
    return {
        "global_mse": global_mse,
        "weak_mse": weak_mse,
        "dense_mse": dense_mse,
        "gap": gap,
        "slope": slope,
        "pos_slope": pos_slope,
        "profile_var": profile_var,
        "profile_rows": profile_rows,
    }


def pairwise_sqdist(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    sqX = np.sum(X * X, axis=1)[:, None]
    sqZ = np.sum(Z * Z, axis=1)[None, :]
    return np.maximum(sqX + sqZ - 2.0 * X @ Z.T, 0.0)


def farthest_first_select_positions(
    X_candidates: np.ndarray,
    X_ref: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    n_candidates = X_candidates.shape[0]
    if n_candidates <= batch_size:
        return np.arange(n_candidates, dtype=np.int64)

    if X_ref.shape[0] == 0:
        min_sqdist = np.full(n_candidates, np.inf, dtype=np.float64)
    else:
        min_sqdist = np.min(pairwise_sqdist(X_candidates, X_ref), axis=1)

    selected: List[int] = []
    available = np.ones(n_candidates, dtype=bool)
    for _ in range(batch_size):
        scores = np.where(available, min_sqdist, -np.inf)
        idx = int(np.argmax(scores))
        selected.append(idx)
        available[idx] = False
        new_sqdist = pairwise_sqdist(X_candidates, X_candidates[idx : idx + 1])[:, 0]
        min_sqdist = np.minimum(min_sqdist, new_sqdist)
    return np.asarray(selected, dtype=np.int64)


def select_random(n_pool: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(rng.choice(n_pool, size=batch_size, replace=False), dtype=np.int64)


def select_top_diverse_scores(
    X_pool: np.ndarray,
    X_train: np.ndarray,
    scores: np.ndarray,
    batch_size: int,
    shortlist_multiplier: int = 8,
) -> np.ndarray:
    shortlist_size = min(X_pool.shape[0], max(300, shortlist_multiplier * batch_size))
    shortlist = np.argsort(-scores, kind="mergesort")[:shortlist_size]
    pos = farthest_first_select_positions(X_pool[shortlist], X_train, batch_size=batch_size)
    return np.asarray(shortlist[pos], dtype=np.int64)


def select_from_ranked_bins_diverse(
    X_pool: np.ndarray,
    X_train: np.ndarray,
    pool_bin_ids: np.ndarray,
    ranked_bins: Sequence[int],
    batch_size: int,
) -> np.ndarray:
    selected: List[int] = []
    for bin_id in ranked_bins:
        candidates = np.flatnonzero(pool_bin_ids == int(bin_id))
        if candidates.size == 0:
            continue
        X_ref = (
            X_train
            if not selected
            else np.vstack([X_train, X_pool[np.asarray(selected, dtype=np.int64)]])
        )
        need = batch_size - len(selected)
        pos = farthest_first_select_positions(
            X_pool[candidates], X_ref, batch_size=min(need, candidates.size)
        )
        selected.extend(int(candidates[p]) for p in pos)
        if len(selected) >= batch_size:
            break

    if len(selected) < batch_size:
        remaining = np.setdiff1d(
            np.arange(X_pool.shape[0]), np.asarray(selected, dtype=np.int64), assume_unique=False
        )
        X_ref = (
            X_train
            if not selected
            else np.vstack([X_train, X_pool[np.asarray(selected, dtype=np.int64)]])
        )
        need = batch_size - len(selected)
        pos = farthest_first_select_positions(
            X_pool[remaining], X_ref, batch_size=min(need, remaining.size)
        )
        selected.extend(int(remaining[p]) for p in pos)
    return np.asarray(selected[:batch_size], dtype=np.int64)


def rule_short_name(rule: str) -> str:
    return rule.replace(" acquisition", "")


def regime_short_name(regime: str) -> str:
    return REGIME_SHORT.get(regime, regime.replace("_", " "))


def profile_row_maps(
    profile_rows: Sequence[Dict[str, float]]
) -> tuple[Dict[int, float], Dict[int, float], Dict[int, int]]:
    mse_map: Dict[int, float] = {}
    h_map: Dict[int, float] = {}
    count_map: Dict[int, int] = {}
    for row in profile_rows:
        bin_zero = int(row["bin_id"]) - 1
        mse_map[bin_zero] = float(row["mse_bin"])
        h_map[bin_zero] = float(row["h_bin_center"])
        count_map[bin_zero] = int(row["n_bin"])
    return mse_map, h_map, count_map


def in_circle_region(X: np.ndarray, center: Sequence[float], radius: float) -> np.ndarray:
    center_arr = np.asarray(center, dtype=np.float64)
    sqdist = np.sum((np.asarray(X, dtype=np.float64) - center_arr[None, :]) ** 2, axis=1)
    return sqdist <= float(radius) ** 2


def in_distractor_region(X: np.ndarray, cfg: Dict[str, object]) -> np.ndarray:
    region = cfg.get("distractor_region")
    if region is None:
        return np.zeros(X.shape[0], dtype=bool)
    return (X[:, 0] >= float(region["x1_min"])) & (X[:, 1] >= float(region["x2_min"]))


def robust_support_contrast(h_values: np.ndarray) -> float:
    q05 = float(np.quantile(h_values, 0.05))
    q95 = float(np.quantile(h_values, 0.95))
    return float(q95 / max(q05, 1e-12))


def run_acquisition_rule(
    regime_name: str,
    cfg: Dict[str, object],
    rule: str,
    rep: int,
    rng: np.random.Generator,
    X_train_init: np.ndarray,
    y_train_init: np.ndarray,
    X_pool_init: np.ndarray,
    y_pool_clean_init: np.ndarray,
    y_pool_noisy_init: np.ndarray,
    X_eval: np.ndarray,
    y_eval_clean: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, List[Dict[str, float | int | str | bool]]]:
    X_train = np.asarray(X_train_init, dtype=np.float64).copy()
    y_train = np.asarray(y_train_init, dtype=np.float64).copy()
    X_pool = np.asarray(X_pool_init, dtype=np.float64).copy()
    y_pool_clean = np.asarray(y_pool_clean_init, dtype=np.float64).copy()
    y_pool_noisy = np.asarray(y_pool_noisy_init, dtype=np.float64).copy()

    curve_rows: List[Dict[str, float | int | str | bool]] = []
    profile_rows_out: List[Dict[str, float | int | str | bool]] = []
    selected_rows: List[Dict[str, float | int | str | bool]] = []

    relevant_mask_eval = in_circle_region(
        X_eval, center=cfg["bump_center"], radius=float(cfg["relevant_radius"])
    )

    for round_idx in range(args.n_rounds + 1):
        model = fit_krr(
            X_train=X_train,
            y_train=y_train,
            gamma=float(args.gamma),
            lambda_reg=float(args.lambda_reg),
        )
        nbrs = make_support_index(X_train=X_train, k_support=int(args.k_support))
        h_eval = compute_support_scores(X_query=X_eval, nbrs=nbrs)
        bins = make_support_bins(h_eval=h_eval, n_bins=int(args.n_bins))
        edges = np.asarray(bins["edges"], dtype=np.float64)
        n_bins_eff = int(np.asarray(bins["n_bins"], dtype=np.int64)[0])
        n_region_bins = max(1, int(math.ceil(0.2 * n_bins_eff)))
        dense_bins = tuple(range(n_region_bins))
        weak_bins = tuple(range(n_bins_eff - n_region_bins, n_bins_eff))

        pred_eval = predict_krr(model=model, X_eval=X_eval)
        errors_eval = (pred_eval - y_eval_clean) ** 2
        metrics = compute_profile_metrics(
            errors=errors_eval,
            h_eval=h_eval,
            bin_edges=edges,
            weak_bins=weak_bins,
            dense_bins=dense_bins,
        )
        mse_map, h_map, _ = profile_row_maps(metrics["profile_rows"])

        curve_rows.append(
            {
                "regime": regime_name,
                "repetition": int(rep),
                "rule": rule,
                "round": int(round_idx),
                "global_mse": float(metrics["global_mse"]),
                "weak_mse": float(metrics["weak_mse"]),
                "dense_mse": float(metrics["dense_mse"]),
                "gap": float(metrics["gap"]),
                "slope": float(metrics["slope"]),
                "pos_slope": float(metrics["pos_slope"]),
                "profile_var": float(metrics["profile_var"]),
                "support_contrast": robust_support_contrast(h_eval),
                "relevant_region_eval_fraction": float(np.mean(relevant_mask_eval)),
                "n_train": int(X_train.shape[0]),
                "n_pool_remaining": int(X_pool.shape[0]),
            }
        )
        for row in metrics["profile_rows"]:
            profile_rows_out.append(
                {
                    "regime": regime_name,
                    "repetition": int(rep),
                    "rule": rule,
                    "round": int(round_idx),
                    "bin_id": int(row["bin_id"]),
                    "h_bin_center": float(row["h_bin_center"]),
                    "mse_bin": float(row["mse_bin"]),
                    "n_bin": int(row["n_bin"]),
                }
            )

        if round_idx == args.n_rounds:
            break
        if X_pool.shape[0] < args.batch_size:
            raise RuntimeError("Acquisition pool does not contain enough points for the next round.")

        h_pool = compute_support_scores(X_query=X_pool, nbrs=nbrs)
        pool_bin_ids = assign_bins(h_pool, edges)
        pred_pool = predict_krr(model=model, X_eval=X_pool)
        oracle_sq_error_pool = (pred_pool - y_pool_clean) ** 2
        predicted_sq_error_proxy = np.asarray([mse_map[int(b)] for b in pool_bin_ids], dtype=np.float64)
        posterior_var_pool = None

        if rule == "Random acquisition":
            selected_idx = select_random(
                n_pool=X_pool.shape[0], batch_size=int(args.batch_size), rng=rng
            )
            pool_score = np.full(X_pool.shape[0], np.nan, dtype=np.float64)
        elif rule == "Support-only acquisition":
            pool_score = h_pool.copy()
            selected_idx = select_top_diverse_scores(
                X_pool=X_pool,
                X_train=X_train,
                scores=pool_score,
                batch_size=int(args.batch_size),
            )
        elif rule == "Error-only acquisition":
            pool_score = predicted_sq_error_proxy.copy()
            ranked_bins = sorted(mse_map.keys(), key=lambda b: (-mse_map[b], b))
            selected_idx = select_from_ranked_bins_diverse(
                X_pool=X_pool,
                X_train=X_train,
                pool_bin_ids=pool_bin_ids,
                ranked_bins=ranked_bins,
                batch_size=int(args.batch_size),
            )
        elif rule == "Posterior-variance acquisition":
            posterior_var_pool = posterior_variance_krr(model=model, X_query=X_pool)
            pool_score = posterior_var_pool.copy()
            selected_idx = select_top_diverse_scores(
                X_pool=X_pool,
                X_train=X_train,
                scores=pool_score,
                batch_size=int(args.batch_size),
            )
        elif rule == "Profile-aware acquisition":
            eps = 1e-12
            median_e = float(np.median(list(mse_map.values())))
            median_h = float(np.median(list(h_map.values())))
            bin_scores = {
                b: (mse_map[b] / (median_e + eps))
                * ((h_map[b] / (median_h + eps)) ** float(args.gamma_acq))
                for b in mse_map
            }
            pool_score = np.asarray([bin_scores[int(b)] for b in pool_bin_ids], dtype=np.float64)
            ranked_bins = sorted(bin_scores.keys(), key=lambda b: (-bin_scores[b], b))
            selected_idx = select_from_ranked_bins_diverse(
                X_pool=X_pool,
                X_train=X_train,
                pool_bin_ids=pool_bin_ids,
                ranked_bins=ranked_bins,
                batch_size=int(args.batch_size),
            )
        else:
            raise ValueError(f"Unknown acquisition rule: {rule}")

        relevant_mask_pool = in_circle_region(
            X_pool, center=cfg["bump_center"], radius=float(cfg["relevant_radius"])
        )
        distractor_mask_pool = in_distractor_region(X_pool, cfg)
        for idx in selected_idx:
            bin_zero = int(pool_bin_ids[idx])
            selected_rows.append(
                {
                    "regime": regime_name,
                    "repetition": int(rep),
                    "rule": rule,
                    "round": int(round_idx + 1),
                    "x1": float(X_pool[idx, 0]),
                    "x2": float(X_pool[idx, 1]),
                    "support_score": float(h_pool[idx]),
                    "oracle_sq_error": float(oracle_sq_error_pool[idx]),
                    "predicted_sq_error_proxy": float(predicted_sq_error_proxy[idx]),
                    "score_used": float(pool_score[idx]),
                    "selected_bin": int(bin_zero + 1),
                    "is_weak_bin": bool(bin_zero in weak_bins),
                    "in_relevant_region": bool(relevant_mask_pool[idx]),
                    "in_distractor_region": bool(distractor_mask_pool[idx]),
                }
            )

        X_selected = X_pool[selected_idx]
        y_selected = y_pool_noisy[selected_idx]
        X_train = np.vstack([X_train, X_selected])
        y_train = np.concatenate([y_train, y_selected], axis=0)

        keep = np.ones(X_pool.shape[0], dtype=bool)
        keep[selected_idx] = False
        X_pool = X_pool[keep]
        y_pool_clean = y_pool_clean[keep]
        y_pool_noisy = y_pool_noisy[keep]

    return {
        "curve_rows": curve_rows,
        "profile_rows": profile_rows_out,
        "selected_rows": selected_rows,
    }


def aggregate_round_metric(
    curve_rows: Sequence[Dict[str, float | int | str | bool]],
    metric_name: str,
    regime_name: str,
    rule: str,
    round_idx: int,
) -> np.ndarray:
    return np.asarray(
        [
            float(row[metric_name])
            for row in curve_rows
            if row["regime"] == regime_name and row["rule"] == rule and int(row["round"]) == int(round_idx)
        ],
        dtype=np.float64,
    )


def aggregate_final_and_auc(
    curve_rows: Sequence[Dict[str, float | int | str | bool]],
    regimes: Sequence[str],
    final_round: int,
) -> tuple[
    List[Dict[str, float | str]],
    List[Dict[str, float | str]],
    List[Dict[str, float | str]],
]:
    final_rows: List[Dict[str, float | str]] = []
    auc_rows: List[Dict[str, float | str]] = []
    combined_rows: List[Dict[str, float | str]] = []

    for regime_name in regimes:
        for rule in RULE_DISPLAY_ORDER:
            final_cur = [
                row
                for row in curve_rows
                if row["regime"] == regime_name and row["rule"] == rule and int(row["round"]) == int(final_round)
            ]
            if not final_cur:
                continue
            global_vals = np.asarray([float(row["global_mse"]) for row in final_cur], dtype=np.float64)
            weak_vals = np.asarray([float(row["weak_mse"]) for row in final_cur], dtype=np.float64)
            dense_vals = np.asarray([float(row["dense_mse"]) for row in final_cur], dtype=np.float64)
            gap_vals = np.asarray([float(row["gap"]) for row in final_cur], dtype=np.float64)
            slope_vals = np.asarray([float(row["slope"]) for row in final_cur], dtype=np.float64)
            pos_slope_vals = np.asarray([float(row["pos_slope"]) for row in final_cur], dtype=np.float64)
            profile_var_vals = np.asarray([float(row["profile_var"]) for row in final_cur], dtype=np.float64)

            per_rep_weak_auc = []
            per_rep_gap_auc = []
            per_rep_slope_auc = []
            reps = sorted(set(int(row["repetition"]) for row in final_cur))
            for rep in reps:
                rep_rows = [
                    row
                    for row in curve_rows
                    if row["regime"] == regime_name and row["rule"] == rule and int(row["repetition"]) == rep
                ]
                rep_rows = sorted(rep_rows, key=lambda row: int(row["round"]))
                per_rep_weak_auc.append(float(np.mean([float(row["weak_mse"]) for row in rep_rows])))
                per_rep_gap_auc.append(float(np.mean([float(row["gap"]) for row in rep_rows])))
                per_rep_slope_auc.append(float(np.mean([float(row["pos_slope"]) for row in rep_rows])))

            weak_auc = np.asarray(per_rep_weak_auc, dtype=np.float64)
            gap_auc = np.asarray(per_rep_gap_auc, dtype=np.float64)
            slope_auc = np.asarray(per_rep_slope_auc, dtype=np.float64)

            final_row = {
                "regime": regime_name,
                "rule": rule,
                "global_mse": float(np.mean(global_vals)),
                "weak_mse": float(np.mean(weak_vals)),
                "dense_mse": float(np.mean(dense_vals)),
                "gap": float(np.mean(gap_vals)),
                "slope": float(np.nanmean(slope_vals)),
                "pos_slope": float(np.nanmean(pos_slope_vals)),
                "profile_var": float(np.mean(profile_var_vals)),
            }
            auc_row = {
                "regime": regime_name,
                "rule": rule,
                "weak_auc": float(np.mean(weak_auc)),
                "gap_auc": float(np.mean(gap_auc)),
                "slope_auc": float(np.mean(slope_auc)),
            }
            combined_row = dict(final_row)
            combined_row.update(auc_row)

            final_rows.append(final_row)
            auc_rows.append(auc_row)
            combined_rows.append(combined_row)
    return final_rows, auc_rows, combined_rows


def summarize_allocation(
    selected_rows: Sequence[Dict[str, float | int | str | bool]],
    regimes: Sequence[str],
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for regime_name in regimes:
        for rule in RULE_DISPLAY_ORDER:
            cur = [row for row in selected_rows if row["regime"] == regime_name and row["rule"] == rule]
            if not cur:
                continue
            support_vals = np.asarray([float(row["support_score"]) for row in cur], dtype=np.float64)
            oracle_vals = np.asarray([float(row["oracle_sq_error"]) for row in cur], dtype=np.float64)
            rows.append(
                {
                    "regime": regime_name,
                    "rule": rule,
                    "n_selected": int(len(cur)),
                    "fraction_relevant_region": float(np.mean([bool(row["in_relevant_region"]) for row in cur])),
                    "fraction_distractor_region": float(np.mean([bool(row["in_distractor_region"]) for row in cur])),
                    "fraction_weak_bin": float(np.mean([bool(row["is_weak_bin"]) for row in cur])),
                    "median_support_radius": float(np.median(support_vals)),
                    "median_oracle_sq_error": float(np.median(oracle_vals)),
                }
            )
    return rows


def summarize_bin_round_allocation(
    selected_rows: Sequence[Dict[str, float | int | str | bool]],
    regimes: Sequence[str],
    n_rounds: int,
    n_bins: int,
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for regime_name in regimes:
        for rule in RULE_DISPLAY_ORDER:
            cur = [row for row in selected_rows if row["regime"] == regime_name and row["rule"] == rule]
            reps = sorted(set(int(row["repetition"]) for row in cur))
            for round_idx in range(1, n_rounds + 1):
                for bin_id in range(1, n_bins + 1):
                    counts = []
                    for rep in reps:
                        count = sum(
                            1
                            for row in cur
                            if int(row["repetition"]) == rep
                            and int(row["round"]) == round_idx
                            and int(row["selected_bin"]) == bin_id
                        )
                        counts.append(count)
                    counts_arr = np.asarray(counts, dtype=np.float64)
                    rows.append(
                        {
                            "regime": regime_name,
                            "rule": rule,
                            "round": int(round_idx),
                            "bin_id": int(bin_id),
                            "mean_count": float(np.mean(counts_arr)),
                            "total_count": int(np.sum(counts_arr)),
                        }
                    )
    return rows


def compare_profile_vs_support(
    combined_rows: Sequence[Dict[str, float | str]],
    regime_name: str,
) -> Dict[str, float]:
    support_row = next(
        row for row in combined_rows if row["regime"] == regime_name and row["rule"] == "Support-only acquisition"
    )
    profile_row = next(
        row for row in combined_rows if row["regime"] == regime_name and row["rule"] == "Profile-aware acquisition"
    )
    weak_improvement = (float(support_row["weak_auc"]) - float(profile_row["weak_auc"])) / max(
        float(support_row["weak_auc"]), 1e-12
    )
    gap_improvement = (float(support_row["gap_auc"]) - float(profile_row["gap_auc"])) / max(
        float(support_row["gap_auc"]), 1e-12
    )
    return {
        "weak_auc_improvement_fraction": float(weak_improvement),
        "gap_auc_improvement_fraction": float(gap_improvement),
    }


def plot_metric_with_band(
    ax: plt.Axes,
    curve_rows: Sequence[Dict[str, float | int | str | bool]],
    regime_name: str,
    metric_name: str,
    ylabel: str,
    n_rounds: int,
) -> None:
    style_map = make_style_map(RULE_DISPLAY_ORDER)
    for rule in RULE_DISPLAY_ORDER:
        xs = []
        ys = []
        yerr = []
        for round_idx in range(n_rounds + 1):
            vals = aggregate_round_metric(curve_rows, metric_name, regime_name, rule, round_idx)
            xs.append(round_idx)
            ys.append(float(np.mean(vals)))
            yerr.append(float(np.std(vals, ddof=0) / math.sqrt(max(1, vals.shape[0]))))
        x_arr = np.asarray(xs, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        err_arr = np.asarray(yerr, dtype=np.float64)
        style = style_map[rule]
        ax.plot(
            x_arr,
            y_arr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=4.0,
            color=style["color"],
            label=rule,
        )
        lower = np.maximum(y_arr - err_arr, 0.0)
        ax.fill_between(x_arr, lower, y_arr + err_arr, color=style["color"], alpha=0.14)
    ax.set_xlabel("Acquisition round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18)


def make_regime_trajectory_figure(
    figures_dir: Path,
    curve_rows: Sequence[Dict[str, float | int | str | bool]],
    regime_name: str,
    n_rounds: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6), constrained_layout=True)
    plot_metric_with_band(axes[0], curve_rows, regime_name, "weak_mse", "Weak-support MSE", n_rounds)
    plot_metric_with_band(axes[1], curve_rows, regime_name, "gap", "Weak/dense gap", n_rounds)
    plot_metric_with_band(axes[2], curve_rows, regime_name, "pos_slope", "Positive profile slope", n_rounds)
    axes[0].set_title(f"{regime_short_name(regime_name)}: Weak MSE")
    axes[1].set_title(f"{regime_short_name(regime_name)}: Gap")
    axes[2].set_title(f"{regime_short_name(regime_name)}: Slope")
    handles, labels = axes[0].get_legend_handles_labels()
    stem = f"exp7_{regime_name}_trajectories"
    for idx, ax in enumerate(axes, start=1):
        save_axes_group_panel(fig, [ax], figures_dir / f"{stem}_panel_{idx}.pdf")
    save_legend_figure(handles, labels, figures_dir / f"{stem}_legend.pdf", ncol=3)
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def representative_round_colorbar(figures_dir: Path, stem: str, n_rounds: int) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 0.8))
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=1, vmax=max(1, n_rounds))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("Acquisition round")
    fig.savefig(figures_dir / f"{stem}_legend.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_regime_overlays(ax: plt.Axes, cfg: Dict[str, object]) -> None:
    center = np.asarray(cfg["bump_center"], dtype=np.float64)
    radius = float(cfg["relevant_radius"])
    circle = patches.Circle(
        (float(center[0]), float(center[1])),
        radius=radius,
        fill=False,
        linewidth=1.6,
        linestyle="-",
        edgecolor="black",
        alpha=1.0,
        zorder=6,
    )
    ax.add_patch(circle)
    region = cfg.get("distractor_region")
    if region is not None:
        rect_base = patches.Rectangle(
            (float(region["x1_min"]), float(region["x2_min"])),
            1.0 - float(region["x1_min"]),
            1.0 - float(region["x2_min"]),
            fill=False,
            linewidth=4.0,
            linestyle="--",
            edgecolor="white",
            alpha=0.95,
            zorder=5,
            clip_on=False,
        )
        rect = patches.Rectangle(
            (float(region["x1_min"]), float(region["x2_min"])),
            1.0 - float(region["x1_min"]),
            1.0 - float(region["x2_min"]),
            fill=False,
            linewidth=2.2,
            linestyle="--",
            edgecolor="black",
            alpha=1.0,
            zorder=6,
            clip_on=False,
        )
        ax.add_patch(rect_base)
        ax.add_patch(rect)


def make_allocation_map_figure(
    figures_dir: Path,
    selected_rows: Sequence[Dict[str, float | int | str | bool]],
    representative_data: Dict[str, Dict[str, np.ndarray]],
    regime_name: str,
    cfg: Dict[str, object],
    n_rounds: int,
) -> None:
    X_train_init = np.asarray(representative_data[regime_name]["X_train_init"], dtype=np.float64)
    rep_rows = [
        row
        for row in selected_rows
        if row["regime"] == regime_name and int(row["repetition"]) == 0
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.6), constrained_layout=True)
    axes = axes.ravel()
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=1, vmax=max(1, n_rounds))
    for ax, rule in zip(axes, RULE_DISPLAY_ORDER):
        cur = [row for row in rep_rows if row["rule"] == rule]
        ax.scatter(
            X_train_init[:, 0],
            X_train_init[:, 1],
            s=10,
            c="#c7c7c7",
            alpha=0.55,
            edgecolors="none",
        )
        if cur:
            xs = np.asarray([float(row["x1"]) for row in cur], dtype=np.float64)
            ys = np.asarray([float(row["x2"]) for row in cur], dtype=np.float64)
            rounds = np.asarray([int(row["round"]) for row in cur], dtype=np.int64)
            ax.scatter(
                xs,
                ys,
                s=18,
                c=rounds,
                cmap=cmap,
                norm=norm,
                edgecolors="black",
                linewidths=0.15,
            )
        draw_regime_overlays(ax, cfg)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(rule_short_name(rule))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.15)
    axes[-1].axis("off")
    stem = f"exp7_{regime_name}_allocation_map"
    for idx, ax in enumerate(axes[:-1], start=1):
        save_axes_group_panel(fig, [ax], figures_dir / f"{stem}_panel_{idx}.pdf")
    representative_round_colorbar(figures_dir, stem, n_rounds)
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_bin_round_heatmap(
    figures_dir: Path,
    bin_round_rows: Sequence[Dict[str, float | int | str]],
    regimes: Sequence[str],
    n_rounds: int,
    n_bins: int,
) -> None:
    fig, axes = plt.subplots(len(regimes), len(RULE_DISPLAY_ORDER), figsize=(16.5, 6.5), constrained_layout=True)
    if len(regimes) == 1:
        axes = np.asarray([axes], dtype=object)
    vmax = max(float(row["mean_count"]) for row in bin_round_rows) if bin_round_rows else 1.0
    for row_idx, regime_name in enumerate(regimes):
        for col_idx, rule in enumerate(RULE_DISPLAY_ORDER):
            ax = axes[row_idx, col_idx]
            grid = np.zeros((n_bins, n_rounds), dtype=np.float64)
            cur = [
                row
                for row in bin_round_rows
                if row["regime"] == regime_name and row["rule"] == rule
            ]
            for row in cur:
                grid[int(row["bin_id"]) - 1, int(row["round"]) - 1] = float(row["mean_count"])
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_title(f"{regime_short_name(regime_name)}\n{rule_short_name(rule)}", fontsize=9)
            ax.set_xlabel("Round")
            ax.set_ylabel("Support bin")
            ax.set_xticks(range(n_rounds))
            ax.set_xticklabels(range(1, n_rounds + 1), fontsize=7)
            ax.set_yticks(range(0, n_bins, max(1, n_bins // 5)))
            ax.set_yticklabels(range(1, n_bins + 1, max(1, n_bins // 5)), fontsize=7)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Mean acquisitions per split")
    fig.savefig(figures_dir / "exp7_bin_round_allocation_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_regime_table_tex(
    path: Path,
    rows: Sequence[Dict[str, float | str]],
    regime_name: str,
) -> None:
    lines = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Acquisition rule & Global MSE & Weak MSE & Dense MSE & Gap & Slope & Weak AUC & Gap AUC & Slope AUC \\\\",
        "\\midrule",
    ]
    for rule in RULE_DISPLAY_ORDER:
        row = next(r for r in rows if r["regime"] == regime_name and r["rule"] == rule)
        lines.append(
            f"{RULE_TABLE_LABELS[rule]} & "
            f"{float(row['global_mse']):.4f} & "
            f"{float(row['weak_mse']):.4f} & "
            f"{float(row['dense_mse']):.4f} & "
            f"{float(row['gap']):.3f} & "
            f"{float(row['slope']):.3f} & "
            f"{float(row['weak_auc']):.4f} & "
            f"{float(row['gap_auc']):.3f} & "
            f"{float(row['slope_auc']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_two_regime_table_tex(path: Path, rows: Sequence[Dict[str, float | str]], regimes: Sequence[str]) -> None:
    lines = [
        "\\begin{tabular}{llcccccccc}",
        "\\toprule",
        "Regime & Acquisition rule & Global MSE & Weak MSE & Dense MSE & Gap & Slope & Weak AUC & Gap AUC & Slope AUC \\\\",
        "\\midrule",
    ]
    for regime_name in regimes:
        first = True
        for rule in RULE_DISPLAY_ORDER:
            row = next(r for r in rows if r["regime"] == regime_name and r["rule"] == rule)
            regime_label = REGIME_SHORT.get(regime_name, regime_name.replace("_", " ")) if first else ""
            first = False
            lines.append(
                f"{regime_label} & {RULE_TABLE_LABELS[rule]} & "
                f"{float(row['global_mse']):.4f} & "
                f"{float(row['weak_mse']):.4f} & "
                f"{float(row['dense_mse']):.4f} & "
                f"{float(row['gap']):.3f} & "
                f"{float(row['slope']):.3f} & "
                f"{float(row['weak_auc']):.4f} & "
                f"{float(row['gap_auc']):.3f} & "
                f"{float(row['slope_auc']):.3f} \\\\"
            )
        if regime_name != regimes[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_allocation_table_tex(path: Path, rows: Sequence[Dict[str, float | str]], regimes: Sequence[str]) -> None:
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Regime & Acquisition rule & Relevant frac. & Distractor frac. & Weak-bin frac. & Median support & Median oracle error \\\\",
        "\\midrule",
    ]
    for regime_name in regimes:
        first = True
        for rule in RULE_DISPLAY_ORDER:
            row = next(r for r in rows if r["regime"] == regime_name and r["rule"] == rule)
            regime_label = REGIME_SHORT.get(regime_name, regime_name.replace("_", " ")) if first else ""
            first = False
            lines.append(
                f"{regime_label} & {RULE_TABLE_LABELS[rule]} & "
                f"{float(row['fraction_relevant_region']):.3f} & "
                f"{float(row['fraction_distractor_region']):.3f} & "
                f"{float(row['fraction_weak_bin']):.3f} & "
                f"{float(row['median_support_radius']):.3f} & "
                f"{float(row['median_oracle_sq_error']):.4f} \\\\"
            )
        if regime_name != regimes[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())


def make_legacy_support_limited_exports(
    outdir: Path,
    combined_rows: Sequence[Dict[str, float | str]],
    curve_rows: Sequence[Dict[str, float | int | str | bool]],
    regime_name: str,
    n_rounds: int,
) -> None:
    support_rows = [row for row in combined_rows if row["regime"] == regime_name]
    legacy_table_rows = [
        {
            "rule": RULE_TABLE_LABELS[str(row["rule"])],
            "global_mse": row["global_mse"],
            "weak_mse": row["weak_mse"],
            "dense_mse": row["dense_mse"],
            "gap": row["gap"],
            "slope": row["slope"],
            "weak_auc": row["weak_auc"],
            "gap_auc": row["gap_auc"],
            "slope_auc": row["slope_auc"],
        }
        for row in support_rows
    ]
    save_csv(outdir / "tables" / "exp7_acquisition.csv", legacy_table_rows)
    save_regime_table_tex(outdir / "tables" / "exp7_acquisition.tex", support_rows, regime_name)

    legacy_curve_rows = [row for row in curve_rows if row["regime"] == regime_name]
    weak_curve_out = outdir / "figures" / "exp7_weak_mse_vs_round.pdf"
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    plot_metric_with_band(ax, legacy_curve_rows, regime_name, "weak_mse", "Weak-support MSE", n_rounds)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(weak_curve_out, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    plot_metric_with_band(axes[0], legacy_curve_rows, regime_name, "gap", "Weak/dense gap", n_rounds)
    plot_metric_with_band(axes[1], legacy_curve_rows, regime_name, "pos_slope", "Positive profile slope", n_rounds)
    handles, labels = axes[0].get_legend_handles_labels()
    save_axes_group_panel(fig, [axes[0]], outdir / "figures" / "exp7_gap_slope_vs_round_panel_gap.pdf")
    save_axes_group_panel(fig, [axes[1]], outdir / "figures" / "exp7_gap_slope_vs_round_panel_slope.pdf")
    save_legend_figure(handles, labels, outdir / "figures" / "exp7_gap_slope_vs_round_legend.pdf", ncol=3)
    fig.savefig(outdir / "figures" / "exp7_gap_slope_vs_round.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = apply_fast_mode(parse_args())
    requested_regimes = parse_regimes(args.regimes)
    outdir = resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    results_dir = outdir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    def run_regimes(regime_names: Sequence[str]) -> tuple[
        List[Dict[str, float | int | str | bool]],
        List[Dict[str, float | int | str | bool]],
        List[Dict[str, float | int | str | bool]],
        Dict[str, Dict[str, np.ndarray]],
    ]:
        curve_rows_local: List[Dict[str, float | int | str | bool]] = []
        profile_rows_local: List[Dict[str, float | int | str | bool]] = []
        selected_rows_local: List[Dict[str, float | int | str | bool]] = []
        representative_data: Dict[str, Dict[str, np.ndarray]] = {}

        for regime_name in regime_names:
            cfg = regime_config(regime_name)
            for rep in range(args.n_reps):
                rng = np.random.default_rng(args.seed + 10000 * REGIME_ORDER.index(regime_name if regime_name in REGIME_ORDER else "error_support_conflict") + 1000 * rep + 29)
                X_train_init = sample_mixture_design(args.n_initial, cfg, rng)
                X_pool_init = sample_uniform_points(args.n_pool, rng)
                X_eval = sample_uniform_points(args.n_eval, rng)
                y_train_clean = f_star(X_train_init, cfg)
                y_train_init = add_noisy_labels(y_train_clean, sigma=float(args.sigma), rng=rng)
                y_pool_clean = f_star(X_pool_init, cfg)
                y_pool_noisy = add_noisy_labels(y_pool_clean, sigma=float(args.sigma), rng=rng)
                y_eval_clean = f_star(X_eval, cfg)

                if rep == 0:
                    representative_data[regime_name] = {
                        "X_train_init": X_train_init,
                        "X_pool_init": X_pool_init,
                        "X_eval": X_eval,
                    }

                for rule_idx, rule in enumerate(RULE_DISPLAY_ORDER):
                    rule_rng = np.random.default_rng(
                        args.seed
                        + 10000 * REGIME_ORDER.index(regime_name if regime_name in REGIME_ORDER else "error_support_conflict")
                        + 1000 * rep
                        + 100 * (rule_idx + 1)
                        + 7
                    )
                    out = run_acquisition_rule(
                        regime_name=regime_name,
                        cfg=cfg,
                        rule=rule,
                        rep=rep,
                        rng=rule_rng,
                        X_train_init=X_train_init,
                        y_train_init=y_train_init,
                        X_pool_init=X_pool_init,
                        y_pool_clean_init=y_pool_clean,
                        y_pool_noisy_init=y_pool_noisy,
                        X_eval=X_eval,
                        y_eval_clean=y_eval_clean,
                        args=args,
                    )
                    curve_rows_local.extend(out["curve_rows"])
                    profile_rows_local.extend(out["profile_rows"])
                    selected_rows_local.extend(out["selected_rows"])
        return curve_rows_local, profile_rows_local, selected_rows_local, representative_data

    curve_rows, profile_rows, selected_rows, representative_data = run_regimes(requested_regimes)
    final_rows, auc_rows, combined_rows = aggregate_final_and_auc(
        curve_rows=curve_rows,
        regimes=requested_regimes,
        final_round=int(args.n_rounds),
    )

    conflict_used = "error_support_conflict" in requested_regimes
    strong_triggered = False
    if conflict_used and not args.disable_strong_conflict_fallback:
        gains = compare_profile_vs_support(combined_rows, "error_support_conflict")
        if gains["weak_auc_improvement_fraction"] < 0.10 and gains["gap_auc_improvement_fraction"] < 0.10:
            strong_triggered = True
            requested_regimes = [r for r in requested_regimes if r != "error_support_conflict"] + ["error_support_conflict_strong"]
            curve_rows, profile_rows, selected_rows, representative_data = run_regimes(requested_regimes)
            final_rows, auc_rows, combined_rows = aggregate_final_and_auc(
                curve_rows=curve_rows,
                regimes=requested_regimes,
                final_round=int(args.n_rounds),
            )

    allocation_rows = summarize_allocation(selected_rows, requested_regimes)
    bin_round_rows = summarize_bin_round_allocation(
        selected_rows=selected_rows,
        regimes=requested_regimes,
        n_rounds=int(args.n_rounds),
        n_bins=int(args.n_bins),
    )

    save_csv(results_dir / "exp7_all_rounds.csv", curve_rows)
    save_csv(results_dir / "exp7_final_metrics_by_regime.csv", final_rows)
    save_csv(results_dir / "exp7_auc_metrics_by_regime.csv", auc_rows)
    save_csv(results_dir / "exp7_allocation_summary.csv", allocation_rows)
    save_csv(results_dir / "exp7_bin_round_allocation.csv", bin_round_rows)
    save_csv(results_dir / "exp7_selected_points.csv", selected_rows)
    save_csv(results_dir / "exp7_acquisition_curves.csv", curve_rows)
    save_csv(results_dir / "exp7_profiles_by_round.csv", profile_rows)

    if any(row["regime"] == "support_limited" for row in combined_rows):
        save_regime_table_tex(
            tables_dir / "exp7_support_limited_main_table.tex",
            combined_rows,
            "support_limited",
        )
    conflict_regime_name = (
        "error_support_conflict_strong"
        if any(row["regime"] == "error_support_conflict_strong" for row in combined_rows)
        else "error_support_conflict"
    )
    if any(row["regime"] == conflict_regime_name for row in combined_rows):
        save_regime_table_tex(tables_dir / "exp7_conflict_main_table.tex", combined_rows, conflict_regime_name)
    save_two_regime_table_tex(tables_dir / "exp7_two_regime_main_table.tex", combined_rows, requested_regimes)
    save_allocation_table_tex(tables_dir / "exp7_allocation_summary_table.tex", allocation_rows, requested_regimes)

    if any(row["regime"] == "support_limited" for row in curve_rows):
        make_regime_trajectory_figure(figures_dir, curve_rows, "support_limited", int(args.n_rounds))
    if any(row["regime"] == conflict_regime_name for row in curve_rows):
        make_regime_trajectory_figure(figures_dir, curve_rows, conflict_regime_name, int(args.n_rounds))
    if "support_limited" in representative_data:
        make_allocation_map_figure(
            figures_dir=figures_dir,
            selected_rows=selected_rows,
            representative_data=representative_data,
            regime_name="support_limited",
            cfg=regime_config("support_limited"),
            n_rounds=int(args.n_rounds),
        )
    if conflict_regime_name in representative_data:
        make_allocation_map_figure(
            figures_dir=figures_dir,
            selected_rows=selected_rows,
            representative_data=representative_data,
            regime_name=conflict_regime_name,
            cfg=regime_config(conflict_regime_name),
            n_rounds=int(args.n_rounds),
        )
        copy_if_exists(
            figures_dir / f"exp7_{conflict_regime_name}_trajectories.pdf",
            figures_dir / "exp7_conflict_trajectories.pdf",
        )
        copy_if_exists(
            figures_dir / f"exp7_{conflict_regime_name}_trajectories_legend.pdf",
            figures_dir / "exp7_conflict_trajectories_legend.pdf",
        )
        for idx in range(1, 4):
            copy_if_exists(
                figures_dir / f"exp7_{conflict_regime_name}_trajectories_panel_{idx}.pdf",
                figures_dir / f"exp7_conflict_trajectories_panel_{idx}.pdf",
            )
        copy_if_exists(
            figures_dir / f"exp7_{conflict_regime_name}_allocation_map.pdf",
            figures_dir / "exp7_conflict_allocation_map.pdf",
        )
        copy_if_exists(
            figures_dir / f"exp7_{conflict_regime_name}_allocation_map_legend.pdf",
            figures_dir / "exp7_conflict_allocation_map_legend.pdf",
        )
        for idx in range(1, 6):
            copy_if_exists(
                figures_dir / f"exp7_{conflict_regime_name}_allocation_map_panel_{idx}.pdf",
                figures_dir / f"exp7_conflict_allocation_map_panel_{idx}.pdf",
            )
    make_bin_round_heatmap(
        figures_dir=figures_dir,
        bin_round_rows=bin_round_rows,
        regimes=requested_regimes,
        n_rounds=int(args.n_rounds),
        n_bins=int(args.n_bins),
    )

    if any(row["regime"] == "support_limited" for row in combined_rows):
        make_legacy_support_limited_exports(
            outdir=outdir,
            combined_rows=combined_rows,
            curve_rows=curve_rows,
            regime_name="support_limited",
            n_rounds=int(args.n_rounds),
        )

    def lookup_row(rows: Sequence[Dict[str, float | str]], regime_name: str, rule: str) -> Dict[str, float | str]:
        return next(row for row in rows if row["regime"] == regime_name and row["rule"] == rule)

    regime_comparisons: Dict[str, Dict[str, float]] = {}
    for regime_name in requested_regimes:
        if not any(row["regime"] == regime_name for row in combined_rows):
            continue
        support = lookup_row(combined_rows, regime_name, "Support-only acquisition")
        profile = lookup_row(combined_rows, regime_name, "Profile-aware acquisition")
        regime_comparisons[regime_name] = {
            "profile_vs_support_weak_auc_improvement_fraction": float(
                (float(support["weak_auc"]) - float(profile["weak_auc"])) / max(float(support["weak_auc"]), 1e-12)
            ),
            "profile_vs_support_gap_auc_improvement_fraction": float(
                (float(support["gap_auc"]) - float(profile["gap_auc"])) / max(float(support["gap_auc"]), 1e-12)
            ),
            "profile_vs_support_final_weak_improvement_fraction": float(
                (float(support["weak_mse"]) - float(profile["weak_mse"])) / max(float(support["weak_mse"]), 1e-12)
            ),
            "profile_vs_support_final_gap_improvement_fraction": float(
                (float(support["gap"]) - float(profile["gap"])) / max(float(support["gap"]), 1e-12)
            ),
        }

    summary = {
        "experiment": "exp7_profile_guided_acquisition",
        "parameters": {
            "seed": int(args.seed),
            "n_initial": int(args.n_initial),
            "n_pool": int(args.n_pool),
            "n_eval": int(args.n_eval),
            "n_rounds": int(args.n_rounds),
            "batch_size": int(args.batch_size),
            "n_reps": int(args.n_reps),
            "n_bins": int(args.n_bins),
            "k_support": int(args.k_support),
            "sigma": float(args.sigma),
            "gamma": float(args.gamma),
            "lambda_reg": float(args.lambda_reg),
            "gamma_acq": float(args.gamma_acq),
            "auc_convention": "mean-over-rounds",
            "regimes_requested": requested_regimes,
            "strong_conflict_fallback_triggered": strong_triggered,
        },
        "regime_descriptions": {
            regime_name: {
                "label": REGIME_LABELS[regime_name],
                "target_description": regime_config(regime_name)["target_description"],
            }
            for regime_name in requested_regimes
        },
        "final_metrics_by_regime": final_rows,
        "auc_metrics_by_regime": auc_rows,
        "allocation_summary": allocation_rows,
        "profile_vs_support_comparisons": regime_comparisons,
        "output_paths": {
            "all_rounds_csv": str((results_dir / "exp7_all_rounds.csv").resolve()),
            "final_metrics_csv": str((results_dir / "exp7_final_metrics_by_regime.csv").resolve()),
            "auc_metrics_csv": str((results_dir / "exp7_auc_metrics_by_regime.csv").resolve()),
            "allocation_summary_csv": str((results_dir / "exp7_allocation_summary.csv").resolve()),
            "bin_round_csv": str((results_dir / "exp7_bin_round_allocation.csv").resolve()),
            "support_limited_table": str((tables_dir / "exp7_support_limited_main_table.tex").resolve()),
            "conflict_table": str((tables_dir / "exp7_conflict_main_table.tex").resolve()),
            "two_regime_table": str((tables_dir / "exp7_two_regime_main_table.tex").resolve()),
            "allocation_table": str((tables_dir / "exp7_allocation_summary_table.tex").resolve()),
            "support_limited_trajectory_figure": str((figures_dir / "exp7_support_limited_trajectories.pdf").resolve()),
            "conflict_trajectory_figure": str((figures_dir / f"exp7_{conflict_regime_name}_trajectories.pdf").resolve()),
            "support_limited_allocation_map": str((figures_dir / "exp7_support_limited_allocation_map.pdf").resolve()),
            "conflict_allocation_map": str((figures_dir / f"exp7_{conflict_regime_name}_allocation_map.pdf").resolve()),
            "bin_round_heatmap": str((figures_dir / "exp7_bin_round_allocation_heatmap.pdf").resolve()),
        },
    }
    (results_dir / "exp7_profile_acquisition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
