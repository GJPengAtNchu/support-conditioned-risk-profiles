from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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

from figure_layout_utils import make_style_map


RULE_DISPLAY_ORDER = (
    "Global-MSE selection",
    "GCV",
    "Constrained weak-MSE",
    "Constrained gap-only",
    "Constrained profile-aware",
)
CONSTRAINED_RULES = (
    "Constrained weak-MSE",
    "Constrained gap-only",
    "Constrained profile-aware",
)
DEFAULT_TAU_GRID = (0.01, 0.03, 0.05, 0.10)


@dataclass(frozen=True)
class Experiment6Setting:
    name: str
    description: str
    weights: Tuple[float, float, float]
    dense_center: Tuple[float, float]
    secondary_center: Tuple[float, float]
    dense_sd: float
    secondary_sd: float
    bump_amplitude: float
    bump_sharpness: float
    bump_center: Tuple[float, float]
    sigma_default: float
    lambda_grid: Tuple[float, ...]
    gamma_grid: Tuple[float, ...]
    default_n_train: int
    default_n_val: int
    default_n_test: int
    default_n_reps: int


SETTING_LIBRARY: Dict[str, Experiment6Setting] = {
    "mild": Experiment6Setting(
        name="mild",
        description="Original mild-regime configuration retained as a sanity check.",
        weights=(0.75, 0.15, 0.10),
        dense_center=(0.25, 0.30),
        secondary_center=(0.60, 0.45),
        dense_sd=0.085,
        secondary_sd=0.095,
        bump_amplitude=0.7,
        bump_sharpness=30.0,
        bump_center=(0.80, 0.78),
        sigma_default=0.08,
        lambda_grid=tuple(float(x) for x in np.logspace(-7, 0, 29)),
        gamma_grid=(5.0, 10.0, 20.0, 40.0),
        default_n_train=600,
        default_n_val=4000,
        default_n_test=8000,
        default_n_reps=50,
    ),
    "stress": Experiment6Setting(
        name="stress",
        description="Stress setting with stronger design heterogeneity and a sharper weak-region bump.",
        weights=(0.85, 0.10, 0.05),
        dense_center=(0.25, 0.30),
        secondary_center=(0.60, 0.45),
        dense_sd=0.08,
        secondary_sd=0.07,
        bump_amplitude=1.2,
        bump_sharpness=60.0,
        bump_center=(0.82, 0.78),
        sigma_default=0.08,
        lambda_grid=tuple(float(x) for x in np.logspace(-8, 0, 33)),
        gamma_grid=(5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
        default_n_train=600,
        default_n_val=4000,
        default_n_test=8000,
        default_n_reps=30,
    ),
    "stress_strong": Experiment6Setting(
        name="stress_strong",
        description="Automatic stronger variant for Experiment 6 if the stress setting remains too mild.",
        weights=(0.90, 0.07, 0.03),
        dense_center=(0.25, 0.30),
        secondary_center=(0.60, 0.45),
        dense_sd=0.06,
        secondary_sd=0.06,
        bump_amplitude=1.4,
        bump_sharpness=80.0,
        bump_center=(0.82, 0.78),
        sigma_default=0.08,
        lambda_grid=tuple(float(x) for x in np.logspace(-8, 0, 33)),
        gamma_grid=(5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0),
        default_n_train=600,
        default_n_val=4000,
        default_n_test=8000,
        default_n_reps=30,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 6: profile-aware model selection with mild/stress configurations."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--setting", type=str, default="stress", choices=tuple(SETTING_LIBRARY.keys()))
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--n-reps", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--k-support", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--tau-grid", type=str, default="0.01,0.03,0.05,0.10")
    parser.add_argument("--eta-w", type=float, default=1.0)
    parser.add_argument("--eta-g", type=float, default=None)
    parser.add_argument("--eta-s", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--skip-auto-strong", action="store_true")
    return parser.parse_args()


def parse_tau_grid(raw: str) -> Tuple[float, ...]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        return DEFAULT_TAU_GRID
    return tuple(sorted(set(values)))


def resolve_outdir(outdir: str) -> Path:
    path = Path(outdir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def fill_defaults(args: argparse.Namespace, setting: Experiment6Setting) -> argparse.Namespace:
    if args.n_train is None:
        args.n_train = setting.default_n_train
    if args.n_val is None:
        args.n_val = setting.default_n_val
    if args.n_test is None:
        args.n_test = setting.default_n_test
    if args.n_reps is None:
        args.n_reps = setting.default_n_reps
    if args.sigma is None:
        args.sigma = setting.sigma_default
    if args.eta_g is None:
        args.eta_g = args.eta
    if args.eta_s is None:
        args.eta_s = args.zeta
    if args.fast:
        args.n_train = min(int(args.n_train), 300)
        args.n_val = min(int(args.n_val), 1500)
        args.n_test = min(int(args.n_test), 2000)
        args.n_reps = min(int(args.n_reps), 8)
    return args


def sample_gaussian_blob(
    n: int,
    center: Tuple[float, float],
    sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2), dtype=np.float64)
    points = rng.normal(loc=np.asarray(center, dtype=np.float64), scale=sd, size=(n, 2))
    return np.clip(points, 0.0, 1.0)


def sample_nonuniform_design(
    n: int,
    rng: np.random.Generator,
    setting: Experiment6Setting,
) -> np.ndarray:
    counts = rng.multinomial(n, np.asarray(setting.weights, dtype=np.float64))
    dense = sample_gaussian_blob(counts[0], setting.dense_center, setting.dense_sd, rng)
    secondary = sample_gaussian_blob(counts[1], setting.secondary_center, setting.secondary_sd, rng)
    background = rng.uniform(0.0, 1.0, size=(counts[2], 2))
    X = np.vstack([dense, secondary, background])
    rng.shuffle(X, axis=0)
    return np.asarray(X, dtype=np.float64)


def f_star(X: np.ndarray, setting: Experiment6Setting) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]
    c1, c2 = setting.bump_center
    return (
        np.sin(2.0 * math.pi * x1) * np.cos(2.0 * math.pi * x2)
        + setting.bump_amplitude * np.exp(
            -setting.bump_sharpness * ((x1 - c1) ** 2 + (x2 - c2) ** 2)
        )
        + 0.25 * x1
    )


def rbf_kernel(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    sqX = np.sum(X * X, axis=1)[:, None]
    sqZ = np.sum(Z * Z, axis=1)[None, :]
    sqdist = np.maximum(sqX + sqZ - 2.0 * X @ Z.T, 0.0)
    return np.exp(-float(gamma) * sqdist)


def fit_krr_basis(X_train: np.ndarray, gamma: float) -> Dict[str, np.ndarray]:
    K = rbf_kernel(X_train, X_train, gamma=gamma)
    if not np.allclose(K, K.T, atol=1e-10):
        raise RuntimeError("Kernel matrix is not numerically symmetric.")
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 0.0)
    return {"K_train": K, "eigvals": eigvals, "eigvecs": eigvecs}


def predict_krr(
    Z_eval: np.ndarray,
    eigvals: np.ndarray,
    y_eig: np.ndarray,
    lambda_reg: float,
    n_train: int,
) -> np.ndarray:
    inv = 1.0 / (eigvals + n_train * float(lambda_reg))
    return np.asarray(Z_eval @ (inv * y_eig), dtype=np.float64)


def compute_support_scores(X_query: np.ndarray, X_train: np.ndarray, k_support: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k_support, algorithm="auto")
    nbrs.fit(X_train)
    distances, _ = nbrs.kneighbors(X_query, return_distance=True)
    return np.asarray(distances[:, -1], dtype=np.float64)


def assign_bins(h_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(h_values, edges[1:-1], right=False).astype(np.int64)


def make_support_bins(h_values: np.ndarray, n_bins: int) -> Dict[str, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(h_values, quantiles)
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, edges.shape[0]):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    bin_ids = assign_bins(h_values, edges)
    centers = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            centers[b] = float(np.median(h_values[mask]))
    return {"edges": edges, "bin_ids": bin_ids, "centers": centers}


def compute_profile_rows(
    errors: np.ndarray,
    h_values: np.ndarray,
    edges: np.ndarray,
) -> List[Dict[str, float]]:
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
    h_values: np.ndarray,
    edges: np.ndarray,
    weak_bins: Sequence[int],
    dense_bins: Sequence[int],
) -> Dict[str, object]:
    bin_ids = assign_bins(h_values, edges)
    weak_mask = np.isin(bin_ids, np.asarray(weak_bins, dtype=np.int64))
    dense_mask = np.isin(bin_ids, np.asarray(dense_bins, dtype=np.int64))
    if not np.any(weak_mask) or not np.any(dense_mask):
        raise RuntimeError("Weak or dense support region is empty.")

    global_mse = float(np.mean(errors))
    weak_mse = float(np.mean(errors[weak_mask]))
    dense_mse = float(np.mean(errors[dense_mask]))
    gap = float(weak_mse / (dense_mse + 1e-12))

    profile_rows = compute_profile_rows(errors=errors, h_values=h_values, edges=edges)
    h_centers = np.asarray([row["h_bin_center"] for row in profile_rows], dtype=np.float64)
    mse_bins = np.asarray([row["mse_bin"] for row in profile_rows], dtype=np.float64)
    slope = slope_loglog(h_centers, np.maximum(mse_bins, 1e-12))
    profile_var = float(np.var(mse_bins))

    return {
        "global_mse": global_mse,
        "weak_mse": weak_mse,
        "dense_mse": dense_mse,
        "gap": gap,
        "slope": slope,
        "pos_slope": slope_pos(slope),
        "profile_var": profile_var,
        "profile_rows": profile_rows,
    }


def compute_gcv_score(y_eig: np.ndarray, eigvals: np.ndarray, lambda_reg: float, n_train: int) -> float:
    inv = 1.0 / (eigvals + n_train * float(lambda_reg))
    smoother_eigs = eigvals * inv
    resid_eig = (1.0 - smoother_eigs) * y_eig
    rss = float(np.sum(resid_eig ** 2))
    tr_s = float(np.sum(smoother_eigs))
    denom = max((1.0 - tr_s / n_train) ** 2, 1e-12)
    return (rss / n_train) / denom


def theta_label(theta: Tuple[float, float]) -> str:
    lambda_reg, gamma = theta
    return f"(lambda={lambda_reg:g}, gamma={gamma:g})"


def latex_scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    if math.isclose(mantissa, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:.2f}\times 10^{{{exponent}}}"


def latex_theta_label(theta: Tuple[float, float]) -> str:
    lambda_reg, gamma = theta
    return rf"$({latex_scientific(float(lambda_reg))}, {float(gamma):g})$"


def slope_pos(value: float) -> float:
    if not math.isfinite(float(value)):
        return float("inf")
    return max(float(value), 0.0)


def select_theta_from_metric(
    theta_grid: Sequence[Tuple[float, float]],
    metric_by_theta: Mapping[Tuple[float, float], float],
    candidate_subset: Sequence[Tuple[float, float]] | None = None,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    candidates = list(candidate_subset) if candidate_subset is not None else list(theta_grid)
    min_val = min(float(metric_by_theta[theta]) for theta in candidates)
    threshold = min_val + tol * max(1.0, abs(min_val))
    nearly_best = [theta for theta in sorted(candidates) if float(metric_by_theta[theta]) <= threshold]
    return tuple(float(x) for x in nearly_best[0])


def mode_theta(
    values: Sequence[Tuple[float, float]],
    theta_grid: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    counts: Dict[Tuple[float, float], int] = {}
    for theta in values:
        counts[theta] = counts.get(theta, 0) + 1
    best = max(theta_grid, key=lambda theta: (counts.get(theta, 0), -list(theta_grid).index(theta)))
    return tuple(float(x) for x in best)


def percentage_delta(profile_value: float, global_value: float, eps: float = 1e-12) -> float:
    return 100.0 * (float(profile_value) - float(global_value)) / (abs(float(global_value)) + eps)


def tau_tag(value: float) -> str:
    return f"{int(round(float(value) * 100.0)):03d}"


def mean_and_se(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=0) / math.sqrt(max(1, arr.size)))
    return mean, se


def build_tau_summary(
    selection_rows: Sequence[Dict[str, object]],
    tau: float,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    rows_tau = [row for row in selection_rows if math.isclose(float(row["tau"]), tau, rel_tol=0.0, abs_tol=1e-15)]
    table_rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, object]] = {}
    for rule in RULE_DISPLAY_ORDER:
        cur = [row for row in rows_tau if row["selection_rule"] == rule]
        theta_values = [
            (float(row["lambda"]), float(row["gamma"]))
            for row in cur
        ]
        theta_mode = mode_theta(theta_values, sorted(set(theta_values)))
        changed_values = [float(row["changed_from_global"]) for row in cur]
        row_out = {
            "selection_rule": rule,
            "global_mse": float(np.mean([float(row["test_global_mse"]) for row in cur])),
            "weak_mse": float(np.mean([float(row["test_weak_mse"]) for row in cur])),
            "dense_mse": float(np.mean([float(row["test_dense_mse"]) for row in cur])),
            "gap": float(np.mean([float(row["test_gap"]) for row in cur])),
            "slope": float(np.nanmean([float(row["test_slope"]) for row in cur])),
            "profile_var": float(np.mean([float(row["test_profile_var"]) for row in cur])),
            "changed_fraction": float(np.mean(changed_values)) if rule != "Global-MSE selection" else float("nan"),
        }
        table_rows.append(row_out)
        summary[rule] = {
            "theta_mode": {
                "lambda": float(theta_mode[0]),
                "gamma": float(theta_mode[1]),
                "label": theta_label(theta_mode),
            },
            "metrics_mean": {
                "global_mse": row_out["global_mse"],
                "weak_mse": row_out["weak_mse"],
                "dense_mse": row_out["dense_mse"],
                "gap": row_out["gap"],
                "slope": row_out["slope"],
                "profile_var": row_out["profile_var"],
            },
            "changed_fraction": row_out["changed_fraction"],
        }
    return table_rows, summary


def build_decision_change_summary(
    selection_rows: Sequence[Dict[str, object]],
    tau_grid: Sequence[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for tau in tau_grid:
        rows_tau = [row for row in selection_rows if math.isclose(float(row["tau"]), tau, rel_tol=0.0, abs_tol=1e-15)]
        for rule in RULE_DISPLAY_ORDER:
            if rule == "Global-MSE selection":
                continue
            cur = [row for row in rows_tau if row["selection_rule"] == rule]
            changed = [row for row in cur if bool(row["changed_from_global"])]
            n_total = len(cur)
            n_changed = len(changed)
            if n_changed == 0:
                rows.append(
                    {
                        "setting": str(cur[0]["setting"]) if cur else "",
                        "tau": float(tau),
                        "selection_rule": rule,
                        "n_splits": int(n_total),
                        "n_changed": int(n_changed),
                        "fraction_changed_from_global": 0.0,
                        "fraction_test_gap_improved": float("nan"),
                        "fraction_test_slope_improved": float("nan"),
                        "fraction_test_weak_mse_improved": float("nan"),
                        "fraction_test_global_mse_within_1pct": float("nan"),
                        "median_delta_gap_percent": float("nan"),
                        "median_delta_slope_percent": float("nan"),
                        "median_delta_weak_mse_percent": float("nan"),
                        "median_delta_global_mse_percent": float("nan"),
                    }
                )
                continue
            delta_gap = np.asarray([float(row["delta_gap_percent"]) for row in changed], dtype=np.float64)
            delta_slope = np.asarray([float(row["delta_slope_percent"]) for row in changed], dtype=np.float64)
            delta_weak = np.asarray([float(row["delta_weak_mse_percent"]) for row in changed], dtype=np.float64)
            delta_global = np.asarray([float(row["delta_global_mse_percent"]) for row in changed], dtype=np.float64)
            rows.append(
                {
                    "setting": str(changed[0]["setting"]),
                    "tau": float(tau),
                    "selection_rule": rule,
                    "n_splits": int(n_total),
                    "n_changed": int(n_changed),
                    "fraction_changed_from_global": float(n_changed / max(1, n_total)),
                    "fraction_test_gap_improved": float(np.mean(delta_gap < 0.0)),
                    "fraction_test_slope_improved": float(np.mean(delta_slope < 0.0)),
                    "fraction_test_weak_mse_improved": float(np.mean(delta_weak < 0.0)),
                    "fraction_test_global_mse_within_1pct": float(np.mean(delta_global <= 1.0)),
                    "median_delta_gap_percent": float(np.median(delta_gap)),
                    "median_delta_slope_percent": float(np.median(delta_slope)),
                    "median_delta_weak_mse_percent": float(np.median(delta_weak)),
                    "median_delta_global_mse_percent": float(np.median(delta_global)),
                }
            )
    return rows


def build_frequency_rows(
    selection_rows: Sequence[Dict[str, object]],
    tau_grid: Sequence[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for tau in tau_grid:
        rows_tau = [row for row in selection_rows if math.isclose(float(row["tau"]), tau, rel_tol=0.0, abs_tol=1e-15)]
        for rule in RULE_DISPLAY_ORDER:
            cur = [row for row in rows_tau if row["selection_rule"] == rule]
            pairs = [(float(row["lambda"]), float(row["gamma"])) for row in cur]
            n = len(pairs)
            counts: Dict[Tuple[float, float], int] = {}
            for theta in pairs:
                counts[theta] = counts.get(theta, 0) + 1
            for theta, count in sorted(counts.items()):
                rows.append(
                    {
                        "setting": str(cur[0]["setting"]),
                        "tau": float(tau),
                        "selection_rule": rule,
                        "lambda": float(theta[0]),
                        "gamma": float(theta[1]),
                        "count": int(count),
                        "fraction": float(count / max(1, n)),
                    }
                )
    return rows


def build_frequency_mode_rows(
    selection_rows: Sequence[Dict[str, object]],
    tau: float,
) -> List[Dict[str, object]]:
    rows_tau = [row for row in selection_rows if math.isclose(float(row["tau"]), tau, rel_tol=0.0, abs_tol=1e-15)]
    out: List[Dict[str, object]] = []
    for rule in RULE_DISPLAY_ORDER:
        cur = [row for row in rows_tau if row["selection_rule"] == rule]
        pairs = [(float(row["lambda"]), float(row["gamma"])) for row in cur]
        theta_mode = mode_theta(pairs, sorted(set(pairs)))
        mode_count = int(sum(theta == theta_mode for theta in pairs))
        n = len(pairs)
        changed = [float(row["changed_from_global"]) for row in cur]
        out.append(
            {
                "selection_rule": rule,
                "mode_lambda": float(theta_mode[0]),
                "mode_gamma": float(theta_mode[1]),
                "mode_count": int(mode_count),
                "mode_fraction": float(mode_count / max(1, n)),
                "changed_fraction": float(np.mean(changed)) if rule != "Global-MSE selection" else float("nan"),
                "n_repetitions": int(n),
            }
        )
    return out


def build_tau_sensitivity_rows(
    selection_rows: Sequence[Dict[str, object]],
    tau_grid: Sequence[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for tau in tau_grid:
        rows_tau, _ = build_tau_summary(selection_rows, tau)
        for row in rows_tau:
            if row["selection_rule"] == "GCV":
                continue
            rows.append(
                {
                    "tau": float(tau),
                    "selection_rule": str(row["selection_rule"]),
                    "global_mse": float(row["global_mse"]),
                    "weak_mse": float(row["weak_mse"]),
                    "dense_mse": float(row["dense_mse"]),
                    "gap": float(row["gap"]),
                    "slope": float(row["slope"]),
                    "profile_var": float(row["profile_var"]),
                    "changed_fraction": float(row["changed_fraction"]) if math.isfinite(float(row["changed_fraction"])) else float("nan"),
                }
            )
    return rows


def make_frontier_figure(
    figures_dir: Path,
    candidate_rows: Sequence[Dict[str, object]],
    selection_rows: Sequence[Dict[str, object]],
    setting: Experiment6Setting,
    tau: float,
    tau_threshold_mean: float,
) -> None:
    rows_val = [row for row in candidate_rows if row["split"] == "val"]
    theta_keys = sorted(set((float(row["lambda"]), float(row["gamma"])) for row in rows_val))
    agg_by_theta: Dict[Tuple[float, float], Dict[str, float]] = {}
    for theta in theta_keys:
        cur = [row for row in rows_val if math.isclose(float(row["lambda"]), theta[0], rel_tol=0.0, abs_tol=1e-15)
               and math.isclose(float(row["gamma"]), theta[1], rel_tol=0.0, abs_tol=1e-15)]
        agg_by_theta[theta] = {
            "global_mse_mean": float(np.mean([float(row["global_mse"]) for row in cur])),
            "weak_mse_mean": float(np.mean([float(row["weak_mse"]) for row in cur])),
            "gap_mean": float(np.mean([float(row["gap"]) for row in cur])),
        }

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    gamma_palette = {gamma: color for gamma, color in zip(setting.gamma_grid, ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6"])}
    gamma_markers = {gamma: marker for gamma, marker in zip(setting.gamma_grid, ["o", "s", "^", "D", "P", "X", "v"])}
    for gamma in setting.gamma_grid:
        rows_gamma = [theta for theta in theta_keys if math.isclose(theta[1], gamma, rel_tol=0.0, abs_tol=1e-15)]
        ax.scatter(
            [agg_by_theta[theta]["global_mse_mean"] for theta in rows_gamma],
            [agg_by_theta[theta]["weak_mse_mean"] for theta in rows_gamma],
            color=gamma_palette[gamma],
            marker=gamma_markers[gamma],
            s=28,
            alpha=0.78,
            label=rf"$\gamma={gamma:g}$",
        )

    ax.axvline(
        float(tau_threshold_mean),
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label=rf"Budget $(1+\tau)\,\mathrm{{MSE}}_{{\mathrm{{val}}}}(\theta_{{\mathrm{{global}}}})$",
    )

    mode_rows = build_frequency_mode_rows(selection_rows, tau=tau)
    marker_map = {
        "Global-MSE selection": ("o", "#111111", (6, -12)),
        "GCV": ("s", "#2f4b7c", (6, 8)),
        "Constrained weak-MSE": ("^", "#e45756", (6, -2)),
        "Constrained gap-only": ("D", "#54a24b", (-12, 6)),
        "Constrained profile-aware": ("P", "#7f3c8d", (-18, -10)),
    }
    for row in mode_rows:
        theta = (float(row["mode_lambda"]), float(row["mode_gamma"]))
        marker, color, offset = marker_map[str(row["selection_rule"])]
        point = agg_by_theta[theta]
        ax.scatter(
            [point["global_mse_mean"]],
            [point["weak_mse_mean"]],
            s=140,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=2.0,
            label=str(row["selection_rule"]),
        )
        ax.annotate(
            str(row["selection_rule"]).replace(" selection", "").replace("Constrained ", "").replace("-MSE", ""),
            (point["global_mse_mean"], point["weak_mse_mean"]),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("Validation global MSE")
    ax.set_ylabel("Validation weak-support MSE")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(figures_dir / f"exp6_{setting.name}_frontier.pdf", bbox_inches="tight")
    if setting.name == "stress":
        fig.savefig(figures_dir / "exp6_pareto_selection.pdf", bbox_inches="tight")
    plt.close(fig)


def make_selected_profiles_figure(
    figures_dir: Path,
    selected_profile_rows: Sequence[Dict[str, object]],
    setting: Experiment6Setting,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    rule_order = ["Global-MSE selection", "Constrained gap-only", "Constrained profile-aware"]
    style_map = make_style_map(rule_order)
    for rule in rule_order:
        rows = [row for row in selected_profile_rows if row["selection_rule"] == rule]
        if not rows:
            continue
        bin_ids = sorted(set(int(row["bin_id"]) for row in rows))
        x = []
        y = []
        yerr = []
        for bin_id in bin_ids:
            cur = [row for row in rows if int(row["bin_id"]) == bin_id]
            x.append(float(np.mean([float(row["h_bin_center"]) for row in cur])))
            vals = np.asarray([float(row["mse_bin"]) for row in cur], dtype=np.float64)
            y.append(float(np.mean(vals)))
            yerr.append(float(np.std(vals, ddof=0) / math.sqrt(max(1, vals.shape[0]))))
        style = style_map[rule]
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.maximum(np.asarray(y, dtype=np.float64), 1e-12)
        err_arr = np.asarray(yerr, dtype=np.float64)
        ax.plot(
            x_arr,
            y_arr,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.1,
            markersize=4.5,
            label=rule,
        )
        ax.fill_between(
            x_arr,
            np.maximum(y_arr - err_arr, 1e-12),
            y_arr + err_arr,
            color=style["color"],
            alpha=0.12,
        )
    ax.set_xlabel("Support bin center $h$")
    ax.set_ylabel("Test clean MSE profile")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.18, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(figures_dir / f"exp6_{setting.name}_selected_profiles.pdf", bbox_inches="tight")
    if setting.name == "stress":
        fig.savefig(figures_dir / "exp6_selected_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def save_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_main_table_tex(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Selection rule & Global MSE $\\times 10^{-3}$ & Weak MSE $\\times 10^{-3}$ & Dense MSE $\\times 10^{-5}$ & Gap & Slope & Profile Var. & Changed from global \\\\",
        "\\midrule",
    ]
    for row in rows:
        changed_text = "--" if row["selection_rule"] == "Global-MSE selection" else f"{100.0 * float(row['changed_fraction']):.1f}\\%"
        global_scaled = 1.0e3 * float(row["global_mse"])
        weak_scaled = 1.0e3 * float(row["weak_mse"])
        dense_scaled = 1.0e5 * float(row["dense_mse"])
        lines.append(
            f"{row['selection_rule']} & "
            f"{global_scaled:.3f} & "
            f"{weak_scaled:.3f} & "
            f"{dense_scaled:.3f} & "
            f"{float(row['gap']):.3f} & "
            f"{float(row['slope']):.3f} & "
            f"{float(row['profile_var']):.6f} & "
            f"{changed_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_main_table_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    save_csv(path, rows)


def save_frequency_table_tex(path: Path, mode_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Selection rule & Most frequent $(\\lambda,\\gamma)$ & Frequency & Changed from global \\\\",
        "\\midrule",
    ]
    for row in mode_rows:
        theta = (float(row["mode_lambda"]), float(row["mode_gamma"]))
        n = int(row["n_repetitions"])
        freq_text = f"{int(row['mode_count'])}/{n} ({100.0 * float(row['mode_fraction']):.1f}\\%)"
        changed_text = "--" if row["selection_rule"] == "Global-MSE selection" else f"{100.0 * float(row['changed_fraction']):.1f}\\%"
        lines.append(
            f"{row['selection_rule']} & "
            f"{latex_theta_label(theta)} & "
            f"{freq_text} & {changed_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_tau_sensitivity_table_tex(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "$\\tau$ & Selection rule & Global MSE & Weak MSE & Gap & Slope & Changed \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["selection_rule"] == "Global-MSE selection":
            changed_text = "--"
        else:
            changed_text = f"{100.0 * float(row['changed_fraction']):.1f}\\%"
        lines.append(
            f"{float(row['tau']):.2f} & "
            f"{row['selection_rule']} & "
            f"{float(row['global_mse']):.4f} & "
            f"{float(row['weak_mse']):.4f} & "
            f"{float(row['gap']):.3f} & "
            f"{float(row['slope']):.3f} & "
            f"{changed_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def build_theta_grid(setting: Experiment6Setting) -> List[Tuple[float, float]]:
    return [(float(lam), float(gamma)) for gamma in setting.gamma_grid for lam in setting.lambda_grid]


def run_setting(
    args: argparse.Namespace,
    outdir: Path,
    setting: Experiment6Setting,
) -> Dict[str, object]:
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    results_dir = outdir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

    theta_grid = build_theta_grid(setting)
    tau_grid = parse_tau_grid(args.tau_grid)
    main_tau = float(args.tau)
    n_region_bins = max(1, int(math.ceil(0.2 * args.n_bins)))
    dense_bins = tuple(range(n_region_bins))
    weak_bins = tuple(range(args.n_bins - n_region_bins, args.n_bins))

    candidate_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    selected_profile_rows: List[Dict[str, object]] = []
    tau_thresholds_main: List[float] = []

    for rep in range(int(args.n_reps)):
        rng = np.random.default_rng(int(args.seed) + 1000 * rep + 17)
        X_train = sample_nonuniform_design(int(args.n_train), rng, setting)
        X_val = sample_nonuniform_design(int(args.n_val), rng, setting)
        X_test = sample_nonuniform_design(int(args.n_test), rng, setting)

        y_train_clean = f_star(X_train, setting)
        y_val_clean = f_star(X_val, setting)
        y_test_clean = f_star(X_test, setting)
        y_train = y_train_clean + float(args.sigma) * rng.normal(size=int(args.n_train))

        h_val = compute_support_scores(X_val, X_train, k_support=int(args.k_support))
        bins_val = make_support_bins(h_val, n_bins=int(args.n_bins))
        edges = np.asarray(bins_val["edges"], dtype=np.float64)
        h_test = compute_support_scores(X_test, X_train, k_support=int(args.k_support))

        val_metrics_by_theta: Dict[Tuple[float, float], Dict[str, object]] = {}
        test_metrics_by_theta: Dict[Tuple[float, float], Dict[str, object]] = {}
        gcv_by_theta: Dict[Tuple[float, float], float] = {}

        for gamma in setting.gamma_grid:
            basis = fit_krr_basis(X_train, gamma=float(gamma))
            eigvals = np.asarray(basis["eigvals"], dtype=np.float64)
            eigvecs = np.asarray(basis["eigvecs"], dtype=np.float64)
            y_eig = eigvecs.T @ y_train

            K_val_train = rbf_kernel(X_val, X_train, gamma=float(gamma))
            K_test_train = rbf_kernel(X_test, X_train, gamma=float(gamma))
            Z_val = K_val_train @ eigvecs
            Z_test = K_test_train @ eigvecs

            for lambda_reg in setting.lambda_grid:
                theta = (float(lambda_reg), float(gamma))
                pred_val = predict_krr(Z_eval=Z_val, eigvals=eigvals, y_eig=y_eig, lambda_reg=float(lambda_reg), n_train=int(args.n_train))
                pred_test = predict_krr(Z_eval=Z_test, eigvals=eigvals, y_eig=y_eig, lambda_reg=float(lambda_reg), n_train=int(args.n_train))
                err_val = (pred_val - y_val_clean) ** 2
                err_test = (pred_test - y_test_clean) ** 2
                val_metrics = compute_profile_metrics(err_val, h_val, edges, weak_bins=weak_bins, dense_bins=dense_bins)
                test_metrics = compute_profile_metrics(err_test, h_test, edges, weak_bins=weak_bins, dense_bins=dense_bins)
                val_metrics_by_theta[theta] = val_metrics
                test_metrics_by_theta[theta] = test_metrics
                gcv_by_theta[theta] = compute_gcv_score(y_eig=y_eig, eigvals=eigvals, lambda_reg=float(lambda_reg), n_train=int(args.n_train))
                candidate_rows.extend(
                    [
                        {
                            "setting": setting.name,
                            "repetition": int(rep),
                            "split": "val",
                            "lambda": float(lambda_reg),
                            "gamma": float(gamma),
                            "theta_label": theta_label(theta),
                            "global_mse": float(val_metrics["global_mse"]),
                            "weak_mse": float(val_metrics["weak_mse"]),
                            "dense_mse": float(val_metrics["dense_mse"]),
                            "gap": float(val_metrics["gap"]),
                            "slope": float(val_metrics["slope"]),
                            "pos_slope": float(val_metrics["pos_slope"]),
                            "profile_var": float(val_metrics["profile_var"]),
                        },
                        {
                            "setting": setting.name,
                            "repetition": int(rep),
                            "split": "test",
                            "lambda": float(lambda_reg),
                            "gamma": float(gamma),
                            "theta_label": theta_label(theta),
                            "global_mse": float(test_metrics["global_mse"]),
                            "weak_mse": float(test_metrics["weak_mse"]),
                            "dense_mse": float(test_metrics["dense_mse"]),
                            "gap": float(test_metrics["gap"]),
                            "slope": float(test_metrics["slope"]),
                            "pos_slope": float(test_metrics["pos_slope"]),
                            "profile_var": float(test_metrics["profile_var"]),
                        },
                    ]
                )

        theta_global = select_theta_from_metric(theta_grid, {theta: float(val_metrics_by_theta[theta]["global_mse"]) for theta in theta_grid})
        theta_gcv = select_theta_from_metric(theta_grid, gcv_by_theta)
        global_val = float(val_metrics_by_theta[theta_global]["global_mse"])
        weak_metric_map = {theta: float(val_metrics_by_theta[theta]["weak_mse"]) for theta in theta_grid}
        gap_metric_map = {theta: float(val_metrics_by_theta[theta]["gap"]) for theta in theta_grid}
        weak_ref = max(float(val_metrics_by_theta[theta_global]["weak_mse"]), 1e-12)
        gap_ref = max(float(val_metrics_by_theta[theta_global]["gap"]), 1e-12)
        slope_ref = float(val_metrics_by_theta[theta_global]["pos_slope"])
        eps = 1e-12
        profile_score_map = {
            theta: (
                float(args.eta_w) * float(val_metrics_by_theta[theta]["weak_mse"]) / (weak_ref + eps)
                + float(args.eta_g) * float(val_metrics_by_theta[theta]["gap"]) / (gap_ref + eps)
                + float(args.eta_s) * float(val_metrics_by_theta[theta]["pos_slope"]) / (slope_ref + eps)
            )
            for theta in theta_grid
        }
        theta_weak_unconstrained = select_theta_from_metric(theta_grid, weak_metric_map)
        theta_gap_unconstrained = select_theta_from_metric(theta_grid, gap_metric_map)
        theta_profile_unconstrained = select_theta_from_metric(theta_grid, profile_score_map)

        for tau in tau_grid:
            tau_threshold = (1.0 + float(tau)) * global_val
            admissible = [
                theta for theta in theta_grid
                if float(val_metrics_by_theta[theta]["global_mse"]) <= tau_threshold + 1e-15
            ]
            if theta_global not in admissible:
                raise RuntimeError("Global baseline theta must belong to the admissible set.")
            if math.isclose(float(tau), main_tau, rel_tol=0.0, abs_tol=1e-15):
                tau_thresholds_main.append(float(tau_threshold))

            theta_weak = select_theta_from_metric(theta_grid, weak_metric_map, candidate_subset=admissible)
            theta_gap = select_theta_from_metric(theta_grid, gap_metric_map, candidate_subset=admissible)
            theta_profile = select_theta_from_metric(theta_grid, profile_score_map, candidate_subset=admissible)

            selected = {
                "Global-MSE selection": (theta_global, False),
                "GCV": (theta_gcv, False),
                "Constrained weak-MSE": (theta_weak, theta_weak != theta_weak_unconstrained),
                "Constrained gap-only": (theta_gap, theta_gap != theta_gap_unconstrained),
                "Constrained profile-aware": (theta_profile, theta_profile != theta_profile_unconstrained),
            }

            for rule_name, (theta_sel, constraint_active) in selected.items():
                lambda_sel, gamma_sel = theta_sel
                val_metrics = val_metrics_by_theta[theta_sel]
                test_metrics = test_metrics_by_theta[theta_sel]
                global_test = test_metrics_by_theta[theta_global]
                changed_from_global = theta_sel != theta_global if rule_name != "Global-MSE selection" else False
                row = {
                    "setting": setting.name,
                    "tau": float(tau),
                    "repetition": int(rep),
                    "selection_rule": rule_name,
                    "lambda": float(lambda_sel),
                    "gamma": float(gamma_sel),
                    "theta_label": theta_label(theta_sel),
                    "theta_global_label": theta_label(theta_global),
                    "changed_from_global": bool(changed_from_global),
                    "constraint_active": bool(constraint_active) if rule_name in CONSTRAINED_RULES else False,
                    "candidate_count": int(len(admissible)),
                    "validation_global_mse": float(val_metrics["global_mse"]),
                    "validation_weak_mse": float(val_metrics["weak_mse"]),
                    "validation_dense_mse": float(val_metrics["dense_mse"]),
                    "validation_gap": float(val_metrics["gap"]),
                    "validation_slope": float(val_metrics["slope"]),
                    "validation_pos_slope": float(val_metrics["pos_slope"]),
                    "validation_profile_var": float(val_metrics["profile_var"]),
                    "validation_profile_score": float(profile_score_map[theta_sel]),
                    "validation_global_budget": float(tau_threshold),
                    "test_global_mse": float(test_metrics["global_mse"]),
                    "test_weak_mse": float(test_metrics["weak_mse"]),
                    "test_dense_mse": float(test_metrics["dense_mse"]),
                    "test_gap": float(test_metrics["gap"]),
                    "test_slope": float(test_metrics["slope"]),
                    "test_pos_slope": float(test_metrics["pos_slope"]),
                    "test_profile_var": float(test_metrics["profile_var"]),
                    "delta_global_mse_percent": percentage_delta(float(test_metrics["global_mse"]), float(global_test["global_mse"])),
                    "delta_weak_mse_percent": percentage_delta(float(test_metrics["weak_mse"]), float(global_test["weak_mse"])),
                    "delta_gap_percent": percentage_delta(float(test_metrics["gap"]), float(global_test["gap"])),
                    "delta_slope_percent": percentage_delta(float(test_metrics["pos_slope"]), float(global_test["pos_slope"])),
                    "delta_profile_var_percent": percentage_delta(float(test_metrics["profile_var"]), float(global_test["profile_var"])),
                }
                selection_rows.append(row)
                if math.isclose(float(tau), main_tau, rel_tol=0.0, abs_tol=1e-15):
                    for prow in test_metrics["profile_rows"]:
                        selected_profile_rows.append(
                            {
                                "setting": setting.name,
                                "tau": float(tau),
                                "repetition": int(rep),
                                "selection_rule": rule_name,
                                "lambda": float(lambda_sel),
                                "gamma": float(gamma_sel),
                                "theta_label": theta_label(theta_sel),
                                "bin_id": int(prow["bin_id"]),
                                "h_bin_center": float(prow["h_bin_center"]),
                                "mse_bin": float(prow["mse_bin"]),
                                "n_bin": int(prow["n_bin"]),
                            }
                        )

    main_table_rows, tau_summary = build_tau_summary(selection_rows, tau=main_tau)
    frequency_rows = build_frequency_rows(selection_rows, tau_grid=tau_grid)
    frequency_mode_rows = build_frequency_mode_rows(selection_rows, tau=main_tau)
    tau_sensitivity_rows = build_tau_sensitivity_rows(selection_rows, tau_grid=tau_grid)
    decision_change_rows = build_decision_change_summary(selection_rows, tau_grid=tau_grid)

    make_frontier_figure(
        figures_dir=figures_dir,
        candidate_rows=candidate_rows,
        selection_rows=selection_rows,
        setting=setting,
        tau=main_tau,
        tau_threshold_mean=float(np.mean(tau_thresholds_main)),
    )
    make_selected_profiles_figure(
        figures_dir=figures_dir,
        selected_profile_rows=selected_profile_rows,
        setting=setting,
    )

    save_csv(results_dir / f"exp6_{setting.name}_all_splits.csv", selection_rows)
    save_csv(
        results_dir / f"exp6_{setting.name}_selected_metrics_tau{tau_tag(main_tau)}.csv",
        [row for row in selection_rows if math.isclose(float(row["tau"]), main_tau, rel_tol=0.0, abs_tol=1e-15)],
    )
    save_csv(results_dir / f"exp6_{setting.name}_selected_frequency.csv", frequency_rows)
    save_csv(results_dir / f"exp6_{setting.name}_tau_sensitivity.csv", tau_sensitivity_rows)
    save_csv(results_dir / f"exp6_{setting.name}_decision_change_summary.csv", decision_change_rows)
    save_csv(results_dir / f"exp6_{setting.name}_candidate_metrics.csv", candidate_rows)
    save_csv(results_dir / f"exp6_{setting.name}_selected_profiles.csv", selected_profile_rows)

    save_main_table_csv(tables_dir / f"exp6_{setting.name}_main_table.csv", main_table_rows)
    save_main_table_tex(tables_dir / f"exp6_{setting.name}_main_table.tex", main_table_rows)
    save_csv(tables_dir / f"exp6_{setting.name}_selected_frequency.csv", frequency_rows)
    save_frequency_table_tex(tables_dir / f"exp6_{setting.name}_frequency_table.tex", frequency_mode_rows)
    save_tau_sensitivity_table_tex(tables_dir / f"exp6_{setting.name}_tau_sensitivity_table.tex", tau_sensitivity_rows)

    if setting.name == "stress":
        save_main_table_csv(tables_dir / "exp6_profile_selection.csv", main_table_rows)
        save_main_table_tex(tables_dir / "exp6_profile_selection.tex", main_table_rows)
        save_csv(tables_dir / "exp6_selected_hyperparameter_frequency.csv", frequency_mode_rows)
        save_frequency_table_tex(tables_dir / "exp6_selected_hyperparameter_frequency.tex", frequency_mode_rows)
        save_csv(results_dir / "exp6_candidate_metrics.csv", candidate_rows)
        save_csv(results_dir / "exp6_selection_results.csv", selection_rows)
        save_csv(results_dir / "exp6_profiles_selected.csv", selected_profile_rows)
        save_csv(results_dir / "exp6_selected_hyperparameters_long.csv", [
            {
                "setting": row["setting"],
                "tau": row["tau"],
                "repetition": row["repetition"],
                "selection_rule": row["selection_rule"],
                "selected_lambda": row["lambda"],
                "selected_gamma": row["gamma"],
                "selected_pair": row["theta_label"],
            }
            for row in selection_rows
            if math.isclose(float(row["tau"]), main_tau, rel_tol=0.0, abs_tol=1e-15)
        ])

    summary = {
        "experiment": "exp6_profile_selection",
        "setting": setting.name,
        "setting_config": asdict(setting),
        "parameters": {
            "seed": int(args.seed),
            "n_train": int(args.n_train),
            "n_val": int(args.n_val),
            "n_test": int(args.n_test),
            "n_reps": int(args.n_reps),
            "n_bins": int(args.n_bins),
            "k_support": int(args.k_support),
            "sigma": float(args.sigma),
            "tau_main": float(main_tau),
            "tau_grid": [float(x) for x in tau_grid],
            "eta_w": float(args.eta_w),
            "eta_g": float(args.eta_g),
            "eta_s": float(args.eta_s),
            "weak_bins": [int(x + 1) for x in weak_bins],
            "dense_bins": [int(x + 1) for x in dense_bins],
            "profile_variance_definition": "Unweighted variance across binwise clean MSE values.",
            "tie_breaking": "Choose the lexicographically smallest (lambda, gamma) among minimizers within floating-point tolerance.",
            "candidate_family": "KRR with theta=(lambda, gamma)",
            "global_risk_budget_mean_tau_main": float(np.mean(tau_thresholds_main)),
        },
        "main_tau_summary": {
            "tau": float(main_tau),
            "table_rows": main_table_rows,
            "selection_summary": tau_summary,
            "frequency_mode_rows": frequency_mode_rows,
        },
        "tau_sensitivity_rows": tau_sensitivity_rows,
        "decision_change_summary_rows": decision_change_rows,
        "output_paths": {
            "frontier_figure": str((figures_dir / f"exp6_{setting.name}_frontier.pdf").resolve()),
            "selected_profiles_figure": str((figures_dir / f"exp6_{setting.name}_selected_profiles.pdf").resolve()),
            "main_table_tex": str((tables_dir / f"exp6_{setting.name}_main_table.tex").resolve()),
            "frequency_table_tex": str((tables_dir / f"exp6_{setting.name}_frequency_table.tex").resolve()),
            "tau_sensitivity_table_tex": str((tables_dir / f"exp6_{setting.name}_tau_sensitivity_table.tex").resolve()),
            "all_splits_csv": str((results_dir / f"exp6_{setting.name}_all_splits.csv").resolve()),
            "selected_metrics_tau_main_csv": str((results_dir / f"exp6_{setting.name}_selected_metrics_tau{tau_tag(main_tau)}.csv").resolve()),
            "selected_frequency_csv": str((results_dir / f"exp6_{setting.name}_selected_frequency.csv").resolve()),
            "tau_sensitivity_csv": str((results_dir / f"exp6_{setting.name}_tau_sensitivity.csv").resolve()),
            "decision_change_summary_csv": str((results_dir / f"exp6_{setting.name}_decision_change_summary.csv").resolve()),
        },
    }
    return summary


def save_summary_json(path: Path, summary: Dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    raw_args = parse_args()
    outdir = resolve_outdir(raw_args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    primary_setting = SETTING_LIBRARY[raw_args.setting]
    args = fill_defaults(raw_args, primary_setting)
    main_summary = run_setting(args=args, outdir=outdir, setting=primary_setting)

    summaries: Dict[str, Dict[str, object]] = {primary_setting.name: main_summary}
    manuscript_setting = primary_setting.name
    auto_strong_triggered = False

    if (
        primary_setting.name == "stress"
        and not bool(raw_args.fast)
        and not bool(raw_args.skip_auto_strong)
    ):
        profile_row = next(
            row for row in main_summary["main_tau_summary"]["table_rows"]
            if row["selection_rule"] == "Constrained profile-aware"
        )
        global_row = next(
            row for row in main_summary["main_tau_summary"]["table_rows"]
            if row["selection_rule"] == "Global-MSE selection"
        )
        gap_improvement = percentage_delta(float(profile_row["gap"]), float(global_row["gap"]))
        slope_improvement = percentage_delta(float(profile_row["slope"]), float(global_row["slope"]))
        if gap_improvement > -10.0 and slope_improvement > -10.0:
            auto_strong_triggered = True
            strong_setting = SETTING_LIBRARY["stress_strong"]
            strong_args = argparse.Namespace(**vars(raw_args))
            strong_args.setting = "stress_strong"
            strong_args = fill_defaults(strong_args, strong_setting)
            strong_summary = run_setting(args=strong_args, outdir=outdir, setting=strong_setting)
            summaries[strong_setting.name] = strong_summary
            manuscript_setting = strong_setting.name

    combined_summary = {
        "experiment": "exp6_profile_selection",
        "requested_setting": primary_setting.name,
        "manuscript_setting": manuscript_setting,
        "auto_strong_triggered": bool(auto_strong_triggered),
        "setting_summaries": summaries,
    }
    save_summary_json(outdir / "results" / "exp6_profile_selection_summary.json", combined_summary)


if __name__ == "__main__":
    main()
