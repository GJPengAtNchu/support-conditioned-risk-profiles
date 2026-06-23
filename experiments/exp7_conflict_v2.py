from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

from figure_layout_utils import make_style_map
from exp7_acquisition import (
    RULE_DISPLAY_ORDER,
    add_noisy_labels,
    assign_bins,
    compute_profile_metrics,
    compute_support_scores,
    fit_krr,
    make_support_bins,
    make_support_index,
    posterior_variance_krr,
    predict_krr,
    robust_support_contrast,
    sample_mixture_design,
    sample_uniform_points,
    select_from_ranked_bins_diverse,
    select_random,
    select_top_diverse_scores,
)


RULE_TABLE_LABELS = {
    "Random acquisition": "Random acquisition",
    "Support-only acquisition": "Support-only acquisition",
    "Error-only acquisition": "Error-only acquisition",
    "Posterior-variance acquisition": "Posterior-variance acquisition",
    "Profile-aware acquisition": "Profile-aware acquisition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 7 conflict-v2: stronger three-region acquisition conflict."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-initial", type=int, default=300)
    parser.add_argument("--n-pool", type=int, default=5000)
    parser.add_argument("--n-eval", type=int, default=8000)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--n-reps", type=int, default=30)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--k-support", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-3)
    parser.add_argument("--gamma-acq", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def apply_fast_mode(args: argparse.Namespace) -> argparse.Namespace:
    if args.fast:
        args.n_initial = 180
        args.n_pool = 2000
        args.n_eval = 3000
        args.n_rounds = 5
        args.batch_size = 20
        args.n_reps = 5
    return args


def resolve_outdir(outdir: str) -> Path:
    path = Path(outdir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def conflict_v2_config(variant: str = "base") -> Dict[str, object]:
    if variant == "strong":
        return {
            "name": "exp7_conflict_v2_strong",
            "train_weights": [0.82, 0.10, 0.06, 0.02],
            "train_components": [
                {"kind": "gaussian", "mean": [0.25, 0.30], "sd": [0.055, 0.055]},
                {"kind": "gaussian", "mean": [0.58, 0.43], "sd": [0.050, 0.050]},
                {"kind": "gaussian", "mean": [0.66, 0.64], "sd": [0.040, 0.040]},
                {"kind": "uniform"},
            ],
            "eval_weights": [0.72, 0.08, 0.15, 0.05],
            "eval_components": [
                {"kind": "gaussian", "mean": [0.25, 0.30], "sd": [0.065, 0.065]},
                {"kind": "gaussian", "mean": [0.58, 0.43], "sd": [0.055, 0.055]},
                {"kind": "gaussian", "mean": [0.70, 0.70], "sd": [0.045, 0.045]},
                {"kind": "uniform"},
            ],
            "bump_center": [0.70, 0.70],
            "bump_amplitude": 1.6,
            "bump_sharpness": 85.0,
            "dense_center": [0.25, 0.30],
            "dense_radius": 0.18,
            "relevant_radius": 0.14,
            "distractor_region": {"x1_min": 0.84, "x2_min": 0.84},
        }
    return {
        "name": "exp7_conflict_v2",
        "train_weights": [0.80, 0.11, 0.06, 0.03],
        "train_components": [
            {"kind": "gaussian", "mean": [0.25, 0.30], "sd": [0.058, 0.058]},
            {"kind": "gaussian", "mean": [0.58, 0.43], "sd": [0.055, 0.055]},
            {"kind": "gaussian", "mean": [0.66, 0.64], "sd": [0.043, 0.043]},
            {"kind": "uniform"},
        ],
        "eval_weights": [0.70, 0.10, 0.15, 0.05],
        "eval_components": [
            {"kind": "gaussian", "mean": [0.25, 0.30], "sd": [0.068, 0.068]},
            {"kind": "gaussian", "mean": [0.58, 0.43], "sd": [0.058, 0.058]},
            {"kind": "gaussian", "mean": [0.72, 0.72], "sd": [0.048, 0.048]},
            {"kind": "uniform"},
        ],
        "bump_center": [0.72, 0.72],
        "bump_amplitude": 1.35,
        "bump_sharpness": 72.0,
        "dense_center": [0.25, 0.30],
        "dense_radius": 0.18,
        "relevant_radius": 0.14,
        "distractor_region": {"x1_min": 0.84, "x2_min": 0.84},
    }


def mixture_cfg(weights: Sequence[float], components: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {"weights": list(weights), "components": list(components)}


def f_conflict_v2(X: np.ndarray, cfg: Dict[str, object]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]
    dense_center = np.asarray(cfg["dense_center"], dtype=np.float64)
    bump_center = np.asarray(cfg["bump_center"], dtype=np.float64)
    dense_window = np.exp(
        -9.0 * ((x1 - dense_center[0]) ** 2 + 0.85 * (x2 - dense_center[1]) ** 2)
    )
    dense_wave = 0.18 * np.sin(2.0 * math.pi * x1) * np.cos(2.0 * math.pi * x2) * dense_window
    smooth_trend = 0.10 * x1 + 0.04 * x2
    high_error_bump = float(cfg["bump_amplitude"]) * np.exp(
        -float(cfg["bump_sharpness"])
        * ((x1 - bump_center[0]) ** 2 + (x2 - bump_center[1]) ** 2)
    )
    return dense_wave + smooth_trend + high_error_bump


def in_circle_region(X: np.ndarray, center: Sequence[float], radius: float) -> np.ndarray:
    center_arr = np.asarray(center, dtype=np.float64)
    sqdist = np.sum((np.asarray(X, dtype=np.float64) - center_arr[None, :]) ** 2, axis=1)
    return sqdist <= float(radius) ** 2


def in_distractor_region(X: np.ndarray, cfg: Dict[str, object]) -> np.ndarray:
    region = cfg["distractor_region"]
    return (X[:, 0] >= float(region["x1_min"])) & (X[:, 1] >= float(region["x2_min"]))


def region_category_masks(X: np.ndarray, cfg: Dict[str, object]) -> Dict[str, np.ndarray]:
    relevant = in_circle_region(X, cfg["bump_center"], float(cfg["relevant_radius"]))
    distractor = in_distractor_region(X, cfg)
    dense = in_circle_region(X, cfg["dense_center"], float(cfg["dense_radius"]))
    other = ~(relevant | distractor | dense)
    dense = dense & ~(relevant | distractor)
    distractor = distractor & ~relevant
    return {
        "relevant": relevant,
        "distractor": distractor,
        "dense": dense,
        "other": other,
    }


def compute_profile_row_maps(profile_rows: Sequence[Dict[str, float]]) -> tuple[Dict[int, float], Dict[int, float]]:
    mse_map: Dict[int, float] = {}
    h_map: Dict[int, float] = {}
    for row in profile_rows:
        bin_zero = int(row["bin_id"]) - 1
        mse_map[bin_zero] = float(row["mse_bin"])
        h_map[bin_zero] = float(row["h_bin_center"])
    return mse_map, h_map


def run_one_rule(
    rule: str,
    rep: int,
    rep_seed: int,
    rng: np.random.Generator,
    cfg: Dict[str, object],
    X_train_init: np.ndarray,
    y_train_init: np.ndarray,
    X_pool_init: np.ndarray,
    y_pool_clean_init: np.ndarray,
    y_pool_noisy_init: np.ndarray,
    X_eval: np.ndarray,
    y_eval_clean: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, List[Dict[str, object]]]:
    X_train = np.asarray(X_train_init, dtype=np.float64).copy()
    y_train = np.asarray(y_train_init, dtype=np.float64).copy()
    X_pool = np.asarray(X_pool_init, dtype=np.float64).copy()
    y_pool_clean = np.asarray(y_pool_clean_init, dtype=np.float64).copy()
    y_pool_noisy = np.asarray(y_pool_noisy_init, dtype=np.float64).copy()

    round_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

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

        round_rows.append(
            {
                "variant": cfg["name"],
                "repetition": int(rep),
                "rep_seed": int(rep_seed),
                "rule": rule,
                "round": int(round_idx),
                "global_mse": float(metrics["global_mse"]),
                "weak_mse": float(metrics["weak_mse"]),
                "dense_mse": float(metrics["dense_mse"]),
                "gap": float(metrics["gap"]),
                "slope": float(metrics["slope"]),
                "positive_slope": float(metrics["pos_slope"]),
                "profile_var": float(metrics["profile_var"]),
                "support_contrast": float(robust_support_contrast(h_eval)),
                "n_train": int(X_train.shape[0]),
                "n_pool_remaining": int(X_pool.shape[0]),
            }
        )

        if round_idx == args.n_rounds:
            break

        h_pool = compute_support_scores(X_query=X_pool, nbrs=nbrs)
        pool_bin_ids = assign_bins(h_pool, edges)
        pred_pool = predict_krr(model=model, X_eval=X_pool)
        oracle_sq_error_pool = (pred_pool - y_pool_clean) ** 2
        mse_map, h_map = compute_profile_row_maps(metrics["profile_rows"])
        predicted_sq_error_proxy = np.asarray([mse_map[int(b)] for b in pool_bin_ids], dtype=np.float64)

        if rule == "Random acquisition":
            pool_score = np.full(X_pool.shape[0], np.nan, dtype=np.float64)
            selected_idx = select_random(X_pool.shape[0], int(args.batch_size), rng)
        elif rule == "Support-only acquisition":
            pool_score = h_pool.copy()
            selected_idx = select_top_diverse_scores(X_pool, X_train, pool_score, int(args.batch_size))
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
            pool_score = posterior_variance_krr(model=model, X_query=X_pool)
            selected_idx = select_top_diverse_scores(X_pool, X_train, pool_score, int(args.batch_size))
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
            raise ValueError(f"Unknown rule: {rule}")

        masks = region_category_masks(X_pool, cfg)
        n_bins_eff = int(np.asarray(bins["n_bins"], dtype=np.int64)[0])
        weak_region_bin_start = n_bins_eff - n_region_bins
        for idx in selected_idx:
            selected_rows.append(
                {
                    "variant": cfg["name"],
                    "repetition": int(rep),
                    "rep_seed": int(rep_seed),
                    "rule": rule,
                    "round": int(round_idx + 1),
                    "x1": float(X_pool[idx, 0]),
                    "x2": float(X_pool[idx, 1]),
                    "support_score": float(h_pool[idx]),
                    "selected_bin": int(pool_bin_ids[idx] + 1),
                    "is_weak_bin": bool(int(pool_bin_ids[idx]) >= weak_region_bin_start),
                    "oracle_sq_error": float(oracle_sq_error_pool[idx]),
                    "predicted_sq_error_proxy": float(predicted_sq_error_proxy[idx]),
                    "score_used": float(pool_score[idx]),
                    "in_relevant_region": bool(masks["relevant"][idx]),
                    "in_distractor_region": bool(masks["distractor"][idx]),
                    "in_dense_region": bool(masks["dense"][idx]),
                    "in_other_region": bool(masks["other"][idx]),
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

    return {"round_rows": round_rows, "selected_rows": selected_rows}


def aggregate_final_and_auc(round_rows: Sequence[Dict[str, object]], n_rounds: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rule in RULE_DISPLAY_ORDER:
        final_cur = [row for row in round_rows if row["rule"] == rule and int(row["round"]) == int(n_rounds)]
        if not final_cur:
            continue
        reps = sorted({int(row["repetition"]) for row in final_cur})
        weak_auc_vals = []
        gap_auc_vals = []
        slope_auc_vals = []
        for rep in reps:
            rep_rows = sorted(
                [row for row in round_rows if row["rule"] == rule and int(row["repetition"]) == rep],
                key=lambda row: int(row["round"]),
            )
            weak_auc_vals.append(float(np.mean([float(row["weak_mse"]) for row in rep_rows])))
            gap_auc_vals.append(float(np.mean([float(row["gap"]) for row in rep_rows])))
            slope_auc_vals.append(float(np.mean([float(row["positive_slope"]) for row in rep_rows])))
        rows.append(
            {
                "variant": str(final_cur[0]["variant"]),
                "rule": rule,
                "global_mse": float(np.mean([float(row["global_mse"]) for row in final_cur])),
                "weak_mse": float(np.mean([float(row["weak_mse"]) for row in final_cur])),
                "dense_mse": float(np.mean([float(row["dense_mse"]) for row in final_cur])),
                "gap": float(np.mean([float(row["gap"]) for row in final_cur])),
                "slope": float(np.mean([float(row["slope"]) for row in final_cur])),
                "positive_slope": float(np.mean([float(row["positive_slope"]) for row in final_cur])),
                "profile_var": float(np.mean([float(row["profile_var"]) for row in final_cur])),
                "weak_auc": float(np.mean(weak_auc_vals)),
                "gap_auc": float(np.mean(gap_auc_vals)),
                "slope_auc": float(np.mean(slope_auc_vals)),
                "n_reps": int(len(reps)),
            }
        )
    return rows


def summarize_allocation(selected_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rule in RULE_DISPLAY_ORDER:
        cur = [row for row in selected_rows if row["rule"] == rule]
        if not cur:
            continue
        support_vals = np.asarray([float(row["support_score"]) for row in cur], dtype=np.float64)
        oracle_vals = np.asarray([float(row["oracle_sq_error"]) for row in cur], dtype=np.float64)
        rows.append(
            {
                "variant": str(cur[0]["variant"]),
                "rule": rule,
                "n_selected": int(len(cur)),
                "fraction_relevant_region": float(np.mean([bool(row["in_relevant_region"]) for row in cur])),
                "fraction_distractor_region": float(np.mean([bool(row["in_distractor_region"]) for row in cur])),
                "fraction_dense_region": float(np.mean([bool(row["in_dense_region"]) for row in cur])),
                "fraction_other_region": float(np.mean([bool(row["in_other_region"]) for row in cur])),
                "fraction_weak_bin": float(np.mean([bool(row["is_weak_bin"]) for row in cur])),
                "median_support_radius": float(np.median(support_vals)),
                "median_oracle_sq_error": float(np.median(oracle_vals)),
            }
        )
    return rows


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_final_metrics_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Acquisition rule & Global MSE & Weak MSE & Dense MSE & Gap & Slope & Profile Var. \\\\",
        "\\midrule",
    ]
    for rule in RULE_DISPLAY_ORDER:
        row = next(r for r in rows if r["rule"] == rule)
        lines.append(
            f"{RULE_TABLE_LABELS[rule]} & "
            f"{float(row['global_mse']):.6f} & "
            f"{float(row['weak_mse']):.6f} & "
            f"{float(row['dense_mse']):.6f} & "
            f"{float(row['gap']):.3f} & "
            f"{float(row['positive_slope']):.3f} & "
            f"{float(row['profile_var']):.6f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_auc_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Acquisition rule & Weak AUC & Gap AUC & Slope AUC \\\\",
        "\\midrule",
    ]
    for rule in RULE_DISPLAY_ORDER:
        row = next(r for r in rows if r["rule"] == rule)
        lines.append(
            f"{RULE_TABLE_LABELS[rule]} & "
            f"{float(row['weak_auc']):.6f} & "
            f"{float(row['gap_auc']):.3f} & "
            f"{float(row['slope_auc']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_allocation_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Acquisition rule & Relevant frac. & Distractor frac. & Dense frac. & Other frac. & Median support & Median oracle error \\\\",
        "\\midrule",
    ]
    for rule in RULE_DISPLAY_ORDER:
        row = next(r for r in rows if r["rule"] == rule)
        lines.append(
            f"{RULE_TABLE_LABELS[rule]} & "
            f"{float(row['fraction_relevant_region']):.3f} & "
            f"{float(row['fraction_distractor_region']):.3f} & "
            f"{float(row['fraction_dense_region']):.3f} & "
            f"{float(row['fraction_other_region']):.3f} & "
            f"{float(row['median_support_radius']):.3f} & "
            f"{float(row['median_oracle_sq_error']):.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def metric_arrays(round_rows: Sequence[Dict[str, object]], rule: str, metric: str, n_rounds: int) -> tuple[np.ndarray, np.ndarray]:
    means = []
    ses = []
    for round_idx in range(n_rounds + 1):
        vals = np.asarray(
            [float(row[metric]) for row in round_rows if row["rule"] == rule and int(row["round"]) == round_idx],
            dtype=np.float64,
        )
        means.append(float(np.mean(vals)))
        ses.append(float(np.std(vals, ddof=0) / math.sqrt(max(1, vals.shape[0]))))
    return np.asarray(means), np.asarray(ses)


def save_metric_trajectory_png(
    path: Path,
    round_rows: Sequence[Dict[str, object]],
    metric: str,
    ylabel: str,
    n_rounds: int,
) -> None:
    style_map = make_style_map(RULE_DISPLAY_ORDER)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    x = np.arange(n_rounds + 1)
    for rule in RULE_DISPLAY_ORDER:
        mean_arr, se_arr = metric_arrays(round_rows, rule, metric, n_rounds)
        style = style_map[rule]
        ax.plot(
            x,
            mean_arr,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=4.8,
            label=RULE_TABLE_LABELS[rule],
        )
        ax.fill_between(x, np.maximum(mean_arr - se_arr, 0.0), mean_arr + se_arr, color=style["color"], alpha=0.15)
    ax.set_xlabel("Acquisition round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_allocation_map_png(
    path: Path,
    selected_rows: Sequence[Dict[str, object]],
    X_train_init: np.ndarray,
    cfg: Dict[str, object],
    n_rounds: int,
) -> None:
    rep_rows = [row for row in selected_rows if int(row["repetition"]) == 0]
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=1, vmax=max(1, n_rounds))
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.8), constrained_layout=True)
    for ax, rule in zip(axes, RULE_DISPLAY_ORDER):
        cur = [row for row in rep_rows if row["rule"] == rule]
        ax.scatter(X_train_init[:, 0], X_train_init[:, 1], s=8, c="#c7c7c7", alpha=0.55, edgecolors="none")
        if cur:
            xs = np.asarray([float(row["x1"]) for row in cur], dtype=np.float64)
            ys = np.asarray([float(row["x2"]) for row in cur], dtype=np.float64)
            rounds = np.asarray([int(row["round"]) for row in cur], dtype=np.int64)
            ax.scatter(
                xs,
                ys,
                s=20,
                c=rounds,
                cmap=cmap,
                norm=norm,
                edgecolors="black",
                linewidths=0.15,
            )
        dense = patches.Circle(
            tuple(cfg["dense_center"]),
            radius=float(cfg["dense_radius"]),
            fill=False,
            linewidth=1.2,
            linestyle=":",
            edgecolor="#4d4d4d",
        )
        relevant = patches.Circle(
            tuple(cfg["bump_center"]),
            radius=float(cfg["relevant_radius"]),
            fill=False,
            linewidth=1.4,
            linestyle="-",
            edgecolor="black",
        )
        distractor = patches.Rectangle(
            (float(cfg["distractor_region"]["x1_min"]), float(cfg["distractor_region"]["x2_min"])),
            1.0 - float(cfg["distractor_region"]["x1_min"]),
            1.0 - float(cfg["distractor_region"]["x2_min"]),
            fill=False,
            linewidth=1.2,
            linestyle="--",
            edgecolor="black",
        )
        ax.add_patch(dense)
        ax.add_patch(relevant)
        ax.add_patch(distractor)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(RULE_TABLE_LABELS[rule], fontsize=9)
        ax.set_xlabel("$x_1$")
        if ax is axes[0]:
            ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.12)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Acquisition round")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_allocation_bar_png(path: Path, allocation_rows: Sequence[Dict[str, object]]) -> None:
    categories = [
        ("Relevant", "fraction_relevant_region"),
        ("Distractor", "fraction_distractor_region"),
        ("Dense", "fraction_dense_region"),
        ("Other", "fraction_other_region"),
    ]
    style_map = make_style_map(RULE_DISPLAY_ORDER)
    x = np.arange(len(categories), dtype=np.float64)
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for idx, rule in enumerate(RULE_DISPLAY_ORDER):
        row = next(r for r in allocation_rows if r["rule"] == rule)
        vals = [float(row[key]) for _, key in categories]
        style = style_map[rule]
        ax.bar(
            x + (idx - 2) * width,
            vals,
            width=width,
            color=style["color"],
            edgecolor="black",
            linewidth=0.2,
            label=RULE_TABLE_LABELS[rule],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in categories])
    ax.set_ylabel("Fraction of acquired points")
    ax.set_ylim(0.0, max(0.18, max(float(row["fraction_dense_region"]) for row in allocation_rows) * 1.15))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def best_rule(rows: Sequence[Dict[str, object]], metric: str) -> Dict[str, object]:
    return min(rows, key=lambda row: float(row[metric]))


def write_interpretation(
    path: Path,
    final_rows: Sequence[Dict[str, object]],
    allocation_rows: Sequence[Dict[str, object]],
) -> None:
    best_final_weak = best_rule(final_rows, "weak_mse")
    best_weak_auc = best_rule(final_rows, "weak_auc")
    best_gap_auc = best_rule(final_rows, "gap_auc")
    support = next(row for row in allocation_rows if row["rule"] == "Support-only acquisition")
    profile = next(row for row in allocation_rows if row["rule"] == "Profile-aware acquisition")
    support_distractor = float(support["fraction_distractor_region"])
    profile_distractor = float(profile["fraction_distractor_region"])
    support_relevant = float(support["fraction_relevant_region"])
    profile_relevant = float(profile["fraction_relevant_region"])
    improved_allocation = (profile_relevant > support_relevant) and (profile_distractor < support_distractor)
    text = (
        f"Variant: {final_rows[0]['variant']}\n"
        f"Best final weak MSE: {best_final_weak['rule']} ({float(best_final_weak['weak_mse']):.6f})\n"
        f"Best weak-MSE AUC: {best_weak_auc['rule']} ({float(best_weak_auc['weak_auc']):.6f})\n"
        f"Best gap AUC: {best_gap_auc['rule']} ({float(best_gap_auc['gap_auc']):.6f})\n"
        f"Support-only distractor fraction: {support_distractor:.4f}\n"
        f"Profile-aware distractor fraction: {profile_distractor:.4f}\n"
        f"Support-only relevant-region fraction: {support_relevant:.4f}\n"
        f"Profile-aware relevant-region fraction: {profile_relevant:.4f}\n"
        f"Profile-aware improves allocation toward the weak-support high-error region: {'yes' if improved_allocation else 'no'}\n"
    )
    path.write_text(text)


def main() -> None:
    args = apply_fast_mode(parse_args())
    outdir = resolve_outdir(args.outdir)
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    results_dir = outdir / "results"
    for directory in (figures_dir, tables_dir, results_dir, PROJECT_ROOT / ".mplconfig"):
        directory.mkdir(parents=True, exist_ok=True)

    def run_variant(variant: str) -> tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], np.ndarray]:
        cfg = conflict_v2_config(variant)
        train_cfg = mixture_cfg(cfg["train_weights"], cfg["train_components"])
        eval_cfg = mixture_cfg(cfg["eval_weights"], cfg["eval_components"])
        round_rows: List[Dict[str, object]] = []
        selected_rows: List[Dict[str, object]] = []
        representative_train = None
        for rep in range(args.n_reps):
            rep_seed = int(args.seed + 1000 * rep + (0 if variant == "base" else 100000))
            rng = np.random.default_rng(rep_seed)
            X_train_init = sample_mixture_design(args.n_initial, train_cfg, rng)
            X_pool_init = sample_uniform_points(args.n_pool, rng)
            X_eval = sample_mixture_design(args.n_eval, eval_cfg, rng)
            y_train_init = add_noisy_labels(f_conflict_v2(X_train_init, cfg), float(args.sigma), rng)
            y_pool_clean = f_conflict_v2(X_pool_init, cfg)
            y_pool_noisy = add_noisy_labels(y_pool_clean, float(args.sigma), rng)
            y_eval_clean = f_conflict_v2(X_eval, cfg)
            if rep == 0:
                representative_train = X_train_init.copy()
            for rule_idx, rule in enumerate(RULE_DISPLAY_ORDER):
                rule_rng = np.random.default_rng(rep_seed + 100 * (rule_idx + 1) + 7)
                out = run_one_rule(
                    rule=rule,
                    rep=rep,
                    rep_seed=rep_seed,
                    rng=rule_rng,
                    cfg=cfg,
                    X_train_init=X_train_init,
                    y_train_init=y_train_init,
                    X_pool_init=X_pool_init,
                    y_pool_clean_init=y_pool_clean,
                    y_pool_noisy_init=y_pool_noisy,
                    X_eval=X_eval,
                    y_eval_clean=y_eval_clean,
                    args=args,
                )
                round_rows.extend(out["round_rows"])
                selected_rows.extend(out["selected_rows"])
        return cfg, round_rows, selected_rows, np.asarray(representative_train, dtype=np.float64)

    cfg, round_rows, selected_rows, representative_train = run_variant("base")
    final_rows = aggregate_final_and_auc(round_rows, int(args.n_rounds))
    allocation_rows = summarize_allocation(selected_rows)

    support_row = next(row for row in final_rows if row["rule"] == "Support-only acquisition")
    profile_row = next(row for row in final_rows if row["rule"] == "Profile-aware acquisition")
    weak_gain = (float(support_row["weak_auc"]) - float(profile_row["weak_auc"])) / max(float(support_row["weak_auc"]), 1e-12)
    gap_gain = (float(support_row["gap_auc"]) - float(profile_row["gap_auc"])) / max(float(support_row["gap_auc"]), 1e-12)
    if weak_gain < 0.10 and gap_gain < 0.10:
        cfg, round_rows, selected_rows, representative_train = run_variant("strong")
        final_rows = aggregate_final_and_auc(round_rows, int(args.n_rounds))
        allocation_rows = summarize_allocation(selected_rows)

    save_csv(results_dir / "results_exp7_conflict_round_metrics.csv", round_rows)
    save_csv(results_dir / "results_exp7_conflict_selected_points.csv", selected_rows)
    save_csv(results_dir / "results_exp7_conflict_final_summary.csv", final_rows)
    save_csv(results_dir / "results_exp7_conflict_allocation_summary.csv", allocation_rows)

    save_final_metrics_table(tables_dir / "table_exp7_conflict_final_metrics.tex", final_rows)
    save_auc_table(tables_dir / "table_exp7_conflict_auc_metrics.tex", final_rows)
    save_allocation_table(tables_dir / "table_exp7_conflict_allocation.tex", allocation_rows)

    save_metric_trajectory_png(
        figures_dir / "exp7_conflict_trajectory_weak_mse.png",
        round_rows,
        "weak_mse",
        "Weak-support MSE",
        int(args.n_rounds),
    )
    save_metric_trajectory_png(
        figures_dir / "exp7_conflict_trajectory_gap.png",
        round_rows,
        "gap",
        "Weak/dense gap",
        int(args.n_rounds),
    )
    save_metric_trajectory_png(
        figures_dir / "exp7_conflict_trajectory_slope.png",
        round_rows,
        "positive_slope",
        "Positive profile slope",
        int(args.n_rounds),
    )
    save_allocation_map_png(
        figures_dir / "exp7_conflict_allocation_map.png",
        selected_rows,
        representative_train,
        cfg,
        int(args.n_rounds),
    )
    save_allocation_bar_png(
        figures_dir / "exp7_conflict_allocation_bar.png",
        allocation_rows,
    )
    write_interpretation(results_dir / "exp7_conflict_interpretation.txt", final_rows, allocation_rows)

    summary = {
        "variant": cfg["name"],
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
            "seeds": [int(args.seed + 1000 * rep) for rep in range(args.n_reps)],
        },
        "final_summary": final_rows,
        "allocation_summary": allocation_rows,
        "output_paths": {
            "round_metrics_csv": str((results_dir / "results_exp7_conflict_round_metrics.csv").resolve()),
            "final_summary_csv": str((results_dir / "results_exp7_conflict_final_summary.csv").resolve()),
            "allocation_summary_csv": str((results_dir / "results_exp7_conflict_allocation_summary.csv").resolve()),
            "final_table": str((tables_dir / "table_exp7_conflict_final_metrics.tex").resolve()),
            "auc_table": str((tables_dir / "table_exp7_conflict_auc_metrics.tex").resolve()),
            "allocation_table": str((tables_dir / "table_exp7_conflict_allocation.tex").resolve()),
            "weak_png": str((figures_dir / "exp7_conflict_trajectory_weak_mse.png").resolve()),
            "gap_png": str((figures_dir / "exp7_conflict_trajectory_gap.png").resolve()),
            "slope_png": str((figures_dir / "exp7_conflict_trajectory_slope.png").resolve()),
            "allocation_map_png": str((figures_dir / "exp7_conflict_allocation_map.png").resolve()),
            "allocation_bar_png": str((figures_dir / "exp7_conflict_allocation_bar.png").resolve()),
            "interpretation_txt": str((results_dir / "exp7_conflict_interpretation.txt").resolve()),
        },
    }
    (results_dir / "results_exp7_conflict_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
