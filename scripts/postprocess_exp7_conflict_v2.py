from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

from figure_layout_utils import make_style_map
from exp7_conflict_v2 import (
    RULE_DISPLAY_ORDER,
    RULE_TABLE_LABELS,
    aggregate_final_and_auc,
    conflict_v2_config,
    f_conflict_v2,
    mixture_cfg,
    run_one_rule,
    sample_mixture_design,
    sample_uniform_points,
    add_noisy_labels,
    summarize_allocation,
)


FINAL_METRICS = [
    ("global_mse", "final global MSE"),
    ("weak_mse", "final weak MSE"),
    ("dense_mse", "final dense MSE"),
    ("gap", "final weak/dense gap"),
    ("positive_slope", "final positive profile slope"),
]

AUC_METRICS = [
    ("weak_auc", "weak-MSE AUC"),
    ("gap_auc", "gap AUC"),
    ("slope_auc", "slope AUC"),
]

ALLOC_METRICS = [
    ("fraction_distractor_region", "sparse low-error distractor allocation fraction"),
    ("fraction_relevant_region", "weak-support high-error relevant-region allocation fraction"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess Experiment 7 conflict_v2 outputs."
    )
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260520)
    parser.add_argument(
        "--gamma-grid",
        type=str,
        default="0,0.5,1,1.5,2",
        help="Comma-separated gamma_acq values for profile-aware sensitivity.",
    )
    return parser.parse_args()


def resolve_outdir(outdir: str) -> Path:
    path = Path(outdir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd_se_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    se = float(sd / math.sqrt(values.size)) if values.size > 0 else float("nan")
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    n = values.size
    for b in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[b] = float(np.mean(sample))
    ci_lower = float(np.quantile(boot_means, 0.025))
    ci_upper = float(np.quantile(boot_means, 0.975))
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def compute_per_rep_final_and_auc(round_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rule in RULE_DISPLAY_ORDER:
        rule_df = round_df.loc[round_df["rule"] == rule].copy()
        if rule_df.empty:
            continue
        final_round = int(rule_df["round"].max())
        for rep, rep_df in rule_df.groupby("repetition"):
            rep_df = rep_df.sort_values("round")
            final_row = rep_df.loc[rep_df["round"] == final_round].iloc[0]
            rows.append(
                {
                    "rule": rule,
                    "repetition": int(rep),
                    "global_mse": float(final_row["global_mse"]),
                    "weak_mse": float(final_row["weak_mse"]),
                    "dense_mse": float(final_row["dense_mse"]),
                    "gap": float(final_row["gap"]),
                    "positive_slope": float(final_row["positive_slope"]),
                    "weak_auc": float(rep_df["weak_mse"].mean()),
                    "gap_auc": float(rep_df["gap"].mean()),
                    "slope_auc": float(rep_df["positive_slope"].mean()),
                }
            )
    return pd.DataFrame(rows)


def compute_per_rep_allocation(selected_df: pd.DataFrame) -> pd.DataFrame:
    grouped = selected_df.groupby(["rule", "repetition"], sort=False)
    rows: List[Dict[str, object]] = []
    for (rule, repetition), g in grouped:
        rows.append(
            {
                "rule": str(rule),
                "repetition": int(repetition),
                "fraction_distractor_region": float(g["in_distractor_region"].astype(float).mean()),
                "fraction_relevant_region": float(g["in_relevant_region"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_uncertainty_summary(
    per_rep_metrics: pd.DataFrame,
    per_rep_alloc: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, object]]:
    merged = pd.merge(per_rep_metrics, per_rep_alloc, on=["rule", "repetition"], how="inner")
    metric_specs = FINAL_METRICS + AUC_METRICS + ALLOC_METRICS
    rows: List[Dict[str, object]] = []
    for rule in RULE_DISPLAY_ORDER:
        rule_df = merged.loc[merged["rule"] == rule].copy()
        if rule_df.empty:
            continue
        for metric_key, metric_label in metric_specs:
            rng = np.random.default_rng(seed + 1000 * RULE_DISPLAY_ORDER.index(rule) + hash(metric_key) % 997)
            stats = mean_sd_se_ci(rule_df[metric_key].to_numpy(dtype=np.float64), n_bootstrap, rng)
            rows.append(
                {
                    "rule": rule,
                    "metric": metric_label,
                    "metric_key": metric_key,
                    "n_reps": int(rule_df.shape[0]),
                    "mean": stats["mean"],
                    "sd": stats["sd"],
                    "se": stats["se"],
                    "bootstrap_ci_lower": stats["ci_lower"],
                    "bootstrap_ci_upper": stats["ci_upper"],
                }
            )
    return rows


def save_uncertainty_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Acquisition rule & Metric & Mean & SD & SE & 95\\% CI low & 95\\% CI high \\\\",
        "\\midrule",
    ]
    last_rule = None
    for row in rows:
        rule = str(row["rule"])
        rule_label = RULE_TABLE_LABELS[rule] if rule != last_rule else ""
        last_rule = rule
        lines.append(
            f"{rule_label} & {row['metric']} & "
            f"{float(row['mean']):.6f} & "
            f"{float(row['sd']):.6f} & "
            f"{float(row['se']):.6f} & "
            f"{float(row['bootstrap_ci_lower']):.6f} & "
            f"{float(row['bootstrap_ci_upper']):.6f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def parse_gamma_grid(gamma_grid: str) -> List[float]:
    out = []
    for item in gamma_grid.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    if not out:
        raise ValueError("Gamma grid must be nonempty.")
    return out


def rerun_profile_aware_gamma_sensitivity(
    summary_path: Path,
    gamma_values: Sequence[float],
) -> pd.DataFrame:
    summary = json.loads(summary_path.read_text())
    params = summary["parameters"]
    seeds = [int(s) for s in params["seeds"]]
    args = argparse.Namespace(
        n_initial=int(params["n_initial"]),
        n_pool=int(params["n_pool"]),
        n_eval=int(params["n_eval"]),
        n_rounds=int(params["n_rounds"]),
        batch_size=int(params["batch_size"]),
        n_reps=int(params["n_reps"]),
        n_bins=int(params["n_bins"]),
        k_support=int(params["k_support"]),
        sigma=float(params["sigma"]),
        gamma=float(params["gamma"]),
        lambda_reg=float(params["lambda_reg"]),
        gamma_acq=1.0,
    )

    cfg = conflict_v2_config("base")
    train_cfg = mixture_cfg(cfg["train_weights"], cfg["train_components"])
    eval_cfg = mixture_cfg(cfg["eval_weights"], cfg["eval_components"])

    per_gamma_rows: List[Dict[str, object]] = []
    per_gamma_global_by_rep: Dict[float, Dict[int, float]] = {}

    for gamma_value in gamma_values:
        round_rows: List[Dict[str, object]] = []
        selected_rows: List[Dict[str, object]] = []
        for rep, rep_seed in enumerate(seeds):
            rng = np.random.default_rng(int(rep_seed))
            X_train_init = sample_mixture_design(args.n_initial, train_cfg, rng)
            X_pool_init = sample_uniform_points(args.n_pool, rng)
            X_eval = sample_mixture_design(args.n_eval, eval_cfg, rng)
            y_train_init = add_noisy_labels(f_conflict_v2(X_train_init, cfg), float(args.sigma), rng)
            y_pool_clean = f_conflict_v2(X_pool_init, cfg)
            y_pool_noisy = add_noisy_labels(y_pool_clean, float(args.sigma), rng)
            y_eval_clean = f_conflict_v2(X_eval, cfg)
            run_args = argparse.Namespace(**vars(args))
            run_args.gamma_acq = float(gamma_value)
            rule_rng = np.random.default_rng(int(rep_seed) + 100 * (RULE_DISPLAY_ORDER.index("Profile-aware acquisition") + 1) + 7)
            out = run_one_rule(
                rule="Profile-aware acquisition",
                rep=int(rep),
                rep_seed=int(rep_seed),
                rng=rule_rng,
                cfg=cfg,
                X_train_init=X_train_init,
                y_train_init=y_train_init,
                X_pool_init=X_pool_init,
                y_pool_clean_init=y_pool_clean,
                y_pool_noisy_init=y_pool_noisy,
                X_eval=X_eval,
                y_eval_clean=y_eval_clean,
                args=run_args,
            )
            round_rows.extend(out["round_rows"])
            selected_rows.extend(out["selected_rows"])

        final_rows = aggregate_final_and_auc(round_rows, int(args.n_rounds))
        alloc_rows = summarize_allocation(selected_rows)
        final_row = next(row for row in final_rows if row["rule"] == "Profile-aware acquisition")
        alloc_row = next(row for row in alloc_rows if row["rule"] == "Profile-aware acquisition")

        per_rep_df = compute_per_rep_final_and_auc(pd.DataFrame(round_rows))
        profile_rep = per_rep_df.loc[per_rep_df["rule"] == "Profile-aware acquisition"].copy()
        per_gamma_global_by_rep[float(gamma_value)] = {
            int(row["repetition"]): float(row["global_mse"]) for _, row in profile_rep.iterrows()
        }

        per_gamma_rows.append(
            {
                "gamma_acq": float(gamma_value),
                "final_global_mse": float(final_row["global_mse"]),
                "final_weak_mse": float(final_row["weak_mse"]),
                "weak_auc": float(final_row["weak_auc"]),
                "gap_auc": float(final_row["gap_auc"]),
                "slope_auc": float(final_row["slope_auc"]),
                "fraction_distractor_region": float(alloc_row["fraction_distractor_region"]),
                "fraction_relevant_region": float(alloc_row["fraction_relevant_region"]),
                "global_mse_change_pct_vs_gamma1": float("nan"),
            }
        )
    gamma_df = pd.DataFrame(per_gamma_rows).sort_values("gamma_acq").reset_index(drop=True)
    if 1.0 not in per_gamma_global_by_rep:
        raise RuntimeError("gamma_acq=1.0 baseline is required for sensitivity comparison.")
    baseline_global_by_rep = per_gamma_global_by_rep[1.0]
    delta_values = []
    for gamma_value in gamma_df["gamma_acq"].to_numpy(dtype=np.float64):
        current = per_gamma_global_by_rep[float(gamma_value)]
        per_rep_changes = []
        for rep, base in baseline_global_by_rep.items():
            per_rep_changes.append(
                100.0 * (current[rep] - base) / max(abs(base), 1e-12)
            )
        delta_values.append(float(np.mean(np.asarray(per_rep_changes, dtype=np.float64))))
    gamma_df["global_mse_change_pct_vs_gamma1"] = np.asarray(delta_values, dtype=np.float64)
    return gamma_df


def save_gamma_table(path: Path, gamma_df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{cccccccc}",
        "\\toprule",
        "$\\gamma_{\\mathrm{acq}}$ & Final weak MSE & Weak AUC & Gap AUC & Slope AUC & Distractor frac. & Relevant frac. & $\\Delta$ global MSE (\\%) \\\\",
        "\\midrule",
    ]
    for _, row in gamma_df.iterrows():
        lines.append(
            f"{float(row['gamma_acq']):.1f} & "
            f"{float(row['final_weak_mse']):.6f} & "
            f"{float(row['weak_auc']):.6f} & "
            f"{float(row['gap_auc']):.3f} & "
            f"{float(row['slope_auc']):.3f} & "
            f"{float(row['fraction_distractor_region']):.4f} & "
            f"{float(row['fraction_relevant_region']):.4f} & "
            f"{float(row['global_mse_change_pct_vs_gamma1']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_gamma_plot(path: Path, gamma_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True)
    axes = axes.ravel()
    gamma = gamma_df["gamma_acq"].to_numpy(dtype=np.float64)
    metric_specs = [
        ("final_weak_mse", "Final weak MSE"),
        ("weak_auc", "Weak-MSE AUC"),
        ("gap_auc", "Gap AUC"),
        ("slope_auc", "Slope AUC"),
        ("fraction_distractor_region", "Distractor frac."),
        ("fraction_relevant_region", "Relevant frac."),
    ]
    line_styles = [
        {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        {"color": "#d62728", "marker": "s", "linestyle": "--"},
        {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
        {"color": "#9467bd", "marker": "D", "linestyle": ":"},
        {"color": "#ff7f0e", "marker": "v", "linestyle": (0, (5, 1, 1, 1))},
        {"color": "#8c564b", "marker": "P", "linestyle": (0, (3, 2))},
    ]
    for ax, (metric, title), style in zip(axes, metric_specs, line_styles):
        values = gamma_df[metric].to_numpy(dtype=np.float64)
        ax.plot(
            gamma,
            values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=5.5,
        )
        ax.axvline(1.0, color="#4d4d4d", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("$\\gamma_{\\mathrm{acq}}$")
        ax.grid(alpha=0.18)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_gamma_interpretation(path: Path, gamma_df: pd.DataFrame) -> None:
    best_weak_auc = gamma_df.loc[gamma_df["weak_auc"].idxmin()]
    best_gap_auc = gamma_df.loc[gamma_df["gap_auc"].idxmin()]
    row0 = gamma_df.loc[np.isclose(gamma_df["gamma_acq"], 0.0)].iloc[0]
    row1 = gamma_df.loc[np.isclose(gamma_df["gamma_acq"], 1.0)].iloc[0]
    row2 = gamma_df.loc[np.isclose(gamma_df["gamma_acq"], 2.0)].iloc[0]
    text = (
        "Conflict-v2 profile-aware gamma_acq sensitivity.\n"
        f"Best weak-MSE AUC: gamma_acq={float(best_weak_auc['gamma_acq']):.1f} ({float(best_weak_auc['weak_auc']):.6f})\n"
        f"Best gap AUC: gamma_acq={float(best_gap_auc['gamma_acq']):.1f} ({float(best_gap_auc['gap_auc']):.6f})\n"
        f"gamma_acq=0.0 final weak MSE={float(row0['final_weak_mse']):.6f}, distractor={float(row0['fraction_distractor_region']):.4f}, relevant={float(row0['fraction_relevant_region']):.4f}\n"
        f"gamma_acq=1.0 final weak MSE={float(row1['final_weak_mse']):.6f}, distractor={float(row1['fraction_distractor_region']):.4f}, relevant={float(row1['fraction_relevant_region']):.4f}\n"
        f"gamma_acq=2.0 final weak MSE={float(row2['final_weak_mse']):.6f}, distractor={float(row2['fraction_distractor_region']):.4f}, relevant={float(row2['fraction_relevant_region']):.4f}\n"
        "Interpretation: gamma_acq=0 behaves more like an error-dominated rule, while larger gamma_acq values place more emphasis on weak-support regions.\n"
        "The default gamma_acq=1 should be read as a pre-specified balanced joint error-support rule, not as a universally optimal exponent.\n"
    )
    path.write_text(text)


def main() -> None:
    args = parse_args()
    outdir = resolve_outdir(args.outdir)
    results_dir = outdir / "results"
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    for directory in (results_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    round_path = results_dir / "results_exp7_conflict_round_metrics.csv"
    selected_path = results_dir / "results_exp7_conflict_selected_points.csv"
    summary_path = results_dir / "results_exp7_conflict_v2_summary.json"

    round_df = pd.read_csv(round_path)
    selected_df = pd.read_csv(selected_path)

    per_rep_metrics = compute_per_rep_final_and_auc(round_df)
    per_rep_alloc = compute_per_rep_allocation(selected_df)
    uncertainty_rows = build_uncertainty_summary(
        per_rep_metrics=per_rep_metrics,
        per_rep_alloc=per_rep_alloc,
        n_bootstrap=int(args.bootstrap_resamples),
        seed=int(args.bootstrap_seed),
    )
    uncertainty_csv = results_dir / "results_exp7_conflict_uncertainty_summary.csv"
    uncertainty_tex = tables_dir / "table_exp7_conflict_uncertainty_summary.tex"
    save_csv(uncertainty_csv, uncertainty_rows)
    save_uncertainty_table(uncertainty_tex, uncertainty_rows)

    gamma_values = parse_gamma_grid(args.gamma_grid)
    gamma_df = rerun_profile_aware_gamma_sensitivity(summary_path=summary_path, gamma_values=gamma_values)
    gamma_csv = results_dir / "results_exp7_conflict_gamma_sensitivity.csv"
    gamma_tex = tables_dir / "table_exp7_conflict_gamma_sensitivity.tex"
    gamma_png = figures_dir / "exp7_conflict_gamma_sensitivity.png"
    gamma_txt = results_dir / "exp7_conflict_gamma_interpretation.txt"
    gamma_df.to_csv(gamma_csv, index=False)
    save_gamma_table(gamma_tex, gamma_df)
    save_gamma_plot(gamma_png, gamma_df)
    write_gamma_interpretation(gamma_txt, gamma_df)


if __name__ == "__main__":
    main()
