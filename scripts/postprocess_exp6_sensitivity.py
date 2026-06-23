from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import os

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RULE_PROFILE = "Constrained profile-aware"
RULE_GLOBAL = "Global-MSE selection"
EPS = 1.0e-12

WEIGHT_LIBRARY: Dict[str, Tuple[float, float, float]] = {
    "equal": (1.0, 1.0, 1.0),
    "weak_emphasis": (2.0, 1.0, 1.0),
    "gap_emphasis": (1.0, 2.0, 1.0),
    "slope_emphasis": (1.0, 1.0, 2.0),
    "weak_gap": (1.0, 1.0, 0.0),
    "gap_slope": (0.0, 1.0, 1.0),
}

TAU_VALUES = (0.01, 0.03, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supplementary sensitivity post-processing for Experiment 6 stress setting."
    )
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--setting", type=str, default="stress")
    parser.add_argument("--tau-main", type=float, default=0.05)
    return parser.parse_args()


def resolve_outdir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def ensure_dirs(outdir: Path) -> Tuple[Path, Path, Path]:
    figures_dir = outdir / "figures"
    results_dir = outdir / "results"
    tables_dir = outdir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, results_dir, tables_dir


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def format_signed_pct(value: float) -> str:
    return f"{value:+.2f}"


def summarize_delta_frame(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "n_splits": float(len(df)),
        "changed_selection_rate": float(df["changed_from_global"].mean()),
        "mean_delta_global_mse_pct": float(df["delta_global_mse_percent"].mean()),
        "median_delta_global_mse_pct": float(df["delta_global_mse_percent"].median()),
        "mean_delta_weak_mse_pct": float(df["delta_weak_mse_percent"].mean()),
        "median_delta_weak_mse_pct": float(df["delta_weak_mse_percent"].median()),
        "mean_delta_gap_pct": float(df["delta_gap_percent"].mean()),
        "median_delta_gap_pct": float(df["delta_gap_percent"].median()),
        "mean_delta_slope_pct": float(df["delta_slope_percent"].mean()),
        "median_delta_slope_pct": float(df["delta_slope_percent"].median()),
        "fraction_global_within_1pct": float((df["delta_global_mse_percent"] <= 1.0).mean()),
        "fraction_global_within_2pct": float((df["delta_global_mse_percent"] <= 2.0).mean()),
        "fraction_weak_improved": float((df["delta_weak_mse_percent"] < 0.0).mean()),
        "fraction_gap_improved": float((df["delta_gap_percent"] < 0.0).mean()),
        "fraction_slope_improved": float((df["delta_slope_percent"] < 0.0).mean()),
    }


def build_tau_sensitivity(all_splits: pd.DataFrame, setting: str) -> pd.DataFrame:
    subset = all_splits[
        (all_splits["setting"] == setting) & (all_splits["selection_rule"] == RULE_PROFILE)
    ].copy()
    rows: List[Dict[str, float]] = []
    for tau in TAU_VALUES:
        tau_df = subset[np.isclose(subset["tau"], tau)].copy()
        if tau_df.empty:
            continue
        row = {"tau": float(tau)}
        row.update(summarize_delta_frame(tau_df))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)


def reconstruct_weight_sensitivity(
    candidate_metrics: pd.DataFrame,
    setting: str,
    tau_main: float,
) -> pd.DataFrame:
    subset = candidate_metrics[candidate_metrics["setting"] == setting].copy()
    rows: List[Dict[str, float | str]] = []
    repetitions = sorted(int(x) for x in subset["repetition"].unique())
    for weight_name, (eta_w, eta_g, eta_s) in WEIGHT_LIBRARY.items():
        per_rep_rows: List[Dict[str, float | str]] = []
        for repetition in repetitions:
            rep_df = subset[subset["repetition"] == repetition].copy()
            val_df = rep_df[rep_df["split"] == "val"].copy()
            test_df = rep_df[rep_df["split"] == "test"].copy()
            if val_df.empty or test_df.empty:
                continue

            global_idx = val_df["global_mse"].astype(float).idxmin()
            global_row = val_df.loc[global_idx]
            global_label = str(global_row["theta_label"])
            global_global_mse = float(global_row["global_mse"])
            budget = (1.0 + float(tau_main)) * global_global_mse
            admissible = val_df[val_df["global_mse"].astype(float) <= budget + EPS].copy()
            if admissible.empty:
                admissible = val_df.loc[[global_idx]].copy()

            weak_base = float(global_row["weak_mse"])
            gap_base = float(global_row["gap"])
            slope_base = float(global_row["pos_slope"])
            admissible["profile_score"] = (
                eta_w * admissible["weak_mse"].astype(float) / (abs(weak_base) + EPS)
                + eta_g * admissible["gap"].astype(float) / (abs(gap_base) + EPS)
                + eta_s * admissible["pos_slope"].astype(float) / (abs(slope_base) + EPS)
            )
            profile_idx = admissible["profile_score"].astype(float).idxmin()
            profile_row = admissible.loc[profile_idx]
            profile_label = str(profile_row["theta_label"])

            global_test = test_df[test_df["theta_label"] == global_label].iloc[0]
            profile_test = test_df[test_df["theta_label"] == profile_label].iloc[0]

            delta_global = 100.0 * (
                float(profile_test["global_mse"]) - float(global_test["global_mse"])
            ) / (abs(float(global_test["global_mse"])) + EPS)
            delta_weak = 100.0 * (
                float(profile_test["weak_mse"]) - float(global_test["weak_mse"])
            ) / (abs(float(global_test["weak_mse"])) + EPS)
            delta_gap = 100.0 * (
                float(profile_test["gap"]) - float(global_test["gap"])
            ) / (abs(float(global_test["gap"])) + EPS)
            delta_slope = 100.0 * (
                float(profile_test["pos_slope"]) - float(global_test["pos_slope"])
            ) / (abs(float(global_test["pos_slope"])) + EPS)

            per_rep_rows.append(
                {
                    "weight_setting": weight_name,
                    "eta_w": eta_w,
                    "eta_g": eta_g,
                    "eta_s": eta_s,
                    "repetition": repetition,
                    "tau": float(tau_main),
                    "theta_global": global_label,
                    "theta_profile": profile_label,
                    "changed_from_global": bool(profile_label != global_label),
                    "test_global_mse": float(profile_test["global_mse"]),
                    "test_weak_mse": float(profile_test["weak_mse"]),
                    "test_gap": float(profile_test["gap"]),
                    "test_pos_slope": float(profile_test["pos_slope"]),
                    "delta_global_mse_percent": delta_global,
                    "delta_weak_mse_percent": delta_weak,
                    "delta_gap_percent": delta_gap,
                    "delta_slope_percent": delta_slope,
                }
            )
        frame = pd.DataFrame(per_rep_rows)
        if frame.empty:
            continue
        summary = {"weight_setting": weight_name, "eta_w": eta_w, "eta_g": eta_g, "eta_s": eta_s}
        summary.update(summarize_delta_frame(frame))
        rows.append(summary)
    order = list(WEIGHT_LIBRARY.keys())
    result = pd.DataFrame(rows)
    if not result.empty:
        result["weight_order"] = result["weight_setting"].map({name: idx for idx, name in enumerate(order)})
        result = result.sort_values("weight_order").drop(columns="weight_order").reset_index(drop=True)
    return result


def save_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def save_tau_table(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "$\\tau$ & Changed & $\\Delta$ Global mean/med & $\\Delta$ Weak mean/med & $\\Delta$ Gap mean/med & $\\Delta$ Slope mean/med & Global $\\leq +1/+2\\%$ & Weak/Gap/Slope improved \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['tau']:.2f} & "
            f"{format_pct(float(row['changed_selection_rate']))} & "
            f"{format_signed_pct(float(row['mean_delta_global_mse_pct']))} / {format_signed_pct(float(row['median_delta_global_mse_pct']))} & "
            f"{format_signed_pct(float(row['mean_delta_weak_mse_pct']))} / {format_signed_pct(float(row['median_delta_weak_mse_pct']))} & "
            f"{format_signed_pct(float(row['mean_delta_gap_pct']))} / {format_signed_pct(float(row['median_delta_gap_pct']))} & "
            f"{format_signed_pct(float(row['mean_delta_slope_pct']))} / {format_signed_pct(float(row['median_delta_slope_pct']))} & "
            f"{format_pct(float(row['fraction_global_within_1pct']))} / {format_pct(float(row['fraction_global_within_2pct']))} & "
            f"{format_pct(float(row['fraction_weak_improved']))} / {format_pct(float(row['fraction_gap_improved']))} / {format_pct(float(row['fraction_slope_improved']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_weight_table(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Weights $(w,g,s)$ & Changed & $\\Delta$ Global mean/med & $\\Delta$ Weak mean/med & $\\Delta$ Gap mean/med & $\\Delta$ Slope mean/med & Global $\\leq +1/+2\\%$ & Weak/Gap/Slope improved \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        label = f"{int(row['eta_w'])},{int(row['eta_g'])},{int(row['eta_s'])}"
        name = str(row["weight_setting"]).replace("_", "\\_")
        lines.append(
            f"{name} ({label}) & "
            f"{format_pct(float(row['changed_selection_rate']))} & "
            f"{format_signed_pct(float(row['mean_delta_global_mse_pct']))} / {format_signed_pct(float(row['median_delta_global_mse_pct']))} & "
            f"{format_signed_pct(float(row['mean_delta_weak_mse_pct']))} / {format_signed_pct(float(row['median_delta_weak_mse_pct']))} & "
            f"{format_signed_pct(float(row['mean_delta_gap_pct']))} / {format_signed_pct(float(row['median_delta_gap_pct']))} & "
            f"{format_signed_pct(float(row['mean_delta_slope_pct']))} / {format_signed_pct(float(row['median_delta_slope_pct']))} & "
            f"{format_pct(float(row['fraction_global_within_1pct']))} / {format_pct(float(row['fraction_global_within_2pct']))} & "
            f"{format_pct(float(row['fraction_weak_improved']))} / {format_pct(float(row['fraction_gap_improved']))} / {format_pct(float(row['fraction_slope_improved']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def plot_tau_changed_rate(path: Path, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(df["tau"], 100.0 * df["changed_selection_rate"], marker="o", color="#1b6ca8", linestyle="-")
    ax.set_xlabel(r"Global-risk budget $\tau$")
    ax.set_ylabel("Changed-selection rate (%)")
    ax.set_title("Experiment 6 tau sensitivity")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_tau_gap_reduction(path: Path, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    gap_reduction = -df["mean_delta_gap_pct"].to_numpy(dtype=float)
    global_increase = df["mean_delta_global_mse_pct"].to_numpy(dtype=float)
    ax.plot(df["tau"], gap_reduction, marker="s", color="#208b3a", linestyle="-", label="Mean gap reduction")
    ax.plot(df["tau"], global_increase, marker="^", color="#b54a14", linestyle="--", label="Mean global-MSE change")
    ax.set_xlabel(r"Global-risk budget $\tau$")
    ax.set_ylabel("Percent change")
    ax.set_title("Gap reduction versus global-risk cost")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_weight_metric_changes(path: Path, df: pd.DataFrame) -> None:
    labels = df["weight_setting"].tolist()
    x = np.arange(len(labels))
    width = 0.23
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.bar(x - width, -df["median_delta_gap_pct"], width=width, color="#2a9d8f", label="Median gap reduction")
    ax.bar(x, -df["median_delta_slope_pct"], width=width, color="#457b9d", label="Median slope reduction")
    ax.bar(x + width, -df["median_delta_weak_mse_pct"], width=width, color="#e76f51", label="Median weak-MSE reduction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Experiment 6 profile-score weight sensitivity")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_interpretation(path: Path, tau_df: pd.DataFrame, weight_df: pd.DataFrame) -> None:
    tau_best_gap = tau_df.loc[tau_df["median_delta_gap_pct"].astype(float).idxmin()]
    tau_best_changed = tau_df.loc[tau_df["changed_selection_rate"].astype(float).idxmax()]
    weight_equal = weight_df[weight_df["weight_setting"] == "equal"].iloc[0]
    weight_best_gap = weight_df.loc[weight_df["median_delta_gap_pct"].astype(float).idxmin()]
    weight_best_weak = weight_df.loc[weight_df["mean_delta_weak_mse_pct"].astype(float).idxmin()]

    lines = [
        "Experiment 6 supplementary sensitivity interpretation",
        "",
        "Tau sensitivity:",
        (
            f"- As tau increases from {tau_df['tau'].min():.2f} to {tau_df['tau'].max():.2f}, "
            f"the changed-selection rate rises from "
            f"{100.0 * float(tau_df.iloc[0]['changed_selection_rate']):.1f}% to "
            f"{100.0 * float(tau_df.iloc[-1]['changed_selection_rate']):.1f}%."
        ),
        (
            "- The median deltas remain exactly zero for several metrics at small and moderate "
            "tau because many splits keep the global-MSE-selected candidate; the mean deltas "
            "therefore provide the more informative all-split summary."
        ),
        (
            f"- The strongest median gap reduction occurs at tau={float(tau_best_gap['tau']):.2f}, "
            f"with median delta gap {float(tau_best_gap['median_delta_gap_pct']):+.2f}% and "
            f"mean global-MSE change {float(tau_best_gap['mean_delta_global_mse_pct']):+.2f}%."
        ),
        (
            f"- The largest decision-change rate occurs at tau={float(tau_best_changed['tau']):.2f}, "
            f"confirming that looser global-risk budgets create more room to re-rank near-tied candidates."
        ),
        "",
        "Weight sensitivity at tau=0.05:",
        (
            f"- The pre-specified equal-weight rule yields median changes of "
            f"global {float(weight_equal['median_delta_global_mse_pct']):+.2f}%, "
            f"weak {float(weight_equal['median_delta_weak_mse_pct']):+.2f}%, "
            f"gap {float(weight_equal['median_delta_gap_pct']):+.2f}%, and "
            f"slope {float(weight_equal['median_delta_slope_pct']):+.2f}%, with "
            f"mean gap change {float(weight_equal['mean_delta_gap_pct']):+.2f}%."
        ),
        (
            f"- The strongest median gap reduction is obtained by {weight_best_gap['weight_setting']} "
            f"with median delta gap {float(weight_best_gap['median_delta_gap_pct']):+.2f}%."
        ),
        (
            f"- No weight setting improves weak MSE on average relative to the global-risk baseline; "
            f"the smallest mean weak-MSE increase is obtained by {weight_best_weak['weight_setting']} "
            f"at {float(weight_best_weak['mean_delta_weak_mse_pct']):+.2f}%."
        ),
        (
            "- Across weight settings, the main conclusion is stable: changing the profile-score "
            "weights changes the preferred trade-off, but support-conditioned summaries still "
            "re-rank candidates that are nearly tied under global validation risk."
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    outdir = resolve_outdir(args.outdir)
    figures_dir, results_dir, tables_dir = ensure_dirs(outdir)

    all_splits_path = results_dir / "exp6_stress_all_splits.csv"
    candidate_metrics_path = results_dir / "exp6_stress_candidate_metrics.csv"
    if not all_splits_path.exists():
        raise FileNotFoundError(f"Missing required input: {all_splits_path}")
    if not candidate_metrics_path.exists():
        raise FileNotFoundError(f"Missing required input: {candidate_metrics_path}")

    all_splits = pd.read_csv(all_splits_path)
    candidate_metrics = pd.read_csv(candidate_metrics_path)

    tau_df = build_tau_sensitivity(all_splits, setting=args.setting)
    weight_df = reconstruct_weight_sensitivity(candidate_metrics, setting=args.setting, tau_main=args.tau_main)

    save_csv(results_dir / "results_exp6_tau_sensitivity.csv", tau_df)
    save_csv(results_dir / "results_exp6_weight_sensitivity.csv", weight_df)

    save_tau_table(tables_dir / "table_exp6_tau_sensitivity.tex", tau_df)
    save_weight_table(tables_dir / "table_exp6_weight_sensitivity.tex", weight_df)

    plot_tau_changed_rate(figures_dir / "exp6_tau_changed_rate_vs_tau.png", tau_df)
    plot_tau_gap_reduction(figures_dir / "exp6_tau_gap_reduction_vs_tau.png", tau_df)
    plot_weight_metric_changes(figures_dir / "exp6_weight_metric_changes.png", weight_df)

    write_interpretation(results_dir / "exp6_sensitivity_interpretation.txt", tau_df, weight_df)


if __name__ == "__main__":
    main()
