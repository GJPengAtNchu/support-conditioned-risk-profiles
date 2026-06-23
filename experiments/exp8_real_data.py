from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

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
from matplotlib.lines import Line2D
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from figure_layout_utils import make_style_map, save_axes_group_panel, save_legend_figure


RULE_DISPLAY_ORDER = (
    "Global-MSE selection",
    "Constrained profile-aware selection",
)
MODEL_FAMILY_LABELS = {
    "krr": "KRR",
    "rf": "RandomForestRegressor",
    "gbrt": "GradientBoostingRegressor",
    "mlp": "MLPRegressor",
}
MAX_DATASETS_DEFAULT = 8
MAX_DATASETS_FAST = 4
MAX_SAMPLES_DEFAULT = 1800
MAX_SAMPLES_FAST = 800
TARGET_CANDIDATES = [
    "target",
    "y",
    "label",
    "response",
    "medv",
    "quality",
    "cnt",
    "traffic_volume",
    "global_active_power",
    "rented bike count",
    "appliances",
    "nox",
    "tey",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 8: model-agnostic and real-data applicability."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--k-support", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--zeta", type=float, default=0.25)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--models", type=str, default="krr,rf,gbrt")
    return parser.parse_args()


def apply_fast_mode(args: argparse.Namespace) -> argparse.Namespace:
    if args.fast:
        args.n_splits = 3
    return args


def resolve_outdir(outdir: str) -> Path:
    path = Path(outdir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def data_roots(data_dir: str) -> list[Path]:
    roots: list[Path] = []
    if data_dir:
        p = Path(data_dir)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        roots.append(p)
    roots.extend(
        [
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "datasets",
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "datasets",
        ]
    )
    out = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root.exists() and root not in seen:
            seen.add(root)
            out.append(root)
    return out


def _find_existing(paths: Sequence[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def deterministic_subsample(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
    n = y.shape[0]
    if max_samples <= 0 or n <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_samples, replace=False))
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx].reset_index(drop=True), y[idx]
    return X[idx], y[idx]


def load_gas_turbine_dataset(roots: Sequence[Path]) -> dict | None:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "gas_turbine/gt_full.csv",
                root / "gas_turbine_emissions/gt_full.csv",
                root / "gt_full.csv",
            ]
        )
    p = _find_existing(candidates)
    if p is None:
        return None
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    if "NOX" not in df.columns:
        return None
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    X = df.drop(columns=["NOX", *drop_cols], errors="ignore").apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["NOX"], errors="coerce").to_numpy(dtype=np.float64)
    return {
        "name": "GasTurbineNOx",
        "X": X,
        "y": y,
        "task_type": "regression",
        "source": str(p),
    }


def load_bike_dataset(roots: Sequence[Path]) -> dict | None:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "bike/day.csv",
                root / "Bike-Sharing-Dataset/day.csv",
                root / "day.csv",
            ]
        )
    p = _find_existing(candidates)
    if p is None:
        return None
    df = pd.read_csv(p)
    feat_cols = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
    ]
    if "cnt" not in df.columns or any(c not in df.columns for c in feat_cols):
        return None
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["cnt"], errors="coerce").to_numpy(dtype=np.float64)
    return {
        "name": "BikeDemand",
        "X": X,
        "y": y,
        "task_type": "regression",
        "source": str(p),
    }


def load_seoul_dataset(roots: Sequence[Path]) -> dict | None:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "seoul/SeoulBikeData.csv",
                root / "SeoulBikeData.csv",
            ]
        )
    p = _find_existing(candidates)
    if p is None:
        return None
    encodings = ("utf-8-sig", "cp949", "euc-kr", "latin-1")
    last_err: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(p, encoding=enc)
            if "Rented Bike Count" not in df.columns:
                continue
            season_map = {"Spring": 0.0, "Summer": 1.0, "Autumn": 2.0, "Winter": 3.0, "Fall": 2.0}
            holiday_map = {"No Holiday": 0.0, "Holiday": 1.0}
            functioning_map = {"Yes": 1.0, "No": 0.0}
            X = pd.DataFrame(
                {
                    "Hour": pd.to_numeric(df.get("Hour"), errors="coerce"),
                    "Temperature": pd.to_numeric(df.get("Temperature(°C)", df.get("Temperature(�C)")), errors="coerce"),
                    "Humidity": pd.to_numeric(df.get("Humidity(%)"), errors="coerce"),
                    "WindSpeed": pd.to_numeric(df.get("Wind speed (m/s)"), errors="coerce"),
                    "Visibility": pd.to_numeric(df.get("Visibility (10m)"), errors="coerce"),
                    "DewPoint": pd.to_numeric(df.get("Dew point temperature(°C)", df.get("Dew point temperature(�C)")), errors="coerce"),
                    "SolarRadiation": pd.to_numeric(df.get("Solar Radiation (MJ/m2)"), errors="coerce"),
                    "Rainfall": pd.to_numeric(df.get("Rainfall(mm)"), errors="coerce"),
                    "Snowfall": pd.to_numeric(df.get("Snowfall (cm)"), errors="coerce"),
                    "Season": df.get("Seasons").map(season_map),
                    "Holiday": df.get("Holiday").map(holiday_map),
                    "FunctioningDay": df.get("Functioning Day").map(functioning_map),
                }
            )
            y = pd.to_numeric(df["Rented Bike Count"], errors="coerce").to_numpy(dtype=np.float64)
            return {
                "name": "SeoulBikeSharingDemand",
                "X": X,
                "y": y,
                "task_type": "regression",
                "source": str(p),
            }
        except Exception as e:
            last_err = e
    print(f"Warning: failed to load SeoulBikeData.csv from {p}: {last_err}")
    return None


def load_energy_dataset(roots: Sequence[Path]) -> dict | None:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "energy_data/energydata_complete.csv",
                root / "energy/energydata_complete.csv",
                root / "energydata_complete.csv",
            ]
        )
    p = _find_existing(candidates)
    if p is None:
        return None
    df = pd.read_csv(p)
    if "Appliances" not in df.columns:
        return None
    out = df.copy()
    if "date" in out.columns:
        dt = pd.to_datetime(out["date"], errors="coerce")
        out["hour"] = dt.dt.hour.astype(float)
        out["dayofweek"] = dt.dt.dayofweek.astype(float)
        out["month"] = dt.dt.month.astype(float)
        out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
        out = out.drop(columns=["date"])
    X = out.drop(columns=["Appliances"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(out["Appliances"], errors="coerce").to_numpy(dtype=np.float64)
    return {
        "name": "AppliancesEnergy",
        "X": X,
        "y": y,
        "task_type": "regression",
        "source": str(p),
    }


def load_boston_dataset(roots: Sequence[Path]) -> dict | None:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "boston/housing.data",
                root / "housing.data",
            ]
        )
    p = _find_existing(candidates)
    if p is None:
        return None
    try:
        arr = np.loadtxt(p, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < 14:
        return None
    X = pd.DataFrame(arr[:, :-1])
    y = arr[:, -1].astype(np.float64)
    return {
        "name": "BostonHousing",
        "X": X,
        "y": y,
        "task_type": "regression",
        "source": str(p),
    }


def load_ccpp_dataset(roots: Sequence[Path]) -> dict | None:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "CCPP/Folds5x2_pp.xlsx",
                root / "CCPP/Folds5x2_pp.ods",
                root / "Folds5x2_pp.xlsx",
            ]
        )
    p = _find_existing(candidates)
    if p is None:
        return None
    try:
        if p.suffix.lower() == ".xlsx":
            df = pd.read_excel(p)
        else:
            df = pd.read_excel(p, engine="odf")
    except Exception:
        return None
    if "PE" not in df.columns:
        return None
    X = df.drop(columns=["PE"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["PE"], errors="coerce").to_numpy(dtype=np.float64)
    return {
        "name": "CCPPPowerOutput",
        "X": X,
        "y": y,
        "task_type": "regression",
        "source": str(p),
    }


def load_household_power_dataset(roots: Sequence[Path]) -> dict | None:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "electricity/household_power_consumption.txt",
                root / "household_power_consumption.txt",
            ]
        )
    p = _find_existing(candidates)
    if p is None:
        return None
    usecols = [
        "Date",
        "Time",
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    try:
        df = pd.read_csv(
            p,
            sep=";",
            usecols=usecols,
            na_values="?",
            low_memory=False,
        )
    except Exception:
        return None
    if "Global_active_power" not in df.columns:
        return None
    out = pd.DataFrame(
        {
            "Date": df.get("Date"),
            "Time": df.get("Time"),
            "Global_reactive_power": pd.to_numeric(df.get("Global_reactive_power"), errors="coerce"),
            "Voltage": pd.to_numeric(df.get("Voltage"), errors="coerce"),
            "Global_intensity": pd.to_numeric(df.get("Global_intensity"), errors="coerce"),
            "Sub_metering_1": pd.to_numeric(df.get("Sub_metering_1"), errors="coerce"),
            "Sub_metering_2": pd.to_numeric(df.get("Sub_metering_2"), errors="coerce"),
            "Sub_metering_3": pd.to_numeric(df.get("Sub_metering_3"), errors="coerce"),
        }
    )
    out["timestamp"] = out["Date"].astype(str) + " " + out["Time"].astype(str)
    out = out.drop(columns=["Date", "Time"])
    y = pd.to_numeric(df["Global_active_power"], errors="coerce").to_numpy(dtype=np.float64)
    return {
        "name": "HouseholdPowerConsumption",
        "X": out,
        "y": y,
        "task_type": "regression",
        "source": str(p),
    }


def read_csv_flexible(path: Path) -> pd.DataFrame:
    attempts = [
        {"sep": None, "engine": "python", "encoding": "utf-8"},
        {"sep": None, "engine": "python", "encoding": "latin-1"},
        {"sep": ";", "engine": "python", "encoding": "utf-8"},
        {"sep": ";", "engine": "python", "encoding": "latin-1"},
        {"sep": ",", "engine": "python", "encoding": "utf-8"},
        {"sep": ",", "engine": "python", "encoding": "latin-1"},
    ]
    last_err: Exception | None = None
    for kw in attempts:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path}: {last_err}")


def normalize_col(name: str) -> str:
    return str(name).strip().lower()


def infer_target_column(df: pd.DataFrame) -> str | None:
    norm_to_col = {normalize_col(c): str(c) for c in df.columns}
    for cand in TARGET_CANDIDATES:
        if cand in norm_to_col:
            return norm_to_col[cand]
    return None


def generic_load_dataset_from_csv(path: Path) -> dict | None:
    try:
        df = read_csv_flexible(path)
    except Exception as e:
        print(f"Warning: skipping {path}: {e}")
        return None
    if df.shape[1] < 3:
        return None
    target_col = infer_target_column(df)
    if target_col is None:
        print(f"Warning: no obvious target column in {path}; skipping.")
        return None
    out = df.copy()
    for col in list(out.columns):
        if normalize_col(col).startswith("unnamed"):
            out = out.drop(columns=[col])
    y = pd.to_numeric(out[target_col], errors="coerce").to_numpy(dtype=np.float64)
    X = out.drop(columns=[target_col])
    X = expand_datetime_columns(X)
    return {
        "name": path.stem,
        "X": X,
        "y": y,
        "task_type": "regression",
        "source": str(path),
    }


def expand_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s, errors="coerce")
        elif pd.api.types.is_object_dtype(s) or "date" in normalize_col(col) or "time" in normalize_col(col):
            dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        else:
            continue
        valid_ratio = float(np.mean(~dt.isna()))
        if valid_ratio < 0.8:
            continue
        out[f"{col}_year"] = dt.dt.year.astype(float)
        out[f"{col}_month"] = dt.dt.month.astype(float)
        out[f"{col}_day"] = dt.dt.day.astype(float)
        out[f"{col}_dayofweek"] = dt.dt.dayofweek.astype(float)
        out[f"{col}_hour"] = dt.dt.hour.astype(float)
        out = out.drop(columns=[col])
    return out


def load_sklearn_fallback_datasets() -> list[dict]:
    out: list[dict] = []
    ds = load_diabetes()
    X = pd.DataFrame(ds.data, columns=[str(c) for c in ds.feature_names])
    y = np.asarray(ds.target, dtype=np.float64)
    out.append(
        {
            "name": "Diabetes",
            "X": X,
            "y": y,
            "task_type": "regression",
            "source": "sklearn.load_diabetes",
        }
    )
    return out


def discover_real_datasets(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    roots = data_roots(args.data_dir)
    datasets: list[dict] = []
    skipped: list[dict] = []
    seen_names: set[str] = set()

    known_loaders = [
        load_gas_turbine_dataset,
        load_bike_dataset,
        load_seoul_dataset,
        load_energy_dataset,
        load_boston_dataset,
        load_ccpp_dataset,
        load_household_power_dataset,
    ]
    for loader in known_loaders:
        try:
            ds = loader(roots)
        except Exception as e:
            skipped.append({"dataset": loader.__name__, "reason": str(e)})
            continue
        if ds is None:
            continue
        if ds["name"] not in seen_names:
            seen_names.add(ds["name"])
            datasets.append(ds)

    generic_candidates: list[Path] = []
    for root in roots:
        generic_candidates.extend(sorted(root.rglob("*.csv")))
        generic_candidates.extend(sorted(root.rglob("*.tsv")))
    for path in generic_candidates:
        stem = path.stem.lower()
        if any(key in stem for key in ["gt_full", "day", "seoulbikedata", "energydata_complete"]):
            continue
        ds = generic_load_dataset_from_csv(path)
        if ds is None or ds["name"] in seen_names:
            continue
        seen_names.add(ds["name"])
        datasets.append(ds)

    if len(datasets) < 2:
        for ds in load_sklearn_fallback_datasets():
            if ds["name"] not in seen_names:
                seen_names.add(ds["name"])
                datasets.append(ds)

    max_datasets = MAX_DATASETS_FAST if args.fast else MAX_DATASETS_DEFAULT
    priority = {
        "GasTurbineNOx": 0,
        "BikeDemand": 1,
        "SeoulBikeSharingDemand": 2,
        "AppliancesEnergy": 3,
        "BostonHousing": 4,
        "CCPPPowerOutput": 5,
        "Metro_Interstate_Traffic_Volume": 6,
        "HouseholdPowerConsumption": 7,
        "Diabetes": 8,
    }
    datasets = sorted(datasets, key=lambda d: (priority.get(d["name"], 99), d["name"]))[:max_datasets]
    return datasets, skipped


def clean_dataset(ds: dict, seed: int, max_samples: int) -> tuple[dict | None, dict | None]:
    X = ds["X"]
    y = np.asarray(ds["y"], dtype=np.float64)
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()
    X_df = expand_datetime_columns(X_df)
    X_df = X_df.loc[:, [c for c in X_df.columns if not normalize_col(c).startswith("unnamed")]].copy()
    X_df = X_df.reset_index(drop=True)
    y = np.asarray(y, dtype=np.float64)
    keep = ~np.isnan(y)
    X_df = X_df.loc[keep].reset_index(drop=True)
    y = y[keep]

    # Drop extremely high-cardinality categorical columns to keep one-hot features sane.
    for col in list(X_df.columns):
        s = X_df[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            nunique = int(s.nunique(dropna=True))
            if nunique > 100 or nunique > max(20, int(0.2 * max(1, len(s)))):
                X_df = X_df.drop(columns=[col])

    if X_df.shape[0] < 200:
        return None, {"dataset": ds["name"], "reason": f"too few samples after cleaning ({X_df.shape[0]})"}

    # Keep datasets with at least two usable columns before preprocessing.
    usable_cols = [c for c in X_df.columns if not X_df[c].isna().all()]
    X_df = X_df[usable_cols].copy()
    if X_df.shape[1] < 2:
        return None, {"dataset": ds["name"], "reason": f"too few usable features after cleaning ({X_df.shape[1]})"}

    X_df, y = deterministic_subsample(X_df, y, max_samples=max_samples, seed=seed)
    out = {
        "name": ds["name"],
        "X": X_df.reset_index(drop=True),
        "y": np.asarray(y, dtype=np.float64),
        "task_type": ds["task_type"],
        "source": ds["source"],
        "samples": int(y.shape[0]),
        "raw_features": int(X_df.shape[1]),
    }
    return out, None


def make_repeated_splits(n_samples: int, n_splits: int, seed: int) -> list[dict]:
    out: list[dict] = []
    n_train = int(round(0.60 * n_samples))
    n_val = int(round(0.20 * n_samples))
    n_train = min(max(n_train, 1), n_samples - 2)
    n_val = min(max(n_val, 1), n_samples - n_train - 1)
    n_test = n_samples - n_train - n_val
    for split_id in range(n_splits):
        rng = np.random.default_rng(seed + 10007 * split_id + 17)
        perm = rng.permutation(n_samples)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]
        if test_idx.shape[0] != n_test:
            raise RuntimeError("Split construction failed.")
        out.append({"split_id": split_id, "train": train_idx, "val": val_idx, "test": test_idx})
    return out


def make_preprocessor(X_train_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X_train_df.columns if pd.api.types.is_numeric_dtype(X_train_df[c])]
    categorical_cols = [c for c in X_train_df.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def transform_dense(preprocessor: ColumnTransformer, X_df: pd.DataFrame) -> np.ndarray:
    arr = preprocessor.transform(X_df)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.asarray(arr, dtype=np.float64)


def compute_support_scores(X_query: np.ndarray, X_train: np.ndarray, k_support: int) -> np.ndarray:
    k_eff = min(k_support, X_train.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nbrs.fit(X_train)
    distances, _ = nbrs.kneighbors(X_query, return_distance=True)
    return np.asarray(distances[:, -1], dtype=np.float64)


def assign_bins(h_values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    return np.digitize(h_values, bin_edges[1:-1], right=False).astype(np.int64)


def make_support_bins(h_val: np.ndarray, n_bins: int) -> dict:
    max_bins = min(int(n_bins), int(h_val.shape[0]))
    for bins in range(max_bins, 1, -1):
        edges = np.quantile(h_val, np.linspace(0.0, 1.0, bins + 1))
        edges = np.asarray(edges, dtype=np.float64)
        for i in range(1, edges.shape[0]):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        bin_ids = assign_bins(h_val, edges)
        counts = np.bincount(bin_ids, minlength=bins)
        if int(np.min(counts)) > 0:
            centers = np.asarray(
                [float(np.median(h_val[bin_ids == b])) for b in range(bins)],
                dtype=np.float64,
            )
            return {
                "edges": edges,
                "bin_ids": bin_ids,
                "centers": centers,
                "counts": counts,
                "n_bins_eff": bins,
            }
    raise RuntimeError("Unable to construct nonempty support bins.")


def compute_support_contrast_summary(h_values: np.ndarray, eps: float = 1e-12) -> dict:
    h = np.asarray(h_values, dtype=np.float64)
    if h.ndim != 1 or h.size == 0:
        raise ValueError("Support contrast summary requires a nonempty 1D array.")

    positive = h[h > 0.0]
    used_min_positive_eps = positive.size == 0
    min_positive = float(np.min(positive)) if positive.size > 0 else float(eps)
    raw_support_contrast = float(np.max(h) / max(min_positive, float(eps)))

    q05 = float(np.quantile(h, 0.05))
    q10 = float(np.quantile(h, 0.10))
    q90 = float(np.quantile(h, 0.90))
    q95 = float(np.quantile(h, 0.95))

    robust_95_05_used_eps = q05 <= 0.0
    robust_90_10_used_eps = q10 <= 0.0
    robust_support_contrast_95_05 = float(q95 / max(q05, float(eps)))
    robust_support_contrast_90_10 = float(q90 / max(q10, float(eps)))

    return {
        "min_positive_support": min_positive,
        "used_min_positive_eps": bool(used_min_positive_eps),
        "q05_support": q05,
        "q10_support": q10,
        "q90_support": q90,
        "q95_support": q95,
        "robust_95_05_used_eps": bool(robust_95_05_used_eps),
        "robust_90_10_used_eps": bool(robust_90_10_used_eps),
        "raw_support_contrast": raw_support_contrast,
        "robust_support_contrast_95_05": robust_support_contrast_95_05,
        "robust_support_contrast_90_10": robust_support_contrast_90_10,
    }


def slope_loglog(h_centers: np.ndarray, mse_values: np.ndarray) -> float:
    mask = (h_centers > 0.0) & (mse_values > 0.0)
    if int(np.sum(mask)) < 2:
        return float("nan")
    beta, _ = np.polyfit(np.log(h_centers[mask]), np.log(mse_values[mask]), deg=1)
    return float(beta)


def compute_profile_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    h_values: np.ndarray,
    bin_edges: np.ndarray,
    weak_bins: Sequence[int],
    dense_bins: Sequence[int],
) -> dict:
    errors = (np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) ** 2
    y_var = float(np.var(np.asarray(y_true, dtype=np.float64)))
    y_var = max(y_var, 1e-12)
    nmse_errors = errors / y_var
    bin_ids = assign_bins(h_values, bin_edges)
    weak_mask = np.isin(bin_ids, np.asarray(weak_bins, dtype=np.int64))
    dense_mask = np.isin(bin_ids, np.asarray(dense_bins, dtype=np.int64))
    if not np.any(weak_mask) or not np.any(dense_mask):
        raise RuntimeError("Weak or dense region is empty.")
    global_mse = float(np.mean(errors))
    global_nmse = float(np.mean(nmse_errors))
    weak_mse = float(np.mean(errors[weak_mask]))
    weak_nmse = float(np.mean(nmse_errors[weak_mask]))
    dense_mse = float(np.mean(errors[dense_mask]))
    dense_nmse = float(np.mean(nmse_errors[dense_mask]))
    gap = float(weak_nmse / (dense_nmse + 1e-12))
    n_bins_eff = bin_edges.shape[0] - 1
    profile_rows: list[dict] = []
    for b in range(n_bins_eff):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        err_bin = errors[mask]
        nmse_bin = nmse_errors[mask]
        profile_rows.append(
            {
                "bin_id": int(b + 1),
                "h_bin_left": float(bin_edges[b]),
                "h_bin_right": float(bin_edges[b + 1]),
                "h_bin_center": float(np.median(h_values[mask])),
                "mse_bin_raw": float(np.mean(err_bin)),
                "mse_bin_norm": float(np.mean(nmse_bin)),
                "mse_bin_raw_se": float(np.std(err_bin, ddof=0) / math.sqrt(max(1, err_bin.shape[0]))),
                "mse_bin_norm_se": float(np.std(nmse_bin, ddof=0) / math.sqrt(max(1, nmse_bin.shape[0]))),
                "n_bin": int(mask.sum()),
            }
        )
    h_centers = np.asarray([row["h_bin_center"] for row in profile_rows], dtype=np.float64)
    mse_bins_norm = np.asarray([row["mse_bin_norm"] for row in profile_rows], dtype=np.float64)
    slope = slope_loglog(h_centers, np.maximum(mse_bins_norm, 1e-12))
    profile_var = float(np.var(mse_bins_norm))
    return {
        "global_mse": global_mse,
        "global_nmse": global_nmse,
        "weak_mse": weak_mse,
        "weak_nmse": weak_nmse,
        "dense_mse": dense_mse,
        "dense_nmse": dense_nmse,
        "gap": gap,
        "slope": slope,
        "profile_var": profile_var,
        "profile_rows": profile_rows,
        "n_bins_eff": int(n_bins_eff),
        "y_variance": y_var,
    }


def estimate_gamma_base(X_train: np.ndarray) -> float:
    n = X_train.shape[0]
    if n <= 1:
        return 1.0
    cap = min(n, 400)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=cap, replace=False)
    Z = X_train[idx]
    sq = np.sum(Z * Z, axis=1)[:, None]
    sqdist = np.maximum(sq + sq.T - 2.0 * Z @ Z.T, 0.0)
    vals = sqdist[np.triu_indices_from(sqdist, k=1)]
    vals = vals[vals > 0.0]
    if vals.size == 0:
        return 1.0
    med = float(np.median(vals))
    return 1.0 / max(med, 1e-6)


def generate_candidate_grid(model_family: str, X_train: np.ndarray, fast: bool) -> list[dict]:
    if model_family == "krr":
        gamma_base = estimate_gamma_base(X_train)
        gamma_mults = [0.5, 1.0, 2.0] if not fast else [0.5, 2.0]
        alphas = [1e-3, 1e-2, 1e-1] if not fast else [1e-3, 1e-1]
        out = []
        order = 0
        for gm in gamma_mults:
            for alpha in alphas:
                out.append(
                    {
                        "candidate_id": f"krr|gm={gm:g}|alpha={alpha:g}",
                        "hyperparams": {"gamma_multiplier": gm, "gamma": gamma_base * gm, "alpha": alpha},
                        "complexity_rank": order,
                    }
                )
                order += 1
        return out
    if model_family == "rf":
        depths = [6, None] if not fast else [6, None]
        leaves = [1, 5] if not fast else [1]
        out = []
        order = 0
        for depth in depths:
            for leaf in leaves:
                depth_label = "None" if depth is None else str(depth)
                out.append(
                    {
                        "candidate_id": f"rf|depth={depth_label}|leaf={leaf}",
                        "hyperparams": {"n_estimators": 100, "max_depth": depth, "min_samples_leaf": leaf},
                        "complexity_rank": order,
                    }
                )
                order += 1
        return out
    if model_family == "gbrt":
        depths = [2, 3]
        leaves = [1, 5] if not fast else [1]
        out = []
        order = 0
        for depth in depths:
            for leaf in leaves:
                out.append(
                    {
                        "candidate_id": f"gbrt|depth={depth}|leaf={leaf}",
                        "hyperparams": {
                            "n_estimators": 100,
                            "learning_rate": 0.1,
                            "max_depth": depth,
                            "min_samples_leaf": leaf,
                        },
                        "complexity_rank": order,
                    }
                )
                order += 1
        return out
    if model_family == "mlp":
        hidden_sizes = [(64,), (128,)] if not fast else [(64,)]
        alphas = [1e-4, 1e-3] if not fast else [1e-4, 1e-3]
        out = []
        order = 0
        for hidden in hidden_sizes:
            for alpha in alphas:
                hidden_label = "x".join(str(v) for v in hidden)
                out.append(
                    {
                        "candidate_id": f"mlp|hidden={hidden_label}|alpha={alpha:g}",
                        "hyperparams": {
                            "hidden_layer_sizes": hidden,
                            "alpha": alpha,
                            "learning_rate_init": 1e-3,
                            "max_iter": 400 if not fast else 250,
                        },
                        "complexity_rank": order,
                    }
                )
                order += 1
        return out
    raise ValueError(f"Unknown model family: {model_family}")


def fit_model_family_candidate(
    model_family: str,
    hyperparams: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
):
    if model_family == "krr":
        model = KernelRidge(
            kernel="rbf",
            gamma=float(hyperparams["gamma"]),
            alpha=float(hyperparams["alpha"]),
        )
    elif model_family == "rf":
        model = RandomForestRegressor(
            n_estimators=int(hyperparams["n_estimators"]),
            max_depth=hyperparams["max_depth"],
            min_samples_leaf=int(hyperparams["min_samples_leaf"]),
            random_state=random_state,
            n_jobs=1,
        )
    elif model_family == "gbrt":
        model = GradientBoostingRegressor(
            n_estimators=int(hyperparams["n_estimators"]),
            learning_rate=float(hyperparams["learning_rate"]),
            max_depth=int(hyperparams["max_depth"]),
            min_samples_leaf=int(hyperparams["min_samples_leaf"]),
            random_state=random_state,
        )
    elif model_family == "mlp":
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std < 1e-8:
            y_std = 1.0
        y_scaled = (np.asarray(y_train, dtype=np.float64) - y_mean) / y_std
        model = MLPRegressor(
            hidden_layer_sizes=tuple(hyperparams["hidden_layer_sizes"]),
            alpha=float(hyperparams["alpha"]),
            learning_rate_init=float(hyperparams["learning_rate_init"]),
            max_iter=int(hyperparams["max_iter"]),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if model_family == "mlp":
            model.fit(X_train, y_scaled)
            return {"wrapped_model": model, "y_mean": y_mean, "y_std": y_std, "family": "mlp"}
        model.fit(X_train, y_train)
    return model


def predict_model(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, dict) and model.get("family") == "mlp":
        pred_scaled = np.asarray(model["wrapped_model"].predict(X), dtype=np.float64)
        return pred_scaled * float(model["y_std"]) + float(model["y_mean"])
    return np.asarray(model.predict(X), dtype=np.float64)


def select_candidate(
    candidate_rows: Sequence[dict],
    metric_key: str,
    tol: float = 1e-12,
) -> dict:
    best_val = min(float(row[metric_key]) for row in candidate_rows)
    threshold = best_val + tol * max(1.0, abs(best_val))
    eligible = [row for row in candidate_rows if float(row[metric_key]) <= threshold]
    eligible = sorted(
        eligible,
        key=lambda row: (
            float(row["global_mse"]),
            int(row["complexity_rank"]),
            str(row["candidate_id"]),
        ),
    )
    return eligible[0]


def build_admissible_global_set(
    candidate_rows: Sequence[dict],
    tau: float,
    tol: float = 1e-12,
) -> tuple[dict, list[dict], bool]:
    theta_global = select_candidate(candidate_rows, "global_mse", tol=tol)
    baseline = float(theta_global["global_mse"])
    threshold = (1.0 + float(tau)) * baseline + tol * max(1.0, abs(baseline))
    admissible = [row for row in candidate_rows if float(row["global_mse"]) <= threshold]
    if not admissible:
        admissible = [theta_global]
    return theta_global, admissible, len(admissible) < len(candidate_rows)


def select_constrained_profile_candidate(
    admissible_rows: Sequence[dict],
    theta_global: dict,
    eta: float,
    zeta: float,
    eps: float = 1e-12,
) -> dict:
    weak_ref = max(float(theta_global["weak_mse"]), eps)
    gap_ref = max(float(theta_global["gap"]), eps)
    slope_ref = max(float(theta_global["slope"]), 0.0) + eps
    scored = []
    for row in admissible_rows:
        slope_pos = max(float(row["slope"]), 0.0)
        score = (
            float(row["weak_mse"]) / weak_ref
            + float(eta) * float(row["gap"]) / gap_ref
            + float(zeta) * slope_pos / slope_ref
        )
        scored.append({**row, "profile_score": score})
    return select_candidate(scored, "profile_score")


def compute_profile_score(
    row: dict,
    theta_global: dict,
    eta: float,
    zeta: float,
    eps: float = 1e-12,
) -> float:
    weak_ref = max(float(theta_global["weak_mse"]), eps)
    gap_ref = max(float(theta_global["gap"]), eps)
    slope_ref = max(float(theta_global["slope"]), 0.0) + eps
    slope_pos = max(float(row["slope"]), 0.0)
    return float(
        float(row["weak_mse"]) / weak_ref
        + float(eta) * float(row["gap"]) / gap_ref
        + float(zeta) * slope_pos / slope_ref
    )


def attach_profile_scores(
    candidate_rows: Sequence[dict],
    theta_global: dict,
    eta: float,
    zeta: float,
    eps: float = 1e-12,
) -> list[dict]:
    out: list[dict] = []
    for row in candidate_rows:
        out.append(
            {
                **row,
                "profile_score": compute_profile_score(
                    row=row,
                    theta_global=theta_global,
                    eta=eta,
                    zeta=zeta,
                    eps=eps,
                ),
                "pos_slope": max(float(row["slope"]), 0.0),
            }
        )
    return out


def aggregate_candidate_points(
    candidate_rows: Sequence[dict],
    dataset: str,
    model_family: str,
    split: str,
) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        if row["dataset"] == dataset and row["model_family"] == model_family and row["split"] == split:
            groups[str(row["candidate_id"])].append(row)
    out: list[dict] = []
    for candidate_id, rows in groups.items():
        out.append(
            {
                "dataset": dataset,
                "model_family": model_family,
                "candidate_id": candidate_id,
                "global_nmse_mean": float(np.mean([float(r["global_nmse"]) for r in rows])),
                "weak_nmse_mean": float(np.mean([float(r["weak_nmse"]) for r in rows])),
                "dense_nmse_mean": float(np.mean([float(r["dense_nmse"]) for r in rows])),
                "gap_mean": float(np.mean([float(r["gap"]) for r in rows])),
                "slope_mean": float(np.nanmean([float(r["slope"]) for r in rows])),
            }
        )
    return sorted(out, key=lambda row: (row["global_nmse_mean"], row["weak_nmse_mean"], row["candidate_id"]))


def percentage_change(new: float, base: float) -> float:
    return 100.0 * (new - base) / max(abs(base), 1e-12)


def percentage_delta(profile_value: float, global_value: float, eps_delta: float = 1e-12) -> float:
    return 100.0 * (float(profile_value) - float(global_value)) / (abs(float(global_value)) + float(eps_delta))


def rank_map_from_values(items: Sequence[tuple[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    series = pd.Series(
        [float(value) for _, value in items],
        index=[str(key) for key, _ in items],
        dtype=np.float64,
    )
    ranked = series.rank(method="average", ascending=True)
    return {str(idx): float(val) for idx, val in ranked.items()}


def spearman_rank_correlation(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return float("nan")
    rank_a = pd.Series(np.asarray(values_a, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    rank_b = pd.Series(np.asarray(values_b, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(rank_a) < 1e-12 or np.std(rank_b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def summarize_pair_group(group_name: str, rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "group": group_name,
            "n": 0,
            "changed_fraction": 0.0,
            "fraction_gap_improved": 0.0,
            "fraction_pos_slope_improved": 0.0,
            "fraction_weak_improved": 0.0,
            "fraction_global_not_worse_1pct": 0.0,
            "mean_delta_global_pct": 0.0,
            "median_delta_global_pct": 0.0,
            "mean_delta_weak_pct": 0.0,
            "median_delta_weak_pct": 0.0,
            "mean_delta_gap_pct": 0.0,
            "median_delta_gap_pct": 0.0,
            "mean_delta_pos_slope_pct": 0.0,
            "median_delta_pos_slope_pct": 0.0,
        }
    delta_global = np.asarray([float(row["delta_global_pct"]) for row in rows], dtype=np.float64)
    delta_weak = np.asarray([float(row["delta_weak_pct"]) for row in rows], dtype=np.float64)
    delta_gap = np.asarray([float(row["delta_gap_pct"]) for row in rows], dtype=np.float64)
    delta_pos_slope = np.asarray([float(row["delta_pos_slope_pct"]) for row in rows], dtype=np.float64)
    changed = np.asarray([bool(row["selection_changed"]) for row in rows], dtype=bool)
    return {
        "group": group_name,
        "n": int(len(rows)),
        "changed_fraction": float(np.mean(changed)),
        "fraction_gap_improved": float(np.mean(delta_gap < 0.0)),
        "fraction_pos_slope_improved": float(np.mean(delta_pos_slope < 0.0)),
        "fraction_weak_improved": float(np.mean(delta_weak < 0.0)),
        "fraction_global_not_worse_1pct": float(np.mean(delta_global <= 1.0)),
        "mean_delta_global_pct": float(np.mean(delta_global)),
        "median_delta_global_pct": float(np.median(delta_global)),
        "mean_delta_weak_pct": float(np.mean(delta_weak)),
        "median_delta_weak_pct": float(np.median(delta_weak)),
        "mean_delta_gap_pct": float(np.mean(delta_gap)),
        "median_delta_gap_pct": float(np.median(delta_gap)),
        "mean_delta_pos_slope_pct": float(np.mean(delta_pos_slope)),
        "median_delta_pos_slope_pct": float(np.median(delta_pos_slope)),
    }


def summarize_support_stratum(stratum: str, rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "stratum": stratum,
            "n": 0,
            "median_robust_contrast": 0.0,
            "changed_fraction": 0.0,
            "fraction_gap_improved": 0.0,
            "fraction_pos_slope_improved": 0.0,
            "fraction_weak_improved": 0.0,
            "mean_delta_global_pct": 0.0,
            "median_delta_global_pct": 0.0,
            "mean_delta_gap_pct": 0.0,
            "median_delta_gap_pct": 0.0,
            "mean_delta_pos_slope_pct": 0.0,
            "median_delta_pos_slope_pct": 0.0,
        }
    contrast = np.asarray([float(row["robust_support_contrast"]) for row in rows], dtype=np.float64)
    delta_global = np.asarray([float(row["delta_global_pct"]) for row in rows], dtype=np.float64)
    delta_gap = np.asarray([float(row["delta_gap_pct"]) for row in rows], dtype=np.float64)
    delta_weak = np.asarray([float(row["delta_weak_pct"]) for row in rows], dtype=np.float64)
    delta_pos_slope = np.asarray([float(row["delta_pos_slope_pct"]) for row in rows], dtype=np.float64)
    changed = np.asarray([bool(row["selection_changed"]) for row in rows], dtype=bool)
    return {
        "stratum": stratum,
        "n": int(len(rows)),
        "median_robust_contrast": float(np.median(contrast)),
        "changed_fraction": float(np.mean(changed)),
        "fraction_gap_improved": float(np.mean(delta_gap < 0.0)),
        "fraction_pos_slope_improved": float(np.mean(delta_pos_slope < 0.0)),
        "fraction_weak_improved": float(np.mean(delta_weak < 0.0)),
        "mean_delta_global_pct": float(np.mean(delta_global)),
        "median_delta_global_pct": float(np.median(delta_global)),
        "mean_delta_gap_pct": float(np.mean(delta_gap)),
        "median_delta_gap_pct": float(np.median(delta_gap)),
        "mean_delta_pos_slope_pct": float(np.mean(delta_pos_slope)),
        "median_delta_pos_slope_pct": float(np.median(delta_pos_slope)),
    }


def summarize_disagreement_group(group_name: str, rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "group": group_name,
            "n": 0,
            "median_global_rank_of_profile_selected": 0.0,
            "median_profile_rank_of_global_selected": 0.0,
            "median_validation_global_cost_pct": 0.0,
            "median_validation_profile_score_gain_pct": 0.0,
            "median_spearman_global_vs_profile": 0.0,
            "fraction_selection_changed": 0.0,
            "fraction_profile_selected_within_top_5_global_rank": 0.0,
            "fraction_global_selected_outside_top_5_profile_rank": 0.0,
        }
    return {
        "group": group_name,
        "n": int(len(rows)),
        "median_global_rank_of_profile_selected": float(np.median([float(row["global_rank_of_profile_selected"]) for row in rows])),
        "median_profile_rank_of_global_selected": float(np.median([float(row["profile_rank_of_global_selected"]) for row in rows])),
        "median_validation_global_cost_pct": float(np.median([float(row["validation_global_cost_pct"]) for row in rows])),
        "median_validation_profile_score_gain_pct": float(np.median([float(row["validation_profile_score_gain_pct"]) for row in rows])),
        "median_spearman_global_vs_profile": float(np.nanmedian([float(row["spearman_global_vs_profile"]) for row in rows])),
        "fraction_selection_changed": float(np.mean([bool(row["selection_changed"]) for row in rows])),
        "fraction_profile_selected_within_top_5_global_rank": float(np.mean([float(row["global_rank_of_profile_selected"]) <= 5.0 for row in rows])),
        "fraction_global_selected_outside_top_5_profile_rank": float(np.mean([float(row["profile_rank_of_global_selected"]) > 5.0 for row in rows])),
    }


def summarize_changed_selection_group(group_name: str, rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "group": group_name,
            "n": 0,
            "gap_improved_fraction": 0.0,
            "positive_slope_improved_fraction": 0.0,
            "weak_nmse_improved_fraction": 0.0,
            "profile_var_improved_fraction": 0.0,
            "global_nmse_improved_fraction": 0.0,
            "global_nmse_within_1pct_fraction": 0.0,
            "global_nmse_within_2pct_fraction": 0.0,
            "median_delta_gap_pct": 0.0,
            "median_delta_positive_slope_pct": 0.0,
            "median_delta_weak_nmse_pct": 0.0,
            "median_delta_global_nmse_pct": 0.0,
        }
    delta_global = np.asarray([float(row["delta_global_nmse_pct"]) for row in rows], dtype=np.float64)
    delta_weak = np.asarray([float(row["delta_weak_nmse_pct"]) for row in rows], dtype=np.float64)
    delta_gap = np.asarray([float(row["delta_gap_pct"]) for row in rows], dtype=np.float64)
    delta_pos_slope = np.asarray([float(row["delta_positive_slope_pct"]) for row in rows], dtype=np.float64)
    delta_profile_var = np.asarray([float(row["delta_profile_var_pct"]) for row in rows], dtype=np.float64)
    return {
        "group": group_name,
        "n": int(len(rows)),
        "gap_improved_fraction": float(np.mean(delta_gap < 0.0)),
        "positive_slope_improved_fraction": float(np.mean(delta_pos_slope < 0.0)),
        "weak_nmse_improved_fraction": float(np.mean(delta_weak < 0.0)),
        "profile_var_improved_fraction": float(np.mean(delta_profile_var < 0.0)),
        "global_nmse_improved_fraction": float(np.mean(delta_global < 0.0)),
        "global_nmse_within_1pct_fraction": float(np.mean(delta_global <= 1.0)),
        "global_nmse_within_2pct_fraction": float(np.mean(delta_global <= 2.0)),
        "median_delta_gap_pct": float(np.median(delta_gap)),
        "median_delta_positive_slope_pct": float(np.median(delta_pos_slope)),
        "median_delta_weak_nmse_pct": float(np.median(delta_weak)),
        "median_delta_global_nmse_pct": float(np.median(delta_global)),
    }


def aggregate_stratum_rows(level: str, stratum: str, rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "level": level,
            "group": stratum,
            "n": 0,
            "median_robust_contrast": 0.0,
            "fraction_changed_from_global": 0.0,
            "fraction_gap_improved": 0.0,
            "fraction_positive_slope_improved": 0.0,
            "fraction_weak_nmse_improved": 0.0,
            "fraction_global_nmse_within_1pct": 0.0,
            "fraction_global_nmse_within_2pct": 0.0,
            "median_delta_gap_pct": 0.0,
            "median_delta_positive_slope_pct": 0.0,
            "median_delta_weak_nmse_pct": 0.0,
            "median_delta_global_nmse_pct": 0.0,
        }
    return {
        "level": level,
        "group": stratum,
        "n": int(len(rows)),
        "median_robust_contrast": float(np.median([float(row["robust_support_contrast_median"]) for row in rows])),
        "fraction_changed_from_global": float(np.mean([float(row["fraction_changed_from_global"]) for row in rows])),
        "fraction_gap_improved": float(np.mean([float(row["fraction_gap_improved"]) for row in rows])),
        "fraction_positive_slope_improved": float(np.mean([float(row["fraction_positive_slope_improved"]) for row in rows])),
        "fraction_weak_nmse_improved": float(np.mean([float(row["fraction_weak_nmse_improved"]) for row in rows])),
        "fraction_global_nmse_within_1pct": float(np.mean([float(row["fraction_global_nmse_within_1pct"]) for row in rows])),
        "fraction_global_nmse_within_2pct": float(np.mean([float(row["fraction_global_nmse_within_2pct"]) for row in rows])),
        "median_delta_gap_pct": float(np.median([float(row["median_delta_gap_pct"]) for row in rows])),
        "median_delta_positive_slope_pct": float(np.median([float(row["median_delta_positive_slope_pct"]) for row in rows])),
        "median_delta_weak_nmse_pct": float(np.median([float(row["median_delta_weak_nmse_pct"]) for row in rows])),
        "median_delta_global_nmse_pct": float(np.median([float(row["median_delta_global_nmse_pct"]) for row in rows])),
    }


def assign_terciles(values: Sequence[float], labels: Sequence[str] = ("Low", "Medium", "High")) -> list[str]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    q1, q2 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    out: list[str] = []
    for value in arr:
        if value <= q1:
            out.append(str(labels[0]))
        elif value <= q2:
            out.append(str(labels[1]))
        else:
            out.append(str(labels[2]))
    return out


def choose_representative_cases(
    selection_pair_rows: Sequence[dict],
    split_profiles_rows: Sequence[dict],
) -> list[dict]:
    grouped_profiles: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in split_profiles_rows:
        grouped_profiles[(row["dataset"], row["model_family"], int(row["split_id"]))].append(row)
    candidates: list[dict] = []
    for row in selection_pair_rows:
        key = (row["dataset"], row["model_family"], int(row["split_id"]))
        profile_rows = grouped_profiles.get(key, [])
        global_rows = sorted(
            [r for r in profile_rows if r["selection_rule_short"] == "global"],
            key=lambda r: int(r["support_bin_id"]),
        )
        profile_rows_cur = sorted(
            [r for r in profile_rows if r["selection_rule_short"] == "profile-aware"],
            key=lambda r: int(r["support_bin_id"]),
        )
        if not global_rows or not profile_rows_cur:
            continue
        weak_ids = sorted({int(r["support_bin_id"]) for r in global_rows})[-2:]
        global_weak = [r for r in global_rows if int(r["support_bin_id"]) in weak_ids]
        profile_weak = [r for r in profile_rows_cur if int(r["support_bin_id"]) in weak_ids]
        if not global_weak or not profile_weak:
            continue
        weak_tail_global = float(np.mean([float(r["bin_nmse"]) for r in global_weak]))
        weak_tail_profile = float(np.mean([float(r["bin_nmse"]) for r in profile_weak]))
        weak_bin_count = int(np.sum([int(r["bin_count"]) for r in global_weak]))
        weak_tail_improvement_pct = 100.0 * (
            weak_tail_global - weak_tail_profile
        ) / (abs(weak_tail_global) + 1e-12)
        improvement_gap_pct = max(0.0, -float(row["delta_gap_pct"]))
        improvement_pos_slope_pct = max(0.0, -float(row["delta_positive_slope_pct"]))
        improvement_weak_tail_pct = max(0.0, weak_tail_improvement_pct)
        penalty_for_global_nmse_increase = max(0.0, float(row["delta_global_nmse_pct"]))
        score = (
            0.4 * improvement_gap_pct
            + 0.4 * improvement_pos_slope_pct
            + 0.2 * improvement_weak_tail_pct
            - penalty_for_global_nmse_increase
        )
        if (
            bool(row["changed_selection"])
            and float(row["delta_global_nmse_pct"]) <= 2.0
            and (
                float(row["delta_gap_pct"]) <= -10.0
                or float(row["delta_positive_slope_pct"]) <= -10.0
            )
            and weak_bin_count >= 20
            and weak_tail_profile < weak_tail_global
        ):
            candidates.append(
                {
                    **row,
                    "weak_tail_global_nmse": weak_tail_global,
                    "weak_tail_profile_nmse": weak_tail_profile,
                    "weak_tail_improvement_pct": weak_tail_improvement_pct,
                    "weak_bin_count": weak_bin_count,
                    "representative_score": score,
                    "global_nmse_budget_label": "<=1%" if float(row["delta_global_nmse_pct"]) <= 1.0 else "<=2%",
                }
            )
    candidates = sorted(
        candidates,
        key=lambda row: (
            0 if abs(float(row["delta_global_nmse_pct"])) <= 1.0 else (1 if abs(float(row["delta_global_nmse_pct"])) <= 2.0 else 2),
            -float(row["representative_score"]),
            abs(float(row["delta_global_nmse_pct"])),
            str(row["dataset"]),
            str(row["model_family"]),
            int(row["split_id"]),
        ),
    )
    if not candidates:
        return []
    chosen: list[dict] = [candidates[0]]
    used_keys = {(chosen[0]["dataset"], chosen[0]["model_family"], int(chosen[0]["split_id"]))}
    for row in candidates[1:]:
        key = (row["dataset"], row["model_family"], int(row["split_id"]))
        if key in used_keys:
            continue
        chosen.append(row)
        used_keys.add(key)
        if len(chosen) >= 4:
            break
    return chosen


def choose_representative_pairs(summary_rows: Sequence[dict]) -> list[dict]:
    if not summary_rows:
        return []
    improved = sorted(
        summary_rows,
        key=lambda row: (
            float(row["delta_gap_pct"]),
            float(row["delta_weak_pct"]),
            float(row["delta_global_pct"]),
        ),
    )
    best = improved[0]
    remaining = [
        row
        for row in summary_rows
        if not (row["dataset"] == best["dataset"] and row["model_family"] == best["model_family"])
    ]
    if not remaining:
        return [best]
    tradeoff = sorted(
        remaining,
        key=lambda row: (
            abs(float(row["delta_gap_pct"])) + abs(float(row["delta_weak_pct"])) + abs(float(row["delta_global_pct"])),
            abs(float(row["delta_slope_pct"])),
        ),
    )[0]
    return [best, tradeoff]


def make_realdata_profiles_figure(
    figures_dir: Path,
    profiles_selected: Sequence[dict],
    representative_pairs: Sequence[dict],
) -> None:
    n = max(1, len(representative_pairs))
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    style_map = make_style_map(RULE_DISPLAY_ORDER)
    for ax, rep in zip(axes, representative_pairs):
        dataset = rep["dataset"]
        family = rep["model_family"]
        for rule in RULE_DISPLAY_ORDER:
            cur = [
                row
                for row in profiles_selected
                if row["dataset"] == dataset and row["model_family"] == family and row["selection_rule"] == rule
            ]
            if not cur:
                continue
            bin_ids = sorted(set(int(row["bin_id"]) for row in cur))
            xs, ys, yerr = [], [], []
            for bin_id in bin_ids:
                rows_bin = [row for row in cur if int(row["bin_id"]) == bin_id]
                xs.append(float(np.mean([float(row["h_bin_center"]) for row in rows_bin])))
                vals = np.asarray([float(row["nmse_bin"]) for row in rows_bin], dtype=np.float64)
                ys.append(float(np.mean(vals)))
                yerr.append(float(np.std(vals, ddof=0) / math.sqrt(max(1, vals.shape[0]))))
            x_arr = np.asarray(xs, dtype=np.float64)
            y_arr = np.maximum(np.asarray(ys, dtype=np.float64), 1e-12)
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
                label=rule.replace(" selection", ""),
            )
            ax.fill_between(
                x_arr,
                np.maximum(y_arr - err_arr, 1e-12),
                y_arr + err_arr,
                color=style["color"],
                alpha=0.12,
            )
        ax.set_title(f"{dataset} ({MODEL_FAMILY_LABELS.get(family, family)})")
        ax.set_xlabel("Support radius h")
        ax.set_ylabel("Test NMSE profile")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.18, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    for idx, ax in enumerate(axes, start=1):
        save_axes_group_panel(fig, [ax], figures_dir / f"exp8_realdata_profiles_panel_{idx}.pdf")
    save_legend_figure(
        handles,
        labels,
        figures_dir / "exp8_realdata_profiles_legend.pdf",
        ncol=2,
    )
    plt.close(fig)


def make_model_agnostic_pareto_figure(
    figures_dir: Path,
    candidate_rows: Sequence[dict],
    selection_rows: Sequence[dict],
    representative_pairs: Sequence[dict],
    tau: float,
) -> None:
    pairs = representative_pairs if representative_pairs else []
    if not pairs:
        return
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    family_colors = {"krr": "#2f4b7c", "rf": "#54a24b", "gbrt": "#f58518", "mlp": "#7f3c8d"}
    rule_markers = {
        "Global-MSE selection": "o",
        "Constrained profile-aware selection": "P",
    }
    for ax, rep in zip(axes, pairs):
        dataset = rep["dataset"]
        family = rep["model_family"]
        points = aggregate_candidate_points(candidate_rows, dataset=dataset, model_family=family, split="val")
        if not points:
            continue
        ax.scatter(
            [row["global_nmse_mean"] for row in points],
            [row["weak_nmse_mean"] for row in points],
            color=family_colors.get(family, "#666666"),
            alpha=0.9,
            s=42,
        )
        point_map = {row["candidate_id"]: row for row in points}
        for rule in RULE_DISPLAY_ORDER:
            cur = [
                row["candidate_id"]
                for row in selection_rows
                if row["dataset"] == dataset and row["model_family"] == family and row["selection_rule"] == rule
            ]
            if not cur:
                continue
            candidate_id = Counter(cur).most_common(1)[0][0]
            if candidate_id not in point_map:
                continue
            row = point_map[candidate_id]
            ax.scatter(
                [row["global_nmse_mean"]],
                [row["weak_nmse_mean"]],
                marker=rule_markers[rule],
                s=150,
                facecolors="none",
                edgecolors="#111111",
                linewidths=2.0,
            )
        global_rows = [
            row
            for row in selection_rows
            if row["dataset"] == dataset
            and row["model_family"] == family
            and row["selection_rule"] == "Global-MSE selection"
        ]
        if global_rows:
            threshold = (1.0 + float(tau)) * float(
                np.mean([float(row["validation_global_nmse"]) for row in global_rows])
            )
            ax.axvline(threshold, color="#666666", linestyle="--", linewidth=1.4, alpha=0.8)
        ax.set_title(f"{dataset} ({MODEL_FAMILY_LABELS.get(family, family)})")
        ax.set_xlabel("Validation global NMSE")
        ax.set_ylabel("Validation weak NMSE")
        ax.grid(alpha=0.18)
    rule_handles = [
        Line2D(
            [0],
            [0],
            marker=rule_markers[r],
            linestyle="",
            color="#111111",
            label=r.replace(" selection", ""),
            markersize=8,
            markerfacecolor="none",
        )
        for r in RULE_DISPLAY_ORDER
    ]
    rule_handles.append(
        Line2D([0], [0], linestyle="--", color="#666666", label="Admissible global-risk threshold")
    )
    candidate_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color="#666666",
        label="Candidate settings",
        markersize=6,
    )
    legend_handles = [candidate_handle, *rule_handles]
    legend_labels = [handle.get_label() for handle in legend_handles]
    for idx, ax in enumerate(axes, start=1):
        save_axes_group_panel(fig, [ax], figures_dir / f"exp8_model_agnostic_pareto_panel_{idx}.pdf")
    save_legend_figure(
        legend_handles,
        legend_labels,
        figures_dir / "exp8_model_agnostic_pareto_legend.pdf",
        ncol=3,
    )
    plt.close(fig)


def save_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_dataset_table_tex(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Dataset & Samples & Features & Raw contrast & Robust contrast & Min bin size \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {int(row['samples'])} & {int(row['features'])} & "
            f"{float(row['raw_support_contrast_median']):.2f} & "
            f"{float(row['robust_support_contrast_95_05_median']):.2f} & "
            f"{int(row['min_bin_size'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_selection_table_tex(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Dataset & Selection rule & Global NMSE & Weak NMSE & Dense NMSE & Gap & Slope & Profile Var. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['selection_rule']} & "
            f"{float(row['global_nmse']):.4f} & {float(row['weak_nmse']):.4f} & "
            f"{float(row['dense_nmse']):.4f} & {float(row['gap']):.3f} & "
            f"{float(row['slope']):.3f} & {float(row['profile_var']):.6f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_model_agnostic_summary_tex(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Model family & Selection changed? & Delta Global (\\%) & Delta Weak (\\%) & Delta Gap (\\%) & Delta Slope (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['model_family_label']} & {row['selection_changed']} & "
            f"{float(row['delta_global_pct']):+.1f}\\% & "
            f"{float(row['delta_weak_pct']):+.1f}\\% & "
            f"{float(row['delta_gap_pct']):+.1f}\\% & "
            f"{float(row['delta_slope_pct']):+.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_aggregate_summary_tex(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Criterion & Fraction improved & Mean change & Median change \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['criterion']} & {float(row['fraction_improved']):.2f} & "
            f"{float(row['mean_change']):+.1f}\\% & {float(row['median_change']):+.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_decision_change_usefulness_tex(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Group & $n$ & Gap improved & Pos. slope improved & Weak NMSE improved & Global NMSE $\\leq$ +1\\% & Med. $\\Delta$ Gap (\\%) & Med. $\\Delta$ Pos. slope (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['group']} & {int(row['n'])} & "
            f"{float(row['fraction_gap_improved']):.2f} & "
            f"{float(row['fraction_pos_slope_improved']):.2f} & "
            f"{float(row['fraction_weak_improved']):.2f} & "
            f"{float(row['fraction_global_not_worse_1pct']):.2f} & "
            f"{float(row['median_delta_gap_pct']):+.1f}\\% & "
            f"{float(row['median_delta_pos_slope_pct']):+.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_support_heterogeneity_stratification_tex(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Stratum & $n$ & Med. contrast & Changed frac. & Gap improved & Pos. slope improved & Weak improved & Med. $\\Delta$ Gap (\\%) & Med. $\\Delta$ Pos. slope (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        stratum = row.get("stratum", row.get("group", ""))
        lines.append(
            f"{stratum} & {int(row['n'])} & "
            f"{float(row['median_robust_contrast']):.2f} & "
            f"{float(row.get('changed_fraction', row.get('fraction_changed_from_global', 0.0))):.2f} & "
            f"{float(row.get('fraction_gap_improved', 0.0)):.2f} & "
            f"{float(row.get('fraction_pos_slope_improved', row.get('fraction_positive_slope_improved', 0.0))):.2f} & "
            f"{float(row.get('fraction_weak_improved', row.get('fraction_weak_nmse_improved', 0.0))):.2f} & "
            f"{float(row['median_delta_gap_pct']):+.1f}\\% & "
            f"{float(row.get('median_delta_pos_slope_pct', row.get('median_delta_positive_slope_pct', 0.0))):+.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_profile_global_disagreement_tex(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Group & $n$ & Med. global rank of profile sel. & Med. profile rank of global sel. & Med. val. global cost (\\%) & Med. profile-score gain (\\%) & Med. Spearman & Changed frac. & Profile sel. in top-5 global \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['group']} & {int(row['n'])} & "
            f"{float(row['median_global_rank_of_profile_selected']):.1f} & "
            f"{float(row['median_profile_rank_of_global_selected']):.1f} & "
            f"{float(row['median_validation_global_cost_pct']):+.1f}\\% & "
            f"{float(row['median_validation_profile_score_gain_pct']):+.1f}\\% & "
            f"{float(row['median_spearman_global_vs_profile']):.2f} & "
            f"{float(row['fraction_selection_changed']):.2f} & "
            f"{float(row['fraction_profile_selected_within_top_5_global_rank']):.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_changed_selection_conditional_tex(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{lcccccccccc}",
        "\\toprule",
        "Group & $n$ & Gap imp. & Pos. slope imp. & Weak NMSE imp. & Prof. var. imp. & Global NMSE imp. & Global $\\leq$ +1\\% & Global $\\leq$ +2\\% & Med. $\\Delta$ Gap (\\%) & Med. $\\Delta$ Pos. slope (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['group']} & {int(row['n'])} & "
            f"{float(row['gap_improved_fraction']):.2f} & "
            f"{float(row['positive_slope_improved_fraction']):.2f} & "
            f"{float(row['weak_nmse_improved_fraction']):.2f} & "
            f"{float(row['profile_var_improved_fraction']):.2f} & "
            f"{float(row['global_nmse_improved_fraction']):.2f} & "
            f"{float(row['global_nmse_within_1pct_fraction']):.2f} & "
            f"{float(row['global_nmse_within_2pct_fraction']):.2f} & "
            f"{float(row['median_delta_gap_pct']):+.1f}\\% & "
            f"{float(row['median_delta_positive_slope_pct']):+.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_support_heterogeneity_pair_split_tex(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{lcccccccccc}",
        "\\toprule",
        "Group & $n$ & Med. contrast & Changed frac. & Gap imp. & Pos. slope imp. & Weak imp. & Global $\\leq$ +1\\% & Global $\\leq$ +2\\% & Med. $\\Delta$ Gap (\\%) & Med. $\\Delta$ Pos. slope (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['group']} & {int(row['n'])} & "
            f"{float(row['median_robust_contrast']):.2f} & "
            f"{float(row['fraction_changed_from_global']):.2f} & "
            f"{float(row['fraction_gap_improved']):.2f} & "
            f"{float(row['fraction_positive_slope_improved']):.2f} & "
            f"{float(row['fraction_weak_nmse_improved']):.2f} & "
            f"{float(row['fraction_global_nmse_within_1pct']):.2f} & "
            f"{float(row['fraction_global_nmse_within_2pct']):.2f} & "
            f"{float(row['median_delta_gap_pct']):+.1f}\\% & "
            f"{float(row['median_delta_positive_slope_pct']):+.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def make_representative_profile_figure(
    path: Path,
    split_profiles_rows: Sequence[dict],
    representative_case: dict,
) -> None:
    dataset = representative_case["dataset"]
    family = representative_case["model_family"]
    split_id = int(representative_case["split_id"])
    cur = [
        row for row in split_profiles_rows
        if row["dataset"] == dataset
        and row["model_family"] == family
        and int(row["split_id"]) == split_id
    ]
    global_rows = sorted(
        [row for row in cur if row["selection_rule_short"] == "global"],
        key=lambda row: int(row["support_bin_id"]),
    )
    profile_rows = sorted(
        [row for row in cur if row["selection_rule_short"] == "profile-aware"],
        key=lambda row: int(row["support_bin_id"]),
    )
    if not global_rows or not profile_rows:
        return
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
        constrained_layout=True,
    )
    style_map = {
        "global": {"color": "#2f4b7c", "marker": "o", "linestyle": "-"},
        "profile-aware": {"color": "#f58518", "marker": "s", "linestyle": "--"},
    }
    for key, rows in (("global", global_rows), ("profile-aware", profile_rows)):
        x = np.asarray([float(r["support_bin_center"]) for r in rows], dtype=np.float64)
        y = np.asarray([float(r["bin_nmse"]) for r in rows], dtype=np.float64)
        se = np.asarray([float(r["bin_standard_error"]) for r in rows], dtype=np.float64)
        counts = np.asarray([float(r["bin_count"]) for r in rows], dtype=np.float64)
        style = style_map[key]
        axes[0].plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=4.5,
            label="Global-selected" if key == "global" else "Profile-aware-selected",
        )
        axes[0].fill_between(x, np.maximum(y - se, 1e-12), y + se, color=style["color"], alpha=0.14)
        axes[1].plot(
            x,
            counts,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=4.0,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Test binwise NMSE")
    axes[0].grid(alpha=0.18, which="both")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Support bin center h")
    axes[1].set_ylabel("Bin count")
    axes[1].grid(alpha=0.18, which="both")
    axes[0].legend(frameon=False, loc="best")
    title = (
        f"{dataset} ({MODEL_FAMILY_LABELS.get(family, family)}), split {split_id}\n"
        f"Global NMSE {representative_case['test_global_nmse_global_selected']:.3f} vs {representative_case['test_global_nmse_profile_selected']:.3f}; "
        f"Gap {representative_case['test_gap_global_selected']:.3f} vs {representative_case['test_gap_profile_selected']:.3f}; "
        f"Slope+ {representative_case['test_pos_slope_global_selected']:.3f} vs {representative_case['test_pos_slope_profile_selected']:.3f}"
    )
    axes[0].set_title(title, fontsize=10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_representative_profiles_supplement_figure(
    figures_dir: Path,
    split_profiles_rows: Sequence[dict],
    representative_cases: Sequence[dict],
) -> None:
    if not representative_cases:
        return
    cases = list(representative_cases[:3])
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.3), constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]
    style_map = {
        "global": {"color": "#2f4b7c", "marker": "o", "linestyle": "-"},
        "profile-aware": {"color": "#f58518", "marker": "s", "linestyle": "--"},
    }
    for ax, case in zip(axes, cases):
        dataset = case["dataset"]
        family = case["model_family"]
        split_id = int(case["split_id"])
        cur = [
            row for row in split_profiles_rows
            if row["dataset"] == dataset
            and row["model_family"] == family
            and int(row["split_id"]) == split_id
        ]
        global_rows = sorted(
            [row for row in cur if row["selection_rule_short"] == "global"],
            key=lambda row: int(row["support_bin_id"]),
        )
        profile_rows = sorted(
            [row for row in cur if row["selection_rule_short"] == "profile-aware"],
            key=lambda row: int(row["support_bin_id"]),
        )
        for key, rows in (("global", global_rows), ("profile-aware", profile_rows)):
            x = np.asarray([float(r["support_bin_center"]) for r in rows], dtype=np.float64)
            y = np.asarray([float(r["bin_nmse"]) for r in rows], dtype=np.float64)
            se = np.asarray([float(r["bin_standard_error"]) for r in rows], dtype=np.float64)
            style = style_map[key]
            ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=4.0,
                label="Global-selected" if key == "global" else "Profile-aware-selected",
            )
            ax.fill_between(x, np.maximum(y - se, 1e-12), y + se, color=style["color"], alpha=0.14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Support bin center h")
        ax.set_ylabel("Test binwise NMSE")
        ax.set_title(f"{dataset} ({MODEL_FAMILY_LABELS.get(family, family)}), split {split_id}")
        ax.grid(alpha=0.18, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    for idx, ax in enumerate(axes, start=1):
        save_axes_group_panel(fig, [ax], figures_dir / f"exp8_representative_profiles_supplement_panel_{idx}.pdf")
    save_legend_figure(
        handles,
        labels,
        figures_dir / "exp8_representative_profiles_supplement_legend.pdf",
        ncol=2,
    )
    fig.savefig(figures_dir / "exp8_representative_profiles_supplement.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = apply_fast_mode(parse_args())
    outdir = resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    results_dir = outdir / "results"
    exp8_results_dir = results_dir / "exp8_real_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    exp8_results_dir.mkdir(parents=True, exist_ok=True)

    model_families = [x.strip() for x in args.models.split(",") if x.strip()]
    max_samples = MAX_SAMPLES_FAST if args.fast else MAX_SAMPLES_DEFAULT

    discovered, skipped = discover_real_datasets(args)
    cleaned: list[dict] = []
    for idx, ds in enumerate(discovered):
        ds_clean, skip_info = clean_dataset(ds, seed=args.seed + 97 * (idx + 1), max_samples=max_samples)
        if skip_info is not None:
            skipped.append(skip_info)
            continue
        cleaned.append(ds_clean)
    if len(cleaned) == 0:
        raise RuntimeError("No usable datasets were found for Experiment 8.")
    if len(cleaned) < 2:
        print("Warning: fewer than two usable datasets are available; proceeding anyway.")

    candidate_rows: list[dict] = []
    candidate_metrics_wide_rows: list[dict] = []
    selection_rows: list[dict] = []
    selection_pair_rows: list[dict] = []
    profile_rows_selected: list[dict] = []
    split_profiles_rows: list[dict] = []
    fit_failures: list[dict] = []
    support_split_rows: list[dict] = []
    disagreement_rows: list[dict] = []
    dataset_metadata_rows: list[dict] = []

    for ds_idx, ds in enumerate(cleaned):
        X_df = ds["X"]
        y = ds["y"]
        splits = make_repeated_splits(n_samples=y.shape[0], n_splits=int(args.n_splits), seed=args.seed + 1000 * ds_idx + 31)
        support_bins_seen: list[int] = []

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

            preprocessor = make_preprocessor(X_train_df)
            preprocessor.fit(X_train_df)
            X_train = transform_dense(preprocessor, X_train_df)
            X_val = transform_dense(preprocessor, X_val_df)
            X_test = transform_dense(preprocessor, X_test_df)
            if X_train.shape[1] < 2:
                fit_failures.append(
                    {
                        "dataset": ds["name"],
                        "split_id": split_id,
                        "model_family": "*",
                        "candidate_id": "*",
                        "reason": "too few transformed features",
                    }
                )
                continue

            h_val = compute_support_scores(X_val, X_train, k_support=int(args.k_support))
            bins = make_support_bins(h_val, n_bins=int(args.n_bins))
            bin_edges = np.asarray(bins["edges"], dtype=np.float64)
            n_bins_eff = int(bins["n_bins_eff"])
            support_bins_seen.append(n_bins_eff)
            n_region_bins = max(1, int(math.ceil(0.2 * n_bins_eff)))
            dense_bins = tuple(range(n_region_bins))
            weak_bins = tuple(range(n_bins_eff - n_region_bins, n_bins_eff))
            counts = np.asarray(bins["counts"], dtype=np.int64)
            contrast = compute_support_contrast_summary(h_val)
            split_support_meta = {
                "support_bins": int(n_bins_eff),
                "raw_support_contrast": float(contrast["raw_support_contrast"]),
                "robust_support_contrast": float(contrast["robust_support_contrast_95_05"]),
                "robust_support_contrast_95_05": float(contrast["robust_support_contrast_95_05"]),
                "robust_support_contrast_90_10": float(contrast["robust_support_contrast_90_10"]),
                "min_bin_size": int(np.min(counts)),
                "max_bin_size": int(np.max(counts)),
                "weak_bin_fraction": float(np.sum(counts[list(weak_bins)]) / max(1, np.sum(counts))),
                "dense_bin_fraction": float(np.sum(counts[list(dense_bins)]) / max(1, np.sum(counts))),
            }
            support_split_rows.append(
                {
                    "dataset": ds["name"],
                    "split_id": split_id,
                    "support_bins": int(split_support_meta["support_bins"]),
                    "raw_support_contrast": float(split_support_meta["raw_support_contrast"]),
                    "robust_support_contrast_95_05": float(split_support_meta["robust_support_contrast_95_05"]),
                    "robust_support_contrast_90_10": float(split_support_meta["robust_support_contrast_90_10"]),
                    "min_positive_support": float(contrast["min_positive_support"]),
                    "used_min_positive_eps": bool(contrast["used_min_positive_eps"]),
                    "q05_support": float(contrast["q05_support"]),
                    "q10_support": float(contrast["q10_support"]),
                    "q90_support": float(contrast["q90_support"]),
                    "q95_support": float(contrast["q95_support"]),
                    "robust_95_05_used_eps": bool(contrast["robust_95_05_used_eps"]),
                    "robust_90_10_used_eps": bool(contrast["robust_90_10_used_eps"]),
                    "min_bin_size": int(split_support_meta["min_bin_size"]),
                    "max_bin_size": int(split_support_meta["max_bin_size"]),
                    "weak_bin_fraction": float(split_support_meta["weak_bin_fraction"]),
                    "dense_bin_fraction": float(split_support_meta["dense_bin_fraction"]),
                }
            )
            h_test = compute_support_scores(X_test, X_train, k_support=int(args.k_support))

            for family_idx, family in enumerate(model_families):
                grid = generate_candidate_grid(model_family=family, X_train=X_train, fast=bool(args.fast))
                family_candidate_rows_val: list[dict] = []
                family_candidate_rows_test: list[dict] = []
                for cand in grid:
                    candidate_id = str(cand["candidate_id"])
                    hyperparams = dict(cand["hyperparams"])
                    complexity_rank = int(cand["complexity_rank"])
                    try:
                        model = fit_model_family_candidate(
                            model_family=family,
                            hyperparams=hyperparams,
                            X_train=X_train,
                            y_train=y_train,
                            random_state=args.seed + 100000 * ds_idx + 1000 * split_id + 17 * (family_idx + 1),
                        )
                        pred_val = predict_model(model, X_val)
                        pred_test = predict_model(model, X_test)
                    except Exception as e:
                        fit_failures.append(
                            {
                                "dataset": ds["name"],
                                "split_id": split_id,
                                "model_family": family,
                                "candidate_id": candidate_id,
                                "reason": str(e),
                            }
                        )
                        continue
                    val_metrics = compute_profile_metrics(
                        y_true=y_val,
                        y_pred=pred_val,
                        h_values=h_val,
                        bin_edges=bin_edges,
                        weak_bins=weak_bins,
                        dense_bins=dense_bins,
                    )
                    test_metrics = compute_profile_metrics(
                        y_true=y_test,
                        y_pred=pred_test,
                        h_values=h_test,
                        bin_edges=bin_edges,
                        weak_bins=weak_bins,
                        dense_bins=dense_bins,
                    )
                    row_common = {
                        "dataset": ds["name"],
                        "split_id": split_id,
                        "model_family": family,
                        "candidate_id": candidate_id,
                        "hyperparams": json.dumps(hyperparams, sort_keys=True),
                        "complexity_rank": complexity_rank,
                        "n_bins_eff": n_bins_eff,
                    }
                    val_row = {
                        **row_common,
                        "split": "val",
                        "global_mse": float(val_metrics["global_mse"]),
                        "global_nmse": float(val_metrics["global_nmse"]),
                        "weak_mse": float(val_metrics["weak_mse"]),
                        "weak_nmse": float(val_metrics["weak_nmse"]),
                        "dense_mse": float(val_metrics["dense_mse"]),
                        "dense_nmse": float(val_metrics["dense_nmse"]),
                        "gap": float(val_metrics["gap"]),
                        "slope": float(val_metrics["slope"]),
                        "pos_slope": float(max(float(val_metrics["slope"]), 0.0)),
                        "profile_var": float(val_metrics["profile_var"]),
                        "y_variance": float(val_metrics["y_variance"]),
                        "raw_support_contrast": float(split_support_meta["raw_support_contrast"]),
                        "robust_support_contrast": float(split_support_meta["robust_support_contrast"]),
                        "robust_support_contrast_95_05": float(split_support_meta["robust_support_contrast_95_05"]),
                        "robust_support_contrast_90_10": float(split_support_meta["robust_support_contrast_90_10"]),
                        "min_bin_size": int(split_support_meta["min_bin_size"]),
                        "max_bin_size": int(split_support_meta["max_bin_size"]),
                        "weak_bin_fraction": float(split_support_meta["weak_bin_fraction"]),
                        "dense_bin_fraction": float(split_support_meta["dense_bin_fraction"]),
                    }
                    test_row = {
                        **row_common,
                        "split": "test",
                        "global_mse": float(test_metrics["global_mse"]),
                        "global_nmse": float(test_metrics["global_nmse"]),
                        "weak_mse": float(test_metrics["weak_mse"]),
                        "weak_nmse": float(test_metrics["weak_nmse"]),
                        "dense_mse": float(test_metrics["dense_mse"]),
                        "dense_nmse": float(test_metrics["dense_nmse"]),
                        "gap": float(test_metrics["gap"]),
                        "slope": float(test_metrics["slope"]),
                        "pos_slope": float(max(float(test_metrics["slope"]), 0.0)),
                        "profile_var": float(test_metrics["profile_var"]),
                        "y_variance": float(test_metrics["y_variance"]),
                        "raw_support_contrast": float(split_support_meta["raw_support_contrast"]),
                        "robust_support_contrast": float(split_support_meta["robust_support_contrast"]),
                        "robust_support_contrast_95_05": float(split_support_meta["robust_support_contrast_95_05"]),
                        "robust_support_contrast_90_10": float(split_support_meta["robust_support_contrast_90_10"]),
                        "min_bin_size": int(split_support_meta["min_bin_size"]),
                        "max_bin_size": int(split_support_meta["max_bin_size"]),
                        "weak_bin_fraction": float(split_support_meta["weak_bin_fraction"]),
                        "dense_bin_fraction": float(split_support_meta["dense_bin_fraction"]),
                    }
                    candidate_rows.append(val_row)
                    candidate_rows.append(test_row)
                    family_candidate_rows_val.append({**val_row, "profile_rows": val_metrics["profile_rows"]})
                    family_candidate_rows_test.append({**test_row, "profile_rows": test_metrics["profile_rows"]})

                if not family_candidate_rows_val:
                    continue
                theta_global, admissible_rows, constraint_active = build_admissible_global_set(
                    family_candidate_rows_val,
                    tau=float(args.tau),
                )
                theta_profile = select_constrained_profile_candidate(
                    admissible_rows,
                    theta_global=theta_global,
                    eta=float(args.eta),
                    zeta=float(args.zeta),
                )
                scored_val_rows = attach_profile_scores(
                    family_candidate_rows_val,
                    theta_global=theta_global,
                    eta=float(args.eta),
                    zeta=float(args.zeta),
                )
                threshold_mse = (1.0 + float(args.tau)) * float(theta_global["global_mse"]) + 1e-12
                if float(theta_profile["global_mse"]) > threshold_mse:
                    raise RuntimeError("Constrained profile-aware selection violated the global-risk budget.")
                selected_by_rule = {
                    "Global-MSE selection": theta_global,
                    "Constrained profile-aware selection": theta_profile,
                }
                test_map = {row["candidate_id"]: row for row in family_candidate_rows_test}
                val_map = {row["candidate_id"]: row for row in scored_val_rows}
                global_candidate_id = str(theta_global["candidate_id"])
                profile_candidate_id = str(theta_profile["candidate_id"])

                for scored_row in scored_val_rows:
                    sel_test = test_map[str(scored_row["candidate_id"])]
                    candidate_metrics_wide_rows.append(
                        {
                            "dataset": ds["name"],
                            "model_family": family,
                            "split_id": split_id,
                            "n_train": int(len(tr_idx)),
                            "n_val": int(len(val_idx)),
                            "n_test": int(len(te_idx)),
                            "candidate_id": str(scored_row["candidate_id"]),
                            "hyperparams": str(scored_row["hyperparams"]),
                            "validation_global_mse": float(scored_row["global_mse"]),
                            "validation_global_nmse": float(scored_row["global_nmse"]),
                            "validation_weak_mse": float(scored_row["weak_mse"]),
                            "validation_weak_nmse": float(scored_row["weak_nmse"]),
                            "validation_dense_mse": float(scored_row["dense_mse"]),
                            "validation_dense_nmse": float(scored_row["dense_nmse"]),
                            "validation_gap": float(scored_row["gap"]),
                            "validation_slope": float(scored_row["slope"]),
                            "validation_pos_slope": float(scored_row["pos_slope"]),
                            "validation_profile_var": float(scored_row["profile_var"]),
                            "validation_profile_score": float(scored_row["profile_score"]),
                            "test_global_mse": float(sel_test["global_mse"]),
                            "test_global_nmse": float(sel_test["global_nmse"]),
                            "test_weak_mse": float(sel_test["weak_mse"]),
                            "test_weak_nmse": float(sel_test["weak_nmse"]),
                            "test_dense_mse": float(sel_test["dense_mse"]),
                            "test_dense_nmse": float(sel_test["dense_nmse"]),
                            "test_gap": float(sel_test["gap"]),
                            "test_slope": float(sel_test["slope"]),
                            "test_pos_slope": float(sel_test["pos_slope"]),
                            "test_profile_var": float(sel_test["profile_var"]),
                            "raw_support_contrast": float(split_support_meta["raw_support_contrast"]),
                            "robust_support_contrast": float(split_support_meta["robust_support_contrast"]),
                            "robust_support_contrast_95_05": float(split_support_meta["robust_support_contrast_95_05"]),
                            "robust_support_contrast_90_10": float(split_support_meta["robust_support_contrast_90_10"]),
                            "min_bin_size": int(split_support_meta["min_bin_size"]),
                            "max_bin_size": int(split_support_meta["max_bin_size"]),
                            "weak_bin_fraction": float(split_support_meta["weak_bin_fraction"]),
                            "dense_bin_fraction": float(split_support_meta["dense_bin_fraction"]),
                        }
                    )

                global_rank_map = rank_map_from_values(
                    [(str(row["candidate_id"]), float(row["global_nmse"])) for row in scored_val_rows]
                )
                profile_rank_map = rank_map_from_values(
                    [(str(row["candidate_id"]), float(row["profile_score"])) for row in scored_val_rows]
                )
                spearman = spearman_rank_correlation(
                    [float(row["global_nmse"]) for row in scored_val_rows],
                    [float(row["profile_score"]) for row in scored_val_rows],
                )

                for rule, selected_val in selected_by_rule.items():
                    sel_test = test_map[selected_val["candidate_id"]]
                    selection_rule_short = "global" if rule == "Global-MSE selection" else "profile-aware"
                    selection_rows.append(
                        {
                            "dataset": ds["name"],
                            "split_id": split_id,
                            "model_family": family,
                            "n_train": int(len(tr_idx)),
                            "n_val": int(len(val_idx)),
                            "n_test": int(len(te_idx)),
                            "selection_rule": rule,
                            "selection_rule_short": selection_rule_short,
                            "candidate_id": str(selected_val["candidate_id"]),
                            "hyperparams": str(selected_val["hyperparams"]),
                            "theta_global_candidate_id": str(theta_global["candidate_id"]),
                            "constraint_active": bool(constraint_active) if rule == "Constrained profile-aware selection" else False,
                            "constraint_threshold_mse": float((1.0 + float(args.tau)) * float(theta_global["global_mse"])),
                            "constraint_threshold_nmse": float((1.0 + float(args.tau)) * float(theta_global["global_nmse"])),
                            "validation_global_mse": float(selected_val["global_mse"]),
                            "validation_global_nmse": float(selected_val["global_nmse"]),
                            "validation_weak_mse": float(selected_val["weak_mse"]),
                            "validation_weak_nmse": float(selected_val["weak_nmse"]),
                            "validation_dense_mse": float(selected_val["dense_mse"]),
                            "validation_dense_nmse": float(selected_val["dense_nmse"]),
                            "validation_gap": float(selected_val["gap"]),
                            "validation_slope": float(selected_val["slope"]),
                            "validation_pos_slope": float(max(float(selected_val["slope"]), 0.0)),
                            "validation_profile_var": float(selected_val["profile_var"]),
                            "validation_profile_score": float(val_map[str(selected_val["candidate_id"])]["profile_score"]),
                            "global_mse": float(sel_test["global_mse"]),
                            "global_nmse": float(sel_test["global_nmse"]),
                            "weak_mse": float(sel_test["weak_mse"]),
                            "weak_nmse": float(sel_test["weak_nmse"]),
                            "dense_mse": float(sel_test["dense_mse"]),
                            "dense_nmse": float(sel_test["dense_nmse"]),
                            "gap": float(sel_test["gap"]),
                            "slope": float(sel_test["slope"]),
                            "pos_slope": float(sel_test["pos_slope"]),
                            "profile_var": float(sel_test["profile_var"]),
                            "y_variance": float(sel_test["y_variance"]),
                            "raw_support_contrast": float(split_support_meta["raw_support_contrast"]),
                            "robust_support_contrast": float(split_support_meta["robust_support_contrast"]),
                            "robust_support_contrast_95_05": float(split_support_meta["robust_support_contrast_95_05"]),
                            "robust_support_contrast_90_10": float(split_support_meta["robust_support_contrast_90_10"]),
                            "min_bin_size": int(split_support_meta["min_bin_size"]),
                            "max_bin_size": int(split_support_meta["max_bin_size"]),
                            "weak_bin_fraction": float(split_support_meta["weak_bin_fraction"]),
                            "dense_bin_fraction": float(split_support_meta["dense_bin_fraction"]),
                        }
                    )
                    for prow in sel_test["profile_rows"]:
                        profile_rows_selected.append(
                            {
                                "dataset": ds["name"],
                                "split_id": split_id,
                                "model_family": family,
                                "selection_rule": rule,
                                "selection_rule_short": selection_rule_short,
                                "bin_id": int(prow["bin_id"]),
                                "h_bin_center": float(prow["h_bin_center"]),
                                "h_bin_left": float(prow["h_bin_left"]),
                                "h_bin_right": float(prow["h_bin_right"]),
                                "mse_bin_raw": float(prow["mse_bin_raw"]),
                                "nmse_bin": float(prow["mse_bin_norm"]),
                                "nmse_bin_se": float(prow["mse_bin_norm_se"]),
                                "n_bin": int(prow["n_bin"]),
                            }
                        )
                        split_profiles_rows.append(
                            {
                                "dataset": ds["name"],
                                "model_family": family,
                                "split_id": split_id,
                                "selection_rule": rule,
                                "selection_rule_short": selection_rule_short,
                                "support_bin_id": int(prow["bin_id"]),
                                "support_bin_center": float(prow["h_bin_center"]),
                                "support_bin_left": float(prow["h_bin_left"]),
                                "support_bin_right": float(prow["h_bin_right"]),
                                "bin_count": int(prow["n_bin"]),
                                "bin_mse": float(prow["mse_bin_raw"]),
                                "bin_nmse": float(prow["mse_bin_norm"]),
                                "bin_standard_error": float(prow["mse_bin_norm_se"]),
                            }
                        )

                global_val_selected = val_map[global_candidate_id]
                profile_val_selected = val_map[profile_candidate_id]
                global_test_selected = test_map[global_candidate_id]
                profile_test_selected = test_map[profile_candidate_id]
                selection_changed = global_candidate_id != profile_candidate_id
                selection_pair_rows.append(
                    {
                        "dataset": ds["name"],
                        "model_family": family,
                        "split_id": split_id,
                        "n_train": int(len(tr_idx)),
                        "n_val": int(len(val_idx)),
                        "n_test": int(len(te_idx)),
                        "theta_global": str(global_val_selected["hyperparams"]),
                        "theta_profile": str(profile_val_selected["hyperparams"]),
                        "theta_global_candidate_id": global_candidate_id,
                        "theta_profile_candidate_id": profile_candidate_id,
                        "changed_selection": bool(selection_changed),
                        "selection_changed": bool(selection_changed),
                        "robust_support_contrast": float(split_support_meta["robust_support_contrast"]),
                        "raw_support_contrast": float(split_support_meta["raw_support_contrast"]),
                        "min_bin_size": int(split_support_meta["min_bin_size"]),
                        "global_val_nmse_global_selected": float(global_val_selected["global_nmse"]),
                        "global_val_nmse_profile_selected": float(profile_val_selected["global_nmse"]),
                        "profile_score_global_selected": float(global_val_selected["profile_score"]),
                        "profile_score_profile_selected": float(profile_val_selected["profile_score"]),
                        "test_global_nmse_global_selected": float(global_test_selected["global_nmse"]),
                        "test_global_nmse_profile_selected": float(profile_test_selected["global_nmse"]),
                        "test_weak_nmse_global_selected": float(global_test_selected["weak_nmse"]),
                        "test_weak_nmse_profile_selected": float(profile_test_selected["weak_nmse"]),
                        "test_dense_nmse_global_selected": float(global_test_selected["dense_nmse"]),
                        "test_dense_nmse_profile_selected": float(profile_test_selected["dense_nmse"]),
                        "test_gap_global_selected": float(global_test_selected["gap"]),
                        "test_gap_profile_selected": float(profile_test_selected["gap"]),
                        "test_slope_global_selected": float(global_test_selected["slope"]),
                        "test_slope_profile_selected": float(profile_test_selected["slope"]),
                        "test_pos_slope_global_selected": float(global_test_selected["pos_slope"]),
                        "test_pos_slope_profile_selected": float(profile_test_selected["pos_slope"]),
                        "test_profile_var_global_selected": float(global_test_selected["profile_var"]),
                        "test_profile_var_profile_selected": float(profile_test_selected["profile_var"]),
                        "delta_global_pct": float(percentage_delta(profile_test_selected["global_nmse"], global_test_selected["global_nmse"])),
                        "delta_global_nmse_pct": float(percentage_delta(profile_test_selected["global_nmse"], global_test_selected["global_nmse"])),
                        "delta_weak_pct": float(percentage_delta(profile_test_selected["weak_nmse"], global_test_selected["weak_nmse"])),
                        "delta_weak_nmse_pct": float(percentage_delta(profile_test_selected["weak_nmse"], global_test_selected["weak_nmse"])),
                        "delta_gap_pct": float(percentage_delta(profile_test_selected["gap"], global_test_selected["gap"])),
                        "delta_slope_pct": float(percentage_delta(profile_test_selected["slope"], global_test_selected["slope"])),
                        "delta_positive_slope_pct": float(percentage_delta(profile_test_selected["pos_slope"], global_test_selected["pos_slope"])),
                        "delta_pos_slope_pct": float(percentage_delta(profile_test_selected["pos_slope"], global_test_selected["pos_slope"])),
                        "delta_profile_var_pct": float(percentage_delta(profile_test_selected["profile_var"], global_test_selected["profile_var"])),
                        "global_nmse_within_1pct": bool(
                            percentage_delta(profile_test_selected["global_nmse"], global_test_selected["global_nmse"]) <= 1.0
                        ),
                        "global_nmse_within_2pct": bool(
                            percentage_delta(profile_test_selected["global_nmse"], global_test_selected["global_nmse"]) <= 2.0
                        ),
                    }
                )

                disagreement_rows.append(
                    {
                        "dataset": ds["name"],
                        "model_family": family,
                        "split_id": split_id,
                        "theta_global_candidate_id": global_candidate_id,
                        "theta_profile_candidate_id": profile_candidate_id,
                        "selection_changed": bool(selection_changed),
                        "global_rank_of_profile_selected": float(global_rank_map[profile_candidate_id]),
                        "profile_rank_of_global_selected": float(profile_rank_map[global_candidate_id]),
                        "validation_global_cost_pct": float(
                            percentage_delta(profile_val_selected["global_nmse"], global_val_selected["global_nmse"])
                        ),
                        "validation_profile_score_gain_pct": float(
                            100.0
                            * (
                                float(global_val_selected["profile_score"])
                                - float(profile_val_selected["profile_score"])
                            )
                            / (abs(float(global_val_selected["profile_score"])) + 1e-12)
                        ),
                        "absolute_test_gap_difference": float(
                            abs(float(profile_test_selected["gap"]) - float(global_test_selected["gap"]))
                        ),
                        "signed_test_gap_delta_pct": float(
                            percentage_delta(profile_test_selected["gap"], global_test_selected["gap"])
                        ),
                        "spearman_global_vs_profile": float(spearman),
                    }
                )

        ds_support_rows = [row for row in support_split_rows if row["dataset"] == ds["name"]]
        dataset_metadata_rows.append(
            {
                "dataset": ds["name"],
                "samples": int(ds["samples"]),
                "features": int(ds["raw_features"]),
                "raw_support_contrast_median": float(np.median([float(row["raw_support_contrast"]) for row in ds_support_rows])) if ds_support_rows else float("nan"),
                "robust_support_contrast_95_05_median": float(np.median([float(row["robust_support_contrast_95_05"]) for row in ds_support_rows])) if ds_support_rows else float("nan"),
                "min_bin_size": int(np.min([int(row["min_bin_size"]) for row in ds_support_rows])) if ds_support_rows else 0,
            }
        )

    summary_rows: list[dict] = []
    pair_level_summary_rows: list[dict] = []
    aggregate_summary_rows: list[dict] = []
    by_dataset_family: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selection_rows:
        by_dataset_family[(row["dataset"], row["model_family"])].append(row)

    split_level_flags = {
        "selection_changed": [],
        "weak_improved": [],
        "gap_improved": [],
        "slope_improved": [],
        "profile_var_improved": [],
        "global_improved": [],
    }
    pair_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selection_pair_rows:
        pair_groups[(row["dataset"], row["model_family"])].append(row)
        split_level_flags["selection_changed"].append(bool(row["selection_changed"]))
        split_level_flags["weak_improved"].append(float(row["delta_weak_nmse_pct"]) < 0.0)
        split_level_flags["gap_improved"].append(float(row["delta_gap_pct"]) < 0.0)
        split_level_flags["slope_improved"].append(float(row["delta_positive_slope_pct"]) < 0.0)
        split_level_flags["profile_var_improved"].append(float(row["delta_profile_var_pct"]) < 0.0)
        split_level_flags["global_improved"].append(float(row["delta_global_nmse_pct"]) < 0.0)

    for (dataset, family), rows in sorted(by_dataset_family.items()):
        global_rows = [r for r in rows if r["selection_rule"] == "Global-MSE selection"]
        profile_rows = [r for r in rows if r["selection_rule"] == "Constrained profile-aware selection"]
        if not global_rows or not profile_rows:
            continue
        global_nmse_mean = float(np.mean([float(r["global_nmse"]) for r in global_rows]))
        weak_nmse_mean = float(np.mean([float(r["weak_nmse"]) for r in global_rows]))
        gap_mean = float(np.mean([float(r["gap"]) for r in global_rows]))
        slope_pos_mean = float(np.mean([max(float(r["slope"]), 0.0) for r in global_rows]))
        profile_var_mean = float(np.mean([float(r["profile_var"]) for r in global_rows]))
        profile_global_mean = float(np.mean([float(r["global_nmse"]) for r in profile_rows]))
        profile_weak_mean = float(np.mean([float(r["weak_nmse"]) for r in profile_rows]))
        profile_gap_mean = float(np.mean([float(r["gap"]) for r in profile_rows]))
        profile_slope_pos_mean = float(np.mean([max(float(r["slope"]), 0.0) for r in profile_rows]))
        profile_profile_var_mean = float(np.mean([float(r["profile_var"]) for r in profile_rows]))
        pair_rows = pair_groups.get((dataset, family), [])
        changed_fraction = float(np.mean([bool(r["selection_changed"]) for r in pair_rows])) if pair_rows else 0.0
        pair_summary = {
            "dataset": dataset,
            "model_family": family,
            "model_family_label": MODEL_FAMILY_LABELS.get(family, family),
            "selection_changed": "Yes" if changed_fraction >= 0.30 else "No",
            "selection_changed_fraction": changed_fraction,
            "constraint_active_fraction": float(np.mean([bool(r["constraint_active"]) for r in profile_rows])) if profile_rows else 0.0,
            "weak_improved_fraction": float(np.mean([float(r["delta_weak_nmse_pct"]) < 0.0 for r in pair_rows])) if pair_rows else 0.0,
            "gap_improved_fraction": float(np.mean([float(r["delta_gap_pct"]) < 0.0 for r in pair_rows])) if pair_rows else 0.0,
            "positive_slope_improved_fraction": float(np.mean([float(r["delta_positive_slope_pct"]) < 0.0 for r in pair_rows])) if pair_rows else 0.0,
            "profile_var_improved_fraction": float(np.mean([float(r["delta_profile_var_pct"]) < 0.0 for r in pair_rows])) if pair_rows else 0.0,
            "global_nmse_improved_fraction": float(np.mean([float(r["delta_global_nmse_pct"]) < 0.0 for r in pair_rows])) if pair_rows else 0.0,
            "fraction_global_nmse_within_1pct": float(np.mean([bool(r["global_nmse_within_1pct"]) for r in pair_rows])) if pair_rows else 0.0,
            "fraction_global_nmse_within_2pct": float(np.mean([bool(r["global_nmse_within_2pct"]) for r in pair_rows])) if pair_rows else 0.0,
            "robust_support_contrast_median": float(np.median([float(r["robust_support_contrast"]) for r in pair_rows])) if pair_rows else float("nan"),
            "raw_support_contrast_median": float(np.median([float(r["raw_support_contrast"]) for r in pair_rows])) if pair_rows else float("nan"),
            "delta_global_pct": percentage_change(profile_global_mean, global_nmse_mean),
            "delta_weak_pct": percentage_change(profile_weak_mean, weak_nmse_mean),
            "delta_gap_pct": percentage_change(profile_gap_mean, gap_mean),
            "delta_slope_pct": percentage_change(profile_slope_pos_mean, slope_pos_mean),
            "delta_profile_var_pct": percentage_change(profile_profile_var_mean, profile_var_mean),
            "fraction_changed_from_global": changed_fraction,
            "fraction_gap_improved": float(np.mean([float(r["delta_gap_pct"]) < 0.0 for r in pair_rows])) if pair_rows else 0.0,
            "fraction_positive_slope_improved": float(np.mean([float(r["delta_positive_slope_pct"]) < 0.0 for r in pair_rows])) if pair_rows else 0.0,
            "fraction_weak_nmse_improved": float(np.mean([float(r["delta_weak_nmse_pct"]) < 0.0 for r in pair_rows])) if pair_rows else 0.0,
            "median_delta_global_nmse_pct": percentage_change(profile_global_mean, global_nmse_mean),
            "median_delta_weak_nmse_pct": percentage_change(profile_weak_mean, weak_nmse_mean),
            "median_delta_gap_pct": percentage_change(profile_gap_mean, gap_mean),
            "median_delta_positive_slope_pct": percentage_change(profile_slope_pos_mean, slope_pos_mean),
        }
        pair_level_summary_rows.append(pair_summary)
        summary_rows.append(pair_summary)

    representative_pairs = choose_representative_pairs(summary_rows)
    representative_cases = choose_representative_cases(selection_pair_rows, split_profiles_rows)
    main_representative_case = representative_cases[0] if representative_cases else None
    supplementary_representative_cases = representative_cases[1:4] if len(representative_cases) > 1 else []

    realdata_selection_rows: list[dict] = []
    for rep in representative_pairs:
        dataset = rep["dataset"]
        family = rep["model_family"]
        label = f"{dataset} ({MODEL_FAMILY_LABELS.get(family, family)})"
        for rule in RULE_DISPLAY_ORDER:
            cur = [
                row for row in selection_rows
                if row["dataset"] == dataset and row["model_family"] == family and row["selection_rule"] == rule
            ]
            if not cur:
                continue
            realdata_selection_rows.append(
                {
                    "dataset": label,
                    "selection_rule": rule,
                    "global_nmse": float(np.mean([float(r["global_nmse"]) for r in cur])),
                    "weak_nmse": float(np.mean([float(r["weak_nmse"]) for r in cur])),
                    "dense_nmse": float(np.mean([float(r["dense_nmse"]) for r in cur])),
                    "gap": float(np.mean([float(r["gap"]) for r in cur])),
                    "slope": float(np.nanmean([float(r["slope"]) for r in cur])),
                    "profile_var": float(np.mean([float(r["profile_var"]) for r in cur])),
                }
            )

    make_realdata_profiles_figure(figures_dir, profile_rows_selected, representative_pairs)
    if main_representative_case is not None:
        make_representative_profile_figure(
            figures_dir / "exp8_representative_profile.pdf",
            split_profiles_rows,
            main_representative_case,
        )
    make_representative_profiles_supplement_figure(
        figures_dir,
        split_profiles_rows,
        supplementary_representative_cases,
    )
    make_model_agnostic_pareto_figure(
        figures_dir,
        candidate_rows,
        selection_rows,
        representative_pairs,
        tau=float(args.tau),
    )

    metric_specs = [
        ("Global MSE", "delta_global_pct"),
        ("Weak MSE", "delta_weak_pct"),
        ("Gap", "delta_gap_pct"),
        ("Slope", "delta_slope_pct"),
        ("Profile variance", "delta_profile_var_pct"),
    ]
    for label, key in metric_specs:
        vals = np.asarray([float(row[key]) for row in summary_rows], dtype=np.float64)
        aggregate_summary_rows.append(
            {
                "criterion": label,
                "fraction_improved": float(np.mean(vals < 0.0)) if vals.size else float("nan"),
                "mean_change": float(np.mean(vals)) if vals.size else float("nan"),
                "median_change": float(np.median(vals)) if vals.size else float("nan"),
            }
        )

    changed_pair_rows = [row for row in selection_pair_rows if bool(row["changed_selection"])]
    unchanged_pair_rows = [row for row in selection_pair_rows if not bool(row["changed_selection"])]
    changed_selection_conditional_rows = [
        summarize_changed_selection_group("Changed selections only", changed_pair_rows),
        summarize_changed_selection_group("Unchanged selections only", unchanged_pair_rows),
        summarize_changed_selection_group("All splits", selection_pair_rows),
    ]
    decision_change_usefulness_rows = [
        summarize_pair_group("Changed selections only", changed_pair_rows),
        summarize_pair_group("All splits", selection_pair_rows),
        summarize_pair_group("Unchanged selections", unchanged_pair_rows),
    ]

    split_level_summary_rows: list[dict] = []
    for row in selection_pair_rows:
        split_level_summary_rows.append(
            {
                "dataset": row["dataset"],
                "model_family": row["model_family"],
                "split_id": int(row["split_id"]),
                "robust_support_contrast_median": float(row["robust_support_contrast"]),
                "fraction_changed_from_global": float(bool(row["changed_selection"])),
                "fraction_gap_improved": float(float(row["delta_gap_pct"]) < 0.0),
                "fraction_positive_slope_improved": float(float(row["delta_positive_slope_pct"]) < 0.0),
                "fraction_weak_nmse_improved": float(float(row["delta_weak_nmse_pct"]) < 0.0),
                "fraction_global_nmse_within_1pct": float(bool(row["global_nmse_within_1pct"])),
                "fraction_global_nmse_within_2pct": float(bool(row["global_nmse_within_2pct"])),
                "median_delta_gap_pct": float(row["delta_gap_pct"]),
                "median_delta_positive_slope_pct": float(row["delta_positive_slope_pct"]),
                "median_delta_weak_nmse_pct": float(row["delta_weak_nmse_pct"]),
                "median_delta_global_nmse_pct": float(row["delta_global_nmse_pct"]),
            }
        )

    pair_terciles = assign_terciles([float(row["robust_support_contrast_median"]) for row in pair_level_summary_rows])
    for row, stratum in zip(pair_level_summary_rows, pair_terciles):
        row["contrast_group"] = stratum
    pair_level_stratification_rows = [
        aggregate_stratum_rows(
            "pair_level",
            stratum,
            [row for row in pair_level_summary_rows if row.get("contrast_group") == stratum],
        )
        for stratum in ("Low", "Medium", "High")
    ]

    split_terciles = assign_terciles([float(row["robust_support_contrast_median"]) for row in split_level_summary_rows])
    for row, stratum in zip(split_level_summary_rows, split_terciles):
        row["contrast_group"] = stratum
    split_level_stratification_rows = [
        aggregate_stratum_rows(
            "split_level",
            stratum,
            [row for row in split_level_summary_rows if row.get("contrast_group") == stratum],
        )
        for stratum in ("Low", "Medium", "High")
    ]
    support_heterogeneity_rows = [*pair_level_stratification_rows, *split_level_stratification_rows]

    disagreement_summary_rows = [
        summarize_disagreement_group("All splits", disagreement_rows),
        summarize_disagreement_group("Changed selections only", [row for row in disagreement_rows if bool(row["selection_changed"])]),
    ]

    save_csv(tables_dir / "exp8_datasets.csv", dataset_metadata_rows)
    save_dataset_table_tex(tables_dir / "exp8_datasets.tex", dataset_metadata_rows)
    save_csv(tables_dir / "exp8_realdata_selection.csv", realdata_selection_rows)
    save_selection_table_tex(tables_dir / "exp8_realdata_selection.tex", realdata_selection_rows)
    save_csv(tables_dir / "exp8_model_agnostic_summary.csv", summary_rows)
    save_model_agnostic_summary_tex(tables_dir / "exp8_model_agnostic_summary.tex", summary_rows)
    save_csv(tables_dir / "exp8_aggregate_summary.csv", aggregate_summary_rows)
    save_aggregate_summary_tex(tables_dir / "exp8_aggregate_summary.tex", aggregate_summary_rows)
    save_csv(tables_dir / "exp8_changed_selection_conditional_table.csv", changed_selection_conditional_rows)
    save_changed_selection_conditional_tex(
        tables_dir / "exp8_changed_selection_conditional_table.tex",
        changed_selection_conditional_rows,
    )
    save_csv(tables_dir / "exp8_decision_change_usefulness.csv", decision_change_usefulness_rows)
    save_decision_change_usefulness_tex(
        tables_dir / "exp8_decision_change_usefulness.tex",
        [row for row in decision_change_usefulness_rows if row["group"] in {"Changed selections only", "All splits"}],
    )
    save_support_heterogeneity_pair_split_tex(
        tables_dir / "exp8_support_heterogeneity_stratification_pair_level.tex",
        pair_level_stratification_rows,
    )
    save_support_heterogeneity_pair_split_tex(
        tables_dir / "exp8_support_heterogeneity_stratification_split_level.tex",
        split_level_stratification_rows,
    )
    save_csv(tables_dir / "exp8_support_heterogeneity_stratification.csv", support_heterogeneity_rows)
    save_support_heterogeneity_stratification_tex(
        tables_dir / "exp8_support_heterogeneity_stratification.tex",
        support_heterogeneity_rows,
    )
    save_csv(tables_dir / "exp8_profile_global_disagreement.csv", disagreement_summary_rows)
    save_profile_global_disagreement_tex(
        tables_dir / "exp8_profile_global_disagreement.tex",
        disagreement_summary_rows,
    )
    save_csv(results_dir / "exp8_candidate_metrics.csv", candidate_rows)
    save_csv(results_dir / "exp8_realdata_candidate_metrics.csv", candidate_metrics_wide_rows)
    save_csv(results_dir / "exp8_profiles_selected.csv", profile_rows_selected)
    save_csv(results_dir / "exp8_selection_results.csv", selection_rows)
    save_csv(results_dir / "exp8_realdata_selection_pairs.csv", selection_pair_rows)
    save_csv(exp8_results_dir / "exp8_split_profiles.csv", split_profiles_rows)
    save_csv(exp8_results_dir / "exp8_changed_selection_conditional.csv", changed_selection_conditional_rows)
    save_csv(exp8_results_dir / "exp8_support_heterogeneity_stratification.csv", support_heterogeneity_rows)
    save_csv(exp8_results_dir / "exp8_representative_cases.csv", representative_cases)
    save_csv(exp8_results_dir / "exp8_selection_pairs.csv", selection_pair_rows)
    save_csv(results_dir / "exp8_decision_change_usefulness.csv", decision_change_usefulness_rows)
    save_csv(results_dir / "exp8_support_heterogeneity_stratification.csv", support_heterogeneity_rows)
    save_csv(results_dir / "exp8_profile_global_disagreement_per_split.csv", disagreement_rows)
    save_csv(results_dir / "exp8_profile_global_disagreement_summary.csv", disagreement_summary_rows)

    selection_frequencies: dict[str, dict[str, dict[str, int]]] = {}
    for (dataset, family), rows in by_dataset_family.items():
        selection_frequencies.setdefault(dataset, {})[family] = {}
        for rule in RULE_DISPLAY_ORDER:
            cur = [str(r["candidate_id"]) for r in rows if r["selection_rule"] == rule]
            selection_frequencies[dataset][family][rule] = {k: int(v) for k, v in Counter(cur).items()}

    summary = {
        "experiment": "exp8_model_agnostic_realdata",
        "parameters": {
            "seed": int(args.seed),
            "n_splits": int(args.n_splits),
            "n_bins": int(args.n_bins),
            "k_support": int(args.k_support),
            "tau": float(args.tau),
            "eta": float(args.eta),
            "zeta": float(args.zeta),
            "models": model_families,
            "max_datasets": int(MAX_DATASETS_FAST if args.fast else MAX_DATASETS_DEFAULT),
            "max_samples_per_dataset": int(max_samples),
            "support_contrast_aggregation": "median across splits",
            "profile_variance_definition": "Unweighted variance across binwise NMSE values.",
            "selection_tie_breaking": "metric, then lower global MSE, then simpler candidate, then sorted candidate id.",
        },
        "support_contrast_notes": {
            "raw_definition": "max(h_val) / min_positive(h_val)",
            "robust_95_05_definition": "quantile_0.95(h_val) / quantile_0.05(h_val)",
            "robust_90_10_definition": "quantile_0.90(h_val) / quantile_0.10(h_val)",
            "aggregation_method": "median across splits",
            "epsilon_used_for_nonpositive_percentiles": {
                "robust_95_05_count": int(sum(bool(row["robust_95_05_used_eps"]) for row in support_split_rows)),
                "robust_90_10_count": int(sum(bool(row["robust_90_10_used_eps"]) for row in support_split_rows)),
                "min_positive_fallback_count": int(sum(bool(row["used_min_positive_eps"]) for row in support_split_rows)),
            },
        },
        "datasets_used": [row["dataset"] for row in dataset_metadata_rows],
        "dataset_metadata": dataset_metadata_rows,
        "support_heterogeneity_by_split": support_split_rows,
        "representative_pairs": representative_pairs,
        "representative_cases": representative_cases,
        "selection_frequencies": selection_frequencies,
        "model_agnostic_summary": summary_rows,
        "pair_level_summary": pair_level_summary_rows,
        "split_level_summary": split_level_summary_rows,
        "aggregate_summary": aggregate_summary_rows,
        "decision_change_usefulness": decision_change_usefulness_rows,
        "changed_selection_conditional": changed_selection_conditional_rows,
        "support_heterogeneity_stratification": support_heterogeneity_rows,
        "support_heterogeneity_stratification_pair_level": pair_level_stratification_rows,
        "support_heterogeneity_stratification_split_level": split_level_stratification_rows,
        "profile_global_disagreement_summary": disagreement_summary_rows,
        "profile_global_disagreement_per_split": disagreement_rows,
        "realdata_selection_rows": realdata_selection_rows,
        "selection_results": selection_rows,
        "selection_pairs": selection_pair_rows,
        "split_profiles": {
            "path": str((exp8_results_dir / "exp8_split_profiles.csv").resolve()),
            "n_rows": int(len(split_profiles_rows)),
        },
        "candidate_metrics_wide": {
            "path": str((results_dir / "exp8_realdata_candidate_metrics.csv").resolve()),
            "n_rows": int(len(candidate_metrics_wide_rows)),
        },
        "fraction_splits_changed_selection": float(np.mean(split_level_flags["selection_changed"])) if split_level_flags["selection_changed"] else 0.0,
        "fraction_splits_weak_nmse_improved": float(np.mean(split_level_flags["weak_improved"])) if split_level_flags["weak_improved"] else 0.0,
        "fraction_splits_gap_improved": float(np.mean(split_level_flags["gap_improved"])) if split_level_flags["gap_improved"] else 0.0,
        "fraction_splits_slope_improved": float(np.mean(split_level_flags["slope_improved"])) if split_level_flags["slope_improved"] else 0.0,
        "fraction_splits_profile_var_improved": float(np.mean(split_level_flags["profile_var_improved"])) if split_level_flags["profile_var_improved"] else 0.0,
        "fraction_splits_global_nmse_improved": float(np.mean(split_level_flags["global_improved"])) if split_level_flags["global_improved"] else 0.0,
        "skipped_datasets_or_failures": skipped,
        "fit_failures": fit_failures,
        "output_paths": {
            "profiles_figure": str((figures_dir / "exp8_realdata_profiles.pdf").resolve()),
            "pareto_figure": str((figures_dir / "exp8_model_agnostic_pareto.pdf").resolve()),
            "dataset_table": str((tables_dir / "exp8_datasets.tex").resolve()),
            "selection_table": str((tables_dir / "exp8_realdata_selection.tex").resolve()),
            "model_agnostic_summary_table": str((tables_dir / "exp8_model_agnostic_summary.tex").resolve()),
            "aggregate_summary_table": str((tables_dir / "exp8_aggregate_summary.tex").resolve()),
            "decision_change_usefulness_table": str((tables_dir / "exp8_decision_change_usefulness.tex").resolve()),
            "changed_selection_conditional_table": str((tables_dir / "exp8_changed_selection_conditional_table.tex").resolve()),
            "support_heterogeneity_stratification_table": str((tables_dir / "exp8_support_heterogeneity_stratification.tex").resolve()),
            "support_heterogeneity_pair_level_table": str((tables_dir / "exp8_support_heterogeneity_stratification_pair_level.tex").resolve()),
            "support_heterogeneity_split_level_table": str((tables_dir / "exp8_support_heterogeneity_stratification_split_level.tex").resolve()),
            "profile_global_disagreement_table": str((tables_dir / "exp8_profile_global_disagreement.tex").resolve()),
            "representative_profile_figure": str((figures_dir / "exp8_representative_profile.pdf").resolve()),
            "representative_profiles_supplement_figure": str((figures_dir / "exp8_representative_profiles_supplement.pdf").resolve()),
            "candidate_metrics_csv": str((results_dir / "exp8_candidate_metrics.csv").resolve()),
            "candidate_metrics_wide_csv": str((results_dir / "exp8_realdata_candidate_metrics.csv").resolve()),
            "selection_results_csv": str((results_dir / "exp8_selection_results.csv").resolve()),
            "selection_pairs_csv": str((results_dir / "exp8_realdata_selection_pairs.csv").resolve()),
            "split_profiles_csv": str((exp8_results_dir / "exp8_split_profiles.csv").resolve()),
            "changed_selection_conditional_csv": str((exp8_results_dir / "exp8_changed_selection_conditional.csv").resolve()),
            "support_heterogeneity_stratification_csv": str((exp8_results_dir / "exp8_support_heterogeneity_stratification.csv").resolve()),
            "representative_cases_csv": str((exp8_results_dir / "exp8_representative_cases.csv").resolve()),
        },
    }
    (results_dir / "exp8_realdata_applicability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
