from __future__ import annotations

import os
import sys
import math
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from autogluon.tabular import TabularPredictor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class CountModelConfig:
    data_path: str = "1minute_data_newyork.csv"
    target_appliance: str = "car1"
    timestamp_col: str = "localminute"
    id_col: str = "dataid"
    grid_col: str = "grid"
    drop_cols: Tuple[str, ...] = ("leg1v", "leg2v")
    threshold_kw: float = 0.5
    window_size: int = 599
    max_rows: Optional[int] = 1_000_000
    development_fraction: float = 0.90  # oldest 90% for train+val
    validation_fraction_within_development: float = 0.10  # 81/9/10 overall split
    ag_time_limit_seconds: int = 30 * 60
    ag_presets: str = "best_quality"
    predictor_path_prefix: str = "AG_regression"
    random_seed: int = 42
    openblas_threads: str = "8"
    omp_threads: str = "8"


@dataclass
class SyntheticLoadConfig:
    # ---- Key ambiguity controls requested by the user ----
    session_library_mode: str = "full_runs"  # "full_runs" or "minute_fragments"
    count_assignment_mode: str = "segment_constant"  # "segment_constant" or "minute_resample"
    mc_stat_scope: str = "full_series"  # "full_series" or "active_only"

    # ---- Defaults chosen to match the thesis discussion ----
    active_segment_count_choices: Tuple[int, ...] = tuple(range(1, 8))
    validation_fixed_counts: Tuple[int, ...] = (8, 9, 10)
    monte_carlo_runs: int = 500
    random_seed: int = 42

    # ---- Full-run handling ----
    full_run_projection_mode: str = "random_crop_or_pad"  # or "resample_to_segment"

    # ---- PMF handling ----
    pmf_mode: str = "uniform"  # or "custom"
    custom_pmf: Optional[Dict[int, float]] = None

    # ---- Plotting / regression ----
    max_count_to_plot: int = 10


@dataclass
class PipelineArtifacts:
    S_Total: pd.Series
    A_Total: pd.DataFrame
    num_active: pd.Series
    df_combined: pd.DataFrame
    predictor: TabularPredictor
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    y_test: np.ndarray
    time_test: pd.Series
    count_results: pd.DataFrame
    preliminary_summary: pd.DataFrame
    preliminary_models: Dict[str, LinearRegression]


# =============================================================================
# LOGGING
# =============================================================================


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return sys.__stdout__.isatty()


# =============================================================================
# LOW-LEVEL HELPERS
# =============================================================================


def set_thread_env(cfg: CountModelConfig) -> None:
    os.environ["OPENBLAS_NUM_THREADS"] = cfg.openblas_threads
    os.environ["OMP_NUM_THREADS"] = cfg.omp_threads



def ensure_datetime_index(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    out = out.dropna(subset=[timestamp_col])
    out = out.sort_values(timestamp_col)
    out = out.set_index(timestamp_col)
    return out



def contiguous_true_segments(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Returns list of (start_timestamp, end_timestamp_inclusive) for contiguous True runs.
    Assumes a datetime index sorted ascending.
    """
    mask = mask.fillna(False).astype(bool)
    if mask.empty:
        return []

    idx = mask.index
    values = mask.to_numpy()
    segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None

    for i, flag in enumerate(values):
        if flag and start is None:
            start = idx[i]
        is_break = False
        if flag:
            if i == len(values) - 1:
                is_break = True
            else:
                next_gap = idx[i + 1] - idx[i]
                if (not values[i + 1]) or (next_gap != pd.Timedelta(minutes=1)):
                    is_break = True
        if flag and is_break and start is not None:
            segments.append((start, idx[i]))
            start = None
    return segments



def contiguous_on_runs(group: pd.DataFrame, power_col: str, threshold_kw: float) -> List[np.ndarray]:
    """
    Extract contiguous ON runs from a single household's timeseries.
    Runs break when power <= threshold or the timestamp gap is not exactly 1 minute.
    """
    if group.empty:
        return []

    group = group.sort_index()
    on_mask = group[power_col].fillna(0) > threshold_kw
    runs: List[np.ndarray] = []
    current: List[float] = []
    prev_ts: Optional[pd.Timestamp] = None

    for ts, is_on, value in zip(group.index, on_mask.to_numpy(), group[power_col].to_numpy()):
        gap_ok = prev_ts is None or (ts - prev_ts == pd.Timedelta(minutes=1))
        if is_on and gap_ok:
            current.append(float(value))
        elif is_on and not gap_ok:
            if current:
                runs.append(np.asarray(current, dtype=float))
            current = [float(value)]
        else:
            if current:
                runs.append(np.asarray(current, dtype=float))
                current = []
        prev_ts = ts

    if current:
        runs.append(np.asarray(current, dtype=float))
    return runs



def normalise_windows(windows: Sequence[np.ndarray], mean_x: float, std_x: float) -> List[np.ndarray]:
    std_x = std_x if std_x != 0 else 1.0
    return [np.asarray((np.asarray(w) - mean_x) / std_x, dtype=float) for w in windows]



def safe_percentile(x: Sequence[float], q: float) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.percentile(arr, q)) if arr.size else float("nan")



def build_probability_vector(choices: Sequence[int], syn_cfg: SyntheticLoadConfig) -> np.ndarray:
    if syn_cfg.pmf_mode == "uniform":
        probs = np.ones(len(choices), dtype=float)
        return probs / probs.sum()

    if syn_cfg.pmf_mode == "custom":
        if not syn_cfg.custom_pmf:
            raise ValueError("custom_pmf must be provided when pmf_mode='custom'.")
        probs = np.asarray([syn_cfg.custom_pmf.get(int(c), 0.0) for c in choices], dtype=float)
        if probs.sum() <= 0:
            raise ValueError("custom_pmf produced zero total probability.")
        return probs / probs.sum()

    raise ValueError(f"Unsupported pmf_mode: {syn_cfg.pmf_mode}")


# =============================================================================
# PART 1 — COUNT MODEL
# =============================================================================



def load_and_prepare_core_series(cfg: CountModelConfig) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    print("📥 Loading data...")
    df = pd.read_csv(cfg.data_path)
    df = ensure_datetime_index(df, cfg.timestamp_col)

    required = [cfg.id_col, cfg.grid_col, cfg.target_appliance]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[cfg.id_col])
    df = df.drop(columns=list(cfg.drop_cols), errors="ignore")

    print("⚙️ Building S_Total and A_Total...")
    S_Total = df.groupby(level=0)[cfg.grid_col].sum().rename("S_Total")

    A_Total = (
        df[[cfg.target_appliance]]
        .dropna()
        .groupby(level=0)[cfg.target_appliance]
        .sum()
        .rename("A_Total")
        .to_frame()
    )

    print("🔍 Building num_active ground truth...")
    active = df[[cfg.id_col, cfg.target_appliance]].dropna()
    active = active[active[cfg.target_appliance] > cfg.threshold_kw]
    num_active = (
        active.groupby(level=0)[cfg.id_col]
        .nunique()
        .rename("num_active")
    )

    print("🔗 Merging A_Total with num_active...")
    df_combined = A_Total.join(num_active, how="left")
    df_combined["num_active"] = df_combined["num_active"].fillna(0).astype(int)
    df_combined = df_combined.sort_index()

    if cfg.max_rows is not None:
        df_combined = df_combined.iloc[: cfg.max_rows].copy()

    return S_Total, A_Total, num_active, df_combined, df



def create_seq2point_dataset(df: pd.DataFrame, feature_col: str, target_col: str, window_size: int):
    half = window_size // 2
    X, y, times = [], [], []
    for i in tqdm(range(half, len(df) - half), desc="Creating Seq2Point windows"):
        window = df[feature_col].iloc[i - half : i + half + 1].to_numpy(dtype=float)
        X.append(window)
        y.append(df[target_col].iloc[i])
        times.append(df.index[i])
    return X, y, times



def gen_feats(window: np.ndarray, timestamp: pd.Timestamp) -> Dict[str, float]:
    w = np.asarray(window, dtype=float)
    center = len(w) // 2
    diff = np.diff(w)
    lags = {f"lag_{s}": w[center - s] if center - s >= 0 else 0.0 for s in [1, 5, 15, 30, 75, 150, 300]}
    rolls = {
        "roll_mean_1min": float(w[max(0, center - 7) : center + 1].mean()),
        "roll_std_1min": float(w[max(0, center - 7) : center + 1].std()),
        "roll_mean_5min": float(w[max(0, center - 37) : center + 1].mean()),
        "roll_std_5min": float(w[max(0, center - 37) : center + 1].std()),
    }
    peak = int(np.argmax(w)) if len(w) else 0
    slope = float(np.polyfit(np.arange(len(w)), w, 1)[0]) if len(w) > 1 else 0.0
    feats = {
        "mean": float(w.mean()),
        "std": float(w.std()),
        "max": float(w.max()),
        "min": float(w.min()),
        "range": float(w.max() - w.min()),
        "leading_edges": float(np.sum(diff > 30)),
        "trailing_edges": float(np.sum(diff < -30)),
        "slope": slope,
        "iqr": float(np.percentile(w, 75) - np.percentile(w, 25)),
        "slope_in": float(w[peak] - w[0]),
        "slope_out": float(w[-1] - w[peak]),
        "hour": float(timestamp.hour),
        "dayofweek": float(timestamp.dayofweek),
        **lags,
        **rolls,
    }
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in feats.items()}



def make_feature_frames(
    X_train: Sequence[np.ndarray],
    X_val: Sequence[np.ndarray],
    X_test: Sequence[np.ndarray],
    time_train: Sequence[pd.Timestamp],
    time_val: Sequence[pd.Timestamp],
    time_test: Sequence[pd.Timestamp],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("🔧 Generating engineered features...")
    train_df = pd.DataFrame([
        gen_feats(w, t) for w, t in tqdm(zip(X_train, time_train), total=len(X_train), desc="Train features")
    ])
    val_df = pd.DataFrame([
        gen_feats(w, t) for w, t in tqdm(zip(X_val, time_val), total=len(X_val), desc="Val features")
    ])
    test_df = pd.DataFrame([
        gen_feats(w, t) for w, t in tqdm(zip(X_test, time_test), total=len(X_test), desc="Test features")
    ])

    train_df["num_active"] = y_train
    val_df["num_active"] = y_val
    test_df["num_active"] = y_test
    return train_df, val_df, test_df



def split_and_prepare_feature_data(df_combined: pd.DataFrame, cfg: CountModelConfig):
    print("📦 Creating Seq2Point dataset...")
    X_raw, y_raw, time_raw = create_seq2point_dataset(df_combined, "A_Total", "num_active", cfg.window_size)

    total = len(X_raw)
    dev_split = int(total * cfg.development_fraction)
    val_start = int(dev_split * (1 - cfg.validation_fraction_within_development))

    X_train = np.asarray(X_raw[:val_start], dtype=float)
    y_train = np.asarray(y_raw[:val_start], dtype=float)
    X_val = np.asarray(X_raw[val_start:dev_split], dtype=float)
    y_val = np.asarray(y_raw[val_start:dev_split], dtype=float)
    X_test = np.asarray(X_raw[dev_split:], dtype=float)
    y_test = np.asarray(y_raw[dev_split:], dtype=float)

    time_train = time_raw[:val_start]
    time_val = time_raw[val_start:dev_split]
    time_test = time_raw[dev_split:]

    mean_x = float(X_train.mean())
    std_x = float(X_train.std())
    X_train_n = normalise_windows(X_train, mean_x, std_x)
    X_val_n = normalise_windows(X_val, mean_x, std_x)
    X_test_n = normalise_windows(X_test, mean_x, std_x)

    train_df, val_df, test_df = make_feature_frames(
        X_train_n, X_val_n, X_test_n,
        time_train, time_val, time_test,
        y_train, y_val, y_test,
    )
    return train_df, val_df, test_df, y_test, pd.Series(time_test, name="Time")



def train_count_model(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: CountModelConfig) -> TabularPredictor:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    predictor_path = f"{cfg.predictor_path_prefix}_{cfg.target_appliance}_{timestamp}"
    if os.path.exists(predictor_path):
        rmtree(predictor_path)

    print("🚀 Training AutoGluon regressor for num_active...")
    predictor = TabularPredictor(
        label="num_active",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=predictor_path,
    ).fit(
        train_data=train_df,
        tuning_data=val_df,
        time_limit=cfg.ag_time_limit_seconds,
        use_bag_holdout=True,
        presets=cfg.ag_presets,
    )
    return predictor



def evaluate_count_model(predictor: TabularPredictor, test_df: pd.DataFrame, y_test: np.ndarray, time_test: pd.Series) -> pd.DataFrame:
    print("✅ Evaluating count model...")
    y_pred_float = predictor.predict(test_df.drop(columns=["num_active"]))
    y_pred_rounded = np.round(y_pred_float).astype(int)
    y_pred_rounded[y_pred_rounded < 0] = 0

    mae = mean_absolute_error(y_test, y_pred_rounded)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred_rounded))
    r2 = r2_score(y_test, y_pred_rounded)

    print("\n--- Count Model Evaluation ---")
    print(f"MAE :  {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²  :  {r2:.4f}")
    print("-------------------------------\n")

    return pd.DataFrame({
        "Time": time_test.values,
        "Actual_Count": y_test,
        "Predicted_Float": np.asarray(y_pred_float),
        "Predicted_Rounded": y_pred_rounded,
    }).set_index("Time")



def build_preliminary_weighted_regression(count_results: pd.DataFrame, S_Total: pd.Series):
    df_min_pred = pd.DataFrame({
        "count": count_results["Predicted_Rounded"],
        "S_Total": S_Total.reindex(count_results.index).ffill(),
    }).dropna()

    freq = df_min_pred["count"].value_counts().sort_index()
    summary = df_min_pred.groupby("count")["S_Total"].agg(
        mean="mean",
        max="max",
        min="min",
        p90=lambda x: np.percentile(x, 90),
    ).reset_index()
    summary["freq"] = summary["count"].map(freq)

    models: Dict[str, LinearRegression] = {}
    print("📈 Preliminary weighted regressions on observed/predicted counts...")
    for stat in ["mean", "max", "min", "p90"]:
        lr = LinearRegression().fit(summary[["count"]], summary[stat], sample_weight=summary["freq"])
        models[stat] = lr
        print(f"Weighted {stat:>4s} fit: S_Total ≈ {lr.coef_[0]:.3f}·count + {lr.intercept_:.3f}")

    return summary, models



def run_count_model_pipeline(cfg: CountModelConfig) -> Tuple[PipelineArtifacts, pd.DataFrame]:
    set_thread_env(cfg)
    S_Total, A_Total, num_active, df_combined, raw_df = load_and_prepare_core_series(cfg)
    train_df, val_df, test_df, y_test, time_test = split_and_prepare_feature_data(df_combined, cfg)
    predictor = train_count_model(train_df, val_df, cfg)
    count_results = evaluate_count_model(predictor, test_df, y_test, time_test)
    preliminary_summary, preliminary_models = build_preliminary_weighted_regression(count_results, S_Total)

    artifacts = PipelineArtifacts(
        S_Total=S_Total,
        A_Total=A_Total,
        num_active=num_active,
        df_combined=df_combined,
        predictor=predictor,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        y_test=y_test,
        time_test=time_test,
        count_results=count_results,
        preliminary_summary=preliminary_summary,
        preliminary_models=preliminary_models,
    )
    return artifacts, raw_df


# =============================================================================
# PART 2 — SYNTHETIC LOAD CREATION (CONFIGURABLE)
# =============================================================================



def build_session_library(raw_df: pd.DataFrame, cfg: CountModelConfig, syn_cfg: SyntheticLoadConfig):
    print(f"🧱 Building EV session library using mode='{syn_cfg.session_library_mode}'...")
    appliance_df = raw_df[[cfg.id_col, cfg.target_appliance]].dropna().copy()

    if syn_cfg.session_library_mode == "full_runs":
        library: List[np.ndarray] = []
        for _, group in tqdm(appliance_df.groupby(cfg.id_col), desc="Extracting contiguous EV runs"):
            runs = contiguous_on_runs(group, cfg.target_appliance, cfg.threshold_kw)
            library.extend([run for run in runs if len(run) > 0])
        if not library:
            raise RuntimeError("No EV runs found for the full_runs session library.")
        return library

    if syn_cfg.session_library_mode == "minute_fragments":
        fragments = appliance_df.loc[appliance_df[cfg.target_appliance] > cfg.threshold_kw, cfg.target_appliance].astype(float).to_numpy()
        if fragments.size == 0:
            raise RuntimeError("No ON-minute fragments found for the minute_fragments session library.")
        return fragments

    raise ValueError(f"Unsupported session_library_mode: {syn_cfg.session_library_mode}")



def sample_count(rng: np.random.Generator, syn_cfg: SyntheticLoadConfig) -> int:
    choices = np.asarray(syn_cfg.active_segment_count_choices, dtype=int)
    probs = build_probability_vector(choices, syn_cfg)
    return int(rng.choice(choices, p=probs))



def project_full_run_to_segment(run: np.ndarray, length: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    run = np.asarray(run, dtype=float)
    if length <= 0:
        return np.zeros(0, dtype=float)

    if mode == "random_crop_or_pad":
        out = np.zeros(length, dtype=float)
        if len(run) >= length:
            start = int(rng.integers(0, len(run) - length + 1))
            return run[start : start + length].copy()
        start = int(rng.integers(0, length - len(run) + 1))
        out[start : start + len(run)] = run
        return out

    if mode == "resample_to_segment":
        if len(run) == 1:
            return np.full(length, float(run[0]), dtype=float)
        old_x = np.linspace(0, 1, len(run))
        new_x = np.linspace(0, 1, length)
        return np.interp(new_x, old_x, run).astype(float)

    raise ValueError(f"Unsupported full_run_projection_mode: {mode}")



def sample_minute_power(session_library, session_library_mode: str, rng: np.random.Generator, m: int) -> float:
    if m <= 0:
        return 0.0

    if session_library_mode == "minute_fragments":
        picks = rng.choice(session_library, size=m, replace=True)
        return float(np.sum(picks))

    if session_library_mode == "full_runs":
        runs = rng.choice(np.arange(len(session_library)), size=m, replace=True)
        total = 0.0
        for idx in runs:
            run = np.asarray(session_library[int(idx)], dtype=float)
            minute_idx = int(rng.integers(0, len(run)))
            total += float(run[minute_idx])
        return total

    raise ValueError(f"Unsupported session_library_mode: {session_library_mode}")



def sample_segment_profile(
    session_library,
    session_library_mode: str,
    rng: np.random.Generator,
    m: int,
    segment_length: int,
    projection_mode: str,
) -> np.ndarray:
    if segment_length <= 0:
        return np.zeros(0, dtype=float)
    if m <= 0:
        return np.zeros(segment_length, dtype=float)

    if session_library_mode == "minute_fragments":
        return np.asarray([
            sample_minute_power(session_library, session_library_mode, rng, m)
            for _ in range(segment_length)
        ], dtype=float)

    if session_library_mode == "full_runs":
        chosen = rng.choice(np.arange(len(session_library)), size=m, replace=True)
        composite = np.zeros(segment_length, dtype=float)
        for idx in chosen:
            run = np.asarray(session_library[int(idx)], dtype=float)
            composite += project_full_run_to_segment(run, segment_length, rng, projection_mode)
        return composite

    raise ValueError(f"Unsupported session_library_mode: {session_library_mode}")



def generate_synthetic_ev_trace(
    count_seed_series: pd.Series,
    session_library,
    syn_cfg: SyntheticLoadConfig,
    fixed_count: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (synthetic_count_series, synthetic_ev_kw_series).

    count_seed_series is usually the predicted rounded count series from the ACE model.
    Active periods are taken from count_seed_series > 0.
    """
    rng = np.random.default_rng(syn_cfg.random_seed if fixed_count is None else syn_cfg.random_seed + int(fixed_count))
    idx = count_seed_series.index
    active_mask = count_seed_series.fillna(0).astype(float) > 0

    synth_count = pd.Series(0, index=idx, dtype=int)
    synth_ev = pd.Series(0.0, index=idx, dtype=float)

    if syn_cfg.count_assignment_mode == "segment_constant":
        segments = contiguous_true_segments(active_mask)
        for start, end in segments:
            seg_idx = idx[(idx >= start) & (idx <= end)]
            seg_len = len(seg_idx)
            if seg_len == 0:
                continue
            m = int(fixed_count) if fixed_count is not None else sample_count(rng, syn_cfg)
            synth_count.loc[seg_idx] = m
            synth_ev.loc[seg_idx] = sample_segment_profile(
                session_library=session_library,
                session_library_mode=syn_cfg.session_library_mode,
                rng=rng,
                m=m,
                segment_length=seg_len,
                projection_mode=syn_cfg.full_run_projection_mode,
            )
        return synth_count, synth_ev

    if syn_cfg.count_assignment_mode == "minute_resample":
        for ts in idx[active_mask]:
            m = int(fixed_count) if fixed_count is not None else sample_count(rng, syn_cfg)
            synth_count.loc[ts] = m
            synth_ev.loc[ts] = sample_minute_power(
                session_library=session_library,
                session_library_mode=syn_cfg.session_library_mode,
                rng=rng,
                m=m,
            )
        return synth_count, synth_ev

    raise ValueError(f"Unsupported count_assignment_mode: {syn_cfg.count_assignment_mode}")



def build_background_load(S_Total: pd.Series, A_Total: pd.DataFrame, target_index: pd.Index) -> pd.Series:
    a_total = A_Total["A_Total"].reindex(target_index).fillna(0.0)
    s_total = S_Total.reindex(target_index).ffill().bfill()
    background = s_total - a_total
    return background.rename("Background_kW")



def combine_background_and_synthetic_ev(background: pd.Series, synth_ev: pd.Series) -> pd.Series:
    return (background.reindex(synth_ev.index).fillna(0.0) + synth_ev.fillna(0.0)).rename("Synthetic_Grid_kW")



def summarise_by_count(count_series: pd.Series, grid_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"count": count_series, "S_Total": grid_series}).dropna()
    freq = df["count"].value_counts().sort_index()
    summary = df.groupby("count")["S_Total"].agg(
        mean="mean",
        max="max",
        min="min",
        p90=lambda x: np.percentile(x, 90),
    ).reset_index()
    summary["freq"] = summary["count"].map(freq)
    return summary.sort_values("count").reset_index(drop=True)



def fit_weighted_regression_from_summary(summary: pd.DataFrame, stats: Sequence[str] = ("mean", "max", "p90")):
    models: Dict[str, LinearRegression] = {}
    for stat in stats:
        lr = LinearRegression().fit(summary[["count"]], summary[stat], sample_weight=summary["freq"])
        models[stat] = lr
        print(f"Weighted {stat:>4s}: S ≈ {lr.coef_[0]:.3f}·count + {lr.intercept_:.3f}")
    return models



def _select_mc_stat_series(
    synth_grid: pd.Series,
    synth_count: pd.Series,
    mc_stat_scope: str,
) -> pd.Series:
    if mc_stat_scope == "full_series":
        stat_series = synth_grid.dropna()
    elif mc_stat_scope == "active_only":
        stat_series = synth_grid.loc[synth_count.fillna(0).astype(int) > 0].dropna()
    else:
        raise ValueError(f"Unsupported mc_stat_scope: {mc_stat_scope}")

    if stat_series.empty:
        raise RuntimeError(
            "No timesteps available for Monte Carlo statistics under the chosen mc_stat_scope."
        )
    return stat_series



def monte_carlo_validate_fixed_counts(
    count_seed_series: pd.Series,
    session_library,
    S_Total: pd.Series,
    A_Total: pd.DataFrame,
    syn_cfg: SyntheticLoadConfig,
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    For each fixed count M (e.g. 8, 9, 10), repeatedly generate a synthetic trace and
    collect distribution summaries for mean / max / p90.

    mc_stat_scope controls whether statistics are computed across the whole synthetic
    feeder trace ("full_series") or only timesteps where EV activity is present
    ("active_only").
    """
    mc_runs: Dict[int, pd.DataFrame] = {}
    background = build_background_load(S_Total, A_Total, count_seed_series.index)
    rows = []

    for fixed_count in syn_cfg.validation_fixed_counts:
        run_records = []
        for i in range(syn_cfg.monte_carlo_runs):
            run_cfg = SyntheticLoadConfig(**{**syn_cfg.__dict__, "random_seed": syn_cfg.random_seed + i})
            synth_count, synth_ev = generate_synthetic_ev_trace(
                count_seed_series=count_seed_series,
                session_library=session_library,
                syn_cfg=run_cfg,
                fixed_count=fixed_count,
            )
            synth_grid = combine_background_and_synthetic_ev(background, synth_ev)
            stat_series = _select_mc_stat_series(
                synth_grid=synth_grid,
                synth_count=synth_count,
                mc_stat_scope=syn_cfg.mc_stat_scope,
            )
            run_records.append({
                "run": i,
                "count": fixed_count,
                "mc_stat_scope": syn_cfg.mc_stat_scope,
                "mean": float(stat_series.mean()),
                "max": float(stat_series.max()),
                "p90": float(np.percentile(stat_series, 90)),
            })

        mc_df = pd.DataFrame(run_records)
        mc_runs[fixed_count] = mc_df

        for stat in ["mean", "max", "p90"]:
            empirical = float(mc_df[stat].mean())
            ci_low = float(np.percentile(mc_df[stat], 2.5))
            ci_high = float(np.percentile(mc_df[stat], 97.5))
            rows.append({
                "Count": fixed_count,
                "Statistic": stat.upper() if stat != "p90" else "P90",
                "Empirical": empirical,
                "CI_Low": ci_low,
                "CI_High": ci_high,
            })

    validation_summary = pd.DataFrame(rows)
    return mc_runs, validation_summary



def compare_empirical_vs_fitted(validation_summary: pd.DataFrame, regression_models: Dict[str, LinearRegression]) -> pd.DataFrame:
    mapping = {"MEAN": "mean", "MAX": "max", "P90": "p90"}
    rows = []
    for _, row in validation_summary.iterrows():
        count = int(row["Count"])
        stat_label = str(row["Statistic"])
        stat_key = mapping[stat_label]
        fitted = float(regression_models[stat_key].predict(np.array([[count]], dtype=float))[0])
        empirical = float(row["Empirical"])
        pct_diff = ((fitted - empirical) / empirical * 100.0) if empirical != 0 else np.nan
        rows.append({
            "Count": count,
            "Statistic": stat_label,
            "Empirical": empirical,
            "Fitted": fitted,
            "Pct_diff(%)": pct_diff,
            "CI_Low": row["CI_Low"],
            "CI_High": row["CI_High"],
        })
    return pd.DataFrame(rows).sort_values(["Count", "Statistic"]).reset_index(drop=True)


# =============================================================================
# PLOTTING
# =============================================================================



def plot_count_predictions(count_results: pd.DataFrame, S_Total: pd.Series, A_Total: pd.DataFrame, target_appliance: str) -> None:
    time_slice = count_results.index
    s_total_slice = S_Total.reindex(time_slice).ffill()
    a_total_slice = A_Total["A_Total"].reindex(time_slice).ffill()

    fig, ax1 = plt.subplots(figsize=(18, 8))
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Active Count", color="tab:blue")
    ax1.plot(time_slice, count_results["Actual_Count"], label="Ground Truth", color="tab:blue", linewidth=2.5)
    ax1.plot(time_slice, count_results["Predicted_Rounded"], label="Predicted", color="red", linestyle=":", marker="o", markersize=2, alpha=0.7)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Power (kW)")
    ax2.plot(s_total_slice.index, s_total_slice.values, label="S_Total (Grid)", color="green", alpha=0.6)
    ax2.plot(a_total_slice.index, a_total_slice.values, label=f"A_Total ({target_appliance})", color="purple", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.legend(loc="upper right")

    fig.suptitle(f"Appliance Count vs. Grid Power – {target_appliance}", fontsize=16)
    fig.autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_weighted_regression(summary: pd.DataFrame, models: Dict[str, LinearRegression], title: str, max_count_to_plot: int = 10) -> None:
    plt.figure(figsize=(12, 7))
    for stat, marker, style in [("mean", "o", "-"), ("max", "s", "--"), ("p90", "v", "-."), ("min", "d", ":")]:
        if stat in summary.columns:
            plt.plot(summary["count"], summary[stat], marker + style, linewidth=2, label=stat.upper())

    x_line = np.linspace(0, max_count_to_plot, 200).reshape(-1, 1)
    for stat, lr in models.items():
        plt.plot(x_line, lr.predict(x_line), linewidth=2.5, linestyle="--", label=f"Weighted {stat} fit")

    plt.xlim(0, max_count_to_plot)
    plt.xlabel("EV Count")
    plt.ylabel("Substation Power (kW)")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()



def plot_synthetic_trace(real_grid: pd.Series, synth_grid: pd.Series, pred_count: pd.Series, synth_count: pd.Series, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(18, 8))
    ax1.set_xlabel("Time")
    ax1.set_ylabel("EV Count")
    ax1.plot(pred_count.index, pred_count.values, label="Predicted Count", alpha=0.7)
    ax1.plot(synth_count.index, synth_count.values, label="Synthetic Count", alpha=0.8)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Power (kW)")
    ax2.plot(real_grid.index, real_grid.values, label="Real Grid Load", color="green", alpha=0.5)
    ax2.plot(synth_grid.index, synth_grid.values, label="Synthetic Grid", color="red", alpha=0.7)
    ax2.legend(loc="upper right")

    plt.title(title)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()



def plot_mc_histograms(mc_runs: Dict[int, pd.DataFrame], comparison_df: pd.DataFrame, stat: str = "mean") -> None:
    counts = sorted(mc_runs.keys())
    fig, axes = plt.subplots(len(counts), 1, figsize=(10, 3.5 * len(counts)), sharex=False)
    if len(counts) == 1:
        axes = [axes]

    for ax, count in zip(axes, counts):
        mc_df = mc_runs[count]
        values = mc_df[stat].to_numpy()
        ax.hist(values, bins=30, alpha=0.75)
        empirical = values.mean()
        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)
        fitted = comparison_df.loc[(comparison_df["Count"] == count) & (comparison_df["Statistic"] == ("P90" if stat == "p90" else stat.upper())), "Fitted"].iloc[0]
        ax.axvline(empirical, linestyle="-", linewidth=2, label=f"Empirical mean ({empirical:.2f})")
        ax.axvline(ci_low, linestyle="--", linewidth=1.5, label="95% CI")
        ax.axvline(ci_high, linestyle="--", linewidth=1.5)
        ax.axvline(fitted, linestyle=":", linewidth=2, label=f"Fit mean ({fitted:.2f})")
        ax.set_title(f"MC distribution of {stat.upper()} @ {count} EVs vs. fit")
        ax.set_xlabel("Substation Load (kW)")
        ax.set_ylabel("Frequency")
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


# =============================================================================
# END-TO-END REBUILD
# =============================================================================



def run_full_rebuild(count_cfg: CountModelConfig, syn_cfg: SyntheticLoadConfig):
    artifacts, raw_df = run_count_model_pipeline(count_cfg)

    pred_count = artifacts.count_results["Predicted_Rounded"].astype(int)
    session_library = build_session_library(raw_df, count_cfg, syn_cfg)
    background = build_background_load(artifacts.S_Total, artifacts.A_Total, pred_count.index)

    print("\n🧪 Building synthetic 1–7 scenario trace...")
    synth_count, synth_ev = generate_synthetic_ev_trace(pred_count, session_library, syn_cfg, fixed_count=None)
    synth_grid = combine_background_and_synthetic_ev(background, synth_ev)
    synth_summary = summarise_by_count(synth_count, synth_grid)

    print("\n📈 Weighted regressions on synthetic 1–7 trace...")
    synth_models = fit_weighted_regression_from_summary(synth_summary, stats=("mean", "max", "p90"))

    print("\n🎲 Running Monte Carlo validation for fixed counts...")
    mc_runs, validation_summary = monte_carlo_validate_fixed_counts(
        count_seed_series=pred_count,
        session_library=session_library,
        S_Total=artifacts.S_Total,
        A_Total=artifacts.A_Total,
        syn_cfg=syn_cfg,
    )

    comparison_df = compare_empirical_vs_fitted(validation_summary, synth_models)

    outputs = {
        "artifacts": artifacts,
        "raw_df": raw_df,
        "session_library": session_library,
        "background": background,
        "synthetic_count": synth_count,
        "synthetic_ev": synth_ev,
        "synthetic_grid": synth_grid,
        "synthetic_summary": synth_summary,
        "synthetic_models": synth_models,
        "mc_runs": mc_runs,
        "validation_summary": validation_summary,
        "comparison_df": comparison_df,
    }
    return outputs


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    count_cfg = CountModelConfig(
        data_path="1minute_data_newyork.csv",
        target_appliance="car1",
        threshold_kw=0.5,
        window_size=599,
        max_rows=1_000_000,
        ag_time_limit_seconds=30 * 60,
        ag_presets="best_quality",
    )

    syn_cfg = SyntheticLoadConfig(
        # Defaults chosen to match the likely thesis implementation:
        session_library_mode="full_runs",
        count_assignment_mode="segment_constant",

        # Synthetic scenario counts / validation:
        active_segment_count_choices=tuple(range(1, 8)),
        validation_fixed_counts=(8, 9, 10),
        monte_carlo_runs=500,

        # Optional knobs:
        pmf_mode="uniform",
        custom_pmf=None,
        full_run_projection_mode="random_crop_or_pad",
        max_count_to_plot=10,
        random_seed=42,
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"nilm_mc_forecasting_rebuild_{count_cfg.target_appliance}_{timestamp}.log"

    with open(log_filename, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = sys.stdout
        try:
            outputs = run_full_rebuild(count_cfg, syn_cfg)

            print("\n=== FINAL COMPARISON TABLE ===")
            print(outputs["comparison_df"].round(3))

            # Optional plots
            plot_count_predictions(
                outputs["artifacts"].count_results,
                outputs["artifacts"].S_Total,
                outputs["artifacts"].A_Total,
                count_cfg.target_appliance,
            )
            plot_weighted_regression(
                outputs["synthetic_summary"],
                outputs["synthetic_models"],
                title="Synthetic Loads vs. EV Count — Empirical 1–7, Extrapolate to 10",
                max_count_to_plot=syn_cfg.max_count_to_plot,
            )
            plot_synthetic_trace(
                real_grid=outputs["artifacts"].S_Total.reindex(outputs["synthetic_grid"].index).ffill(),
                synth_grid=outputs["synthetic_grid"],
                pred_count=outputs["artifacts"].count_results["Predicted_Rounded"],
                synth_count=outputs["synthetic_count"],
                title="Predicted vs Synthetic Count & Grid Load",
            )
            plot_mc_histograms(outputs["mc_runs"], outputs["comparison_df"], stat="mean")
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout
