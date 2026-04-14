# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:29:08 2025


@author: Administrator
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import random

random.seed(42)
np.random.seed(42)

# --- Load and preprocess Pecan Street data ---
df = pd.read_csv("1minute_data_newyork.csv")
df = df[df["dataid"] == 3000].copy()

df["localminute"] = pd.to_datetime(df["localminute"], utc=True, errors='coerce')
df.dropna(subset=["localminute"], inplace=True)
df.set_index("localminute", inplace=True)

df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq="1min"))

# --- Visual check (optional) ---
plt.figure(figsize=(14, 4))
plt.plot(df["grid"].iloc[:1000], label="Grid", alpha=0.7)
plt.plot(df["car1"].iloc[:1000], label="Car1", alpha=0.7)
plt.title("Grid vs Car1 (First 1000 points)")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Parameters ---
target_appliance = "car1" #Configure appliance
window_size = 599
threshold = 0.05  # kW
min_on_duration = 3

# --- Clean & Subset ---
df_clean = df[["grid", target_appliance]].dropna()
df_clean = df_clean.iloc[:1_000_000].copy()

# --- Create Seq2Point Dataset ---
def create_seq2point_dataset(df, feature_col, target_col, window_size):
    half = window_size // 2
    X, y, center_times = [], [], []
    for i in range(half, len(df) - half):
        window = df[feature_col].iloc[i - half:i + half + 1].values
        X.append(window)
        y.append(df[target_col].iloc[i])
        center_times.append(df.index[i])
    return X, y, center_times

print("📦 Creating Seq2Point dataset...")
X_raw, y_raw, time_raw = create_seq2point_dataset(df_clean, "grid", target_appliance, window_size)

# --- Normalisation ---
split_index = int(len(X_raw) * 0.9)
val_index = int(split_index * 0.9)

X_train_raw = np.array(X_raw[:val_index])
y_train_raw = np.array(y_raw[:val_index])
X_val_raw = np.array(X_raw[val_index:split_index])
y_val_raw = np.array(y_raw[val_index:split_index])
X_test_raw = np.array(X_raw[split_index:])
y_test_raw = np.array(y_raw[split_index:])
time_test_full = pd.Series(time_raw[split_index:], name="Time")

mean_X = X_train_raw.mean()
std_X = X_train_raw.std()
mean_y = y_train_raw.mean()
std_y = y_train_raw.std()

X_train_norm = [(x - mean_X) / std_X for x in X_train_raw]
X_val_norm = [(x - mean_X) / std_X for x in X_val_raw]
X_test_norm = [(x - mean_X) / std_X for x in X_test_raw]

y_train_norm = (y_train_raw - mean_y) / std_y
y_val_norm = (y_val_raw - mean_y) / std_y
y_test_norm = (y_test_raw - mean_y) / std_y

# --- Advanced Feature Engineering ---
def generate_features(window, timestamp):
    window = np.array(window)
    center = len(window) // 2
    diff = np.diff(window)

    lags = {f"lag_{s}": window[center - s] if center - s >= 0 else 0 for s in [1, 5, 15, 30, 75, 150, 300]}
    roll_feats = {
        "roll_mean_1min": window[center-7:center+1].mean(),
        "roll_std_1min": window[center-7:center+1].std(),
        "roll_mean_5min": window[center-37:center+1].mean(),
        "roll_std_5min": window[center-37:center+1].std(),
        "roll_min_5min": window[center-37:center+1].min(),
        "roll_max_5min": window[center-37:center+1].max(),
    }
    peak = np.argmax(window)
    engineered = {
        "mean": window.mean(),
        "std": window.std(),
        "max": window.max(),
        "min": window.min(),
        "range": window.max() - window.min(),
        "leading_edges": np.sum(diff > 0.03),  # 30W threshold in kW
        "trailing_edges": np.sum(diff < -0.03),
        "slope": np.polyfit(np.arange(len(window)), window, 1)[0],
        "iqr": np.percentile(window, 75) - np.percentile(window, 25),
        "slope_in": window[peak] - window[0],
        "slope_out": window[-1] - window[peak],
        "hour": timestamp.hour,
        "dayofweek": timestamp.dayofweek,
        **lags,
        **roll_feats
    }

    raw_window_feats = {f"w_{i}": float(v) for i, v in enumerate(window)}
    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in {**engineered, **raw_window_feats}.items()}

print("🔧 Generating full FE2 features (engineered + raw normalized)...")
df_train = pd.DataFrame([generate_features(w, t) for w, t in tqdm(zip(X_train_norm, time_raw[:val_index]))])
df_val = pd.DataFrame([generate_features(w, t) for w, t in tqdm(zip(X_val_norm, time_raw[val_index:split_index]))])
df_test = pd.DataFrame([generate_features(w, t) for w, t in tqdm(zip(X_test_norm, time_raw[split_index:]))])

df_train[target_appliance] = y_train_norm
df_val[target_appliance] = y_val_norm
df_test[target_appliance] = y_test_norm

# --- Train AutoGluon ---
print("🚀 Training AutoGluon (FE2)...")
predictor = TabularPredictor(label=target_appliance, problem_type='regression', eval_metric='mean_absolute_error')
predictor.fit(train_data=df_train, tuning_data=df_val, use_bag_holdout=True, time_limit=60*60, presets='best_quality')

# --- Predict ---
y_pred_norm = predictor.predict(df_test.drop(columns=[target_appliance]))
y_pred = y_pred_norm * std_y + mean_y
y_true = df_test[target_appliance] * std_y + mean_y

# --- Post-Processing ---
def duration_threshold_filter(pred, threshold=threshold, min_on_duration=min_on_duration):
    pred = pred.copy()
    on = pred > threshold
    filtered = on.copy()
    count = 0
    for i in range(len(on)):
        if on[i]: count += 1
        else:
            if 0 < count < min_on_duration:
                filtered[i - count:i] = False
            count = 0
    if 0 < count < min_on_duration:
        filtered[len(on) - count:] = False
    pred[~filtered] = 0
    return pred

y_pred_filtered = duration_threshold_filter(y_pred)

# --- Evaluation ---
def print_metrics(y_true, y_pred, label=""):
    print(f"--- {label} ---")
    print(f"🔍 RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f} kW")
    print(f"📉 MAE: {mean_absolute_error(y_true, y_pred):.4f} kW")
    print(f"📈 R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"⚡ SAE: {abs(y_true.sum() - y_pred.sum()) / y_true.sum():.4f}\n")

print_metrics(y_true, y_pred, label="Raw Prediction")
print_metrics(y_true, y_pred_filtered, label="Post-Processed Prediction")

# --- Save ---
results_df = pd.DataFrame({
    "Time": time_test_full,
    "GroundTruth": y_true.values,
    "Prediction_Raw": y_pred.values,
    "Prediction_PostProcessed": y_pred_filtered.values,
})
results_df.to_csv(f"{target_appliance}_FE2_predictions_pecanstreet_1hr_1mil.csv", index=False)
print("✅ Results saved to CSV.")

# --- Plotting ---
plt.figure(figsize=(14, 5))
plt.plot(time_test_full, y_true, label="Actual", alpha=0.6)
plt.plot(time_test_full, y_pred, label="Prediction (Raw)", linestyle='--', alpha=0.7)
plt.plot(time_test_full, y_pred_filtered, label="Post-Processed", linestyle='--', alpha=0.9)
plt.title(f"Prediction vs Actual ({target_appliance}) - Full Test Timeline")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.3, label="Raw")
plt.plot([0, y_true.max()], [0, y_true.max()], 'r--', label="Ideal")
plt.xlabel("Actual Power (kW)")
plt.ylabel("Predicted Power (kW)")
plt.title("Raw Prediction vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred_filtered, alpha=0.3, label="Post-Processed")
plt.plot([0, y_true.max()], [0, y_true.max()], 'r--', label="Ideal")
plt.xlabel("Actual Power (kW)")
plt.ylabel("Predicted Power (kW)")
plt.title("Post-Processed Prediction vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
