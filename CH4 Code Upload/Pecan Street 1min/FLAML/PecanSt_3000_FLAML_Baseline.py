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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import random
import time
from flaml import AutoML

start_time = time.time()

# Reproducibility
random.seed(42)
np.random.seed(42)

# --- Load and preprocess Pecan Street data ---
df = pd.read_csv("1minute_data_newyork.csv") #Dataset
df = df[df["dataid"] == 3000].copy()

df["localminute"] = pd.to_datetime(df["localminute"], utc=True, errors='coerce')
df.dropna(subset=["localminute"], inplace=True)
df.set_index("localminute", inplace=True)

# Resample to 1-minute intervals
df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq="1min"))

# --- Parameters ---
target_appliance = "car1" #Configure Appliance
window_size = 599
threshold = 0.05
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

# --- Split ---
split_index = int(len(X_raw) * 0.9)
val_index = int(split_index * 0.9)
X_train_raw = np.array(X_raw[:val_index])
y_train_raw = np.array(y_raw[:val_index])
X_val_raw = np.array(X_raw[val_index:split_index])
y_val_raw = np.array(y_raw[val_index:split_index])
X_test_raw = np.array(X_raw[split_index:])
y_test_raw = np.array(y_raw[split_index:])
time_test_full = pd.Series(time_raw[split_index:], name="Time")

# --- Normalisation ---
mean_y = y_train_raw.mean()
std_y = y_train_raw.std()
y_train_norm = (y_train_raw - mean_y) / std_y
y_val_norm = (y_val_raw - mean_y) / std_y
y_test_norm = (y_test_raw - mean_y) / std_y

# --- Baseline Feature Engineering ---
def generate_baseline_features(window, timestamp):
    center = len(window) // 2
    return {
        "agg_center": float(window[center]),
        "hour": timestamp.hour,
        "dayofweek": timestamp.dayofweek,
    }

print("🔧 Generating baseline features...")
df_train = pd.DataFrame([generate_baseline_features(w, t) for w, t in zip(X_train_raw, time_raw[:val_index])])
df_val = pd.DataFrame([generate_baseline_features(w, t) for w, t in zip(X_val_raw, time_raw[val_index:split_index])])
df_test = pd.DataFrame([generate_baseline_features(w, t) for w, t in zip(X_test_raw, time_raw[split_index:])])
df_train[target_appliance] = y_train_norm
df_val[target_appliance] = y_val_norm
df_test[target_appliance] = y_test_norm

# --- FLAML Training ---
print("🚀 Training FLAML...")
automl = AutoML()
automl_settings = {
    "time_budget": 3600,
    "metric": 'mae',
    "task": 'regression',
    "log_file_name": f"flaml_{target_appliance}_baseline.log",
    "eval_method": "holdout",
    "verbose": 2,
}
automl.fit(
    X_train=df_train.drop(columns=[target_appliance]),
    y_train=df_train[target_appliance],
    X_val=df_val.drop(columns=[target_appliance]),
    y_val=df_val[target_appliance],
    **automl_settings
)

# --- Predict ---
y_pred_norm = automl.predict(df_test.drop(columns=[target_appliance]))
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

print_metrics(y_true, y_pred, "Raw Prediction")
print_metrics(y_true, y_pred_filtered, "Post-Processed Prediction")

# --- Save Results ---
results_df = pd.DataFrame({
    "Time": time_test_full,
    "GroundTruth": y_true,
    "Prediction_Raw": y_pred,
    "Prediction_PostProcessed": y_pred_filtered,
})
results_df.to_csv(f"{target_appliance}_baseline_FLAML_predictions_pecanstreet.csv", index=False)
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

# --- Plotting ---
plt.figure(figsize=(14, 5))
plt.plot(time_test_full, y_true, label="Actual", alpha=0.6)
plt.plot(time_test_full, y_pred, label="Prediction (Raw)", linestyle='--', alpha=0.7)
#plt.plot(time_test_full, y_pred_filtered, label="Post-Processed", linestyle='--', alpha=0.9)
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

end_time = time.time()
print(f"🕒 Total Runtime: {((end_time - start_time) / 60):.2f} minutes")
