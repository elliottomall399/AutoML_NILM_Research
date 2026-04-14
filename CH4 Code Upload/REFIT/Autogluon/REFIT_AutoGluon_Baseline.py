# -*- coding: utf-8 -*-
"""

"House1": 
    "Fridge": "Appliance1", done
    "Freezer": "Appliance2",
    "Freezer": "Appliance3",
    "Washer Dryer": "Appliance4",
    "Washing Machine": "Appliance5",done
    "Dishwasher": "Appliance6",done
    "Computer": "Appliance7",
    "Television Site": "Appliance8",
    "Electric Heater": "Appliance9", done


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

# --- Load Data ---
file_name = "CLEAN_House1.csv"
df = pd.read_csv(file_name, parse_dates=["Time"])
df.set_index("Time", inplace=True)

# --- Resample to 8-second intervals ---
df_resampled = df.resample("8s").mean().interpolate("linear")

# --- Parameters ---
target_appliance = "Appliance5"
window_size = 599
threshold = 5

# --- Clean & Subset Data ---
appliance_cols = [f"Appliance{i}" for i in range(1, 10)]
df_clean = df_resampled[["Aggregate"] + appliance_cols].dropna()
df_clean = df_clean.iloc[:1_000_000].copy()

# --- Create Seq2Point dataset ---
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
X_raw, y_raw, time_raw = create_seq2point_dataset(df_clean, "Aggregate", target_appliance, window_size)

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

mean_y = y_train_raw.mean()
std_y = y_train_raw.std()

y_train_norm = (y_train_raw - mean_y) / std_y
y_val_norm = (y_val_raw - mean_y) / std_y
y_test_norm = (y_test_raw - mean_y) / std_y

# --- Baseline Level 2: Only center agg + time-of-day features ---
def generate_baseline_features(window, timestamp):
    center = len(window) // 2
    return {
        "agg_center": float(window[center]),
        "hour": timestamp.hour,
        "dayofweek": timestamp.dayofweek,
    }

print("🔧 Generating baseline features (Level 2)...")
df_train = pd.DataFrame([generate_baseline_features(w, t) for w, t in tqdm(zip(X_train_raw, time_raw[:val_index]))])
df_val = pd.DataFrame([generate_baseline_features(w, t) for w, t in tqdm(zip(X_val_raw, time_raw[val_index:split_index]))])
df_test = pd.DataFrame([generate_baseline_features(w, t) for w, t in tqdm(zip(X_test_raw, time_raw[split_index:]))])

df_train[target_appliance] = y_train_norm
df_val[target_appliance] = y_val_norm
df_test[target_appliance] = y_test_norm

# --- Train AutoGluon ---
print("🚀 Training AutoGluon (Baseline Level 2)...")
predictor = TabularPredictor(label=target_appliance, problem_type='regression', eval_metric='mean_absolute_error')
predictor.fit(train_data=df_train, tuning_data=df_val, use_bag_holdout=True, time_limit=3600, presets='best_quality')

# --- Predict ---
y_pred_norm = predictor.predict(df_test.drop(columns=[target_appliance]))
y_pred = y_pred_norm * std_y + mean_y
y_true = df_test[target_appliance] * std_y + mean_y

# --- Duration threshold filter ---
def duration_threshold_filter(pred, threshold=threshold, min_on_duration=3):
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
    print(f"🔍 RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f} W")
    print(f"📉 MAE: {mean_absolute_error(y_true, y_pred):.2f} W")
    print(f"📈 R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"⚡ SAE: {abs(y_true.sum() - y_pred.sum()) / y_true.sum():.4f}\n")

print_metrics(y_true, y_pred, label="All Test Timesteps (Raw)")
print_metrics(y_true, y_pred_filtered, label="All Test Timesteps (Post-Processed)")

# --- Save Results ---
results_df = pd.DataFrame({
    "Time": time_test_full,
    "GroundTruth": y_true.values,
    "Prediction_Raw": y_pred.values,
    "Prediction_PostProcessed": y_pred_filtered.values,
})
results_df.to_csv(f"{target_appliance}_baseline_lvl2_predictions_1mil_1hr_AG.csv", index=False)
print("csv saved!")

# --- Plotting ---
plt.figure(figsize=(14, 5))
plt.plot(time_test_full, y_true.values, label="Actual", alpha=0.6)
plt.plot(time_test_full, y_pred.values, label="Prediction (Raw)", linestyle='--', alpha=0.7)
plt.plot(time_test_full, y_pred_filtered.values, label="Post-Processed", linestyle='--', alpha=0.9)
plt.title(f"Prediction vs Actual ({target_appliance}) - Full Test Timeline")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#Extra plot!!
plt.figure(figsize=(14, 5))
plt.plot(time_test_full, y_true.values, label="Actual", alpha=0.6)
plt.plot(time_test_full, y_pred.values, label="Prediction (Raw)", linestyle='--', alpha=0.7)
plt.title(f"Prediction vs Actual ({target_appliance}) - Full Test Timeline")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.3, label="Raw")
plt.plot([0, y_true.max()], [0, y_true.max()], 'r--', label="Ideal")
plt.xlabel("Actual Power (W)")
plt.ylabel("Predicted Power (W)")
plt.title("Raw Prediction vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred_filtered, alpha=0.3, label="Post-Processed")
plt.plot([0, y_true.max()], [0, y_true.max()], 'r--', label="Ideal")
plt.xlabel("Actual Power (W)")
plt.ylabel("Predicted Power (W)")
plt.title("Post-Processed Prediction vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

