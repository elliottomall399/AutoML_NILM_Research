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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import random
from flaml import AutoML

random.seed(42)
np.random.seed(42)

file_name = "CLEAN_House1.csv"
df = pd.read_csv(file_name, parse_dates=["Time"])
df.set_index("Time", inplace=True)
df_resampled = df.resample("8s").mean().interpolate("linear")

target_appliance = "Appliance1"
window_size = 599
threshold = 5

appliance_cols = [f"Appliance{i}" for i in range(1, 10)]
df_clean = df_resampled[["Aggregate"] + appliance_cols].dropna()
df_clean = df_clean.iloc[:1_000_000].copy()

def create_seq2point_dataset(df, feature_col, target_col, window_size):
    half = window_size // 2
    X, y, center_times = [], [], []
    for i in range(half, len(df) - half):
        window = df[feature_col].iloc[i - half:i + half + 1].values
        X.append(window)
        y.append(df[target_col].iloc[i])
        center_times.append(df.index[i])
    return X, y, center_times

print("\U0001f4e6 Creating Seq2Point dataset...")
X_raw, y_raw, time_raw = create_seq2point_dataset(df_clean, "Aggregate", target_appliance, window_size)

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

def generate_advanced_features(window, timestamp):
    window = np.array(window)
    center = len(window) // 2
    diff = np.diff(window)

    lags = {f"lag_{s}": window[center - s] if center - s >= 0 else 0 for s in [1, 5, 15, 30, 75, 150, 300]}
    roll_feats = {
        "roll_mean_1min": window[center - 7:center + 1].mean(),
        "roll_std_1min": window[center - 7:center + 1].std(),
        "roll_mean_5min": window[center - 37:center + 1].mean(),
        "roll_std_5min": window[center - 37:center + 1].std(),
        "roll_min_5min": window[center - 37:center + 1].min(),
        "roll_max_5min": window[center - 37:center + 1].max(),
    }
    peak = np.argmax(window)
    engineered = {
        "mean": window.mean(),
        "std": window.std(),
        "max": window.max(),
        "min": window.min(),
        "range": window.max() - window.min(),
        "leading_edges": np.sum(diff > 30),
        "trailing_edges": np.sum(diff < -30),
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

print("🔧 Generating advanced features...")
df_train = pd.DataFrame([generate_advanced_features(w, t) for w, t in tqdm(zip(X_train_norm, time_raw[:val_index]))])
df_val = pd.DataFrame([generate_advanced_features(w, t) for w, t in tqdm(zip(X_val_norm, time_raw[val_index:split_index]))])
df_test = pd.DataFrame([generate_advanced_features(w, t) for w, t in tqdm(zip(X_test_norm, time_raw[split_index:]))])

df_train[target_appliance] = y_train_norm
df_val[target_appliance] = y_val_norm
df_test[target_appliance] = y_test_norm

print("🚀 Training FLAML...")
automl = AutoML()
automl_settings = {
    "time_budget": 60 * 60,
    "metric": 'mae',
    "task": 'regression',
    "log_file_name": f"flaml_{target_appliance}.log",
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
print("Best learner:", automl.best_estimator)
print("Best config:", automl.best_config)
print("Best validation score:", automl.best_loss)

y_pred_norm = automl.predict(df_test.drop(columns=[target_appliance]))
y_pred = y_pred_norm * std_y + mean_y
y_true = df_test[target_appliance] * std_y + mean_y

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

def print_metrics(y_true, y_pred, label=""):
    print(f"--- {label} ---")
    print(f"\U0001f50d RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f} W")
    print(f"\U0001f4c9 MAE: {mean_absolute_error(y_true, y_pred):.2f} W")
    print(f"\U0001f4c8 R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"\u26a1 SAE: {abs(y_true.sum() - y_pred.sum()) / y_true.sum():.4f}\n")

print_metrics(y_true, y_pred, label="All Test Timesteps (Raw)")
print_metrics(y_true, y_pred_filtered, label="All Test Timesteps (Post-Processed)")

# --- Save Results ---
results_df = pd.DataFrame({
    "Time": time_test_full,
    "GroundTruth": y_true.values,
    "Prediction_Raw": y_pred,
    "Prediction_PostProcessed": y_pred_filtered,
})
results_df.to_csv(f"{target_appliance}_FE2_lvl2_predictions_1mil_1hr_FLAML.csv", index=False)
print("csv saved!")

# --- Plotting ---
plt.figure(figsize=(14, 5))
plt.plot(time_test_full, y_true, label="Actual", alpha=0.6)
plt.plot(time_test_full, y_pred, label="Prediction (Raw)", linestyle='--', alpha=0.7)
plt.plot(time_test_full, y_pred_filtered, label="Post-Processed", linestyle='--', alpha=0.9)
plt.title(f"Prediction vs Actual ({target_appliance}) - Full Test Timeline")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(time_test_full, y_true, label="Actual", alpha=0.6)
plt.plot(time_test_full, y_pred, label="Prediction (Raw)", linestyle='--', alpha=0.7)
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

