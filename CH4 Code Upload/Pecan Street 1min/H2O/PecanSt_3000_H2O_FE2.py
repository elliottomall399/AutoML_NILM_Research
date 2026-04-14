# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:29:08 2025


@author: Administrator
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-24"  # Adjust this if needed
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import h2o
from h2o.automl import H2OAutoML

# --- Init ---
random.seed(42)
np.random.seed(42)
h2o.init()

# --- Load Data ---
df = pd.read_csv("1minute_data_newyork.csv")
df = df[df["dataid"] == 3000].copy()
df["localminute"] = pd.to_datetime(df["localminute"], utc=True, errors='coerce')
df.dropna(subset=["localminute"], inplace=True)
df.set_index("localminute", inplace=True)
df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq="1min"))

# --- Params ---
target_appliance = "car1"
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

# --- FE2: Advanced Features (engineered + full normalized window) ---
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
        "leading_edges": np.sum(diff > 0.03),
        "trailing_edges": np.sum(diff < -0.03),
        "slope": np.polyfit(np.arange(len(window)), window, 1)[0],
        "iqr": np.percentile(window, 75) - np.percentile(window, 25),
        "slope_in": window[peak] - window[0],
        "slope_out": window[-1] - window[peak],
        "hour": timestamp.hour,
        "dayofweek": timestamp.dayofweek,
        **lags,
        **roll_feats,
    }

    raw_feats = {f"w_{i}": float(v) for i, v in enumerate(window)}
    return {**engineered, **raw_feats}

print("🔧 Generating advanced features for H2O...")
df_train = pd.DataFrame([generate_features(w, t) for w, t in tqdm(zip(X_train_norm, time_raw[:val_index]))])
df_val = pd.DataFrame([generate_features(w, t) for w, t in tqdm(zip(X_val_norm, time_raw[val_index:split_index]))])
df_test = pd.DataFrame([generate_features(w, t) for w, t in tqdm(zip(X_test_norm, time_raw[split_index:]))])

df_train[target_appliance] = y_train_norm
df_val[target_appliance] = y_val_norm
df_test[target_appliance] = y_test_norm

# --- Train H2O ---
train_h2o = h2o.H2OFrame(df_train)
valid_h2o = h2o.H2OFrame(df_val)
test_h2o = h2o.H2OFrame(df_test)

aml = H2OAutoML(max_runtime_secs=60*60, stopping_metric="MAE", seed=42, nfolds=0)
aml.train(x=df_train.columns.drop(target_appliance).tolist(),
          y=target_appliance,
          training_frame=train_h2o,
          validation_frame=valid_h2o)

print("✅ Best Model Leader:")
print(aml.leader)

# --- Predict and Denormalize ---
y_pred_norm = aml.predict(test_h2o).as_data_frame().values.flatten()
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

# --- Save Results ---
results_df = pd.DataFrame({
    "Time": time_test_full,
    "GroundTruth": y_true.values,
    "Prediction_Raw": y_pred,
    "Prediction_PostProcessed": y_pred_filtered,
})
results_df.to_csv(f"{target_appliance}_FE2_predictions_H2O.csv", index=False)

# --- Plotting ---
plt.figure(figsize=(14, 5))
plt.plot(time_test_full, y_true.values, label="Actual", alpha=0.6)
plt.plot(time_test_full, y_pred, label="Prediction (Raw)", linestyle='--')
plt.plot(time_test_full, y_pred_filtered, label="Post-Processed", linestyle='--')
plt.title(f"Prediction vs Actual ({target_appliance}) - Full Timeline")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plotting ---
plt.figure(figsize=(14, 5))
plt.plot(time_test_full, y_true.values, label="Actual", alpha=0.6)
plt.plot(time_test_full, y_pred, label="Prediction (Raw)", linestyle='--')
#plt.plot(time_test_full, y_pred_filtered, label="Post-Processed", linestyle='--')
plt.title(f"Prediction vs Actual ({target_appliance}) - Full Timeline")
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

import smtplib
from email.mime.text import MIMEText

def send_email_notification(): #Function to email on completion of the script. Delete or add own credentials
    from_email = ""
    to_email = ""  # Can be the same
    subject = "Script Completed ✅"
    body = f"Your Python script for '{target_appliance}' H2O FE2 has finished running."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, "")
            server.send_message(msg)
        print("📧 Email notification sent!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Send the email at the end
send_email_notification()