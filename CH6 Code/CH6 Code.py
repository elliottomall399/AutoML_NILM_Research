# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:29:08 2025

@author: Administrator
"""
# =============================================================================
# Full Python Script for Appliance Count Forecasting as a REGRESSION Problem
# =============================================================================

import os
# Setting thread environment variables can sometimes help with performance on multi-core CPUs
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import datetime
import sys
from shutil import rmtree

# --- Parameters ---
target_appliance = "car1"
window_size      = 599
threshold        = 0.5 # kW threshold for active definition

# --- Logging setup ---
log_filename     = f"autogluon_regression_log_{target_appliance}_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
    def isatty(self):
        return sys.__stdout__.isatty()

log_file         = open(log_filename, "w")
original_stdout = sys.stdout
sys.stdout       = Tee(sys.stdout, log_file)
sys.stderr       = sys.stdout

print("--- PART 1: PREDICTING APPLIANCE COUNTS (CODE A) ---")

# ------------------------------
# 1.1) LOAD + PREPARE DATA
# ------------------------------
print("📥 Loading data...")
df = pd.read_csv("1minute_data_newyork.csv")
df["localminute"] = pd.to_datetime(df["localminute"], utc=True, errors="coerce")
df.dropna(subset=["localminute","dataid"], inplace=True)
df.drop(columns=["leg1v","leg2v"], errors="ignore", inplace=True)

print(" Crunching aggregate features...")
S_Total = df.groupby("localminute")["grid"].sum().rename("S_Total")

print(f"🔢 Computing total {target_appliance} per timestamp...")
df_appl_values = df[["localminute", target_appliance]].dropna()
df_feature     = df_appl_values.groupby("localminute")[target_appliance]\
                           .sum().rename("A_Total").to_frame()

print("🔍 Computing number of active devices per timestamp (Ground Truth)...")
df_appl_act = df[["localminute","dataid",target_appliance]].dropna()
df_appl_act = df_appl_act[df_appl_act[target_appliance] > threshold]
df_label    = df_appl_act.groupby("localminute")["dataid"]\
                           .nunique().rename("num_active")

print("🔗 Merging feature and label...")
df_combined           = df_feature.join(df_label, how="left")
df_combined["num_active"] = df_combined["num_active"].fillna(0).astype(int)
df_combined           = df_combined.iloc[:1_000_000].copy()

# ------------------------------
# 1.2) SEQ2POINT DATASET CREATION
# ------------------------------
def create_seq2point_dataset(df, feature_col, target_col, window_size):
    half = window_size // 2
    X, y, times = [], [], []
    for i in tqdm(range(half, len(df) - half), desc="Creating Seq2Point windows"):
        window = df[feature_col].iloc[i-half:i+half+1].values
        X.append(window)
        y.append(df[target_col].iloc[i])
        times.append(df.index[i])
    return X, y, times

print("📦 Creating Seq2Point dataset...")
X_raw, y_raw, time_raw = create_seq2point_dataset(df_combined, "A_Total", "num_active", window_size)

# ------------------------------
# 1.3) SPLIT, NORMALIZE, AND ENGINEER FEATURES
# ------------------------------
total     = len(X_raw)
split     = int(total * 0.9)
val_split = int(split * 0.9)

X_train   = np.array(X_raw[:val_split])
y_train   = np.array(y_raw[:val_split])
X_val     = np.array(X_raw[val_split:split])
y_val     = np.array(y_raw[val_split:split])
X_test    = np.array(X_raw[split:])
y_test    = np.array(y_raw[split:])
time_test = pd.Series(time_raw[split:], name="Time")

mean_X, std_X = X_train.mean(), X_train.std()
X_train = [(x-mean_X)/std_X for x in X_train]
X_val   = [(x-mean_X)/std_X for x in X_val]
X_test  = [(x-mean_X)/std_X for x in X_test]

print("🔧 Generating window features...")
def gen_feats(window, timestamp):
    w      = np.array(window)
    center = len(w)//2
    diff   = np.diff(w)
    lags   = {f"lag_{s}": w[center-s] if center-s>=0 else 0 for s in [1,5,15,30,75,150,300]}
    rolls  = {
        "roll_mean_1min": w[center-7:center+1].mean(),
        "roll_std_1min":  w[center-7:center+1].std(),
        "roll_mean_5min": w[center-37:center+1].mean(),
        "roll_std_5min":  w[center-37:center+1].std(),
    }
    peak   = np.argmax(w)
    feats  = {
        "mean": w.mean(), "std": w.std(), "max": w.max(), "min": w.min(),
        "range": w.max()-w.min(),
        "leading_edges":  np.sum(diff>30),
        "trailing_edges": np.sum(diff<-30),
        "slope": np.polyfit(np.arange(len(w)), w,1)[0],
        "iqr":   np.percentile(w,75)-np.percentile(w,25),
        "slope_in":  w[peak]-w[0],
        "slope_out": w[-1]-w[peak],
        "hour": timestamp.hour,
        "dayofweek": timestamp.dayofweek,
        **lags, **rolls
    }
    return {k: float(v) if np.isfinite(v) else 0.0 for k,v in feats.items()}

train_df = pd.DataFrame(
    [gen_feats(w,t) for w,t in tqdm(zip(X_train, time_raw[:val_split]), total=len(X_train), desc="Features for Train")]
)
val_df   = pd.DataFrame(
    [gen_feats(w,t) for w,t in tqdm(zip(X_val,   time_raw[val_split:split]), total=len(X_val), desc="Features for Val")]
)
test_df  = pd.DataFrame(
    [gen_feats(w,t) for w,t in tqdm(zip(X_test,  time_raw[split:]), total=len(X_test), desc="Features for Test")]
)

train_df["num_active"] = y_train
val_df["num_active"]   = y_val
test_df["num_active"]  = y_test

# ------------------------------
# 1.4) TRAIN AND EVALUATE COUNT MODEL
# ------------------------------
print("🚀 Training AutoGluon regressor for appliance counts…")
predictor_path = f"AG_regression_{target_appliance}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
if os.path.exists(predictor_path):
    rmtree(predictor_path)

predictor = TabularPredictor(
    label='num_active',
    problem_type='regression',
    eval_metric='root_mean_squared_error',
    path=predictor_path
).fit(
    train_data = train_df,
    tuning_data= val_df,
    time_limit  = 60*6,
    use_bag_holdout=True,
    presets='best_quality'
)

print("✅ Evaluating count model...")
y_pred_float   = predictor.predict(test_df.drop(columns=["num_active"]))
y_pred_rounded = np.round(y_pred_float).astype(int)
y_pred_rounded[y_pred_rounded < 0] = 0 # Ensure no negative counts

mae  = mean_absolute_error(y_test, y_pred_rounded)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
r2   = r2_score(y_test, y_pred_rounded)

print("\n--- Model Evaluation (Counts) ---")
print(f"✅ MAE:  {mae:.4f} appliances")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R²:   {r2:.4f}")
print("---------------------------------\n")

results = pd.DataFrame({
    "Time":              time_test,
    "Actual_Count":      y_test,
    "Predicted_Float":   y_pred_float,
    "Predicted_Rounded": y_pred_rounded
}).set_index("Time")

print("✅ Count predictions generated.")
print("\n\n--- PART 2: WEIGHTED REGRESSION ON PREDICTED COUNTS (CODE B) ---")

# ------------------------------
# 2.1) AGGREGATE & PREPARE DATA FOR FITTING
# ------------------------------
print("📊 Preparing data for weighted fits using predicted counts...")
df_min_pred = pd.DataFrame({
    "count":   results["Predicted_Rounded"],  # Use predicted counts from Part 1
    "S_Total": S_Total.reindex(results.index).ffill()
}).dropna()

vc_pred = df_min_pred["count"].value_counts().sort_index()
agg_pred = df_min_pred.groupby("count")["S_Total"].agg(
    mean = "mean",
    max  = "max",
    min  = "min",
    p90  = lambda x: np.percentile(x, 90)
).reset_index()
agg_pred["freq"] = agg_pred["count"].map(vc_pred)

print("\nPer-count summary based on PREDICTED counts:\n", agg_pred, "\n")

# ------------------------------
# 2.2) PERFORM WEIGHTED LINEAR REGRESSION
# ------------------------------
print("📈 Performing weighted linear regression on summary statistics...")
models = {}
for stat in ["mean", "max", "min", "p90"]:
    lr = LinearRegression().fit(
        agg_pred[["count"]],
        agg_pred[stat].values,
        sample_weight=agg_pred["freq"]
    )
    models[stat] = lr
    a, b = lr.coef_[0], lr.intercept_
    print(f"Weighted {stat:>4s} fit (predicted): S_Total ≈ {a:.3f}·count + {b:.3f}")

# ------------------------------
# 2.3) PLOT RESULTS
# ------------------------------
print("🖼️ Generating final plots...")
sys.stdout.flush()
sys.stdout = original_stdout # Switch back to original stdout for plotting

# PLOT 1: Weighted Regression Plot
plt.figure(figsize=(12, 7))
plt.scatter(df_min_pred["count"], df_min_pred["S_Total"],
            s=10, alpha=0.15, label="Data points (using predicted count)", zorder=1)
plt.plot(agg_pred["count"], agg_pred["mean"], 'o-',  linewidth=2, label="Mean Power", zorder=2)
plt.plot(agg_pred["count"], agg_pred["max"],  's--', linewidth=2, label="Max Power", zorder=2)
plt.plot(agg_pred["count"], agg_pred["p90"],  'v-.', linewidth=2, label="90th Percentile Power", zorder=2)
plt.plot(agg_pred["count"], agg_pred["min"],  'd:',  linewidth=2, label="Min Power", zorder=2)

max_count_to_plot = 10
x_line = np.linspace(0, max_count_to_plot, 200).reshape(-1,1)
for stat, lr in models.items():
    plt.plot(x_line, lr.predict(x_line),
             linewidth=2.5, linestyle='--', label=f"Weighted {stat} fit", zorder=3)

max_observed_count = df_min_pred["count"].max()
if max_observed_count <= max_count_to_plot:
    plt.axvline(max_observed_count, color="gray", linestyle=":", label=f"Max observed predicted count ({max_observed_count})")
plt.axvline(max_count_to_plot, color="black", linestyle="--", label=f"{max_count_to_plot}-device mark")

plt.xlim(0, max_count_to_plot)
plt.xlabel("Predicted Active-Device Count")
plt.ylabel("Substation Power (kW)")
plt.title(f"Substation Power vs. Predicted Number of Active '{target_appliance}' Appliances")
plt.legend(loc="upper right")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


# PLOT 2: Appliance Count vs. Grid Power Time-Series
print("🖼️ Generating time-series plot...")
time_slice    = results.index
s_total_slice = S_Total.reindex(time_slice).ffill()
a_total_slice = df_feature.reindex(time_slice).ffill()

fig, ax1 = plt.subplots(figsize=(18,8))
ax1.set_xlabel('Time')
ax1.set_ylabel('Active Count', color='tab:blue')
ax1.plot(time_slice, results['Actual_Count'],
         label='Ground Truth', color='tab:blue', linewidth=2.5)
ax1.plot(time_slice, results['Predicted_Rounded'],
         label='Predicted', color='red', linestyle=':', marker='o', markersize=2, alpha=0.7)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', linewidth=0.5)

ax2 = ax1.twinx()
ax2.set_ylabel('Power (kW)')
ax2.plot(s_total_slice.index, s_total_slice.values, label='S_Total (Grid)',
         color='green', alpha=0.6)
ax2.plot(a_total_slice.index, a_total_slice.values,
         label=f"A_Total ({target_appliance})", color='purple', alpha=0.7)
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend(loc='upper right')

fig.suptitle(f'Appliance Count vs. Grid Power – {target_appliance} (Full Test Set)', fontsize=16)
fig.autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(time_slice) // (24*60) // 10))) # Auto-adjust date ticks
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()