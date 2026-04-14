
"""
Created on Fri Apr 18 14:29:08 2025


@author: Administrator
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"


import os
import sys
import datetime
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from autogluon.core.models.ensemble.weighted_ensemble_model import WeightedEnsembleModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- Environment threads ---
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

# --- Parameters ---
target_appliance = "car1"
target_dataid = 3000      # household ID for initial dataset generation
window_size = 599         # half-window = 299 on each side
threshold = 0.05          # kW threshold for post-processing
min_on_duration = 3       # minutes
EDGE_THRESH = 1.0         # kW threshold for edges in FE generation

# Feature case: choose 'FE1', 'FE2' or 'Baseline'
CASE = os.getenv('CASE', 'Baseline')

# --- Logging setup ---
ts_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
log_filename = f"autogluon_{target_appliance}_{CASE}_{ts_now}.log"

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, msg):
        for f in self.files:
            try:
                f.write(msg)
                f.flush()
            except UnicodeEncodeError:
                f.write(msg.encode('utf-8','replace').decode())
                f.flush()
    def flush(self):
        for f in self.files: f.flush()
    def isatty(self): return False

log_file = open(log_filename, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = sys.stdout

print(f"===== Feature case: {CASE} =====")

# --- Load and preprocess ---
print("📦 Loading and preprocessing data...")
df = pd.read_csv("1minute_data_newyork.csv")
df["localminute"] = pd.to_datetime(df["localminute"], utc=True, errors='coerce')
df.dropna(subset=["localminute", "dataid"], inplace=True)
# Compute total grid consumption (S_Total)
df_s_total = df.groupby("localminute")["grid"].sum().rename("S_Total")
# Extract target appliance series
df_target = df[df["dataid"] == target_dataid].set_index("localminute")
if target_appliance not in df_target.columns:
    raise ValueError(f"Appliance '{target_appliance}' not found for dataid {target_dataid}.")
# Merge

# Align on timestamps
idx = df_s_total.index.intersection(df_target.index)
df_clean = pd.DataFrame({
    'S_Total': df_s_total.loc[idx],
    target_appliance: df_target.loc[idx, target_appliance]
}).dropna().iloc[:1_000_000]

# --- Create Seq2Point dataset ---
def create_seq2point_dataset(df, feature_col, target_col, window_size):
    half = window_size // 2
    X, y, times = [], [], []
    for i in range(half, len(df) - half):
        w = df[feature_col].iloc[i-half:i+half+1].values
        X.append(w)
        y.append(df[target_col].iloc[i])
        times.append(df.index[i])
    return X, y, times

print("📦 Creating Seq2Point dataset...")
X_raw, y_raw, time_raw = create_seq2point_dataset(df_clean, 'S_Total', target_appliance, window_size)

# --- Split into train/val/test ---
split_idx = int(len(X_raw) * 0.9)
val_idx = int(split_idx * 0.9)
X_tr, y_tr = np.array(X_raw[:val_idx]), np.array(y_raw[:val_idx])
X_v,  y_v  = np.array(X_raw[val_idx:split_idx]), np.array(y_raw[val_idx:split_idx])
X_te, y_te = np.array(X_raw[split_idx:]),      np.array(y_raw[split_idx:])
times_te = time_raw[split_idx:]

# --- Normalize ---
mX, sX = X_tr.mean(), X_tr.std()
my, sy = y_tr.mean(), y_tr.std()
Xtr_n = [(x - mX) / sX for x in X_tr]
Xv_n  = [(x - mX) / sX for x in X_v]
Xte_n = [(x - mX) / sX for x in X_te]
ytr_n = (y_tr - my) / sy
yv_n  = (y_v  - my) / sy
yte_n = (y_te - my) / sy

# --- Feature engineering functions ---
def generate_fe1(window, ts):
    w = np.array(window)
    center = len(w)//2
    diff = np.diff(w)
    lags = {f"lag_{s}": w[center-s] if center-s>=0 else 0 for s in [1,5,15,30,75,150,300]}
    roll = {
        "roll_mean_1min": w[center-7:center+1].mean(),
        "roll_std_1min":  w[center-7:center+1].std(),
        "roll_mean_5min": w[center-37:center+1].mean(),
        "roll_std_5min":  w[center-37:center+1].std(),
        "roll_min_5min":  w[center-37:center+1].min(),
        "roll_max_5min":  w[center-37:center+1].max(),
    }
    peak = np.argmax(w)
    feats = {
        "mean": w.mean(), "std": w.std(), "max": w.max(), "min": w.min(),
        "range": w.max() - w.min(),
        "leading_edges": np.sum(diff > EDGE_THRESH),
        "trailing_edges": np.sum(diff < -EDGE_THRESH),
        "slope": np.polyfit(np.arange(len(w)), w, 1)[0],
        "iqr": np.percentile(w,75) - np.percentile(w,25),
        "slope_in": w[peak] - w[0],
        "slope_out": w[-1] - w[peak],
        "hour": ts.hour, "dayofweek": ts.dayofweek,
        **lags, **roll
    }
    return {k: float(v) if np.isfinite(v) else 0.0 for k,v in feats.items()}

def generate_fe2(window, ts):
    base = generate_fe1(window, ts)
    raw = {f"w_{i}": float(v) for i,v in enumerate(window)}
    return {**base, **raw}

def generate_baseline(window, ts):
    c = len(window)//2
    return {"agg_center": float(window[c]), "hour": ts.hour, "dayofweek": ts.dayofweek}

FEATURE_FUNCS = {"FE1": generate_fe1, "FE2": generate_fe2, "Baseline": generate_baseline}
fe_func = FEATURE_FUNCS.get(CASE)
if fe_func is None:
    raise ValueError(f"Unknown CASE '{CASE}'. Choose from {list(FEATURE_FUNCS.keys())}.")

# --- Build feature datasets ---
print("🔧 Generating features...")
df_tr = pd.DataFrame([fe_func(w, t) for w, t in tqdm(zip(Xtr_n, time_raw[:val_idx]), total=len(Xtr_n))])
df_tr[target_appliance] = ytr_n
df_v = pd.DataFrame([fe_func(w, t) for w, t in tqdm(zip(Xv_n, time_raw[val_idx:split_idx]), total=len(Xv_n))])
df_v[target_appliance] = yv_n
df_te = pd.DataFrame([fe_func(w, t) for w, t in tqdm(zip(Xte_n, times_te), total=len(Xte_n))])
df_te[target_appliance] = yte_n

# --- Train AutoGluon ---
print("🚀 Training AutoGluon...")
predictor = TabularPredictor(label=target_appliance, problem_type='regression', eval_metric='mean_absolute_error')
predictor.fit(train_data=df_tr, tuning_data=df_v, use_bag_holdout=True, time_limit=60*60, presets='best_quality')

# --- Leaderboard & ensemble weights ---
lb = predictor.leaderboard(silent=True)
print("\nFull leaderboard:")
print(lb.to_string(index=False))
best = predictor.model_best
print(f"\n🏆 Best model: {best}")
for m in predictor.model_names():
    if m.startswith("WeightedEnsemble"):
        ens = predictor._trainer.load_model(m)
        if isinstance(ens, WeightedEnsembleModel):
            print(f"=== {m} weights ===")
            for mod, w in ens._get_model_weights().items():
                print(f"  {mod}: {w:.3f}")

# --- Predict & metrics ---
y_pred_n = predictor.predict(df_te.drop(columns=[target_appliance]))
y_pred = y_pred_n * sy + my
y_true = df_te[target_appliance] * sy + my

# Post-processing filter
def duration_threshold_filter(pred, threshold=threshold, min_on_duration=min_on_duration):
    on = pred > threshold
    filtered = on.copy()
    count = 0
    for i in range(len(on)):
        if on[i]: count += 1
        else:
            if 0 < count < min_on_duration:
                filtered[i-count:i] = False
            count = 0
    if 0 < count < min_on_duration:
        filtered[len(on)-count:] = False
    out = pred.copy()
    out[~filtered] = 0
    return out

print_metrics = lambda y_t, y_p, label: print(
    f"--- {label} ---\n"
    f"🔍 RMSE: {np.sqrt(mean_squared_error(y_t, y_p)):.4f} kW\n"
    f"📉 MAE: {mean_absolute_error(y_t, y_p):.4f} kW\n"
    f"📈 R²:  {r2_score(y_t, y_p):.4f}\n"
    f"⚡ SAE: {abs(y_t.sum()-y_p.sum())/y_t.sum():.4f}\n"
)

print_metrics(y_true, y_pred, f"{CASE} - Raw")
y_pp = duration_threshold_filter(y_pred)
print_metrics(y_true, y_pp, f"{CASE} - PostProcessed")

# --- Save results ---
csv_filename = f"{target_appliance}_{CASE}_predictions.csv"
out = pd.DataFrame({"Time": times_te, "GroundTruth": y_true, "Pred_Raw": y_pred, "Pred_Post": y_pp})
out.to_csv(csv_filename, index=False)
print(f"✅ Saved {csv_filename}")

# --- Email with attachments ---
def send_email_with_attachments(sender, password, recipient, subject, body, attachments):
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    for filepath in attachments:
        part = MIMEBase('application', 'octet-stream')
        with open(filepath, 'rb') as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(filepath)}')
        msg.attach(part)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)
    print("📧 Email with attachments sent")

# Email config
SENDER = ""
PASSWORD = ""
RECIPIENT = ""
SUBJECT = f"{target_appliance} {CASE} results_Machine15!!!"
BODY = (
    f"Feature case {CASE} complete.\n\n"
    f"Attached:\n"
    f" • Log file: {log_filename}\n"
    f" • Predictions CSV: {csv_filename}\n"
)
ATTACHMENTS = [log_filename, csv_filename]

send_email_with_attachments(SENDER, PASSWORD, RECIPIENT, SUBJECT, BODY, ATTACHMENTS)

# --- Plot predictions vs. ground truth ---
print("📊 Plotting predictions vs. ground truth...")
plt.figure(figsize=(12, 6))
plt.plot(times_te, y_true, label='Ground Truth')
plt.plot(times_te, y_pred, label='Raw Prediction', alpha=0.7)
plt.plot(times_te, y_pp, label='PostProcessed Prediction', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.title(f'{target_appliance} Predictions vs Ground Truth ({CASE})')
plt.legend()
plt.tight_layout()
plt.show()

# --- Cleanup ---
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()