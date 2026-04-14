
"""
Created on Fri Apr 18 14:29:08 2025

@author: Administrator
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from autogluon.core.models.ensemble.weighted_ensemble_model import WeightedEnsembleModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# set environment threads
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"]    = "8"

# --- Parameters ---
target_appliance = "car1" #Change me!!
window_size     = 599
threshold       = 0.05  # 50W instead of 0.05kW was. Remember the context is device power. Removes near zero noise and stil allows for low power modes etc.
min_on_duration = 3     # consecutive mins
EDGE_THRESH    = 1.0    # kW, count any jump >1 kW as an edge. Context is substation power. 1 is used as it is likely real events and not jitter.

# --- Select the case you want to run: 'FE1', 'FE2' or 'Baseline' ---
CASE = "Baseline"   # ← change me!

# --- Utility functions ---
def create_seq2point_dataset(df, feature_col, target_col, window_size):
    half = window_size // 2
    X, y, times = [], [], []
    for i in range(half, len(df) - half):
        w = df[feature_col].iloc[i-half:i+half+1].values
        X.append(w)
        y.append(df[target_col].iloc[i])
        times.append(df.index[i])
    return X, y, times

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
        "range": w.max()-w.min(),
        "leading_edges": np.sum(diff> EDGE_THRESH),
        "trailing_edges": np.sum(diff<-EDGE_THRESH),
        "slope": np.polyfit(np.arange(len(w)), w, 1)[0],
        "iqr": np.percentile(w,75)-np.percentile(w,25),
        "slope_in": w[peak]-w[0],
        "slope_out": w[-1]-w[peak],
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

def duration_threshold_filter(pred, threshold=threshold, min_on_duration=min_on_duration):
    on = pred > threshold
    filtered = on.copy()
    count = 0
    for i in range(len(on)):
        if on[i]:
            count += 1
        else:
            if 0 < count < min_on_duration:
                filtered[i-count:i] = False
            count = 0
    if 0 < count < min_on_duration:
        filtered[len(on)-count:] = False
    out = pred.copy()
    out[~filtered] = 0
    return out

def print_metrics(y_t, y_p, label):
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae  = mean_absolute_error(y_t, y_p)
    r2   = r2_score(y_t, y_p)
    sae  = abs(y_t.sum() - y_p.sum()) / y_t.sum()
    print(f"--- {label} ---")
    print(f"🔍 RMSE: {rmse:.4f} kW")
    print(f"📉 MAE:  {mae:.4f} kW")
    print(f"📈 R²:   {r2:.4f}")
    print(f"⚡ SAE:  {sae:.4f}\n")

# --- Map cases to feature‐generation functions ---
FEATURE_FUNCS = {
    "FE1": generate_fe1,
    "FE2": generate_fe2,
    "Baseline": generate_baseline,
}
fe_func = FEATURE_FUNCS[CASE]

# --- Load & aggregate substation data once ---
print("📦 Loading data…")
df = pd.read_csv("1minute_data_newyork.csv")
df["localminute"] = pd.to_datetime(df["localminute"], utc=True, errors="coerce")
df.dropna(subset=["localminute","dataid"], inplace=True)
df.drop(columns=["leg1v","leg2v"], errors="ignore", inplace=True)
value_cols = [c for c in df.columns if c not in ["dataid","localminute"]]
df_sub = df.groupby("localminute")[value_cols].sum()
df_sub["S_Total"] = df_sub.get("grid",0)
full_idx = pd.date_range(df_sub.index.min(), df_sub.index.max(), freq="1min")
df_sub = df_sub.reindex(full_idx)
df_sub.index.name = "localminute"
keep_cols = ["car1","car2","solar","solar2","air1","heater1","S_Total"]
df_sub = df_sub[keep_cols]
# ─── NEW: Inspect 1-minute differences in S_Total ───
import matplotlib.pyplot as plt

# drop any NaNs, get the values and compute diffs
series = df_sub["S_Total"].dropna().values
diffs   = np.diff(series)

# print percentiles of the diffs
pcts = np.percentile(diffs, [1,5,25,50,75,95,99]).round(4)
print("S_Total Δ percentiles (kW):")
print(f" 1%: {pcts[0]:>6}   5%: {pcts[1]:>6}   25%: {pcts[2]:>6}")
print(f"50%: {pcts[3]:>6}  75%: {pcts[4]:>6}   95%: {pcts[5]:>6}   99%: {pcts[6]:>6}")

# histogram
plt.figure(figsize=(8,4))
plt.hist(diffs, bins=200)
plt.title("Histogram of 1-minute Δ S_Total")
plt.xlabel("Δ Power (kW)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

series = df_sub[target_appliance].dropna().values

# 1) Histogram of raw values
plt.figure(figsize=(6,4))
plt.hist(series, bins=200)
plt.title(f"Histogram of {target_appliance} power")
plt.xlabel("Power (kW)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2) Print low‐end percentiles
pct = np.percentile(series, [50,75,90,95,99])
print(f"{target_appliance} power percentiles (kW): "
      f"50%={pct[0]:.3f}, 75%={pct[1]:.3f}, "
      f"90%={pct[2]:.3f}, 95%={pct[3]:.3f}, 99%={pct[4]:.3f}")

# --- Prepare logging ---
ts_now       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
log_filename = f"autogluon_{target_appliance}_{CASE}_{ts_now}.log"



class Tee:
    def __init__(self,*files): self.files = files
    def write(self,msg):
        for f in self.files:
            try:
                f.write(msg)
                f.flush()
            except UnicodeEncodeError:
                f.write(msg.encode('utf-8','replace').decode())
                f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def isatty(self):
        return False

log_file = open(log_filename, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = sys.stdout

print(f"\n===== Feature case: {CASE} =====")

# --- Build datasets ---
df_clean   = df_sub[["S_Total", target_appliance]].dropna().iloc[:1_000_000]
X_raw, y_raw, times = create_seq2point_dataset(df_clean, "S_Total", target_appliance, window_size)
split_idx  = int(len(X_raw)*0.9)
val_idx    = int(split_idx*0.9)
X_tr, y_tr = np.array(X_raw[:val_idx]), np.array(y_raw[:val_idx])
X_v , y_v  = np.array(X_raw[val_idx:split_idx]), np.array(y_raw[val_idx:split_idx])
X_te, y_te = np.array(X_raw[split_idx:]),      np.array(y_raw[split_idx:])
times_te   = times[split_idx:]

# --- Normalize ---
mX, sX = X_tr.mean(), X_tr.std()
my, sy = y_tr.mean(), y_tr.std()
Xtr_n = [(x-mX)/sX for x in X_tr]
Xv_n  = [(x-mX)/sX for x in X_v]
Xte_n = [(x-mX)/sX for x in X_te]
ytr_n = (y_tr-my)/sy
yv_n  = (y_v-my)/sy
yte_n = (y_te-my)/sy

# --- Feature engineering ---
df_tr = pd.DataFrame([fe_func(w,t) for w,t in zip(Xtr_n, times[:val_idx])])
df_tr[target_appliance] = ytr_n
df_v  = pd.DataFrame([fe_func(w,t) for w,t in zip(Xv_n,   times[val_idx:split_idx])])
df_v[target_appliance]  = yv_n
df_te = pd.DataFrame([fe_func(w,t) for w,t in zip(Xte_n,  times_te)])
df_te[target_appliance] = yte_n

# --- Train ---
print("🚀 Training AutoGluon…")
predictor = TabularPredictor(
    label=target_appliance,
    problem_type='regression',
    eval_metric='mean_absolute_error'
).fit(
    train_data=df_tr,
    tuning_data=df_v,
    use_bag_holdout=True,
    time_limit=600*6,
    presets='best_quality'
)

# --- Evaluate leaderboard & ensembles ---
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
y_pred   = y_pred_n * sy + my
y_true   = df_te[target_appliance] * sy + my
print_metrics(y_true, y_pred, f"{CASE} - Raw")

# --- Post-process & metrics ---
y_pp = duration_threshold_filter(y_pred)
print_metrics(y_true, y_pp, f"{CASE} - PostProcessed")

# --- Save results ---
out = pd.DataFrame({
    "Time":        times_te,
    "GroundTruth": y_true,
    "Pred_Raw":    y_pred,
    "Pred_Post":   y_pp
})
csv_filename = f"{target_appliance}_{CASE}_predictions.csv"
out.to_csv(csv_filename, index=False)
print(f"✅ Saved {csv_filename}")

# --- Email with attachments ---
def send_email_with_attachments(sender, password, recipient, subject, body, attachments):
    msg = MIMEMultipart()
    msg['From']    = sender
    msg['To']      = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    for filepath in attachments:
        part = MIMEBase('application', 'octet-stream')
        with open(filepath, 'rb') as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename={os.path.basename(filepath)}'
        )
        msg.attach(part)

    # connect & send
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)
    print("📧 Email with attachments sent")


# —– CONFIGURE EMAIL DETAILS —–
SENDER      = ""
PASSWORD    = ""   # ← your literal App Password
RECIPIENT   = ""
SUBJECT     = f"{target_appliance} {CASE} results!!"
BODY        = (
    f"Feature case {CASE} is complete.\n\n"
    f"Attached are:\n"
    f" • Log file:       {log_filename}\n"
    f" • Predictions CSV: {csv_filename}\n"
)
ATTACHMENTS = [log_filename, csv_filename]

# Send the email
send_email_with_attachments(
    sender=SENDER,
    password=PASSWORD,
    recipient=RECIPIENT,
    subject=SUBJECT,
    body=BODY,
    attachments=ATTACHMENTS
)

# --- Restore stdout/stderr & close log ---
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()