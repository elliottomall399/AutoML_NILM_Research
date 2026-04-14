
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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# H2O setup
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"]    = "8"
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-24"  # update if needed
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]
import h2o
from h2o.automl import H2OAutoML

# initialize H2O cluster
h2o.init()

# --- Parameters ---
target_appliance = "car1"
window_size     = 599
threshold       = 0.05  # 50W threshold
default_min_on   = 3    # mins
EDGE_THRESH     = 1.0   # kW jump threshold
CASE = "Baseline"     # FE1, FE2, Baseline

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

def duration_threshold_filter(pred, threshold=threshold, min_on_duration=default_min_on):
    on = pred > threshold
    filtered = on.copy()
    count = 0
    for i in range(len(on)):
        if on[i]: count += 1
        else:
            if 0 < count < min_on_duration: filtered[i-count:i] = False
            count = 0
    if 0 < count < min_on_duration: filtered[len(on)-count:] = False
    out = pred.copy()
    out[~filtered] = 0
    return out

def print_metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    sae  = abs(y_true.sum() - y_pred.sum()) / y_true.sum()
    print(f"--- {label} ---")
    print(f"🔍 RMSE: {rmse:.4f} kW")
    print(f"📉 MAE:  {mae:.4f} kW")
    print(f"📈 R²:   {r2:.4f}")
    print(f"⚡ SAE:  {sae:.4f}\n")

# Feature functions (from Code1)
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
FEATURE_FUNCS = {"FE1": generate_fe1, "FE2": generate_fe2, "Baseline": generate_baseline}
fe_func = FEATURE_FUNCS[CASE]

# --- Load & aggregate data ---
df = pd.read_csv("1minute_data_newyork.csv")
df["localminute"] = pd.to_datetime(df["localminute"], utc=True, errors='coerce')
df.dropna(subset=["localminute","dataid"], inplace=True)
df.drop(columns=["leg1v","leg2v"], errors="ignore", inplace=True)
vals = [c for c in df.columns if c not in ["dataid","localminute"]]
df_sub = df.groupby("localminute")[vals].sum()
df_sub["S_Total"] = df_sub.get("grid",0)
idx = pd.date_range(df_sub.index.min(), df_sub.index.max(), freq="1min")
df_sub = df_sub.reindex(idx)
df_sub.index.name = "localminute"
df_sub = df_sub[["car1","car2","solar","solar2","air1","heater1","S_Total"]]

# Logging
ts_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
log_filename   = f"h2o_{target_appliance}_{CASE}_{ts_now}.log"
pred_filename  = f"{target_appliance}_{CASE}_h2o_preds.csv"
log_file = open(log_filename, 'w', encoding='utf-8')
class Tee:  # capture stdout
    def __init__(self,*files): self.files=files
    def write(self,msg): [f.write(msg) or f.flush() for f in self.files]
    def flush(self): [f.flush() for f in self.files]
    def isatty(self): return False
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = sys.stdout
print(f"\n===== H2O Feature case: {CASE} =====")

# --- Build datasets ---
df_clean = df_sub[["S_Total", target_appliance]].dropna().iloc[:1_000_000]
X_raw,y_raw,times = create_seq2point_dataset(df_clean, "S_Total", target_appliance, window_size)
si = int(len(X_raw)*0.9)
vi = int(si*0.9)
X_tr,y_tr = np.array(X_raw[:vi]), np.array(y_raw[:vi])
X_v ,y_v  = np.array(X_raw[vi:si]), np.array(y_raw[vi:si])
X_te,y_te = np.array(X_raw[si:]), np.array(y_raw[si:])
times_te = times[si:]
# Normalize
mX,sX = X_tr.mean(), X_tr.std()
my,sy = y_tr.mean(), y_tr.std()
Xtr_n = [(x-mX)/sX for x in X_tr]
Xv_n  = [(x-mX)/sX for x in X_v]
Xte_n = [(x-mX)/sX for x in X_te]
ytr_n = (y_tr-my)/sy
yv_n  = (y_v-my)/sy
yte_n = (y_te-my)/sy
# FE
df_tr = pd.DataFrame([fe_func(w,t) for w,t in zip(Xtr_n, times[:vi])]); df_tr[target_appliance]=ytr_n
df_v  = pd.DataFrame([fe_func(w,t) for w,t in zip(Xv_n,   times[vi:si])]); df_v[target_appliance]=yv_n
df_te = pd.DataFrame([fe_func(w,t) for w,t in zip(Xte_n,  times_te)]);    df_te[target_appliance]=yte_n

# --- Train H2O AutoML ---
train_h2o = h2o.H2OFrame(df_tr)
valid_h2o = h2o.H2OFrame(df_v)
h2o_Model = H2OAutoML(max_runtime_secs=60*5, stopping_metric="MAE", seed=42, nfolds=0)
h2o_Model.train(x=df_tr.columns.drop(target_appliance).tolist(),
               y=target_appliance,
               training_frame=train_h2o,
               validation_frame=valid_h2o)
print("✅ H2O Leader:", h2o_Model.leader)

# --- Predict & metrics ---
pred_h2o = h2o_Model.predict(h2o.H2OFrame(df_te)).as_data_frame().values.flatten()
y_pred   = pred_h2o * sy + my
y_true   = df_te[target_appliance] * sy + my
print_metrics(y_true, y_pred, f"{CASE} - Raw")
y_pp = duration_threshold_filter(y_pred)
print_metrics(y_true, y_pp, f"{CASE} - PostProcessed")

# --- Save results & restore ---
out = pd.DataFrame({"Time": times_te, "GroundTruth": y_true, "Pred_Raw": y_pred, "Pred_Post": y_pp})
out.to_csv(pred_filename, index=False)
print(f"✅ Saved {pred_filename}")
sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__; log_file.close()

# --- Email ---
def send_email_with_attachments(sender, password, recipient, subject, body, attachments):
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    msg = MIMEMultipart(); msg['From']=sender; msg['To']=recipient; msg['Subject']=subject
    msg.attach(MIMEText(body,'plain'))
    for fpath in attachments:
        part=MIMEBase('application','octet-stream')
        with open(fpath,'rb') as f: part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',f'attachment; filename={os.path.basename(fpath)}')
        msg.attach(part)
    import smtplib
    with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
        smtp.login(sender,password)
        smtp.send_message(msg)
    print("📧 Email sent")
# configure & send
SENDER=""; PASSWORD=""; RECIPIENT=""
SUBJECT=f"{target_appliance} {CASE} H2O results!!"; BODY=f"Feature case {CASE} complete with H2O.\nAttached: {log_filename}, {pred_filename}\n"
send_email_with_attachments(SENDER,PASSWORD,RECIPIENT,SUBJECT,BODY,[log_filename,pred_filename])
