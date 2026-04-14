
# -*- coding: utf-8 -*-
"""
This script is for AutoGluon
H2O and FLAML will be similar to this, but with the AutoML specific pipelines as used in the other CH5 Code.
"""

# ---------- env/thread caps: set BEFORE numpy/pandas imports ----------
import os
os.environ["OPENBLAS_NUM_THREADS"] = "24"
os.environ["OMP_NUM_THREADS"]      = "24"
os.environ["MKL_NUM_THREADS"]      = "24"
os.environ["NUMEXPR_NUM_THREADS"]  = "24"
os.environ["AG_MAX_CPU_COUNT"]     = "24"
os.environ["TQDM_NOTEBOOK"]        = "0"   # avoid IProgress warnings

# ---------- imports ----------
import sys
import gc
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from autogluon.tabular import TabularPredictor
from autogluon.core.models.ensemble.weighted_ensemble_model import WeightedEnsembleModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ---------- parameters ----------
target_appliance   = "car1"
window_size        = 599
threshold          = 0.05      # kW, used for post-processing only
min_on_duration    = 3         # mins
EDGE_THRESH        = 1.0       # kW, edge detector for FE
COUNT_THRESHOLD_KW = 0.5       # for num_active definition
CASE               = "FE2"     # "Baseline" | "FE1" | "FE2"
TIME_LIMIT_SEC     = 60*60      # AutoGluon time budget

# Email creds: set as env vars to avoid hardcoding
SENDER    = os.getenv("NILM_EMAIL_SENDER", "")
PASSWORD  = os.getenv("NILM_EMAIL_PASSWORD", "")
RECIPIENT = os.getenv("NILM_EMAIL_RECIPIENT", "")

# ---------- helpers ----------
def create_seq2point_dataset(df, feature_col, target_col, window_size):
    half = window_size // 2
    X, y, times = [], [], []
    vals  = df[feature_col].values
    tvals = df[target_col].values
    idx   = df.index
    for i in range(half, len(df) - half):
        X.append(vals[i-half:i+half+1])
        y.append(tvals[i])
        times.append(idx[i])
    return X, y, times

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
    sae  = abs(y_t.sum() - y_p.sum()) / (y_t.sum() + 1e-12)
    print(f"--- {label} ---")
    print(f"RMSE: {rmse:.4f} kW  |  MAE: {mae:.4f} kW  |  R²: {r2:.4f}  |  SAE: {sae:.4f}")

def downcast_float32(df_):
    for c in df_.columns:
        dt = df_[c].dtype
        if dt == "float64":
            df_[c] = df_[c].astype("float32")
        elif dt == "int64":
            df_[c] = df_[c].astype("int32")
        elif dt == "bool":
            df_[c] = df_[c].astype("int8")
    return df_

# ---------- logging ----------
ts_now       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
log_filename = f"autogluon_{target_appliance}_{CASE}_{ts_now}.log"

class Tee:
    def __init__(self,*files): self.files = files
    def write(self,msg):
        for f in self.files:
            try: f.write(msg); f.flush()
            except UnicodeEncodeError:
                f.write(msg.encode('utf-8','replace').decode()); f.flush()
    def flush(self):
        for f in self.files: f.flush()
    def isatty(self): return False

log_file = open(log_filename, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = sys.stdout

print("📦 Loading data…")
print(f"\n===== Feature case: {CASE} =====")

# ---------- load data ----------
df = pd.read_csv("1minute_data_newyork.csv")
df["localminute"] = pd.to_datetime(df["localminute"], utc=True, errors="coerce")
df.dropna(subset=["localminute","dataid"], inplace=True)
df.drop(columns=["leg1v","leg2v"], errors="ignore", inplace=True)
df["dataid"] = df["dataid"].astype("int32")

# ---------- compute num_active (Series -> dict, bind into accessor) ----------
mask = df[target_appliance].values > COUNT_THRESHOLD_KW
df_act = df.loc[mask, ["localminute","dataid"]]
num_active_series = (
    df_act.groupby("localminute")["dataid"]
          .nunique()
          .astype("int16")
)
num_active_lookup = num_active_series.to_dict()   # compact mapping
del mask, df_act, num_active_series
gc.collect()

# Accessor bound to the lookup at definition time (won't break if we delete the global later)
def get_num_active(ts, _lookup=num_active_lookup):
    return int(_lookup.get(ts, 0))

# ---------- aggregate substation per minute ----------
value_cols = [c for c in df.columns if c not in ["dataid","localminute"]]
df_sub = df.groupby("localminute")[value_cols].sum()
df_sub["S_Total"] = df_sub.get("grid", 0)
keep_cols = ["car1","car2","solar","solar2","air1","heater1","S_Total"]
df_sub = df_sub[keep_cols]

# free raw df before feature building
del df, value_cols
gc.collect()

# ---------- diagnostics (optional) ----------
series_st = df_sub["S_Total"].dropna().values
if len(series_st) > 2:
    diffs = np.diff(series_st)
    pcts = np.percentile(diffs, [1,5,25,50,75,95,99]).round(4)
    print("S_Total Δ percentiles (kW):", dict(zip([1,5,25,50,75,95,99], pcts.tolist())))

series_tg = df_sub[target_appliance].dropna().values
if len(series_tg) > 0:
    pct = np.percentile(series_tg, [50,75,90,95,99])
    print(f"{target_appliance} power percentiles (kW): 50%={pct[0]:.3f}, 75%={pct[1]:.3f}, 90%={pct[2]:.3f}, 95%={pct[3]:.3f}, 99%={pct[4]:.3f}")

# ---------- FE functions (include num_active) ----------
EDGE_THRESH = float(EDGE_THRESH)

def generate_fe1(window, ts):
    w = np.array(window, dtype=float)
    center = len(w)//2
    diff = np.diff(w)
    lags = {f"lag_{s}": (w[center-s] if center-s>=0 else 0.0) for s in [1,5,15,30,75,150,300]}
    src1 = w[max(0,center-7):center+1]
    src5 = w[max(0,center-37):center+1]
    slope = np.polyfit(np.arange(len(w)), w, 1)[0] if np.isfinite(w).all() else 0.0
    peak = int(np.argmax(w))
    feats = {
        "mean": w.mean(), "std": w.std(), "max": w.max(), "min": w.min(),
        "range": w.max()-w.min(),
        "leading_edges": float(np.sum(diff> EDGE_THRESH)),
        "trailing_edges": float(np.sum(diff<-EDGE_THRESH)),
        "slope": float(slope),
        "iqr": float(np.percentile(w,75)-np.percentile(w,25)),
        "slope_in":  float(w[peak]-w[0]),
        "slope_out": float(w[-1]-w[peak]),
        "hour": int(ts.hour), "dayofweek": int(ts.dayofweek),
        "roll_mean_1min": (src1.mean() if len(src1) else 0.0),
        "roll_std_1min":  (src1.std()  if len(src1) else 0.0),
        "roll_mean_5min": (src5.mean() if len(src5) else 0.0),
        "roll_std_5min":  (src5.std()  if len(src5) else 0.0),
        "roll_min_5min":  (src5.min()  if len(src5) else 0.0),
        "roll_max_5min":  (src5.max()  if len(src5) else 0.0),
        "num_active": np.float32(get_num_active(ts)),
        **lags
    }
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k,v in feats.items()}

def generate_fe2(window, ts):
    base = generate_fe1(window, ts)    # includes num_active
    # If memory gets tight, thin raw points: window[::2] / window[::3]
    raw = {f"w_{i}": float(v) for i,v in enumerate(window)}
    base.update(raw)
    return base

def generate_baseline(window, ts):
    c = len(window)//2
    return {
        "agg_center": float(window[c]),
        "hour": int(ts.hour),
        "dayofweek": int(ts.dayofweek),
        "num_active": np.float32(get_num_active(ts)),
    }

FEATURE_FUNCS = {"Baseline": generate_baseline, "FE1": generate_fe1, "FE2": generate_fe2}
fe_func = FEATURE_FUNCS[CASE]

# ---------- build windows ----------
df_clean = df_sub[["S_Total", target_appliance]].dropna().iloc[:1_000_000]
X_raw, y_raw, times = create_seq2point_dataset(df_clean, "S_Total", target_appliance, window_size)

split_idx = int(len(X_raw)*0.9)
val_idx   = int(split_idx*0.9)
X_tr, y_tr = np.array(X_raw[:val_idx]),           np.array(y_raw[:val_idx])
X_v,  y_v  = np.array(X_raw[val_idx:split_idx]),  np.array(y_raw[val_idx:split_idx])
X_te, y_te = np.array(X_raw[split_idx:]),         np.array(y_raw[split_idx:])
times_te   = times[split_idx:]

# Free base tables before FE to lower peak RAM
del df_sub, df_clean
gc.collect()

# ---------- normalize ----------
mX, sX = X_tr.mean(), X_tr.std() + 1e-12
my, sy = y_tr.mean(), y_tr.std() + 1e-12
Xtr_n = [(x-mX)/sX for x in X_tr]
Xv_n  = [(x-mX)/sX for x in X_v]
Xte_n = [(x-mX)/sX for x in X_te]
ytr_n = (y_tr-my)/sy
yv_n  = (y_v -my)/sy
yte_n = (y_te-my)/sy

# ---------- feature engineering ----------
df_tr = pd.DataFrame([fe_func(w,t) for w,t in zip(Xtr_n, times[:val_idx])])
df_tr[target_appliance] = ytr_n
df_v  = pd.DataFrame([fe_func(w,t) for w,t in zip(Xv_n,   times[val_idx:split_idx])])
df_v[target_appliance]  = yv_n
df_te = pd.DataFrame([fe_func(w,t) for w,t in zip(Xte_n,  times_te)])
df_te[target_appliance] = yte_n

# Downcast to float32 / int32 to reduce memory
df_tr = downcast_float32(df_tr)
df_v  = downcast_float32(df_v)
df_te = downcast_float32(df_te)

# Free window arrays
del X_raw, y_raw, X_tr, X_v, X_te, y_tr, y_v, y_te, Xtr_n, Xv_n, Xte_n, ytr_n, yv_n, yte_n
gc.collect()

# ---------- train ----------
print("🚀 Training AutoGluon…")
predictor = TabularPredictor(
    label=target_appliance,
    problem_type='regression',
    eval_metric='mean_absolute_error'
).fit(
    train_data=df_tr,
    tuning_data=df_v,
    use_bag_holdout=True,
    time_limit=TIME_LIMIT_SEC,
    presets='best_quality'
)

# ---------- leaderboard ----------
lb = predictor.leaderboard(silent=True)
print("\nFull leaderboard:")
print(lb.to_string(index=False))
best = predictor.model_best
print(f"\nBest model: {best}")
for m in predictor.model_names():
    if m.startswith("WeightedEnsemble"):
        ens = predictor._trainer.load_model(m)
        if isinstance(ens, WeightedEnsembleModel):
            print(f"=== {m} weights ===")
            for mod, w in ens._get_model_weights().items():
                print(f"  {mod}: {w:.3f}")

# ---------- predict & metrics ----------
y_pred_n = predictor.predict(df_te.drop(columns=[target_appliance]))
y_pred   = y_pred_n.astype("float64") * sy + my
y_true   = df_te[target_appliance].astype("float64") * sy + my
print_metrics(y_true, y_pred, f"{CASE} - Raw")

# post-process
y_pp = duration_threshold_filter(y_pred, threshold=threshold, min_on_duration=min_on_duration)
print_metrics(y_true, y_pp, f"{CASE} - PostProcessed")

# ---------- plots ----------
def plot_timeseries(gt, pred, tindex, title, outfile):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(tindex, gt,   linewidth=1.5, label="Ground Truth")
    ax.plot(tindex, pred, linewidth=1.2, linestyle="--", label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("Time"); ax.set_ylabel(f"{target_appliance} Power (kW)")
    ax.legend(loc="center right"); ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    total_days = max(1, (pd.to_datetime(tindex[-1]) - pd.to_datetime(tindex[0])).days)
    interval = max(1, total_days // 10)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    fig.autofmt_xdate()
    plt.tight_layout(); plt.savefig(outfile, dpi=600); plt.show()

raw_plot  = f"{target_appliance}_{CASE}_timeseries_raw.png"
post_plot = f"{target_appliance}_{CASE}_timeseries_post.png"

plot_timeseries(
    gt=y_true, pred=y_pred, tindex=pd.to_datetime(times_te),
    title=f"{target_appliance} – Raw Prediction vs Ground Truth", outfile=raw_plot)
plot_timeseries(
    gt=y_true, pred=y_pp, tindex=pd.to_datetime(times_te),
    title=f"{target_appliance} – Post-Processed Prediction vs Ground Truth", outfile=post_plot)

# ---------- save csv ----------
csv_filename = f"{target_appliance}_{CASE}_predictions.csv"
out = pd.DataFrame({
    "Time":        pd.to_datetime(times_te),
    "GroundTruth": y_true,
    "Pred_Raw":    y_pred,
    "Pred_Post":   y_pp
})
out.to_csv(csv_filename, index=False)
print(f"✅ Saved {csv_filename}")

# ---------- email ----------
def send_email_with_attachments(sender, password, recipient, subject, body, attachments):
    msg = MIMEMultipart()
    msg['From'] = sender; msg['To'] = recipient; msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    for filepath in attachments:
        if not os.path.exists(filepath): continue
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

SUBJECT = f"{target_appliance} {CASE} results"
BODY    = (
    f"Feature case {CASE} complete.\n\n"
    f"Attached:\n"
    f" • Log:  {log_filename}\n"
    f" • CSV:  {csv_filename}\n"
    f" • Plot: {raw_plot}\n"
    f" • Plot: {post_plot}\n"
)
attachments = [log_filename, csv_filename, raw_plot, post_plot]

if SENDER and PASSWORD and RECIPIENT:
    try:
        send_email_with_attachments(SENDER, PASSWORD, RECIPIENT, SUBJECT, BODY, attachments)
    except Exception as e:
        print(f"Email send failed: {e}")
else:
    print("✉️  Email skipped (set NILM_EMAIL_SENDER/PASSWORD/RECIPIENT env vars to enable).")

# ---------- restore stdout & close log ----------
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
print("✅ Done.")
