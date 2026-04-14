#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

"""
Feature Analysis Script for NILM Thesis
----------------------------------------
Reads Pecan Street CSV, builds S2P windows, generates Baseline/FE1/FE2 features,
then runs:
  1. Correlation dendrogram (FE1 full, FE2 truncated)
  2. Surrogate Random Forest with feature importance
  3. SHAP summary plot (feature significance + direction of effect)
  4. SHAP dependence plots for top 5 features (linear/nonlinear effects + interactions)
  5. Ablation summary table (metric changes across feature sets)
 
Run on one appliance at a time. Change APPLIANCE and TARGET_DATAID as needed.
Suggested runs: car1 (EV — good performance) and air1 (air conditioning — sparse/poor)
"""
 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
 
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
# ============================================================
# PARAMETERS — change these per appliance/run
# ============================================================
 
CSV_PATH         = "1minute_data_newyork.csv"
TARGET_DATAID    = 3000
APPLIANCE        = "solar"    # e.g. "car1", "air1", "solar"
 
WINDOW_SIZE      = 599       # must be odd — same as thesis
EDGE_THRESH      = 1.0       # same as thesis
 
DATASET_FRACTION = 0.10      # use 10% of data for speed
OUT_DIR          = "feature_analysis_outputs"
 
N_SHAP_SAMPLES   = 200       # number of test samples for SHAP (keep low for speed)
N_DEPEND_FEATS   = 5         # number of top features to generate dependence plots for
 
# ============================================================
# SETUP
# ============================================================
 
os.makedirs(OUT_DIR, exist_ok=True)
 
 
# ============================================================
# STEP 1 — LOAD AND ALIGN DATA
# ============================================================
 
def load_data(csv_path, target_dataid, appliance, fraction=1.0):
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    df["localminute"] = pd.to_datetime(df["localminute"], utc=True, errors="coerce")
    df.dropna(subset=["localminute", "dataid"], inplace=True)
 
    S_Total = df.groupby("localminute")["grid"].sum().rename("S_Total")
 
    df_target = df[df["dataid"] == target_dataid].set_index("localminute")
    if appliance not in df_target.columns:
        raise ValueError(f"Appliance '{appliance}' not found for dataid {target_dataid}.")
 
    idx = S_Total.index.intersection(df_target.index)
    df_clean = pd.DataFrame({
        "S_Total": S_Total.loc[idx],
        appliance: df_target.loc[idx, appliance]
    }).dropna()
 
    if fraction < 1.0:
        n = max(1, int(len(df_clean) * fraction))
        df_clean = df_clean.iloc[:n].copy()
 
    print(f"Data loaded: {len(df_clean)} rows")
    return df_clean
 
 
# ============================================================
# STEP 2 — SEQ2POINT WINDOWS
# ============================================================
 
def make_windows(df_clean, appliance, window_size):
    half = window_size // 2
    X, y, times = [], [], []
    vals_x = df_clean["S_Total"].values
    vals_y = df_clean[appliance].values
    idx    = df_clean.index
 
    for i in tqdm(range(half, len(df_clean) - half), desc="S2P windows"):
        X.append(vals_x[i - half: i + half + 1])
        y.append(vals_y[i])
        times.append(idx[i])
 
    return np.array(X), np.array(y), times
 
 
def train_test_split_90(X, y, times):
    n  = len(X)
    tr = int(n * 0.8)
    te = int(n * 0.9)
    return (
        (X[:tr],   y[:tr],   times[:tr]),
        (X[tr:te], y[tr:te], times[tr:te]),
        (X[te:],   y[te:],   times[te:])
    )
 
 
# ============================================================
# STEP 3 — FEATURE ENGINEERING
# (identical logic to thesis)
# ============================================================
 
def baseline_feats(window, ts):
    c = len(window) // 2
    return {
        "agg_center": float(window[c]),
        "hour":       int(ts.hour),
        "dayofweek":  int(ts.dayofweek),
    }
 
 
def fe1_feats(window, ts):
    w    = np.array(window, dtype=float)
    c    = len(w) // 2
    diff = np.diff(w)
 
    lags = {f"lag_{s}": float(w[c - s]) if (c - s) >= 0 else 0.0
            for s in [1, 5, 15, 30, 75, 150, 300]}
 
    w_1min = w[max(0, c - 7):  c + 1]
    w_5min = w[max(0, c - 37): c + 1]
    peak   = int(np.argmax(w))
 
    try:
        slope = float(np.polyfit(np.arange(len(w)), w, 1)[0])
    except Exception:
        slope = 0.0
 
    feats = {
        "agg_center":     float(w[c]),
        "hour":           int(ts.hour),
        "dayofweek":      int(ts.dayofweek),
        "mean":           float(np.mean(w)),
        "std":            float(np.std(w)),
        "max":            float(np.max(w)),
        "min":            float(np.min(w)),
        "range":          float(np.max(w) - np.min(w)),
        "iqr":            float(np.percentile(w, 75) - np.percentile(w, 25)),
        "leading_edges":  float(np.sum(diff >  EDGE_THRESH)),
        "trailing_edges": float(np.sum(diff < -EDGE_THRESH)),
        "slope":          slope,
        "slope_in":       float(w[peak] - w[0]),
        "slope_out":      float(w[-1]   - w[peak]),
        "roll_mean_1min": float(np.mean(w_1min)),
        "roll_std_1min":  float(np.std(w_1min)),
        "roll_mean_5min": float(np.mean(w_5min)),
        "roll_std_5min":  float(np.std(w_5min)),
        "roll_min_5min":  float(np.min(w_5min)),
        "roll_max_5min":  float(np.max(w_5min)),
        **lags,
    }
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in feats.items()}
 
 
def fe2_feats(window, ts):
    base = fe1_feats(window, ts)
    raw  = {f"w_{i}": float(v) for i, v in enumerate(window)}
    return {**base, **raw}
 
 
FEAT_FUNCS = {"Baseline": baseline_feats, "FE1": fe1_feats, "FE2": fe2_feats}
 
 
def build_features(X_windows, times, case):
    fn   = FEAT_FUNCS[case]
    rows = [fn(w, t) for w, t in tqdm(
        zip(X_windows, times), total=len(X_windows), desc=f"Features {case}")]
    df   = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df
 
 
# ============================================================
# STEP 4 — CORRELATION DENDROGRAM
# FE1: fully labelled | FE2: truncated (high dimensional)
# Low merge height = highly correlated (redundant) features
# ============================================================
 
def plot_dendrogram(X_df, case_name, out_dir, n_clusters=6, truncate_p=25):
    print(f"\nDendrogram: {case_name}")
 
    corr = X_df.corr(method="spearman").fillna(0.0)
    dist = np.clip(1.0 - np.abs(corr.values), 0.0, None)
    np.fill_diagonal(dist, 0.0)
 
    condensed = squareform(dist, checks=False)
    Z         = linkage(condensed, method="average")
 
    n_feats  = X_df.shape[1]
    is_large = n_feats > 60
 
    fig, ax = plt.subplots(figsize=(16, 6))
 
    if is_large:
        dendrogram(Z, ax=ax, truncate_mode="lastp", p=truncate_p,
                   show_leaf_counts=True, leaf_rotation=90, leaf_font_size=9)
        ax.set_xlabel(f"Merged clusters shown (total features: {n_feats})")
    else:
        dendrogram(Z, ax=ax, labels=X_df.columns.tolist(),
                   leaf_rotation=90, leaf_font_size=8)
 
    ax.set_ylabel("Distance (1 - |Spearman|)")
    ax.set_title(f"{case_name} — Feature Correlation Dendrogram\n"
                 f"Low merge height = highly correlated (redundant) features")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{case_name}_dendrogram.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")
 
    labels     = fcluster(Z, t=n_clusters, criterion="maxclust")
    cluster_df = pd.DataFrame({
        "feature": X_df.columns,
        "cluster": labels
    }).sort_values("cluster")
    cluster_df.to_csv(os.path.join(out_dir, f"{case_name}_clusters.csv"), index=False)
 
    return cluster_df
 
 
# ============================================================
# STEP 5 — SURROGATE RF + FEATURE IMPORTANCE
#           + SHAP SUMMARY + SHAP DEPENDENCE PLOTS
#
# SHAP summary plot:
#   - Which features matter most (ranked by mean |SHAP|)
#   - Direction of effect (red = high value increases prediction,
#     blue = high value decreases prediction)
#
# SHAP dependence plots (one per top feature):
#   - X axis: raw feature value
#   - Y axis: SHAP contribution to prediction
#   - Shape of curve reveals linear vs nonlinear effect
#     (straight line = linear, curve/step = nonlinear)
#   - Colour = automatically chosen interacting feature
#     (colour gradient running diagonally = interaction present)
# ============================================================
 
def run_surrogate(X_tr, y_tr, X_te, y_te, case_name, out_dir,
                  n_shap=200, n_depend_feats=5):
    print(f"\nSurrogate RF: {case_name}")
 
    scaler  = MinMaxScaler()
    y_tr_sc = scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
 
    rf = RandomForestRegressor(n_estimators=100, max_depth=12,
                               n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr_sc)
 
    y_pred_sc       = rf.predict(X_te)
    y_pred_unscaled = scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    mae  = mean_absolute_error(y_te, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred_unscaled))
    print(f"  Surrogate RF  MAE={mae:.4f}  RMSE={rmse:.4f}")
 
    # Feature importance bar chart
    importance = pd.Series(rf.feature_importances_, index=X_tr.columns)
    top20      = importance.nlargest(20).sort_values()
 
    fig, ax = plt.subplots(figsize=(8, 7))
    top20.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{case_name} — Top 20 Feature Importances (Surrogate RF)")
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{case_name}_feature_importance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")
 
    importance.sort_values(ascending=False).to_csv(
        os.path.join(out_dir, f"{case_name}_feature_importance.csv"),
        header=["importance"])
 
    # SHAP values
    bg_idx    = np.random.choice(len(X_tr), size=min(100, len(X_tr)), replace=False)
    explainer = shap.TreeExplainer(rf, X_tr.iloc[bg_idx])
 
    shap_idx  = np.random.choice(len(X_te), size=min(n_shap, len(X_te)), replace=False)
    X_shap    = X_te.iloc[shap_idx]
    shap_vals = explainer.shap_values(X_shap)
 
    # SHAP summary plot
    shap.summary_plot(shap_vals, X_shap, max_display=20, show=False)
    plt.title(f"{case_name} — SHAP Summary\n"
              f"Red = high value increases prediction | Blue = decreases prediction")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{case_name}_shap_summary.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {path}")
 
    # SHAP dependence plots — top N engineered features only
    # For FE2 exclude raw w_ points as they are not interpretable
    if case_name == "FE2":
        candidate_feats = [f for f in importance.index if not f.startswith("w_")]
    else:
        candidate_feats = importance.index.tolist()
 
    top_feats = [f for f in importance.sort_values(ascending=False).index
                 if f in candidate_feats][:n_depend_feats]
 
    print(f"\nSHAP dependence plots for: {top_feats}")
 
    for feat in top_feats:
        if feat not in X_shap.columns:
            continue
 
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(
            feat,
            shap_vals,
            X_shap,
            interaction_index="auto",
            ax=ax,
            show=False
        )
        ax.set_title(
            f"{case_name} — SHAP Dependence: {feat}\n"
            f"Curve shape = linear/nonlinear effect | Colour = interacting feature"
        )
        fig.tight_layout()
        safe_feat = feat.replace("/", "_")
        path = os.path.join(out_dir, f"{case_name}_shap_dependence_{safe_feat}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved: {path}")
 
    return rf, importance, mae, rmse
 
 
# ============================================================
# STEP 6 — ABLATION TABLE
# Surrogate RF MAE/RMSE across Baseline, FE1, FE2
# Shows whether adding features helped or hurt (adverse effects)
# ============================================================
 
def save_ablation_table(results, out_dir):
    df = pd.DataFrame(results, columns=["Feature Set", "MAE", "RMSE"])
    df.to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False)
    print("\nAblation summary (Surrogate RF):")
    print(df.to_string(index=False))
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
    df_clean = load_data(CSV_PATH, TARGET_DATAID, APPLIANCE, DATASET_FRACTION)
 
    X_raw, y_raw, times_raw = make_windows(df_clean, APPLIANCE, WINDOW_SIZE)
    (X_tr, y_tr, t_tr), _, (X_te, y_te, t_te) = train_test_split_90(
        X_raw, y_raw, times_raw)
 
    ablation_results = []
 
    for case in ["Baseline", "FE1", "FE2"]:
        print(f"\n{'=' * 55}")
        print(f"  Case: {case}")
        print(f"{'=' * 55}")
 
        case_dir = os.path.join(OUT_DIR, APPLIANCE, case)
        os.makedirs(case_dir, exist_ok=True)
 
        X_feats_tr = build_features(X_tr, t_tr, case)
        X_feats_te = build_features(X_te, t_te, case)
 
        # Dendrogram — skip Baseline (only 3 features)
        if case in ["FE1", "FE2"]:
            plot_dendrogram(X_feats_tr, case, case_dir)
 
        # Surrogate RF + all SHAP plots
        _, _, mae, rmse = run_surrogate(
            X_feats_tr, y_tr,
            X_feats_te, y_te,
            case_name=case,
            out_dir=case_dir,
            n_shap=N_SHAP_SAMPLES,
            n_depend_feats=N_DEPEND_FEATS
        )
 
        ablation_results.append([case, round(mae, 4), round(rmse, 4)])
 
    save_ablation_table(ablation_results, os.path.join(OUT_DIR, APPLIANCE))
 
    print(f"\nAll done. Outputs saved to: {OUT_DIR}")
 
 
if __name__ == "__main__":
    main()