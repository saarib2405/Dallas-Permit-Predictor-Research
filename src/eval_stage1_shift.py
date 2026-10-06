"""
eval_stage1_shift.py
====================
Evaluates Stage 1 (zero-value classification) under the zero-value recording pattern shift.

Specifically:
1. Evaluates forward-chaining Stage 1 performance grouped into:
   - Pre-Shift Evaluation: Fold 1 (2019 Q1)
   - Transition Period: Fold 2 (2019 Q2)
   - Post-Shift Evaluation: Fold 3 (2019 Q3), Fold 4 (2019 Q4), and 2020 Holdout
2. Counterfactual Training Horizon Experiment:
   - Evaluates the 2020 Holdout using a Stage 1 classifier trained ONLY on Pre-Shift data
     (2018-01-02 to 2019-03-31, matching Fold 2's training window, before the shift was established)
   - vs. the Production Stage 1 classifier trained on the Full Development Corpus
     (2018-01-02 to 2019-12-31, which covers the new data pattern).
   This demonstrates quantitatively how much holdout generalization depends on
   the training window covering the new recording pattern.
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "outputs" / "permits_research_cleaned.csv"
OUTPUT_FILE = ROOT / "outputs" / "stage1_shift_eval_results.json"

STAGE1_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", verbosity=0
)
BURN_IN_DAYS = 60
M_SMOOTHING = 10.0

print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE, parse_dates=["Issued Date"])
df = df.sort_values("Issued Date").reset_index(drop=True)

df["contractor_name"] = df["contractor_name"].fillna("Unknown")
df["contractor_city"] = df["contractor_city"].fillna("Unknown")
df["permit_code"]     = df["permit_code"].fillna("NA")
df["Land Use"]        = df["Land Use"].fillna("Unknown")

# 1. Continuous expanding frequency features
def compute_continuous_expanding_count(df_sorted, col_name, output_col):
    daily_counts = df_sorted.groupby(["Issued Date", col_name]).size().unstack(fill_value=0)
    cum_prior = daily_counts.cumsum().shift(1, fill_value=0)
    stacked = cum_prior.stack().rename(output_col)
    merged = df_sorted.join(stacked, on=["Issued Date", col_name])
    merged[output_col] = merged[output_col].fillna(0).astype(int)
    return merged[output_col]

for col, out_col in [
    ("contractor_name", "contractor_permit_count"),
    ("contractor_city", "contractor_city_freq"),
    ("permit_code",     "permit_code_freq"),
    ("Land Use",        "land_use_freq"),
]:
    df[out_col] = compute_continuous_expanding_count(df, col, out_col)

df["is_repeat_contractor"] = (df["contractor_permit_count"] >= 10).astype(int)

# Helper functions identical to modeling_pipeline.py
def impute_zip_codes(df_train, df_apply):
    df_apply = df_apply.copy()
    mapsco_zip_mode = (
        df_train.dropna(subset=["Zip Code", "Mapsco"])
        .groupby("Mapsco")["Zip Code"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
        .to_dict()
    )
    overall_zip_mode = df_train["Zip Code"].mode().iloc[0]
    df_apply["Zip Code"] = (
        df_apply["Zip Code"]
        .fillna(df_apply["Mapsco"].map(mapsco_zip_mode))
        .fillna(overall_zip_mode)
    )
    return df_apply

def impute_area(df_train, df_apply):
    df_apply = df_apply.copy()
    area_group_medians = (
        df_train[df_train["Area"] > 0]
        .groupby(["permit_category", "sub_category", "work_type"])["Area"]
        .median()
        .to_dict()
    )
    median_area_train = float(df_train.loc[df_train["Area"] > 0, "Area"].median())

    def _impute_row(row):
        if row["Area"] == 0 or pd.isna(row["Area"]):
            key = (row["permit_category"], row["sub_category"], row["work_type"])
            return area_group_medians.get(key, median_area_train)
        return row["Area"]

    df_apply["Area"] = df_apply.apply(_impute_row, axis=1)
    df_apply["log_area"] = np.where(df_apply["Area"] > 0, np.log1p(df_apply["Area"]), 0.0)
    return df_apply

def compute_bayesian_target_encodings_train(df_train):
    df_t = df_train.copy()
    daily_zeros = df_t.groupby(["Issued Date", "Zip Code"])["is_zero_value"].sum().unstack(fill_value=0)
    daily_total = df_t.groupby(["Issued Date", "Zip Code"]).size().unstack(fill_value=0)
    cum_zeros = daily_zeros.cumsum().shift(1, fill_value=0)
    cum_total = daily_total.cumsum().shift(1, fill_value=0)

    daily_g_zeros = df_t.groupby("Issued Date")["is_zero_value"].sum().cumsum().shift(1, fill_value=0)
    daily_g_total = df_t.groupby("Issued Date").size().cumsum().shift(1, fill_value=0)
    g_rate = pd.Series(
        np.where(daily_g_total > 0, daily_g_zeros / daily_g_total, 0.0),
        index=cum_zeros.index
    )
    smoothed_zero = (cum_zeros.add(g_rate * M_SMOOTHING, axis=0)).div(cum_total + M_SMOOTHING)
    joined = df_t.join(smoothed_zero.stack().rename("zip_zero_rate_encoded"), on=["Issued Date", "Zip Code"])
    
    row_g_rate = df_t["Issued Date"].map(g_rate).fillna(0.0)
    df_t["zip_zero_rate_encoded"] = joined["zip_zero_rate_encoded"].fillna(row_g_rate)

    final_zip_zeros = df_t.groupby("Zip Code")["is_zero_value"].sum()
    final_zip_counts = df_t.groupby("Zip Code").size()
    global_zero_train = float(df_t["is_zero_value"].mean())
    frozen_zero_map = ((final_zip_zeros + M_SMOOTHING * global_zero_train) / (final_zip_counts + M_SMOOTHING)).to_dict()

    return df_t, frozen_zero_map, global_zero_train

def add_temporal_features(df_input):
    d = df_input.copy()
    d["issue_year"]        = d["Issued Date"].dt.year
    d["issue_month"]       = d["Issued Date"].dt.month
    d["issue_quarter"]     = d["Issued Date"].dt.quarter
    d["days_since_epoch"]  = (d["Issued Date"] - pd.Timestamp("2000-01-01")).dt.days
    d["issue_day_of_week"] = d["Issued Date"].dt.dayofweek
    d["is_q4"]             = (d["issue_quarter"] == 4).astype(int)
    return d

def add_ohe_features(df_train, df_val):
    categorical_cols = ["permit_category", "sub_category", "work_type"]
    df_train = pd.get_dummies(df_train, columns=categorical_cols, prefix_sep="_", dtype=int)
    ohe_cols = [c for c in df_train.columns if any(c.startswith(p + "_") for p in categorical_cols)]

    df_val = pd.get_dummies(df_val, columns=categorical_cols, prefix_sep="_", dtype=int)
    for c in ohe_cols:
        if c not in df_val.columns:
            df_val[c] = 0
    extra = [c for c in df_val.columns if any(c.startswith(p + "_") for p in categorical_cols) and c not in ohe_cols]
    df_val = df_val.drop(columns=extra)
    return df_train, df_val, ohe_cols

def build_feature_list(ohe_cols):
    base = [
        "Area", "log_area",
        "issue_year", "issue_month", "issue_quarter",
        "days_since_epoch", "issue_day_of_week", "is_q4",
        "contractor_permit_count", "is_repeat_contractor",
        "contractor_city_freq", "permit_code_freq", "land_use_freq",
    ] + ohe_cols
    return base + ["zip_zero_rate_encoded"]

def prepare_pipeline_split(df_train_raw, df_val_raw):
    df_tr = df_train_raw.copy()
    df_vl = df_val_raw.copy()

    df_vl = impute_zip_codes(df_tr, df_vl)
    df_tr = impute_zip_codes(df_tr, df_tr)

    df_vl = impute_area(df_tr, df_vl)
    df_tr = impute_area(df_tr, df_tr)

    df_tr, frozen_zero, g_zero = compute_bayesian_target_encodings_train(df_tr)
    df_vl["zip_zero_rate_encoded"] = df_vl["Zip Code"].map(frozen_zero).fillna(g_zero)

    df_tr = add_temporal_features(df_tr)
    df_vl = add_temporal_features(df_vl)

    df_tr, df_vl, ohe_cols = add_ohe_features(df_tr, df_vl)
    features_s1 = build_feature_list(ohe_cols)
    return df_tr, df_vl, features_s1

# ==============================================================================
# EXPERIMENT 1: Pre-Shift Trained Model vs Post-Shift Trained Model on 2020 Holdout
# ==============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT: EVALUATING 2020 HOLDOUT WITH PRE-SHIFT VS FULL TRAINING")
print("=" * 80)

df_holdout_2020 = df[df["Issued Date"] >= "2020-01-01"].copy()

# Condition A: Pre-shift training window (up to 2019-03-31, exactly before shift started)
df_preshift_train = df[(df["Issued Date"] >= "2018-01-02") & (df["Issued Date"] <= "2019-03-31")].copy()
train_start_a = df_preshift_train["Issued Date"].min()
burn_in_a = train_start_a + pd.Timedelta(days=BURN_IN_DAYS)

df_tr_a, df_te_a, feat_a = prepare_pipeline_split(df_preshift_train, df_holdout_2020)
df_fit_a = df_tr_a[df_tr_a["Issued Date"] > burn_in_a].copy()

clf_preshift = xgb.XGBClassifier(**STAGE1_PARAMS)
clf_preshift.fit(df_fit_a[feat_a], df_fit_a["is_zero_value"])

prob_zero_a = clf_preshift.predict_proba(df_te_a[feat_a])[:, 1]
pred_zero_a = (prob_zero_a >= 0.5).astype(int)
y_true_2020 = df_te_a["is_zero_value"].values

metrics_preshift_on_2020 = {
    "training_window": "2018-01-02 to 2019-03-31 (Pre-Shift Only)",
    "training_effective_n": len(df_fit_a),
    "accuracy": float(accuracy_score(y_true_2020, pred_zero_a)),
    "precision": float(precision_score(y_true_2020, pred_zero_a)),
    "recall": float(recall_score(y_true_2020, pred_zero_a)),
    "f1": float(f1_score(y_true_2020, pred_zero_a)),
    "roc_auc": float(roc_auc_score(y_true_2020, prob_zero_a)),
}

# Condition B: Full development training window (2018-01-02 to 2019-12-31, covering shift)
df_full_train = df[df["Issued Date"] < "2020-01-01"].copy()
train_start_b = df_full_train["Issued Date"].min()
burn_in_b = train_start_b + pd.Timedelta(days=BURN_IN_DAYS)

df_tr_b, df_te_b, feat_b = prepare_pipeline_split(df_full_train, df_holdout_2020)
df_fit_b = df_tr_b[df_tr_b["Issued Date"] > burn_in_b].copy()

clf_full = xgb.XGBClassifier(**STAGE1_PARAMS)
clf_full.fit(df_fit_b[feat_b], df_fit_b["is_zero_value"])

prob_zero_b = clf_full.predict_proba(df_te_b[feat_b])[:, 1]
pred_zero_b = (prob_zero_b >= 0.5).astype(int)

metrics_full_on_2020 = {
    "training_window": "2018-01-02 to 2019-12-31 (Full Development, covering shift)",
    "training_effective_n": len(df_fit_b),
    "accuracy": float(accuracy_score(y_true_2020, pred_zero_b)),
    "precision": float(precision_score(y_true_2020, pred_zero_b)),
    "recall": float(recall_score(y_true_2020, pred_zero_b)),
    "f1": float(f1_score(y_true_2020, pred_zero_b)),
    "roc_auc": float(roc_auc_score(y_true_2020, prob_zero_b)),
}

print("\n--- 2020 Holdout Stage 1 Performance by Training Window Coverage ---")
print(f"Pre-Shift Trained Model (Fold 2 Horizon: 2018 to 2019 Q1):")
print(f"  Accuracy:  {metrics_preshift_on_2020['accuracy']:.4f}")
print(f"  Precision: {metrics_preshift_on_2020['precision']:.4f}")
print(f"  Recall:    {metrics_preshift_on_2020['recall']:.4f}")
print(f"  F1-Score:  {metrics_preshift_on_2020['f1']:.4f}")
print(f"  ROC-AUC:   {metrics_preshift_on_2020['roc_auc']:.4f}")

print(f"\nFull-Period Trained Model (Covering 2019 Q2-Q4 Shift):")
print(f"  Accuracy:  {metrics_full_on_2020['accuracy']:.4f}")
print(f"  Precision: {metrics_full_on_2020['precision']:.4f}")
print(f"  Recall:    {metrics_full_on_2020['recall']:.4f}")
print(f"  F1-Score:  {metrics_full_on_2020['f1']:.4f}")
print(f"  ROC-AUC:   {metrics_full_on_2020['roc_auc']:.4f}")

# Breakdown specifically on Building Permits in 2020
bldg_mask_2020 = df_te_b["permit_category_Building"] == 1
y_bldg_true = y_true_2020[bldg_mask_2020]

pred_bldg_a = pred_zero_a[bldg_mask_2020]
prob_bldg_a = prob_zero_a[bldg_mask_2020]
pred_bldg_b = pred_zero_b[bldg_mask_2020]
prob_bldg_b = prob_zero_b[bldg_mask_2020]

bldg_eval = {
    "building_permits_total_2020": int(bldg_mask_2020.sum()),
    "building_permits_zero_count_2020": int(y_bldg_true.sum()),
    "building_permits_zero_rate_2020": float(y_bldg_true.mean()),
    "preshift_trained_bldg_recall": float(recall_score(y_bldg_true, pred_bldg_a)),
    "preshift_trained_bldg_precision": float(precision_score(y_bldg_true, pred_bldg_a)),
    "preshift_trained_bldg_f1": float(f1_score(y_bldg_true, pred_bldg_a)),
    "full_trained_bldg_recall": float(recall_score(y_bldg_true, pred_bldg_b)),
    "full_trained_bldg_precision": float(precision_score(y_bldg_true, pred_bldg_b)),
    "full_trained_bldg_f1": float(f1_score(y_bldg_true, pred_bldg_b)),
}

print(f"\n--- Specific Evaluation on 2020 Building Permits (N={bldg_eval['building_permits_total_2020']:,}, Zero % = {bldg_eval['building_permits_zero_rate_2020']*100:.1f}%) ---")
print(f"Pre-Shift Trained Model Building Zero Recall: {bldg_eval['preshift_trained_bldg_recall']:.4f} (F1: {bldg_eval['preshift_trained_bldg_f1']:.4f})")
print(f"Full Trained Model Building Zero Recall:      {bldg_eval['full_trained_bldg_recall']:.4f} (F1: {bldg_eval['full_trained_bldg_f1']:.4f})")

# Breakdown on Non-Building Permits (Signs, Demolitions, Electrical, etc.)
non_bldg_mask = df_te_b["permit_category_Building"] == 0
y_non_bldg = y_true_2020[non_bldg_mask]
pred_non_bldg_a = pred_zero_a[non_bldg_mask]
pred_non_bldg_b = pred_zero_b[non_bldg_mask]

non_bldg_eval = {
    "non_building_total_2020": int(non_bldg_mask.sum()),
    "non_building_zero_count_2020": int(y_non_bldg.sum()),
    "preshift_trained_recall": float(recall_score(y_non_bldg, pred_non_bldg_a)),
    "full_trained_recall": float(recall_score(y_non_bldg, pred_non_bldg_b)),
}

print(f"\n--- Specific Evaluation on 2020 Non-Building Permits (N={non_bldg_eval['non_building_total_2020']:,}) ---")
print(f"Pre-Shift Trained Model Recall: {non_bldg_eval['preshift_trained_recall']:.4f}")
print(f"Full Trained Model Recall:      {non_bldg_eval['full_trained_recall']:.4f}")

# Cross-Fold Grouping Summary
fold_groups = {
    "pre_shift_group": {
        "folds": ["Fold 1 (2019 Q1)"],
        "mean_accuracy": 0.9793,
        "mean_recall": 0.8722,
        "mean_precision": 0.9865,
        "mean_f1": 0.9256,
        "mean_auc": 0.9766,
        "val_zero_rate": 0.1478,
    },
    "transition_period": {
        "folds": ["Fold 2 (2019 Q2)"],
        "accuracy": 0.9480,
        "recall": 0.6969,
        "precision": 0.9774,
        "f1": 0.8137,
        "auc": 0.8535,
        "val_zero_rate": 0.1630,
    },
    "post_shift_group": {
        "folds": ["Fold 3 (2019 Q3)", "Fold 4 (2019 Q4)", "2020 Holdout"],
        "mean_accuracy": float(np.mean([0.8984, 0.9361, metrics_full_on_2020["accuracy"]])),
        "mean_recall": float(np.mean([0.7091, 0.8598, metrics_full_on_2020["recall"]])),
        "mean_precision": float(np.mean([0.9535, 0.9551, metrics_full_on_2020["precision"]])),
        "mean_f1": float(np.mean([0.8134, 0.9050, metrics_full_on_2020["f1"]])),
        "mean_auc": float(np.mean([0.9528, 0.9656, metrics_full_on_2020["roc_auc"]])),
        "mean_val_zero_rate": float(np.mean([0.3124, 0.3539, 0.3745])),
    }
}

out_data = {
    "metrics_preshift_on_2020": metrics_preshift_on_2020,
    "metrics_full_on_2020": metrics_full_on_2020,
    "building_permits_breakdown": bldg_eval,
    "non_building_breakdown": non_bldg_eval,
    "cross_fold_groups": fold_groups
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nResults successfully written to {OUTPUT_FILE}")
