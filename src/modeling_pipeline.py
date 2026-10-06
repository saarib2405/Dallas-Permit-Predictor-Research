"""
Dallas Building Permits -- Research Paper: Leakage-Free Modeling Pipeline
=========================================================================
Project: ASDS 6302 Final Project -- Predict Building Permit Value
Architecture: Two-Stage Hurdle Model (Redesigned, Forward-Chaining)

This script implements the complete pipeline specified in pipeline_redesign_outline.md:
  1. Continuous expanding frequency features across the entire 2018-2020 timeline
  2. 4-fold forward-chaining validation (quarterly 2019 validation windows)
  3. Per-fold imputation (Zip via Mapsco, Area via group medians)
  4. Stage-specific Bayesian target encodings with 60-day cold-start burn-in
  5. Two-stage hurdle training: XGBoost Classifier + XGBoost Regressor
  6. Primary uncapped Stage 2 + sensitivity p99-capped model
  7. Duan Smearing correction for retransformation bias
  8. Dual hurdle scoring: hard-threshold + probability-weighted expected value
  9. Cross-fold aggregation with mean +/- std
 10. Final one-shot 2020 holdout evaluation

Input: permits_research_cleaned.csv (from data_cleaning_v2.py)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import os
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = str(ROOT / "outputs" / "permits_research_cleaned.csv")
OUTPUT_DIR = str(ROOT / "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Smoothing parameter for Bayesian target encodings
M_SMOOTHING = 10.0

# Cold-start burn-in duration (days from start of training)
BURN_IN_DAYS = 60

# XGBoost hyperparameters (shared across folds -- tuning is out of scope here)
STAGE1_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", verbosity=0
)
STAGE2_PARAMS = dict(
    n_estimators=400, max_depth=7, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="rmse", verbosity=0
)

# Forward-chaining fold definitions
# Train always starts at the data origin; validation is the next 2019 quarter
FOLD_SPEC = [
    {"name": "Fold 1", "train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-03-31", "label": "2019 Q1 (Winter)"},
    {"name": "Fold 2", "train_end": "2019-03-31", "val_start": "2019-04-01", "val_end": "2019-06-30", "label": "2019 Q2 (Spring)"},
    {"name": "Fold 3", "train_end": "2019-06-30", "val_start": "2019-07-01", "val_end": "2019-09-30", "label": "2019 Q3 (Summer)"},
    {"name": "Fold 4", "train_end": "2019-09-30", "val_start": "2019-10-01", "val_end": "2019-12-31", "label": "2019 Q4 (Year-end)"},
]

print("=" * 80)
print("RESEARCH PIPELINE -- LEAKAGE-FREE FORWARD-CHAINING HURDLE MODEL")
print("=" * 80)


# ============================================================================
# STEP 1: LOAD & PREPARE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOAD & PREPARE DATA")
print("=" * 80)

df = pd.read_csv(INPUT_FILE, parse_dates=["Issued Date"])
df = df.sort_values("Issued Date").reset_index(drop=True)

# Explicit missingness labeling for categorical features
df["contractor_name"] = df["contractor_name"].fillna("Unknown")
df["contractor_city"] = df["contractor_city"].fillna("Unknown")
df["permit_code"]     = df["permit_code"].fillna("NA")
df["Land Use"]        = df["Land Use"].fillna("Unknown")

print(f"  Loaded {INPUT_FILE}: {len(df):,} rows x {len(df.columns)} columns")
print(f"  Date range: {df['Issued Date'].min().date()} -- {df['Issued Date'].max().date()}")
print(f"  Zero-value permits: {df['is_zero_value'].sum():,} ({df['is_zero_value'].mean()*100:.1f}%)")
print(f"  Positive permits:   {(df['is_zero_value']==0).sum():,} ({(df['is_zero_value']==0).mean()*100:.1f}%)")


# ============================================================================
# STEP 2: CONTINUOUS EXPANDING FREQUENCY FEATURES (ENTIRE TIMELINE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CONTINUOUS EXPANDING FREQUENCY FEATURES")
print("=" * 80)
print("  Computing point-in-time counts across the entire 2018-2020 timeline...")
print("  Each record counts STRICTLY PRIOR records (Issued Date < own date).")


def compute_continuous_expanding_count(df_sorted, col_name, output_col):
    """
    Vectorized continuous expanding count strictly prior to each row's calendar date.

    Algorithm:
      1. Pivot to daily_counts[date, category] = count of records on that date
      2. Cumsum + shift(1) -> each date sees the cumulative count of ALL prior dates
      3. Re-join to the original dataframe

    First occurrences stay 0 (no artificial floor).
    """
    daily_counts = df_sorted.groupby(["Issued Date", col_name]).size().unstack(fill_value=0)
    # Shift by 1 calendar day so date D only sees sum of days < D
    cum_prior = daily_counts.cumsum().shift(1, fill_value=0)
    stacked = cum_prior.stack().rename(output_col)

    merged = df_sorted.join(stacked, on=["Issued Date", col_name])
    merged[output_col] = merged[output_col].fillna(0).astype(int)
    return merged[output_col]


freq_features = [
    ("contractor_name", "contractor_permit_count"),
    ("contractor_city", "contractor_city_freq"),
    ("permit_code",     "permit_code_freq"),
    ("Land Use",        "land_use_freq"),
]

for col, out_col in freq_features:
    df[out_col] = compute_continuous_expanding_count(df, col, out_col)
    print(f"  {out_col}: range [{df[out_col].min()}, {df[out_col].max()}]")

# Re-derive is_repeat_contractor from the continuous expanding count
df["is_repeat_contractor"] = (df["contractor_permit_count"] >= 10).astype(int)
repeat_pct = df["is_repeat_contractor"].mean() * 100
print(f"  is_repeat_contractor: {repeat_pct:.1f}% of permits are from repeat contractors")


# ============================================================================
# STEP 3: QUARANTINE 2020 HOLDOUT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: QUARANTINE 2020 HOLDOUT")
print("=" * 80)

df_dev = df[df["Issued Date"] < "2020-01-01"].copy()
df_holdout_2020 = df[df["Issued Date"] >= "2020-01-01"].copy()

print(f"  Development set (2018-2019): {len(df_dev):,} rows")
print(f"  2020 Holdout (LOCKED):       {len(df_holdout_2020):,} rows "
      f"({df_holdout_2020['Issued Date'].min().date()} -- {df_holdout_2020['Issued Date'].max().date()})")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def impute_zip_codes(df_train, df_apply):
    """
    Fit ZIP imputation from Mapsco grid mode on df_train,
    apply to df_apply. Returns imputed copy.
    """
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
    return df_apply, mapsco_zip_mode, overall_zip_mode


def impute_area(df_train, df_apply):
    """
    Fit area imputation (group medians of positive area) on df_train,
    apply to df_apply. Unseen groups fall back to training positive-area median.
    Returns imputed copy.
    """
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
    return df_apply, area_group_medians, median_area_train


def compute_bayesian_target_encodings_train(df_train):
    """
    Compute stage-specific Bayesian target encodings on df_train using
    expanding priors.

    DESIGN CHOICE (Non-Leaky, Explicit):
      Frozen validation-time ZIP maps are built from the FULL fold (including
      the 60-day burn-in window up to train_end).
      Rationale: All records in the burn-in window occurred strictly prior to
      the validation period. Including them provides maximum valid historical
      geographic coverage for empirical Bayes shrinkage without temporal leakage,
      while model parameter fitting (df_train_fit) strictly excludes the burn-in window.

    Returns:
      - df_train with 'zip_zero_rate_encoded' and 'zip_log_val_encoded' added
      - frozen maps for validation/test application
      - final expanding global priors for unseen-category fallback
    """
    df_t = df_train.copy()

    # ----- Stage 1: Expanding Bayesian zero-rate -----
    daily_zeros = df_t.groupby(["Issued Date", "Zip Code"])["is_zero_value"].sum().unstack(fill_value=0)
    daily_total = df_t.groupby(["Issued Date", "Zip Code"]).size().unstack(fill_value=0)
    cum_zeros = daily_zeros.cumsum().shift(1, fill_value=0)
    cum_total = daily_total.cumsum().shift(1, fill_value=0)

    # Expanding global zero rate strictly prior to date t
    daily_g_zeros = df_t.groupby("Issued Date")["is_zero_value"].sum().cumsum().shift(1, fill_value=0)
    daily_g_total = df_t.groupby("Issued Date").size().cumsum().shift(1, fill_value=0)
    g_rate = pd.Series(
        np.where(daily_g_total > 0, daily_g_zeros / daily_g_total, 0.0),
        index=cum_zeros.index
    )
    # Natural Bayesian smoothing: (cum_zeros + M * g(t)) / (cum_total + M)
    # When cum_total == 0, naturally evaluates to g(t)
    smoothed_zero = (cum_zeros.add(g_rate * M_SMOOTHING, axis=0)).div(cum_total + M_SMOOTHING)
    joined = df_t.join(smoothed_zero.stack().rename("zip_zero_rate_encoded"), on=["Issued Date", "Zip Code"])
    
    # Fallback to date-specific expanding global rate
    row_g_rate = df_t["Issued Date"].map(g_rate).fillna(0.0)
    df_t["zip_zero_rate_encoded"] = joined["zip_zero_rate_encoded"].fillna(row_g_rate)

    # Freeze Stage 1 map at train boundary from FULL fold
    final_zip_zeros = df_t.groupby("Zip Code")["is_zero_value"].sum()
    final_zip_counts = df_t.groupby("Zip Code").size()
    global_zero_train = float(df_t["is_zero_value"].mean())
    frozen_zero_map = ((final_zip_zeros + M_SMOOTHING * global_zero_train) / (final_zip_counts + M_SMOOTHING)).to_dict()

    # ----- Stage 2: Expanding Bayesian mean log(value) on positives -----
    pos_mask_all = df_t["is_zero_value"] == 0
    df_pos = df_t[pos_mask_all].copy()

    daily_log_sum = df_pos.groupby(["Issued Date", "Zip Code"])["log_value"].sum().unstack(fill_value=0)
    daily_pos_cnt = df_pos.groupby(["Issued Date", "Zip Code"]).size().unstack(fill_value=0)
    cum_log_sum = daily_log_sum.cumsum().shift(1, fill_value=0)
    cum_pos_cnt = daily_pos_cnt.cumsum().shift(1, fill_value=0)

    daily_g_log = df_pos.groupby("Issued Date")["log_value"].sum().cumsum().shift(1, fill_value=0)
    daily_g_pos = df_pos.groupby("Issued Date").size().cumsum().shift(1, fill_value=0)
    g_log_mean = pd.Series(
        np.where(daily_g_pos > 0, daily_g_log / daily_g_pos, 0.0),
        index=cum_log_sum.index
    )
    # Natural Bayesian smoothing
    smoothed_log = (cum_log_sum.add(g_log_mean * M_SMOOTHING, axis=0)).div(cum_pos_cnt + M_SMOOTHING)
    joined2 = df_t.join(smoothed_log.stack().rename("zip_log_val_encoded"), on=["Issued Date", "Zip Code"])
    
    row_g_log = df_t["Issued Date"].map(g_log_mean).fillna(0.0)
    df_t["zip_log_val_encoded"] = joined2["zip_log_val_encoded"].fillna(row_g_log)

    # Freeze Stage 2 map at train boundary from FULL fold
    final_zip_log = df_pos.groupby("Zip Code")["log_value"].sum()
    final_zip_pos = df_pos.groupby("Zip Code").size()
    global_log_train = float(df_pos["log_value"].mean())
    frozen_log_map = ((final_zip_log + M_SMOOTHING * global_log_train) / (final_zip_pos + M_SMOOTHING)).to_dict()

    return df_t, frozen_zero_map, frozen_log_map, global_zero_train, global_log_train


def add_temporal_features(df_input):
    """Add deterministic temporal features."""
    d = df_input.copy()
    d["issue_year"]        = d["Issued Date"].dt.year
    d["issue_month"]       = d["Issued Date"].dt.month
    d["issue_quarter"]     = d["Issued Date"].dt.quarter
    d["days_since_epoch"]  = (d["Issued Date"] - pd.Timestamp("2000-01-01")).dt.days
    d["issue_day_of_week"] = d["Issued Date"].dt.dayofweek
    d["is_q4"]             = (d["issue_quarter"] == 4).astype(int)
    return d


def add_ohe_features(df_train, df_val):
    """
    One-hot encode categorical features. Align validation columns to training schema.
    Returns (df_train, df_val, ohe_cols).
    """
    categorical_cols = ["permit_category", "sub_category", "work_type"]
    df_train = pd.get_dummies(df_train, columns=categorical_cols, prefix_sep="_", dtype=int)
    ohe_cols = [c for c in df_train.columns if any(c.startswith(p + "_") for p in categorical_cols)]

    df_val = pd.get_dummies(df_val, columns=categorical_cols, prefix_sep="_", dtype=int)
    for c in ohe_cols:
        if c not in df_val.columns:
            df_val[c] = 0
    # Drop any extra columns in val not in train
    extra = [c for c in df_val.columns if any(c.startswith(p + "_") for p in categorical_cols) and c not in ohe_cols]
    df_val = df_val.drop(columns=extra)

    return df_train, df_val, ohe_cols


def build_feature_lists(ohe_cols):
    """
    Build Stage 1 and Stage 2 feature lists.
    NOTE: area_value_ratio is ELIMINATED (direct target leakage).
    """
    base = [
        "Area", "log_area",
        "issue_year", "issue_month", "issue_quarter",
        "days_since_epoch", "issue_day_of_week", "is_q4",
        "contractor_permit_count", "is_repeat_contractor",
        "contractor_city_freq", "permit_code_freq", "land_use_freq",
    ] + ohe_cols

    features_s1 = base + ["zip_zero_rate_encoded"]
    features_s2 = base + ["zip_log_val_encoded"]
    return features_s1, features_s2


def compute_metrics(y_true_log, y_pred_log, y_true_dollars, forecast_hard, forecast_soft,
                    y_true_zero, pred_prob_zero, duan_factor, val_pos_mask):
    """Compute the full scorecard of metrics for a single fold or holdout."""
    results = {}

    # Stage 1 classification metrics
    pred_zero_label = (pred_prob_zero >= 0.5).astype(int)
    results["s1_accuracy"] = accuracy_score(y_true_zero, pred_zero_label)
    results["s1_precision"] = precision_score(y_true_zero, pred_zero_label, zero_division=0)
    results["s1_recall"] = recall_score(y_true_zero, pred_zero_label, zero_division=0)
    results["s1_f1"] = f1_score(y_true_zero, pred_zero_label, zero_division=0)
    try:
        results["s1_auc"] = roc_auc_score(y_true_zero, pred_prob_zero)
    except ValueError:
        results["s1_auc"] = float("nan")

    # Primary log-space metrics (on positive permits only)
    if val_pos_mask.sum() > 0:
        results["r2_log"]   = r2_score(y_true_log, y_pred_log)
        results["rmse_log"] = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        results["mae_log"]  = mean_absolute_error(y_true_log, y_pred_log)
    else:
        results["r2_log"]   = float("nan")
        results["rmse_log"] = float("nan")
        results["mae_log"]  = float("nan")

    # Secondary dollar-space metrics
    results["mae_hard"]  = mean_absolute_error(y_true_dollars, forecast_hard)
    results["rmse_hard"] = np.sqrt(mean_squared_error(y_true_dollars, forecast_hard))
    results["r2_hard"]   = r2_score(y_true_dollars, forecast_hard)

    results["mae_soft"]  = mean_absolute_error(y_true_dollars, forecast_soft)
    results["rmse_soft"] = np.sqrt(mean_squared_error(y_true_dollars, forecast_soft))
    results["r2_soft"]   = r2_score(y_true_dollars, forecast_soft)

    results["duan_factor"] = duan_factor

    return results


def print_scorecard(name, results):
    """Pretty-print a fold/holdout scorecard."""
    print(f"\n{'=' * 80}")
    print(f"  {name} SCORECARD")
    print(f"{'=' * 80}")
    print(f"  Stage 1 Classifier:")
    print(f"    Accuracy = {results['s1_accuracy']:.4f}  |  "
          f"Precision = {results['s1_precision']:.4f}  |  "
          f"Recall = {results['s1_recall']:.4f}")
    print(f"    F1 = {results['s1_f1']:.4f}  |  ROC-AUC = {results['s1_auc']:.4f}")
    print(f"  {'-' * 74}")
    print(f"  PRIMARY STABILITY METRICS (Log-Space, Positive Permits):")
    print(f"    R2 (Log-Space):   {results['r2_log']:.4f}  <-- Primary Stability Indicator")
    print(f"    RMSE (Log-Space): {results['rmse_log']:.4f}")
    print(f"    MAE (Log-Space):  {results['mae_log']:.4f}")
    print(f"  {'-' * 74}")
    print(f"  Duan Smearing Factor: S = {results['duan_factor']:.4f}")
    print(f"  SECONDARY ECONOMIC METRICS (Uncapped Dollars):")
    print(f"    [Hard-Threshold]:  MAE = ${results['mae_hard']:>12,.2f}  |  "
          f"RMSE = ${results['rmse_hard']:>14,.2f}  |  R2 = {results['r2_hard']:.4f}")
    print(f"    [Prob-Weighted]:   MAE = ${results['mae_soft']:>12,.2f}  |  "
          f"RMSE = ${results['rmse_soft']:>14,.2f}  |  R2 = {results['r2_soft']:.4f}")
    print(f"{'=' * 80}")


# ============================================================================
# STEP 4: FORWARD-CHAINING FOLD LOOP
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: FORWARD-CHAINING CROSS-VALIDATION")
print("=" * 80)

all_fold_results = []
sensitivity_fold_results = []

for fold_spec in FOLD_SPEC:
    fold_name = fold_spec["name"]
    train_end = pd.Timestamp(fold_spec["train_end"])
    val_start = pd.Timestamp(fold_spec["val_start"])
    val_end   = pd.Timestamp(fold_spec["val_end"])

    print(f"\n{'-' * 80}")
    print(f"  {fold_name}: Train <= {fold_spec['train_end']}  |  "
          f"Val {fold_spec['val_start']} - {fold_spec['val_end']}  ({fold_spec['label']})")
    print(f"{'-' * 80}")

    # 4.1 Partition fold windows (continuous expanding freqs already computed)
    train_mask = df_dev["Issued Date"] <= train_end
    val_mask = (df_dev["Issued Date"] >= val_start) & (df_dev["Issued Date"] <= val_end)

    df_train_fold = df_dev[train_mask].copy().reset_index(drop=True)
    df_val_fold   = df_dev[val_mask].copy().reset_index(drop=True)

    print(f"  Train: {len(df_train_fold):,} rows  |  Val: {len(df_val_fold):,} rows")

    # 4.2 Per-fold imputation (fit on full train fold, apply to both)
    # Using full fold for imputation gives maximum observed spatial coverage prior to validation
    df_train_fold, _, _ = impute_zip_codes(df_train_fold, df_train_fold)
    df_val_fold, _, _ = impute_zip_codes(df_train_fold, df_val_fold)

    df_train_fold, area_medians, area_fallback = impute_area(df_train_fold, df_train_fold)
    df_val_fold, _, _ = impute_area(df_train_fold, df_val_fold)

    # 4.3 Bayesian target encodings (train expanding + frozen map for val)
    # Frozen maps are built from the FULL fold (including burn-in) up to train_end
    df_train_fold, frozen_zero_map, frozen_log_map, global_zero_train, global_log_train = \
        compute_bayesian_target_encodings_train(df_train_fold)

    df_val_fold["zip_zero_rate_encoded"] = df_val_fold["Zip Code"].map(frozen_zero_map).fillna(global_zero_train)
    df_val_fold["zip_log_val_encoded"]   = df_val_fold["Zip Code"].map(frozen_log_map).fillna(global_log_train)

    # 4.4 Temporal features
    df_train_fold = add_temporal_features(df_train_fold)
    df_val_fold   = add_temporal_features(df_val_fold)

    # 4.5 One-hot encoding (aligned)
    df_train_fold, df_val_fold, ohe_cols = add_ohe_features(df_train_fold, df_val_fold)

    # 4.6 Build feature lists
    features_s1, features_s2 = build_feature_lists(ohe_cols)

    # Ensure all features exist
    for f in features_s1 + features_s2:
        if f not in df_train_fold.columns:
            df_train_fold[f] = 0
        if f not in df_val_fold.columns:
            df_val_fold[f] = 0

    # 4.7 EXCLUDE 60-DAY BURN-IN FROM MODEL FITTING (Leakage Fix)
    fold_train_start = df_train_fold["Issued Date"].min()
    burn_in_end = fold_train_start + pd.Timedelta(days=BURN_IN_DAYS)

    df_train_fit = df_train_fold[df_train_fold["Issued Date"] > burn_in_end].copy().reset_index(drop=True)

    # Assertions: No row used in X_train_s1, X_train_s2, or validation has Issued Date <= burn_in_end
    assert (df_train_fit["Issued Date"] > burn_in_end).all(), f"Leak detected: burn-in rows in {fold_name} train_fit!"
    assert (df_val_fold["Issued Date"] > burn_in_end).all(), f"Leak detected: burn-in rows in {fold_name} val_fold!"

    print(f"  Train full history:   {len(df_train_fold):,} rows")
    print(f"  Burn-in excluded:     {len(df_train_fold) - len(df_train_fit):,} rows (Issued Date <= {burn_in_end.date()})")
    print(f"  Effective train fit:  {len(df_train_fit):,} rows")

    # ---- STAGE 1: XGBoost Classifier (is_zero_value) on df_train_fit ----
    X_train_s1 = df_train_fit[features_s1]
    y_train_s1 = df_train_fit["is_zero_value"]

    clf_s1 = xgb.XGBClassifier(**STAGE1_PARAMS)
    clf_s1.fit(X_train_s1, y_train_s1)

    # ---- STAGE 2 PRIMARY: XGBoost Regressor (uncapped log_value) on df_train_fit ----
    train_pos = df_train_fit["is_zero_value"] == 0
    X_train_s2 = df_train_fit.loc[train_pos, features_s2]
    y_train_s2 = df_train_fit.loc[train_pos, "log_value"]

    reg_s2_primary = xgb.XGBRegressor(**STAGE2_PARAMS)
    reg_s2_primary.fit(X_train_s2, y_train_s2)

    # ---- DUAN SMEARING FACTOR (on train_fit residuals only) ----
    train_pred_log = reg_s2_primary.predict(X_train_s2)
    train_residuals = y_train_s2.values - train_pred_log
    duan_factor = float(np.mean(np.exp(train_residuals)))
    print(f"  Duan Smearing Factor S = {duan_factor:.4f}")

    # ---- STAGE 2 SENSITIVITY: p99-capped model on df_train_fit ----
    p99_train = df_train_fit.loc[train_pos, "Value"].quantile(0.99)
    y_train_s2_capped = np.log(np.minimum(df_train_fit.loc[train_pos, "Value"], p99_train))

    reg_s2_sensitivity = xgb.XGBRegressor(**STAGE2_PARAMS)
    reg_s2_sensitivity.fit(X_train_s2, y_train_s2_capped)

    # Sensitivity Duan factor
    sens_pred_log = reg_s2_sensitivity.predict(X_train_s2)
    sens_residuals = y_train_s2_capped.values - sens_pred_log
    duan_factor_sens = float(np.mean(np.exp(sens_residuals)))

    # ---- VALIDATION SCORING ----
    X_val_s1 = df_val_fold[features_s1]
    X_val_s2 = df_val_fold[features_s2]

    val_prob_zero = clf_s1.predict_proba(X_val_s1)[:, 1]
    val_prob_pos  = 1.0 - val_prob_zero

    # PRIMARY model predictions
    val_pred_log = reg_s2_primary.predict(X_val_s2)
    val_pred_pos_dollars = duan_factor * np.exp(val_pred_log)

    forecast_hard = np.where(val_prob_zero >= 0.5, 0.0, val_pred_pos_dollars)
    forecast_soft = val_prob_pos * val_pred_pos_dollars

    # Compute & print PRIMARY metrics
    val_pos_mask = df_val_fold["is_zero_value"] == 0
    y_val_true_log = df_val_fold.loc[val_pos_mask, "log_value"].values
    y_val_pred_log = val_pred_log[val_pos_mask]

    fold_results = compute_metrics(
        y_val_true_log, y_val_pred_log,
        df_val_fold["Value"].values, forecast_hard, forecast_soft,
        df_val_fold["is_zero_value"].values, val_prob_zero,
        duan_factor, val_pos_mask
    )
    fold_results["fold"] = fold_name
    fold_results["train_size_full"] = len(df_train_fold)
    fold_results["train_size"] = len(df_train_fit)
    fold_results["burn_in_excluded"] = len(df_train_fold) - len(df_train_fit)
    fold_results["val_size"] = len(df_val_fold)
    fold_results["p99_cap"] = float(p99_train)
    all_fold_results.append(fold_results)
    print_scorecard(f"{fold_name} -- PRIMARY (Uncapped)", fold_results)

    # SENSITIVITY model predictions
    sens_val_pred_log = reg_s2_sensitivity.predict(X_val_s2)
    sens_val_pred_pos_dollars = duan_factor_sens * np.exp(sens_val_pred_log)
    sens_forecast_hard = np.where(val_prob_zero >= 0.5, 0.0, sens_val_pred_pos_dollars)
    sens_forecast_soft = val_prob_pos * sens_val_pred_pos_dollars

    sens_results = compute_metrics(
        y_val_true_log, sens_val_pred_log[val_pos_mask],
        df_val_fold["Value"].values, sens_forecast_hard, sens_forecast_soft,
        df_val_fold["is_zero_value"].values, val_prob_zero,
        duan_factor_sens, val_pos_mask
    )
    sens_results["fold"] = fold_name
    sens_results["model_type"] = "sensitivity_p99"
    sensitivity_fold_results.append(sens_results)
    print(f"\n  Sensitivity (p99-capped @ ${p99_train:,.0f}):")
    print(f"    R2 (Log) = {sens_results['r2_log']:.4f}  |  "
          f"RMSE (Log) = {sens_results['rmse_log']:.4f}  |  "
          f"MAE$ (Hard) = ${sens_results['mae_hard']:,.2f}")


# ============================================================================
# STEP 5: CROSS-FOLD AGGREGATION (u +/- sigma)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: CROSS-FOLD AGGREGATION (4-Fold Forward-Chaining)")
print("=" * 80)

metric_keys = [
    "s1_accuracy", "s1_f1", "s1_auc",
    "r2_log", "rmse_log", "mae_log",
    "mae_hard", "rmse_hard", "r2_hard",
    "mae_soft", "rmse_soft", "r2_soft",
    "duan_factor",
]

print(f"\n  {'Metric':<25s}  {'Mean':>12s}  {'+/- Std':>12s}")
print(f"  {'-'*25}  {'-'*12}  {'-'*12}")
for key in metric_keys:
    vals = [r[key] for r in all_fold_results if not np.isnan(r.get(key, float("nan")))]
    if vals:
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        if key.startswith("mae_") or key.startswith("rmse_"):
            print(f"  {key:<25s}  ${mean_v:>11,.2f}  ${std_v:>11,.2f}")
        else:
            print(f"  {key:<25s}  {mean_v:>12.4f}  {std_v:>12.4f}")

print(f"\n  NOTE: Fold-length asymmetry -- training grows from 12 to 21 months.")
print(f"        Performance gains may reflect both larger samples and temporal proximity.")


# ============================================================================
# STEP 6: FINAL 2020 HOLDOUT EVALUATION (ONE-SHOT)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: FINAL 2020 HOLDOUT EVALUATION")
print("=" * 80)
print("  Training on ALL 2018-2019 data, evaluating on 2020.")

# Train on all 2018-2019 development data
df_final_train = df_dev.copy().reset_index(drop=True)
df_final_test  = df_holdout_2020.copy().reset_index(drop=True)

print(f"  Final Train: {len(df_final_train):,} rows (2018-01-02 - 2019-12-31)")
print(f"  Final Test:  {len(df_final_test):,} rows (2020-01-01 - 2020-08-29)")

# 6.1 Imputation (fit on 2018-2019, apply to 2020)
df_final_train, _, _ = impute_zip_codes(df_final_train, df_final_train)
df_final_test, _, _ = impute_zip_codes(df_final_train, df_final_test)

df_final_train, _, _ = impute_area(df_final_train, df_final_train)
df_final_test, _, _ = impute_area(df_final_train, df_final_test)

# 6.2 Target encodings (expanding on full 2018-2019, frozen for 2020)
# Frozen maps are built from the FULL 2018-2019 development set (including burn-in)
df_final_train, frozen_zero_final, frozen_log_final, global_zero_final, global_log_final = \
    compute_bayesian_target_encodings_train(df_final_train)

df_final_test["zip_zero_rate_encoded"] = df_final_test["Zip Code"].map(frozen_zero_final).fillna(global_zero_final)
df_final_test["zip_log_val_encoded"]   = df_final_test["Zip Code"].map(frozen_log_final).fillna(global_log_final)

# 6.3 Temporal + OHE
df_final_train = add_temporal_features(df_final_train)
df_final_test  = add_temporal_features(df_final_test)

df_final_train, df_final_test, ohe_cols_final = add_ohe_features(df_final_train, df_final_test)
features_s1_final, features_s2_final = build_feature_lists(ohe_cols_final)

for f in features_s1_final + features_s2_final:
    if f not in df_final_train.columns:
        df_final_train[f] = 0
    if f not in df_final_test.columns:
        df_final_test[f] = 0

# 6.4 EXCLUDE 60-DAY BURN-IN FROM FINAL MODEL FITTING (Leakage Fix)
final_train_start = df_final_train["Issued Date"].min()
final_burn_in_end = final_train_start + pd.Timedelta(days=BURN_IN_DAYS)

df_final_fit = df_final_train[df_final_train["Issued Date"] > final_burn_in_end].copy().reset_index(drop=True)

assert (df_final_fit["Issued Date"] > final_burn_in_end).all(), "Leak detected: burn-in rows in final train fit!"
assert (df_final_test["Issued Date"] > final_burn_in_end).all(), "Leak detected: burn-in rows in 2020 holdout!"

print(f"  Final train full history:  {len(df_final_train):,} rows")
print(f"  Burn-in excluded:          {len(df_final_train) - len(df_final_fit):,} rows (Issued Date <= {final_burn_in_end.date()})")
print(f"  Effective final train fit: {len(df_final_fit):,} rows")

# 6.5 Train final models on df_final_fit
clf_final = xgb.XGBClassifier(**STAGE1_PARAMS)
clf_final.fit(df_final_fit[features_s1_final], df_final_fit["is_zero_value"])

pos_final = df_final_fit["is_zero_value"] == 0
reg_final = xgb.XGBRegressor(**STAGE2_PARAMS)
reg_final.fit(df_final_fit.loc[pos_final, features_s2_final],
              df_final_fit.loc[pos_final, "log_value"])

# Duan smearing factor (final) on df_final_fit residuals
final_train_pred = reg_final.predict(df_final_fit.loc[pos_final, features_s2_final])
final_residuals = df_final_fit.loc[pos_final, "log_value"].values - final_train_pred
duan_final = float(np.mean(np.exp(final_residuals)))
print(f"  Final Duan Smearing Factor S = {duan_final:.4f}")

# 6.6 Score 2020 holdout
test_prob_zero = clf_final.predict_proba(df_final_test[features_s1_final])[:, 1]
test_prob_pos  = 1.0 - test_prob_zero

test_pred_log = reg_final.predict(df_final_test[features_s2_final])
test_pred_pos_dollars = duan_final * np.exp(test_pred_log)

forecast_hard_test = np.where(test_prob_zero >= 0.5, 0.0, test_pred_pos_dollars)
forecast_soft_test = test_prob_pos * test_pred_pos_dollars

test_pos_mask = df_final_test["is_zero_value"] == 0
y_test_true_log = df_final_test.loc[test_pos_mask, "log_value"].values
y_test_pred_log = test_pred_log[test_pos_mask]

holdout_results = compute_metrics(
    y_test_true_log, y_test_pred_log,
    df_final_test["Value"].values, forecast_hard_test, forecast_soft_test,
    df_final_test["is_zero_value"].values, test_prob_zero,
    duan_final, test_pos_mask
)
holdout_results["fold"] = "2020 Holdout"
holdout_results["train_size_full"] = len(df_final_train)
holdout_results["train_size"] = len(df_final_fit)
holdout_results["burn_in_excluded"] = len(df_final_train) - len(df_final_fit)
holdout_results["test_size"] = len(df_final_test)

print_scorecard("2020 HOLDOUT (ONE-SHOT)", holdout_results)


# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: SAVE RESULTS")
print("=" * 80)

# Save cross-fold results to v3.1 (burn-in leak fixed, non-leaky expanding priors)
results_summary = {
    "pipeline_version": "v3.1_burnin_fixed",
    "date_run": datetime.now().isoformat(),
    "data_source": INPUT_FILE,
    "burn_in_days": BURN_IN_DAYS,
    "frozen_zip_maps_scope": "full_fold_including_burnin",
    "folds": all_fold_results,
    "sensitivity_folds": sensitivity_fold_results,
    "holdout_2020": holdout_results,
    "cross_fold_summary": {}
}

for key in metric_keys:
    vals = [r[key] for r in all_fold_results if not np.isnan(r.get(key, float("nan")))]
    if vals:
        results_summary["cross_fold_summary"][key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

results_path = os.path.join(OUTPUT_DIR, "pipeline_results.json")
with open(results_path, "w") as f:
    json.dump(results_summary, f, indent=2, default=str)
print(f"  Results saved to: {results_path}")

# Save feature importance (from final model)
importance_s1 = pd.DataFrame({
    "feature": features_s1_final,
    "importance": clf_final.feature_importances_
}).sort_values("importance", ascending=False)

importance_s2 = pd.DataFrame({
    "feature": features_s2_final,
    "importance": reg_final.feature_importances_
}).sort_values("importance", ascending=False)

imp_path_s1 = os.path.join(OUTPUT_DIR, "feature_importance_stage1.csv")
imp_path_s2 = os.path.join(OUTPUT_DIR, "feature_importance_stage2.csv")
importance_s1.to_csv(imp_path_s1, index=False)
importance_s2.to_csv(imp_path_s2, index=False)
print(f"  Feature importance saved to: {imp_path_s1}")
print(f"  Feature importance saved to: {imp_path_s2}")

print(f"\n{'=' * 80}")
print("PIPELINE COMPLETE")
print(f"{'=' * 80}")
print(f"  * 4-fold forward-chaining cross-validation: DONE")
print(f"  * Continuous expanding frequencies: DONE (no static freezing)")
print(f"  * Per-fold imputation: DONE (no global leakage)")
print(f"  * Bayesian target encodings with cold-start: DONE")
print(f"  * Duan smearing correction: DONE")
print(f"  * Dual hurdle scoring (hard + soft): DONE")
print(f"  * area_value_ratio: ELIMINATED (direct target leakage)")
print(f"  * 2020 Holdout (one-shot): DONE")
