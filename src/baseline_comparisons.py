"""
baseline_comparisons.py
=======================
Comparative baseline models evaluated on the exact same 4-fold forward-chaining
cross-validation and 2020 holdout splits as the primary two-stage hurdle model.

Models:
  1. Naive Baselines:
     - Naive Median: predicts the training set median of positive permits
     - Naive Two-Rate Expected Value: (1 - p0) * positive_median
  2. Conventional Single-Stage Log-Linear:
     - Single-stage linear regression (Ridge alpha=1.0) on log(1 + Value) on ALL permits
     - Dual evaluation: log-space and exponentiated dollars with Duan smearing correction
  3. Non-Hurdle Tweedie XGBoost:
     - Single-stage compound Poisson-Gamma regressor (objective='reg:tweedie', tweedie_variance_power=1.5)
     - Fits directly on uncapped dollar Value across ALL permits (including exact zeros)

All models adhere strictly to:
  - 60-day burn-in exclusion from model fitting
  - Zero target leakage
  - Identical fold partitions and preprocessing
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = str(ROOT / "outputs" / "permits_research_cleaned.csv")
OUTPUT_DIR = str(ROOT / "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

M_SMOOTHING = 10.0
BURN_IN_DAYS = 60

FOLD_SPEC = [
    {"name": "Fold 1", "train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-03-31", "label": "2019 Q1 (Winter)"},
    {"name": "Fold 2", "train_end": "2019-03-31", "val_start": "2019-04-01", "val_end": "2019-06-30", "label": "2019 Q2 (Spring)"},
    {"name": "Fold 3", "train_end": "2019-06-30", "val_start": "2019-07-01", "val_end": "2019-09-30", "label": "2019 Q3 (Summer)"},
    {"name": "Fold 4", "train_end": "2019-09-30", "val_start": "2019-10-01", "val_end": "2019-12-31", "label": "2019 Q4 (Year-end)"},
]

print("=" * 80)
print("BENCHMARK COMPARISONS: NAIVE, LOG-LINEAR, AND TWEEDIE XGBOOST")
print("=" * 80)

# 1. LOAD DATA
df = pd.read_csv(INPUT_FILE, parse_dates=["Issued Date"])
df = df.sort_values("Issued Date").reset_index(drop=True)

df["contractor_name"] = df["contractor_name"].fillna("Unknown")
df["contractor_city"] = df["contractor_city"].fillna("Unknown")
df["permit_code"]     = df["permit_code"].fillna("NA")
df["Land Use"]        = df["Land Use"].fillna("Unknown")

# 2. CONTINUOUS EXPANDING FREQUENCIES
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
    ("permit_code", "permit_code_freq"),
    ("Land Use", "land_use_freq")
]:
    df[out_col] = compute_continuous_expanding_count(df, col, out_col)

df["is_repeat_contractor"] = (df["contractor_permit_count"] >= 10).astype(int)

df_dev = df[df["Issued Date"] < "2020-01-01"].copy()
df_holdout_2020 = df[df["Issued Date"] >= "2020-01-01"].copy()

# HELPERS
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
    return df_apply, mapsco_zip_mode, overall_zip_mode

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
    return df_apply, area_group_medians, median_area_train

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
    smoothed_log = (cum_log_sum.add(g_log_mean * M_SMOOTHING, axis=0)).div(cum_pos_cnt + M_SMOOTHING)
    joined2 = df_t.join(smoothed_log.stack().rename("zip_log_val_encoded"), on=["Issued Date", "Zip Code"])
    row_g_log = df_t["Issued Date"].map(g_log_mean).fillna(0.0)
    df_t["zip_log_val_encoded"] = joined2["zip_log_val_encoded"].fillna(row_g_log)

    final_zip_log = df_pos.groupby("Zip Code")["log_value"].sum()
    final_zip_pos = df_pos.groupby("Zip Code").size()
    global_log_train = float(df_pos["log_value"].mean())
    frozen_log_map = ((final_zip_log + M_SMOOTHING * global_log_train) / (final_zip_pos + M_SMOOTHING)).to_dict()

    return df_t, frozen_zero_map, frozen_log_map, global_zero_train, global_log_train

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

def build_features(ohe_cols):
    base = [
        "Area", "log_area",
        "issue_year", "issue_month", "issue_quarter",
        "days_since_epoch", "issue_day_of_week", "is_q4",
        "contractor_permit_count", "is_repeat_contractor",
        "contractor_city_freq", "permit_code_freq", "land_use_freq",
        "zip_zero_rate_encoded", "zip_log_val_encoded"
    ] + ohe_cols
    return base

# BENCHMARK EVALUATION FUNCTION
def evaluate_split(df_train_fold, df_val_fold, split_name):
    # Preprocessing
    df_train_fold, _, _ = impute_zip_codes(df_train_fold, df_train_fold)
    df_val_fold, _, _ = impute_zip_codes(df_train_fold, df_val_fold)
    df_train_fold, _, _ = impute_area(df_train_fold, df_train_fold)
    df_val_fold, _, _ = impute_area(df_train_fold, df_val_fold)

    df_train_fold, frozen_zero_map, frozen_log_map, g_zero, g_log = \
        compute_bayesian_target_encodings_train(df_train_fold)
    df_val_fold["zip_zero_rate_encoded"] = df_val_fold["Zip Code"].map(frozen_zero_map).fillna(g_zero)
    df_val_fold["zip_log_val_encoded"]   = df_val_fold["Zip Code"].map(frozen_log_map).fillna(g_log)

    df_train_fold = add_temporal_features(df_train_fold)
    df_val_fold   = add_temporal_features(df_val_fold)
    df_train_fold, df_val_fold, ohe_cols = add_ohe_features(df_train_fold, df_val_fold)

    feature_cols = build_features(ohe_cols)
    for c in feature_cols:
        if c not in df_train_fold.columns:
            df_train_fold[c] = 0
        if c not in df_val_fold.columns:
            df_val_fold[c] = 0

    # Burn-in exclusion for model fitting
    train_start = df_train_fold["Issued Date"].min()
    burn_in_end = train_start + pd.Timedelta(days=BURN_IN_DAYS)
    df_train_fit = df_train_fold[df_train_fold["Issued Date"] > burn_in_end].copy().reset_index(drop=True)
    assert (df_train_fit["Issued Date"] > burn_in_end).all()
    assert (df_val_fold["Issued Date"] > burn_in_end).all()

    y_val_dollars = df_val_fold["Value"].values
    val_pos_mask = df_val_fold["is_zero_value"] == 0
    y_val_true_log = df_val_fold.loc[val_pos_mask, "log_value"].values

    split_res = {
        "split": split_name,
        "effective_train_size": len(df_train_fit),
        "val_size": len(df_val_fold)
    }

    # -------------------------------------------------------------
    # 1. NAIVE BASELINES
    # -------------------------------------------------------------
    train_pos_vals = df_train_fit.loc[df_train_fit["Value"] > 0, "Value"]
    naive_median_pos = float(train_pos_vals.median())
    naive_mean_pos = float(train_pos_vals.mean())
    p_zero_train = float(df_train_fit["is_zero_value"].mean())

    # Baseline 1a: Naive Constant Positive Median
    pred_naive_median = np.full(len(df_val_fold), naive_median_pos)
    split_res["naive_median"] = {
        "mae_dollars": float(mean_absolute_error(y_val_dollars, pred_naive_median)),
        "rmse_dollars": float(np.sqrt(mean_squared_error(y_val_dollars, pred_naive_median))),
        "r2_dollars": float(r2_score(y_val_dollars, pred_naive_median)),
    }

    # Baseline 1b: Naive Two-Rate Expected Value: (1 - p0) * positive_median
    pred_naive_tworate = np.full(len(df_val_fold), (1.0 - p_zero_train) * naive_median_pos)
    split_res["naive_tworate"] = {
        "mae_dollars": float(mean_absolute_error(y_val_dollars, pred_naive_tworate)),
        "rmse_dollars": float(np.sqrt(mean_squared_error(y_val_dollars, pred_naive_tworate))),
        "r2_dollars": float(r2_score(y_val_dollars, pred_naive_tworate)),
    }

    # -------------------------------------------------------------
    # 2. CONVENTIONAL SINGLE-STAGE LOG-LINEAR (RIDGE REGRESSION)
    # -------------------------------------------------------------
    # Target: log1p(Value) across ALL rows (zeros become 0.0).
    # In standard econometric log-linear modeling, raw linear Area is excluded
    # to avoid exponential scaling, retaining log_area as the elasticity feature.
    # Features are standardized via StandardScaler and regularized via RidgeCV.
    ridge_features = [c for c in feature_cols if c != "Area"]
    y_train_log1p = np.log1p(df_train_fit["Value"].values)
    X_train_ridge = np.nan_to_num(df_train_fit[ridge_features].values, nan=0.0)
    X_val_ridge   = np.nan_to_num(df_val_fold[ridge_features].values, nan=0.0)

    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]))
    ])
    ridge_pipeline.fit(X_train_ridge, y_train_log1p)
    best_alpha = float(ridge_pipeline.named_steps["ridge"].alpha_)

    train_pred_log1p = ridge_pipeline.predict(X_train_ridge)
    ols_residuals = y_train_log1p - train_pred_log1p
    ols_duan_s = float(np.mean(np.exp(ols_residuals)))

    val_pred_log1p = ridge_pipeline.predict(X_val_ridge)
    # Dollar prediction using Duan smearing: S * exp(log1p) - 1, bounded at 0
    val_pred_dollars_ols = np.maximum(0.0, ols_duan_s * np.expm1(val_pred_log1p))

    # Log metrics on positives (native log optimization space)
    val_pred_log_pos = val_pred_log1p[val_pos_mask]
    r2_log_ols = float(r2_score(y_val_true_log, val_pred_log_pos))
    rmse_log_ols = float(np.sqrt(mean_squared_error(y_val_true_log, val_pred_log_pos)))
    mae_log_ols = float(mean_absolute_error(y_val_true_log, val_pred_log_pos))

    split_res["log_linear_ridge"] = {
        "best_alpha": best_alpha,
        "r2_log": r2_log_ols,
        "rmse_log": rmse_log_ols,
        "mae_log": mae_log_ols,
        "duan_factor": ols_duan_s,
        "mae_dollars": float(mean_absolute_error(y_val_dollars, val_pred_dollars_ols)),
        "rmse_dollars": float(np.sqrt(mean_squared_error(y_val_dollars, val_pred_dollars_ols))),
        "r2_dollars": float(r2_score(y_val_dollars, val_pred_dollars_ols)),
    }

    # -------------------------------------------------------------
    # 3. NON-HURDLE TWEEDIE XGBOOST
    # -------------------------------------------------------------
    # Fits directly on Value (dollars) using compound Poisson-Gamma distribution
    tweedie_params = dict(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        n_estimators=400,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    reg_tweedie = xgb.XGBRegressor(**tweedie_params)
    reg_tweedie.fit(df_train_fit[feature_cols], df_train_fit["Value"].values)

    val_pred_dollars_tweedie = np.maximum(0.0, reg_tweedie.predict(df_val_fold[feature_cols]))

    split_res["tweedie_xgboost"] = {
        "mae_dollars": float(mean_absolute_error(y_val_dollars, val_pred_dollars_tweedie)),
        "rmse_dollars": float(np.sqrt(mean_squared_error(y_val_dollars, val_pred_dollars_tweedie))),
        "r2_dollars": float(r2_score(y_val_dollars, val_pred_dollars_tweedie)),
    }

    return split_res

# RUN FORWARD-CHAINING CROSS-VALIDATION
benchmark_folds = []
for spec in FOLD_SPEC:
    train_end = pd.Timestamp(spec["train_end"])
    val_start = pd.Timestamp(spec["val_start"])
    val_end   = pd.Timestamp(spec["val_end"])

    train_mask = df_dev["Issued Date"] <= train_end
    val_mask = (df_dev["Issued Date"] >= val_start) & (df_dev["Issued Date"] <= val_end)

    df_train_fold = df_dev[train_mask].copy().reset_index(drop=True)
    df_val_fold   = df_dev[val_mask].copy().reset_index(drop=True)

    print(f"\nRunning Benchmark: {spec['name']} ({spec['label']})...")
    res = evaluate_split(df_train_fold, df_val_fold, spec["name"])
    benchmark_folds.append(res)
    print(f"  Naive Median:     R2$ = {res['naive_median']['r2_dollars']:.4f} | MAE$ = ${res['naive_median']['mae_dollars']:,.2f}")
    print(f"  Log-Linear Ridge: R2$ = {res['log_linear_ridge']['r2_dollars']:.4f} | MAE$ = ${res['log_linear_ridge']['mae_dollars']:,.2f} | R2_log = {res['log_linear_ridge']['r2_log']:.4f}")
    print(f"  Tweedie XGBoost:  R2$ = {res['tweedie_xgboost']['r2_dollars']:.4f} | MAE$ = ${res['tweedie_xgboost']['mae_dollars']:,.2f}")

# RUN 2020 HOLDOUT
print("\nRunning Benchmark: 2020 Holdout...")
df_final_train = df_dev.copy().reset_index(drop=True)
df_final_test  = df_holdout_2020.copy().reset_index(drop=True)
benchmark_holdout = evaluate_split(df_final_train, df_final_test, "2020 Holdout")
print(f"  Naive Median:     R2$ = {benchmark_holdout['naive_median']['r2_dollars']:.4f} | MAE$ = ${benchmark_holdout['naive_median']['mae_dollars']:,.2f}")
print(f"  Log-Linear Ridge: R2$ = {benchmark_holdout['log_linear_ridge']['r2_dollars']:.4f} | MAE$ = ${benchmark_holdout['log_linear_ridge']['mae_dollars']:,.2f} | R2_log = {benchmark_holdout['log_linear_ridge']['r2_log']:.4f}")
print(f"  Tweedie XGBoost:  R2$ = {benchmark_holdout['tweedie_xgboost']['r2_dollars']:.4f} | MAE$ = ${benchmark_holdout['tweedie_xgboost']['mae_dollars']:,.2f}")

# COMPUTE CROSS-FOLD MEANS
summary = {}
for model in ["naive_median", "naive_tworate", "log_linear_ridge", "tweedie_xgboost"]:
    summary[model] = {}
    sample_res = benchmark_folds[0][model]
    for metric in sample_res.keys():
        vals = [f[model][metric] for f in benchmark_folds]
        summary[model][f"{metric}_mean"] = float(np.mean(vals))
        summary[model][f"{metric}_std"] = float(np.std(vals))

final_output = {
    "description": "Benchmark comparative models (Naive, Log-Linear Ridge, Tweedie XGBoost) evaluated across identical forward-chaining folds and 2020 holdout (burn-in fixed)",
    "folds": benchmark_folds,
    "holdout_2020": benchmark_holdout,
    "cross_fold_summary": summary
}

out_path = os.path.join(OUTPUT_DIR, "baseline_comparison_results.json")
with open(out_path, "w") as f:
    json.dump(final_output, f, indent=2)

print(f"\nBenchmark results successfully saved to: {out_path}")
print("=" * 80)
