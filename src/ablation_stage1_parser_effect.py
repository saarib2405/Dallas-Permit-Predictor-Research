"""
ablation_stage1_parser_effect.py
================================
Ablation study: Does simply REMOVING work_type_Other and sub_category_General
from Stage 1 features (without adding the new Sign/Demolition categories)
reproduce the same Fold 2 improvement / Folds 1,3,4 regression pattern seen
in the full v3 pipeline?

Approach:
  1. Load v3 cleaned data (has raw Permit Type column)
  2. Re-derive v2 parser outputs inline (old regex, producing work_type_Other
     and sub_category_General for the 13 fallback strings)
  3. Run the FULL v2 pipeline (frequencies, target encodings, OHE, etc.)
  4. For Stage 1 ONLY: drop work_type_Other and sub_category_General from
     the feature list
  5. Stage 2 is completely unchanged (uses all v2 features)
  6. Report metrics side-by-side with v2 and v3
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = str(ROOT / "outputs" / "permits_research_cleaned.csv")
OUTPUT_DIR = str(ROOT / "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
M_SMOOTHING = 10.0
BURN_IN_DAYS = 60

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

FOLD_SPEC = [
    {"name": "Fold 1", "train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-03-31", "label": "2019 Q1"},
    {"name": "Fold 2", "train_end": "2019-03-31", "val_start": "2019-04-01", "val_end": "2019-06-30", "label": "2019 Q2"},
    {"name": "Fold 3", "train_end": "2019-06-30", "val_start": "2019-07-01", "val_end": "2019-09-30", "label": "2019 Q3"},
    {"name": "Fold 4", "train_end": "2019-09-30", "val_start": "2019-10-01", "val_end": "2019-12-31", "label": "2019 Q4"},
]

ABLATION_DROP_COLS = ["work_type_Other", "sub_category_General"]

print("=" * 80)
print("ABLATION: Stage 1 with v2 data, MINUS work_type_Other & sub_category_General")
print("=" * 80)

# ============================================================================
# STEP 0: LOAD DATA AND RE-DERIVE V2 PARSER OUTPUT
# ============================================================================
print("\n[Step 0] Loading data and re-deriving v2 parser output...")
df = pd.read_csv(INPUT_FILE, parse_dates=["Issued Date"])
df = df.sort_values("Issued Date").reset_index(drop=True)
print(f"  Loaded {len(df):,} rows")

# Re-derive v2 parser output from raw Permit Type
def parse_permit_type_v2(pt):
    """Original v2 parser (before Sign/Demolition fix)."""
    match = re.match(r"^(.+?)\s*\((\w{2})\)\s+(.+?)\s{2,}(.+)$", pt)
    if match:
        return (match.group(1).strip(), match.group(2).strip(),
                match.group(3).strip(), match.group(4).strip())
    match2 = re.match(r"^(.+?)\s{2,}(.+)$", pt)
    if match2:
        return (match2.group(1).strip(), "NA", "General", match2.group(2).strip())
    return (pt.strip(), "NA", "General", "Other")

parsed = df["Permit Type"].apply(parse_permit_type_v2)
df["permit_category"] = parsed.apply(lambda x: x[0])
df["permit_code"]     = parsed.apply(lambda x: x[1])
df["sub_category"]    = parsed.apply(lambda x: x[2])
df["work_type"]       = parsed.apply(lambda x: x[3])

# Confirm v2 features restored
print(f"  work_type_Other count:       {(df['work_type'] == 'Other').sum():,}")
print(f"  sub_category_General count:  {(df['sub_category'] == 'General').sum():,}")

# Explicit missingness labeling
df["contractor_name"] = df["contractor_name"].fillna("Unknown")
df["contractor_city"] = df["contractor_city"].fillna("Unknown")
df["permit_code"]     = df["permit_code"].fillna("NA")
df["Land Use"]        = df["Land Use"].fillna("Unknown")

# ============================================================================
# STEP 1: CONTINUOUS EXPANDING FREQUENCY FEATURES
# ============================================================================
print("\n[Step 1] Computing continuous expanding frequency features...")

def compute_continuous_expanding_count(df_sorted, col_name, output_col):
    daily_counts = df_sorted.groupby(["Issued Date", col_name]).size().unstack(fill_value=0)
    cum_prior = daily_counts.cumsum().shift(1, fill_value=0)
    stacked = cum_prior.stack().rename(output_col)
    merged = df_sorted.join(stacked, on=["Issued Date", col_name])
    merged[output_col] = merged[output_col].fillna(0).astype(int)
    return merged[output_col]

for col, out_col in [("contractor_name", "contractor_permit_count"),
                     ("contractor_city", "contractor_city_freq"),
                     ("permit_code", "permit_code_freq"),
                     ("Land Use", "land_use_freq")]:
    df[out_col] = compute_continuous_expanding_count(df, col, out_col)
    print(f"  {out_col}: [{df[out_col].min()}, {df[out_col].max()}]")

df["is_repeat_contractor"] = (df["contractor_permit_count"] >= 10).astype(int)

# ============================================================================
# STEP 2: QUARANTINE 2020 HOLDOUT
# ============================================================================
df_dev = df[df["Issued Date"] < "2020-01-01"].copy()
df_holdout_2020 = df[df["Issued Date"] >= "2020-01-01"].copy()
print(f"\n[Step 2] Dev: {len(df_dev):,}  |  2020 Holdout: {len(df_holdout_2020):,}")

# ============================================================================
# HELPER FUNCTIONS (identical to modeling_pipeline_v2.py)
# ============================================================================

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
    """
    Compute stage-specific Bayesian target encodings on df_train using
    expanding priors. Frozen validation-time ZIP maps are built from the FULL fold.
    """
    df_t = df_train.copy()

    # ----- Stage 1: Expanding Bayesian zero-rate -----
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
    d["days_since_epoch"]  = (d["Issued Date"] - pd.Timestamp("2018-01-01")).dt.days
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


def build_feature_lists(ohe_cols):
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


# ============================================================================
# STEP 3: FOLD LOOP (ABLATION: DROP work_type_Other, sub_category_General FROM S1)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FORWARD-CHAINING CROSS-VALIDATION (ABLATED STAGE 1)")
print("=" * 80)

ablation_results = []

for fold_spec in FOLD_SPEC:
    fold_name = fold_spec["name"]
    train_end = pd.Timestamp(fold_spec["train_end"])
    val_start = pd.Timestamp(fold_spec["val_start"])
    val_end   = pd.Timestamp(fold_spec["val_end"])

    print(f"\n  {fold_name}: Train <= {fold_spec['train_end']}  |  Val {fold_spec['val_start']} - {fold_spec['val_end']}")

    train_mask = df_dev["Issued Date"] <= train_end
    val_mask = (df_dev["Issued Date"] >= val_start) & (df_dev["Issued Date"] <= val_end)

    df_train_fold = df_dev[train_mask].copy().reset_index(drop=True)
    df_val_fold   = df_dev[val_mask].copy().reset_index(drop=True)
    print(f"  Train: {len(df_train_fold):,}  |  Val: {len(df_val_fold):,}")

    # Per-fold imputation
    df_train_fold, _, _ = impute_zip_codes(df_train_fold, df_train_fold)
    df_val_fold, _, _ = impute_zip_codes(df_train_fold, df_val_fold)
    df_train_fold, _, _ = impute_area(df_train_fold, df_train_fold)
    df_val_fold, _, _ = impute_area(df_train_fold, df_val_fold)

    # Target encodings (frozen from FULL fold)
    df_train_fold, frozen_zero_map, frozen_log_map, global_zero_train, global_log_train = \
        compute_bayesian_target_encodings_train(df_train_fold)
    df_val_fold["zip_zero_rate_encoded"] = df_val_fold["Zip Code"].map(frozen_zero_map).fillna(global_zero_train)
    df_val_fold["zip_log_val_encoded"]   = df_val_fold["Zip Code"].map(frozen_log_map).fillna(global_log_train)

    # Temporal + OHE
    df_train_fold = add_temporal_features(df_train_fold)
    df_val_fold   = add_temporal_features(df_val_fold)
    df_train_fold, df_val_fold, ohe_cols = add_ohe_features(df_train_fold, df_val_fold)
    features_s1, features_s2 = build_feature_lists(ohe_cols)

    for f in features_s1 + features_s2:
        if f not in df_train_fold.columns:
            df_train_fold[f] = 0
        if f not in df_val_fold.columns:
            df_val_fold[f] = 0

    # ABLATION: Drop work_type_Other and sub_category_General from Stage 1 ONLY
    features_s1_ablated = [f for f in features_s1 if f not in ABLATION_DROP_COLS]
    dropped = [f for f in ABLATION_DROP_COLS if f in features_s1]
    print(f"  ABLATION: Dropped {dropped} from Stage 1 ({len(features_s1)} -> {len(features_s1_ablated)} features)")

    # Exclude 60-day burn-in from model fitting
    fold_train_start = df_train_fold["Issued Date"].min()
    burn_in_end = fold_train_start + pd.Timedelta(days=BURN_IN_DAYS)
    df_train_fit = df_train_fold[df_train_fold["Issued Date"] > burn_in_end].copy().reset_index(drop=True)
    assert (df_train_fit["Issued Date"] > burn_in_end).all()
    assert (df_val_fold["Issued Date"] > burn_in_end).all()

    # Stage 1 (ABLATED on df_train_fit)
    clf = xgb.XGBClassifier(**STAGE1_PARAMS)
    clf.fit(df_train_fit[features_s1_ablated], df_train_fit["is_zero_value"])

    # Stage 2 (UNCHANGED - uses full v2 feature set on df_train_fit)
    train_pos = df_train_fit["is_zero_value"] == 0
    reg = xgb.XGBRegressor(**STAGE2_PARAMS)
    reg.fit(df_train_fit.loc[train_pos, features_s2], df_train_fit.loc[train_pos, "log_value"])

    # Duan smearing
    train_pred_log = reg.predict(df_train_fit.loc[train_pos, features_s2])
    train_residuals = df_train_fit.loc[train_pos, "log_value"].values - train_pred_log
    duan_factor = float(np.mean(np.exp(train_residuals)))

    # Score validation
    val_prob_zero = clf.predict_proba(df_val_fold[features_s1_ablated])[:, 1]
    val_prob_pos  = 1.0 - val_prob_zero
    val_pred_log = reg.predict(df_val_fold[features_s2])
    val_pred_pos_dollars = duan_factor * np.exp(val_pred_log)
    forecast_hard = np.where(val_prob_zero >= 0.5, 0.0, val_pred_pos_dollars)
    forecast_soft = val_prob_pos * val_pred_pos_dollars

    val_pos_mask = df_val_fold["is_zero_value"] == 0
    y_val_true_log = df_val_fold.loc[val_pos_mask, "log_value"].values
    y_val_pred_log = val_pred_log[val_pos_mask]

    pred_zero_label = (val_prob_zero >= 0.5).astype(int)
    fold_res = {
        "fold": fold_name,
        "s1_recall": recall_score(df_val_fold["is_zero_value"], pred_zero_label, zero_division=0),
        "s1_f1": f1_score(df_val_fold["is_zero_value"], pred_zero_label, zero_division=0),
        "s1_auc": roc_auc_score(df_val_fold["is_zero_value"], val_prob_zero),
        "r2_log": r2_score(y_val_true_log, y_val_pred_log),
        "r2_hard": r2_score(df_val_fold["Value"].values, forecast_hard),
        "duan_factor": duan_factor,
    }
    ablation_results.append(fold_res)
    print(f"  S1 Recall={fold_res['s1_recall']:.4f}  F1={fold_res['s1_f1']:.4f}  "
          f"AUC={fold_res['s1_auc']:.4f}  R2_log={fold_res['r2_log']:.4f}  "
          f"R2_hard={fold_res['r2_hard']:.4f}")

# ============================================================================
# STEP 4: 2020 HOLDOUT (ABLATED STAGE 1)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: 2020 HOLDOUT (ABLATED STAGE 1)")
print("=" * 80)

df_final_train = df_dev.copy().reset_index(drop=True)
df_final_test  = df_holdout_2020.copy().reset_index(drop=True)

df_final_train, _, _ = impute_zip_codes(df_final_train, df_final_train)
df_final_test, _, _ = impute_zip_codes(df_final_train, df_final_test)
df_final_train, _, _ = impute_area(df_final_train, df_final_train)
df_final_test, _, _ = impute_area(df_final_train, df_final_test)

df_final_train, frozen_zero_final, frozen_log_final, global_zero_final, global_log_final = \
    compute_bayesian_target_encodings_train(df_final_train)
df_final_test["zip_zero_rate_encoded"] = df_final_test["Zip Code"].map(frozen_zero_final).fillna(global_zero_final)
df_final_test["zip_log_val_encoded"]   = df_final_test["Zip Code"].map(frozen_log_final).fillna(global_log_final)

df_final_train = add_temporal_features(df_final_train)
df_final_test  = add_temporal_features(df_final_test)
df_final_train, df_final_test, ohe_cols_final = add_ohe_features(df_final_train, df_final_test)
features_s1_final, features_s2_final = build_feature_lists(ohe_cols_final)

for f in features_s1_final + features_s2_final:
    if f not in df_final_train.columns:
        df_final_train[f] = 0
    if f not in df_final_test.columns:
        df_final_test[f] = 0

features_s1_final_ablated = [f for f in features_s1_final if f not in ABLATION_DROP_COLS]
dropped_final = [f for f in ABLATION_DROP_COLS if f in features_s1_final]
print(f"  ABLATION: Dropped {dropped_final} from Stage 1 ({len(features_s1_final)} -> {len(features_s1_final_ablated)} features)")

final_train_start = df_final_train["Issued Date"].min()
final_burn_in_end = final_train_start + pd.Timedelta(days=BURN_IN_DAYS)
df_final_fit = df_final_train[df_final_train["Issued Date"] > final_burn_in_end].copy().reset_index(drop=True)
assert (df_final_fit["Issued Date"] > final_burn_in_end).all()
assert (df_final_test["Issued Date"] > final_burn_in_end).all()

clf_final = xgb.XGBClassifier(**STAGE1_PARAMS)
clf_final.fit(df_final_fit[features_s1_final_ablated], df_final_fit["is_zero_value"])

pos_final = df_final_fit["is_zero_value"] == 0
reg_final = xgb.XGBRegressor(**STAGE2_PARAMS)
reg_final.fit(df_final_fit.loc[pos_final, features_s2_final],
              df_final_fit.loc[pos_final, "log_value"])

final_train_pred = reg_final.predict(df_final_fit.loc[pos_final, features_s2_final])
final_residuals = df_final_fit.loc[pos_final, "log_value"].values - final_train_pred
duan_final = float(np.mean(np.exp(final_residuals)))

test_prob_zero = clf_final.predict_proba(df_final_test[features_s1_final_ablated])[:, 1]
test_prob_pos  = 1.0 - test_prob_zero
test_pred_log = reg_final.predict(df_final_test[features_s2_final])
test_pred_pos_dollars = duan_final * np.exp(test_pred_log)
forecast_hard_test = np.where(test_prob_zero >= 0.5, 0.0, test_pred_pos_dollars)

test_pos_mask = df_final_test["is_zero_value"] == 0
y_test_true_log = df_final_test.loc[test_pos_mask, "log_value"].values
y_test_pred_log = test_pred_log[test_pos_mask]

pred_zero_label_test = (test_prob_zero >= 0.5).astype(int)
holdout_res = {
    "fold": "2020 Holdout",
    "s1_recall": recall_score(df_final_test["is_zero_value"], pred_zero_label_test, zero_division=0),
    "s1_f1": f1_score(df_final_test["is_zero_value"], pred_zero_label_test, zero_division=0),
    "s1_auc": roc_auc_score(df_final_test["is_zero_value"], test_prob_zero),
    "r2_log": r2_score(y_test_true_log, y_test_pred_log),
    "r2_hard": r2_score(df_final_test["Value"].values, forecast_hard_test),
    "duan_factor": duan_final,
}
print(f"  S1 Recall={holdout_res['s1_recall']:.4f}  F1={holdout_res['s1_f1']:.4f}  "
      f"AUC={holdout_res['s1_auc']:.4f}  R2_log={holdout_res['r2_log']:.4f}  "
      f"R2_hard={holdout_res['r2_hard']:.4f}")

# ============================================================================
# STEP 5: COMPARISON TABLE (v3.1 vs Ablation)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: COMPARISON TABLE (Ablation vs v3.1 Burn-in Fixed)")
print("=" * 80)

v3_1_path = os.path.join(OUTPUT_DIR, "pipeline_results.json")
if os.path.exists(v3_1_path):
    v3_1 = json.load(open(v3_1_path))
    compare_metrics = ["s1_recall", "s1_f1", "s1_auc", "r2_log", "r2_hard"]

    print(f"\n  {'Fold':>12s}  {'Metric':>10s}  {'Ablation':>10s}  {'Pipeline':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    for i in range(4):
        for m in compare_metrics:
            abl_val = ablation_results[i][m]
            v3_val = v3_1['folds'][i][m]
            print(f"  {FOLD_SPEC[i]['name']:>12s}  {m:>10s}  {abl_val:>10.4f}  {v3_val:>10.4f}")
        print()

    print(f"  {'Holdout':>12s}")
    for m in compare_metrics:
        abl_val = holdout_res[m]
        v3_val = v3_1['holdout_2020'][m]
        print(f"  {'':>12s}  {m:>10s}  {abl_val:>10.4f}  {v3_val:>10.4f}")

# Save results
output = {
    "description": "Ablation: v2 data with work_type_Other and sub_category_General dropped from Stage 1 only (burn-in fixed)",
    "folds": ablation_results,
    "holdout_2020": holdout_res,
}
abl_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
with open(abl_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Results saved to: {abl_path}")

print("\n" + "=" * 80)
print("ABLATION COMPLETE")
print("=" * 80)
