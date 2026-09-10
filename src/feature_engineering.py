"""
Dallas Building Permits - Phase 1 Step 4: Feature Engineering
==============================================================
Transforms permits_cleaned.csv into a modeling-ready feature matrix.

Architecture: Two-Stage Hurdle Model
    Stage 1 (Classifier): Uses ALL records - predicts zero vs positive value
    Stage 2 (Regressor): Uses positive-value records only - predicts log(Value)

Both stages share the same feature set (except the target column).
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

INPUT_FILE = "permits_cleaned.csv"
OUTPUT_FILE = "permits_features.csv"

print("=" * 70)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 70)

# ============================================================================
# LOAD CLEANED DATA
# ============================================================================
df = pd.read_csv(INPUT_FILE, parse_dates=["Issued Date"])
print(f"\nLoaded {INPUT_FILE}: {len(df):,} rows x {len(df.columns)} columns")
print(f"Stage 1 records: {len(df):,}")
print(f"Stage 2 records: {(df['is_zero_value']==0).sum():,}")


# ============================================================================
# 1. TEMPORAL FEATURES
# ============================================================================
print(f"\n{'='*70}")
print("1. TEMPORAL FEATURES")
print(f"{'='*70}")

# Extract temporal components
df["issue_year"] = df["Issued Date"].dt.year
df["issue_month"] = df["Issued Date"].dt.month
df["issue_quarter"] = df["Issued Date"].dt.quarter

# Days since epoch (2000-01-01) - captures long-term economic trends
epoch = pd.Timestamp("2000-01-01")
df["days_since_epoch"] = (df["Issued Date"] - epoch).dt.days

# Day of week (0=Mon, 6=Sun) - captures weekly filing patterns
df["issue_day_of_week"] = df["Issued Date"].dt.dayofweek

# Is it filed in Q4? (budget year-end rush effect common in government)
df["is_q4"] = (df["issue_quarter"] == 4).astype(int)

print("  Created: issue_year, issue_month, issue_quarter, days_since_epoch,")
print("           issue_day_of_week, is_q4")
print(f"  WHY: Temporal features capture seasonality (monthly/quarterly cycles in")
print(f"       construction), long-term economic trends (real estate boom/bust),")
print(f"       and bureaucratic patterns (year-end filing rushes).")

# Verify
for col in ["issue_year", "issue_month", "issue_quarter", "days_since_epoch"]:
    print(f"  {col}: range [{df[col].min()}, {df[col].max()}]")


# ============================================================================
# 2. CONTRACTOR FEATURES
# ============================================================================
print(f"\n{'='*70}")
print("2. CONTRACTOR FEATURES")
print(f"{'='*70}")

# Contractor permit count - total permits filed by each contractor
contractor_counts = df["contractor_name"].value_counts().to_dict()
df["contractor_permit_count"] = df["contractor_name"].map(contractor_counts).fillna(1)

# Repeat contractor flag - contractors with 10+ permits
df["is_repeat_contractor"] = (df["contractor_permit_count"] >= 10).astype(int)

repeat_pct = df["is_repeat_contractor"].mean() * 100
print(f"  contractor_permit_count: range [{df['contractor_permit_count'].min()}, "
      f"{df['contractor_permit_count'].max()}]")
print(f"  is_repeat_contractor: {repeat_pct:.1f}% of permits are from repeat contractors (10+ permits)")
print(f"  WHY: Repeat contractors likely file similar permits and may correlate with")
print(f"       specific value ranges. A large-volume commercial contractor vs a one-time")
print(f"       homeowner filing a fence permit represent very different value profiles.")

# Contractor city frequency encoding
city_freq = df["contractor_city"].value_counts().to_dict()
df["contractor_city_freq"] = df["contractor_city"].map(city_freq).fillna(1)

print(f"  contractor_city_freq: range [{df['contractor_city_freq'].min()}, "
      f"{df['contractor_city_freq'].max()}]")
print(f"  WHY: Frequency encoding replaces city names with how often they appear.")
print(f"       Dallas-based contractors appear most frequently; out-of-state contractors")
print(f"       are rare but may handle specialty high-value projects.")


# ============================================================================
# 3. CATEGORICAL ENCODING
# ============================================================================
print(f"\n{'='*70}")
print("3. CATEGORICAL ENCODING")
print(f"{'='*70}")

# --- 3A: One-Hot Encode Low-Cardinality Categoricals ---
# Parsed permit components have manageable cardinality
one_hot_cols = ["permit_category", "sub_category", "work_type"]

print(f"  One-hot encoding: {one_hot_cols}")
for col in one_hot_cols:
    n = df[col].nunique()
    print(f"    {col}: {n} unique values")

df = pd.get_dummies(df, columns=one_hot_cols, prefix_sep="_", dtype=int)
new_ohe_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in 
                ["permit_category", "sub_category", "work_type"])]
print(f"  Created {len(new_ohe_cols)} one-hot columns")
print(f"  WHY: One-hot encoding is appropriate for these features because they have")
print(f"       low cardinality (<40 unique values each). The ML model needs numeric")
print(f"       inputs, and one-hot preserves the non-ordinal nature of these categories.")

# --- 3B: Label/Frequency Encode High-Cardinality Categoricals ---

# Land Use - frequency encoding (189 unique is too many for one-hot)
land_use_freq = df["Land Use"].value_counts().to_dict()
df["land_use_freq"] = df["Land Use"].map(land_use_freq)
print(f"\n  Land Use frequency encoded: {df['Land Use'].nunique()} categories -> land_use_freq")
print(f"  WHY: 189 unique Land Use categories would create 189 sparse columns with one-hot.")
print(f"       Frequency encoding captures the relative commonality of each land use type")
print(f"       (single-family dwelling is by far the most common) without explosion of dimensions.")

# Zip Code - target encoding (mean log_value per zip, computed on training data concept)
# For now, compute on the positive-value subset to avoid bias from zeros
pos_mask = df["is_zero_value"] == 0
zip_target_mean = df.loc[pos_mask].groupby("Zip Code")["log_value"].mean().to_dict()
overall_mean = df.loc[pos_mask, "log_value"].mean()
df["zip_target_encoded"] = df["Zip Code"].map(zip_target_mean).fillna(overall_mean)

print(f"\n  Zip Code target encoded: {df['Zip Code'].nunique()} zips -> zip_target_encoded")
print(f"  Range: [{df['zip_target_encoded'].min():.2f}, {df['zip_target_encoded'].max():.2f}]")
print(f"  WHY: Target encoding replaces each ZIP with the mean log(Value) of permits in that")
print(f"       ZIP. This captures neighborhood wealth/development intensity — ZIP 75201")
print(f"       (Uptown Dallas) has higher-value permits than ZIP 75217 (South Dallas).")
print(f"       NOTE: In production, this must be computed on training data only to prevent")
print(f"       data leakage. We will re-encode after the train/val/test split.")

# permit_code - frequency encoding (26 unique, but some are "NA" from parsing)
code_freq = df["permit_code"].value_counts().to_dict()
df["permit_code_freq"] = df["permit_code"].map(code_freq).fillna(1)
print(f"\n  permit_code frequency encoded: {df['permit_code'].nunique()} codes -> permit_code_freq")


# ============================================================================
# 4. INTERACTION FEATURES
# ============================================================================
print(f"\n{'='*70}")
print("4. INTERACTION FEATURES")
print(f"{'='*70}")

# Area per unit of value (for nonzero values) - captures cost efficiency
# Only meaningful for positive-value permits
df["area_value_ratio"] = np.where(
    (df["Value"] > 0) & (df["Area"] > 0),
    df["Area"] / df["Value"],
    0
)

print(f"  area_value_ratio: captures cost per square foot (inverse)")
print(f"  WHY: A permit with 5,000 sqft at $100K vs 500 sqft at $100K represent very")
print(f"       different project types. This ratio helps the model learn construction")
print(f"       cost efficiency patterns.")

# Log-transformed Area (for those > 0)
df["log_area"] = np.where(df["Area"] > 0, np.log1p(df["Area"]), 0)
print(f"  log_area: log(1+Area) for positive areas, 0 otherwise")
print(f"  WHY: Area is right-skewed like Value. Log transformation normalizes it.")


# ============================================================================
# 5. DROP RAW / IDENTIFIER COLUMNS
# ============================================================================
print(f"\n{'='*70}")
print("5. DROP RAW / IDENTIFIER COLUMNS")
print(f"{'='*70}")

# Keep these for reference but exclude from model features
id_cols_to_drop = [
    "Permit Number",    # Unique ID - no predictive signal
    "Permit Type",      # Replaced by parsed components (one-hot encoded)
    "Issued Date",      # Replaced by temporal features
    "Land Use",         # Replaced by land_use_freq
    "Zip Code",         # Replaced by zip_target_encoded
    "permit_code",      # Replaced by permit_code_freq
    "contractor_name",  # Replaced by contractor_permit_count, is_repeat_contractor
    "contractor_city",  # Replaced by contractor_city_freq
]

# Save identifier info separately for later reference (SHAP waterfall, etc.)
df_identifiers = df[["Permit Number", "Permit Type", "Issued Date", "Zip Code", "Land Use"]].copy()
df_identifiers.to_csv("permits_identifiers.csv", index=False)
print(f"  Saved identifier columns to permits_identifiers.csv for later reference")

# Drop from feature matrix
df = df.drop(columns=id_cols_to_drop)

print(f"  Dropped {len(id_cols_to_drop)} raw/identifier columns:")
for c in id_cols_to_drop:
    print(f"    - {c}")
print(f"  WHY: These columns either have no predictive signal (IDs), are too")
print(f"       high-cardinality for direct use, or have been replaced by engineered features.")


# ============================================================================
# 6. FINAL CHECKS & SAVE
# ============================================================================
print(f"\n{'='*70}")
print("6. FINAL CHECKS & SAVE")
print(f"{'='*70}")

# Identify feature columns (everything except targets)
target_cols = ["Value", "is_zero_value", "log_value"]
feature_cols = [c for c in df.columns if c not in target_cols]

print(f"\n  Total columns: {len(df.columns)}")
print(f"  Feature columns: {len(feature_cols)}")
print(f"  Target columns: {len(target_cols)} ({target_cols})")

# Check for any remaining nulls in features
print(f"\n  Null check on feature columns:")
has_nulls = False
for col in feature_cols:
    null_count = df[col].isna().sum()
    if null_count > 0:
        print(f"    {col}: {null_count} nulls!")
        has_nulls = True
if not has_nulls:
    print(f"    All feature columns: 0 nulls")

# Check for infinite values
inf_cols = []
for col in feature_cols:
    if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_cols.append((col, inf_count))
if inf_cols:
    print(f"  WARNING: Infinite values found in: {inf_cols}")
else:
    print(f"  No infinite values in any feature column")

# Print all feature columns with types
print(f"\n  Feature columns:")
for i, col in enumerate(feature_cols, 1):
    print(f"    {i:3d}. {col} ({df[col].dtype})")

# Save
df.to_csv(OUTPUT_FILE, index=False)
file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

print(f"\n{'='*70}")
print("FEATURE ENGINEERING COMPLETE")
print(f"{'='*70}")
print(f"  Output: {OUTPUT_FILE}")
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Size: {file_size:.1f} MB")
print(f"  Feature columns: {len(feature_cols)}")
print(f"  Stage 1 target: is_zero_value (binary)")
print(f"  Stage 2 target: log_value (continuous, {(~df['log_value'].isna()).sum():,} valid)")

print(f"\nReady for Phase 2: Train/Val/Test Split + Modeling")
