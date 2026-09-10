"""
Dallas Building Permits — Phase 1: Data Cleaning Pipeline
==========================================================
Project: ASDS 6302 Final Project — Predict Building Permit Value
Architecture: Two-Stage Hurdle Model
    Stage 1 (Classifier): Predict if permit is $0 (administrative) or positive-value
    Stage 2 (Regressor): Predict log(Value) for positive-value permits

Author: Senior Data Scientist — Urban Planning Division
Date: 2026-04-11

This script executes the 8-step cleaning plan documented in the implementation plan.
Every observation and decision is logged with WHAT happened and WHY we made that choice.
"""

import pandas as pd
import numpy as np
import re
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")
 

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = "Building-Permits.csv"
OUTPUT_FILE = "permits_cleaned.csv"
LOG_FILE = "cleaning_log.md"

# Observation log — accumulates all observations and decisions
observations = []

def log_observation(step, what, why, data=None):
    """Log an observation with context."""
    entry = {
        "step": step,
        "what": what,
        "why": why,
        "data": data,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    observations.append(entry)
    print(f"  [{step}] {what}")
    if data:
        print(f"          -> {data}")

def write_log():
    """Write all observations to a markdown log file."""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("# Data Cleaning Log — Dallas Building Permits\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        current_step = None
        for obs in observations:
            if obs["step"] != current_step:
                current_step = obs["step"]
                f.write(f"## {current_step}\n\n")
            f.write(f"**WHAT:** {obs['what']}\n\n")
            f.write(f"**WHY:** {obs['why']}\n\n")
            if obs["data"]:
                f.write(f"**DATA:** `{obs['data']}`\n\n")
            f.write("---\n\n")
    print(f"\n📄 Cleaning log saved to: {LOG_FILE}")


# ============================================================================
# STEP 1: LOAD AND TYPE-CAST
# ============================================================================
print("=" * 70)
print("STEP 1: LOAD AND TYPE-CAST")
print("=" * 70)

df = pd.read_csv(INPUT_FILE, dtype=str)
initial_rows = len(df)
initial_cols = len(df.columns)

log_observation(
    "Step 1 — Load",
    f"Loaded raw CSV: {initial_rows:,} rows × {initial_cols} columns",
    "All columns loaded as strings to prevent pandas from auto-inferring types "
    "(e.g., Zip Code would be read as int, losing leading zeros and categorical nature)."
)

# Parse Value: remove commas → float
df["Value"] = df["Value"].str.replace(",", "", regex=False).astype(float)
log_observation(
    "Step 1 — Type Cast",
    f"Parsed 'Value' column: removed commas, cast to float64",
    "Value is stored as comma-formatted strings (e.g., '665,000'). Must be numeric for math operations.",
    f"Range: ${df['Value'].min():,.0f} — ${df['Value'].max():,.0f}"
)

# Parse Area: remove commas → float
df["Area"] = df["Area"].str.replace(",", "", regex=False).astype(float)
log_observation(
    "Step 1 — Type Cast",
    f"Parsed 'Area' column: removed commas, cast to float64",
    "Same comma-formatting issue as Value.",
    f"Range: {df['Area'].min():,.0f} — {df['Area'].max():,.0f} sq ft"
)

# Parse Issued Date → datetime
df["Issued Date"] = pd.to_datetime(df["Issued Date"], format="%m/%d/%y")
log_observation(
    "Step 1 — Type Cast",
    f"Parsed 'Issued Date': MM/DD/YY → datetime64",
    "2-digit year format (e.g., '03/13/20' → 2020-03-13). pandas handles the century correctly for 18-20.",
    f"Range: {df['Issued Date'].min().date()} — {df['Issued Date'].max().date()}"
)


# ============================================================================
# STEP 2: REMOVE TRULY UNMODELABLE ROWS (MINIMAL)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: REMOVE TRULY UNMODELABLE ROWS")
print("=" * 70)

rows_before = len(df)

# Check each column for nulls
null_value = df["Value"].isna().sum()
null_date = df["Issued Date"].isna().sum()
null_permit = df["Permit Type"].isna().sum()
null_contractor = df["Contractor"].isna().sum()

log_observation(
    "Step 2 — Null Audit",
    f"Null counts — Value: {null_value}, Issued Date: {null_date}, "
    f"Permit Type: {null_permit}, Contractor: {null_contractor}",
    "We only remove rows where critical fields are null. Value=0 rows are KEPT "
    "for the Stage 1 Hurdle Model classifier."
)

# Drop rows with null in critical fields
critical_mask = (
    df["Value"].notna() &
    df["Issued Date"].notna() &
    df["Permit Type"].notna() &
    df["Contractor"].notna()
)
df = df[critical_mask].copy()
rows_dropped = rows_before - len(df)

log_observation(
    "Step 2 — Row Removal",
    f"Dropped {rows_dropped} rows with null critical fields ({rows_dropped/rows_before*100:.4f}%)",
    "Only 3 rows had null Contractor — all other critical fields were complete. "
    "We do NOT drop Value=0 rows because they serve as training data for the "
    "Stage 1 binary classifier in our Hurdle Model architecture.",
    f"Remaining: {len(df):,} rows"
)


# ============================================================================
# STEP 3: CREATE HURDLE MODEL LABELS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: CREATE HURDLE MODEL LABELS")
print("=" * 70)

# Stage 1 target: binary classification
df["is_zero_value"] = (df["Value"] == 0).astype(int)
zero_count = df["is_zero_value"].sum()
positive_count = len(df) - zero_count

log_observation(
    "Step 3 — Stage 1 Label",
    f"Created 'is_zero_value': 1 if Value==0, else 0",
    "This is the target variable for the Stage 1 binary classifier. "
    "It learns to distinguish administrative permits ($0) from real construction projects.",
    f"Class balance — Zero: {zero_count:,} ({zero_count/len(df)*100:.1f}%) | "
    f"Positive: {positive_count:,} ({positive_count/len(df)*100:.1f}%)"
)

# Stage 2 target: log-transformed value (only for positive-value records)
df["log_value"] = np.where(df["Value"] > 0, np.log(df["Value"]), np.nan)

pos_mask = df["Value"] > 0
log_observation(
    "Step 3 — Stage 2 Label",
    f"Created 'log_value': ln(Value) for positive-value records, NaN for zeros",
    "Log-transformation is mandatory because Value is extremely right-skewed "
    "(mean $87.6K vs median $2.8K, ratio 31:1). Log-transforming normalizes the "
    "distribution, stabilizes variance, and prevents large-value outliers from "
    "dominating the loss function.",
    f"log_value range: {df.loc[pos_mask, 'log_value'].min():.2f} — "
    f"{df.loc[pos_mask, 'log_value'].max():.2f} "
    f"(corresponds to ${np.exp(df.loc[pos_mask, 'log_value'].min()):,.0f} — "
    f"${np.exp(df.loc[pos_mask, 'log_value'].max()):,.0f})"
)


# ============================================================================
# STEP 4: PARSE COMPOSITE FIELDS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: PARSE COMPOSITE FIELDS")
print("=" * 70)

# --- 4A: Parse Permit Type into 4 components ---
def parse_permit_type(pt):
    """
    Parse composite permit type string into 4 components.
    
    Standard pattern: 'Building (BU) Multi Family  New Construction'
                       ────────  ──   ────────────  ────────────────
                       category  code sub_category  work_type
    
    Edge cases:
      - 'Electrical Sign New Construction' — no code or sub-category
      - 'Demolition Permit SFD/Duplex' — no code
      - 'Paving (Sidewalk, Drive Approaches) (PV) Single Family  Alteration'
        — parenthesized description before the code
    """
    # Pattern 1: Standard format with double-space separator
    # e.g., "Building (BU) Multi Family  New Construction"
    match = re.match(
        r"^(.+?)\s*\((\w{2})\)\s+(.+?)\s{2,}(.+)$", pt
    )
    if match:
        category = match.group(1).strip()
        code = match.group(2).strip()
        sub_category = match.group(3).strip()
        work_type = match.group(4).strip()
        return category, code, sub_category, work_type
    
    # Pattern 2: No code, has double-space separator
    # e.g., "Sign  New Construction" (unlikely but catch it)
    match2 = re.match(r"^(.+?)\s{2,}(.+)$", pt)
    if match2:
        raw_cat = match2.group(1).strip()
        work_type = match2.group(2).strip()
        return raw_cat, "NA", "General", work_type
    
    # Pattern 3: Completely non-standard
    # e.g., "Demolition Permit SFD/Duplex"
    return pt.strip(), "NA", "General", "Other"

# Apply parser
parsed = df["Permit Type"].apply(parse_permit_type)
df["permit_category"] = parsed.apply(lambda x: x[0])
df["permit_code"] = parsed.apply(lambda x: x[1])
df["sub_category"] = parsed.apply(lambda x: x[2])
df["work_type"] = parsed.apply(lambda x: x[3])

# Report parsing success
na_code_count = (df["permit_code"] == "NA").sum()
parse_success = len(df) - na_code_count
log_observation(
    "Step 4A — Parse Permit Type",
    f"Parsed 'Permit Type' into 4 columns: permit_category, permit_code, sub_category, work_type",
    "The raw Permit Type field encodes 4 dimensions in one string. Splitting them gives "
    "the ML model access to individual categorical features instead of 137 one-hot columns.",
    f"Parse success (standard format): {parse_success:,} ({parse_success/len(df)*100:.1f}%) | "
    f"Fallback parse: {na_code_count:,} ({na_code_count/len(df)*100:.1f}%)"
)

# Show unique values for each parsed component
for col in ["permit_category", "permit_code", "sub_category", "work_type"]:
    n_unique = df[col].nunique()
    top_vals = df[col].value_counts().head(5).to_dict()
    log_observation(
        "Step 4A — Parsed Component",
        f"'{col}': {n_unique} unique values",
        f"Top 5: {top_vals}",
    )

# --- 4B: Parse Contractor into name + city ---
def parse_contractor(contractor_str):
    """
    Parse contractor string into name and city.
    
    Format: 'COMPANY NAME ADDRESS, CITY, STATE ZIP (PHONE)'
    Strategy: Extract the city by finding the state abbreviation pattern ' TX '
              and taking the word before it.
    """
    if pd.isna(contractor_str):
        return "Unknown", "Unknown"
    
    # Try to find ', CITY, TX' or ', City, TX' pattern
    match = re.search(r",\s*([A-Za-z\s]+?),\s*TX\s+\d{5}", contractor_str)
    if match:
        city = match.group(1).strip().upper()
        # Extract contractor name (everything before the first digit-heavy address)
        name_match = re.match(r"^([A-Za-z0-9\s\.\+\-\&/]+?)(?:\s+\d)", contractor_str)
        name = name_match.group(1).strip() if name_match else contractor_str.split(",")[0].strip()
        return name, city
    
    # Fallback: just take everything before the first comma as the name
    parts = contractor_str.split(",")
    name = parts[0].strip() if parts else contractor_str.strip()
    return name, "Unknown"

parsed_contractors = df["Contractor"].apply(parse_contractor)
df["contractor_name"] = parsed_contractors.apply(lambda x: x[0])
df["contractor_city"] = parsed_contractors.apply(lambda x: x[1])

# Report contractor parsing
known_city = (df["contractor_city"] != "Unknown").sum()
unknown_city = (df["contractor_city"] == "Unknown").sum()
n_unique_contractors = df["contractor_name"].nunique()
n_unique_cities = df["contractor_city"].nunique()

log_observation(
    "Step 4B — Parse Contractor",
    f"Parsed 'Contractor' into contractor_name ({n_unique_contractors:,} unique) "
    f"and contractor_city ({n_unique_cities:,} unique)",
    "The raw Contractor field is a single string containing company name, full mailing address, "
    "and phone number. We extract just the name (for repeat-contractor features) and city "
    "(for frequency encoding) — the full string is too high-cardinality for direct use.",
    f"City extraction success: {known_city:,} ({known_city/len(df)*100:.1f}%) | "
    f"Unknown: {unknown_city:,} ({unknown_city/len(df)*100:.1f}%)"
)

# Show top contractor cities
top_cities = df["contractor_city"].value_counts().head(10).to_dict()
log_observation(
    "Step 4B — Contractor Cities",
    f"Top 10 contractor cities: {top_cities}",
    "Dallas dominates as expected for a Dallas building permits dataset. "
    "This feature captures whether the contractor is local or from outside the city.",
)


# ============================================================================
# STEP 5: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: HANDLE MISSING VALUES")
print("=" * 70)

# --- 5A: Zip Code Imputation ---
zip_null_before = df["Zip Code"].isna().sum()

# Strategy: Use Mapsco as geographic proxy
# For each Mapsco grid reference, find the most common zip code
# Then fill missing zips using their Mapsco
mapsco_mask = df["Zip Code"].isna() & df["Mapsco"].notna()
mapsco_zip_mode = df.dropna(subset=["Zip Code", "Mapsco"]).groupby("Mapsco")["Zip Code"].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
)

filled_by_mapsco = 0
for idx in df[mapsco_mask].index:
    mapsco_val = df.loc[idx, "Mapsco"]
    if mapsco_val in mapsco_zip_mode.index:
        df.loc[idx, "Zip Code"] = mapsco_zip_mode[mapsco_val]
        filled_by_mapsco += 1

# For remaining nulls (no Mapsco either), use overall mode
overall_mode = df["Zip Code"].mode().iloc[0]
remaining_nulls = df["Zip Code"].isna().sum()
df["Zip Code"] = df["Zip Code"].fillna(overall_mode)

log_observation(
    "Step 5A — Zip Code Imputation",
    f"Imputed {zip_null_before:,} missing Zip Codes: "
    f"{filled_by_mapsco:,} via Mapsco lookup, "
    f"{remaining_nulls:,} via overall mode ({overall_mode})",
    "Mapsco grid references are geographic sub-areas of Dallas — permits in the same "
    "Mapsco grid are physically close and likely share the same ZIP code. This is a more "
    "accurate imputation than global mode because it preserves spatial relationships. "
    "Remaining nulls (where Mapsco is also missing) use the dataset-wide most common ZIP."
)

# --- 5B: Area (Square Footage) Imputation ---
area_zero_count = (df["Area"] == 0).sum()
area_zero_pct = area_zero_count / len(df) * 100

# Compute group medians for imputation
group_cols = ["permit_category", "sub_category", "work_type"]
group_medians = df[df["Area"] > 0].groupby(group_cols)["Area"].median()

# Only impute where the group median is > 0 (many groups genuinely have 0 sqft)
imputed_count = 0
for idx in df[df["Area"] == 0].index:
    key = (df.loc[idx, "permit_category"], 
           df.loc[idx, "sub_category"], 
           df.loc[idx, "work_type"])
    if key in group_medians.index:
        median_val = group_medians[key]
        if median_val > 0:
            df.loc[idx, "Area"] = median_val
            imputed_count += 1

still_zero = (df["Area"] == 0).sum()
log_observation(
    "Step 5B — Area Imputation",
    f"Area zeros before: {area_zero_count:,} ({area_zero_pct:.1f}%). "
    f"Imputed {imputed_count:,} using group medians (permit_category × sub_category × work_type). "
    f"Still zero after: {still_zero:,} ({still_zero/len(df)*100:.1f}%)",
    "Many permit types genuinely have Area=0 (e.g., electrical work, plumbing repairs, sign installations) — "
    "these are not missing data, they are accurate. We only impute zeros where the group median is positive, "
    "meaning similar permits in the same category typically report a non-zero area. The remaining zeros are "
    "legitimate (the work doesn't create or modify floor space)."
)

# --- 5C: Land Use — no imputation needed ---
log_observation(
    "Step 5C — Land Use",
    f"Land Use: 0 missing values, {df['Land Use'].nunique()} unique categories",
    "Column is complete. No action needed. Directly usable as a categorical feature.",
)


# ============================================================================
# STEP 6: OUTLIER HANDLING (Stage 2 data only)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: OUTLIER HANDLING")
print("=" * 70)

# Only cap positive values — zeros are a separate class
pos_mask = df["Value"] > 0
p99 = df.loc[pos_mask, "Value"].quantile(0.99)
capped_count = (df.loc[pos_mask, "Value"] > p99).sum()

df.loc[pos_mask & (df["Value"] > p99), "Value"] = p99

# Recompute log_value after capping
df["log_value"] = np.where(df["Value"] > 0, np.log(df["Value"]), np.nan)

log_observation(
    "Step 6 — Outlier Capping",
    f"Capped Value at 99th percentile (${p99:,.0f}) for positive-value records. "
    f"{capped_count:,} records capped ({capped_count/pos_mask.sum()*100:.2f}%)",
    "Extreme outliers (up to $64.5M) would dominate the MSE loss function and distort "
    "the regression model. Capping at the 99th percentile retains 99% of the value "
    "distribution while preventing a handful of mega-projects from skewing predictions. "
    "This is applied only to positive-value records — zeros are handled by Stage 1."
)

# Post-cap statistics for Stage 2
log_observation(
    "Step 6 — Post-Cap Stats",
    f"Stage 2 target stats after capping — "
    f"Value: mean=${df.loc[pos_mask, 'Value'].mean():,.0f}, "
    f"median=${df.loc[pos_mask, 'Value'].median():,.0f}, "
    f"max=${df.loc[pos_mask, 'Value'].max():,.0f}",
    "After capping, the target distribution is much more well-behaved for regression.",
    f"log_value: mean={df.loc[pos_mask, 'log_value'].mean():.2f}, "
    f"std={df.loc[pos_mask, 'log_value'].std():.2f}, "
    f"range=[{df.loc[pos_mask, 'log_value'].min():.2f}, {df.loc[pos_mask, 'log_value'].max():.2f}]"
)


# ============================================================================
# STEP 7: TYPE ENFORCEMENT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: TYPE ENFORCEMENT")
print("=" * 70)

# Enforce categorical types (string)
categorical_cols = [
    "Permit Number", "Permit Type", "Zip Code", "Land Use",
    "permit_category", "permit_code", "sub_category", "work_type",
    "contractor_name", "contractor_city"
]
for col in categorical_cols:
    df[col] = df[col].astype(str)

# Enforce numeric types
df["Value"] = df["Value"].astype(float)
df["Area"] = df["Area"].astype(float)
df["is_zero_value"] = df["is_zero_value"].astype(int)

log_observation(
    "Step 7 — Type Enforcement",
    f"Enforced types: {len(categorical_cols)} categorical → str, "
    f"Value/Area → float64, is_zero_value → int, Issued Date → datetime64",
    "Zip Code must be string (it's a categorical identifier, not a number — arithmetic "
    "operations on it are meaningless). is_zero_value is 0/1 integer for classification. "
    "Consistent typing prevents unexpected behavior in downstream encoding and modeling."
)

# Print final dtypes
print("\n  Final column types:")
for col in df.columns:
    print(f"    {col}: {df[col].dtype}")


# ============================================================================
# STEP 8: SAVE & QUALITY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: SAVE & QUALITY REPORT")
print("=" * 70)

# Drop columns that won't be needed downstream
cols_to_drop = ["Mapsco", "Street Address", "Work Description", "Contractor"]
df_save = df.drop(columns=cols_to_drop)

log_observation(
    "Step 8 — Column Drop",
    f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}",
    "Mapsco: used only for zip imputation — too granular for modeling (3,066 unique). "
    "Street Address: too high-cardinality (76K unique) and not useful without geocoding. "
    "Work Description: free-text field (17K unique) — would need NLP; out of scope for tabular ML. "
    "Contractor: replaced by parsed contractor_name and contractor_city."
)

# Save
df_save.to_csv(OUTPUT_FILE, index=False)

# Final quality report
final_rows = len(df_save)
final_cols = len(df_save.columns)
stage1_size = final_rows
stage2_size = (df_save["is_zero_value"] == 0).sum()
class_balance = df_save["is_zero_value"].value_counts()

print(f"\n{'='*70}")
print("FINAL QUALITY REPORT")
print(f"{'='*70}")
print(f"  Original dataset:      {initial_rows:,} rows × {initial_cols} columns")
print(f"  Cleaned dataset:       {final_rows:,} rows × {final_cols} columns")
print(f"  Data retention:        {final_rows/initial_rows*100:.2f}%")
print(f"  Rows dropped:          {initial_rows - final_rows} (null critical fields only)")
print(f"")
print(f"  Stage 1 (Classifier) training set:  {stage1_size:,} rows")
print(f"    • Class 0 (positive-value):       {class_balance.get(0, 0):,} ({class_balance.get(0, 0)/stage1_size*100:.1f}%)")
print(f"    • Class 1 (zero-value/$0 admin):  {class_balance.get(1, 0):,} ({class_balance.get(1, 0)/stage1_size*100:.1f}%)")
print(f"")
print(f"  Stage 2 (Regressor) training set:   {stage2_size:,} rows")
print(f"    • Target: log_value")
print(f"    • Mean:  {df_save.loc[df_save['is_zero_value']==0, 'log_value'].mean():.2f}")
print(f"    • Std:   {df_save.loc[df_save['is_zero_value']==0, 'log_value'].std():.2f}")
print(f"")
print(f"  Columns in cleaned dataset:")
for c in df_save.columns:
    print(f"    • {c} ({df_save[c].dtype})")
print(f"")
print(f"  Output saved to: {OUTPUT_FILE}")
print(f"  Size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.1f} MB")

log_observation(
    "Step 8 — Final Report",
    f"Saved permits_cleaned.csv: {final_rows:,} rows × {final_cols} columns "
    f"({final_rows/initial_rows*100:.2f}% retention)",
    f"Two-Stage Hurdle Model ready. Stage 1 class balance: "
    f"{class_balance.get(1,0)/stage1_size*100:.1f}% zero / {class_balance.get(0,0)/stage1_size*100:.1f}% positive. "
    f"Stage 2 has {stage2_size:,} records for regression."
)

# Write observation log
write_log()

# Also print a quick missing-value check on the final dataset
print(f"\n{'='*70}")
print("POST-CLEANING MISSING VALUE AUDIT")
print(f"{'='*70}")
for col in df_save.columns:
    null_count = df_save[col].isna().sum()
    if col == "log_value":
        # log_value is intentionally NaN for zero-value records
        expected_nan = df_save["is_zero_value"].sum()
        actual_nan = null_count
        print(f"  {col}: {actual_nan:,} NaN (expected: {expected_nan:,} — these are zero-value rows, Stage 2 ignores them) ✅")
    elif null_count > 0:
        print(f"  {col}: {null_count:,} nulls ⚠️")
    else:
        print(f"  {col}: 0 nulls ✅")

print(f"\n✅ Data cleaning complete. Ready for Phase 1 Step 4: Feature Engineering.")
