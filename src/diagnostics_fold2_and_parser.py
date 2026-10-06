"""
diagnostics_fold2_and_parser.py
===============================
Investigative script for:
1. Root-causing Fold 2 (2019 Q2) Stage 1 recall / AUC / dollar R2 anomaly
2. Measuring permit-type parser fallback rate and extracting candidate regex patterns

Run from project root or Dallas-Permit-Value-Research directory.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# 0. Locate Data
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "permits_research_cleaned.csv"
if not DATA_PATH.exists():
    # Fallback to local if running in outputs dir
    DATA_PATH = Path("permits_research_cleaned.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find permits_research_cleaned.csv at {DATA_PATH}")

DATA_PATH = str(DATA_PATH)

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, parse_dates=["Issued Date"])
df = df.sort_values("Issued Date").reset_index(drop=True)
print(f"Total rows loaded: {len(df):,}")

# Folds definition matching modeling_pipeline_v2.py
FOLD_SPEC = [
    {"name": "Fold 1", "train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-03-31", "label": "2019 Q1"},
    {"name": "Fold 2", "train_end": "2019-03-31", "val_start": "2019-04-01", "val_end": "2019-06-30", "label": "2019 Q2"},
    {"name": "Fold 3", "train_end": "2019-06-30", "val_start": "2019-07-01", "val_end": "2019-09-30", "label": "2019 Q3"},
    {"name": "Fold 4", "train_end": "2019-09-30", "val_start": "2019-10-01", "val_end": "2019-12-31", "label": "2019 Q4"},
]

print("\n" + "=" * 80)
print("PART 1: ROOT-CAUSE ANALYSIS OF FOLD 2 (2019 Q2) ANOMALY")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1.1 Zero-Value Rate Shift Across Folds (Train vs Val)
# -----------------------------------------------------------------------------
print("\n--- 1.1 Zero-Value Rate Shift (Train vs Val) ---")
fold_zero_stats = []

for spec in FOLD_SPEC:
    train_mask = df["Issued Date"] <= spec["train_end"]
    val_mask = (df["Issued Date"] >= spec["val_start"]) & (df["Issued Date"] <= spec["val_end"])
    
    tr_df = df[train_mask]
    val_df = df[val_mask]
    
    tr_zeros = tr_df["is_zero_value"].sum()
    tr_rate = tr_zeros / len(tr_df) if len(tr_df) > 0 else 0
    
    val_zeros = val_df["is_zero_value"].sum()
    val_rate = val_zeros / len(val_df) if len(val_df) > 0 else 0
    
    delta = val_rate - tr_rate
    rel_delta = (val_rate - tr_rate) / tr_rate if tr_rate > 0 else 0
    
    fold_zero_stats.append({
        "Fold": spec["name"],
        "Val Quarter": spec["label"],
        "Train N": len(tr_df),
        "Train Zero %": round(tr_rate * 100, 2),
        "Val N": len(val_df),
        "Val Zero %": round(val_rate * 100, 2),
        "Diff (pp)": round(delta * 100, 2),
        "Rel Change %": round(rel_delta * 100, 1)
    })

stats_df = pd.DataFrame(fold_zero_stats)
print(stats_df.to_string(index=False))

# Also quarterly zero rate across all 2019
df_2019 = df[(df["Issued Date"] >= "2019-01-01") & (df["Issued Date"] <= "2019-12-31")].copy()
df_2019["Quarter"] = df_2019["Issued Date"].dt.to_period("Q")
print("\nQuarterly Breakdown for 2019:")
q_stats = df_2019.groupby("Quarter").agg(
    Total=("is_zero_value", "count"),
    Zeros=("is_zero_value", "sum"),
    Zero_Pct=("is_zero_value", lambda x: round(x.mean() * 100, 2)),
    Median_Positive_Value=("Value", lambda x: round(x[x > 0].median(), 2)),
    Mean_Positive_Value=("Value", lambda x: round(x[x > 0].mean(), 2))
)
print(q_stats)

# -----------------------------------------------------------------------------
# 1.2 Categorical Distribution Across 2019 Quarters
# -----------------------------------------------------------------------------
print("\n--- 1.2 Categorical Distribution Shift Across 2019 Quarters ---")

for col in ["permit_category", "sub_category", "work_type"]:
    print(f"\nDistribution for {col} (Percentages across Q1-Q4):")
    ct = pd.crosstab(df_2019[col], df_2019["Quarter"], normalize="columns") * 100
    ct["Avg_Q1_Q3_Q4"] = ct[[pd.Period('2019Q1', 'Q'), pd.Period('2019Q3', 'Q'), pd.Period('2019Q4', 'Q')]].mean(axis=1)
    ct["Q2_vs_Avg_Ratio"] = (ct[pd.Period('2019Q2', 'Q')] / ct["Avg_Q1_Q3_Q4"]).round(2)
    ct = ct.sort_values(by=pd.Period('2019Q2', 'Q'), ascending=False)
    print(ct.head(10).round(2).to_string())

print("\nDistribution for top 8 Land Use:")
top_lu = df_2019["Land Use"].value_counts().head(8).index
df_2019["Land_Use_Top"] = df_2019["Land Use"].apply(lambda x: x if x in top_lu else "Other")
ct_lu = pd.crosstab(df_2019["Land_Use_Top"], df_2019["Quarter"], normalize="columns") * 100
ct_lu["Avg_Q1_Q3_Q4"] = ct_lu[[pd.Period('2019Q1', 'Q'), pd.Period('2019Q3', 'Q'), pd.Period('2019Q4', 'Q')]].mean(axis=1)
ct_lu["Q2_vs_Avg_Ratio"] = (ct_lu[pd.Period('2019Q2', 'Q')] / ct_lu["Avg_Q1_Q3_Q4"]).round(2)
print(ct_lu.sort_values(by=pd.Period('2019Q2', 'Q'), ascending=False).round(2).to_string())

# -----------------------------------------------------------------------------
# 1.3 Unseen ZIP Code Fallback Coverage
# -----------------------------------------------------------------------------
print("\n--- 1.3 Unseen ZIP Code Fallback Coverage ---")
zip_coverage = []

for spec in FOLD_SPEC:
    tr_df = df[df["Issued Date"] <= spec["train_end"]]
    val_df = df[(df["Issued Date"] >= spec["val_start"]) & (df["Issued Date"] <= spec["val_end"])]
    
    tr_zips = set(tr_df["Zip Code"].dropna().unique())
    
    val_total = len(val_df)
    null_zips = val_df["Zip Code"].isna().sum()
    
    # Unseen = non-null ZIP not present in training (excludes null ZIPs)
    val_nonnull = val_df[val_df["Zip Code"].notna()]
    unseen_rows = val_nonnull[~val_nonnull["Zip Code"].isin(tr_zips)]
    unseen_count = len(unseen_rows)
    unseen_pct = (unseen_count / val_total) * 100 if val_total > 0 else 0
    null_pct = (null_zips / val_total) * 100 if val_total > 0 else 0
    
    zip_coverage.append({
        "Fold": spec["name"],
        "Val Quarter": spec["label"],
        "Train Unique ZIPs": len(tr_zips),
        "Val Total Rows": val_total,
        "Null ZIP Rows": null_zips,
        "Null ZIP %": round(null_pct, 3),
        "Unseen ZIP Rows": unseen_count,
        "Unseen ZIP %": round(unseen_pct, 3)
    })

print(pd.DataFrame(zip_coverage).to_string(index=False))

# -----------------------------------------------------------------------------
# 1.4 Temporal Volume & Ingestion Anomaly Detection
# -----------------------------------------------------------------------------
print("\n--- 1.4 Permit Volume & Ingestion Anomaly Analysis ---")

df_2018_2019 = df[(df["Issued Date"] >= "2018-01-01") & (df["Issued Date"] <= "2019-12-31")].copy()
daily_counts = df_2018_2019.groupby("Issued Date").size()

print(f"Daily count overall (2018-2019): Mean = {daily_counts.mean():.1f}, Median = {daily_counts.median():.1f}, Std = {daily_counts.std():.1f}")

for q in ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]:
    q_mask = (df_2018_2019["Issued Date"].dt.to_period("Q") == q)
    q_daily = df_2018_2019[q_mask].groupby("Issued Date").size()
    print(f"{q}: Days = {len(q_daily)}, Mean/day = {q_daily.mean():.1f}, Min = {q_daily.min()}, Max = {q_daily.max()}, Total = {q_daily.sum()}")

# Check for zero or near-zero days on weekdays
daily_df = df_2018_2019.groupby("Issued Date").agg(
    count=("Permit Number", "count"),
    zeros=("is_zero_value", "sum")
).reset_index()
daily_df["weekday"] = daily_df["Issued Date"].dt.day_name()
weekdays_only = daily_df[~daily_df["weekday"].isin(["Saturday", "Sunday"])]

# Days with abnormally low volume (< 30 on weekday) in 2019 Q2
q2_low = weekdays_only[(weekdays_only["Issued Date"] >= "2019-04-01") & (weekdays_only["Issued Date"] <= "2019-06-30") & (weekdays_only["count"] < 30)]
print(f"\n2019 Q2 Weekdays with unusually low volume (< 30): {len(q2_low)}")
if len(q2_low) > 0:
    print(q2_low.to_string(index=False))

# Days with abnormally high volume (> 350) in 2019 Q2
q2_high = weekdays_only[(weekdays_only["Issued Date"] >= "2019-04-01") & (weekdays_only["Issued Date"] <= "2019-06-30") & (weekdays_only["count"] > 350)]
print(f"\n2019 Q2 Weekdays with unusually high volume (> 350): {len(q2_high)}")
if len(q2_high) > 0:
    print(q2_high.to_string(index=False))

# -----------------------------------------------------------------------------
# 1.5 Per-Fold Positive-Value Variance (Dollar R2 Denominator Compression)
# -----------------------------------------------------------------------------
print("\n--- 1.5 Per-Fold Positive-Value Variance (Dollar R2 Denominator) ---")
variance_stats = []
for spec in FOLD_SPEC:
    val_df = df[(df["Issued Date"] >= spec["val_start"]) & (df["Issued Date"] <= spec["val_end"])]
    val_values = val_df["Value"]
    val_pos = val_values[val_values > 0]
    variance_stats.append({
        "Fold": spec["name"],
        "Val Quarter": spec["label"],
        "Val N": len(val_df),
        "Pos N": len(val_pos),
        "Var(y) All": f"{np.var(val_values):.3e}",
        "Var(y) Pos": f"{np.var(val_pos):.3e}",
        "Max Value": f"${val_pos.max():,.0f}",
        "Mean Pos Value": f"${val_pos.mean():,.0f}"
    })
print(pd.DataFrame(variance_stats).to_string(index=False))

print("\n" + "=" * 80)
print("PART 2: PERMIT-TYPE PARSER FALLBACK RATE & PATTERN AUDIT")
print("=" * 80)

# -----------------------------------------------------------------------------
# 2.1 Parse Permit Type Pattern Breakdown
# -----------------------------------------------------------------------------
p1_regex = re.compile(r"^(.+?)\s*\((\w{2})\)\s+(.+?)\s{2,}(.+)$")
p2_regex = re.compile(r"^(.+?)\s{2,}(.+)$")

def tag_pattern(pt):
    """Match production parse_permit_type exactly (no pre-strip)."""
    if pd.isna(pt):
        return "Null"
    pt_str = str(pt)
    if p1_regex.match(pt_str):
        return "Pattern 1 (Full 4-Part)"
    if p2_regex.match(pt_str):
        return "Pattern 2 (2-Part Double-Space)"
    return "Pattern 3 (Fallback)"

df["parser_pattern"] = df["Permit Type"].apply(tag_pattern)
pattern_counts = df["parser_pattern"].value_counts()
pattern_pcts = df["parser_pattern"].value_counts(normalize=True) * 100

summary_parser = pd.DataFrame({
    "Pattern": pattern_counts.index,
    "Count": pattern_counts.values,
    "Percentage": pattern_pcts.values.round(2)
})
print("\nOverall Permit Type Parser Pattern Breakdown:")
print(summary_parser.to_string(index=False))

# Breakdown by year
df["Year"] = df["Issued Date"].dt.year
ct_pat_year = pd.crosstab(df["parser_pattern"], df["Year"], normalize="columns") * 100
print("\nParser Pattern % by Year:")
print(ct_pat_year.round(2).to_string())

# -----------------------------------------------------------------------------
# 2.2 Sample Fallback Strings
# -----------------------------------------------------------------------------
fallback_records = df[df["parser_pattern"] == "Pattern 3 (Fallback)"]
fallback_strings = fallback_records["Permit Type"].dropna().value_counts()

print(f"\nTotal distinct Permit Type strings hitting Fallback: {len(fallback_strings)}")
print("\nTop 30 Most Frequent Fallback Strings (with occurrences):")
top_30_fallback = fallback_strings.head(30)
for idx, (raw_str, count) in enumerate(top_30_fallback.items(), 1):
    print(f"  {idx:2d}. [{count:5d}x] \"{raw_str}\"")

# Check relationship between parser pattern and work_type / sub_category
print("\nTop 10 work_type values for Pattern 3 Fallback:")
print(fallback_records["work_type"].value_counts().head(10))

print("\nTop 10 sub_category values for Pattern 3 Fallback:")
print(fallback_records["sub_category"].value_counts().head(10))

# Also check for Pattern 2 top work_types
p2_records = df[df["parser_pattern"] == "Pattern 2 (2-Part Double-Space)"]
print("\nTop 10 work_type values for Pattern 2:")
print(p2_records["work_type"].value_counts().head(10))

# -----------------------------------------------------------------------------
# 2.3 Check why sub_category_General and work_type_Other are top features
# -----------------------------------------------------------------------------
print("\nDistribution of work_type across all data:")
print(df["work_type"].value_counts().head(10))

print("\nDistribution of sub_category across all data:")
print(df["sub_category"].value_counts().head(10))

# Check zero-value rate by work_type
print("\nZero-value rate by work_type (top 8 work types):")
top_wt = df["work_type"].value_counts().head(8).index
print(df[df["work_type"].isin(top_wt)].groupby("work_type")["is_zero_value"].agg(
    Total="count",
    Zero_Rate=lambda x: f"{x.mean()*100:.1f}%",
    Median_Value=lambda x: f"${df.loc[x.index, 'Value'][df.loc[x.index, 'Value'] > 0].median():,.0f}"
))

print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
