"""
Dallas Building Permits — Research Paper: Minimal Data Cleaning
================================================================
Project: ASDS 6302 Final Project — Predict Building Permit Value
Architecture: Two-Stage Hurdle Model (Redesigned, Leakage-Free)

This script performs ONLY type-casting and field parsing.
It deliberately does NOT impute missing values, does NOT cap outliers,
and does NOT compute any aggregate statistics — those steps are
deferred to the per-fold modeling pipeline to prevent data leakage.

Output: permits_research_cleaned.csv
  - Retains Mapsco column (needed for per-fold ZIP imputation)
  - Retains raw Area zeros (imputed per-fold)
  - Retains uncapped Value (capping is a per-fold sensitivity choice)
"""

import pandas as pd
import numpy as np
import re
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = str(ROOT / "data" / "Building-Permits.csv")
OUTPUT_FILE = str(ROOT / "outputs" / "permits_research_cleaned.csv")


# ============================================================================
# COMPOSITE FIELD PARSERS (REUSABLE ACROSS PIPELINE & API)
# ============================================================================

def parse_permit_type(pt):
    """
    Parse composite permit type string into 4 components:
    (permit_category, permit_code, sub_category, work_type)
    
    Standard pattern: 'Building (BU) Multi Family  New Construction'
                       ────────  ──   ────────────  ────────────────
                       category  code sub_category  work_type
    """
    pt_str = str(pt).strip()

    # Pattern 1A: Standard format with known sub_category and work_type (single or multi-space)
    match_std = re.match(
        r"^(.+?)\s*\((\w{2})\)\s+(Single Family|Commercial|Multi Family)\s+"
        r"(New Construction|Alteration|Renovation|Addition|Reconstruction|Finish Out)$",
        pt_str
    )
    if match_std:
        return (match_std.group(1).strip(), match_std.group(2).strip(),
                match_std.group(3).strip(), match_std.group(4).strip())

    # Pattern 1B: Standard format with double-space separator
    match = re.match(
        r"^(.+?)\s*\((\w{2})\)\s+(.+?)\s{2,}(.+)$", pt_str
    )
    if match:
        return (match.group(1).strip(), match.group(2).strip(),
                match.group(3).strip(), match.group(4).strip())

    # Pattern 2 (NEW): Sign permits — "[Category] [WorkType]"
    #   e.g. "Electrical Sign New Construction", "Sign Alteration",
    #        "Supergraphics Alteration", "Special Purpose Sign New Construction"
    sign_match = re.match(
        r"^((?:Electrical\s+)?(?:Special\s+Purpose\s+)?Sign|Supergraphics)"
        r"\s+(New\s+Construction|Alteration|Renovation|Reconstruction)$",
        pt_str
    )
    if sign_match:
        return (sign_match.group(1).strip(), "NA",
                "Commercial", sign_match.group(2).strip())

    # Pattern 3 (NEW): Demolition permits — "Demolition Permit [SubCategory]"
    #   e.g. "Demolition Permit SFD/Duplex", "Demolition Permit Commercial"
    demo_match = re.match(
        r"^Demolition\s+Permit\s+(.+)$", pt_str
    )
    if demo_match:
        raw_sub = demo_match.group(1).strip()
        sub = "Single Family" if "SFD" in raw_sub or "Duplex" in raw_sub else raw_sub
        return ("Demolition", "NA", sub, "Demolition")

    # Pattern 4: No code, has double-space separator
    match2 = re.match(r"^(.+?)\s{2,}(.+)$", pt_str)
    if match2:
        return (match2.group(1).strip(), "NA", "General", match2.group(2).strip())

    # Pattern 5: Completely non-standard (true last-resort fallback)
    return (pt_str, "NA", "General", "Other")


def parse_contractor(contractor_str):
    """Parse contractor string into name and city."""
    if pd.isna(contractor_str):
        return "Unknown", "Unknown"

    c_str = str(contractor_str).strip()
    match = re.search(r",\s*([A-Za-z\s]+?),\s*TX\s+\d{5}", c_str)
    if match:
        city = match.group(1).strip().upper()
        name_match = re.match(r"^([A-Za-z0-9\s\.+\-\&/]+?)(?:\s+\d)", c_str)
        name = name_match.group(1).strip() if name_match else c_str.split(",")[0].strip()
        return name, city

    parts = c_str.split(",")
    name = parts[0].strip() if parts else c_str
    return name, "Unknown"


# ============================================================================
# MAIN DATA CLEANING PIPELINE
# ============================================================================

def clean_data(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    print("=" * 70)
    print("RESEARCH PIPELINE — MINIMAL DATA CLEANING")
    print("=" * 70)

    # Step 1: Load and type-cast
    print("\n[Step 1] Loading and type-casting...")
    df = pd.read_csv(input_file, dtype=str)
    initial_rows = len(df)
    print(f"  Loaded {input_file}: {initial_rows:,} rows x {len(df.columns)} columns")

    df["Value"] = df["Value"].str.replace(",", "", regex=False).astype(float)
    print(f"  Value range: ${df['Value'].min():,.0f} — ${df['Value'].max():,.0f}")

    df["Area"] = df["Area"].str.replace(",", "", regex=False).astype(float)
    df["Issued Date"] = pd.to_datetime(df["Issued Date"], format="%m/%d/%y")
    print(f"  Date range: {df['Issued Date'].min().date()} — {df['Issued Date'].max().date()}")

    # Step 2: Remove rows with null critical fields
    print("\n[Step 2] Removing rows with null critical fields...")
    rows_before = len(df)
    critical_mask = (
        df["Value"].notna() &
        df["Issued Date"].notna() &
        df["Permit Type"].notna() &
        df["Contractor"].notna()
    )
    df = df[critical_mask].copy()
    rows_dropped = rows_before - len(df)
    print(f"  Dropped {rows_dropped} rows with null critical fields")
    print(f"  Remaining: {len(df):,} rows")

    # Step 3: Create hurdle targets
    print("\n[Step 3] Creating hurdle model targets (NO outlier capping)...")
    df["is_zero_value"] = (df["Value"] == 0).astype(int)
    zero_count = df["is_zero_value"].sum()
    positive_count = len(df) - zero_count
    print(f"  is_zero_value: {zero_count:,} zeros ({zero_count/len(df)*100:.1f}%) | "
          f"{positive_count:,} positive ({positive_count/len(df)*100:.1f}%)")

    df["log_value"] = np.where(df["Value"] > 0, np.log(df["Value"]), np.nan)
    pos_mask = df["Value"] > 0
    print(f"  log_value range: {df.loc[pos_mask, 'log_value'].min():.2f} — "
          f"{df.loc[pos_mask, 'log_value'].max():.2f}")

    # Step 4: Parse composite fields
    print("\n[Step 4] Parsing composite fields...")
    parsed = df["Permit Type"].apply(parse_permit_type)
    df["permit_category"] = parsed.apply(lambda x: x[0])
    df["permit_code"]     = parsed.apply(lambda x: x[1])
    df["sub_category"]    = parsed.apply(lambda x: x[2])
    df["work_type"]       = parsed.apply(lambda x: x[3])

    for col in ["permit_category", "permit_code", "sub_category", "work_type"]:
        print(f"  {col}: {df[col].nunique()} unique values")

    parsed_contractors = df["Contractor"].apply(parse_contractor)
    df["contractor_name"] = parsed_contractors.apply(lambda x: x[0])
    df["contractor_city"] = parsed_contractors.apply(lambda x: x[1])

    known_city = (df["contractor_city"] != "Unknown").sum()
    print(f"  contractor_name: {df['contractor_name'].nunique():,} unique")
    print(f"  contractor_city: {df['contractor_city'].nunique():,} unique "
          f"({known_city:,} resolved, {len(df)-known_city:,} 'Unknown')")

    # Step 5: Type enforcement
    print("\n[Step 5] Enforcing types...")
    categorical_cols = [
        "Permit Number", "Permit Type", "Zip Code", "Land Use", "Mapsco",
        "permit_category", "permit_code", "sub_category", "work_type",
        "contractor_name", "contractor_city"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace("nan", np.nan)

    df["Value"] = df["Value"].astype(float)
    df["Area"] = df["Area"].astype(float)
    df["is_zero_value"] = df["is_zero_value"].astype(int)

    # Step 6: Select columns & save
    print("\n[Step 6] Selecting columns and saving...")
    cols_to_drop = ["Street Address", "Work Description", "Contractor"]
    df_save = df.drop(columns=cols_to_drop)

    df_save = df_save.sort_values("Issued Date").reset_index(drop=True)
    df_save.to_csv(output_file, index=False)

    file_size = os.path.getsize(output_file) / (1024 * 1024)

    print(f"\n{'='*70}")
    print("MINIMAL CLEANING COMPLETE")
    print(f"{'='*70}")
    print(f"  Output: {output_file}")
    print(f"  Shape:  {df_save.shape[0]:,} rows x {df_save.shape[1]} columns")
    print(f"  Size:   {file_size:.1f} MB")
    print(f"  Date range: {df_save['Issued Date'].min().date()} — {df_save['Issued Date'].max().date()}")
    print()


if __name__ == "__main__":
    clean_data()
