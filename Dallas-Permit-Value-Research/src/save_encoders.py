"""
Save all encoding artifacts needed by the Flask API.
The API must apply the same transformations used during training.

Updated: Stage 2 regressor uses feature_cols_reg (excludes area_value_ratio)
         while Stage 1 classifier uses all features (feature_cols).
"""
import pandas as pd
import numpy as np
import joblib
import json

print("Saving encoding artifacts for Flask API...")

# Load the cleaned + feature data to extract encoding maps
df_clean = pd.read_csv("permits_cleaned.csv")
df_features = pd.read_csv("permits_features.csv")

# 1. Zip Code target encoding map
pos_mask = df_clean["is_zero_value"] == 0
zip_target_map = df_clean.loc[pos_mask].groupby("Zip Code")["log_value"].mean().to_dict()
zip_overall_mean = df_clean.loc[pos_mask, "log_value"].mean()

# 2. Land Use frequency map
land_use_freq_map = df_clean["Land Use"].value_counts().to_dict()

# 3. Permit code frequency map
permit_code_freq_map = df_clean["permit_code"].value_counts().to_dict()

# 4. Contractor city frequency map
city_freq_map = df_clean["contractor_city"].value_counts().to_dict()

# 5. Contractor permit count map
contractor_count_map = df_clean["contractor_name"].value_counts().to_dict()

# 6. Feature column order — Stage 1 uses ALL features (63)
target_cols = ["Value", "is_zero_value", "log_value"]
feature_cols = [c for c in df_features.columns if c not in target_cols]

# 7. Feature column order — Stage 2 EXCLUDES area_value_ratio (62)
#    This matches the updated modeling pipeline where area_value_ratio
#    is removed from regressor features to avoid data leakage
#    (area_value_ratio = Area / Value, and Value is the prediction target)
feature_cols_reg = [c for c in feature_cols if c != "area_value_ratio"]

# 8. One-hot column names for permit_category, sub_category, work_type
ohe_permit_cats = [c for c in feature_cols if c.startswith("permit_category_")]
ohe_sub_cats = [c for c in feature_cols if c.startswith("sub_category_")]
ohe_work_types = [c for c in feature_cols if c.startswith("work_type_")]

# 9. Median values for imputation at prediction time
median_area = df_clean.loc[pos_mask, "Area"].median()
median_contractor_count = df_clean["contractor_name"].map(contractor_count_map).median()

# Save everything
encoders = {
    "zip_target_map": {str(k): float(v) for k, v in zip_target_map.items()},
    "zip_overall_mean": float(zip_overall_mean),
    "land_use_freq_map": {str(k): int(v) for k, v in land_use_freq_map.items()},
    "permit_code_freq_map": {str(k): int(v) for k, v in permit_code_freq_map.items()},
    "city_freq_map": {str(k): int(v) for k, v in city_freq_map.items()},
    "feature_cols": feature_cols,
    "feature_cols_reg": feature_cols_reg,
    "ohe_permit_cats": ohe_permit_cats,
    "ohe_sub_cats": ohe_sub_cats,
    "ohe_work_types": ohe_work_types,
    "median_area": float(median_area),
    "median_contractor_count": float(median_contractor_count),
}

with open("models/encoders.json", "w") as f:
    json.dump(encoders, f, indent=2)

print(f"  Saved models/encoders.json")
print(f"  Stage 1 feature columns: {len(feature_cols)} (all features)")
print(f"  Stage 2 feature columns: {len(feature_cols_reg)} (excludes area_value_ratio)")
print(f"  One-hot columns: {len(ohe_permit_cats)} permit_category + {len(ohe_sub_cats)} sub_category + {len(ohe_work_types)} work_type")
print(f"  Zip codes mapped: {len(zip_target_map)}")
print(f"  Land use categories mapped: {len(land_use_freq_map)}")
print(f"  Done!")
