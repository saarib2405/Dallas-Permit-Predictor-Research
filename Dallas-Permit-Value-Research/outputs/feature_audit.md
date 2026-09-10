# Feature Leakage Audit — Dallas Building Permits

**Generated:** 2026-09-10  
**Scope:** Audit five engineered features for target leakage and future-data leakage.  
**Source scripts:** `src/feature_engineering.py`, `src/data_cleaning.py`  
**Constraint:** READ-ONLY audit — no code modifications made.

---

## Audit Methodology

Each feature was traced through the source code to determine:
1. **How it is calculated** — the exact lines of code and data used.
2. **Target Leakage Risk** — whether the feature uses the target variable (`Value`, `log_value`, or `is_zero_value`) in its computation, giving the model direct or indirect access to the answer it is trying to predict.
3. **Future Data Risk** — whether the feature uses information from validation/test-period records during training, violating the chronological split (train: 2018–mid-2019, val: mid-2019–early 2020, test: early–Aug 2020).

### Risk Levels
- 🔴 **HIGH** — Direct leakage that will inflate metrics and invalidate results.
- 🟡 **MEDIUM** — Indirect or partial leakage that may subtly inflate performance.
- 🟢 **LOW** — No leakage detected for this risk category.

---

## Feature Audit Table

| Feature | Calculation Method | Target Leakage Risk | Future Data Risk |
|---------|-------------------|---------------------|------------------|
| `area_value_ratio` | Computed as `Area / Value` for rows where both `Area > 0` and `Value > 0`; set to `0` otherwise. *(feature_engineering.py, lines 167–171)* | 🔴 **HIGH** — Directly divides by the target variable `Value`. This gives the model a near-perfect mathematical relationship to the prediction target. The README acknowledges this and excludes it from Stage 2 features, but it remains in Stage 1. | 🟢 **LOW** — Uses only same-row `Area` and `Value`; no cross-row or cross-time computation. |
| `zip_target_encoded` | Each `Zip Code` is replaced with the mean `log_value` of positive-value permits sharing that ZIP. Computed on the **full dataset** before the train/val/test split. *(feature_engineering.py, lines 139–142)* | 🔴 **HIGH** — Uses the target variable `log_value` directly in the encoding formula. The code author notes at line 149–150: *"In production, this must be computed on training data only to prevent data leakage. We will re-encode after the train/val/test split."* However, this re-encoding is **never implemented** in the pipeline — the pre-split encoding persists into the modeling stage. | 🟡 **MEDIUM** — Because the encoding is computed pre-split on all rows, the mean `log_value` for each ZIP includes future (val/test) observations. Training rows see ZIP averages that incorporate future target values. |
| `contractor_permit_count` | Total number of permits filed by each `contractor_name`, computed via `value_counts()` across the **entire dataset** before the train/val/test split. *(feature_engineering.py, lines 78–79)* | 🟢 **LOW** — Does not reference `Value`, `log_value`, or `is_zero_value` in its computation. Purely a frequency count of contractor names. | 🟡 **MEDIUM** — The count includes permits from the validation and test periods. For a contractor with 50 lifetime permits, the training-period rows see the full count of 50 rather than the ~35 permits that would have existed at the time of the chronological training cutoff. This inflates contractor activity signals for training observations. |
| `contractor_city_freq` | Total number of permits originating from each `contractor_city`, computed via `value_counts()` across the **entire dataset** before the train/val/test split. *(feature_engineering.py, lines 93–94)* | 🟢 **LOW** — Does not reference any target variable. Purely a frequency count of contractor cities. | 🟡 **MEDIUM** — Same issue as `contractor_permit_count`: frequency counts include future-period rows. A city's permit frequency at training time is inflated by validation/test permits that had not yet been issued. |
| `permit_code_freq` | Total number of permits for each `permit_code`, computed via `value_counts()` across the **entire dataset** before the train/val/test split. *(feature_engineering.py, lines 153–154)* | 🟢 **LOW** — Does not reference any target variable. Purely a frequency count of permit codes. | 🟡 **MEDIUM** — Same pattern: frequency is computed pre-split, so permit codes' counts include future observations. Training rows see inflated frequency values. |

---

## Summary of Findings

### Target Leakage

| Severity | Features | Issue |
|----------|----------|-------|
| 🔴 HIGH | `area_value_ratio` | Divides directly by the target variable `Value`. |
| 🔴 HIGH | `zip_target_encoded` | Mean of `log_value` (the target) per ZIP, computed on the full dataset. The planned fix (re-encode after split) was never implemented. |
| 🟢 LOW | `contractor_permit_count`, `contractor_city_freq`, `permit_code_freq` | No target variable used in computation. |

### Future Data Leakage

| Severity | Features | Issue |
|----------|----------|-------|
| 🟡 MEDIUM | `zip_target_encoded`, `contractor_permit_count`, `contractor_city_freq`, `permit_code_freq` | All four features compute aggregate statistics (means or counts) on the **full dataset** before the chronological train/val/test split. Training rows see statistics that include future observations. |
| 🟢 LOW | `area_value_ratio` | Uses only same-row data; no cross-row aggregation. |

### Recommendations (for future phases — no code changes in Week 1)

1. **`area_value_ratio`** — Must be excluded from any model stage where `Value` is the target. Currently excluded from Stage 2 (confirmed in README), but its inclusion in Stage 1 should also be reviewed since `is_zero_value` is derived from `Value`.
2. **`zip_target_encoded`** — Must be recomputed after the train/val/test split using only training data. Consider adding regularization (smoothing with the global mean) to handle low-count ZIPs.
3. **`contractor_permit_count`**, **`contractor_city_freq`**, **`permit_code_freq`** — Must be recomputed after the split using only training-period data. The `value_counts()` calls should operate on `df_train` only, with unseen categories in val/test mapped to a default value (e.g., 1).
