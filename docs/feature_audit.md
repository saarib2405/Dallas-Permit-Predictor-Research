# Feature-Audit & Data-Leakage Report — Dallas Building Permits Pipeline

**Document Status:** Complete Feature Audit  
**Date:** September 14, 2026  
**Scope:** Rigorous investigation of all engineered features, data transformations, target dependencies, and train/val/test contamination across the Dallas Permit Predictor codebase.  
**Source Code Examined:**
- `src/data_cleaning.py` (571 lines)
- `src/feature_engineering.py` (276 lines)
- `src/modeling_pipeline.py` (694 lines)
- `src/save_encoders.py` (81 lines)
- `app.py` (694 lines)

---

## 1. Executive Summary & Audit Context

### 1.1 Dataset & Experimental Setup
- **Raw Dataset:** Dallas Building Permits (`Building-Permits.csv`), 126,840 records spanning **January 2, 2018 through August 29, 2020**.
- **Post-Cleaning Dataset:** `permits_cleaned.csv` (126,837 rows; 3 dropped due to null `Contractor`).
- **Two-Stage Hurdle Architecture:**
  - **Stage 1 (Binary Classification):** Distinguishes administrative/zero-value permits ($0; 30,054 rows / 23.7%) from positive-value construction permits (> $0; 96,783 rows / 76.3%).
  - **Stage 2 (Regression):** Predicts continuous declared permit value $\ln(\text{Value})$ on positive-value records only.
- **Chronological Split:**
  - **Train Period (70%):** 2018-01-02 to 2019-10-17 (88,785 records)
  - **Validation Period (15%):** 2019-10-17 to 2020-03-04 (19,026 records)
  - **Test Period (15%):** 2020-03-04 to 2020-08-29 (19,026 records) — **2020 COVID test-period records**

### 1.2 Leakage Taxonomies Evaluated
1. **Direct Target Leakage:** A feature mathematically contains or directly derives from the prediction target (`Value`, `log_value`, or `is_zero_value`).
2. **Temporal Lookahead Leakage:** A feature uses information, frequencies, or statistics generated after the record's issuance date, specifically incorporating validation and **2020 test-period** records into training-time features.
3. **Train-Test Contamination via Global Aggregation:** Statistics (means, modes, medians, value counts, percentiles) computed across the entire 126,837 rows before the chronological split, allowing test distributions to contaminate training representations.

---

## 2. Core Feature-Audit Table (Requested Primary Features)

Below is the structured audit table for the 5 specific features requested, formatted in accordance with research audit standards:

| Feature | Script & Location | How It Is Calculated (Logic / Formula) | Input Columns / Data | Uses Target Variable? | Uses Full Dataset / 2020 Test Records? | Leakage Risk & Impact Assessment |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **`area_value_ratio`** | `src/feature_engineering.py`<br>Lines 167–171<br>*(Also in `modeling_pipeline.py` L133, `save_encoders.py` L40–44, `app.py` L173–180)* | `np.where((Value > 0) & (Area > 0), Area / Value, 0)`<br>Permit square footage divided by declared permit dollar value. | `Area`<br>`Value` | **YES**<br>*(Direct mathematical dependency on `Value`)* | **Not Cross-Row**<br>*(Row-level computation, but relies on current row's target)* | 🔴 **CRITICAL DIRECT TARGET LEAKAGE**<br>• **Stage 2:** Excluded from regressor (`feature_cols_reg`) because dividing by target is trivial cheating.<br>• **Stage 1 (FATAL DEFECT):** **Retained in classifier** (`X_train = df_train[feature_cols]`). Because `area_value_ratio > 0` iff `Value > 0`, the classifier is handed the answer to `is_zero_value`.<br>• **Deployment Failure:** At inference time in `app.py`, `Value` is unknown, forcing a hardcoded dummy monkey-patch (`0.03`). |
| **`zip_target_encoded`** | `src/feature_engineering.py`<br>Lines 139–142<br>*(Referenced in `save_encoders.py` L20–23, `app.py` L156–158)* | `pos_mask = is_zero_value == 0`<br>`zip_target_mean = df[pos_mask].groupby('Zip Code')['log_value'].mean()`<br>`df['Zip Code'].map(zip_target_mean).fillna(overall_mean)` | `Zip Code`<br>`log_value`<br>`is_zero_value` | **YES**<br>*(Directly computes mean of target `log_value`)* | **YES**<br>*(Calculated on all 126,837 rows across 2018–2020 prior to split)* | 🔴 **HIGH TARGET & TEMPORAL LEAKAGE**<br>• **In-Sample Target Leakage:** Computed globally without k-fold out-of-sample splits or smoothing; each training row's target directly inflates its own feature.<br>• **Future Temporal Leakage:** Training records from 2018 see neighborhood averages that incorporate 2020 COVID-era permit values.<br>• **Broken Promise:** Code lines 149–150 state: *"In production, this must be computed on training data only... We will re-encode after the train/val/test split."* **This re-encoding was never implemented** in `modeling_pipeline.py`. |
| **`contractor_permit_count`** | `src/feature_engineering.py`<br>Lines 78–79<br>*(Referenced in `save_encoders.py` L34, `app.py` L148–150)* | `counts = df['contractor_name'].value_counts().to_dict()`<br>`df['contractor_name'].map(counts).fillna(1)` | `contractor_name`<br>*(parsed from raw `Contractor` in `data_cleaning.py`)* | **NO**<br>*(Purely frequency of contractor)* | **YES**<br>*(Calculated on all 126,837 rows across 2018–2020 prior to split)* | 🟡 **MEDIUM TEMPORAL LOOKAHEAD LEAKAGE**<br>• **Lookahead Bias:** Includes permits issued in late 2019 and the 2020 test period. A contractor filing 3 permits in 2018 and 40 in 2020 receives a count of 43 in 2018.<br>• At historical prediction time, a real system would only know prior permit counts, not future volume.<br>• Directly propagates to `is_repeat_contractor`. |
| **`contractor_city_freq`** | `src/feature_engineering.py`<br>Lines 93–94<br>*(Referenced in `save_encoders.py` L31, `app.py` L152–154)* | `city_freq = df['contractor_city'].value_counts().to_dict()`<br>`df['contractor_city'].map(city_freq).fillna(1)` | `contractor_city`<br>*(parsed from raw `Contractor` in `data_cleaning.py`)* | **NO**<br>*(Frequency count of contractor origin city)* | **YES**<br>*(Calculated on all 126,837 rows across 2018–2020 prior to split)* | 🟡 **MEDIUM TEMPORAL LOOKAHEAD LEAKAGE**<br>• City frequencies aggregate future 2020 activity into 2018–2019 training rows.<br>• Leaks geographic market evolution and out-of-town contractor influx into historical observations. |
| **`permit_code_freq`** | `src/feature_engineering.py`<br>Lines 153–154<br>*(Referenced in `save_encoders.py` L27–29, `app.py` L170–172)* | `code_freq = df['permit_code'].value_counts().to_dict()`<br>`df['permit_code'].map(code_freq).fillna(1)` | `permit_code`<br>*(parsed from raw `Permit Type` in `data_cleaning.py`)* | **NO**<br>*(Frequency count of 2-letter permit code, e.g., 'BU', 'EL')* | **YES**<br>*(Calculated on all 126,837 rows across 2018–2020 prior to split)* | 🟡 **MEDIUM TEMPORAL LOOKAHEAD LEAKAGE**<br>• Global frequency aggregates future permit distributions (including 2020 commercial freeze/shifts) into training set.<br>• Ranked as the #3 most important feature in SHAP explanations, meaning the regressor heavily relies on an aggregated feature that saw test-period counts. |

---

## 3. Audit of All Remaining Features in the Pipeline

The feature matrix (`permits_features.csv`) contains **63 features** (plus 3 target columns). Below is the comprehensive audit of every remaining feature:

| Feature Name / Group | Script & Location | How Calculated / Logic | Inputs Used | Uses Target? | Uses Full Dataset (2020 Test Records)? | Leakage Risk & Assessment |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **`is_repeat_contractor`** | `feature_engineering.py`<br>Lines 81–82 | `(contractor_permit_count >= 10).astype(int)` | `contractor_permit_count` | **NO** | **YES** | 🟡 **MEDIUM:** Inherits lookahead leakage from `contractor_permit_count`. Contractors who only became "repeat" during 2020 are treated as repeat contractors in 2018. |
| **`land_use_freq`** | `feature_engineering.py`<br>Lines 130–131 | `df['Land Use'].value_counts().to_dict()` mapped to `df['Land Use']` | `Land Use` | **NO** | **YES** | 🟡 **MEDIUM:** Global frequency encoding incorporates future land-use filing trends from 2020 into 2018–2019 records. |
| **`log_area`** | `feature_engineering.py`<br>Lines 179–181 | `np.where(Area > 0, np.log1p(Area), 0)` | `Area` | **NO** | **NO** (Indirectly through imputation) | 🟢 **LOW:** Row-wise monotonic transform. (Inherits minor contamination if `Area` was imputed). |
| **`Area`** (Raw / Imputed) | `data_cleaning.py`<br>Lines 371–401 | Where `Area == 0`, filled with median of positive areas for matching `(permit_category, sub_category, work_type)`. | `Area`, `permit_category`, `sub_category`, `work_type` | **NO** | **YES** | 🟡 **MEDIUM:** Group medians computed across all 2018–2020 records. 2020 project sizes influenced 2018 imputed values. |
| **`issue_year`** | `feature_engineering.py`<br>Line 45 | `df['Issued Date'].dt.year` | `Issued Date` | **NO** | **NO** | 🟢 **LOW (No Leakage):** Strict per-row date extraction. *(Note: Tree models cannot extrapolate beyond 2019 during test evaluation).* |
| **`issue_month`** | `feature_engineering.py`<br>Line 46 | `df['Issued Date'].dt.month` | `Issued Date` | **NO** | **NO** | 🟢 **LOW (No Leakage):** Strict per-row date extraction (1–12). |
| **`issue_quarter`** | `feature_engineering.py`<br>Line 47 | `df['Issued Date'].dt.quarter` | `Issued Date` | **NO** | **NO** | 🟢 **LOW (No Leakage):** Strict per-row date extraction (1–4). |
| **`days_since_epoch`** | `feature_engineering.py`<br>Lines 50–51 | `(Issued Date - pd.Timestamp('2000-01-01')).dt.days` | `Issued Date` | **NO** | **NO** | 🟢 **LOW (No Leakage):** Linear days elapsed. No future data used. |
| **`issue_day_of_week`** | `feature_engineering.py`<br>Line 54 | `df['Issued Date'].dt.dayofweek` | `Issued Date` | **NO** | **NO** | 🟢 **LOW (No Leakage):** Strict per-row date extraction (0=Mon, 6=Sun). |
| **`is_q4`** | `feature_engineering.py`<br>Line 57 | `(issue_quarter == 4).astype(int)` | `issue_quarter` | **NO** | **NO** | 🟢 **LOW (No Leakage):** Binary indicator for Q4 filing rush. |
| **`permit_category_*`**<br>*(37 one-hot columns)* | `data_cleaning.py` L206–250<br>`feature_engineering.py` L112–122 | Regex parsed from `Permit Type`, then `pd.get_dummies()` | `Permit Type` | **NO** | **YES (Global dummy creation)** | 🟢 **LOW-NEGLIGIBLE:** Full-dataset dummy columns ensure category schema alignment, but categories rare in training could appear as zero-variance columns. |
| **`sub_category_*`**<br>*(4 one-hot columns)* | `data_cleaning.py` L206–250<br>`feature_engineering.py` L112–122 | Regex parsed from `Permit Type`, then `pd.get_dummies()` | `Permit Type` | **NO** | **YES** | 🟢 **LOW-NEGLIGIBLE:** Fixed set ('Single Family', 'Multi Family', 'Commercial', 'General'). |
| **`work_type_*`**<br>*(7 one-hot columns)* | `data_cleaning.py` L206–250<br>`feature_engineering.py` L112–122 | Regex parsed from `Permit Type`, then `pd.get_dummies()` | `Permit Type` | **NO** | **YES** | 🟢 **LOW-NEGLIGIBLE:** Fixed set ('New Construction', 'Alteration', 'Renovation', 'Addition', 'Reconstruction', 'Finish Out', 'Other'). |

---

## 4. Root-Level Preprocessing & Data Cleaning Audits

In addition to explicit feature engineering, critical data transformations in `src/data_cleaning.py` introduce subtle dataset-level leakage prior to modeling:

### 4.1 Missing Zip Code Imputation (`data_cleaning.py`, Lines 343–359)
- **Method:** For rows missing `Zip Code` (5,397 rows, 4.3%), the pipeline groups by `Mapsco` (Dallas geographic grid reference) and computes the mode `Zip Code`. Remaining nulls receive the dataset-wide mode (`75229`).
- **Contamination:** The `Mapsco`-to-`Zip Code` mode table and dataset-wide mode are computed across all 126,837 rows simultaneously. Test-period records contribute to resolving missing training locations.
- **Severity:** 🟡 **MEDIUM**.

### 4.2 Outlier Target Capping (`data_cleaning.py`, Lines 419–427)
- **Method:** `p99 = df.loc[pos_mask, "Value"].quantile(0.99)`. Capped all positive values above `p99` ($1.2M), then derived `log_value`.
- **Contamination:** The 99th percentile was calculated on the entire 2018–2020 dataset rather than strictly on `df_train`. Information about the upper tail of the 2020 test period influenced the training target values.
- **Severity:** 🟡 **MEDIUM TARGET CONTAMINATION**.

### 4.3 Target Derivations (`data_cleaning.py`, Lines 166–196)
- **`is_zero_value`:** `(df["Value"] == 0).astype(int)` (Stage 1 Target).
- **`log_value`:** `np.where(df["Value"] > 0, np.log(df["Value"]), np.nan)` (Stage 2 Target).
- **Status:** Legitimate target transformations, but must be insulated from feature matrices.

---

## 5. Architectural & Pipeline Flaws Discovered

### Flaw 1: The Stage 1 Classifier "Silver Platter" Leakage
In `src/modeling_pipeline.py`:
- Stage 2 Regressor explicitly removed `area_value_ratio`:
  ```python
  feature_cols_reg = [c for c in feature_cols if c != 'area_value_ratio']
  X_train_reg = df_train.loc[train_pos, feature_cols_reg].values
  ```
- **Stage 1 Classifier left `area_value_ratio` in the feature vector:**
  ```python
  X_train = df_train[feature_cols].values # feature_cols contains area_value_ratio!
  y_train_cls = df_train["is_zero_value"].values
  ```
Because `area_value_ratio = Area / Value`, it is mathematically $0.0$ when $\text{Value} = 0$, and $> 0.0$ when $\text{Value} > 0$ and $\text{Area} > 0$. The classifier was effectively trained with the ground truth label embedded in its input features, artificially inflating Stage 1 accuracy and ROC-AUC.

### Flaw 2: The Production API "Placeholder Hack"
Because `area_value_ratio` requires knowing `Value` before predicting `Value`, the deployed Flask API (`app.py`, lines 173–180) could not compute the feature for new incoming permits. The author was forced to insert an arbitrary constant:
```python
if area > 0:
    area_value_ratio = 0.03  # Approximate median from training data
else:
    area_value_ratio = 0.0
```
This demonstrates the classic signature of data leakage: **a model that requires features at inference time that cannot exist without knowing the target.**

### Flaw 3: The Broken Promise of Post-Split Re-Encoding
In `src/feature_engineering.py`, lines 149–150:
> *"NOTE: In production, this must be computed on training data only to prevent data leakage. We will re-encode after the train/val/test split."*

However, an inspection of `src/modeling_pipeline.py` reveals that `df = pd.read_csv("permits_features.csv")` is split chronologically, but **no re-encoding is ever performed**. The globally target-encoded `zip_target_encoded` from `permits_features.csv` is fed directly into XGBoost.

---

## 6. Scientific Summary & Recommendations for Paper Rebuild

To produce uncompromised, peer-reviewable research findings for the academic paper, the pipeline must be restructured according to strict chronological isolation:

1. **Delete `area_value_ratio` Completely:**
   Remove this feature from both Stage 1 and Stage 2. It has no valid representation at inference time.
2. **Move Feature Engineering Downstream of Chronological Split:**
   The workflow order must strictly be:
   $$\text{Raw Data} \longrightarrow \text{Train / Val / Test Chronological Split} \longrightarrow \text{Fit Encoders on Train} \longrightarrow \text{Transform Val / Test}$$
3. **Out-of-Fold Target Encoding for `Zip Code`:**
   Use 5-fold out-of-fold target encoding with empirical Bayes / m-estimate smoothing computed **exclusively within the training period** (2018–Oct 2019). For validation and test sets, map the learned training dictionary; assign unseen ZIPs the global training mean.
4. **Historical Point-in-Time Frequency Encoding:**
   `contractor_permit_count`, `contractor_city_freq`, and `permit_code_freq` must be fitted exclusively on `df_train`. Val/Test frequencies must reflect what was known up to the split boundary, or contractor features should be reformulated as expanding-window historical counts.
5. **Fit Imputers & Quantiles on `df_train` Only:**
   The 99th percentile cap on `Value` and the `Area` group medians must be computed solely on training records and saved as frozen transformation parameters for validation and testing.
