# Data Cleaning Log — Dallas Building Permits

**Generated:** 2026-04-11 18:59:20

---

## Step 1 — Load

**WHAT:** Loaded raw CSV: 126,840 rows × 11 columns

**WHY:** All columns loaded as strings to prevent pandas from auto-inferring types (e.g., Zip Code would be read as int, losing leading zeros and categorical nature).

---

## Step 1 — Type Cast

**WHAT:** Parsed 'Value' column: removed commas, cast to float64

**WHY:** Value is stored as comma-formatted strings (e.g., '665,000'). Must be numeric for math operations.

**DATA:** `Range: $0 — $64,565,291`

---

**WHAT:** Parsed 'Area' column: removed commas, cast to float64

**WHY:** Same comma-formatting issue as Value.

**DATA:** `Range: 0 — 2,478,361 sq ft`

---

**WHAT:** Parsed 'Issued Date': MM/DD/YY → datetime64

**WHY:** 2-digit year format (e.g., '03/13/20' → 2020-03-13). pandas handles the century correctly for 18-20.

**DATA:** `Range: 2018-01-02 — 2020-08-29`

---

## Step 2 — Null Audit

**WHAT:** Null counts — Value: 0, Issued Date: 0, Permit Type: 0, Contractor: 3

**WHY:** We only remove rows where critical fields are null. Value=0 rows are KEPT for the Stage 1 Hurdle Model classifier.

---

## Step 2 — Row Removal

**WHAT:** Dropped 3 rows with null critical fields (0.0024%)

**WHY:** Only 3 rows had null Contractor — all other critical fields were complete. We do NOT drop Value=0 rows because they serve as training data for the Stage 1 binary classifier in our Hurdle Model architecture.

**DATA:** `Remaining: 126,837 rows`

---

## Step 3 — Stage 1 Label

**WHAT:** Created 'is_zero_value': 1 if Value==0, else 0

**WHY:** This is the target variable for the Stage 1 binary classifier. It learns to distinguish administrative permits ($0) from real construction projects.

**DATA:** `Class balance — Zero: 30,051 (23.7%) | Positive: 96,786 (76.3%)`

---

## Step 3 — Stage 2 Label

**WHAT:** Created 'log_value': ln(Value) for positive-value records, NaN for zeros

**WHY:** Log-transformation is mandatory because Value is extremely right-skewed (mean $87.6K vs median $2.8K, ratio 31:1). Log-transforming normalizes the distribution, stabilizes variance, and prevents large-value outliers from dominating the loss function.

**DATA:** `log_value range: -4.61 — 17.98 (corresponds to $0 — $64,565,291)`

---

## Step 4A — Parse Permit Type

**WHAT:** Parsed 'Permit Type' into 4 columns: permit_category, permit_code, sub_category, work_type

**WHY:** The raw Permit Type field encodes 4 dimensions in one string. Splitting them gives the ML model access to individual categorical features instead of 137 one-hot columns.

**DATA:** `Parse success (standard format): 116,476 (91.8%) | Fallback parse: 10,361 (8.2%)`

---

## Step 4A — Parsed Component

**WHAT:** 'permit_category': 37 unique values

**WHY:** Top 5: {'Building': 44735, 'Plumbing': 25360, 'Electrical': 18427, 'Mechanical': 12140, 'Electrical Sign New Construction': 5171}

---

**WHAT:** 'permit_code': 26 unique values

**WHY:** Top 5: {'BU': 44735, 'PL': 25360, 'EL': 15616, 'ME': 12140, 'NA': 10361}

---

**WHAT:** 'sub_category': 4 unique values

**WHY:** Top 5: {'Single Family': 76754, 'Commercial': 29746, 'General': 10361, 'Multi Family': 9976}

---

**WHAT:** 'work_type': 7 unique values

**WHY:** Top 5: {'Alteration': 81005, 'Renovation': 22246, 'Other': 10361, 'New Construction': 8384, 'Addition': 3804}

---

## Step 4B — Parse Contractor

**WHAT:** Parsed 'Contractor' into contractor_name (13,621 unique) and contractor_city (549 unique)

**WHY:** The raw Contractor field is a single string containing company name, full mailing address, and phone number. We extract just the name (for repeat-contractor features) and city (for frequency encoding) — the full string is too high-cardinality for direct use.

**DATA:** `City extraction success: 113,891 (89.8%) | Unknown: 12,946 (10.2%)`

---

## Step 4B — Contractor Cities

**WHAT:** Top 10 contractor cities: {'DALLAS': 36482, 'Unknown': 12946, 'MESQUITE': 6577, 'GARLAND': 6085, 'IRVING': 5804, 'PLANO': 4490, 'FORT WORTH': 4147, 'CARROLLTON': 3281, 'RICHARDSON': 2853, 'ARLINGTON': 2812}

**WHY:** Dallas dominates as expected for a Dallas building permits dataset. This feature captures whether the contractor is local or from outside the city.

---

## Step 5A — Zip Code Imputation

**WHAT:** Imputed 5,397 missing Zip Codes: 900 via Mapsco lookup, 4,497 via overall mode (75229)

**WHY:** Mapsco grid references are geographic sub-areas of Dallas — permits in the same Mapsco grid are physically close and likely share the same ZIP code. This is a more accurate imputation than global mode because it preserves spatial relationships. Remaining nulls (where Mapsco is also missing) use the dataset-wide most common ZIP.

---

## Step 5B — Area Imputation

**WHAT:** Area zeros before: 77,165 (60.8%). Imputed 46,713 using group medians (permit_category × sub_category × work_type). Still zero after: 30,452 (24.0%)

**WHY:** Many permit types genuinely have Area=0 (e.g., electrical work, plumbing repairs, sign installations) — these are not missing data, they are accurate. We only impute zeros where the group median is positive, meaning similar permits in the same category typically report a non-zero area. The remaining zeros are legitimate (the work doesn't create or modify floor space).

---

## Step 5C — Land Use

**WHAT:** Land Use: 0 missing values, 189 unique categories

**WHY:** Column is complete. No action needed. Directly usable as a categorical feature.

---

## Step 6 — Outlier Capping

**WHAT:** Capped Value at 99th percentile ($2,449,100) for positive-value records. 967 records capped (1.00%)

**WHY:** Extreme outliers (up to $64.5M) would dominate the MSE loss function and distort the regression model. Capping at the 99th percentile retains 99% of the value distribution while preventing a handful of mega-projects from skewing predictions. This is applied only to positive-value records — zeros are handled by Stage 1.

---

## Step 6 — Post-Cap Stats

**WHAT:** Stage 2 target stats after capping — Value: mean=$85,996, median=$5,495, max=$2,449,100

**WHY:** After capping, the target distribution is much more well-behaved for regression.

**DATA:** `log_value: mean=8.89, std=2.00, range=[-4.61, 14.71]`

---

## Step 7 — Type Enforcement

**WHAT:** Enforced types: 10 categorical → str, Value/Area → float64, is_zero_value → int, Issued Date → datetime64

**WHY:** Zip Code must be string (it's a categorical identifier, not a number — arithmetic operations on it are meaningless). is_zero_value is 0/1 integer for classification. Consistent typing prevents unexpected behavior in downstream encoding and modeling.

---

## Step 8 — Column Drop

**WHAT:** Dropped 4 columns: ['Mapsco', 'Street Address', 'Work Description', 'Contractor']

**WHY:** Mapsco: used only for zip imputation — too granular for modeling (3,066 unique). Street Address: too high-cardinality (76K unique) and not useful without geocoding. Work Description: free-text field (17K unique) — would need NLP; out of scope for tabular ML. Contractor: replaced by parsed contractor_name and contractor_city.

---

## Step 8 — Final Report

**WHAT:** Saved permits_cleaned.csv: 126,837 rows × 15 columns (100.00% retention)

**WHY:** Two-Stage Hurdle Model ready. Stage 1 class balance: 23.7% zero / 76.3% positive. Stage 2 has 96,786 records for regression.

---

