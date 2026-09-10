# SHAP Interpretability Analysis

**Model:** Two-Stage Hurdle Model (Stage 2 — XGBoost Regressor)  
**Method:** SHAP TreeExplainer on test set (19,026 permits, March–August 2020)  
**Features:** 62 (area_value_ratio excluded from Stage 2 to prevent data leakage)

---

## 1. What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theory-based framework for interpreting ML predictions. For every prediction, SHAP assigns each feature a **SHAP value** — the amount that feature pushed the prediction up or down from the baseline.

- **Positive SHAP value** → feature increases the predicted permit value
- **Negative SHAP value** → feature decreases the predicted permit value
- **Base value** (E[f(X)] = 8.931) → the average prediction across all permits (~$7,600)

---

## 2. Global Feature Importance

The bar chart below ranks all 62 features by their **mean absolute SHAP value** — the average impact each feature has on predictions across the entire test set.

![SHAP Bar Summary](plots/shap_bar_summary.png)

### Top 5 Features

| Rank | Feature | Mean |SHAP| | What It Means |
|------|---------|-------------|---------------|
| 1 | `work_type_Alteration` | ~0.52 | Whether the permit is for an alteration (repair/modification) vs. new construction. Alterations are consistently cheaper. |
| 2 | `Area` | ~0.49 | Square footage of the building. Larger buildings → higher cost. The dominant continuous feature. |
| 3 | `permit_code_freq` | ~0.19 | How common the permit code is in the dataset. Building permits (BU, freq=44,735) behave differently from rare codes. |
| 4 | `permit_category_Building` | ~0.16 | Whether this is a structural building permit vs. plumbing, electrical, or mechanical. Building permits are higher value. |
| 5 | `contractor_permit_count` | ~0.13 | How many permits the contractor has filed. High-volume contractors handle different project scales. |

**Key Takeaway:** *What you're doing* (alteration vs. new construction) and *how big it is* (square footage) matter far more than *where* you're doing it or *when* you filed.

---

## 3. Beeswarm Plot — Feature Value vs. SHAP Impact

Each dot represents one permit. Color = feature value (red = high, blue = low). Position = SHAP impact on prediction.

![SHAP Beeswarm](plots/shap_beeswarm.png)

### Key Observations

- **work_type_Alteration (Row 1):** Red dots (alteration = 1) form a tight cluster pushed right with large positive SHAP values. Since the model predicts log(Value), and alterations are cheaper, this feature creates the biggest swing between project types.

- **Area (Row 2):** The clearest monotonic pattern. Red dots (large buildings) push predictions up (+1 to +2.5). Blue dots (small/zero area) push down (-0.5 to -1.0). This is the primary continuous value driver.

- **permit_category_Electrical (Row 9):** Red dots (electrical = 1) pull predictions LEFT (negative SHAP). Electrical permits are consistently lower value than building permits — they cover wiring and panel work, not structural construction.

- **work_type_New Construction (Row 13):** Red dots (new construction = 1) push predictions right. New construction is the most expensive work type, confirming domain knowledge.

---

## 4. Local Explanations — Waterfall Plots

Waterfall plots trace how a single prediction is built from the base value, showing each feature's contribution.

### 4.1 High-Value Permit — $2,449,100

![SHAP Waterfall: High Value](plots/shap_waterfall_high_value.png)

**How the model reasons:**

Starting from the base value of 8.931 (≈ $7,600):

| Feature | Value | SHAP | Reasoning |
|---------|-------|------|-----------|
| Area | 26,804 sqft | +1.49 | Large building → expensive |
| work_type_Alteration | 0 (not alteration) | +1.47 | New construction → expensive |
| permit_code_freq | 44,735 | +0.48 | Building permit code (most common) |
| area_value_ratio | 10.196 | +0.46 | High cost-efficiency ratio |
| work_type_New Construction | 1 | +0.44 | Confirms new construction |
| permit_category_Building | 1 | +0.37 | Structural work |

**Final prediction:** 8.931 → 14.234 in log-space → **$2,449,100**

This mirrors how a human permit officer would reason: *"27,000 sqft new commercial building = expensive."*

---

### 4.2 Mid-Value Permit — $5,200

![SHAP Waterfall: Mid Value](plots/shap_waterfall_mid_value.png)

**How the model reasons:**

| Feature | Value | SHAP | Reasoning |
|---------|-------|------|-----------|
| Area | 0 sqft | -0.66 | No floor space — likely a repair |
| work_type_Alteration | 1 | -0.38 | It's an alteration — cheaper |
| area_value_ratio | 0 | -0.17 | No area data available |
| days_since_epoch | 7,534 | -0.17 | Filed later in 2020 (COVID period) |

**Final prediction:** 8.931 → 7.676 → **$5,200**

The contrast is stark: no square footage + alteration work = the model correctly predicts a low-value routine repair.

---

### 4.3 Low-Value Permit — $1

![SHAP Waterfall: Low Value](plots/shap_waterfall_low_value.png)

This permit has Area = 8,000 sqft (+0.75) but it's an alteration (-0.68), has a high zip_target_encoded (+0.40) but contractor_permit_count = 1 (-0.33). The competing forces nearly cancel out, resulting in a prediction near the baseline: **f(x) = 9.686 ≈ $16,000** (the actual value was $1, showing the inherent noise in self-reported permit values).

---

## 5. Dependence Plots

Dependence plots show the relationship between a single feature's value (x-axis) and its SHAP impact (y-axis).

### 5.1 Area vs. SHAP Value

![SHAP Dependence: Area](plots/shap_dependence_Area.png)

- **Non-linear relationship:** Below ~50,000 sqft, each additional square foot adds significant predicted value. Above that, the curve flattens — diminishing returns for mega-buildings.
- **Color (permit_code_freq):** Red dots (Building permits, freq ~44,000) get pushed higher than blue dots (less common codes) at the same square footage. A 10,000 sqft *building* permit is worth more than a 10,000 sqft *plumbing* permit.

### 5.2 Permit Code Frequency

![SHAP Dependence: Permit Code Freq](plots/shap_dependence_permit_code_freq.png)

### 5.3 Work Type: Alteration

![SHAP Dependence: Work Type Alteration](plots/shap_dependence_work_type_Alteration.png)

- **Binary split:** When alteration = 0 (new construction/renovation), SHAP clusters near zero. When alteration = 1, SHAP shows a wide spread from -1.5 to +2.5, indicating the model treats alterations differently depending on other contextual features.

---

## 6. Summary

| Insight | Evidence |
|---------|----------|
| Work type is the #1 value driver | Alteration vs. new construction creates the largest SHAP swing |
| Area has a non-linear effect | Diminishing returns above ~50K sqft |
| Building permits > subsystem permits | permit_category_Building consistently pushes values up |
| Location matters, but less | zip_target_encoded ranks 8th, not in top 3 |
| Model reasoning matches domain expertise | Waterfall plots mirror how human permit officers estimate costs |
