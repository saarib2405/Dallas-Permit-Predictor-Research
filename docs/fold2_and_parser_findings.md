# Diagnostic Findings: Fold 2 Anomaly & Permit-Type Parser Audit

This report presents the empirical findings from the investigation of:
1. **The Fold 2 (2019 Q2) Anomaly**: Root-causing the lower Stage 1 recall ($0.669$) and dollar $R^2$ ($0.201$) relative to other folds.
2. **Permit-Type Parser Fallback Rate**: Measuring pattern hit rates in `parse_permit_type` and explaining why `sub_category_General` and `work_type_Other` dominate Stage 1 feature importance.

---

## 1. Task 1: Root-Cause Analysis of the Fold 2 Anomaly

### 1.1 Zero-Value Rate Shift (Train vs. Validation Across Folds)

We evaluated the `is_zero_value` percentage across the training window and validation window for each forward-chaining fold:

| Fold | Validation Window | Train $N$ | Train Zero % | Val $N$ | Val Zero % | Diff (pp) | Rel Change (%) |
|------|-------------------|-----------|--------------|---------|------------|-----------|----------------|
| **Fold 1** | 2019 Q1 (Winter)  | 47,343 | 15.32% | 11,334 | 14.78% | -0.54 pp | -3.5% |
| **Fold 2** | 2019 Q2 (Spring)  | 58,677 | 15.21% | 12,653 | 16.30% | +1.08 pp | +7.1% |
| **Fold 3** | 2019 Q3 (Summer)  | 71,330 | 15.41% | 14,890 | 31.24% | +15.83 pp| +102.8% |
| **Fold 4** | 2019 Q4 (Year-end)| 86,220 | 18.14% | 13,160 | 35.39% | +17.25 pp| +95.1% |

#### Key Insight: The Structural Regime Shift in Building Permits
While Fold 2's aggregate zero rate only rose modestly (+1.08 pp), decomposing by permit category reveals a profound structural shift that begins specifically in **2019 Q2**:

| Year & Quarter | Building Permits Total | Building Permits Zero-Value Count | Building Permit Zero % |
|----------------|------------------------|-----------------------------------|-----------------------|
| 2018 Q1 | 3,953 | 82 | **2.07%** |
| 2018 Q2 | 4,541 | 105 | **2.31%** |
| 2018 Q3 | 3,744 | 79 | **2.11%** |
| 2018 Q4 | 3,798 | 97 | **2.55%** |
| **2019 Q1** | 4,188 | 108 | **2.58%** |
| **2019 Q2 (Fold 2 Val)** | 4,356 | 350 | **8.03%** *(3x surge)* |
| **2019 Q3** | 6,184 | 2,235 | **36.14%** *(14x surge)* |
| **2019 Q4** | 4,974 | 2,147 | **43.16%** *(17x surge)* |
| **2020 (Holdout)** | 8,997 | 4,073 | **45.27%** |

- **Why Stage 1 Recall Dropped in Fold 2**:
  Throughout 2018 and 2019 Q1 (Fold 2's training data), standard Building permits (e.g., Single Family Alterations, Commercial Renovations) were $>97.5\%$ positive-value. In 2019 Q2, Dallas began recording zero-value building permits at triple the historical rate. Because Fold 2's model had never seen regular building categories with zero value, it predicted them as positive (false negatives for zero), reducing zero-class recall to $0.669$. By Folds 3 and 4, the training set included Q2/Q3 data where the model learned this new pattern.

---

### 1.2 Categorical Distribution Across 2019 Quarters

Comparing category distributions across the four validation quarters:

- **Permit Category**:
  - Mechanical permits spiked to $11.26\%$ in Q2 (vs. $7.41\%$ average across other quarters; ratio $1.52$).
  - Electrical permits rose to $16.75\%$ (vs. $14.92\%$).
  - Building permits were slightly lower at $34.43\%$ (vs. $38.76\%$).
- **Sub-Category**:
  - Extremely stable: Single Family ($62.0\%$), Commercial ($23.0\%$), Multi Family ($7.6\%$), General ($7.4\%$).
- **Work Type**:
  - Extremely stable: Alteration ($64.7\%$), Renovation ($16.8\%$), New Construction ($7.5\%$), Other ($7.4\%$).
- **Land Use**:
  - Single Family Dwelling ($62.5\%$), Multi-Family ($8.0\%$), Office ($5.5\%$) remained consistent with historical shares.

---

### 1.3 Unseen ZIP Code Fallback Coverage

We checked whether Fold 2 experienced an out-of-vocabulary spike in unmapped ZIP codes:

| Fold | Val Quarter | Training Unique ZIPs | Val Total Rows | Unseen ZIP Rows | Unseen ZIP % |
|------|-------------|----------------------|----------------|-----------------|--------------|
| **Fold 1** | 2019 Q1 | 69 | 11,334 | 446 | **3.94%** |
| **Fold 2** | 2019 Q2 | 69 | 12,653 | 540 | **4.27%** |
| **Fold 3** | 2019 Q3 | 69 | 14,890 | 618 | **4.15%** |
| **Fold 4** | 2019 Q4 | 70 | 13,160 | 512 | **3.89%** |

- **Conclusion**: Unseen ZIP rates are flat at $\sim 3.9\%\text{--}4.3\%$ across all folds. Unseen ZIP fallback did **not** cause the anomaly.

---

### 1.4 Temporal Volume & Ingestion Anomaly Analysis

- **Daily Volume Consistency**:
  - 2019 Q1: 87 reporting days, mean $130.3$ permits/day.
  - 2019 Q2: 89 reporting days, mean $142.2$ permits/day.
  - 2019 Q3: 91 reporting days, mean $163.6$ permits/day.
  - 2019 Q4: 92 reporting days, mean $143.0$ permits/day.
- **Outlier Days**:
  - Only one unusually low weekday: Memorial Day (2019-05-27, 17 permits).
  - Only one unusually high weekday: Good Friday (2019-04-19, 359 permits, normal seasonal surge).
- **Batch Duplicates**: No anomalous duplicate ingestion bursts or missing periods were found.

---

### 1.5 The Dollar $R^2$ ($0.201$) Explanation: Denominator Compression

A critical mathematical insight explains why Fold 2's dollar $R^2$ appears artificially low:

| Fold | Val Quarter | Val Positive Value Variance $\text{Var}(y)$ | Dollar RMSE (Hard) | Dollar $R^2$ (Hard) | Log-Space $R^2$ |
|------|-------------|---------------------------------------------|--------------------|---------------------|-----------------|
| **Fold 1** | 2019 Q1 | $9.37 \times 10^{11}$ | **$675,892** | **0.5124** | **0.7452** |
| **Fold 2** | 2019 Q2 | $4.67 \times 10^{11}$ | **$610,872** | **0.2007** | **0.7455** |
| **Fold 3** | 2019 Q3 | $1.81 \times 10^{11}$ | **$323,214** | **0.4218** | **0.6593** |
| **Fold 4** | 2019 Q4 | $2.69 \times 10^{11}$ | **$352,044** | **0.5401** | **0.6926** |

> [!IMPORTANT]
> **Fold 2 actually achieved a LOWER Dollar RMSE ($610,872) than Fold 1 ($675,892)!**
> However, because Fold 1 contained two mega-permits ($43M and $38.5M), its variance denominator $\sum (y - \bar{y})^2$ was more than double Fold 2's. In contrast, in **log space** (which is scale-invariant and immune to dollar variance distortion), **Fold 2's $R^2$ is $0.7455$, virtually identical to Fold 1's $0.7452$**.

### Task 1 Conclusion
The Fold 2 metrics reflect a **genuine real-world administrative shift** (the initial emergence of zero-value Building permits in Dallas in spring 2019) combined with a **dollar-space metric artifact** (smaller variance denominator). It is **not** a data pipeline bug or leakage defect. It should be preserved as an authentic finding and discussed in the paper's empirical evaluation section.

---

## 2. Task 2: Permit-Type Parser Fallback Rate & Pattern Audit

### 2.1 Parser Matching Breakdown

We audited `parse_permit_type` across the entire 126,837-row cleaned dataset:

| Pattern | Definition | Count | Percentage |
|---------|------------|-------|------------|
| **Pattern 1** | Standard 4-part: `Category (Code) SubCategory  WorkType` | 116,476 | **91.83%** |
| **Pattern 2** | 2-part double space: `Category  WorkType` | 0 | **0.00%** |
| **Pattern 3** | Catch-all Fallback: `(pt, 'NA', 'General', 'Other')` | 10,361 | **8.17%** |

- **Year-over-Year Stability**:
  - 2018: $8.57\%$ fallback
  - 2019: $7.58\%$ fallback
  - 2020: $8.59\%$ fallback

---

### 2.2 Complete Inventory of All Fallback Strings

Across the entire dataset, **exactly 13 distinct strings** hit Pattern 3 Fallback. Every single one is listed below:

| # | Occurrences | Raw `Permit Type` String | Domain Interpretation |
|---|-------------|--------------------------|-----------------------|
| 1 | 5,171 | `"Electrical Sign New Construction"` | Commercial Sign (Electrical) |
| 2 | 2,230 | `"Demolition Permit SFD/Duplex"` | Demolition (Single Family / Duplex) |
| 3 | 1,016 | `"Sign New Construction"` | Commercial Sign (Non-electrical) |
| 4 | 977 | `"Electrical Sign Alteration"` | Commercial Sign Alteration |
| 5 | 415 | `"Demolition Permit Commercial"` | Commercial Demolition |
| 6 | 245 | `"Supergraphics Alteration"` | Supergraphics Sign Alteration |
| 7 | 230 | `"Sign Alteration"` | Commercial Sign Alteration |
| 8 | 31 | `"Electrical Sign Reconstruction"` | Commercial Sign Reconstruction |
| 9 | 21 | `"Supergraphics New Construction"` | Supergraphics Sign |
| 10| 8 | `"Special Purpose Sign New Construction"` | Special Purpose Sign |
| 11| 6 | `"Electrical Sign Renovation"` | Commercial Sign Renovation |
| 12| 6 | `"Sign Reconstruction"` | Commercial Sign Reconstruction |
| 13| 5 | `"Sign Renovation"` | Commercial Sign Renovation |
| **Total** | **10,361** | *(All 13 strings accounted for)* | |

---

### 2.3 The Root Cause of `work_type_Other` & `sub_category_General` Feature Importance

The audit uncovered a definitive explanation for the feature importance rankings in Stage 1:

1. **Every single instance** of `work_type == "Other"` ($N = 10,361$) and `sub_category == "General"` ($N = 10,361$) in the entire dataset comes exclusively from these 13 strings hitting Pattern 3.
2. For all 10,361 of these permits, the declared value is **$0 (100.0% zero-value rate)**!
3. Because the parser hardcoded `work_type = "Other"` for unparsed strings, and these 13 permit types happen to be administrative regulatory permits (Signs and Demolitions) with zero declared value, **`work_type == "Other"` became a perfect mathematical indicator for $V = \$0$**.
4. The tree-based classifier naturally picked this up as its strongest splitting feature.

### 2.4 Structural Anatomy of the 13 Strings
Notice that these 13 strings actually follow regular patterns:
- **Sign permits**: `[Category] [WorkType]` where WorkType is one of `New Construction`, `Alteration`, `Renovation`, `Reconstruction`. They lack the two-letter parenthesized code (e.g. `(BU)`) and the double-space separator.
- **Demolition permits**: `"Demolition Permit [SubCategory]"` where SubCategory is `SFD/Duplex` or `Commercial`. The WorkType is inherently "Demolition".

---

## 3. Recommendations & Next Steps

1. **For Fold 2**:
   - Retain the current results. The performance dip is a documented historical transition (the onset of Dallas recording zero-value building permits in spring 2019) plus variance denominator compression for dollar $R^2$.
2. **For the Permit Type Parser**:
   - We now have the exact 13 strings. If desired, we can introduce a clean supplementary regex to parse:
     - Signs into: category=`"Sign"`, sub_category=`"Commercial"`, work_type=`"New Construction" / "Alteration" / etc.`
     - Demolitions into: category=`"Demolition"`, sub_category=`"Single Family" / "Commercial"`, work_type=`"Demolition"`
   - This would replace the synthetic `work_type == "Other"` with genuine domain categories.
