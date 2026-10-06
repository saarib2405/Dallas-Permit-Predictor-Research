# Predictive Modeling of Municipal Building Permit Valuations: A Leakage-Free Two-Stage Hurdle Framework with Empirical Regime Diagnostics

**Authors:** Data Science & Analytics Research Group  
**Project:** ASDS 6302 Final Project — Dallas Building Permits Valuation Study  
**Document Version:** v3.1 (Burn-In Fixed Research Draft)  
**Date:** September 2026  

---

## Abstract

Accurate estimation of declared building-permit values from municipal building permit applications is essential for urban infrastructure planning and regulatory compliance. However, municipal permit records exhibit two challenging statistical properties: (1) an extreme zero-inflation spike ($\approx 24\%$) associated with non-construction administrative, regulatory, and demolition filings, and (2) severe right-skewness spanning seven orders of magnitude (from $\$0$ to over $\$64.5\text{M}$). Prior attempts to model this dataset suffered from subtle methodological flaws, including lookahead leakage in static frequency encodings, target leakage via self-labeled encoding and target-derived features, and uncorrected retransformation bias under log-normal transformations. 

In this work, we propose a rigorous, leakage-free **Two-Stage Hurdle Architecture** combining an XGBoost binary classification hurdle with an uncapped log-scale XGBoost regressor corrected by empirical **Duan Smearing**. Crucially, we enforce a strict 60-day cold-start burn-in exclusion on model parameter fitting (`df_train_fit`) while preserving complete continuous expanding historical counts, guaranteeing zero leakage. We evaluate our framework using a strict 4-fold forward-chaining cross-validation schedule across 2019 and a quarantined, one-shot holdout evaluation on 2020 ($N = 27,457$, through August 29, 2020). Across 2019 folds, our model achieves a primary log-space $R^2$ of $0.7145 \pm 0.0360$ and log-RMSE of $1.07 \pm 0.04$. On the 2020 holdout, the model achieves a log-space $R^2$ of $0.6349$, with a Stage 1 classifier ROC-AUC of $0.9678$ and an uncapped Dollar $R^2$ of $0.5192$. 

Furthermore, we present two major empirical and data-engineering contributions: (1) an in-depth root-cause investigation of a historical performance anomaly in 2019 Q2, identifying an observed change in recording pattern (Dallas began recording zero-value building permits at triple the historical rate) and metric denominator compression in dollar space; and (2) a forensic audit of the composite permit parser, identifying 13 unparsed regulatory string patterns (10,361 permits, $100\%$ zero-value) that had created a synthetic binary shortcut feature (`work_type_Other`), which we resolved to restore genuine domain feature importance. Finally, comparative evaluations against Naive, Log-Linear Ridge, and Non-Hurdle Tweedie XGBoost baselines reveal clear trade-offs between modeling paradigms: while the Two-Stage Hurdle model achieves higher overall dollar $R^2$ (cross-fold mean $0.4542$ vs. $0.3888$) and superior generalization on the 2020 holdout ($0.5192$ vs. $0.3236$), the single-stage Tweedie model wins across all three dollar metrics in Fold 3 (Dollar $R^2$ of $0.4762$ vs. $0.4344$, MAE of $\$43,437$ vs. $\$50,216$, RMSE of $\$307,618$ vs. $\$319,678$) and achieves lower MAE in Fold 2 ($\$68,718$ vs. $\$72,340$).

---

## 1. Introduction and Methodological Corrections

Predicting municipal permit valuation is fundamentally a hurdle regression problem. Standard single-stage regression models fail because administrative permits (such as sign postings, barricades, and single-family demolitions) require municipal review but legally declare $\$0$ in construction value. A two-stage hurdle model addresses this by separating the probability of positive construction value from the conditional dollar amount:

$$\mathbb{E}[V \mid X] = P(V > 0 \mid X) \cdot \mathbb{E}[V \mid V > 0, X]$$

### 1.1 Flaws in Naive Baseline Designs and Methodological Solutions

An audit of preliminary baseline pipelines revealed five severe methodological vulnerabilities that compromised previous findings. Table 1 summarizes these flaws and the formal corrections implemented in our framework.

#### Table 1: Methodological Corrections over Baseline Designs
*Source: Methodological audit and redesign specification (`feature_audit.md`, `pipeline_redesign_outline.md`)*

| # | Flawed Baseline Practice | Failure Mode / Consequence | Methodological Correction in Framework |
|---|---|---|---|
| **1** | Static Whole-Dataset Frequency Freezing | **Lookahead Leakage:** Validation and holdout frequencies were computed using future record counts frozen at training time. | **Continuous Point-in-Time Expanding Counts:** Every record counts only strictly prior events ($t' < t$) across the entire sorted timeline. |
| **2** | In-Fold Global Target Encodings & Cold-Start Noise | **Target Leakage & Initial Noise:** Training records received target encodings computed from their own labels, or initial arbitrary whole-window priors. | **Expanding Empirical Bayes Encodings & 60-Day Burn-In Fitting Exclusion:** Dynamic expanding priors strictly prior to date $t$. Records in the first 60 days of training are excluded from model fitting (`df_train_fit`) to prevent cold-start noise feedback. |
| **3** | Direct Target Leakage via Derived Ratios | **Mathematical Infeasibility:** Feature `area_value_ratio` ($Area / Value$) requires the target at inference time. | **Complete Elimination:** `area_value_ratio` completely removed from both Stage 1 and Stage 2 feature vectors. |
| **4** | Silent Target Capping (p99) on Primary Regressor | **Biased Retransformation:** Training on $\min(V, p99)$ artificially deflated upper-tail dollar forecasts. | **Uncapped Primary Target Specification:** Primary Stage 2 fitted on full uncapped log-value; p99-capped model retained solely as a labeled sensitivity check. |
| **5** | Naive Exponentiation $\exp(\hat{y})$ | **Retransformation Underestimation:** Jensen's inequality implies $\mathbb{E}[\exp(\epsilon)] > 1$ for non-zero error variance. | **Non-Parametric Duan Smearing Correction:** Residual smearing factor $S = \frac{1}{N}\sum \exp(y_i - \hat{y}_i)$ fit on training residuals before retransformation. |

---

## 2. Leakage-Free Methodology & Architecture

### 2.1 Dataset Scope and Global Distribution

The empirical framework operates on 126,837 cleaned permit records spanning January 2, 2018 to August 29, 2020. Table 1B presents the global dataset characteristics, establishing the ground truth for the valuation range and overall zero-inflation rate referenced throughout this study.

#### Table 1B: Dataset-Wide Overview and Target Distribution (2018–2020)
*Source: Full cleaned corpus (`permits_research_cleaned.csv`)*

| Metric / Attribute | Global Value | Context / Empirical Notes |
|---|---|---|
| **Total Filings ($N$)** | **126,837** | 2018-01-02 to 2020-08-29 (entire corpus) |
| **Development Horizon (2018–2019)** | 99,380 ($78.35\%$) | 4-fold forward-chaining cross-validation corpus |
| **Quarantined Holdout (2020)** | 27,457 ($21.65\%$) | One-shot prospective test set (Jan 1 – Aug 29, 2020) |
| **Zero-Valuation Filings ($V = \$0$)** | **30,051 (23.69%)** | Regulatory, administrative, sign, and demolition permits |
| **Positive Valuation Filings ($V > \$0$)**| **96,786 (76.31%)** | Construction projects subject to Stage 2 log regression |
| **Minimum Value** | $\$0.00$ | Baseline administrative permits |
| **Median Positive Value** | $\$10,000.00$ | Typical residential trade / alteration work |
| **Mean Positive Value** | $\$124,314.12$ | Skewed by major commercial developments |
| **Maximum Declared Valuation** | **$\$64,565,291.00$** | Commercial Warehouse New Construction (Issued: 2018-07-26) |
| **Valuation Span** | $7$ orders of magnitude | $\log(V)$ range: $-4.61$ to $17.98$ for positive filings |

```
+-----------------------------------------------------------------------------------------+
|                               LEAKAGE-FREE PIPELINE ARCHITECTURE                        |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  Raw Inputs: [Permit Type, Issued Date, Zip Code, Mapsco, Square Feet, Contractor, ...] |
|                                       |                                                 |
|                                       v                                                 |
|  [Deterministic Parsers] --------> 4-Part Permit Type + Contractor City                |
|                                       |                                                 |
|                                       v                                                 |
|  [Point-in-Time Expanding Frequencies] (Strictly prior records: t' < t)                 |
|                                       |                                                 |
|                                       v                                                 |
|  [Per-Fold Imputation] ----------> Mapsco Mode -> ZIP | Group Median -> Area             |
|                                       |                                                 |
|                                       v                                                 |
|  [Bayesian Target Encodings] ----> 60-Day Cold Start -> Expanding Prior Smoothing       |
|                                       |                                                 |
|                                       v                                                 |
|               [60-Day Burn-In Exclusion: df_train_fit (Issued Date > Start + 60d)]      |
|                                       |                                                 |
|                   +-------------------+-------------------+                             |
|                   |                                       |                             |
|                   v                                       v                             |
|        [Stage 1: XGBClassifier]                [Stage 2: XGBRegressor]                  |
|           Target: is_zero_value                   Target: log(Value) [Uncapped]         |
|                   |                                       |                             |
|                   v                                       v                             |
|          P(Positive Value) = 1 - p           Predicted log(V) -> Duan Smearing S * exp()|
|                   |                                       |                             |
|                   +-------------------+-------------------+                             |
|                                       |                                                 |
|                                       v                                                 |
|                       [Dual Hurdle Retransformation]                                    |
|              Soft: E[V] = (1 - p) * S * exp(y_log)                                      |
|              Hard (Decision-Rule Prediction): V_hat = S * exp(y_log) if p < 0.5 else $0 |
+-----------------------------------------------------------------------------------------+
```

### 2.1 Point-in-Time Continuous Expanding Counts

For categorical entities (contractor name, contractor city, permit code, and land use occupancy), frequency features must reflect only historical volume known at the exact moment of permit filing:

$$C_{i}(k) = \sum_{j: \text{Date}_j < \text{Date}_i} \mathbf{1}(X_j = k)$$

This is implemented via a vectorized daily pivot cumsum shifted by 1 calendar day. A contractor is classified as a repeat contractor if and only if their continuous prior permit count satisfies $C_{i}(\text{contractor}) \ge 10$.

### 2.2 Per-Fold Imputations

To prevent global leakage across validation boundaries:
1. **Missing ZIP Codes** are imputed using the empirical mode of the property's Mapsco navigation grid coordinate computed exclusively on the active training window, falling back to the training window's overall modal ZIP (`75229`).
2. **Missing or Zero Area** records are imputed using the positive-area median of the corresponding `(permit_category, sub_category, work_type)` stratum within the training window, falling back to the global positive median area ($1,400\text{ sq ft}$).

### 2.3 Stage-Specific Bayesian Target Encodings with Cold-Start Burn-In Exclusion

Target encodings are computed separately for Stage 1 ($P(\text{Zero Value})$) and Stage 2 ($\mathbb{E}[\log(V) \mid V > 0]$) using an empirical Bayes m-estimate smoothing formulation ($M = 10.0$):

$$\hat{S}_{\text{ZIP}}(t_i) = \frac{\sum_{j \in \text{ZIP}, t_j < t_i} y_j + M \cdot g(t_i)}{N_{\text{ZIP}}(t_i) + M}$$

where $g(t_i)$ is the strictly expanding global rate computed across all observed filings prior to date $t_i$. In contrast to naive approaches that rely on static window averages, $g(t_i)$ updates dynamically as each day passes, ensuring zero future information enters the encoding.

To eliminate cold-start target feedback at the inception of each training window, records falling within the first 60 days ($t \le \text{Start} + 60\text{ days}$) serve as an expanding count burn-in window. These records are strictly excluded from model parameter fitting (`df_train_fit`), but flow through the expanding pipeline to establish baseline history.

> **Deliberate Design Choice (Validation ZIP Maps):** Frozen validation-time empirical Bayes target encoding maps are constructed from the **full historical training fold** (including the 60-day burn-in window up to `train_end`). Because every record in the burn-in window occurred strictly prior to the validation period, incorporating them provides maximum valid historical geographic coverage and stabilizes shrinkage estimates without introducing temporal leakage. Meanwhile, model parameter fitting (`df_train_fit`) strictly excludes the burn-in window.

### 2.4 Duan Smearing Retransformation

Because $\mathbb{E}[\exp(\epsilon)] \ge \exp(\mathbb{E}[\epsilon]) = 1$, naively exponentiating log-scale predictions introduces systematic underestimation. We correct for this using Duan's non-parametric smearing factor:

$$S = \frac{1}{N_{\text{pos}}} \sum_{i=1}^{N_{\text{pos}}} \exp\left(y_i - \hat{y}_i\right)$$

The smearing factor $S$ is fitted strictly on positive training residuals ($S \approx 1.46\text{--}1.55$ across cross-validation folds, and $1.61$ on the full production model) and applied during dollar-space retransformation.

For downstream valuation, predictions are evaluated under two distinct paradigms: (1) **soft continuous expected value**, $\mathbb{E}[V] = (1 - p) \cdot S \cdot \exp(\hat{y}_{\log})$, and (2) the discrete **decision-rule prediction** (hard hurdle), $\hat{V} = S \cdot \exp(\hat{y}_{\log})$ if $p < 0.5$ else $\$0$. All primary dollar-space evaluation metrics reported throughout this study reflect this hard decision-rule prediction.

---

## 3. Empirical Evaluation and Results

Evaluation follows a strict 4-fold forward-chaining quarterly schedule across 2019, followed by an out-of-sample holdout evaluation on 2020. 

### 3.1 4-Fold Forward-Chaining Results (2019 Quarters)

Table 2 presents the fold-by-fold results for the primary uncapped model under the locked v3.1 production pipeline (with 60-day burn-in exclusion).

#### Table 2: 4-Fold Forward-Chaining Primary Performance (v3.1 Pipeline)
*Source: `outputs/pipeline_results.json`*

| Fold | Training Window | Validation Quarter | Effective Train $N$ | Val $N$ | Stage 1 Acc | Stage 1 F1 | Stage 1 AUC | Log $R^2$ (Primary) | Log RMSE | Duan $S$ | Dollar $R^2$ (Hard) | Dollar MAE (Hard) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Fold 1** | 2018-01-02 to 2018-12-31 | 2019 Q1 (Winter) | 40,361 | 11,334 | 0.9793 | 0.9256 | 0.9766 | **0.7422** | 1.0796 | 1.4645 | **0.5068** | $74,955 |
| **Fold 2** | 2018-01-02 to 2019-03-31 | 2019 Q2 (Spring) | 51,695 | 12,653 | 0.9480 | 0.8137 | 0.8535 | **0.7566** | 1.0042 | 1.4954 | **0.3370** | $72,340 |
| **Fold 3** | 2018-01-02 to 2019-06-30 | 2019 Q3 (Summer) | 64,348 | 14,890 | 0.8984 | 0.8134 | 0.9528 | **0.6693** | 1.1040 | 1.5179 | **0.4344** | $50,216 |
| **Fold 4** | 2018-01-02 to 2019-09-30 | 2019 Q4 (Year-end) | 79,238 | 13,160 | 0.9361 | 0.9050 | 0.9656 | **0.6899** | 1.1038 | 1.5504 | **0.5386** | $47,685 |

*Note: Effective Train $N$ excludes the 6,982 records occurring in the first 60 days of the 2018 training timeline (burn-in period). Full historical counts (47,343; 58,677; 71,330; 86,220) were used to seed continuous frequency counts and frozen validation ZIP maps.*

#### Table 3: Cross-Fold Aggregate Performance ($\mu \pm \sigma$)
*Source: `outputs/pipeline_results.json` (`cross_fold_summary`)*

| Evaluation Metric | Primary Mean ($\mu$) | Standard Deviation ($\sigma$) | Evaluation Space / Scope |
|---|---|---|---|
| **Log-Scale $R^2$** | **0.7145** | $\pm 0.0360$ | Log Space (Positive Permits Only) |
| **Log-Scale RMSE** | **1.0729** | $\pm 0.0409$ | Log Space (Positive Permits Only) |
| **Log-Scale MAE** | **0.7836** | $\pm 0.0424$ | Log Space (Positive Permits Only) |
| **Stage 1 Accuracy** | **0.9404** | $\pm 0.0289$ | Binary Classification ($N = 13,009\text{ avg}$) |
| **Stage 1 F1-Score** | **0.8644** | $\pm 0.0514$ | Binary Classification |
| **Stage 1 ROC-AUC** | **0.9371** | $\pm 0.0490$ | Binary Classification |
| **Duan Smearing Factor ($S$)** | **1.5071** | $\pm 0.0314$ | Retransformation Multiplier |
| **Dollar-Scale $R^2$ (Hard)** | **0.4542** | $\pm 0.0775$ | Full Evaluation Set (Uncapped USD) |
| **Dollar-Scale MAE (Hard)** | **$61,299** | $\pm \$12,415$ | Full Evaluation Set (Uncapped USD) |
| **Dollar-Scale $R^2$ (Soft)** | **0.4577** | $\pm 0.0756$ | Full Evaluation Set (Uncapped USD) |
| **Dollar-Scale MAE (Soft)** | **$62,152** | $\pm \$12,325$ | Full Evaluation Set (Uncapped USD) |

### 3.2 Quarantined 2020 Holdout Results (One-Shot Evaluation)

Table 4 displays the evaluation on the quarantined 2020 holdout dataset ($N = 27,457$, January 1 to August 29, 2020), trained once on the complete 2018–2019 development corpus ($N = 92,398$ effective fit rows post burn-in; 99,380 total).

#### Table 4: 2020 Holdout Evaluation (One-Shot Generalization)
*Source: `outputs/pipeline_results.json` (`holdout_2020`)*

| Metric Category | Metric Name | Holdout Value | Comparison vs. 2019 Cross-Fold Mean |
|---|---|---|---|
| **Primary Stability** | **Log-Scale $R^2$** | **0.6349** | $-0.0796$ (Macroeconomic/COVID regime shift) |
| | **Log-Scale RMSE** | **1.1483** | $+0.0754$ |
| | **Log-Scale MAE** | **0.8374** | $+0.0538$ |
| **Stage 1 Classifier** | **Accuracy** | **0.9345** | $-0.0059$ |
| | **Precision** | **0.9264** | Stable high precision |
| | **Recall** | **0.8861** | $+0.0217$ (High zero-class identification) |
| | **F1-Score** | **0.9058** | $+0.0414$ |
| | **ROC-AUC** | **0.9678** | $+0.0307$ |
| **Retransformation** | **Duan Factor ($S$)** | **1.5776** | Fit on 2018–2019 residuals |
| **Economic Dollar Space**| **Dollar $R^2$ (Hard)** | **0.5192** | $+0.0650$ vs cross-fold mean |
| | **Dollar MAE (Hard)** | **$51,911** | $-\$9,388$ lower error |
| | **Dollar $R^2$ (Soft)** | **0.5177** | $+0.0600$ vs cross-fold mean |
| | **Dollar MAE (Soft)** | **$52,872** | $-\$9,280$ lower error |

---

## 4. Case Study: Deconstructing the Fold 2 (2019 Q2) Anomaly

In initial cross-validation runs, Fold 2 (2019 Q2) exhibited an apparent performance anomaly: Stage 1 recall dropped to $0.669$ (vs. $0.86\text{--}0.95$ in other folds), and Dollar $R^2$ was $0.201$ (vs. $0.42\text{--}0.54$). We investigated this systematically.

### 4.1 Root Cause 1: Observed Change in Zero-Value Recording Patterns

Decomposing the zero-value rate across time revealed that Fold 2 coincided with an observed change in recording pattern in how the City of Dallas filed building permits.

#### Table 5: Emergence of Zero-Value Building Permits Across Quarters
*Source: `src/diagnostics_fold2_and_parser.py` (Sections 1.1 & 1.2)*

| Quarter | Window Context | Total Building Permits | Zero-Value Building Permits | Zero-Value Rate (%) |
|---|---|---|---|---|
| **2018 Q1** | Training Baseline | 3,953 | 82 | **2.07%** |
| **2018 Q2** | Training Baseline | 4,541 | 105 | **2.31%** |
| **2018 Q3** | Training Baseline | 3,744 | 79 | **2.11%** |
| **2018 Q4** | Training Baseline | 3,798 | 97 | **2.55%** |
| **2019 Q1** | Fold 1 Validation | 4,188 | 108 | **2.58%** |
| **2019 Q2** | **Fold 2 Validation** | **4,356** | **350** | **8.03%** *(3.1x jump)* |
| **2019 Q3** | Fold 3 Validation | 6,184 | 2,235 | **36.14%** *(14.0x jump)* |
| **2019 Q4** | Fold 4 Validation | 4,974 | 2,147 | **43.16%** *(16.7x jump)* |
| **2020** | Holdout Period | 8,997 | 4,073 | **45.27%** |

Throughout 2018 and 2019 Q1 (Fold 2's training data), standard Building permits were $>97.5\%$ positive-value. In 2019 Q2, Dallas began recording zero-value building permits at triple the historical rate—reflecting an observed change in recording pattern. Because Fold 2's training set had never observed building permits with declared zero values, the classifier initially predicted them as positive (false negatives for zero), reducing zero-class recall. By Folds 3 and 4, the training set included Q2/Q3 data, and the model adapted to the newly observed recording patterns.

### 4.2 Root Cause 2: Dollar-Space Denominator Compression

The apparent drop in Fold 2's Dollar $R^2$ ($0.3370$ vs $0.5068$ in Fold 1) is mathematically explained by **variance denominator compression**, rather than higher prediction error.

#### Table 6: Per-Fold Positive-Value Variance and Metric Comparison
*Source: `src/diagnostics_fold2_and_parser.py` (Section 1.5) and `outputs/pipeline_results.json`*

| Fold | Val Quarter | Total Val $N$ | Positive Val $N$ | $\text{Var}(y)$ All Rows | $\text{Var}(y)$ Positive Rows | Max Declared Value | Dollar RMSE (Hard) | Dollar $R^2$ (Hard) | Log-Scale $R^2$ (Primary) |
|---|---|---|---|---|---|---|---|---|---|
| **Fold 1** | 2019 Q1 | 11,334 | 9,659 | $9.370 \times 10^{11}$ | $1.097 \times 10^{12}$ | $\$43,000,000$ | $\$679,805$ | **0.5068** | **0.7422** |
| **Fold 2** | 2019 Q2 | 12,653 | 10,591 | $4.669 \times 10^{11}$ | $5.551 \times 10^{11}$ | $\$21,250,000$ | **$\$556,364$** | **0.3370** | **0.7566** |
| **Fold 3** | 2019 Q3 | 14,890 | 10,239 | $1.807 \times 10^{11}$ | $2.607 \times 10^{11}$ | $\$22,267,046$ | $\$319,678$ | **0.4344** | **0.6693** |
| **Fold 4** | 2019 Q4 | 13,160 | 8,503 | $2.695 \times 10^{11}$ | $4.117 \times 10^{11}$ | $\$15,600,000$ | $\$352,633$ | **0.5386** | **0.6899** |

> **Key Finding:** In Fold 2, the model achieved a **lower Dollar RMSE ($\$556,364$)** than in Fold 1 ($\$679,805$). However, because Fold 1 contained two multi-million commercial projects ($\$43\text{M}$ and $\$38.5\text{M}$), its variance denominator $\sum (y_i - \bar{y})^2$ was twice as large. In **log space**, which is invariant to single-outlier variance inflation, **Fold 2's $R^2$ ($0.7566$) is higher than Fold 1's ($0.7422$)**.

### 4.3 Stage 1 Evaluation Under Recording Pattern Shift (Pre-Shift vs. Post-Shift Split)

To evaluate how much of the Stage 1 classification accuracy and prospective holdout generalization depends on the training window covering the newly observed recording patterns, we split the empirical performance into two temporal regimes and conducted a counterfactual training horizon experiment (`src/eval_stage1_shift.py`).

#### Table 5B: Stage 1 Performance Grouped by Pre-Shift vs. Post-Shift Regimes
*Source: `outputs/stage1_shift_eval_results.json` and `outputs/pipeline_results.json`*

| Evaluation Window | Regime Group | Val $N$ | Val Zero Rate | Stage 1 Acc | Stage 1 Recall | Stage 1 Precision | Stage 1 F1 | Stage 1 ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **Fold 1** (2019 Q1) | **Pre-Shift Baseline** | 11,334 | 14.78% | **0.9793** | **0.8722** | **0.9865** | **0.9256** | **0.9766** |
| **Fold 2** (2019 Q2) | **Shift Transition Onset** | 12,653 | 16.30% | 0.9480 | 0.6969 | 0.9774 | 0.8137 | 0.8535 |
| **Fold 3** (2019 Q3) | **Post-Shift Early** | 14,890 | 31.24% | 0.8984 | 0.7091 | 0.9535 | 0.8134 | 0.9528 |
| **Fold 4** (2019 Q4) | **Post-Shift Established** | 13,160 | 35.39% | 0.9361 | 0.8598 | 0.9551 | 0.9050 | 0.9656 |
| **2020 Holdout** | **Post-Shift Prospective** | 27,457 | 37.45% | 0.9345 | 0.8861 | 0.9264 | 0.9058 | 0.9678 |

#### Table 5C: Counterfactual 2020 Holdout Generalization by Training Window Coverage
*Source: `outputs/stage1_shift_eval_results.json` (`src/eval_stage1_shift.py`)*

| Model Training Window | Shift Coverage in Train | 2020 Overall Acc | 2020 Overall Recall | 2020 Overall F1 | 2020 Overall AUC | 2020 Building Zero Recall | 2020 Building Zero F1 |
|---|---|---|---|---|---|---|---|
| **Pre-Shift Only** (2018 to 2019 Q1) | None (Pre-shift baseline only) | 0.7668 | 0.3515 | 0.5172 | 0.4883 | **0.0264** | **0.0511** |
| **Full Development** (2018 to 2019 Q4) | Complete (Includes 2019 Q2–Q4) | **0.9345** | **0.8861** | **0.9058** | **0.9678** | **0.9402** | **0.9564** |
| **Empirical Delta ($\Delta$)** | Coverage Recovery | $+0.1677$ | $+0.5346$ | $+0.3886$ | $+0.4795$ | **+0.9138** | **+0.9053** |

> **Key Takeaway on Holdout Dependency:** The counterfactual experiment demonstrates that the Stage 1 model's high prospective performance on the 2020 holdout (Recall $0.8861$, ROC-AUC $0.9678$) is critically dependent on the training window extending through 2019 Q2–Q4 to capture the newly observed recording patterns. When trained strictly on pre-shift data, the classifier identifies only $2.6\%$ of zero-value building permits in 2020 (overall recall collapses to $0.3515$), because it was never exposed to building permits declaring $\$0$. Once training data encompasses the emergent patterns, building zero recall jumps to $94.02\%$, establishing that prospective generalization reflects empirical adaptation to updated municipal filing patterns rather than static distributional stability.

---

## 5. Data-Quality Contribution: Permit Type Parser Forensic Audit

### 5.1 Discovery of 13 Unparsed Regulatory Permit Strings

An audit of the initial regex parser in `data_cleaning.py` revealed that **$8.17\%$ of the entire dataset ($N = 10,361$ permits)** failed standard 4-part regex parsing and fell through to a generic fallback rule that assigned:
$$\text{sub\_category} = \text{"General"}, \quad \text{work\_type} = \text{"Other"}$$

Forensic extraction identified that exactly 13 distinct strings accounted for all 10,361 fallback records.

#### Table 7: Complete Inventory of the 13 Fallback Permit Strings
*Source: `diagnostics_fold2_and_parser.py` (Section 2.2)*

| # | Occurrences | Raw `Permit Type` String | Structural Domain Meaning | Re-Parsed Category | Re-Parsed Sub-Category | Re-Parsed Work Type |
|---|---|---|---|---|---|---|
| 1 | 5,171 | `"Electrical Sign New Construction"` | Commercial Electrical Sign | `Electrical Sign` | `Commercial` | `New Construction` |
| 2 | 2,230 | `"Demolition Permit SFD/Duplex"` | Single-Family / Duplex Demolition | `Demolition` | `Single Family` | `Demolition` |
| 3 | 1,016 | `"Sign New Construction"` | Commercial Non-Electrical Sign | `Sign` | `Commercial` | `New Construction` |
| 4 | 977 | `"Electrical Sign Alteration"` | Electrical Sign Alteration | `Electrical Sign` | `Commercial` | `Alteration` |
| 5 | 415 | `"Demolition Permit Commercial"` | Commercial Building Demolition | `Demolition` | `Commercial` | `Demolition` |
| 6 | 245 | `"Supergraphics Alteration"` | Supergraphics Sign Alteration | `Supergraphics` | `Commercial` | `Alteration` |
| 7 | 230 | `"Sign Alteration"` | Commercial Sign Alteration | `Sign` | `Commercial` | `Alteration` |
| 8 | 31 | `"Electrical Sign Reconstruction"` | Electrical Sign Reconstruction | `Electrical Sign` | `Commercial` | `Reconstruction` |
| 9 | 21 | `"Supergraphics New Construction"` | Supergraphics Sign Erection | `Supergraphics` | `Commercial` | `New Construction` |
| 10| 8 | `"Special Purpose Sign New Construction"`| Special Purpose Commercial Sign | `Special Purpose Sign` | `Commercial` | `New Construction` |
| 11| 6 | `"Electrical Sign Renovation"` | Electrical Sign Renovation | `Electrical Sign` | `Commercial` | `Renovation` |
| 12| 6 | `"Sign Reconstruction"` | Sign Reconstruction | `Sign` | `Commercial` | `Reconstruction` |
| 13| 5 | `"Sign Renovation"` | Sign Renovation | `Sign` | `Commercial` | `Renovation` |
| **Total**| **10,361** | *(100% of all fallback records)* | | | | |

### 5.2 The 100% Zero-Value Correlation and Feature Importance Distortion

Crucially, **every single one of these 10,361 permits had a declared value of exactly $\$0$ ($100.0\%$ zero-value rate)**. Because the original parser assigned `work_type = "Other"` exclusively to these fallback records, `work_type_Other` became a deterministic proxy for zero valuation ($V = \$0$).

In the baseline model, `work_type_Other` and `sub_category_General` captured an artificial **$58.5\%$ of all Stage 1 feature importance** ($0.4436$ and $0.1418$ respectively), blinding the tree ensembles to genuine predictors. Implementing the multi-pattern parser in `data_cleaning_v2.py` eliminated fallback records completely and redistributed importance across genuine domain categories.

#### Table 8: Stage 1 Feature Importance Redistribution (v2 Baseline vs. v3.1 Production)
*Source: `outputs/feature_importance_stage1.csv`*

| Rank | v2 Baseline Feature | v2 Importance | v3.1 Production Feature | v3.1 Importance | Category Status |
|---|---|---|---|---|---|
| **1** | `work_type_Other` | **0.4436** | `permit_category_Electrical Sign` | **0.1495** | Genuine Domain Category |
| **2** | `sub_category_General` | **0.1418** | `work_type_Demolition` | **0.1089** | Genuine Domain Work Type |
| **3** | `permit_category_Electrical Sign New Const.` | 0.0925 | `permit_category_Demolition` | **0.0922** | Genuine Domain Category |
| **4** | `permit_category_Barricade` | 0.0504 | `work_type_Alteration` | **0.0812** | Genuine Domain Work Type |
| **5** | `work_type_Alteration` | 0.0380 | `permit_category_Barricade` | **0.0685** | Genuine Domain Category |
| **6** | `permit_category_Demolition Permit SFD/Duplex`| 0.0375 | `permit_category_Building` | **0.0677** | Genuine Domain Category |

---

## 6. Ablation Study: Evaluating the Parser-Fix Effect

To test whether the performance shifts were linked to removing the synthetic proxy, we conducted an ablation study comparing:
1. **v2 Baseline**: Raw dataset with synthetic `work_type_Other` and `sub_category_General`.
2. **Ablation Variant**: Identical v2 data and pipeline, but **dropping only `work_type_Other` and `sub_category_General`** from Stage 1 (no new categories added), evaluated under the strict 60-day burn-in exclusion.
3. **v3.1 Production**: Full multi-pattern parser with domain categories and burn-in exclusion.

#### Table 9: Three-Way Ablation Comparison Across All Folds and 2020 Holdout
*Source: `outputs/ablation_results.json` and `outputs/pipeline_results.json`*

| Evaluation Window | Metric | v2 Baseline | Ablation (Drop Only) | v3.1 Production | Ablation vs. v2 | v3.1 vs. v2 |
|---|---|---|---|---|---|---|
| **Fold 1** (2019 Q1) | Stage 1 Recall | 0.8633 | **0.8167** | **0.8722** | $-0.0466$ | $+0.0089$ |
| | Stage 1 F1 | 0.9207 | 0.8932 | **0.9256** | $-0.0275$ | $+0.0049$ |
| | Stage 1 AUC | **0.9782** | 0.9737 | 0.9766 | $-0.0045$ | $-0.0016$ |
| | Dollar $R^2$ (Hard) | 0.5124 | **0.5184** | 0.5068 | $+0.0060$ | $-0.0560$ |
| **Fold 2** (2019 Q2) | Stage 1 Recall | 0.6688 | **0.6717** | **0.6969** | $+0.0029$ | **+0.0281** |
| | Stage 1 F1 | 0.7932 | 0.7946 | **0.8137** | $+0.0014$ | **+0.0205** |
| | Stage 1 AUC | **0.8541** | 0.8488 | 0.8535 | $-0.0053$ | $-0.0006$ |
| | Dollar $R^2$ (Hard) | 0.2007 | 0.3559 | **0.3370** | $+0.1552$ | **+0.1363** |
| **Fold 3** (2019 Q3) | Stage 1 Recall | 0.6932 | 0.6953 | **0.7091** | $+0.0021$ | $+0.0159$ |
| | Stage 1 F1 | 0.8018 | 0.7994 | **0.8134** | $-0.0024$ | $+0.0116$ |
| | Stage 1 AUC | 0.9515 | 0.9494 | **0.9528** | $-0.0021$ | $+0.0013$ |
| | Dollar $R^2$ (Hard) | 0.4218 | **0.4434** | 0.4344 | $+0.0216$ | $+0.0126$ |
| **Fold 4** (2019 Q4) | Stage 1 Recall | **0.8591** | 0.8557 | 0.8598 | $-0.0034$ | $+0.0007$ |
| | Stage 1 F1 | 0.8945 | 0.9035 | **0.9050** | $+0.0090$ | $+0.0105$ |
| | Stage 1 AUC | 0.9642 | **0.9660** | 0.9656 | $+0.0018$ | $+0.0014$ |
| | Dollar $R^2$ (Hard) | 0.5401 | 0.5244 | **0.5386** | $-0.0157$ | $-0.0015$ |
| **2020 Holdout** | Stage 1 Recall | 0.8852 | **0.8933** | 0.8861 | $+0.0081$ | $+0.0009$ |
| | Stage 1 F1 | 0.8839 | **0.9119** | 0.9058 | $+0.0280$ | $+0.0219$ |
| | Stage 1 AUC | 0.9646 | **0.9685** | 0.9678 | $+0.0039$ | $+0.0032$ |
| | Dollar $R^2$ (Hard) | 0.5278 | 0.5052 | **0.5192** | $-0.0226$ | $-0.0086$ |

### 6.1 Evaluation of the Splitting-Budget Redistribution Hypothesis

The ablation empirically evaluates this mechanism:
1. **Classifier Split Redistribution**: Removing the synthetic shortcut features (`work_type_Other` and `sub_category_General`) encourages the tree ensembles to distribute their splitting budget across true contextual features (geographic location, contractor history, land use, and structural permit categories). In Fold 2, classification recall rises from $0.6688$ to $0.6717$ in the ablation and $0.6969$ in v3.1, while F1 improves from $0.7932$ to $0.8137$.
2. **Domain Categories and Valuation Fidelity**: Supplying explicit regulatory categories (`Electrical Sign`, `Demolition`, etc.) provides critical structural discriminators. An empirical confusion audit on all 10,361 Sign and Demolition records (`src/ablation_stage1_parser_effect.py`) confirmed that both the ablation model and v3.1 classify $\ge 99.9\%$ of Sign/Demolition permits as $\$0$ under hard thresholding ($0$ misclassifications in Folds 1 and 2; only $1$ in Fold 3 and Fold 4). Under continuous soft probability weighting, v3.1's calibrated class probabilities substantially reduce dollar leakage on zero-value regulatory filings: soft MAE on Sign/Demolition filings drops from $\$2,175$ to $\$553$ in Fold 1, and from $\$1,076$ to $\$216$ in Fold 2.
3. **Mixed Dollar-Scale Impact**: Crucially, while Stage 1 classification metrics and zero-leakage suppression improve consistently, the downstream dollar-scale effect is mixed across evaluation splits rather than a uniform improvement. In Fold 2, Dollar $R^2$ recovers substantially ($0.2007 \to 0.3370$, $+0.1363$), and Fold 3 exhibits modest gains ($0.4218 \to 0.4344$, $+0.0126$). However, in Fold 1 ($0.5124 \to 0.5068$, $-0.0056$), Fold 4 ($0.5401 \to 0.5386$, $-0.0015$), and the 2020 holdout ($0.5278 \to 0.5192$, $-0.0086$), the dollar-scale impact is essentially flat or slightly lower. The primary justification for the parser fix is therefore architectural integrity and domain interpretability—eliminating artificial feature shortcuts and restoring valid regulatory categories—rather than an unequivocal dollar-scale metric gain across every fold.

---

## 7. Sensitivity Analysis: Primary Uncapped vs. p99-Capped Specification

To validate the decision to adopt the uncapped model as the primary specification, we evaluated both the primary uncapped model and a p99-capped sensitivity variant against the true uncapped validation values.

#### Table 10: Empirical Sensitivity Comparison (Primary Uncapped vs. p99-Capped)
*Source: `outputs/pipeline_results.json` (`folds` vs. `sensitivity_folds`)*

| Evaluation Window | Model Specification | Primary Log $R^2$ | Log RMSE | Log MAE | Dollar $R^2$ (Hard) | Dollar MAE (Hard) | Dollar RMSE (Hard) |
|---|---|---|---|---|---|---|---|
| **Fold 1** | Primary Uncapped | **0.7422** | **1.0796** | **0.7953** | **0.5068** | **$74,955** | **$679,805** |
| | Sensitivity (p99-Capped) | 0.7423 | 1.0793 | 0.7933 | 0.1815 | $81,662 | $875,727 |
| **Fold 2** | Primary Uncapped | **0.7566** | **1.0042** | **0.7162** | **0.3370** | $72,340 | **$556,364** |
| | Sensitivity (p99-Capped) | 0.7554 | 1.0066 | 0.7218 | 0.2978 | **$69,099** | $572,576 |
| **Fold 3** | Primary Uncapped | **0.6693** | **1.1040** | **0.7895** | **0.4344** | $50,216 | **$319,678** |
| | Sensitivity (p99-Capped) | 0.6631 | 1.1142 | 0.7966 | 0.3942 | **$46,358** | $330,839 |
| **Fold 4** | Primary Uncapped | **0.6899** | **1.1038** | **0.8336** | **0.5386** | **$47,685** | **$352,633** |
| | Sensitivity (p99-Capped) | 0.6905 | 1.1029 | 0.8351 | 0.4835 | $51,013 | $373,079 |

> **Methodological Takeaway:** In Fold 1, training on a p99-capped target was associated with a Dollar $R^2$ collapse from $0.5068$ to $0.1815$ and an RMSE increase of over $\$195,000$. Artificially truncating the training target introduces severe negative bias when predicting high-value commercial construction projects. The uncapped model provides superior fidelity across the full distribution.

---

## 7.1 Comparative Model Baselines

To rigorously establish whether the two-stage hurdle architecture provides tangible benefits over simpler or alternative single-stage formulations, we benchmarked the primary hurdle pipeline against three comparative models evaluated across the exact same forward-chaining folds and 2020 holdout:
1. **Naive Median Baseline**: Unconditionally predicts the historical training set median positive value for all permits.
2. **Conventional Single-Stage Log-Linear (RidgeCV)**: Fits a cross-validated regularized linear model (`RidgeCV` with $\alpha \in [0.1, 10^4]$ and feature standardization via `StandardScaler`) on $\log(1 + \text{Value})$ across all permits without zero-inflation separation. In accordance with standard econometric specification, raw linear `Area` is excluded to prevent exponential dollar scaling, retaining `log_area` as the elasticity regressor. Predictions are retransformed to dollar scale using empirical Duan smearing.
3. **Non-Hurdle Tweedie XGBoost**: A single-stage compound Poisson-Gamma regressor (`reg:tweedie`, variance power $= 1.5$) fit directly on uncapped dollar values across all permits.

#### Table 11: Comparative Baseline Model Evaluation Across Folds and 2020 Holdout
*Source: `outputs/baseline_comparison_results.json` and `outputs/pipeline_results.json`*

| Evaluation Window | Model Architecture | Log $R^2$ (Positives) | Dollar $R^2$ | Dollar MAE | Dollar RMSE |
|---|---|---|---|---|---|
| **Fold 1** (2019 Q1) | Naive Median Baseline | — | $-0.0134$ | $\$117,234$ | $\$974,446$ |
| | Conventional Log-Linear (RidgeCV) | 0.4053 | $-7.4142$ | $\$951,925$ | $\$2,807,800$ |
| | Non-Hurdle Tweedie XGBoost | — | 0.3111 | $\$79,128$ | $\$803,435$ |
| | **Two-Stage Hurdle (Primary)** | **0.7422** | **0.5068** | **$74,955** | **$679,805** |
| **Fold 2** (2019 Q2) | Naive Median Baseline | — | $-0.0218$ | $\$105,374$ | $\$690,670$ |
| | Conventional Log-Linear (RidgeCV) | 0.4752 | $-26.2190$ | $\$1,217,358$ | $\$3,564,775$ |
| | Non-Hurdle Tweedie XGBoost | — | 0.3206 | **$68,718** | $\$563,203$ |
| | **Two-Stage Hurdle (Primary)** | **0.7566** | **0.3370** | $\$72,340$ | **$556,364** |
| **Fold 3** (2019 Q3) | Naive Median Baseline | — | $-0.0134$ | $\$55,137$ | $\$427,881$ |
| | Conventional Log-Linear (RidgeCV) | 0.4848 | $-14.8619$ | $\$541,613$ | $\$1,692,842$ |
| | Non-Hurdle Tweedie XGBoost | — | **0.4762** | **$43,437** | **$307,618** |
| | **Two-Stage Hurdle (Primary)** | **0.6693** | 0.4344 | $\$50,216$ | $\$319,678$ |
| **Fold 4** (2019 Q4) | Naive Median Baseline | — | $-0.0200$ | $\$80,174$ | $\$524,297$ |
| | Conventional Log-Linear (RidgeCV) | $-0.1257$ | 0.1870 | $\$116,580$ | $\$468,060$ |
| | Non-Hurdle Tweedie XGBoost | — | 0.4472 | $\$53,857$ | $\$385,968$ |
| | **Two-Stage Hurdle (Primary)** | **0.6899** | **0.5386** | **$47,685** | **$352,633** |
| **2020 Holdout** | Naive Median Baseline | — | $-0.0107$ | $\$77,749$ | $\$696,387$ |
| | Conventional Log-Linear (RidgeCV) | $-1.7021$ | $-0.3138$ | $\$147,417$ | $\$793,975$ |
| | Non-Hurdle Tweedie XGBoost | — | 0.3236 | $\$55,411$ | $\$569,691$ |
| | **Two-Stage Hurdle (Primary)** | **0.6349** | **0.5192** | **$51,911** | **$480,333** |

> **Key Comparative Findings:**
> 1. **Log-Space Competence vs. Dollar Retransformation Degradation in Log-Linear Models:** In native log space on positive permits, regularized log-linear Ridge regression demonstrates reasonable explanatory power across the first three folds ($R^2_{\log} \in [0.405, 0.485]$). However, when predictions are retransformed to the dollar scale across the entire permit population via Duan smearing, performance degrades substantially (Dollar $R^2 \le 0$ across Folds 1–3 and the 2020 Holdout). This illustrates the fundamental instability of applying single-stage log transformations to zero-inflated distributions without an explicit probability hurdle.
> 2. **Tweedie Under-Fitting in Upper Tails:** While the single-stage Tweedie XGBoost regressor natively accommodates exact zeros without retransformation bias, it exhibits reduced upper-tail flexibility, yielding a 2020 Holdout Dollar $R^2$ of $0.3236$ (compared to **$0.5192$** for the Two-Stage Hurdle model).
> 3. **Architectural Trade-Offs Between Hurdle and Compound Models:** Decoupling occurrence probability from conditional valuation is not unconditionally superior across all metrics and time horizons, but highlights a distinct practical trade-off:
>    - **Hurdle Model Wins**: Achieves higher overall dollar $R^2$ across cross-validation (cross-fold mean $0.4542$ vs. $0.3888$) and substantially better prospective holdout generalization on 2020 ($0.5192$ vs. $0.3236$). It also maintains lower Dollar RMSE on Fold 1 ($\$679,805$ vs. $\$803,435$), Fold 2 ($\$556,364$ vs. $\$563,203$), Fold 4 ($\$352,633$ vs. $\$385,968$), and the 2020 Holdout ($\$480,333$ vs. $\$569,691$).
>    - **Tweedie Model Wins**: Wins across all three dollar metrics in Fold 3 (higher Dollar $R^2$ of $0.4762$ vs. $0.4344$, lower MAE of $\$43,437$ vs. $\$50,216$, and lower RMSE of $\$307,618$ vs. $\$319,678$), and achieves lower dollar MAE in Fold 2 ($\$68,718$ vs. $\$72,340$).
>    - **Practical Implication**: The Two-Stage Hurdle model is preferable when long-term prospective holdout stability, upper-tail commercial valuation fidelity, and overall variance explained ($R^2$) are prioritized, while single-stage Tweedie offers strong competition during transitional quarters where median/MAE absolute error is the primary operational benchmark.

---

## 8. Limitations and Threats to Validity

1. **Burn-In Window Sample Size Reduction**:  
   Excluding the first 60 days of the training corpus ($6,982$ records) eliminated cold-start target leakage from model parameter fitting (`df_train_fit`). However, it reduced the effective training set size (e.g., from $47,343$ to $40,361$ in Fold 1, and $99,380$ to $92,398$ in the 2020 Holdout).
2. **Sample Size Constraints under 2020 Holdout Quarantine**:  
   Quarantining the entire 2020 calendar year ($27,457$ records) for clean out-of-sample evaluation restricted the development corpus to 2018–2019 ($99,380$ total rows), limiting the forward-chaining schedule to 4 quarterly folds.
3. **Forward-Chaining Fold-Length Asymmetry**:  
   Under forward-chaining cross-validation, the effective training horizon expands from 10 months in Fold 1 ($N = 40,361$) to 19 months in Fold 4 ($N = 79,238$). Variations in performance across folds reflect both sample growth and temporal proximity to the validation window.
4. **Self-Reported Declared Valuations**:  
   Permit values represent developer-declared construction estimates at the time of filing, which may diverge from final realized construction expenditures or assessed market values.

---

## 9. Data Source & Reproducibility Traceability Registry

Every table and figure in this manuscript is programmatically reproducible from source code and outputs in the repository:

| Manuscript Section / Table | Source File / Artifact | Generation Script |
|---|---|---|
| **Table 1: Methodological Corrections** | [`docs/feature_audit.md`](docs/feature_audit.md) | Architectural audit |
| **Tables 2, 3, 4, 10: Performance & Sensitivity** | [`outputs/pipeline_results.json`](outputs/pipeline_results.json) | [`src/modeling_pipeline.py`](src/modeling_pipeline.py) |
| **Tables 5, 6, 7: Diagnostics, Regimes, Fallback** | [`docs/fold2_and_parser_findings.md`](docs/fold2_and_parser_findings.md) | [`src/diagnostics_fold2_and_parser.py`](src/diagnostics_fold2_and_parser.py) |
| **Tables 5B, 5C: Stage 1 Shift & Counterfactual** | [`outputs/stage1_shift_eval_results.json`](outputs/stage1_shift_eval_results.json) | [`src/eval_stage1_shift.py`](src/eval_stage1_shift.py) |
| **Table 8: Feature Importance Redistribution** | [`outputs/feature_importance_stage1.csv`](outputs/feature_importance_stage1.csv) | [`src/modeling_pipeline.py`](src/modeling_pipeline.py) |
| **Table 9: Three-Way Ablation Study** | [`outputs/ablation_results.json`](outputs/ablation_results.json) | [`src/ablation_stage1_parser_effect.py`](src/ablation_stage1_parser_effect.py) |
| **Section 6.1: Sign/Demo Confusion Analysis** | `outputs/ablation_results.json` | [`src/ablation_stage1_parser_effect.py`](src/ablation_stage1_parser_effect.py) |
| **Table 11: Comparative Baseline Models** | [`outputs/baseline_comparison_results.json`](outputs/baseline_comparison_results.json) | [`src/baseline_comparisons.py`](src/baseline_comparisons.py) |

---

## 10. References

1. **Cragg, J. G.** (1971). *Some statistical models for limited dependent variables with application to the demand for durable goods*. Econometrica: Journal of the Econometric Society, 829-844.
2. **Duan, N.** (1983). *Smearing estimate: a nonparametric retransformation method*. Journal of the American Statistical Association, 78(383), 605-610.
3. **Chen, T., & Guestrin, C.** (2016). *XGBoost: A scalable tree boosting system*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794).
4. **Miceli, T. J.** (2002). *The economics of building codes*. Journal of Real Estate Finance and Economics, 25(1), 79-95.
5. **Mullahy, J.** (1986). *Specification and testing of some modified count data models*. Journal of Econometrics, 33(3), 341-365.
6. **Jorgensen, B.** (1997). *The theory of dispersion models*. CRC Press.
