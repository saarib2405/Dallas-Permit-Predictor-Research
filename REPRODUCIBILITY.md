# Dallas Building Permit Value Research: Reproducibility Guide

This document specifies the exact steps, environment configurations, and shell commands required to reproduce all empirical findings, cross-validation metrics, ablation experiments, and baseline comparisons reported in the research manuscript.

---

## 1. Environment & Dependencies

- **Python Version**: Python 3.12+ (tested on Windows 64-bit and Linux x86_64)
- **Dependency Installation**:
  ```bash
  pip install -r requirements.txt
  ```

### Pinned Dependencies (`requirements.txt`)
```text
pandas==3.0.0
numpy==2.4.3
xgboost==3.2.0
scikit-learn==1.8.0
scipy==1.17.0
```

---

## 2. Step-by-Step Reproduction Workflow

Execute all commands from the repository root:

### Step 2.1: Data Cleaning & Permit Parsing
Reads the raw municipal permit records (`data/Building-Permits.csv`), parses the composite permit types using domain regex rules, and produces the un-leaked research dataset:

```bash
python src/data_cleaning.py
```

- **Input**: `data/Building-Permits.csv`
- **Output**: `outputs/permits_research_cleaned.csv` ($126,837$ rows, $0$ generic fallbacks)

---

### Step 2.2: Forward-Chaining Hurdle Pipeline (Primary Model)
Executes the primary Two-Stage Hurdle model across 4 quarterly forward-chaining folds and the quarantined 2020 holdout split:

```bash
python src/modeling_pipeline.py
```

- **Methodological Guarantees**:
  - Continuous expanding frequency counts strictly prior to each filing date.
  - Initial 60-day cold-start burn-in strictly excluded from training fitting sets (`df_train_fit`), preventing target and frequency leakage.
  - Validation-time empirical Bayes target encodings built from the full historical fold prior to validation window.
  - Retransformation bias corrected via fold-specific empirical Duan smearing factors.
- **Outputs**:
  - `outputs/pipeline_results.json`
  - `outputs/feature_importance_stage1.csv`
  - `outputs/feature_importance_stage2.csv`

---

### Step 2.3: Stage 1 Ablation Experiment
Evaluates whether dropping synthetic fallback features (`work_type_Other`, `sub_category_General`) alone accounts for the Stage 1 performance recovery:

```bash
python src/ablation_stage1_parser_effect.py
```

- **Input**: `outputs/permits_research_cleaned.csv`
- **Output**: `outputs/ablation_results.json`

---

### Step 2.4: Forensic Diagnostics (Fold 2 & Fallback Strings Audit)
Reproduces the empirical zero-rate regime shift audit, volume diagnostics, and the positive-value variance denominator compression analysis:

```bash
python src/diagnostics_fold2_and_parser.py
```

- **Outputs**: Detailed console diagnostic tables matching Sections 1.5, 5, and 6 in `docs/fold2_and_parser_findings.md`.

---

### Step 2.5: Comparative Model Baselines
Benchmarks the primary Two-Stage Hurdle model against three comparator architectures (Naive Positive Median, Standardized Log-Linear `RidgeCV`, and Non-Hurdle Tweedie XGBoost):

```bash
python src/baseline_comparisons.py
```

- **Input**: `outputs/permits_research_cleaned.csv`
- **Output**: `outputs/baseline_comparison_results.json`

---

## 3. Results Verification & Traceability

Every table in `docs/research_paper_draft.md` traces directly to the generated output files:

| Manuscript Table | Source Output File | Generating Script |
| :--- | :--- | :--- |
| **Table 1** (Methodological Audit) | `docs/feature_audit.md` | Architectural review |
| **Tables 2, 3, 4** (Validation & Holdout) | `outputs/pipeline_results.json` | `src/modeling_pipeline.py` |
| **Table 5** (Zero-Rate Regime Shift) | `docs/fold2_and_parser_findings.md` | `src/diagnostics_fold2_and_parser.py` |
| **Table 6** (Variance / Denominator) | `docs/fold2_and_parser_findings.md` | `src/diagnostics_fold2_and_parser.py` |
| **Table 7** (Fallback Strings Audit) | `docs/fold2_and_parser_findings.md` | `src/diagnostics_fold2_and_parser.py` |
| **Table 8** (Parser Effect Across Folds) | `outputs/pipeline_results.json` | `src/modeling_pipeline.py` |
| **Table 9** (Stage 1 Ablation Study) | `outputs/ablation_results.json` | `src/ablation_stage1_parser_effect.py` |
| **Table 10** (Feature Importances) | `outputs/feature_importance_stage1.csv` | `src/modeling_pipeline.py` |
| **Table 11** (Baseline Comparisons) | `outputs/baseline_comparison_results.json` | `src/baseline_comparisons.py` |
| **Consolidated Metrics Table** | `outputs/consolidated_summary_table.md` | Consolidated reporting script |
