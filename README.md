# Dallas Building Permit Value Research

This repository contains the econometric and machine learning research codebase for predicting municipal building permit declared valuations using an empirical two-stage hurdle model with continuous expanding features and cold-start burn-in exclusion.

## Repository Structure

```text
Dallas-Permit-Value-Research/
├── README.md                          # Project overview and navigation guide
├── requirements.txt                   # Minimal pinned dependencies for the research pipeline
├── REPRODUCIBILITY.md                 # Step-by-step reproduction instructions and commands
├── data/
│   └── Building-Permits.csv           # Raw Dallas municipal building permit dataset (2018–2020)
├── src/
│   ├── data_cleaning.py               # Minimal cleaning and composite permit type parser
│   ├── modeling_pipeline.py           # Primary two-stage hurdle pipeline (leakage-free, 4 folds + holdout)
│   ├── ablation_stage1_parser_effect.py # Stage 1 ablation isolating synthetic feature removal
│   ├── diagnostics_fold2_and_parser.py  # Forensic diagnostics on Fold 2 regime shift & parser fallbacks
│   └── baseline_comparisons.py        # Benchmark models (Naive Median, RidgeCV, Tweedie XGBoost)
├── outputs/
│   ├── permits_research_cleaned.csv   # Cleaned research dataset (126,837 rows)
│   ├── pipeline_results.json          # Primary cross-validation and 2020 holdout metrics
│   ├── ablation_results.json          # Stage 1 ablation metrics
│   ├── baseline_comparison_results.json # Comparative baseline model metrics
│   ├── feature_importance_stage1.csv  # Stage 1 classifier feature importance
│   ├── feature_importance_stage2.csv  # Stage 2 regressor feature importance
│   └── consolidated_summary_table.md  # Consolidated table of all empirical metrics
└── docs/
    ├── research_paper_draft.md        # Academic research manuscript draft
    ├── feature_audit.md               # Methodological audit identifying baseline flaws
    └── fold2_and_parser_findings.md   # Root-cause diagnostic findings report
```

## Data Access

The raw dataset (`data/Building-Permits.csv`, ~31 MB, 126,837 records) is sourced from the City of Dallas Open Data portal spanning January 2018 through August 2020. If not already present in `data/`, place the CSV file in `data/Building-Permits.csv` or track it via Git LFS.

## How to Reproduce

For end-to-end instructions and exact shell commands to execute the pipeline from raw CSV to final tables and figures, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
