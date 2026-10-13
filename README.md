# Dallas Building Permit Value Research

Research code, cleaned corpus and result files for a leakage-free two-stage hurdle
model of **declared building-permit value**, built on City of Dallas permit records
issued between January 2018 and August 2020.

The pipeline enforces chronological feature construction: historical count features
use only permits issued before each prediction date, stage-specific target encodings
are frozen at every evaluation boundary, a 60-day cold-start window is retained for
feature construction but excluded from model fitting, validation is forward-chaining
across four 2019 quarters, and the 2020 period is held out and evaluated once.

## Repository structure

```text
Dallas-Permit-Predictor-Research/
├── README.md                             # Project overview and navigation guide
├── REPRODUCIBILITY.md                    # Step-by-step reproduction commands
├── requirements.txt                      # Pinned dependencies for the pipeline
├── CITATION.cff                          # How to cite this repository
├── LICENSE                               # MIT (code) and CC BY 4.0 (docs, outputs)
├── data/
│   └── Building-Permits.csv              # Raw Dallas permit extract (2018–2020)
├── src/
│   ├── data_cleaning.py                  # Cleaning and composite permit-type parser
│   ├── modeling_pipeline.py              # Two-stage hurdle pipeline, 4 folds + holdout
│   ├── ablation_stage1_parser_effect.py  # Stage 1 ablation isolating the parser fix
│   ├── diagnostics_fold2_and_parser.py   # Fold 2 and parser fallback diagnostics
│   ├── baseline_comparisons.py           # Naive, RidgeCV and Tweedie XGBoost comparators
│   └── eval_stage1_shift.py              # Pre/post recording-shift Stage 1 evaluation
├── outputs/
│   ├── permits_research_cleaned.csv      # Cleaned corpus (126,837 rows)
│   ├── pipeline_results.json             # Cross-validation and 2020 holdout metrics
│   ├── ablation_results.json             # Stage 1 ablation metrics
│   ├── baseline_comparison_results.json  # Comparator metrics
│   ├── stage1_shift_eval_results.json    # Shift and counterfactual metrics
│   ├── feature_importance_stage1.csv     # Stage 1 feature importance
│   ├── feature_importance_stage2.csv     # Stage 2 feature importance
│   └── consolidated_summary_table.md     # Consolidated empirical metrics
├── docs/
│   ├── feature_audit.md                  # Audit identifying the baseline flaws
│   └── fold2_and_parser_findings.md      # Root-cause diagnostic findings
└── tests/
    └── test_manuscript_verification.py   # Verifies manuscript figures against results
```

## Data

The raw extract (`data/Building-Permits.csv`, approximately 31 MB, 126,837 records)
comes from the City of Dallas Open Data portal and covers 2 January 2018 to
29 August 2020. It is committed to this repository unmodified so the pipeline runs
end to end without an external download.

## Reproducing the results

`REPRODUCIBILITY.md` gives the environment, the exact commands, and the order in
which the four scripts run. The test suite in `tests/` checks that the figures
reported in the manuscript match the generated result files:

```bash
pip install -r requirements.txt
python -m pytest tests/
```

## Licence

Code under `src/` and `tests/` is released under the MIT Licence. Documentation and
derived outputs under `docs/` and `outputs/` are released under CC BY 4.0. The raw
permit extract in `data/` remains City of Dallas open data and is redistributed
unmodified under their terms. See `LICENSE` for details.

## Citation

If you use this software or the derived data, please cite the archived release and
the associated article. See `CITATION.cff`, or use the "Cite this repository" button
on the GitHub page.
