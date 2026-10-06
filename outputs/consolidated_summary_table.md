# Consolidated Model Evaluation and Baseline Benchmark Metrics

All metrics evaluated under strict chronological forward-chaining splits with initial 60-day burn-in window (N=6,982 records) excluded from training model fitting. Validation-time ZIP encoding maps are built from the full historical fold prior to validation window start.

## 1. Primary Two-Stage Hurdle Model and Sensitivity Variant (p99-Capped)

| Split / Fold | Model Variant | Train N (Fit) | Val N | S1 Acc | S1 F1 | S1 AUC | Log R2 | Log RMSE | Log MAE | Duan Factor | Dollar R2 (Hard) | Dollar MAE (Hard) | Dollar R2 (Soft) | Dollar MAE (Soft) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | Two-Stage Hurdle (Primary Uncapped) | 40361 | 11334 | 0.9793 | 0.9256 | 0.9766 | 0.7422 | 1.0796 | 0.7953 | 1.4645 | 0.5068 | $74,955.27 | 0.5042 | $75,972.41 |
| Fold 2 | Two-Stage Hurdle (Primary Uncapped) | 51695 | 12653 | 0.9480 | 0.8137 | 0.8535 | 0.7566 | 1.0042 | 0.7162 | 1.4954 | 0.3370 | $72,340.22 | 0.3400 | $72,875.00 |
| Fold 3 | Two-Stage Hurdle (Primary Uncapped) | 64348 | 14890 | 0.8984 | 0.8134 | 0.9528 | 0.6693 | 1.1040 | 0.7895 | 1.5179 | 0.4344 | $50,216.37 | 0.4470 | $50,364.02 |
| Fold 4 | Two-Stage Hurdle (Primary Uncapped) | 79238 | 13160 | 0.9361 | 0.9050 | 0.9656 | 0.6899 | 1.1038 | 0.8336 | 1.5504 | 0.5386 | $47,684.82 | 0.5396 | $49,396.73 |
| Cross-Fold Mean +- Std | Two-Stage Hurdle (Primary Uncapped) | — | — | 0.9404 +- 0.0289 | 0.8644 +- 0.0514 | 0.9371 +- 0.0490 | 0.7145 +- 0.0360 | 1.0729 +- 0.0409 | 0.7836 +- 0.0424 | 1.5071 +- 0.0314 | 0.4542 +- 0.0775 | $61,299.17 +- $12,415.44 | 0.4577 +- 0.0756 | $62,152.04 +- $12,325.18 |
| 2020 Holdout | Two-Stage Hurdle (Primary Uncapped) | 92398 | 27457 | 0.9345 | 0.9058 | 0.9678 | 0.6349 | 1.1483 | 0.8374 | 1.5776 | 0.5192 | $51,910.89 | 0.5177 | $52,872.40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fold 1 | Sensitivity Variant (p99 Capped Stage 2) | 40361 | 11334 | 0.9793 | 0.9256 | 0.9766 | 0.7423 | 1.0793 | 0.7933 | 1.4546 | 0.1815 | $81,661.79 | 0.1800 | $82,703.89 |
| Fold 2 | Sensitivity Variant (p99 Capped Stage 2) | 51695 | 12653 | 0.9480 | 0.8137 | 0.8535 | 0.7554 | 1.0066 | 0.7218 | 1.4924 | 0.2978 | $69,099.23 | 0.2943 | $70,002.96 |
| Fold 3 | Sensitivity Variant (p99 Capped Stage 2) | 64348 | 14890 | 0.8984 | 0.8134 | 0.9528 | 0.6631 | 1.1142 | 0.7966 | 1.5097 | 0.3942 | $46,358.25 | 0.3988 | $46,723.70 |
| Fold 4 | Sensitivity Variant (p99 Capped Stage 2) | 79238 | 13160 | 0.9361 | 0.9050 | 0.9656 | 0.6905 | 1.1029 | 0.8351 | 1.5345 | 0.4835 | $51,013.36 | 0.4826 | $52,924.50 |
| Cross-Fold Mean +- Std | Sensitivity Variant (p99 Capped Stage 2) | — | — | 0.9404 +- 0.0289 | 0.8644 +- 0.0514 | 0.9371 +- 0.0490 | 0.7128 +- 0.0376 | 1.0757 +- 0.0418 | 0.7867 +- 0.0409 | 1.4978 +- 0.0291 | 0.3392 +- 0.1123 | $62,033.16 +- $14,162.90 | 0.3389 +- 0.1134 | $63,088.77 +- $14,174.69 |

## 2. Comparative Baselines vs. Primary Two-Stage Hurdle Model

| Split / Fold | Model Specification | Log R2 | Log RMSE | Log MAE | Duan Factor | Dollar R2 | Dollar MAE | Dollar RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | **Two-Stage Hurdle (Primary)** | 0.7422 | 1.0796 | 0.7953 | 1.4645 | **0.5068** | **$74,955.27** | $679,804.62 |
| Fold 1 | Naive Constant Positive Median | — | — | — | — | -0.0134 | $117,234.29 | $974,446.09 |
| Fold 1 | Log-Linear Ridge (CV-scaled, log-area) | 0.4053 | 1.6397 | 1.2371 | 45.9002 | -7.4142 | $951,925.04 | $2,807,800.02 |
| Fold 1 | Non-Hurdle Tweedie XGBoost | — | — | — | — | 0.3111 | $79,128.33 | $803,435.04 |
| Fold 2 | **Two-Stage Hurdle (Primary)** | 0.7566 | 1.0042 | 0.7162 | 1.4954 | **0.3370** | **$72,340.22** | $556,364.41 |
| Fold 2 | Naive Constant Positive Median | — | — | — | — | -0.0218 | $105,374.09 | $690,669.92 |
| Fold 2 | Log-Linear Ridge (CV-scaled, log-area) | 0.4752 | 1.4744 | 1.0857 | 32.8654 | -26.2190 | $1,217,357.87 | $3,564,775.50 |
| Fold 2 | Non-Hurdle Tweedie XGBoost | — | — | — | — | 0.3206 | $68,717.73 | $563,202.80 |
| Fold 3 | **Two-Stage Hurdle (Primary)** | 0.6693 | 1.1040 | 0.7895 | 1.5179 | **0.4344** | **$50,216.37** | $319,677.51 |
| Fold 3 | Naive Constant Positive Median | — | — | — | — | -0.0134 | $55,136.68 | $427,880.62 |
| Fold 3 | Log-Linear Ridge (CV-scaled, log-area) | 0.4848 | 1.3778 | 0.9986 | 25.0428 | -14.8619 | $541,612.58 | $1,692,841.70 |
| Fold 3 | Non-Hurdle Tweedie XGBoost | — | — | — | — | 0.4762 | $43,437.36 | $307,617.97 |
| Fold 4 | **Two-Stage Hurdle (Primary)** | 0.6899 | 1.1038 | 0.8336 | 1.5504 | **0.5386** | **$47,684.82** | $352,633.15 |
| Fold 4 | Naive Constant Positive Median | — | — | — | — | -0.0200 | $80,174.17 | $524,297.19 |
| Fold 4 | Log-Linear Ridge (CV-scaled, log-area) | -0.1257 | 2.1032 | 1.7103 | 22.7857 | 0.1870 | $116,579.82 | $468,059.81 |
| Fold 4 | Non-Hurdle Tweedie XGBoost | — | — | — | — | 0.4472 | $53,857.45 | $385,967.93 |
| 2020 Holdout | **Two-Stage Hurdle (Primary)** | 0.6349 | 1.1483 | 0.8374 | 1.5776 | **0.5192** | **$51,910.89** | $480,333.06 |
| 2020 Holdout | Naive Constant Positive Median | — | — | — | — | -0.0107 | $77,748.64 | $696,387.46 |
| 2020 Holdout | Log-Linear Ridge (CV-scaled, log-area) | -1.7021 | 3.1239 | 2.6677 | 33.2388 | -0.3138 | $147,416.76 | $793,975.35 |
| 2020 Holdout | Non-Hurdle Tweedie XGBoost | — | — | — | — | 0.3236 | $55,411.30 | $569,691.13 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cross-Fold Mean +- Std | **Two-Stage Hurdle (Primary)** | 0.7145 +- 0.0360 | 1.0729 +- 0.0409 | 0.7836 +- 0.0424 | 1.5071 +- 0.0314 | **0.4542 +- 0.0775** | **$61,299.17 +- $12,415.44** | $477,119.92 +- $148,025.21 |
| Cross-Fold Mean +- Std | Naive Constant Positive Median | — | — | — | — | -0.0172 +- 0.0038 | $89,479.81 +- $23,921.67 | $654,323.46 +- $207,354.02 |
| Cross-Fold Mean +- Std | Log-Linear Ridge (CV-scaled, log-area) | 0.3099 +- 0.2534 | 1.6488 +- 0.2786 | 1.2579 +- 0.2748 | 31.6485 +- 9.0385 | -12.0770 +- 9.7455 | $706,868.83 +- $417,253.63 | $2,133,369.26 +- $1,169,519.21 |
| Cross-Fold Mean +- Std | Non-Hurdle Tweedie XGBoost | — | — | — | — | 0.3888 +- 0.0737 | $61,285.22 +- $13,668.73 | $515,055.94 +- $190,509.09 |