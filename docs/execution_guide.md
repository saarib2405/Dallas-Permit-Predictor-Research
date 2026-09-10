# Dallas Building Permits — ML Project Execution Guide
**Course:** ASDS 6302 Advanced Machine Learning | University of Texas at Arlington  
**Goal:** Predict declared construction value of Dallas building permits, cluster permit archetypes, and deploy an AI agent for interactive queries.  
**Stack:** Python (Colab), XGBoost, SHAP, K-Means, Flask, n8n Cloud  
**Data Source:** Dallas OpenData Socrata API — live government data, no pre-packaged file

---

## How to Read This Guide

This is a single-person, top-to-bottom execution plan. Work through each phase in order. Each phase has a clear output that the next phase depends on. Do not start Phase 2 until Phase 1's output exists. The project has three logical workstreams — EDA, ML Modeling, and Agent Deployment — but they are written here as a sequential flow for one person executing the full pipeline.

---

## Phase 1 — Data Pull and EDA

**Goal:** Pull the raw dataset from the Dallas Socrata API, explore it, clean it, and produce a clean CSV ready for modeling.

### Step 1 — Pull data from the Socrata API

Use the `requests` library (or `sodapy`) to call the Dallas OpenData API endpoint for building permits. The dataset ID is `e7gq-4sah`. Pull all available records using pagination — the default Socrata limit is 1000 rows per call, so loop with `$offset` increments until you have exhausted the dataset. Store the raw pull as a pandas DataFrame. Print the shape, column names, and a sample of five rows to confirm the pull worked.

Target columns to keep: `permit_number`, `permit_type`, `permit_subtype`, `issue_date`, `council_district`, `contractor_name`, `declared_value`, `square_footage`, `work_description`, `occupancy_type`, `address`, `zip_code`.

**Output of this step:** A raw DataFrame saved as `raw_permits.csv`.

### Step 2 — Exploratory Data Analysis

Before cleaning anything, run a full EDA pass on the raw data. Check the following: total row count and column count, missing value counts per column (express as both count and percentage of total rows), data types for every column, distribution of `declared_value` using a histogram and log-scale histogram, distribution of `permit_type` using a value counts bar chart, distribution of `issue_date` by year and month, and a correlation heatmap of numeric columns.

Write observations as markdown comments in the notebook. Note which columns have more than 20% missing values, which columns will need type conversion, and what the shape of the target variable looks like (is it heavily right-skewed? are there zeros or negatives?).

**Output of this step:** EDA section of the notebook with charts and written observations.

### Step 3 — Data Cleaning

Apply the following cleaning steps in order:

Filter rows. Remove any row where `declared_value` is null, zero, or negative — these cannot be modeled as a regression target. Remove rows where `permit_type` is null. Remove rows where `issue_date` is null.

Handle remaining missing values. For `square_footage`, impute with the median value grouped by `permit_type` (missing footage for a commercial permit likely differs from a residential permit). For `zip_code`, impute with the modal ZIP within the same `council_district`. For `occupancy_type`, impute with the string `"Unknown"` and treat as a valid category.

Type conversions. Parse `issue_date` to datetime. Cast `declared_value` and `square_footage` to float. Cast `council_district` to string (it is a categorical ID, not a numeric quantity).

Outlier handling. For `declared_value`, cap values at the 99th percentile. Log-transform `declared_value` to create `log_declared_value` — this becomes your regression target. Keep the original column for back-transformation during evaluation.

**Output of this step:** A cleaned DataFrame saved as `permits_cleaned.csv`.

### Step 4 — Feature Engineering

Engineer the following features from the cleaned data:

Temporal features from `issue_date`: `issue_year` (int), `issue_month` (int), `issue_quarter` (int), `days_since_epoch` (int — number of days since 2000-01-01, captures long-term economic trends).

Contractor features: `contractor_permit_count` — total number of permits filed by each contractor across the dataset. Flag contractors with more than 10 permits as `is_repeat_contractor` (binary). Encode `contractor_city` as a frequency-encoded column (replace city name with its frequency count across the dataset).

Categorical encoding: One-hot encode `permit_type`, `permit_subtype`, `occupancy_type`, and `council_district`. Label-encode `zip_code` using target encoding — replace each ZIP with the mean `log_declared_value` for permits in that ZIP (compute on training data only; apply to validation and test using training statistics).

Drop columns that should not enter the model: `permit_number`, `address`, `work_description`, `contractor_name`, `issue_date` (the raw date string — the engineered features replace it).

**Output of this step:** A modeling-ready DataFrame saved as `permits_features.csv`. Print final shape and a list of all feature column names.

---

## Phase 2 — ML Modeling and Evaluation

**Goal:** Train a regression model to predict `log_declared_value`, evaluate it rigorously, explain it with SHAP, and run K-Means clustering on the permit space.

### Step 5 — Train/Validation/Test Split

Split the data chronologically, not randomly. Sort by `issue_date`. Use the earliest 70% of records as training data, the next 15% as validation, and the most recent 15% as the test set. This prevents data leakage — a model trained on 2024 data should not be evaluated on 2022 data.

Print the date ranges for each split to confirm the chronological ordering is correct.

**Output of this step:** `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test` — all saved as numpy arrays or stored as variables in the notebook session.

### Step 6 — Baseline Models

Train two baseline models and record their validation metrics before touching XGBoost. This is your performance floor — every subsequent model must beat this.

Train a Linear Regression model on the standardized features. Train a Ridge Regression model and tune the `alpha` hyperparameter using cross-validation on the training set. For both, compute RMSE, MAE, R-squared, and MAPE on the validation set. Log results in a comparison table in the notebook.

**Output of this step:** Baseline metrics table with RMSE, MAE, R², MAPE for both linear models.

### Step 7 — Random Forest

Train a Random Forest Regressor with `n_estimators=200`, `max_features='sqrt'`, and `oob_score=True`. Evaluate on validation set using the same four metrics. Extract the top 20 features by `feature_importances_` and plot them as a horizontal bar chart. This gives a quick, model-internal view of which features matter before running SHAP.

**Output of this step:** RF validation metrics and feature importance chart.

### Step 8 — XGBoost with Hyperparameter Tuning

Train an XGBoost Regressor. Use Optuna (preferred) or GridSearchCV to tune the following hyperparameters: `n_estimators` (range: 200–1000), `max_depth` (range: 3–9), `learning_rate` (range: 0.01–0.3), `subsample` (range: 0.6–1.0), `colsample_bytree` (range: 0.6–1.0), `min_child_weight` (range: 1–10). Run at least 50 Optuna trials. Use 5-fold cross-validation on the training set as the objective. Select the best trial and retrain the final model on the full training set. Evaluate on validation and test sets.

**Output of this step:** Best XGBoost hyperparameters, final model saved as `xgb_model.pkl` using joblib, and final test-set metrics printed clearly.

### Step 9 — Stacking Ensemble (Optional but Recommended)

Build a stacking ensemble with Random Forest and XGBoost as base learners and a Ridge Regression meta-learner. Use `sklearn.ensemble.StackingRegressor`. Train on `X_train`, evaluate on `X_val` and `X_test`. Compare ensemble metrics to standalone XGBoost. If the ensemble improves R² by more than 0.01, use it as the final model for SHAP analysis and deployment. Otherwise, keep XGBoost.

**Output of this step:** Final model selection decision written as a markdown cell in the notebook.

### Step 10 — SHAP Analysis

Use the `shap` library to generate global and local explanations for the final model.

Global explanation: compute SHAP values for the entire test set. Generate a beeswarm summary plot showing the top 20 features by mean absolute SHAP value. Generate dependence plots for the three highest-ranked features — for each, plot SHAP value on the y-axis and feature value on the x-axis, with the second interaction feature auto-selected by SHAP.

Local explanation: pick three representative permits from the test set — one high-value commercial permit, one mid-value residential permit, and one low-value maintenance permit. Generate a waterfall plot for each, showing how each feature pushes the prediction above or below the base value.

Save all SHAP plots as PNG files: `shap_beeswarm.png`, `shap_dependence_sqft.png`, `shap_waterfall_commercial.png`, etc.

**Output of this step:** SHAP plots saved and interpretive markdown written in the notebook explaining what the model has learned.

### Step 11 — K-Means Clustering

Cluster permits by their feature profile to discover natural archetypes in the data. Use a subset of features that describe the permit's characteristics — `log_declared_value`, `square_footage`, encoded `permit_type`, encoded `occupancy_type`, encoded `council_district` — standardized to zero mean and unit variance.

Run K-Means for k = 2 through 10. Plot the elbow curve (inertia vs k) and the silhouette score curve (silhouette score vs k). Select the optimal k based on where the elbow bends and silhouette score peaks. Refit K-Means at the selected k.

Profile each cluster by computing the mean value of the key features per cluster. Name each cluster based on its profile — for example: "High-Value Commercial New Construction," "Residential Renovation Mid-Range," "Low-Value Single-Family Alteration," "Swimming Pool and Specialty Permits." Write these names and their definitions as a markdown table in the notebook.

Visualize clusters using a 2D PCA scatter plot with cluster color labels. Add a second visualization: a bar chart of mean `declared_value` per cluster with cluster names on the x-axis.

**Output of this step:** Cluster assignments saved as a column in the cleaned permits DataFrame, cluster profile table, and cluster plots saved as PNGs.

---

## Phase 3 — Report, Slides, and AI Agent

**Goal:** Communicate findings in a written report and slide deck, then deploy an AI agent that wraps the trained model for interactive queries.

### Step 12 — Written Report

Write the full report as a structured document covering the following sections in order: executive summary (non-technical, two paragraphs), dataset description and data quality findings from EDA, feature engineering decisions and rationale, model performance comparison table across all models, SHAP findings explained in plain English (which features drive permit value and why that makes sense), clustering findings with cluster profiles and policy implications, and limitations and future work.

The report is for both technical and non-technical readers. Keep the body text accessible. Put model metrics and SHAP charts in a dedicated results section. Put raw code references in an appendix.

**Output of this step:** Report saved as a PDF or Word document, approximately 8–12 pages.

### Step 13 — Slide Deck

Build a 12–15 slide presentation covering: title slide, problem statement and why it matters, dataset overview, EDA highlights (one or two most interesting charts), modeling approach (pipeline diagram), model comparison results, SHAP insights (beeswarm and one waterfall), clustering findings (PCA scatter + cluster names), live agent demo (this slide plays during the demo, not pre-filled), policy recommendations, and next steps.

Keep each slide to one idea. Use the SHAP and cluster PNGs generated in earlier steps as chart slides.

**Output of this step:** Slide deck saved as a PPTX or PDF file.

### Step 14 — Flask API for the Model

Build a Flask application that wraps the trained model as a REST API. The API should expose a single POST endpoint at `/predict`. The request body accepts JSON with the following fields: `permit_type` (string), `zip_code` (string), `square_footage` (float), `occupancy_type` (string), `council_district` (string), `issue_month` (int), `issue_year` (int). The endpoint applies the same feature engineering transformations used during training (load the fitted encoders and imputers saved from Phase 1), runs inference with the saved model, back-transforms the log-predicted value to USD, and returns a JSON response with `predicted_value_usd` (float), `confidence_note` (string describing model confidence tier), and `top_features` (list of the three most impactful SHAP features for this prediction with their direction).

Load the model and encoders at app startup using joblib. Do not reload them on every request.

Test the endpoint locally by sending a sample POST request with `curl` or `requests`. Confirm the response is valid JSON and the predicted value is in a reasonable dollar range for the permit type.

Deploy the Flask app to Render (free tier). Push the project to a GitHub repository. Connect the repository to Render. Set the start command to `gunicorn app:app`. Confirm the public URL returns a valid response to a POST request.

**Output of this step:** Flask API live at a public Render URL. Note the URL — it is the backend for the n8n agent.

### Step 15 — n8n AI Agent

Build a conversational agent in n8n Cloud that connects to the Flask API and allows any user to query permit value predictions through natural language.

Create a new n8n workflow. Add a Chat Trigger node as the entry point — this gives the agent a shareable public chat link. Add an AI Agent node using OpenAI GPT-4o as the language model. Give the agent the following system prompt (adapt as needed):

> You are a Dallas Building Permit Value Advisor. When a user describes a construction project in Dallas, ask for the permit type, ZIP code, square footage, occupancy type, council district, and the expected month and year of permit filing. Once you have all required fields, call the prediction tool and return the predicted value in plain language along with a brief explanation of the top factors driving that estimate.

Add an HTTP Request tool node connected to the AI Agent. Configure it to send a POST request to the Flask Render URL with the fields extracted by the agent. Map the response fields (`predicted_value_usd`, `top_features`) into the agent's reply template.

Test the full flow by entering a natural language query in the n8n chat interface: "I'm planning a commercial renovation in ZIP 75201, about 3000 square feet, council district 2, filing in March 2025." Confirm the agent asks clarifying questions if needed, calls the API, and returns a readable prediction with an explanation.

Activate the workflow and copy the public chat link.

**Output of this step:** n8n agent live with a shareable public URL. Test it from a fresh browser tab to confirm it works end-to-end without authentication.

### Step 16 — Dry Run and Final Prep

Run a full end-to-end dry run of the presentation: open the slide deck, walk through each slide at presentation pace, open the n8n agent link in a browser tab during the demo slide, and enter a live query. Time the full run — target 12–15 minutes. Prepare answers for likely Q&A questions: why did you log-transform the target, what does SHAP show about ZIP code effects, what would happen if you retrained monthly on fresh API data, how would you productionize this beyond Render free tier.

**Output of this step:** Final slide deck version saved. Public agent URL confirmed working. Q&A notes written.

---

## Deliverables Checklist

- `raw_permits.csv` — raw API pull  
- `permits_cleaned.csv` — cleaned data  
- `permits_features.csv` — engineered feature matrix  
- `xgb_model.pkl` — trained final model  
- Fitted encoder/imputer objects saved with joblib  
- SHAP plots (beeswarm, dependence, waterfall PNGs)  
- Cluster profile table and PCA scatter plot  
- Full ML pipeline notebook (`.ipynb`)  
- Written report (PDF or DOCX)  
- Slide deck (PPTX or PDF)  
- Flask API live on Render — public POST endpoint  
- n8n agent live — shareable public chat URL  

---

## Dependency Order (Critical Path)

```
API Pull → EDA → Cleaning → Feature Engineering
    → Train/Val/Test Split
        → Baseline Models → Random Forest → XGBoost Tuning → Stacking
            → SHAP Analysis → K-Means Clustering
                → Report + Slides
                    → Flask API → n8n Agent → Dry Run
```

Every step feeds the next. Feature engineering artifacts (encoders, imputers) must be saved with joblib immediately after fitting — the Flask API loads them at runtime and cannot re-derive them from scratch.
