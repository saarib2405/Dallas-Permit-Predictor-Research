# Data Inventory — Dallas Building Permits

**Generated:** 2026-09-10

---

## Dataset Summary

| Field | Value |
|-------|-------|
| **Dataset Name** | Dallas Building Permits |
| **Source** | City of Dallas Open Data Portal |
| **Download Date** | April 2026 (approximate, based on project initiation) |
| **Study Period** | January 2, 2018 – August 29, 2020 |
| **Number of Records** | 126,840 |

---

## Main Variables

| # | Column | Description | Example Value |
|---|--------|-------------|---------------|
| 1 | `Permit Number` | Unique permit identifier | `2003133024` |
| 2 | `Permit Type` | Composite string encoding category, code, sub-category, and work type | `Electrical (EL) Commercial  Alteration` |
| 3 | `Issued Date` | Date the permit was issued (MM/DD/YY format) | `03/13/20` |
| 4 | `Mapsco` | Dallas Mapsco grid reference (geographic sub-area) | `39-N` |
| 5 | `Contractor` | Contractor name, full mailing address, and phone number in a single string | `B+T GROUP HOLDINGS INC 1025 S. Belt Line Road, Suite 400, Coppell, TX 75019 (469) 480-3578` |
| 6 | `Value` | Declared construction value in USD (comma-formatted) | `1,800` |
| 7 | `Area` | Square footage of the permitted work | `0` |
| 8 | `Work Description` | Free-text description of the permitted work | `PERMANENT ELECTRICAL SERVICE` |
| 9 | `Land Use` | Occupancy / land-use classification | `COMMUNICATIONS` |
| 10 | `Street Address` | Street address of the permitted work | `2928 ANDREA LN` |
| 11 | `Zip Code` | ZIP code of the permit location | `75228` |

---

## Target Variable

| Property | Value |
|----------|-------|
| **Column** | `Value` |
| **Description** | Declared construction value in USD |
| **Type** | Continuous (stored as comma-formatted string in raw data) |
| **Range** | $0 – $64,565,291 |
| **Zero-value records** | 30,054 (23.7%) — administrative permits (signs, demolitions, etc.) |
| **Modeling approach** | Two-Stage Hurdle Model: Stage 1 classifies zero vs. positive; Stage 2 predicts log(Value) for positive-value permits |

---

## Missing-Data Issues

| Column | Null / Empty Count | % of Total | Notes |
|--------|-------------------|------------|-------|
| `Permit Number` | 0 | 0.0% | Complete |
| `Permit Type` | 0 | 0.0% | Complete |
| `Issued Date` | 0 | 0.0% | Complete |
| `Mapsco` | 890 | 0.7% | Geographic grid reference; used for Zip Code imputation |
| `Contractor` | 3 | <0.01% | 3 rows dropped during cleaning (critical field) |
| `Value` | 0 | 0.0% | Complete; 23.7% are legitimately $0 |
| `Area` | 0 | 0.0% | Complete as numeric; many genuine zeros (e.g., electrical work) |
| `Work Description` | 902 | 0.7% | Free-text; not used in tabular ML pipeline |
| `Land Use` | 0 | 0.0% | Complete |
| `Street Address` | 3 | <0.01% | Dropped during cleaning (too high-cardinality for modeling) |
| `Zip Code` | 5,397 | 4.3% | Imputed via Mapsco grid lookup + overall mode fallback |

### Key Observations

- **Zip Code** has the largest missing-data issue (4.3%). The cleaning pipeline addresses this using Mapsco-based geographic imputation followed by overall mode fallback.
- **Contractor** has only 3 missing values; these rows are dropped as Contractor is a critical field for feature engineering.
- **Work Description** and **Mapsco** have sub-1% missingness and are either unused (Work Description) or used only as an imputation helper (Mapsco), so their missing values do not propagate into the feature matrix.
- **Value = $0** is not missing data — it represents legitimate administrative permits and is handled by the Stage 1 classifier in the Hurdle Model architecture.
