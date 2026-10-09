# Smart Customer Retention & BI System — Member 1 Scope (v2: Longitudinal)
**Data Preparation + SQL + EDA + KPI/Business Analysis**

> This is v2 — a monthly customer-panel rebuild of the original static
> snapshot version. The static version is preserved at
> `../churn_project_v1_backup/` and as git commit `v1: static customer
> snapshot` in this repo's history, in case anything needs to be reverted
> or compared.

## What changed from v1
- One row per customer → **one row per customer per active month**
  (Jan 2021 - Dec 2025), ~252K rows across 6,000 customers
- Churn is no longer a static label — it's the output of a **monthly hazard
  model**, recomputed from each customer's *current* state, so churn happens
  in a specific month with no lookahead/leakage
- Contract type, payment method, plan, usage, and complaints now evolve
  month to month instead of being fixed at signup
- Added acquisition **cohort fields** (`signup_month`, `cohort_year`,
  `cohort_month`, `cohort_quarter`) for genuine cohort retention analysis
- Pricing trend is calibrated (in shape/magnitude only) against real,
  publicly reported Indian telecom industry ARPU data (TRAI) — see
  `data/reference/industry_benchmarks.csv`
- SQL and EDA extended with real time-based analysis: monthly churn trend,
  customer growth/decline, cohort retention curves, revenue-at-risk-over-time

## Structure
```
churn_project/
├── data/
│   ├── raw/customer_month_raw.csv          # messy synthetic panel
│   ├── processed/customer_month_clean.csv  # <- analytics-ready panel
│   ├── processed/cleaning_log.md
│   └── reference/industry_benchmarks.csv   # REAL external data (TRAI etc.)
├── src/
│   ├── generate_synthetic_data.py   # monthly hazard-based panel generator
│   ├── industry_benchmarks.py       # real benchmark data + sources
│   ├── clean_data.py                # cleaning pipeline (panel-aware)
│   ├── load_to_postgres.py          # loads panel + benchmarks into Postgres
│   └── eda_and_kpis.py              # time-based EDA + KPI framework
├── sql/
│   ├── 01_core_kpis.sql
│   ├── 02_revenue_analysis_over_time.sql
│   ├── 03_cohort_retention_analysis.sql
│   └── 04_customer_value_and_business_questions.sql
├── outputs/                  # charts + kpi_summary.csv
└── docs/
    ├── data_dictionary.md    # includes leakage guard + censoring caveat
    └── business_insights.md
```

## How to reproduce
```bash
python3 src/generate_synthetic_data.py   # -> data/raw/customer_month_raw.csv
python3 src/industry_benchmarks.py       # -> data/reference/industry_benchmarks.csv
python3 src/clean_data.py                # -> data/processed/customer_month_clean.csv
python3 src/load_to_postgres.py          # -> loads into Postgres db `churn_retention`
python3 src/eda_and_kpis.py              # -> outputs/*.png + kpi_summary.csv

psql -d churn_retention -f sql/01_core_kpis.sql
psql -d churn_retention -f sql/02_revenue_analysis_over_time.sql
psql -d churn_retention -f sql/03_cohort_retention_analysis.sql
psql -d churn_retention -f sql/04_customer_value_and_business_questions.sql
```

## Modelling: churn prediction, segmentation, explainability
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# (run the data pipeline above first — the Postgres step is optional for modelling)
.venv/bin/python src/churn_model.py            # -> models/churn_model.joblib, outputs/churn_model_metrics.csv
.venv/bin/python src/customer_segmentation.py  # -> outputs/segment_profiles.csv, retention_priority.csv
.venv/bin/python src/explainability.py         # -> outputs/shap_*.png, feature_importance.csv
```
Run them in this order — segmentation's priority matrix and explainability both
load the saved churn model. `src/features.py` holds the shared, past-only
(leakage-safe) feature engineering.

- **Churn prediction** — HistGradientBoosting vs. logistic regression vs.
  majority-class baseline, time-split (train 2021-2024, test 2025). Accuracy is
  reported but is misleading at a ~0.6% churn rate; ROC-AUC, PR-AUC and
  top-decile capture are the metrics to judge on. An evaluation-only "oracle"
  row scores the generator's true hazard to show the achievable ceiling.
- **Segmentation** — K-Means on customer-level behaviour (k by silhouette),
  validated by churn rate per segment; plus a risk x value retention priority
  matrix for currently active customers.
- **Explainability** — SHAP (global, beeswarm, dependence, per-customer
  reasons, waterfall) with validity checks: additivity, agreement with
  permutation importance, and recovery of the generator's known drivers.

## Key numbers (full write-up in `docs/business_insights.md`)
- 252,298 customer-months, 6,000 customers, 0.92% overall monthly churn rate
- Contract type is still the strongest lever: 1.77%/mo (month-to-month) vs
  0.31%/mo (two-year)
- High-value customers now churn *less*, not the same — a genuine
  longitudinal finding the static snapshot couldn't see correctly
- Synthetic ARPU growth (~51%, 2021-2025) tracks real TRAI-reported industry
  ARPU growth (~57%) in shape and magnitude

## On the external data (read before citing this project)
`data/reference/industry_benchmarks.csv` and `src/industry_benchmarks.py`
contain REAL, publicly reported Indian telecom industry figures (TRAI
reports, public news coverage of listed operators' investor disclosures),
used strictly as **external calibration/context** — to check that this
project's synthetic ARPU trend is plausible. No synthetic customer or
cohort represents an actual customer of any real company; only aggregate
industry-level trend shape is benchmarked. Full source citations are in
`industry_benchmarks.py`.
