# Data Dictionary — `customer_month_clean.csv` (longitudinal panel)

**This is synthetic data.** Generated from documented business rules and a
monthly hazard model (see `src/generate_synthetic_data.py` header), then
calibrated in trend-shape against real, publicly reported Indian telecom
industry statistics (see `data/reference/industry_benchmarks.csv`). No row
represents a real customer of any real company.

**Grain: one row per customer per active month.** A customer contributes
one row for every month they were active, from `max(signup, 2021-01-01)`
through either `2025-12-01` or the month they churned — whichever is first.

| Column | Type | Description |
|---|---|---|
| customer_id | string | Unique customer identifier |
| activity_month | date | The month this row describes (first-of-month) |
| tenure_months | int | Months since signup, as of this row's month |
| region | category | North / South / East / West |
| customer_segment | category | Consumer / SMB / Enterprise |
| age | int | Customer age (static; imputed with segment median where invalid/missing) |
| contract_type | category | Month-to-month / One year / Two year — **can change month to month** (customers occasionally upgrade to longer contracts; they never downgrade in this model) |
| payment_method | category | Electronic check / Mailed check / Bank transfer (auto) / Credit card (auto) / Unknown — can migrate toward autopay over time |
| plan | category | Basic / Standard / Premium — can change (rare) |
| monthly_charges | float | This month's bill. Reflects plan + add-ons + contract discount + the industry-calibrated pricing trend (see below). Outlier-capped per plan (see `monthly_charges_outlier_flag`) |
| usage_gb | float | This month's usage/engagement proxy (mean-reverting random walk + seasonality) |
| num_addon_services | int | Add-on service count this month |
| support_calls_this_month | int | Support/complaint calls in this specific month |
| is_active | int | Always 1 (rows only exist while active) |
| **churn_this_month** | int (0/1) | **Target variable.** 1 only in the exact month a customer churns; 0 every other month. This is the last row for that customer. |
| signup_month | date | First-of-month signup date |
| cohort_year / cohort_month | int | Acquisition cohort — for retention-curve analysis |
| cohort_quarter | string | e.g. "2023-Q2" — finer-grained cohort for less-censored comparisons |
| monthly_charges_outlier_flag | int (0/1) | 1 if this row's charge was capped during cleaning |
| tenure_years, tenure_group | derived | Convenience bucketing of tenure_months |
| is_high_value | int (0/1) | Top quartile of `monthly_charges` **within that calendar month's cross-section** (not a fixed global quartile — a customer's tier can shift as prices rise industry-wide) |
| complaint_flag | int (0/1) | 1 if `support_calls_this_month` >= 2 |

## `industry_benchmarks.csv` (external reference data — NOT synthetic)
Real, publicly reported figures from TRAI and public news coverage of listed
telecom operators, used only to sanity-check the synthetic pricing trend's
shape and magnitude. See `src/industry_benchmarks.py` for exact figures,
units, and full source citations. Columns: `year, metric, value, unit, note, source`.

## LEAKAGE — read this before modelling
- `churn_hazard_latent` (the exact probability used to draw that month's
  churn outcome) was **dropped** during cleaning. Never reconstruct it.
- To predict `churn_this_month` for a given row, only use information from
  **that row or earlier** for that customer. Do not join future months'
  fields (e.g. next month's `usage_gb`) back onto a row as a "feature" —
  that is leakage even though the raw table makes it easy to do by accident.
- `is_high_value` is computed within-month, so it is safe to use as a
  same-month feature; it does not require future information.

## Known statistical caveat — right-censoring at the panel edges
Because the observation window ends at 2025-12, any customer acquired late
enough that they haven't yet reached a given tenure milestone (e.g. the 2025
cohort's 12-month retention) will be **absent, not churned**, from that
milestone's count. This also affects **cohort-year** aggregates (which mix
January-December signups): once tenure exceeds ~12 months, cohort-year
retention curves increasingly reflect only the earlier-in-year signups
(later ones haven't reached that tenure yet), which can look like an
accelerating drop that isn't real attrition. Use `cohort_quarter` for
tighter, less-biased cohort comparisons, and treat any cohort-year curve
past month ~11-12 with caution. This is flagged directly in
`sql/03_cohort_retention_analysis.sql` and in the EDA chart.

## Notes for Member 2 (modelling)
- Natural framing: predict `churn_this_month` from that row's own features —
  this is a standard panel/survival-style churn setup.
- Strongest expected signals: `contract_type`, `payment_method`,
  `support_calls_this_month`, `monthly_charges`, `tenure_months` (non-linear —
  elevated hazard <6 months, declining after).
- `is_high_value` was **not** a strong standalone predictor in v1's
  cross-sectional data either — worth re-testing here, don't assume it matters.

## Notes for Member 3 (forecasting / ROI)
- `activity_month`-grouped revenue/churn series are ready-made time series —
  see `sql/02_revenue_analysis_over_time.sql`.
- The industry-calibrated pricing step (notably around 2024-07, modelled on
  real tariff hikes) is a deliberate, documented structural break — useful
  for testing forecast models against a known regime change.
- Revenue-at-risk-over-time query in `02_revenue_analysis_over_time.sql`
  gives a ready basis for ROI/what-if modelling.
