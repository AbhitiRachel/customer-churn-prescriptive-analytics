# Data Cleaning Log — Longitudinal Panel

- Loaded raw panel: 253811 rows, 19 columns, 6000 unique customers
- Removed 1513 exact duplicate rows -> 252298 rows remain
- Standardized categorical text fields (trimmed whitespace, title-cased)
- Found 10 invalid age rows (<=0 or >100) -> set to NaN, imputed by segment median
- Imputed 3794 missing values in 'age'
- Imputed 2018 missing values in 'payment_method'
- Imputed 3027 missing values in 'usage_gb'
- Flagged and capped 247 monthly_charges outliers (per-plan 3x IQR rule)
- Dropped 'churn_hazard_latent' — generator-internal churn probability, would leak the target
- Added derived features: tenure_years, tenure_group, is_high_value (relative to that month's cross-section), complaint_flag, cohort_quarter
- Passed sanity checks: positive charges, valid ages, window bounds respected, no rows exist after any customer's churn month (leakage guard)
- Churn events in window: 2323 | share of customers who ever churn in window: 0.387