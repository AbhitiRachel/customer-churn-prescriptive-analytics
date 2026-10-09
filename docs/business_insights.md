# Business Insights — Longitudinal Panel (Member 1)

*All figures pulled directly from SQL/EDA output in this repo, not invented
ahead of time. Monthly rates are on a customer-month basis across the full
2021-2025 panel unless stated otherwise.*

## Headline KPIs (see `outputs/kpi_summary.csv` for exact figures)
- 252,298 customer-month rows, 6,000 unique customers, Jan 2021 - Dec 2025
- Overall monthly churn rate: **0.92%** (this compounds to ~39% of the base
  churning at some point across the 5-year window — a monthly rate this low
  still adds up over 60 months, which is a useful intuition check in itself)
- 3,697 customers still active as of Dec 2025
- Current monthly revenue run-rate: ~$362.6K

## Finding 1 — Contract type remains the strongest, clearest churn lever
Month-to-month customers churn at **1.77%/month** vs **0.82%** (one-year)
and **0.31%** (two-year) — a nearly 6x gap between the extremes. This holds
up just as strongly in the longitudinal data as it did in the static
snapshot, which is reassuring: it's not an artifact of the static model.

## Finding 2 — Tenure's protective effect is now much clearer with monthly-hazard data
Monthly churn hazard declines steadily and monotonically with tenure: 1.75%
(0-6mo) → 1.42% (7-12mo) → 1.35% (1-2yr) → 1.04% (2-4yr) → 0.74% (4yr+).
Unlike the earlier static-snapshot version (where this pattern was noisy),
the longitudinal hazard framing shows it cleanly — new customers are
genuinely the highest-risk group, and risk keeps falling the longer someone
stays, not just in the first year.

## Finding 3 — High-value customers now show *lower* churn, once time is modeled correctly
High-value customers (top quartile of that month's pricing) churn at
**0.59%/month** vs **1.03%** for everyone else — the opposite conclusion
from the earlier static dataset (where value tier had no relationship to
churn). This makes sense once you account for tenure and contract type
properly: value in this data is correlated with longer-tenured, longer-
contract customers, who are simply more stable. **This is a good illustration
of why the static snapshot version could mislead** — a naive read of "does
value protect against churn?" depends heavily on whether you control for
the underlying tenure/contract dynamics.

## Finding 4 — A visible churn response around the synthetic tariff-hike month
Monthly churn rate ticks up around July 2024 (0.72% → 0.93% that month)
before settling back down — consistent with the price-shock rule built into
the generator, and directionally consistent with real-world churn upticks
that typically follow tariff hikes. This is a designed, documented effect
(see `generate_synthetic_data.py`), not a spontaneous finding — but it gives
Member 3 a concrete, dated event to test forecast models against.

## Finding 5 — Cohort retention is genuinely useful, with an important caveat
Retention curves by acquisition cohort year (2021-2024) are consistent
with each other at 6 and 12 months (~80-90% and ~80-84% respectively) — no
strong evidence of cohorts getting systematically "worse" over time. **However**,
the apparent steep drop in later months for cohorts close to the panel's
edge (e.g. 2024 past month ~13, 2025 entirely) is a **right-censoring
artifact**, not a real acceleration in churn — see the caveat documented in
`sql/03_cohort_retention_analysis.sql` and `docs/data_dictionary.md`. Anyone
using this chart should compare cohorts only up to a tenure horizon that all
of them have actually reached.

## Calibration against real industry data
Synthetic ARPU grew from ~65 (2021) to ~98 (2025, in project units) — a
~51% increase, matching the shape and rough magnitude of TRAI's reported
real industry blended ARPU growth of ~57% (₹121 → ₹190/month) over the same
period (see `outputs/arpu_calibration_check.png`). This is a trend-shape
match only — we did not force the synthetic numbers to hit the real rupee
figures, and no synthetic customer represents a real customer of any real
telecom operator.

## What this hands off to Member 2 (modelling)
- Predict `churn_this_month` using only same-row-or-earlier information
  (see leakage guard in `data_dictionary.md`)
- Expect `contract_type`, `tenure_months`, `payment_method`, and
  `support_calls_this_month` to be the strongest features
- Re-test `is_high_value` rather than assuming its direction — it flipped
  sign between the static and longitudinal versions of this dataset

## What this hands off to Member 3 (forecasting / ROI)
- Clean monthly revenue and churn time series, with a known, dated
  structural break (Jul 2024 pricing step) to validate against
- Cohort retention curves as a second lens beyond period-over-period churn
- Revenue-at-risk-over-time query ready as a starting point for ROI
  scenario modelling
