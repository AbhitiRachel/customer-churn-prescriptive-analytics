-- =========================================================
-- 03. ACQUISITION COHORT RETENTION ANALYSIS
-- =========================================================
-- "Customers acquired in cohort year Y have an X% retention rate after
-- N months, compared with customers acquired in a different cohort year."

-- Cohort size by acquisition year
SELECT cohort_year, COUNT(DISTINCT customer_id) AS customers_acquired
FROM customer_month
WHERE tenure_months = 0
GROUP BY cohort_year ORDER BY cohort_year;

-- Retention rate at 6, 12, 24 months by acquisition cohort year
-- (retention = fraction of the cohort still active — i.e. still appearing in the panel — at that tenure milestone)
-- CAUTION - RIGHT-CENSORING: cohorts acquired late in the observation window (e.g. 2025, and 2024 for
-- the 24mo milestone) will show artificially low or NULL retention at longer horizons simply because
-- those customers have not yet reached that tenure milestone by Dec-2025 (the panel's last month) --
-- this is NOT necessarily elevated churn. Only compare milestones where the cohort has had enough
-- calendar time to reach them (e.g. don't compare 2025's 6mo retention at face value against 2021's).
WITH cohort_sizes AS (
    SELECT customer_id, cohort_year
    FROM customer_month WHERE tenure_months = 0
),
retention_at_milestone AS (
    SELECT cs.cohort_year,
           cm.tenure_months,
           COUNT(DISTINCT cm.customer_id) AS still_active
    FROM customer_month cm
    JOIN cohort_sizes cs USING (customer_id)
    WHERE cm.tenure_months IN (6, 12, 24)
    GROUP BY cs.cohort_year, cm.tenure_months
)
SELECT
    r.cohort_year,
    c.customers_acquired,
    MAX(CASE WHEN r.tenure_months = 6 THEN r.still_active END) AS active_at_6mo,
    ROUND((100.0 * MAX(CASE WHEN r.tenure_months = 6 THEN r.still_active END) / c.customers_acquired)::numeric, 1) AS retention_6mo_pct,
    MAX(CASE WHEN r.tenure_months = 12 THEN r.still_active END) AS active_at_12mo,
    ROUND((100.0 * MAX(CASE WHEN r.tenure_months = 12 THEN r.still_active END) / c.customers_acquired)::numeric, 1) AS retention_12mo_pct,
    MAX(CASE WHEN r.tenure_months = 24 THEN r.still_active END) AS active_at_24mo,
    ROUND((100.0 * MAX(CASE WHEN r.tenure_months = 24 THEN r.still_active END) / c.customers_acquired)::numeric, 1) AS retention_24mo_pct
FROM retention_at_milestone r
JOIN (SELECT cohort_year, COUNT(DISTINCT customer_id) AS customers_acquired FROM cohort_sizes GROUP BY cohort_year) c
    ON c.cohort_year = r.cohort_year
GROUP BY r.cohort_year, c.customers_acquired
ORDER BY r.cohort_year;

-- Full retention curve (every tenure month, all cohorts) - for a cohort retention-curve chart
WITH cohort_sizes AS (
    SELECT customer_id, cohort_year FROM customer_month WHERE tenure_months = 0
)
SELECT
    cs.cohort_year,
    cm.tenure_months,
    COUNT(DISTINCT cm.customer_id) AS still_active,
    ROUND((100.0 * COUNT(DISTINCT cm.customer_id) /
        FIRST_VALUE(COUNT(DISTINCT cm.customer_id)) OVER (PARTITION BY cs.cohort_year ORDER BY cm.tenure_months))::numeric, 1
    ) AS retention_pct
FROM customer_month cm
JOIN cohort_sizes cs USING (customer_id)
WHERE cm.tenure_months <= 36
GROUP BY cs.cohort_year, cm.tenure_months
ORDER BY cs.cohort_year, cm.tenure_months;

-- Quarterly cohort comparison (finer grain than yearly)
SELECT cohort_quarter, COUNT(DISTINCT customer_id) AS customers_acquired,
       ROUND(AVG(tenure_months)::numeric, 1) AS avg_observed_tenure
FROM customer_month
WHERE cohort_year >= 2021
GROUP BY cohort_quarter