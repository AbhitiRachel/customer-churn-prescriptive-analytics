-- =========================================================
-- 02. REVENUE ANALYSIS OVER TIME
-- =========================================================

-- Monthly revenue trend (actuals across the panel, shows the synthetic tariff-hike step in Jul 2024)
SELECT
    activity_month,
    ROUND(SUM(monthly_charges)::numeric, 2) AS total_monthly_revenue,
    COUNT(*) AS active_customers,
    ROUND(AVG(monthly_charges)::numeric, 2) AS arpu
FROM customer_month
GROUP BY activity_month
ORDER BY activity_month;

-- Year-over-year ARPU trend (for comparison against industry_benchmarks)
SELECT
    EXTRACT(YEAR FROM activity_month)::int AS year,
    ROUND(AVG(monthly_charges)::numeric, 2) AS avg_arpu_synthetic
FROM customer_month
GROUP BY 1 ORDER BY 1;

-- Same benchmark, side by side with real industry ARPU (join against reference table)
SELECT
    s.year, s.avg_arpu_synthetic,
    b.value AS industry_arpu_inr, b.source
FROM (
    SELECT EXTRACT(YEAR FROM activity_month)::int AS year, ROUND(AVG(monthly_charges)::numeric, 2) AS avg_arpu_synthetic
    FROM customer_month GROUP BY 1
) s
LEFT JOIN industry_benchmarks b
    ON b.year = s.year AND b.metric = 'blended_wireless_arpu'
ORDER BY s.year;

-- Revenue by segment / plan / region (current cross-section, latest month)
WITH latest_month AS (SELECT MAX(activity_month) AS m FROM customer_month)
SELECT customer_segment, COUNT(*) AS customers,
       ROUND(SUM(monthly_charges)::numeric, 2) AS monthly_revenue,
       ROUND((100.0 * SUM(monthly_charges) / SUM(SUM(monthly_charges)) OVER())::numeric, 2) AS pct_of_revenue
FROM customer_month, latest_month
WHERE activity_month = latest_month.m
GROUP BY customer_segment ORDER BY monthly_revenue DESC;

-- Revenue at risk over time: monthly revenue tied to customers who churn THAT month
SELECT
    activity_month,
    ROUND(SUM(CASE WHEN churn_this_month = 1 THEN monthly_charges ELSE 0 END)::numeric, 2) AS revenue_lost_this_month,
    ROUND(SUM(monthly_charges)::numeric, 2) AS total_revenue_this_month,
    ROUND((100.0 * SUM(CASE WHEN churn_this_month = 1 THEN monthly_charges ELSE 0 END)
           / NULLIF(SUM(monthly_charges), 0))::numeric, 3) AS pct_revenue_lost
FROM customer_month
GROUP BY activity_month
ORDER BY activity_month;

-- Revenue at risk right now: current monthly charges for active, high-complaint customers (proxy for near-term churn risk)
WITH latest_month AS (SELECT MAX(activity_month) AS m FROM customer_month)
SELECT
    ROUND(SUM(monthly_charges)::numeric, 2) AS monthly_revenue_at_risk,
    COUNT(*) AS customers_at_risk
FROM customer_month, latest_month
WHERE activity_month = latest_month.m AND complaint_flag = 1;
