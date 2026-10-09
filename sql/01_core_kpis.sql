-- =========================================================
-- 01. CORE KPIs (longitudinal panel version)
-- =========================================================

-- Snapshot KPIs as of the latest month in the panel
WITH latest_month AS (SELECT MAX(activity_month) AS m FROM customer_month)
SELECT
    COUNT(*) AS active_customers_latest_month,
    ROUND(AVG(monthly_charges)::numeric, 2) AS avg_monthly_revenue_per_customer,
    ROUND(AVG(tenure_months)::numeric, 1) AS avg_tenure_months,
    ROUND(SUM(monthly_charges)::numeric, 2) AS monthly_revenue_run_rate
FROM customer_month, latest_month
WHERE activity_month = latest_month.m;

-- Overall churn rate across the full panel (customer-months basis)
SELECT
    COUNT(*) AS total_customer_months,
    SUM(churn_this_month) AS churn_events,
    ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct
FROM customer_month;

-- Share of the 6,000-customer base that ever churns within the observation window
SELECT
    COUNT(*) AS total_customers,
    SUM(ever_churned) AS customers_who_churned,
    ROUND((100.0 * SUM(ever_churned) / COUNT(*))::numeric, 2) AS pct_ever_churned
FROM (
    SELECT customer_id, MAX(churn_this_month) AS ever_churned
    FROM customer_month GROUP BY customer_id
) t;

-- Monthly churn rate over time (of customers active that month)
SELECT
    activity_month,
    COUNT(*) AS active_customers,
    SUM(churn_this_month) AS churn_events,
    ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct
FROM customer_month
GROUP BY activity_month
ORDER BY activity_month;

-- Customer growth/decline over time (active customers per month, net adds)
WITH monthly_active AS (
    SELECT activity_month, COUNT(*) AS active_customers
    FROM customer_month GROUP BY activity_month
)
SELECT
    activity_month, active_customers,
    active_customers - LAG(active_customers) OVER (ORDER BY activity_month) AS net_change
FROM monthly_active
ORDER BY activity_month;

-- Churn rate by contract type, payment method, plan, region, segment (current snapshot behavior, panel-wide)
SELECT contract_type, COUNT(*) AS customer_months,
       ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct
FROM customer_month GROUP BY contract_type ORDER BY monthly_churn_rate_pct DESC;

SELECT payment_method, COUNT(*) AS customer_months,
       ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct
FROM customer_month GROUP BY payment_method ORDER BY monthly_churn_rate_pct DESC;

SELECT plan, COUNT(*) AS customer_months,
       ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct
FROM customer_month GROUP BY plan ORDER BY monthly_churn_rate_pct DESC;

SELECT region, COUNT(*) AS customer_months,
       ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct
FROM customer_month GROUP BY region ORDER BY monthly_churn_rate_pct DESC;

SELECT tenure_group, COUNT(*) AS customer_months,
       ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct
FROM customer_month GROUP BY tenure_group
ORDER BY CASE tenure_group
    WHEN '0-6mo' THEN 1 WHEN '7-12mo' THEN 2 WHEN '1-2yr' THEN 3
    WHEN '2-4yr' THEN 4 ELSE 5 END;
