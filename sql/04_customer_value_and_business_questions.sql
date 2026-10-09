-- =========================================================
-- 04. CUSTOMER VALUE & KEY BUSINESS QUESTIONS
-- =========================================================

-- Are high-value customers churning disproportionately? (panel-wide, monthly basis)
SELECT
    is_high_value,
    COUNT(*) AS customer_months,
    ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct
FROM customer_month
GROUP BY is_high_value;

-- Which plan/contract/segment combinations have the highest churn AND meaningful revenue (priority retention targets)?
SELECT
    customer_segment, plan, contract_type,
    COUNT(*) AS customer_months,
    ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct,
    ROUND(SUM(monthly_charges)::numeric, 2) AS total_monthly_revenue
FROM customer_month
GROUP BY customer_segment, plan, contract_type
HAVING COUNT(*) > 500
ORDER BY monthly_churn_rate_pct DESC, total_monthly_revenue DESC
LIMIT 15;

-- Complaints -> churn relationship
SELECT
    complaint_flag,
    COUNT(*) AS customer_months,
    ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct,
    ROUND(AVG(support_calls_this_month)::numeric, 2) AS avg_support_calls
FROM customer_month GROUP BY complaint_flag;

-- Did the synthetic tariff-hike month (Jul 2024) coincide with a churn bump?
-- (checks whether the price-shock rule is visible in the aggregate data)
SELECT
    activity_month,
    ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS monthly_churn_rate_pct
FROM customer_month
WHERE activity_month BETWEEN '2024-04-01' AND '2024-10-01'
GROUP BY activity_month ORDER BY activity_month;

-- Region/segment priority flag: above-average churn AND above-average revenue (latest month)
WITH latest_month AS (SELECT MAX(activity_month) AS m FROM customer_month),
region_stats AS (
    SELECT region,
           ROUND((100.0 * SUM(churn_this_month) / COUNT(*))::numeric, 3) AS churn_rate_pct,
           ROUND(SUM(monthly_charges)::numeric, 2) AS total_revenue
    FROM customer_month, latest_month
    WHERE activity_month = latest_month.m
    GROUP BY region
)
SELECT *,
    CASE WHEN churn_rate_pct > (SELECT AVG(churn_rate_pct) FROM region_stats)
         AND total_revenue > (SELECT AVG(total_revenue) FROM region_stats)
    THEN 'PRIORITY: high churn + high revenue' ELSE 'monitor' END AS flag
FROM region_stats ORDER BY churn_rate_pct DESC;
