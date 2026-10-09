"""
EDA & KPI Framework — Longitudinal Version
===========================================
Charts now emphasize TIME: trends, seasonality, cohort curves, alongside
the cross-sectional churn-driver charts carried over from v1.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

sns.set_style("whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT = PROJECT_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    PROJECT_ROOT / "data" / "processed" / "customer_month_clean.csv",
    parse_dates=["activity_month", "signup_month"]
)

benchmarks = pd.read_csv(
    PROJECT_ROOT / "data" / "reference" / "industry_benchmarks.csv"
)

# --- 1. Monthly churn rate over time ---
fig, ax = plt.subplots(figsize=(10, 4))
monthly_churn = df.groupby("activity_month").apply(
    lambda g: 100 * g["churn_this_month"].sum() / len(g), include_groups=False
)
monthly_churn.plot(ax=ax, color="#C44E52")
ax.set_ylabel("Monthly churn rate (%)"); ax.set_title("Monthly Churn Rate Over Time (2021-2025)")
ax.axvline(pd.Timestamp("2024-07-01"), color="gray", linestyle="--", alpha=0.6)
ax.annotate("synthetic tariff hike", xy=(pd.Timestamp("2024-07-01"), monthly_churn.max()*0.9), fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "monthly_churn_trend.png", dpi=110); plt.close()

# --- 2. Monthly revenue trend (with seasonality + tariff hike step) ---
fig, ax = plt.subplots(figsize=(10, 4))
monthly_rev = df.groupby("activity_month")["monthly_charges"].sum()
monthly_rev.plot(ax=ax, color="#4C72B0")
ax.set_ylabel("Total monthly revenue"); ax.set_title("Monthly Revenue Trend (2021-2025)")
plt.tight_layout(); plt.savefig(OUT / "monthly_revenue_trend.png", dpi=110); plt.close()

# --- 3. Synthetic ARPU vs real industry ARPU benchmark, by year ---
fig, ax = plt.subplots(figsize=(7, 4))
synth_arpu = df.groupby(df["activity_month"].dt.year)["monthly_charges"].mean()
bench = benchmarks[benchmarks["metric"] == "blended_wireless_arpu"].set_index("year")["value"]
ax2 = ax.twinx()
synth_arpu.plot(ax=ax, marker="o", color="#4C72B0", label="Synthetic ARPU (project units)")
bench.plot(ax=ax2, marker="s", color="#DD8452", label="Real industry ARPU (INR, TRAI)")
ax.set_ylabel("Synthetic ARPU"); ax2.set_ylabel("Real industry ARPU (INR)")
ax.set_title("Synthetic ARPU Growth Shape vs Real Industry ARPU (TRAI) — trend calibration check")
lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "arpu_calibration_check.png", dpi=110); plt.close()

# --- 4. Cohort retention curves (first 24 months, by acquisition year) ---
fig, ax = plt.subplots(figsize=(9, 5))
cohort_start = df[df["tenure_months"] == 0][["customer_id", "cohort_year"]]
merged = df.merge(cohort_start, on="customer_id", suffixes=("", "_c"))
for year, grp in merged.groupby("cohort_year"):
    if year == 2025:
        continue  # too censored to plot meaningfully alongside full cohorts
    curve = grp[grp["tenure_months"] <= 24].groupby("tenure_months")["customer_id"].nunique()
    curve_pct = 100 * curve / curve.iloc[0]
    curve_pct.plot(ax=ax, label=str(year), marker="o", markersize=3)
ax.set_xlabel("Months since acquisition"); ax.set_ylabel("Retention (%)")
ax.set_title("Acquisition Cohort Retention Curves (2025 cohort excluded — censored)")
ax.legend(title="Cohort year")
plt.tight_layout(); plt.savefig(OUT / "cohort_retention_curves.png", dpi=110); plt.close()

# --- 5. Churn by contract type (cross-sectional driver, carried over) ---
fig, ax = plt.subplots(figsize=(6, 4))
churn_by_contract = df.groupby("contract_type").apply(
    lambda g: 100 * g["churn_this_month"].sum() / len(g), include_groups=False
).sort_values(ascending=False)
churn_by_contract.plot(kind="bar", ax=ax, color="#55A868")
ax.set_ylabel("Monthly churn rate (%)"); ax.set_title("Churn Rate by Contract Type")
plt.tight_layout(); plt.savefig(OUT / "churn_by_contract.png", dpi=110); plt.close()

# --- 6. Complaints vs churn ---
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df.sample(min(30000, len(df)), random_state=1), x="churn_this_month", y="support_calls_this_month", ax=ax)
ax.set_xlabel("Churned this month (0/1)"); ax.set_title("Support Calls vs Churn (sampled)")
plt.tight_layout(); plt.savefig(OUT / "complaints_vs_churn.png", dpi=110); plt.close()

print("Saved 6 EDA charts to", OUT)

# =========================================================
# KPI FRAMEWORK
# =========================================================
latest_month = df["activity_month"].max()
latest = df[df["activity_month"] == latest_month]
ever_churn = df.groupby("customer_id")["churn_this_month"].max()

kpis = {
    "panel_rows": len(df),
    "unique_customers": df["customer_id"].nunique(),
    "observation_window": f"{df['activity_month'].min().date()} to {df['activity_month'].max().date()}",
    "active_customers_latest_month": len(latest),
    "avg_arpu_latest_month": round(latest["monthly_charges"].mean(), 2),
    "monthly_revenue_run_rate_latest": round(latest["monthly_charges"].sum(), 2),
    "overall_monthly_churn_rate_pct": round(100 * df["churn_this_month"].sum() / len(df), 3),
    "pct_customers_ever_churned_in_window": round(100 * ever_churn.mean(), 2),
    "avg_tenure_months_latest": round(latest["tenure_months"].mean(), 1),
    "revenue_lost_latest_month_to_churn": round(
        df[(df["activity_month"] == latest_month) & (df["churn_this_month"] == 1)]["monthly_charges"].sum(), 2
    ),
}
kpi_df = pd.DataFrame(list(kpis.items()), columns=["kpi", "value"])
kpi_df.to_csv(OUT / "kpi_summary.csv", index=False)
print(kpi_df.to_string(index=False))